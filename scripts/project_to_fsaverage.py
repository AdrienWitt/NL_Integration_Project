#!/usr/bin/env python3
"""
Project encoding result maps onto fsaverage and average them over subjects.

    subject volume  --pycortex trilinear-->  subject native surface
                    --mri_surf2surf------->  fsaverage
                    --mean over subjects-->  group map

Why the order matters
---------------------
Voxel 40,000 is a different piece of cortex in every subject, so a mean over
subject-space maps averages unrelated tissue. fsaverage is the frame in which
averaging is defined, so the projection has to come first. (Note that
`stats/analysis.py` writes `group/mean_*.npy` the other way round.)

Vertices are not covered equally either: a subject's slab does not reach all
of cortex, and `--min-ev` leaves whole regions unfitted. Each subject's
coverage is therefore recorded and uncovered vertices are excluded from the
mean rather than pulled toward zero, with `n_subjects_*.npy` written
alongside every group map so a vertex backed by 2 subjects is not read like
one backed by 9.

Usage
-----
    # every run under results/encoding; group mean per run directory
    python scripts/project_to_fsaverage.py

    # one run
    python scripts/project_to_fsaverage.py \\
        --results-dir results/encoding/gpt2_mean__base_emotion_L11__UTS01_c25

    python scripts/project_to_fsaverage.py --check     # readiness per subject
    python scripts/project_to_fsaverage.py --dry-run   # what it would touch

Needs the pycortex transform in PYCORTEX_DB, FreeSurfer surfaces for each
subject *and* fsaverage in FREESURFER_DIR, a FreeSurfer licence, and
`$FREESURFER_HOME/bin` on PATH. `--check` says which of those is missing.
"""

import argparse
import contextlib
import glob
import json
import logging
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    ENCODING_OUT, FSAVERAGE_DIR, FREESURFER_DIR, FS_LICENSE,
    PYCORTEX_DB, SUBJECTS, SUBJECT_XFMS,
)

log = logging.getLogger("project")

#: Maps worth interpolating onto a surface. Everything else a run writes is
#: either categorical or not a per-voxel quantity:
#:   voxel_mask.npy      bool; a smoothed edge is not a mask
#:   *_best_alphas.npy   regularisation strength, spans orders of magnitude
#:   *_deltas.npy        himalaya's per-band log scalings -- NOT the
#:                       integration statistic, which is contrasts/delta.npy
#:   contrasts/winner.npy  categorical; interpolating model ids is meaningless
DEFAULT_MAP_GLOBS = ("*_corrs.npy", "ev.npy",
                     "contrasts/delta.npy", "contrasts/preference.npy",
                     "contrasts/r_*.npy")

SKIP_SUFFIXES = ("_fsaverage.npy", "voxel_mask.npy", "_best_alphas.npy",
                 "_deltas.npy", "winner.npy", "coverage.npy")

COVERAGE_NAME = "coverage_fsaverage.npy"


@contextlib.contextmanager
def quiet(enabled: bool = True):
    """Silence stdout, including from subprocesses.

    `mri_surf2surf` writes ~1 MB of progress counter per hemisphere. Under
    SLURM stdout is a file on a shared volume, and that is the same flood
    that took down the 36-task sweep array with `OSError: [Errno 121]`.
    Redirect at the file-descriptor level: the noise comes from a child
    process, which `contextlib.redirect_stdout` would not catch.
    """
    if not enabled:
        yield
        return
    sys.stdout.flush()
    saved, devnull = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def setup_pycortex():
    """Point pycortex and FreeSurfer at this project's directories.

    Must run before the first `cortex.db` lookup: the Database caches its
    subject list on first use, so assigning `filestore` afterwards without
    clearing that cache silently keeps the old store.
    """
    os.environ["SUBJECTS_DIR"] = str(FREESURFER_DIR)
    os.environ["FS_LICENSE"] = str(FS_LICENSE)

    import cortex
    cortex.db.filestore = str(PYCORTEX_DB)
    cortex.db._subjects = None
    cortex.freesurfer.subjects_dir = str(FREESURFER_DIR)
    return cortex


def xfm_for(subject: str) -> str:
    if Path(SUBJECT_XFMS).exists():
        mapping = json.loads(Path(SUBJECT_XFMS).read_text())
        if subject in mapping:
            return mapping[subject]
    return f"{subject}_auto"


def preflight(subject: str) -> list[str]:
    """Reasons `subject` cannot be projected; empty means ready."""
    problems = []
    xfm_dir = Path(PYCORTEX_DB) / subject / "transforms" / xfm_for(subject)
    if not (xfm_dir / "matrices.xfm").exists():
        problems.append(f"no pycortex transform at {xfm_dir} "
                        f"(need matrices.xfm + reference.nii.gz + mask_*.nii.gz)")
    surfaces = Path(PYCORTEX_DB) / subject / "surfaces"
    if not surfaces.is_dir() or not any(surfaces.iterdir()):
        problems.append(f"no pycortex surfaces at {surfaces}")

    for fs_subj in (subject, "fsaverage"):
        surf = Path(FREESURFER_DIR) / fs_subj / "surf" / "lh.white"
        if not surf.exists():
            hint = ""
            if fs_subj == "fsaverage":
                home = os.environ.get("FREESURFER_HOME", "$FREESURFER_HOME")
                hint = (f"\n        fix: ln -s {home}/subjects/fsaverage "
                        f"{Path(FREESURFER_DIR) / 'fsaverage'}")
            problems.append(f"missing FreeSurfer surface {surf}{hint}")

    if not Path(FS_LICENSE).exists():
        problems.append(f"no FreeSurfer licence at {FS_LICENSE} "
                        f"(set FS_LICENSE=/path/to/license.txt)")
    return problems


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------
class Projector:
    """Volume -> native surface -> fsaverage, for one subject.

    Both stages are linear and subject-constant, so they are built once and
    reused for every map. The surf2surf matrices are cached on disk as well:
    pycortex estimates them by running `mri_surf2surf` over 40 test images,
    ~25 s per hemisphere every time otherwise.
    """

    def __init__(self, subject: str, cache_dir: Path | None = None,
                 verbose: bool = False):
        import nibabel as nib
        import scipy.sparse as sp

        self.cortex = setup_pycortex()
        self.subject = subject
        self.xfmname = xfm_for(subject)

        self.mapper = self.cortex.get_mapper(subject, self.xfmname, "trilinear")
        pts_lh, _ = self.cortex.db.get_surf(subject, "fiducial", hemisphere="lh")
        self.num_lh = pts_lh.shape[0]

        cache_dir = Path(cache_dir or Path(FSAVERAGE_DIR) / "_surf2surf_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.mapping = {}
        for hemi in ("lh", "rh"):
            cache = cache_dir / f"{subject}_{hemi}_white_to_fsaverage.npz"
            if cache.exists():
                self.mapping[hemi] = sp.load_npz(cache)
                log.info(f"[{subject}] surf2surf {hemi}: cached "
                         f"{self.mapping[hemi].shape}")
            else:
                log.info(f"[{subject}] surf2surf {hemi}: estimating "
                         f"(mri_surf2surf, ~25 s)")
                with quiet(not verbose):
                    m = self.cortex.freesurfer.get_mri_surf2surf_matrix(
                        source_subj=subject, hemi=hemi, surface_type="white",
                        target_subj="fsaverage",
                        subjects_dir=str(FREESURFER_DIR),
                    )
                m = sp.csr_matrix(m)
                sp.save_npz(cache, m)
                self.mapping[hemi] = m

        self.n_vertices = (self.mapping["lh"].shape[0]
                           + self.mapping["rh"].shape[0])

        # Resolve the voxel mask explicitly rather than letting pycortex infer
        # it from the array length: a silent mismatch would scatter values
        # into the wrong voxels and still produce a plausible-looking map.
        self._masks = {}
        pattern = self.cortex.db.get_paths(subject)["masks"].format(
            xfmname=self.xfmname, type="*")
        for mf in sorted(glob.glob(pattern)):
            mask = nib.load(mf).get_fdata().T != 0
            self._masks.setdefault(int(mask.sum()), (Path(mf).name, mask))
        log.info(f"[{subject}] xfm {self.xfmname}, masks "
                 f"{ {n: v[0] for n, v in self._masks.items()} }, "
                 f"{self.n_vertices:,} fsaverage vertices")

    def mask_for(self, n_voxels: int):
        if n_voxels not in self._masks:
            raise ValueError(
                f"{self.subject}: data has {n_voxels:,} voxels, but this "
                f"transform's masks hold {sorted(self._masks)}. Wrong subject, "
                f"wrong transform, or data that is not in this voxel space."
            )
        return self._masks[n_voxels][1]

    def project(self, data: np.ndarray) -> np.ndarray:
        """(n_voxels,) -> (n_fsaverage_vertices,)."""
        mask = self.mask_for(data.shape[-1])

        # NaNs would bleed through the sparse matmul into every vertex the
        # voxel touches; zero them and say how many, rather than returning a
        # map with mysteriously blank patches.
        n_nan = int(np.isnan(data).sum())
        if n_nan:
            log.warning(f"[{self.subject}] {n_nan:,} NaN set to 0 before "
                        f"projection")
            data = np.nan_to_num(data, nan=0.0)

        vol = self.cortex.Volume(np.asarray(data, dtype=np.float32),
                                 self.subject, self.xfmname, mask=mask)
        vtx = np.asarray(self.mapper(vol).data).ravel()
        return np.concatenate([self.mapping["lh"] @ vtx[:self.num_lh],
                               self.mapping["rh"] @ vtx[self.num_lh:]]
                              ).astype(np.float32)

    def coverage(self, n_voxels: int) -> np.ndarray:
        """Vertices this subject's data can actually reach.

        Pushing an all-ones volume through both stages gives each vertex its
        total interpolation weight; zero means no voxel of this subject's
        acquisition contributes to it. Those vertices are missing data, not
        measurements of zero, and must not enter the group mean.
        """
        return self.project(np.ones(n_voxels, dtype=np.float32)) > 0


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def config_key(run_root: Path, subject_dir: Path) -> str:
    """Group label for a result directory, with the subject removed.

    Runs are named `gpt2_mean__base_emotion_9to11__UTS01_c25` -- one
    directory per subject, the subject baked into the name -- so grouping on
    the directory itself gives nine groups of one and averages nothing.
    Strip the `UTS0N` token and keep backend/eval, so cv and holdout stay
    apart and the feature configurations stay apart.
    """
    name = re.sub(r"_*UTS\d{2}_*", "_", run_root.name).strip("_")
    name = re.sub(r"_{3,}", "__", name)
    return str(Path(name) / subject_dir.relative_to(run_root).parent)


def find_result_maps(roots, globs, subjects):
    """-> {config key: {subject: (subject_dir, [map paths])}}.

    Keyed by configuration rather than directory because `audio_corrs.npy`
    names a different model in every run, and pooling across runs would
    silently mix the feature configurations the whole sweep exists to tell
    apart.
    """
    found = defaultdict(dict)
    for root in roots:
        root = Path(root)
        for subject in subjects:
            for subject_dir in sorted(root.glob(f"**/{subject}")):
                if not subject_dir.is_dir():
                    continue
                paths = [p for pattern in globs
                         for p in sorted(subject_dir.glob(pattern))
                         if not p.name.endswith(SKIP_SUFFIXES)]
                if not paths:
                    continue
                # <run>/<backend>/<eval>/<subject> -- the run is 3 levels up
                run_root = subject_dir.parents[2]
                found[config_key(run_root, subject_dir)][subject] = \
                    (subject_dir, paths)
    return found


def fs_name(path: Path) -> Path:
    return path.with_name(path.stem + "_fsaverage.npy")


# ---------------------------------------------------------------------------
def project_run(per_subject, args) -> dict:
    """Project one run's maps. -> {subject: {"maps": {key: path}, ...}}."""
    written = {}
    for subject, (subject_dir, paths) in sorted(per_subject.items()):
        # key relative to the subject dir: "audio_corrs", "contrasts/delta"
        keys = {str(p.relative_to(subject_dir).with_suffix("")): p
                for p in paths}
        todo = [p for p in paths if args.overwrite or not fs_name(p).exists()]
        cov_path = subject_dir / COVERAGE_NAME

        if args.dry_run:
            for path in todo:
                log.info(f"[{subject}] would project {path}")
            continue

        if todo or not cov_path.exists():
            problems = preflight(subject)
            if problems:
                log.error(f"[{subject}] cannot project:")
                for problem in problems:
                    log.error(f"    {problem}")
                continue
            projector = Projector(subject, verbose=args.verbose)
            for path in todo:
                arr = np.load(path).ravel()
                fs = projector.project(arr)
                np.save(fs_name(path), fs)
                log.info(f"[{subject}] {path.name} {arr.shape} -> {fs.shape}")
            if args.overwrite or not cov_path.exists():
                n_vox = np.load(paths[0]).ravel().shape[0]
                np.save(cov_path, projector.coverage(n_vox))
                log.info(f"[{subject}] coverage -> {cov_path.name}")
        else:
            log.info(f"[{subject}] {len(paths)} maps already projected")

        written[subject] = {
            "maps": {k: fs_name(v) for k, v in keys.items()
                     if fs_name(v).exists()},
            "coverage": cov_path,
        }
    return written


def write_group_mean(written: dict, out_dir: Path):
    """Average each map over subjects, in fsaverage space, coverage-aware."""
    by_key = defaultdict(list)
    for subject, entry in written.items():
        for key, path in entry["maps"].items():
            by_key[key].append((subject, path, entry["coverage"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    for key, items in sorted(by_key.items()):
        if len(items) < 2:
            log.info(f"group: {key} has only {len(items)} subject, skipped")
            continue
        stack = []
        for subject, path, cov_path in items:
            arr = np.load(path).astype(np.float64)
            if cov_path.exists():
                arr = np.where(np.load(cov_path), arr, np.nan)
            stack.append(arr)
        stack = np.stack(stack)
        n = np.sum(np.isfinite(stack), axis=0)
        with warnings.catch_warnings():
            # vertices no subject covers are a real category, not a mistake
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(stack, axis=0)
        mean[n == 0] = np.nan

        name = key.replace("/", "_")
        np.save(out_dir / f"mean_{name}_fsaverage.npy", mean.astype(np.float32))
        np.save(out_dir / f"n_subjects_{name}.npy", n.astype(np.int16))
        full = int((n == len(items)).sum())
        log.info(f"group: mean_{name} over {len(items)} subjects "
                 f"({full:,}/{n.size:,} vertices covered by all)")


def show(path: Path):
    """Open one projected map in the pycortex viewer."""
    cortex = setup_pycortex()
    data = np.load(path).astype(np.float64)
    finite = data[np.isfinite(data) & (data != 0)]
    cortex.webshow(cortex.Vertex(
        data, "fsaverage", cmap="hot",
        vmin=float(np.percentile(finite, 1)),
        vmax=float(np.percentile(finite, 99)),
    ))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", nargs="*",
                    help="run directories (default: everything under "
                         "results/encoding)")
    ap.add_argument("--maps", nargs="*",
                    help=f"globs relative to each subject dir (default: "
                         f"{' '.join(DEFAULT_MAP_GLOBS)})")
    ap.add_argument("--subjects", nargs="+", default=["all"])
    ap.add_argument("--no-group", action="store_true",
                    help="project only, skip the across-subject mean")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="report per-subject readiness and stop")
    ap.add_argument("--verbose", action="store_true",
                    help="let mri_surf2surf print its progress; ~1 MB per "
                         "hemisphere, keep it off under SLURM")
    ap.add_argument("--show", metavar="MAP_fsaverage.npy")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    subjects = SUBJECTS if args.subjects == ["all"] else args.subjects

    if args.show:
        show(Path(args.show))
        return

    if args.check:
        for subject in subjects:
            problems = preflight(subject)
            if problems:
                print(f"[{subject}] NOT READY")
                for problem in problems:
                    print(f"    {problem}")
            else:
                print(f"[{subject}] ready ({xfm_for(subject)})")
        return

    roots = ([Path(d) for d in args.results_dir] if args.results_dir
             else [Path(ENCODING_OUT)])
    runs = find_result_maps(roots, args.maps or DEFAULT_MAP_GLOBS, subjects)
    if not runs:
        log.error(f"No result maps under {[str(r) for r in roots]}")
        return

    total = sum(len(v[1]) for r in runs.values() for v in r.values())
    log.info(f"{total} maps in {len(runs)} run directories\n")

    group_root = (roots[0] if (args.results_dir and len(roots) == 1)
                  else Path(ENCODING_OUT)) / "group_fsaverage"

    for key, per_subject in sorted(runs.items()):
        log.info(f"=== {key} ({len(per_subject)} subjects: "
                 f"{', '.join(sorted(per_subject))})")
        written = project_run(per_subject, args)
        if written and not args.no_group and not args.dry_run:
            write_group_mean(written, group_root / key)


if __name__ == "__main__":
    main()
