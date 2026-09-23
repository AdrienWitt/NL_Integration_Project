#!/usr/bin/env python3
"""
fsaverage group maps -> CONN surface files (`.surf.nii`), with no resampling.

    <stat>__mean.npy  (327,684 fsaverage vertices)  ->  <stat>.surf.nii

Why this and not a volume
-------------------------
These maps were computed on fsaverage and CONN's surface space *is* fsaverage,
so the two already agree vertex for vertex. Writing a volume and letting CONN
re-project it onto its surface is a surface -> volume -> surface round trip
that resamples twice for no gain. This writes the vertices straight across.
`export_group_mni.py` remains the fallback for tools that only read a volume.

CONN's surface layout
---------------------
CONN stores fsaverage data in a *volume-shaped* NIfTI whose dimensions
multiply back to the vertex count:

    42 x 83 x 47 = 163,842 = fsaverage ico7 vertices per hemisphere

and one map over both hemispheres is `[42 83 94]`: lh then rh, stacked along
the third axis. That is what `conn_surf_write` produces for 2 x 163,842
vertices (`dims.*[1 1 2]`), and `conn_surf_dimscheck` recognises a surface
file by exactly those first three dimensions -- a `[42 83 47 2]` file (the
first version of this script) reads as `[42 83 47]` and is loaded as a
volume. The reshape is MATLAB's, i.e. column-major, hence `order="F"`
below: a C-order reshape would scramble the vertices into noise while
producing a file of exactly the right size.

Confirm it before trusting a figure
-----------------------------------
The dimensions above are asserted here (the factorisation is exact, which is
what identifies them), but hemisphere order and reshape order are conventions
this script cannot test without CONN. So it also writes

    _check_curvature.surf.nii

which is fsaverage's own curvature in the same layout. Open that one first:
curvature has unmistakable structure, every gyrus light and every sulcus dark.
If it renders as a normal folded brain, the layout is right and every other
file in the folder is right with it. If it renders as noise, the reshape order
is wrong; if the hemispheres look swapped, they are. Two minutes, and it fails
loudly instead of silently producing a plausible-looking wrong map.

In MATLAB, the other half of the check:

    conn_surf_dims(8)      % expect [42 83 47]
    V = spm_vol('_check_curvature.surf.nii'); V(1).dim   % expect [42 83 94]
    conn_surf_dimscheck(V) % expect 1

Usage
-----
    python scripts/visualisation/export_group_conn.py --root results/group/<run>
    python scripts/visualisation/export_group_conn.py --root results/group/<run> \\
        --stats delta --test none
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

_HERE = Path(__file__).resolve().parent
#: scripts/visualisation/ -> the project root is two levels up; `_HERE` itself
#: carries plot_group_maps.py, which these scripts import to stay in step with
#: its colormaps, labels and default tests.
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE))

import config  # noqa: E402
from plot_group_maps import DEFAULT_TEST, FALLBACK_TEST, discover  # noqa: E402

log = logging.getLogger("export_group_conn")

HEMIS = ("lh", "rh")
N_PER_HEMI = 163842
#: `conn_surf_dims(8)`. The product is the vertex count exactly, which is what
#: makes this the right triple rather than a guess.
CONN_DIMS = (42, 83, 47)
assert CONN_DIMS[0] * CONN_DIMS[1] * CONN_DIMS[2] == N_PER_HEMI


def fsaverage_dir() -> Path:
    local = Path(config.FREESURFER_DIR) / "fsaverage"
    if (local / "surf" / "lh.curv").exists():
        return local
    fallback = Path("/usr/local/freesurfer/7.4.1/subjects/fsaverage")
    if (fallback / "surf" / "lh.curv").exists():
        return fallback
    raise SystemExit("no fsaverage found; set FREESURFER_DIR")


def to_conn_volume(values: np.ndarray) -> nib.Nifti1Image:
    """(327,684,) fsaverage vertices, lh then rh -> a [42 83 94] CONN
    surface NIfTI, as `conn_surf_write` lays it out."""
    if values.shape[0] != 2 * N_PER_HEMI:
        raise SystemExit(
            f"expected {2 * N_PER_HEMI} vertices, got {values.shape[0]}")
    dims = (CONN_DIMS[0], CONN_DIMS[1], 2 * CONN_DIMS[2])
    data = values.reshape(dims, order="F").astype(np.float32)
    # The affine is meaningless for surface data -- nothing here is in mm --
    # and CONN reads these files by their dimensions. Identity keeps any
    # viewer that ignores that from inventing a plausible-looking geometry.
    img = nib.Nifti1Image(data, np.eye(4))
    img.set_data_dtype(np.float32)
    return img


def masked_values(root: Path, stat: str, test: str, alpha: float) -> np.ndarray:
    """Group mean, zeroed outside coverage and outside significance.

    Zero rather than NaN: CONN thresholds at a value and a NaN is not a
    number it can compare.
    """
    mean = np.load(root / f"{stat}__mean.npy").astype(np.float32)
    out = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    if test == "none":
        return out
    p = np.load(root / f"{stat}__{test}.npy")
    sig = np.isfinite(p) & (p < alpha) & np.isfinite(mean)
    return np.where(sig, out, 0.0).astype(np.float32)


def write_reference(fsdir: Path, out: Path) -> None:
    """fsaverage curvature in the same layout: the known-answer test."""
    curv = np.concatenate(
        [fsio.read_morph_data(fsdir / "surf" / f"{h}.curv") for h in HEMIS])
    nib.save(to_conn_volume(curv.astype(np.float32)), out)
    log.info("wrote %s  <- open this one first", out.name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--stats", nargs="*", default=None)
    ap.add_argument("--test", default=None,
                    help="p-map to threshold with, or 'none'. Default: as "
                         "plot_group_maps")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--no-reference", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.root.is_dir():
        raise SystemExit(f"no such directory: {args.root}")
    out_dir = args.out_dir or (args.root / "export" / "conn")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_reference:
        write_reference(fsaverage_dir(), out_dir / "_check_curvature.surf.nii")

    for stat in discover(args.root, args.stats):
        test = args.test or DEFAULT_TEST.get(stat, FALLBACK_TEST)
        if test != "none" and not (args.root / f"{stat}__{test}.npy").exists():
            log.warning("%s: no %s, writing unthresholded", stat, test)
            test = "none"
        values = masked_values(args.root, stat, test, args.alpha)
        suffix = "unthresholded" if test == "none" else test
        out = out_dir / f"{stat}__{suffix}.surf.nii"
        nib.save(to_conn_volume(values), out)
        log.info("wrote %s  (%d non-zero vertices, max %.4f)", out.name,
                 int(np.count_nonzero(values)), float(values.max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
