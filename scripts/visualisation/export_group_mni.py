#!/usr/bin/env python3
"""
fsaverage surface maps -> MNI152 volumes, for tools that only read a .nii.

    <stat>__mean.npy  --mri_surf2vol-->  fsaverage ribbon (MNI305, 1 mm)
                      --affine--------->  MNI152 RAS
                      --resample------->  the SPM/CONN 2 mm grid  -> <stat>.nii

Read this before using the output
---------------------------------
These maps were never volumetric. Each subject is fitted in their own
functional space, projected to their own surface and then to fsaverage, and
the group average exists only there -- the nine subjects' functional volumes
are in nine different spaces and there is no group volume to go back to. So
this is a *forward* conversion of a surface result, not a recovery of a
volume result, and two things follow:

- The value at a voxel is the surface value of the nearest ribbon vertex,
  smeared through the cortical ribbon. Nothing lives outside the ribbon:
  no white matter, no subcortex, no cerebellum. A voxel in the middle of the
  thalamus is empty because the analysis never looked there.
- **Peak coordinates from these volumes are not trustworthy** in the way a
  volumetric analysis's peaks are. The surface->volume hop resamples twice and
  the MNI305->MNI152 step is a single affine. Use them to *render*; quote
  vertex-level or parcel-level numbers from the surface, which is where the
  statistics were computed.

If the target tool renders surfaces from fsaverage itself (CONN does), feeding
it the surface directly is exact and this script is the fallback, not the
first choice.

Usage
-----
    python scripts/visualisation/export_group_mni.py --root results/group/<run>
    python scripts/visualisation/export_group_mni.py --root results/group/<run> \\
        --stats delta preference --test none
    python scripts/visualisation/export_group_mni.py --root results/group/<run> \\
        --grid path/to/reference.nii     # match someone else's voxel grid

Needs FreeSurfer on PATH (`mri_surf2vol`) and a licence in $FS_LICENSE.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.image import resample_img

_HERE = Path(__file__).resolve().parent
#: scripts/visualisation/ -> the project root is two levels up; `_HERE` itself
#: carries plot_group_maps.py, which these scripts import to stay in step with
#: its colormaps, labels and default tests.
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE))

import config  # noqa: E402
from plot_group_maps import DEFAULT_TEST, FALLBACK_TEST, discover  # noqa: E402

log = logging.getLogger("export_group_mni")

HEMIS = ("lh", "rh")
N_PER_HEMI = 163842

#: MNI305 RAS -> MNI152 RAS. fsaverage's own space is MNI305; this is the
#: constant affine FreeSurfer publishes for the pair (it is a rigid-ish
#: correction of under 2 mm, not a nonlinear warp, and that is all the
#: accuracy any peak read off these volumes can claim).
MNI305_TO_MNI152 = np.array([
    [0.9975, -0.0073,  0.0176, -0.0429],
    [0.0146,  1.0009, -0.0024,  1.5496],
    [-0.0130, -0.0093, 0.9971,  1.1840],
    [0.0,      0.0,     0.0,    1.0],
])

#: The grid CONN and SPM use for second-level maps, and the one the Study 3
#: maps in the irony project are on: 79 x 95 x 79 at 2 mm.
DEFAULT_SHAPE = (79, 95, 79)
DEFAULT_AFFINE = np.array([
    [-2.0, 0.0, 0.0,   78.0],
    [0.0,  2.0, 0.0, -112.0],
    [0.0,  0.0, 2.0,  -70.0],
    [0.0,  0.0, 0.0,    1.0],
])


def fsaverage_dir() -> Path:
    local = Path(config.FREESURFER_DIR) / "fsaverage"
    if (local / "surf" / "lh.white").exists():
        return local
    fallback = Path("/usr/local/freesurfer/7.4.1/subjects/fsaverage")
    if (fallback / "surf" / "lh.white").exists():
        return fallback
    raise SystemExit("no fsaverage found; set FREESURFER_DIR")


def masked_values(root: Path, stat: str, test: str, alpha: float) -> np.ndarray:
    """The group mean, zeroed outside coverage and outside significance.

    Zero, not NaN: mri_surf2vol has no notion of a missing vertex and a NaN
    propagates through the ribbon fill.
    """
    mean = np.load(root / f"{stat}__mean.npy").astype(np.float32)
    out = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    if test == "none":
        return out
    p = np.load(root / f"{stat}__{test}.npy")
    sig = np.isfinite(p) & (p < alpha) & np.isfinite(mean)
    return np.where(sig, out, 0.0).astype(np.float32)


def surf2vol(values: np.ndarray, fsdir: Path, tmp: Path, out: Path) -> Path:
    """One call, both hemispheres, filled through fsaverage's own ribbon."""
    if values.shape[0] != 2 * N_PER_HEMI:
        raise SystemExit(f"expected {2 * N_PER_HEMI} vertices, got {values.shape[0]}")
    cmd = ["mri_surf2vol", "--subject", "fsaverage", "--o", str(out)]
    for i, hemi in enumerate(HEMIS):
        chunk = values[i * N_PER_HEMI:(i + 1) * N_PER_HEMI]
        ov = tmp / f"{hemi}.mgz"
        nib.save(nib.MGHImage(chunk.reshape(-1, 1, 1).astype(np.float32),
                              np.eye(4)), ov)
        cmd += ["--so", str(fsdir / "surf" / f"{hemi}.white"), str(ov)]
    env = dict(os.environ)
    env["SUBJECTS_DIR"] = str(fsdir.parent)
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0 or not out.exists():
        raise SystemExit(f"mri_surf2vol failed:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
    return out


def to_mni152(fsvol: Path, shape, affine, out: Path) -> None:
    img = nib.load(fsvol)
    # fsaverage's conformed space IS MNI305, so correcting the affine is the
    # whole of the MNI305->MNI152 step; the voxel data is untouched and only
    # the resampling below moves values between grids.
    moved = nib.Nifti1Image(np.asarray(img.dataobj, dtype=np.float32),
                            MNI305_TO_MNI152 @ img.affine)
    out_img = resample_img(moved, target_affine=affine, target_shape=shape,
                           interpolation="linear")
    out_img.set_data_dtype(np.float32)
    nib.save(out_img, out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--stats", nargs="*", default=None)
    ap.add_argument("--test", default=None,
                    help="p-map to threshold with, or 'none' for the "
                         "unthresholded mean. Default: as plot_group_maps")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--grid", type=Path, default=None,
                    help="reference .nii whose shape and affine to match "
                         "(default: the 79x95x79 2 mm SPM/CONN grid)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--keep-fsaverage", action="store_true",
                    help="also write the 1 mm fsaverage-space volume")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if shutil.which("mri_surf2vol") is None:
        raise SystemExit("mri_surf2vol not on PATH -- source FreeSurfer first")
    if not args.root.is_dir():
        raise SystemExit(f"no such directory: {args.root}")

    if args.grid:
        ref = nib.load(args.grid)
        shape, affine = ref.shape[:3], ref.affine
        log.info("grid from %s: %s", args.grid.name, shape)
    else:
        shape, affine = DEFAULT_SHAPE, DEFAULT_AFFINE

    fsdir = fsaverage_dir()
    out_dir = args.out_dir or (args.root / "export" / "mni")
    out_dir.mkdir(parents=True, exist_ok=True)

    for stat in discover(args.root, args.stats):
        test = args.test or DEFAULT_TEST.get(stat, FALLBACK_TEST)
        if test != "none" and not (args.root / f"{stat}__{test}.npy").exists():
            log.warning("%s: no %s, writing unthresholded", stat, test)
            test = "none"
        values = masked_values(args.root, stat, test, args.alpha)
        suffix = "unthresholded" if test == "none" else test
        name = f"{stat}__{suffix}"
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fsvol = surf2vol(values, fsdir, tmp, tmp / "fsaverage.nii")
            if args.keep_fsaverage:
                shutil.copy(fsvol, out_dir / f"{name}__fsaverage1mm.nii")
            to_mni152(fsvol, shape, affine, out_dir / f"{name}.nii")
        vol = nib.load(out_dir / f"{name}.nii").get_fdata()
        log.info("wrote %s.nii  (%d non-zero voxels, max %.4f)", name,
                 int(np.count_nonzero(vol)), float(np.nanmax(vol)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
