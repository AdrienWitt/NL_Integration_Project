#!/usr/bin/env python3
"""
The regions of the context figure as CONN surface masks, one per region.

    psn.annot (scripts/atlas)  ->  results/group/roi_atlas/export/conn/
                                     roi_<key>__mask.surf.nii   1 in the region, 0 elsewhere
                                     roi_<key>.cmap.txt          96 x 3, the region's curve colour
                                     _check_curvature.surf.nii

The third panel of the context figure shows where the curves come from: each
region in the colour of its curve, on the same CONN surface as the maps. One
mask per region rather than one label map: CONN interpolates values across a
face, so a label map would paint a thin line of a third region's colour along
every border between two others. A single-colour colormap makes the edge of a
binary mask fade in its own colour. print_group_conn.m picks up
`<stat>.cmap.txt` by itself; render with `--vmax 1` so the colour range is
[0 1]. The renders are composited into one brain afterwards
(plot_roi_atlas_panel.py).

Colours come from plot_roi_context.ROIS, so the panel and the curves cannot
disagree.

    python scripts/visualisation/export_roi_masks_conn.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.visualisation.export_group_conn import (  # noqa: E402
    HEMIS, fsaverage_dir, to_conn_volume, write_reference)
from scripts.visualisation.plot_roi_context import ROIS  # noqa: E402

log = logging.getLogger("export_roi_masks_conn")
OUT = Path("results/group/roi_atlas/export/conn")
DISCS = Path("results/group/roi_context_discs.json")


def masks(fsdir: Path) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {}
    for hemi in HEMIS:
        lab, _, names = nib.freesurfer.read_annot(fsdir / "label" / f"{hemi}.psn.annot")
        names = [n.decode() for n in names]
        for key, *_ in ROIS:
            out.setdefault(key, []).append(lab == names.index(key))
    return {k: np.concatenate(v) for k, v in out.items()}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fsdir = fsaverage_dir()
    OUT.mkdir(parents=True, exist_ok=True)
    write_reference(fsdir, OUT / "_check_curvature.surf.nii")
    colour = {k: c for k, _, c, _ in ROIS}
    # the 6-mm peak discs of the curves (roi_context_curve.py --discs-out),
    # one mask per region with both hemispheres, drawn over the paler region
    import json
    discs = json.load(open(DISCS)) if DISCS.exists() else {}
    regions = masks(fsdir)
    n_vert = len(next(iter(regions.values())))
    for key, *_ in ROIS:
        v = [i for h in HEMIS for i in discs.get(f"{h}.{key}", {}).get("vertices", [])]
        if not v:
            continue
        m = np.zeros(n_vert, dtype=np.float32)
        m[v] = 1
        nib.save(to_conn_volume(m), OUT / f"roi_{key}_disc__mask.surf.nii")
        rgb = np.array([int(colour[key][i:i + 2], 16) for i in (1, 3, 5)]) / 255
        np.savetxt(OUT / f"roi_{key}_disc.cmap.txt", np.tile(rgb, (96, 1)), fmt="%.6f")
        log.info("roi_%s_disc: %d vertices", key, len(v))
    for key, m in regions.items():
        nib.save(to_conn_volume(m.astype(np.float32)), OUT / f"roi_{key}__mask.surf.nii")
        rgb = np.array([int(colour[key][i:i + 2], 16) for i in (1, 3, 5)]) / 255
        np.savetxt(OUT / f"roi_{key}.cmap.txt", np.tile(rgb, (96, 1)), fmt="%.6f")
        log.info("roi_%s: %d vertices, %s", key, int(m.sum()), colour[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
