#!/usr/bin/env python3
"""
CONN renders of the context-figure regions -> one brain per hemisphere.

    results/group/roi_atlas/export/conn/fig_mosaic8/roi_<key>__mask.surf.png
      -> results/group/roi_atlas/roi_atlas_{lh,rh}.png        lateral view, regions in curve colours
         results/group/roi_atlas/roi_atlas_centroids.json      label anchor of each region per view

Each mask was printed alone in its curve colour (export_roi_masks_conn.py), so
a pixel belongs to a region where its render departs from the bare surface.
The bare surface is the per-pixel median of the renders: every pixel is
coloured in at most a few of them. plot_roi_context.py draws the two views as
a third column, beside the curves of the same hemisphere.

    python scripts/visualisation/plot_roi_atlas_panel.py [--discs-only]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.visualisation.conn_figures import REF_VIEWS  # noqa: E402
from scripts.visualisation.plot_roi_context import ROIS  # noqa: E402

ROOT = Path("results/group/roi_atlas")
PRINTS = ROOT / "export" / "conn" / "fig_mosaic8"
VIEWS = {"lh": 0, "rh": 5}          # REF_VIEWS: L lateral, R lateral
THRESH = 40                          # summed |RGB| departure from the bare surface
PALE = 0.6                           # region tint when its disc is drawn over it


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--discs-only", action="store_true",
                    help="paint only the peak discs, not the whole regions")
    args = ap.parse_args(argv)
    keys = [k for k, *_ in ROIS]
    files = {k: PRINTS / f"roi_{k}__mask.surf.png" for k in keys}
    imgs = {k: np.asarray(Image.open(f).convert("RGB")).astype(np.int16) for k, f in files.items()}
    sizes = {im.shape for im in imgs.values()}
    if len(sizes) != 1:
        raise SystemExit(f"renders differ in size: {sizes}")
    # the 6-mm peak discs of the curves, when rendered: the region is drawn
    # pale and its disc in full colour, and the label points at the disc
    dfiles = {k: PRINTS / f"roi_{k}_disc__mask.surf.png" for k in keys}
    dimgs = {k: np.asarray(Image.open(f).convert("RGB")).astype(np.int16)
             for k, f in dfiles.items() if f.exists()}
    base = np.median(np.stack(list(imgs.values())), axis=0).astype(np.int16)
    out = base.copy()
    masks = {}
    for k in keys:
        m = np.abs(imgs[k] - base).sum(-1) > THRESH
        if not args.discs_only:
            out[m] = (imgs[k][m] * (1 - PALE) + 255 * PALE) if dimgs else imgs[k][m]
        masks[k] = m
    dmasks = {}
    for k, im in dimgs.items():
        m = np.abs(im - base).sum(-1) > THRESH
        out[m] = im[m]
        dmasks[k] = m
    cents, discs = {}, {}
    for hemi, i in VIEWS.items():
        x0, y0, x1, y1 = REF_VIEWS[i]
        Image.fromarray(out[y0:y1, x0:x1].astype(np.uint8)).save(ROOT / f"roi_atlas_{hemi}.png")
        cents[hemi] = {}
        for k in keys:
            ys, xs = np.nonzero(dmasks[k][y0:y1, x0:x1]) if k in dmasks else ([], [])
            on_disc = len(xs) > 50
            if not on_disc and args.discs_only:
                continue
            if not on_disc:     # no disc in this view: point at the region
                ys, xs = np.nonzero(masks[k][y0:y1, x0:x1])
                if len(xs) <= 200:  # region not visible in this view either
                    continue
            cents[hemi][k] = [float(np.median(xs)) / (x1 - x0), float(np.median(ys)) / (y1 - y0)]
            if on_disc:
                discs.setdefault(hemi, []).append(k)
    json.dump({"centroids": cents, "on_disc": discs}, open(ROOT / "roi_atlas_centroids.json", "w"), indent=1)
    print(f"-> {ROOT}/roi_atlas_lh.png, roi_atlas_rh.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
