#!/usr/bin/env python3
"""
Where integration changes between successive context lengths, as maps.

Supplementary Table 4 counts the vertices where delta increases or decreases
from one context length to the next (context_contrast.py, paired sign-flip
TFCE, family = vertices covered by >= 7 subjects), but no figure showed where
they are. This writes, for each adjacent pair already tested, the files
contour_maps.py reads:

    <axis dir>/<pair dir>/delta_change__mean.npy          mean change in delta
    <axis dir>/<pair dir>/delta_change_sigpos__mean.npy   TFCE up, p < alpha/2
    <axis dir>/<pair dir>/delta_change_signeg__mean.npy   TFCE down, p < alpha/2

Nothing is recomputed: the map is `delta__mean_diff.npy` and the masks are
`delta__p_tfce_{up,down}.npy` of the existing contrast. The mask counts are
checked against `summary.json`, so the outlines are exactly the vertices the
table reports. Vertices outside the family are NaN, hence uncoloured.

One symmetric scale serves every change map of both axes (99th percentile of
|change| over the family, all pairs), so steps can be compared across rows and
figures. It is written to `results/group/context_change_maps.json` with the
pair directories, for the CONN prompt and assemble_paper_figures.py.

    PYTHONPATH=. python3 scripts/visualisation/context_change_maps.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

G = Path("results/group")
#: axis -> (contrast dir, levels in order, unit)
AXES = {"text": ("context_contrast_text8", (4, 16, 64, 256, 512, 1024), "words"),
        "audio": ("context_contrast_audio8", (0, 2, 4, 8, 16), "s")}
STAT = "delta_change"


def pair_dir(root: Path, a, b) -> Path:
    """The contrast dir of `a:b`, whatever --prefix it was run with."""
    hits = [p for p in root.iterdir()
            if p.is_dir() and re.fullmatch(rf"[a-z]*{a}_vs_[a-z]*{b}", p.name)]
    if len(hits) != 1:
        raise SystemExit(f"{root}: expected one dir for {a} vs {b}, found {hits}")
    return hits[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--pct", type=float, default=99.0)
    a = ap.parse_args(argv)

    out, absvals = {}, []
    for axis, (cdir, levels, unit) in AXES.items():
        root = G / cdir
        summary = json.loads((root / "summary.json").read_text())
        out[axis] = []
        for lo, hi in zip(levels[:-1], levels[1:]):
            d = pair_dir(root, lo, hi)
            diff = np.load(d / "delta__mean_diff.npy")
            fam = np.load(d / "family.npy").astype(bool)
            masks = {}
            for side, direction in (("sigpos", "up"), ("signeg", "down")):
                p = np.nan_to_num(np.load(d / f"delta__p_tfce_{direction}.npy"), nan=1.0)
                masks[side] = p < a.alpha / 2
                want = summary[f"{lo}:{hi}"]["delta"][f"p_tfce_{direction}"]
                if int(masks[side].sum()) != want:
                    raise SystemExit(f"{d.name} {direction}: {int(masks[side].sum())} "
                                     f"vertices, summary.json says {want}")
            vals = np.where(fam, diff, np.nan).astype(np.float32)
            np.save(d / f"{STAT}__mean.npy", vals)
            for side, m in masks.items():
                np.save(d / f"{STAT}_{side}__mean.npy", m.astype(np.float32))
            absvals.append(np.abs(vals[np.isfinite(vals)]))
            out[axis].append({"pair": f"{lo}:{hi}", "label": f"{lo} → {hi} {unit}",
                              "run": f"{cdir}/{d.name}",
                              "up": int(masks["sigpos"].sum()),
                              "down": int(masks["signeg"].sum())})
            print(f"{axis} {lo}->{hi} {unit}: {d.name}, up {out[axis][-1]['up']}, "
                  f"down {out[axis][-1]['down']}")
    v = float(np.percentile(np.concatenate(absvals), a.pct))
    step = 10 ** (np.floor(np.log10(v)) - 1)       # rounded up, 2 significant digits
    vmax = round(float(np.ceil(v / step) * step), 10)
    out["vmax"] = vmax
    (G / "context_change_maps.json").write_text(json.dumps(out, indent=1))
    print(f"vmax {vmax} (p{a.pct:g} of |change| = {v:.4f}) -> {G / 'context_change_maps.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
