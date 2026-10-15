#!/usr/bin/env python3
"""
Significance contours over an unthresholded signed map, for CONN.

CONN's 3-D display draws one activation map per surface and has no overlay
(`repaintborder` *replaces* the map by its border, and the `fileVOL` argument
draws volume blobs, not lines on the cortex). Its per-vertex colours live in
a closure, so they cannot be edited between opening the display and printing.
What it does honour is the colormap, so the contour is carried *in the map*:

    value(v) = +vmax                     on the border of the positive mask
               -vmax                     on the border of the negative mask
               clip(map, +-vmax * KEEP)  everywhere else

and the 192-row diverging colormap gets its last row (+vmax) set to the
positive contour colour and its first row (-vmax) to the negative one. The
data are clipped just short of those two rows (CONN maps V in [-1, 1] to row
round(96.5 + 95.5 V), so |V| <= KEEP lands on rows 3..191), which changes
nothing visible: it only affects vertices within 1.6 % of the scale's end.
The result is an ordinary `.surf.nii` + `<stat>.cmap.txt` pair, rendered by
the existing print_group_conn.m / conn_print_maps.m path unchanged, so the
camera, print size and layout are exactly those of every other paper map.

The border is computed on the fsaverage mesh (`group_maps.fsaverage_mesh`):
a vertex in the mask with a neighbour outside it is the inner edge, a vertex
outside with a neighbour inside the outer edge. A width of w rings takes
ceil(w/2) rings inward and floor(w/2) outward, so the line is centred on the
true edge. "Outside" includes uncovered vertices, so a significant region
that reaches the edge of coverage is closed there too.

`--min-size N` simplifies the outline for display only: holes of fewer than N
vertices inside a significant region are filled and islands of fewer than N
vertices dropped, so the line follows regions rather than every
non-significant vertex. The map underneath is untouched; say so in the legend
("outlines omit clusters and holes smaller than N vertices").

Dashes (variant c) keep the border vertices in alternate cells of a 3-D grid
laid over the fsaverage *sphere*, where vertex spacing is uniform.

    PYTHONPATH=. python3 scripts/visualisation/contour_maps.py \\
        --root results/group/_contour_test --stat preference --vmax 0.15 \\
        --variants a b c --widths 1 2 3

writes, into <root>/export/conn/,
    <stat>_ctr<variant>w<width>__unthresholded.surf.nii
    <stat>_ctr<variant>w<width>.cmap.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
from conn_figures import SEQUENTIAL, ramp              # noqa: E402
from export_group_conn import to_conn_volume          # noqa: E402

#: |V| <= KEEP maps to colormap rows 3..191 (see module docstring).
KEEP = 1 - 1.5 / 95.5

#: variant -> (positive contour, negative contour, negative dashed)
VARIANTS = {
    "a": ((0, 0, 0), (0, 0, 0), False),          # black both sides
    "b": ((0, 0, 0), (1, 1, 1), False),          # black +, white -
    "g": ((0, 0, 0), (0.3, 0.3, 0.3), False),    # black +, dark grey -
    "c": ((0, 0, 0), (0, 0, 0), True),           # black +, black dashed -
}


def ring_distance(adj, seed: np.ndarray, within: np.ndarray, k: int) -> np.ndarray:
    """Graph distance (in edges, capped at k) from `seed`, moving only through
    `within`; vertices further than k (or unreachable) get k + 1."""
    d = np.full(seed.size, k + 1, dtype=np.int32)
    front = seed & within
    d[front] = 1
    seen = front.copy()
    for step in range(2, k + 1):
        nxt = (adj @ front.astype(np.int8)) > 0
        nxt &= within & ~seen
        d[nxt] = step
        seen |= nxt
        front = nxt
    return d


def border(mask: np.ndarray, adj, width: int) -> np.ndarray:
    """Vertices within the contour band of `mask`, `width` rings wide."""
    if not mask.any():
        return np.zeros_like(mask)
    k_in, k_out = (width + 1) // 2, width // 2
    touches_out = (adj @ (~mask).astype(np.int8)) > 0
    touches_in = (adj @ mask.astype(np.int8)) > 0
    band = np.zeros_like(mask)
    band |= ring_distance(adj, mask & touches_out, mask, k_in) <= k_in
    if k_out:
        band |= ring_distance(adj, ~mask & touches_in, ~mask, k_out) <= k_out
    return band


def simplify(mask: np.ndarray, adj, n: int) -> np.ndarray:
    """Fill holes and drop islands of fewer than n vertices (display only)."""
    from scipy.sparse.csgraph import connected_components
    if n <= 0 or not mask.any():
        return mask
    out = mask.copy()
    for keep_value in (False, True):          # holes first, then islands
        idx = np.flatnonzero(out == keep_value)
        _, lab = connected_components(adj[idx][:, idx], directed=False)
        small = np.bincount(lab)[lab] < n
        out[idx[small]] = not keep_value
    return out


def dash_cells(period: float) -> np.ndarray:
    """Alternate cells of a 3-D grid on the fsaverage sphere, lh then rh."""
    import nibabel.freesurfer.io as fsio
    from plot_group_inflated import fsaverage_dir
    xyz = np.concatenate([fsio.read_geometry(fsaverage_dir() / "surf" / f"{h}.sphere")[0]
                          for h in ("lh", "rh")])
    return (np.floor(xyz / period).astype(int).sum(1) % 2) == 0


def colormap(stat: str, pos, neg) -> np.ndarray:
    """The paper's 192-row diverging map for `stat`, ends replaced."""
    base = stat if stat in SEQUENTIAL else "preference"
    c = np.asarray(ramp(base), float)
    if c.shape != (192, 3):
        raise SystemExit(f"{base}: colormap is {c.shape}, expected (192, 3)")
    c[0], c[-1] = neg, pos
    return c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--stat", required=True,
                    help="<stat>__mean.npy, with <stat>_sigpos / _signeg masks")
    ap.add_argument("--vmax", required=True, type=float,
                    help="the symmetric scale the map is printed on (--vmax of conn_figures)")
    ap.add_argument("--variants", nargs="+", default=["a"], choices=sorted(VARIANTS))
    ap.add_argument("--widths", nargs="+", type=int, default=[2])
    ap.add_argument("--min-size", type=int, default=0,
                    help="fill holes / drop islands smaller than this (vertices)")
    ap.add_argument("--dash", type=float, default=4.0,
                    help="dash cell size on the fsaverage sphere (radius 100)")
    a = ap.parse_args(argv)

    from group_maps import fsaverage_mesh
    adj, _ = fsaverage_mesh()
    vals = np.load(a.root / f"{a.stat}__mean.npy").astype(np.float64)
    pos = np.nan_to_num(np.load(a.root / f"{a.stat}_sigpos__mean.npy")) > 0
    neg = np.nan_to_num(np.load(a.root / f"{a.stat}_signeg__mean.npy")) > 0
    pos, neg = simplify(pos, adj, a.min_size), simplify(neg, adj, a.min_size)
    print(f"{a.stat}: {pos.sum()} positive, {neg.sum()} negative vertices; "
          f"|map| max {np.nanmax(np.abs(vals)):.4f}, printed on +-{a.vmax}")
    base = np.clip(np.nan_to_num(vals), -a.vmax * KEEP, a.vmax * KEEP)
    dashes = None
    out = a.root / "export" / "conn"
    out.mkdir(parents=True, exist_ok=True)
    for w in a.widths:
        bpos, bneg = border(pos, adj, w), border(neg, adj, w)
        for v in a.variants:
            cpos, cneg, dashed = VARIANTS[v]
            bn = bneg
            if dashed:
                if dashes is None:
                    dashes = dash_cells(a.dash)
                bn = bneg & dashes
            x = base.copy()
            x[bn & ~bpos] = -a.vmax
            x[bpos] = a.vmax
            name = f"{a.stat}_ctr{v}w{w}" + (f"m{a.min_size}" if a.min_size else "")
            to_conn_volume(x.astype(np.float32)).to_filename(
                out / f"{name}__unthresholded.surf.nii")
            np.savetxt(out / f"{name}.cmap.txt", colormap(a.stat, cpos, cneg), fmt="%.6f")
            print(f"  {name}: border + {int(bpos.sum())}, - {int(bn.sum())} vertices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
