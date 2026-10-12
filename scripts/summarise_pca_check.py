"""
Read back scripts/cluster/pca_check.sbatch: does PCA on the text band hurt?

Each condition is a single-band text fit under results/encoding/
bands_text__pca_<cond>/banded/<eval>/<subject>/. Rows are paired against the
unprojected band within each subject, on the same mask, so the difference is
the projection and nothing else.

    PYTHONPATH=. python scripts/summarise_pca_check.py
"""

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

from config import ENCODING_OUT

SUBJECTS = [f"UTS0{i}" for i in range(1, 10)]


def score(root: Path, cond: str, ev: str, subj: str):
    d = root / f"bands_text__pca_{cond}" / "banded" / ev / subj
    if not (d / "meta.json").exists():
        return None, None
    r = np.load(d / "text_corrs.npy")
    mask = np.load(d / "voxel_mask.npy") if (d / "voxel_mask.npy").exists() \
        else r != 0
    meta = json.loads((d / "meta.json").read_text())
    return float(np.nanmean(r[mask.astype(bool)])), meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--conds", nargs="+", default=["none", "1024", "256", "0.9"])
    p.add_argument("--root", default=ENCODING_OUT)
    args = p.parse_args()
    root = Path(args.root)

    for ev in ("cv", "holdout"):
        print(f"\n--eval {ev}")
        base = {s: score(root, "none", ev, s)[0] for s in SUBJECTS}
        for c in args.conds:
            rows = {s: score(root, c, ev, s)[0] for s in SUBJECTS}
            ss = [s for s in SUBJECTS if rows[s] is not None and base[s] is not None]
            if not ss:
                print(f"  {c:>6s}  no results yet")
                continue
            m = st.mean(rows[s] for s in ss)
            if c == "none":
                print(f"  {c:>6s}  {m:.4f}   n={len(ss)}")
                continue
            d = [rows[s] - base[s] for s in ss]
            print(f"  {c:>6s}  {m:.4f}  {st.mean(d):+.4f} vs none, "
                  f"PCA ahead {sum(x > 0 for x in d)}/{len(ss)}  "
                  f"(min {min(d):+.4f}, max {max(d):+.4f})")


if __name__ == "__main__":
    main()
