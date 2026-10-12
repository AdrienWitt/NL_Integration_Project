#!/usr/bin/env python3
"""Synthetic check of `group_maps.sign_flip_tfce` on the real fsaverage mesh.

A known effect in a ~300-vertex patch of the n>=7 family must be found, and
pure-noise stacks must not produce hits more often than alpha. Missing
subject-vertex values are injected so the null-fill path is exercised too.

    PYTHONPATH=. python3 scripts/check_tfce.py --root results/group/<run>
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import breadth_first_order

sys.path.insert(0, str(Path(__file__).resolve().parent))
import group_maps as gm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--stat", default="delta_audio_given_text")
    ap.add_argument("--n-noise", type=int, default=5)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    adj, area = gm.fsaverage_mesh()
    count = np.load(args.root / f"{args.stat}__n_subjects.npy")
    fam = count >= 7
    idx = np.flatnonzero(fam)
    print(f"family {fam.sum():,} vertices")

    rng = np.random.default_rng(0)
    order = breadth_first_order(adj[fam][:, fam], len(idx) // 3,
                                directed=False, return_predecessors=False)
    blob = idx[order[:300]]
    outside = np.setdiff1d(idx, blob)

    def make(effect):
        x = np.full((9, count.size), np.nan, np.float32)
        x[:, fam] = rng.normal(0, 0.02, (9, fam.sum()))
        x[:, blob] += effect
        x[rng.integers(0, 9, 3000), rng.choice(idx, 3000)] = np.nan
        return x

    t0 = time.time()
    p, _ = gm.sign_flip_tfce(make(0.012), fam, adj, area)
    print(f"effect: {time.time() - t0:.0f} s | detected in "
          f"{np.mean(p[blob] < .05):.0%} of the patch | "
          f"{int(np.nansum(p[outside] < .05))} false positives outside",
          flush=True)

    hits = 0
    for i in range(args.n_noise):
        p, _ = gm.sign_flip_tfce(make(0.0), fam, adj, area)
        hits += np.nanmin(p) < 0.05
        print(f"noise {i}: min p {np.nanmin(p):.3f}", flush=True)
    print(f"noise stacks with any hit: {hits}/{args.n_noise} "
          f"(expected ~{0.05 * args.n_noise:.2f})")


if __name__ == "__main__":
    main()
