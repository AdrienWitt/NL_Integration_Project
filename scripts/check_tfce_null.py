#!/usr/bin/env python3
"""Empirical false-positive rate of the TFCE sign-flip on REAL nulls.

`check_tfce.py` uses Gaussian noise, which is symmetric by construction. The
real per-subject nulls need not be, and sign-flipping is exact only if the
centred statistic is symmetric about 0 under H0. So: take null draw *i* as if
it were the observed map for every subject (the nine share a seed, so draw i is
the same shuffle of time for all of them), centre it on the mean of the other
draws exactly as `run_statistic` centres the real map, and run the same TFCE
test. Every draw is H0 by construction; the share of draws with any hit
estimates the family-wise error rate.

    PYTHONPATH=. python3 scripts/check_tfce_null.py \
        --perm-dir /root/prosody_nulls/<run>/permutation \
        --group-dir results/group/<run> --stat <statistic> --n-draws 20
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import group_maps as gm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm-dir", type=Path, required=True)
    ap.add_argument("--group-dir", type=Path, required=True)
    ap.add_argument("--stat", required=True)
    ap.add_argument("--n-draws", type=int, default=20)
    ap.add_argument("--min-subjects", type=int, default=7)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    p2f = gm._load_projector_module()
    subject_dirs = sorted(d for d in args.perm_dir.iterdir()
                          if (d / f"{args.stat}_null.npy").exists())
    rng = np.random.default_rng(1)
    n_perms = np.load(subject_dirs[0] / f"{args.stat}_null.npy",
                      mmap_mode="r").shape[0]
    draws = rng.choice(n_perms, args.n_draws, replace=False)

    # centred[k, s, :] = projected (null_i - mean of the other draws)
    centred = None
    for s, d in enumerate(subject_dirs):
        maps = gm.SubjectMaps(p2f.Projector(d.name), d)
        null = np.load(d / f"{args.stat}_null.npy").astype(np.float64)
        total = null.sum(axis=0)
        scratch = np.zeros(maps.voxel_mask.size)
        for k, i in enumerate(draws):
            loo_mean = (total - null[i]) / (n_perms - 1)
            scratch[:] = 0.0
            scratch[maps.voxel_mask] = null[i] - loo_mean
            proj = maps._project_masked(scratch)
            if centred is None:
                centred = np.full((len(draws), len(subject_dirs), proj.size),
                                  np.nan, dtype=np.float32)
            centred[k, s] = proj
        print(f"  projected {d.name}", flush=True)

    count = np.load(args.group_dir / f"{args.stat}__n_subjects.npy")
    family = count >= args.min_subjects
    adj, area = gm.fsaverage_mesh()

    hits_tfce = hits_fwe = 0
    for k in range(len(draws)):
        p, _ = gm.sign_flip_tfce(centred[k], family, adj, area)
        p_old = gm.sign_flip_fwe(centred[k], count >= 9)
        n_t = int(np.nansum(p < args.alpha))
        hits_tfce += n_t > 0
        hits_fwe += np.nansum(p_old < args.alpha) > 0
        print(f"draw {draws[k]:4d}: TFCE {n_t:6,d} vertices (min p "
              f"{np.nanmin(p):.3f}) | vertex FWE min p {np.nanmin(p_old):.3f}",
              flush=True)
    print(f"\nFWER  TFCE {hits_tfce}/{len(draws)}   vertex FWE "
          f"{hits_fwe}/{len(draws)}   nominal {args.alpha}")


if __name__ == "__main__":
    main()
