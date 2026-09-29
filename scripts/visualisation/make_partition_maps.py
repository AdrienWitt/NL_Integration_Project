#!/usr/bin/env python3
"""
Text vs audio, per region: the variance-partitioning maps in one directory.

Reads the two group runs of one design and writes a third, render-ready:

    <cond>/   delta_text_given_audio, delta_audio_given_text, delta  (DS nulls)
    <marg>/   preference = r_text - r_audio, k_favouring_text
      ->  <out>/<name>__mean.npy          already thresholded, 0 outside
          <out>/partition.json            counts and provenance

Every map is thresholded *here*, so `export_group_conn.py --test none` writes
exactly what is below and nothing gets thresholded twice.

    unique_text          r_joint - r_audio, group perm FDR q<.05, clipped >= 0
    unique_audio         r_joint - r_text,  group perm FDR q<.05, clipped >= 0
    integration          delta = min of the two, IU test FDR q<.05, clipped >= 0
    preference           r_text - r_audio, every covered vertex (signed)
    preference_consistent  same, where >= 8 of 9 subjects agree on the sign
    preference_fwe       same, sign-flip FWE p<.05 (n=9 family only)

`preference` is also the difference of the two unique maps -- the joint term
cancels, (r_j - r_a) - (r_j - r_t) = r_t - r_a -- so it is the partition's own
contrast, not a separate analysis. It has no permutation null (the prediction
shuffle tests "any signal", not "text beats audio"), and an exact sign test at
n=9 cannot survive FDR, hence the consistency map: 8/9 is a two-sided sign
test p = 0.039 per vertex, uncorrected, and is labelled as such.

Negative means are clipped from the magnitude maps: a group permutation can
reject at a vertex whose mean is slightly negative (the null is more
negative), and CONN would draw it in the opposite colour.

    python scripts/visualisation/make_partition_maps.py \\
        --conditional results/group/k256L8__base_emotionL11__conditional \\
        --marginal    results/group/k256L8__base_emotionL11__marginalfixed \\
        --out         results/group/k256L8__base_emotionL11__partition
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(d: Path, name: str) -> np.ndarray:
    return np.load(d / f"{name}.npy").astype(np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditional", required=True, type=Path)
    ap.add_argument("--marginal", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-agree", type=int, default=8,
                    help="subjects agreeing on the sign of preference")
    args = ap.parse_args(argv)
    c, m, a = args.conditional, args.marginal, args.alpha

    t = load(c, "delta_text_given_audio__mean")
    au = load(c, "delta_audio_given_text__mean")
    d = load(c, "delta__mean")
    covered = np.isfinite(t)

    def sig(stat: str) -> np.ndarray:
        p = load(c, f"{stat}__p_perm_fdr")
        return covered & np.isfinite(p) & (p < a)

    t_sig = sig("delta_text_given_audio")
    a_sig = sig("delta_audio_given_text")
    d_sig = sig("delta")

    pref = load(m, "preference__mean")
    k = load(m, "preference__k_favouring_text")
    n = load(m, "preference__n_subjects")
    fwe = load(m, "preference__p_signflip_fwe")
    pcov = np.isfinite(pref)
    agree = pcov & ((k >= args.min_agree) | (n - k >= args.min_agree))
    fwe_sig = pcov & np.isfinite(fwe) & (fwe < a)

    def mag(x, mask):
        return np.where(mask, np.clip(np.nan_to_num(x), 0, None), 0).astype(np.float32)

    def signed(mask):
        return np.where(mask, np.nan_to_num(pref), 0).astype(np.float32)

    maps = {
        "unique_text": mag(t, t_sig),
        "unique_audio": mag(au, a_sig),
        "integration": mag(d, d_sig),
        "preference": signed(pcov),
        "preference_consistent": signed(agree),
        "preference_fwe": signed(fwe_sig),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, v in maps.items():
        np.save(args.out / f"{name}__mean.npy", v)

    ok = covered & pcov
    report = {
        "conditional": str(c), "marginal": str(m), "alpha": a,
        "covered_vertices": int(covered.sum()),
        "unique_text_sig": int(t_sig.sum()),
        "unique_audio_sig": int(a_sig.sum()),
        "both_sig": int((t_sig & a_sig).sum()),
        "text_only_sig": int((t_sig & ~a_sig).sum()),
        "audio_only_sig": int((a_sig & ~t_sig).sum()),
        "integration_sig": int(d_sig.sum()),
        "preference_text_ahead": int((pref[pcov] > 0).sum()),
        "preference_audio_ahead": int((pref[pcov] < 0).sum()),
        "consistent_text": int((agree & (pref > 0)).sum()),
        "consistent_audio": int((agree & (pref < 0)).sum()),
        "fwe_text": int((fwe_sig & (pref > 0)).sum()),
        "fwe_audio": int((fwe_sig & (pref < 0)).sum()),
        "fwe_family": int(np.isfinite(fwe).sum()),
        "min_agree": args.min_agree,
        # the identity check: unique_text - unique_audio == preference
        "corr_unique_diff_vs_preference": float(
            np.corrcoef((t - au)[ok], pref[ok])[0, 1]),
    }
    (args.out / "partition.json").write_text(json.dumps(report, indent=2))
    for key, val in report.items():
        print(f"{key:34s} {val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
