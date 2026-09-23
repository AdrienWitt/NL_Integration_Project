"""
Read the surprisal context sweep: what prediction adds beyond word rate, per k.

    PYTHONPATH=. python3 scripts/summarise_surprisal_sweep.py --eval cv

One row per context length, averaged over subjects, over each subject's own
EV mask. The statistic that chooses k is

    r_joint - r_wordrate

not `r_pred`. `pred` alone carries the word-rate component that every k shares
(r(surprisal, word_rate) = +0.74 to +0.98 depending on k), so a comparison of
marginal scores across k is a comparison of how collinear each k is with the
nuisance -- and k=0 wins that by construction. The conditional contribution is
the same quantity the text/audio permutations report, one level down.

Both evals are read because this project chooses on cv and reports on holdout:
over 36 prosody sweeps the two argmaxes agreed in 1, and selecting on holdout
would have inflated the reported score by +0.0083. The holdout column here is
for reporting, never for choosing.

Rows are restricted to the subjects present at EVERY k
------------------------------------------------------
Mean r over an EV mask is not the same quantity across subjects -- at five
repeats each, the masks run 1,706 to 16,395 voxels -- so a k scored over seven
subjects and a k scored over nine are not comparable, and the difference
between them is mostly which subjects happened to finish. The `wordrate` band
is bit-identical across k, which turns this into a free check: on a common
subject set its column MUST repeat exactly down the table. If it does not,
the rows are not the comparison they look like. `--all-subjects` prints the
unrestricted table instead, for watching a run fill in.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from config import ENCODING_OUT


def subject_row(d: Path) -> dict | None:
    """Mean r per model over this subject's EV mask, or None if unfinished."""
    if not (d / "meta.json").exists():
        return None
    mask = np.load(d / "voxel_mask.npy").astype(bool)
    out = {}
    for f in sorted(d.glob("*_corrs.npy")):
        name = f.name[: -len("_corrs.npy")]
        r = np.load(f)
        out[name] = float(np.nanmean(r[mask])) if r.shape[0] == mask.size \
            else float(np.nanmean(r))
    out["_n_voxels"] = int(mask.sum())
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None,
                   help="encoding output root (default: config.ENCODING_OUT)")
    p.add_argument("--prefix", default="bands_wordrate-pred__k")
    p.add_argument("--evals", nargs="+", default=["cv", "holdout"])
    p.add_argument("--tidy", default=None, help="write a tidy CSV here")
    p.add_argument("--all-subjects", action="store_true",
                   help="do not restrict to subjects present at every k")
    args = p.parse_args(argv)

    root = Path(args.root or ENCODING_OUT)
    runs = sorted(root.glob(f"{args.prefix}*"),
                  key=lambda q: int(re.search(r"k(\d+)$", q.name).group(1)))
    if not runs:
        raise SystemExit(f"no {args.prefix}* directories under {root}")

    need = {"wordrate", "pred", "joint"}
    rows = []
    for ev in args.evals:
        # {k: {subject: row}}, keeping only subjects whose three models are
        # all on disk -- a directory exists from the moment a subject STARTS.
        table = {}
        for run in runs:
            k = int(re.search(r"k(\d+)$", run.name).group(1))
            base = run / "banded" / ev
            if not base.is_dir():
                continue
            per = {}
            for sd in sorted(base.iterdir()):
                r = subject_row(sd) if sd.is_dir() else None
                if r and need <= set(r):
                    per[sd.name] = r
            table[k] = per

        common = set.intersection(*(set(v) for v in table.values())) \
            if table and not args.all_subjects else None
        if common is not None and not common:
            counts = ", ".join(f"k{k}={len(v)}"
                               for k, v in sorted(table.items()))
            print(f"\n=== --eval {ev} : no subject is finished at every k "
                  f"({counts}); pass --all-subjects for the partial table ===")
            continue

        scope = ("all finished subjects, NOT comparable across k"
                 if common is None else
                 f"{len(common)} subjects present at every k")
        print(f"\n=== --eval {ev} : mean r over each subject's EV mask "
              f"({scope}) ===")
        print(f"{'k':>6} {'wordrate':>10} {'pred':>9} {'joint':>9} "
              f"{'joint-wr':>10} {'n':>6} {'subj':>5}")
        for k, per in sorted(table.items()):
            use = {s: v for s, v in per.items()
                   if common is None or s in common}
            if not use:
                print(f"{k:>6}   (no finished subject)")
                continue
            got = list(use.values())
            wr = np.mean([v["wordrate"] for v in got])
            pr = np.mean([v["pred"] for v in got])
            jt = np.mean([v["joint"] for v in got])
            d = [v["joint"] - v["wordrate"] for v in got]
            print(f"{k:>6} {wr:10.4f} {pr:9.4f} {jt:9.4f} {np.mean(d):+10.4f} "
                  f"{sum(x > 0 for x in d):>3}/{len(d)} {len(got):>5}")
            for s, v in use.items():
                rows.append({"k": k, "eval": ev, "subject": s,
                             "n_voxels": v["_n_voxels"],
                             **{m: v[m] for m in need},
                             "joint_minus_wordrate": v["joint"] - v["wordrate"]})

        if common is not None:
            wrs = [np.mean([v["wordrate"] for s, v in per.items()
                            if s in common])
                   for k, per in sorted(table.items()) if per]
            spread = max(wrs) - min(wrs) if wrs else 0.0
            flag = "OK" if spread < 5e-4 else "*** MISMATCH ***"
            print(f"       wordrate is one band shared by every k: "
                  f"spread across rows {spread:.5f}  {flag}")

    if args.tidy and rows:
        import csv
        keys = sorted({q for row in rows for q in row})
        with open(args.tidy, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\ntidy -> {args.tidy}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
