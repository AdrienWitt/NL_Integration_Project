"""
Turn the per-subject sweep.csv files into the layer profile.

Reads every ``results/encoding/prosody_sweep/<eval>/<store>__<subject>/sweep.csv``
and reports, per store and layer, the mean correlation *relative to that
subject's own openSMILE score*.

Why relative and paired
-----------------------
Absolute correlation varies about threefold across subjects (UTS05 0.0072,
UTS04 0.0245), for reasons that have nothing to do with which layer was used —
head motion, coverage, how many stories they heard. An unpaired mean over
subjects is therefore dominated by which subjects a store happens to hold, and
two stores fitted on different subject sets are not comparable at all. Taking
each subject's difference from their own baseline first removes that, and the
baseline is free: openSMILE is scored inside every store's task on identical
folds and mask.

That redundancy is also the integrity check this prints first. The same subject's
openSMILE score is computed in up to four separate SLURM jobs on different nodes;
if the shared folds and EV mask are deterministic, those must agree. They do, to
about 1e-8.

    python scripts/summarise_sweep.py
    python scripts/summarise_sweep.py --eval cv --json profiles.json
"""

import argparse
import collections
import csv
import glob
import json
import os
import statistics as st

from config import ENCODING_OUT


def load(root):
    d = collections.defaultdict(dict)
    for path in glob.glob(os.path.join(root, "*", "sweep.csv")):
        name = os.path.basename(os.path.dirname(path))
        store, _, subject = name.partition("__")
        store = store.replace("perlayer_", "")
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                d[(store, subject or row["subject"])][row["config"]] = \
                    float(row["mean_r"])
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", default="cv", choices=["cv", "holdout"])
    p.add_argument("--root", default=None)
    p.add_argument("--json", default=None, help="also write the profile here")
    args = p.parse_args()

    root = args.root or os.path.join(ENCODING_OUT, "prosody_sweep", args.eval)
    data = load(root)
    if not data:
        raise SystemExit(f"No sweep.csv under {root}")

    stores = sorted({k[0] for k in data})
    subs = sorted({k[1] for k in data})
    print(f"{len(data)} store-subject cells · {len(stores)} stores · "
          f"{len(subs)} subjects\n")

    print("openSMILE reproducibility (same subject, independent jobs):")
    for sub in subs:
        vals = [data[(s, sub)]["opensmile"] for s in stores
                if (s, sub) in data and "opensmile" in data[(s, sub)]]
        if len(vals) > 1:
            print(f"  {sub}  n={len(vals)}  r={st.mean(vals):.5f}  "
                  f"spread={max(vals) - min(vals):.1e}")

    out = {}
    for store in stores:
        have = [s for s in subs if (store, s) in data]
        cfgs = sorted((c for c in data[(store, have[0])] if c != "opensmile"),
                      key=int)
        print(f"\n{store}  (Δr vs own openSMILE, n={len(have)})")
        rows = []
        for c in cfgs:
            diff = [data[(store, s)][c] - data[(store, s)]["opensmile"]
                    for s in have]
            m, se = st.mean(diff), st.stdev(diff) / len(diff) ** 0.5
            npos = sum(x > 0 for x in diff)
            rows.append({"layer": int(c), "delta": round(m, 5),
                         "se": round(se, 5), "n_positive": npos,
                         "abs_r": round(st.mean(data[(store, s)][c]
                                                for s in have), 5)})
            print(f"  L{c:<3} {m:+.4f} ± {se:.4f}   {npos}/{len(have)} subjects"
                  f"   {'#' * max(0, round(m * 2000))}")
        out[store] = rows

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
