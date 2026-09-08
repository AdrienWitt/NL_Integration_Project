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
import re
import json
import os
import statistics as st

from config import ENCODING_OUT


def config_sort_key(cfg: str):
    """Order '0','1',...,'23','12-17','15-18' — layers first, ranges after.

    The semantic sweep labels configurations by name rather than by depth
    ('gpt2_k16', 'perlayer_gpt2_k16_L8'), which no numeric key can order.
    Those sort alphabetically, after everything numeric, instead of raising.
    """
    if "-" in cfg:
        try:
            start, stop = cfg.split("-")
            return (1, int(start), int(stop), "")
        except ValueError:
            return (2, 0, 0, cfg)
    try:
        return (0, int(cfg), 0, "")
    except ValueError:
        pass
    # 'perlayer_gpt2_k16_L8' — a named store carrying a depth. Sorting those
    # as text puts L10 between L1 and L2, which destroys the one thing a
    # depth profile is read for.
    m = re.fullmatch(r"(.*)_L(\d+)", cfg)
    if m:
        return (0, int(m.group(2)), 0, m.group(1))
    return (2, 0, 0, cfg)


def config_display(cfg: str) -> str:
    """'8' -> 'L8'; a named configuration is already its own label."""
    return f"L{cfg}" if cfg.lstrip("-").isdigit() else cfg


def load(root):
    d = collections.defaultdict(dict)
    for path in glob.glob(os.path.join(root, "*", "sweep.csv")):
        name = os.path.basename(os.path.dirname(path))
        store, _, subject = name.partition("__")
        store = store.replace("perlayer_", "")
        # `--tag "${SUBJ}${TAGSUF}"` puts the run's tag after the subject, so
        # "semantic__UTS01_layers" is subject UTS01 of the *layers* run. Left
        # in the subject field, two runs of one sweep merge into a single
        # store holding 24 "subjects" and two incompatible config lists.
        m = re.fullmatch(r"(UTS\d+)_(.+)", subject)
        if m:
            store, subject = f"{store}_{m.group(2)}", m.group(1)
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                d[(store, subject or row["subject"])][row["config"]] = \
                    float(row["mean_r"])
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", default="cv", choices=["cv", "holdout"])
    p.add_argument("--root", default=None,
                   help="default: results/encoding/prosody_sweep/<eval>. Point "
                        "it at semantic_sweep/<eval> to read that one, with "
                        "--baseline gpt2_mean")
    p.add_argument("--baseline", default="opensmile",
                   help="the configuration every other one is scored against, "
                        "and which must appear in every cell. 'opensmile' for "
                        "the prosody sweep, 'gpt2_mean' for the semantic one")
    p.add_argument("--json", default=None, help="also write the profile here")
    p.add_argument("--tidy", default=None,
                   help="write one long CSV with every store/subject/layer row "
                        "— the single file to open in R or pandas")
    args = p.parse_args()

    root = args.root or os.path.join(ENCODING_OUT, "prosody_sweep", args.eval)
    data = load(root)
    if not data:
        raise SystemExit(f"No sweep.csv under {root}")

    stores = sorted({k[0] for k in data})
    subs = sorted({k[1] for k in data})
    print(f"{len(data)} store-subject cells · {len(stores)} stores · "
          f"{len(subs)} subjects\n")

    missing = [k for k, cell in data.items() if args.baseline not in cell]
    if missing:
        raise SystemExit(
            f"--baseline {args.baseline!r} is absent from {len(missing)} of "
            f"{len(data)} cells, e.g. {missing[0]}. Every difference reported "
            f"below is paired within a cell, so a cell without the baseline "
            f"cannot contribute. Re-run those with --baseline-features "
            f"{args.baseline}, or name the baseline this sweep actually used.")

    print(f"{args.baseline} reproducibility (same subject, independent jobs):")
    for sub in subs:
        vals = [data[(s, sub)][args.baseline] for s in stores
                if (s, sub) in data and args.baseline in data[(s, sub)]]
        if len(vals) > 1:
            print(f"  {sub}  n={len(vals)}  r={st.mean(vals):.5f}  "
                  f"spread={max(vals) - min(vals):.1e}")

    out = {}
    for store in stores:
        have = [s for s in subs if (store, s) in data]
        # Configs are either a single layer ("18") or an averaged range
        # ("15-18"), so int() alone raises. Single layers sort numerically
        # first, ranges after them by their start, which keeps the profile
        # readable as a depth axis with the composites gathered at the end.
        #
        # The union over subjects, not the first subject's list: a sweep read
        # while it is still running has subjects that stopped at different
        # configurations, and indexing every subject by subject one's list
        # raises on the first configuration the others have not reached.
        cfgs = sorted({c for s in have for c in data[(store, s)]
                       if c != args.baseline}, key=config_sort_key)
        print(f"\n{store}  (Δr vs own {args.baseline}, n={len(have)})")
        rows = []
        for c in cfgs:
            have_c = [s for s in have if c in data[(store, s)]]
            if len(have_c) < 2:
                print(f"  {config_display(c):<24} (only {len(have_c)} subject"
                      f"{'' if len(have_c) == 1 else 's'} so far, skipped)")
                continue
            diff = [data[(store, s)][c] - data[(store, s)][args.baseline]
                    for s in have_c]
            m, se = st.mean(diff), st.stdev(diff) / len(diff) ** 0.5
            npos = sum(x > 0 for x in diff)
            rows.append({"layer": c, "delta": round(m, 5),
                         "se": round(se, 5), "n_positive": npos,
                         "n": len(have_c),
                         "abs_r": round(st.mean(data[(store, s)][c]
                                                for s in have_c), 5)})
            print(f"  {config_display(c):<24} {m:+.4f} ± {se:.4f}   "
                  f"{npos}/{len(have_c)} subjects"
                  f"   {'#' * max(0, round(m * 2000))}")
        out[store] = rows

    if args.tidy:
        # One row per (store, subject, layer). The per-task sweep.csv files are
        # already one-row-per-layer; this only concatenates them and carries the
        # subject's own openSMILE score alongside, so the paired difference can
        # be recomputed downstream without re-reading 36 files.
        import csv as _csv
        with open(args.tidy, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(["store", "subject", "config", "mean_r",
                        f"{args.baseline}_r", f"delta_vs_{args.baseline}"])
            for store in stores:
                for sub in subs:
                    cell = data.get((store, sub))
                    if not cell or args.baseline not in cell:
                        continue
                    base = cell[args.baseline]
                    for c, v in sorted(
                            cell.items(),
                            key=lambda kv: ((3, 0, 0, "") if kv[0] == args.baseline
                                            else config_sort_key(kv[0]))):
                        w.writerow([store, sub, c, f"{v:.6f}",
                                    f"{base:.6f}", f"{v - base:.6f}"])
        print(f"\nwrote {args.tidy}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
