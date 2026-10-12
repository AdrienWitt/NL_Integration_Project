"""
Compare semantic backbones by their depth profiles, against the carried band.

Every depth sweep launched with ``EXTRA_SOURCES=perlayer_gpt2_k256:8`` scores
the current semantic band beside its own layers, on the same folds and the same
EV mask. So "does this model beat GPT-2" is a *paired* difference taken inside
each subject's own task, not two runs compared after the fact -- and the
reference row doubles as an integrity check: it must come out identical in every
tag for a given subject.

Selection follows the project rule: the layer is chosen on cv (argmax of the
nine-subject mean), and holdout is reported at that cv-chosen layer, never at
its own argmax.

    PYTHONPATH=. python scripts/summarise_model_sweep.py \\
        --tags llama1b_k256 llama3b_k256 llama8b_k256 \\
               llama1b_k1024 llama3b_k1024 llama8b_k1024
"""

import argparse
import csv
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from config import ENCODING_OUT

SUBJECTS = [f"UTS0{i}" for i in range(1, 10)]
REF = "perlayer_gpt2_k256_L8"
#: Runs on different GPUs agree to ~1e-8; the faulty gpu050 shifted every row by
#: ~1e-3. Anything between is worth a look, so the bar sits well below the fault.
TOL = 1e-5


def read(root: Path, tag: str):
    """{config: {subject: mean_r}} over every subject that has a sweep.csv."""
    out = defaultdict(dict)
    for subj in SUBJECTS:
        f = root / f"semantic__{subj}_{tag}" / "sweep.csv"
        if not f.exists():
            continue
        for row in csv.DictReader(open(f, encoding="utf-8")):
            if row["score_kind"] == "text_r":
                out[row["config"]][subj] = float(row["mean_r"])
    return out


def layer_of(cfg: str):
    m = re.fullmatch(r"(.*)_L(\d+)", cfg)
    return (m.group(1), int(m.group(2))) if m else (cfg, None)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--root", default=str(Path(ENCODING_OUT) / "semantic_sweep"))
    p.add_argument("--profile", action="store_true",
                   help="print every layer, not only the cv argmax")
    args = p.parse_args()
    root = Path(args.root)

    ref_seen = defaultdict(list)
    chosen = {}                                   # tag -> (cv dict, ho dict, cfg)
    print(f"{'tag':16s} {'n':>3s} {'best':>5s} {'depth':>5s} "
          f"{'cv':>7s} {'vs gpt2':>8s} {'n+':>4s}   "
          f"{'holdout':>7s} {'vs gpt2':>8s} {'n+':>4s}")
    for tag in args.tags:
        cv, ho = read(root / "cv", tag), read(root / "holdout", tag)
        if REF not in cv:
            print(f"{tag:16s} -- no cv results yet")
            continue
        for s, v in cv[REF].items():
            ref_seen[s].append(v)
        layers = {}
        for cfg in cv:
            store, L = layer_of(cfg)
            if L is not None and cfg != REF:
                layers[L] = cfg
        subj = sorted(set.intersection(*(set(cv[c]) for c in layers.values()),
                                       set(cv[REF])))
        n_top = max(layers)

        def mean(d, cfg, ss):
            return st.mean(d[cfg][s] for s in ss)

        best = max(layers, key=lambda L: mean(cv, layers[L], subj))
        b = layers[best]
        chosen[tag] = (cv, ho, b)
        dcv = [cv[b][s] - cv[REF][s] for s in subj]
        line = (f"{tag:16s} {len(subj):3d} {'L%d' % best:>5s} "
                f"{best / n_top:5.2f} {mean(cv, b, subj):7.4f} "
                f"{st.mean(dcv):+8.4f} {sum(d > 0 for d in dcv):2d}/{len(subj)}")
        hs = [s for s in subj if s in ho.get(b, {}) and s in ho.get(REF, {})]
        if hs:
            dho = [ho[b][s] - ho[REF][s] for s in hs]
            line += (f"   {st.mean(ho[b][s] for s in hs):7.4f} "
                     f"{st.mean(dho):+8.4f} {sum(d > 0 for d in dho):2d}/{len(hs)}")
        print(line)
        if args.profile:
            for L in sorted(layers):
                m = mean(cv, layers[L], subj)
                print(f"    L{L:<3d} {L / n_top:4.2f}  {m:.4f}  "
                      f"{m - mean(cv, REF, subj):+.4f}")

    if len(chosen) > 1:
        print("\npaired, each at its own cv-chosen layer  "
              "(row - column, cv mean, n row ahead)")
        tags = list(chosen)
        for a in tags:
            cells = []
            for c in tags:
                if a == c:
                    cells.append(f"{'':>15s}")
                    continue
                (ca, _, ba), (cc, _, bc) = chosen[a], chosen[c]
                ss = sorted(set(ca[ba]) & set(cc[bc]))
                d = [ca[ba][s] - cc[bc][s] for s in ss]
                cells.append(f"{st.mean(d):+.4f} {sum(x > 0 for x in d)}/{len(d)}".rjust(15))
            print(f"  {a:14s}" + "".join(cells))
        print(f"  {'':14s}" + "".join(f"{t:>15s}" for t in tags))

    drift = {s: max(v) - min(v) for s, v in ref_seen.items()
             if max(v) - min(v) > TOL}
    print(f"\nreference row {REF}: "
          + (f"identical across tags in every subject (within {TOL:g})"
             if not drift
             else f"DIFFERS across tags: {drift} -- folds, masks or a GPU "
                  f"are not shared, and the paired differences are not valid"))


if __name__ == "__main__":
    main()
