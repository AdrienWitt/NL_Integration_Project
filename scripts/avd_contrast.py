"""
Read a three-band arousal / dominance / valence encoding run and answer the
two questions it was fit to answer.

    PYTHONPATH=. python3 scripts/avd_contrast.py --eval holdout

**1. Keep three dimensions, or two?**  Reported as the unique contribution of
each dimension beyond the other two,

    d_dominance = r_joint - r_(arousal+valence)

which is the conditional contribution the text/audio permutations already use,
one level down. A dimension that adds nothing here is a dimension the joint
model can shrink to zero, and carrying it costs interpretability for no fit.
This is deliberately *not* a comparison against openSMILE or the emotion layer
-- those answer "which band predicts better", a different question.

**2. Which dimension does the joint model lean on, where?**  Reported from the
banded-ridge `split_corrs`, each band's share of the joint prediction. Note
these are shares that sum to the joint r, not standalone correlations: a band's
split score is what it contributes *inside* the joint fit.

Read the answer to 2 through the answer to the collinearity report in
`scripts/split_avd_bands.py`. Arousal and dominance correlate ~0.95 in this
model, and banded ridge allocates shared variance rather than dissolving it, so
an argmax over their split scores produces a clean map whether or not the split
means anything. The guard is consistency: a real split wins in most subjects,
a noise split is a coin flip. With n=9 the exact sign test bottoms out at
p = 1/512 = 0.002 (9/9), which is the same floor every other group claim in
this project runs into.

Per-voxel cross-subject maps need a common space -- native voxel counts differ
by subject -- so this script reports subject-level consistency and leaves the
vertexwise map to `scripts/project_to_fsaverage.py`.
"""

import argparse
import json
import logging
from math import comb
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import ENCODING_OUT

log = logging.getLogger("avd")


def sign_test(n_pos: int, n: int) -> float:
    """One-sided exact sign test: P(X >= n_pos) under Binomial(n, 0.5)."""
    return sum(comb(n, k) for k in range(n_pos, n + 1)) / 2 ** n


def load_subject(path: Path) -> Optional[Dict]:
    """Every `<model>_corrs.npy` in one subject directory, plus the mask."""
    corrs = {p.name[: -len("_corrs.npy")]: np.load(p)
             for p in sorted(path.glob("*_corrs.npy"))}
    if not corrs:
        return None
    out: Dict[str, object] = {"corrs": corrs}

    mask_path = path / "voxel_mask.npy"
    if mask_path.exists():
        out["mask"] = np.load(mask_path).astype(bool)
    else:
        # No --min-ev on this run: score everywhere the joint model is defined.
        # Saying so matters -- an unmasked mean r is a different quantity from
        # a mean over EV>0.1 voxels, and the two are not comparable.
        any_corr = next(iter(corrs.values()))
        out["mask"] = np.ones(any_corr.shape[-1], dtype=bool)
        out["unmasked"] = True

    split_path = path / "joint_split_corrs.npy"
    names_path = path / "joint_band_names.json"
    if split_path.exists() and names_path.exists():
        out["split"] = np.load(split_path)
        with open(names_path, encoding="utf-8") as f:
            out["band_names"] = json.load(f)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None,
                   help="run directory (default: "
                        "<ENCODING_OUT>/bands_arousal-dominance-valence)")
    p.add_argument("--backend", default="banded")
    p.add_argument("--eval", default="holdout", choices=["holdout", "cv"])
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    root = Path(args.root or
                Path(ENCODING_OUT) / "bands_arousal-dominance-valence")
    base = root / args.backend / args.eval
    if not base.is_dir():
        raise FileNotFoundError(f"{base} not found — has the run finished?")

    subjects = sorted(d.name for d in base.iterdir() if d.is_dir())
    data = {s: load_subject(base / s) for s in subjects}
    data = {s: d for s, d in data.items() if d}
    if not data:
        raise FileNotFoundError(f"no *_corrs.npy under {base}")

    models: List[str] = sorted({m for d in data.values()
                                for m in d["corrs"]})            # type: ignore
    if any(d.get("unmasked") for d in data.values()):
        log.warning("some subjects have no voxel_mask.npy: those means are "
                    "whole-brain, not over EV>min-ev voxels. Do not compare "
                    "them with the sweeps' masked numbers.")

    # ---- 1. mean r per model ---------------------------------------------
    print(f"\nMean r over the selected voxels — {root.name}, "
          f"{args.backend}/{args.eval}\n")
    width = max(len(m) for m in models)
    print(f"{'subject':<9}" + "".join(f"{m:>{width + 3}}" for m in models)
          + f"{'n voxels':>10}")
    means: Dict[str, Dict[str, float]] = {}
    for s, d in data.items():
        mask = d["mask"]
        row = {}
        for m in models:
            c = d["corrs"].get(m)                                # type: ignore
            row[m] = float(np.nanmean(c[mask])) if c is not None else np.nan
        means[s] = row
        print(f"{s:<9}" + "".join(f"{row[m]:>{width + 3}.4f}" for m in models)
              + f"{int(mask.sum()):>10,}")
    print(f"{'MEAN':<9}" + "".join(
        f"{np.nanmean([means[s][m] for s in data]):>{width + 3}.4f}"
        for m in models))

    # ---- 2. unique contribution of each dimension ------------------------
    bands = next((d["band_names"] for d in data.values()
                  if "band_names" in d), None)
    if bands and "joint" in models:
        print("\nUnique contribution beyond the other dimensions "
              "(r_joint - r_subset):\n")
        for band in bands:
            subset = "+".join(b for b in bands if b != band)
            if subset not in models:
                print(f"  {band:<10} needs --models {subset} — not fitted")
                continue
            deltas, n_pos = [], 0
            for s, d in data.items():
                mask = d["mask"]
                delta = float(np.nanmean(
                    d["corrs"]["joint"][mask] - d["corrs"][subset][mask]))
                deltas.append(delta)
                n_pos += delta > 0
            n = len(deltas)
            print(f"  {band:<10} {np.mean(deltas):+.4f}   "
                  f"positive in {n_pos}/{n}   "
                  f"sign test p = {sign_test(n_pos, n):.4f}   "
                  f"(vs {subset})")
        print("\n  A dimension near zero here is one the joint model can drop:"
              "\n  it costs a column and an interpretation for no fit.")

    # ---- 3. which band carries the joint prediction, where ---------------
    have_split = {s: d for s, d in data.items() if "split" in d}
    if have_split:
        bands = next(iter(have_split.values()))["band_names"]     # type: ignore
        print("\nShare of the joint prediction (split scores), and the "
              "fraction of voxels each band wins:\n")
        print(f"{'subject':<9}" + "".join(f"{b:>22}" for b in bands))
        win_fracs = {b: [] for b in bands}
        for s, d in have_split.items():
            mask, split = d["mask"], d["split"]
            sel = split[:, mask]
            winner = np.argmax(sel, axis=0)
            cells = []
            for i, b in enumerate(bands):
                frac = float((winner == i).mean())
                win_fracs[b].append(frac)
                cells.append(f"{np.nanmean(sel[i]):>+9.4f} ({frac:>5.1%})")
            print(f"{s:<9}" + "".join(f"{c:>22}" for c in cells))
        print(f"\n{'MEAN win %':<9}" + "".join(
            f"{np.mean(win_fracs[b]):>21.1%} " for b in bands))
        print("\n  Split scores are shares of the joint r, not standalone "
              "correlations.\n  Where two bands are collinear the split "
              "between them is allocated, not\n  measured — check "
              "scripts/split_avd_bands.py before reading this as a map.")

    print("\nFor a vertexwise cross-subject map, project these with "
          "scripts/project_to_fsaverage.py\nand count winners per vertex; "
          "native voxel spaces differ across subjects.\n")


if __name__ == "__main__":
    main()
