"""
Split the 3-d `emotion_avd` store into one store per affective dimension, and
report how separable those dimensions actually are.

    PYTHONPATH=. python3 scripts/split_avd_bands.py

`extract.emotion_avd` writes one `(n_TRs, 3)` file per story. `run_encoding`
selects *stores*, not columns — `store:9-11` means transformer layers of a 3-D
per-layer store, not columns of a 2-D one — so fitting arousal, dominance and
valence as three bands of one banded ridge needs three directories. They are
tiny (one float per TR) and naming them explicitly is what keeps
`--band arousal=emotion_arousal` readable at the call site.

Why the correlation report is part of this script
-------------------------------------------------
Whether the third dimension is worth keeping is decided by two things, and
this is the cheap one. Measured on the 774 stimuli of ../Clean_Irony, the same
model's outputs give

    arousal x dominance   0.950        variance per component:
    arousal x valence     0.223            0.639 / 0.348 / 0.013
    dominance x valence   0.274

i.e. an effective rank of 2, not 3. If that holds on these stimuli too, then
"where is dominance encoded rather than arousal" is not an identifiable
question: banded ridge does not dissolve collinearity, it *allocates* it, and
a winner-take-all map will still look clean because an argmax never abstains.

This script only measures the design side. The other half — does dominance add
anything to the fit beyond the other two — is `--models arousal+valence joint`
in `run_encoding`, read with `scripts/avd_contrast.py`.

The slice matches the design
----------------------------
Correlations are computed on ``[TR_PAD + trim : -trim]``, the rows
`encoding.preprocess` actually keeps. The head rows are padding duplicated by
`load_simulated_trfiles`; leaving them in would count the same TR several
times.
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

from config import FEATURES_DIR, TR_PAD, ensure_dirs
from finetune import EMOTION_DIMENSIONS

log = logging.getLogger("split_avd")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", default="emotion_avd",
                   help="3-column store written by extract.emotion_avd")
    p.add_argument("--out-prefix", default="emotion",
                   help="stores are <prefix>_<dimension>")
    p.add_argument("--trim", type=int, default=5,
                   help="must match the --trim used for encoding")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--report-only", action="store_true",
                   help="measure the correlations, write nothing")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    src = Path(FEATURES_DIR) / args.store
    if not src.is_dir():
        raise FileNotFoundError(
            f"{src} not found. Run `python -m extract.emotion_avd "
            f"--output avd` first (~20 min on one GPU).")

    files = sorted(src.glob("*.hf5"))
    if not files:
        raise FileNotFoundError(f"{src} holds no .hf5 files")

    dims = list(EMOTION_DIMENSIONS)          # arousal, dominance, valence
    out_dirs = [Path(FEATURES_DIR) / f"{args.out_prefix}_{d}" for d in dims]
    if not args.report_only:
        for d in out_dirs:
            ensure_dirs(d)

    keep = slice(TR_PAD + args.trim, -args.trim if args.trim else None)
    pooled, per_story = [], []
    written = skipped = 0

    for path in files:
        with h5py.File(path, "r") as f:
            arr = np.asarray(f["data"], dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != len(dims):
            raise ValueError(
                f"{path.name}: expected (n_TRs, {len(dims)}), got {arr.shape}. "
                f"Was this written with --output avd rather than hidden/both?")

        block = arr[keep]
        if block.shape[0] > len(dims) + 1:
            pooled.append(block)
            with np.errstate(invalid="ignore"):
                per_story.append(np.corrcoef(block.T))

        if args.report_only:
            continue
        for i, out_dir in enumerate(out_dirs):
            out_path = out_dir / path.name
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue
            with h5py.File(out_path, "w") as f:
                dset = f.create_dataset(
                    "data", data=arr[:, i:i + 1].astype(np.float32))
                dset.attrs["dimension"] = dims[i]
                dset.attrs["source_store"] = args.store
            written += 1

    if not args.report_only:
        log.info(f"wrote {written} files, skipped {skipped} "
                 f"(--overwrite to replace) across {len(out_dirs)} stores")
        for d in out_dirs:
            log.info(f"    {d}")

    X = np.concatenate(pooled)
    C = np.corrcoef(X.T)
    sv = np.linalg.svd(X - X.mean(0), compute_uv=False)
    var = sv ** 2 / (sv ** 2).sum()

    print(f"\n{len(files)} stories, {X.shape[0]:,} TRs on the design slice "
          f"[{TR_PAD + args.trim}:-{args.trim}]\n")
    width = max(len(d) for d in dims)
    print(" " * (width + 2) + "".join(f"{d:>12}" for d in dims))
    for i, d in enumerate(dims):
        print(f"{d:<{width}}  " + "".join(f"{C[i, j]:>12.3f}"
                                          for j in range(len(dims))))

    # Per-story spread says whether the pooled number is a stable property of
    # the model or an artefact of pooling stories with different means.
    S = np.stack(per_story)
    print("\nper-story range of each off-diagonal correlation:")
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            v = S[:, i, j]
            print(f"  {dims[i]:<{width}} x {dims[j]:<{width}}  "
                  f"median {np.median(v):+.3f}   "
                  f"[{v.min():+.3f}, {v.max():+.3f}]")

    print("\nvariance per component: " + "  ".join(f"{v:.3f}" for v in var))
    n_eff = int((np.cumsum(var) < 0.99).sum() + 1)
    print(f"components to reach 99% of the variance: {n_eff} of {len(dims)}")

    worst = max(((abs(C[i, j]), dims[i], dims[j])
                 for i in range(len(dims)) for j in range(i + 1, len(dims))))
    if worst[0] >= 0.9:
        print(f"\n=> {worst[1]} and {worst[2]} share "
              f"{worst[0] ** 2:.0%} of their variance. Their separate maps are "
              f"not identifiable; report the pair as one axis, and gate any "
              f"per-dimension claim on cross-subject replication.")
    else:
        print(f"\n=> largest pairwise |r| is {worst[0]:.3f} "
              f"({worst[1]} x {worst[2]}); all three dimensions are separable.")


if __name__ == "__main__":
    main()
