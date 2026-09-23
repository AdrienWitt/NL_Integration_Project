"""
Split the 3-column surprisal stores into bands that can carry their own alpha.

    PYTHONPATH=. python3 scripts/split_surprisal_bands.py --ks 0 4 16 64 256

`extract.surprisal` writes `(n_TRs, 3)` per story -- surprisal, entropy,
word_rate -- which is the right thing to store and the wrong thing to fit as
one band.

Why word_rate cannot stay in the same band
------------------------------------------
Both prediction columns are word-level signals resampled onto TR times with
`lanczosinterp2D`, so their amplitude scales with how many words fell in the
window. Measured on the design slice, 84 stories, 28,088 TRs:

    r(surprisal, word_rate)   k0 +0.976   k4 +0.841   k16 +0.766   k256 +0.740
    r(entropy,   word_rate)   k0 +1.000   k4 +0.958   k16 +0.935   k256 +0.928

So the stored band is mostly a word-rate regressor wearing a surprisal label,
and at k=0 the entropy column is word_rate exactly (r = 1.000 to three
decimals): with no context the per-word entropy is a constant, and a constant
resampled by word arrival IS the word-rate signal. Banded ridge assigns one
alpha per *band*, not per column, so leaving word_rate inside means the
nuisance and the signal are shrunk together and no contrast can separate them.
Split out, `r_joint - r_wordrate` is the contribution of surprisal and entropy
beyond word rate -- the same conditional-contribution logic the text/audio
permutations use.

It also fixes the k comparison. word_rate is bit-identical across k (this
script checks it), so a marginal comparison across k would hand every k the
same strong nuisance and hand k=0 the most of it -- the argmax would land on
the least contextual model for the least contextual reason.

What is written
---------------
    wordrate              1 column, once, shared by every k
    surp_pred_k<K>        2 columns, [surprisal, entropy] at that context

The prediction pair stays together because surprisal and entropy are the same
distribution's first and second summaries (partial r = 0.555 at k256 with word
rate removed) and nothing here asks which of the two is carrying the effect.
Split them further if that question ever gets asked.
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

from config import FEATURES_DIR, TR_PAD, ensure_dirs

log = logging.getLogger("split_surprisal")

EXPECTED = ["surprisal", "entropy", "word_rate"]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ks", nargs="+", type=int, default=[0, 4, 16, 64, 256],
                   help="context lengths; store is surprisal_gpt2_k<K>")
    p.add_argument("--store-prefix", default="surprisal_gpt2_k")
    p.add_argument("--pred-prefix", default="surp_pred_k")
    p.add_argument("--wordrate-name", default="wordrate")
    p.add_argument("--trim", type=int, default=5,
                   help="must match the --trim used for encoding")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    feats = Path(FEATURES_DIR)
    srcs = {k: feats / f"{args.store_prefix}{k}" for k in args.ks}
    for k, src in srcs.items():
        if not src.is_dir():
            raise FileNotFoundError(
                f"{src} not found. Run scripts/cluster/extract_surprisal.sbatch first.")

    stories = sorted(q.name for q in srcs[args.ks[0]].glob("*.hf5"))
    if not stories:
        raise FileNotFoundError(f"{srcs[args.ks[0]]} holds no .hf5 files")
    log.info(f"{len(stories)} stories, k = {args.ks}")

    wr_dir = feats / args.wordrate_name
    pred_dirs = {k: feats / f"{args.pred_prefix}{k}" for k in args.ks}
    for d in [wr_dir, *pred_dirs.values()]:
        ensure_dirs(d)

    keep = slice(TR_PAD + args.trim, -args.trim if args.trim else None)
    written = 0
    mismatched = []
    corr_rows = []

    for name in stories:
        cols = {}
        for k in args.ks:
            with h5py.File(srcs[k] / name, "r") as f:
                arr = np.asarray(f["data"], dtype=np.float32)
                got = [str(q) for q in f["data"].attrs["columns"]]
            if got != EXPECTED:
                raise ValueError(f"{name} k{k}: columns {got}, expected {EXPECTED}")
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(f"{name} k{k}: shape {arr.shape}, expected (n, 3)")
            cols[k] = arr

        # word_rate must be the same signal at every k, or "the same nuisance
        # across the grid" is false and one shared store would be wrong.
        ref = cols[args.ks[0]][:, 2]
        for k in args.ks[1:]:
            if not np.allclose(ref, cols[k][:, 2], atol=1e-5):
                mismatched.append((name, k))

        wr_path = wr_dir / name
        if args.overwrite or not wr_path.exists():
            with h5py.File(wr_path, "w") as f:
                d = f.create_dataset("data", data=ref[:, None].astype(np.float32))
                d.attrs["columns"] = np.asarray(["word_rate"], dtype=object)
                d.attrs["source_store"] = f"{args.store_prefix}{args.ks[0]}"
            written += 1

        for k in args.ks:
            out = pred_dirs[k] / name
            if not args.overwrite and out.exists():
                continue
            with h5py.File(out, "w") as f:
                d = f.create_dataset("data", data=cols[k][:, :2])
                d.attrs["columns"] = np.asarray(["surprisal", "entropy"],
                                                dtype=object)
                d.attrs["source_store"] = f"{args.store_prefix}{k}"
                d.attrs["context_words"] = int(k)
            written += 1

        block = {k: cols[k][keep] for k in args.ks}
        if block[args.ks[0]].shape[0] > 4:
            corr_rows.append(block)

    if mismatched:
        raise ValueError(
            f"word_rate differs across k in {len(mismatched)} (story, k) pairs, "
            f"e.g. {mismatched[:3]}. One shared wordrate store would be wrong.")
    log.info(f"word_rate identical across k in all {len(stories)} stories")
    log.info(f"wrote {written} files")

    # The report is the reason the split exists; print it so a reader of the
    # log can see the confound rather than trust this docstring.
    pooled = {k: np.vstack([b[k] for b in corr_rows]) for k in args.ks}
    wr = pooled[args.ks[0]][:, 2]
    log.info(f"design slice: {wr.size} TRs")
    for k in args.ks:
        rs = np.corrcoef(pooled[k][:, 0], wr)[0, 1]
        re = np.corrcoef(pooled[k][:, 1], wr)[0, 1]
        log.info(f"  k{k:<4d} r(surprisal, word_rate) {rs:+.3f}   "
                 f"r(entropy, word_rate) {re:+.3f}")


if __name__ == "__main__":
    main()
