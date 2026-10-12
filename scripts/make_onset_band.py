"""
Build a story-onset nuisance band: a few smooth functions of time since story
onset, as one feature store on the same TR grid as the other stores.

    PYTHONPATH=. python3 scripts/make_onset_band.py --like perlayer_gpt2_k16:8

Why this band exists
--------------------
`extract/context_lm.py` fills context from the story start, so for the first
k words the input -- and the absolute token position GPT-2 sees -- grows with
time. At k >= 256 that is ~50 TRs per story and survives the trim. Measured
2026-10-02 (84 stories, ridge, folds by story):

    decode time since onset     TR 0-40    TR 60-140
    perlayer_gpt2_k16:8          0.087      0.024
    perlayer_gpt2_k256:8         0.824      0.064
    perlayer_gpt2_k512:8         0.815      0.579
    perlayer_base_emotion:11     ~0.15

The responses carry an onset ramp too (median |r| with log(1+t) over the first
80 TRs is 4-10x its shuffled null, 9/9 subjects), and the voxels where
text|audio jumps from k16 to k256 are the ramp voxels (9/9). So a long-context
text band can score by telling time. Put this band in every model and the
ramp has a home that is not the text band; then compare k on
`r(text+onset) - r(onset)`.

Columns
-------
t is the row index of the store (row 0 = story start, the same origin the
context fill starts from).

    log1p        log(1 + t)
    bump_0..5    raised cosines evenly spaced in log(1 + t), centres from
                 TR 10 to TR 150, each spanning two neighbouring centres --
                 a smooth, flexible onset response without committing to a
                 shape
    frac         t / n_TRs, slow whole-story drift (k512 decodes it at 0.68)

8 columns; run_encoding delays and z-scores them like any other band.
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

from common.io import load_features
from config import FEATURES_DIR, ensure_dirs

log = logging.getLogger("onset_band")


def onset_basis(n: int, n_bumps: int = 6, first: float = 10.0,
                last: float = 150.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    lt = np.log1p(t)
    centres = np.linspace(np.log1p(first), np.log1p(last), n_bumps)
    width = centres[1] - centres[0]
    cols = [lt]
    for c in centres:
        x = np.clip((lt - c) / (2 * width), -0.5, 0.5)
        cols.append(0.5 * (1 + np.cos(2 * np.pi * x)))
    cols.append(t / n)
    return np.stack(cols, axis=1)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--like", default="perlayer_gpt2_k16:8",
                   help="store whose per-story TR counts the band copies")
    p.add_argument("--n-bumps", type=int, default=6)
    p.add_argument("--out-name", default="onset")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    ref = load_features(args.like)
    out = Path(FEATURES_DIR) / args.out_name
    ensure_dirs(out)
    columns = np.asarray(["log1p"] + [f"bump_{i}" for i in range(args.n_bumps)]
                         + ["frac"], dtype=object)
    written = 0
    for s, x in sorted(ref.items()):
        path = out / f"{s}.hf5"
        if path.exists() and not args.overwrite:
            continue
        with h5py.File(path, "w") as f:
            d = f.create_dataset(
                "data", data=onset_basis(x.shape[0], args.n_bumps).astype(np.float32))
            d.attrs["columns"] = columns
            d.attrs["grid_from"] = args.like
        written += 1
    log.info(f"wrote {written} of {len(ref)} stories -> {out}")


if __name__ == "__main__":
    main()
