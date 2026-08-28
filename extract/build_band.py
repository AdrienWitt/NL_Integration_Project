"""
Materialise an encoding band from a per-layer feature store.

`extract.wav2vec --per-layer` writes ``(n_TRs, n_layers, hidden)`` per story
and records which transformer layer each stack index came from. This script
collapses a chosen subset of those layers to the ``(n_TRs, n_dim)`` shape
`common.io.load_features` expects, so a layer sweep costs one forward pass over
the audio plus a few seconds of disk per candidate — not a new pass each time.

The averaging here is the same operation `extract.wav2vec` performs without
``--per-layer``: mean over the mean-pooled activations of the chosen layers.
Rebuilding ``--layers 18-23`` from the store reproduces the direct extraction
to float32 rounding — verified on `adollshouse`: correlation 1.000000000000,
max absolute difference 4.6e-05 against a mean magnitude of 7.8. The residual
is reduction-order noise between a GPU `torch.stack(...).mean()` and a NumPy
`mean(axis=1)`, not a difference in what is computed. That is what makes the
store a drop-in replacement for direct extraction.

``--mode concat`` is the alternative worth knowing about: it keeps the layers
side by side instead of averaging, so a banded solver can weight each layer
separately. That strictly dominates averaging — a weighted combination can
represent the plain mean, the reverse is not true — at the cost of
``n_layers * hidden`` columns.

Examples
--------
    # single layer
    python -m extract.build_band --source perlayer_ft_robust --layers 20 \\
        --out-name sweep_ft_robust_L20

    # a range, averaged (reproduces extract.wav2vec --layers 18-23)
    python -m extract.build_band --source perlayer_ft_robust --layers 18-23 \\
        --out-name ft_robust_18to23

    # every stored layer, concatenated for a banded solver
    python -m extract.build_band --source perlayer_ft_emotion --layers all \\
        --mode concat --out-name ft_emotion_allcat
"""

import argparse
import logging
from pathlib import Path
from typing import List

import h5py
import numpy as np

from config import FEATURES_DIR, ensure_dirs

log = logging.getLogger("extract.build_band")


def parse_layers(spec: str, available: np.ndarray) -> List[int]:
    """'all' | '20' | '18-23' | '6,12,18' -> a list of transformer indices."""
    spec = spec.strip().lower()
    if spec == "all":
        return [int(i) for i in available]
    if "-" in spec:
        start, stop = spec.split("-")
        wanted = list(range(int(start), int(stop) + 1))
    else:
        wanted = [int(x) for x in spec.split(",") if x.strip()]

    missing = [i for i in wanted if i not in set(int(a) for a in available)]
    if missing:
        raise SystemExit(
            f"--layers {spec} asks for layer(s) {missing}, which this store "
            f"does not hold. It has {sorted(int(a) for a in available)}. "
            f"Re-run extract.wav2vec --per-layer with a wider --layers, or "
            f"pick from what is there."
        )
    return wanted


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", required=True,
                   help="per-layer directory under data/features")
    p.add_argument("--layers", required=True,
                   help="'all', a single index, a range like 18-23, or a list")
    p.add_argument("--mode", default="mean", choices=["mean", "concat"],
                   help="mean = average the layers (default); concat = keep "
                        "them side by side for per-layer band weighting")
    p.add_argument("--out-name", required=True,
                   help="destination directory under data/features")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    src_dir = Path(FEATURES_DIR) / args.source
    if not src_dir.is_dir():
        raise SystemExit(f"No such per-layer store: {src_dir}")
    stories = sorted(p.stem for p in src_dir.glob("*.hf5"))
    if not stories:
        raise SystemExit(f"{src_dir} holds no .hf5 files")

    # The layer axis is only meaningful with the attribute that labels it.
    with h5py.File(src_dir / f"{stories[0]}.hf5", "r") as f:
        dset = f["data"]
        if dset.ndim != 3:
            raise SystemExit(
                f"{src_dir} is not a per-layer store: data has shape "
                f"{dset.shape}, expected (n_TRs, n_layers, hidden). Build it "
                f"with extract.wav2vec --per-layer."
            )
        if "layers" not in dset.attrs:
            raise SystemExit(
                f"{src_dir} has no 'layers' attribute, so the middle axis is "
                f"unlabelled and any range built from it would be a guess. "
                f"Re-extract with the current extract.wav2vec."
            )
        available = np.asarray(dset.attrs["layers"])

    wanted = parse_layers(args.layers, available)
    index = {int(layer): i for i, layer in enumerate(available)}
    take = [index[i] for i in wanted]

    out_dir = Path(FEATURES_DIR) / args.out_name
    ensure_dirs(out_dir)
    log.info(f"Source  : {src_dir}  (layers {sorted(index)})")
    log.info(f"Layers  : {wanted}  mode={args.mode}")
    log.info(f"Output  : {out_dir}")

    done = skipped = 0
    for story in stories:
        out_path = out_dir / f"{story}.hf5"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        with h5py.File(src_dir / f"{story}.hf5", "r") as f:
            block = f["data"][:, take, :]          # (n_TRs, k, hidden)

        if args.mode == "mean":
            band = block.mean(axis=1)
        else:
            band = block.reshape(block.shape[0], -1)

        with h5py.File(out_path, "w") as f:
            dset = f.create_dataset("data", data=band.astype(np.float32))
            dset.attrs["layers"] = np.asarray(wanted, dtype=np.int32)
            dset.attrs["mode"] = args.mode
        done += 1

    log.info(f"Done: {done} written, {skipped} skipped -> {out_dir}")


if __name__ == "__main__":
    main()
