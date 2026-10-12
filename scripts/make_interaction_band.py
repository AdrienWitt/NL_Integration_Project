"""
Build a text x audio interaction band: the products of the top principal
components of the two bands, as one feature store.

    PYTHONPATH=. python3 scripts/make_interaction_band.py \\
        --text perlayer_gpt2_k256:8 --audio perlayer_base_emotion:11 \\
        --n-components 20

Why this band exists
--------------------
`delta = min(audio|text, text|audio)` was ported from ../Clean_Irony, but the
irony combined model carried something ours does not: `semantic_* x prosody_*`
product columns (`--use_interaction`, the `results_agefix_interaction` run).
There, "audio beyond text" included how the two combine. Our joint model is
additive, `[text | audio]`, so the same formula only asks whether both bands
carry unique variance in the same voxel -- co-location. In naturalistic
listening both bands predict the whole speech network, so co-location maps
that network: at k256 the min is the audio term at 85% of vertices and
correlates 0.73 with audio|text.

The naturalistic analogue of irony's `semantic x prosody` term is a product of
the two representations. The full product is 768 x 1024 = 786,432 columns, so
it is taken between low-dimensional summaries: the top N principal components
of each band, N x N columns (400 at N = 20). Then

    r(text+audio+inter) - r(text+audio)

is what a *non-additive* combination explains beyond both bands, and
`run_permutation --shuffle-block inter` gives it a Draper-Stoneman null with
both parents held aligned.

Choices, and why
----------------
- **PCA fitted on the training stories only** (`common_stories_all9` minus the
  held-out story), on the design slice `[TR_PAD + trim : -trim]`. It uses no
  brain data, but the held-out story is kept out of every fitted quantity on
  principle, as elsewhere in this project.
- **Columns z-scored before PCA**, because `build_design` z-scores every column
  before the ridge: the components are those of the space the model sees.
- **Component scores scaled to unit variance before the product**, so each
  product column weights its two parents equally rather than the first
  components dominating by their eigenvalues. `build_design` z-scores the
  products again afterwards.
- **Products at the stimulus level, delayed later.** The FIR delays are applied
  to this store like any other, i.e. the interaction happens in the stimulus
  and the haemodynamics stay linear. Multiplying *delayed* features would put
  products of different time points in one column.

What is written
---------------
    <out-name>/<story>.hf5    (n_TRs, N*N), attrs: columns, source stores,
                              n_components, explained variance, fit stories
"""

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np

from common.io import load_features, load_story_json
from config import ENCODING_SPLIT_DIR, FEATURES_DIR, HELD_OUT_STORY, TR_PAD, ensure_dirs

log = logging.getLogger("interaction_band")


def pca_basis(train: list, n: int):
    """Column mean/sd, the top-n principal axes, and the component sd, all
    from the stacked training design slice."""
    x = np.vstack(train).astype(np.float64)
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd == 0] = 1.0
    z = (x - mu) / sd
    _, s, vt = np.linalg.svd(z, full_matrices=False)
    var = s ** 2 / (z.shape[0] - 1)
    axes = vt[:n].T
    score_sd = np.sqrt(var[:n])
    return mu, sd, axes, score_sd, var[:n] / var.sum()


def scores(x: np.ndarray, basis) -> np.ndarray:
    mu, sd, axes, score_sd, _ = basis
    return ((x - mu) / sd) @ axes / score_sd


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", default="perlayer_gpt2_k256:8")
    p.add_argument("--audio", default="perlayer_base_emotion:11")
    p.add_argument("--n-components", type=int, default=20)
    p.add_argument("--stories-json", default="common_stories_all9.json",
                   help="the PCA is fitted on its stories minus the held-out one")
    p.add_argument("--trim", type=int, default=5,
                   help="must match the --trim used for encoding")
    p.add_argument("--out-name", default=None,
                   help="default: inter_<text>_x_<audio>_pc<N>")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: FEATURES_DIR")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    text = load_features(args.text)
    audio = load_features(args.audio)
    stories = sorted(set(text) & set(audio))
    if set(text) ^ set(audio):
        log.warning(f"stories in only one store, skipped: "
                    f"{sorted(set(text) ^ set(audio))}")
    for s in stories:
        if text[s].shape[0] != audio[s].shape[0]:
            raise ValueError(f"{s}: {text[s].shape[0]} text TRs against "
                             f"{audio[s].shape[0]} audio TRs")

    split = load_story_json(Path(ENCODING_SPLIT_DIR) / args.stories_json)
    fit = [s for s in split["common_stories"] if s != HELD_OUT_STORY]
    missing = [s for s in fit if s not in text]
    if missing:
        raise ValueError(f"fit stories absent from the stores: {missing}")
    keep = slice(TR_PAD + args.trim, -args.trim if args.trim else None)
    n = args.n_components

    bt = pca_basis([text[s][keep] for s in fit], n)
    ba = pca_basis([audio[s][keep] for s in fit], n)
    log.info(f"PCA on {len(fit)} stories, "
             f"{sum(text[s][keep].shape[0] for s in fit):,} TRs; {HELD_OUT_STORY} "
             f"excluded")
    log.info(f"  text  {args.text}: {n} components explain "
             f"{bt[4].sum():.1%} of variance (first {bt[4][0]:.1%})")
    log.info(f"  audio {args.audio}: {n} components explain "
             f"{ba[4].sum():.1%} of variance (first {ba[4][0]:.1%})")

    name = args.out_name or (
        f"inter_{args.text.replace(':', 'L')}_x_"
        f"{args.audio.replace(':', 'L')}_pc{n}")
    out = (args.out_dir or Path(FEATURES_DIR)) / name
    ensure_dirs(out)
    columns = np.asarray([f"t{i}_x_a{j}" for i in range(n) for j in range(n)],
                         dtype=object)

    written = 0
    for s in stories:
        path = out / f"{s}.hf5"
        if path.exists() and not args.overwrite:
            continue
        st, sa = scores(text[s], bt), scores(audio[s], ba)
        prod = (st[:, :, None] * sa[:, None, :]).reshape(st.shape[0], n * n)
        with h5py.File(path, "w") as f:
            d = f.create_dataset("data", data=prod.astype(np.float32))
            d.attrs["columns"] = columns
            d.attrs["text_store"] = args.text
            d.attrs["audio_store"] = args.audio
            d.attrs["n_components"] = n
            d.attrs["text_explained"] = bt[4].astype(np.float32)
            d.attrs["audio_explained"] = ba[4].astype(np.float32)
            d.attrs["fit_stories"] = json.dumps(fit)
        written += 1
    log.info(f"wrote {written} of {len(stories)} stories -> {out}")

    # How much of the product band is already linear in its parents: if the
    # ridge could rebuild it from [text | audio], the interaction test would
    # have nothing to find. Pooled over the fit stories, out of a 5-fold split
    # by story so the check is not trivially in-sample.
    xt = np.vstack([scores(text[s][keep], bt) for s in fit])
    xa = np.vstack([scores(audio[s][keep], ba) for s in fit])
    xp = (xt[:, :, None] * xa[:, None, :]).reshape(xt.shape[0], n * n)
    parents = np.hstack([np.vstack([text[s][keep] for s in fit]),
                         np.vstack([audio[s][keep] for s in fit])])
    parents = (parents - parents.mean(0)) / (parents.std(0) + 1e-8)
    lengths = [text[s][keep].shape[0] for s in fit]
    fold = np.repeat(np.arange(len(fit)) % 5, lengths)
    r2 = []
    for k in range(5):
        tr, te = fold != k, fold == k
        a = parents[tr]
        w = np.linalg.solve(a.T @ a + 1e3 * np.eye(a.shape[1]), a.T @ xp[tr])
        pred = parents[te] @ w
        ss = ((xp[te] - xp[te].mean(0)) ** 2).sum(0)
        r2.append(1 - ((xp[te] - pred) ** 2).sum(0) / ss)
    r2 = np.mean(r2, axis=0)
    log.info(f"product columns linearly predictable from [text | audio] "
             f"(ridge, held-out stories): median R2 {np.median(r2):.3f}, "
             f"90th pct {np.percentile(r2, 90):.3f}")


if __name__ == "__main__":
    main()
