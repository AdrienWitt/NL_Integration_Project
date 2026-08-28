"""
Layer sweep for the prosodic band: which layer of a speech encoder best
predicts brain responses?

Fits a *prosody-only* encoding model — one band, the audio features — once per
candidate layer configuration, and reports a score per configuration. The
answer to "which layer" is the profile, not a single number: where the curve
peaks says where prosodic information lives in that network, and comparing the
profile of a fine-tuned model against its frozen base says whether fine-tuning
moved it.

Why this is not just a loop over `run_encoding`
-----------------------------------------------
Everything except the audio band is identical across configurations: the same
subject, the same stories, the same responses, the same explainable-variance
mask, the same CV folds. Calling `run_encoding` once per layer would reload and
recompute all of it every time — and reloading the responses is the slowest
step in the whole pipeline. Here they are computed once and reused, so the
marginal cost of another layer is one design build plus one ridge fit.

Using the same folds and the same voxel mask for every candidate is also what
makes the comparison across layers meaningful rather than a comparison of fold
assignments.

Selecting a layer honestly
--------------------------
Default ``--eval cv`` scores by cross-validation **within the training
stories**, and never touches the held-out repeated story. That matters: with a
dozen candidate layers per model, picking the winner by held-out correlation
is selection on the test set, and every number reported afterwards from that
story would be optimistically biased. Sweep with ``--eval cv``, pick the
layer, then evaluate that one choice once with `run_encoding --eval holdout`.

``--eval holdout`` is available for inspecting the profile, but a layer chosen
that way is not a clean out-of-sample result.

Prosody versus everything else in the audio
-------------------------------------------
By default this fits the audio band alone, which is what "encode only prosody"
asks for. Be aware of what that measures: speech audio carries words, and
self-supervised speech representations encode a great deal of phonetic and
lexical structure, so an audio-only score in language regions is partly
semantic. ``--with-text`` adds the semantic band as a covariate and reports the
audio band's *split* score instead — "what does this representation explain
beyond semantics", which is the stricter reading of prosody. Layer *rankings*
are usually stable between the two; absolute values are not.

Examples
--------
Coarse sweep of a frozen 24-layer backbone, single layers::

    python -m encoding.run_prosody_sweep --subjects UTS01 \\
        --source perlayer_base_robust --configs "0 3 6 9 12 15 18 21" \\
        --max-stories 25 --eval cv

Fine sweep around a peak, plus two averaged ranges for comparison::

    python -m encoding.run_prosody_sweep --subjects UTS01 \\
        --source perlayer_ft_robust --configs "17 18 19 20 18-23 12-17" \\
        --max-stories 25 --eval cv
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np

from config import (ENCODING_SPLIT_DIR, ENCODING_OUT, FEATURES_DIR,
                    HELD_OUT_STORY, SUBJECTS, ensure_dirs)
from common.io import (load_features, load_response, load_response_repeats,
                       stories_for_subject, subject_has_story)
from .banded import (default_solver_params, fit_banded, fit_banded_cv,
                     set_himalaya_backend)
from .cv import explainable_variance, story_folds
from .preprocess import build_design, prepare_responses, trim_response

log = logging.getLogger("prosody_sweep")


# --------------------------------------------------------------------------
# Layer configurations
# --------------------------------------------------------------------------

def parse_config(spec: str, available: Sequence[int]) -> List[int]:
    """'20' -> [20] ; '18-23' -> [18..23] ; 'all' -> every stored layer."""
    spec = spec.strip().lower()
    if spec == "all":
        return list(available)
    if "-" in spec:
        start, stop = spec.split("-")
        wanted = list(range(int(start), int(stop) + 1))
    else:
        wanted = [int(x) for x in spec.split(",") if x.strip()]

    missing = [i for i in wanted if i not in set(available)]
    if missing:
        raise SystemExit(
            f"Config {spec!r} asks for layer(s) {missing}; the store holds "
            f"{sorted(available)}. Re-extract with a wider --layers, or drop "
            f"this configuration."
        )
    return wanted


def config_label(spec: str, layers: Sequence[int]) -> str:
    return spec if len(layers) > 1 else f"L{layers[0]}"


# --------------------------------------------------------------------------
# Per-layer store
# --------------------------------------------------------------------------

def store_layers(src_dir: Path, story: str) -> np.ndarray:
    with h5py.File(src_dir / f"{story}.hf5", "r") as f:
        dset = f["data"]
        if dset.ndim != 3:
            raise SystemExit(
                f"{src_dir} is not a per-layer store: data has shape "
                f"{dset.shape}. Build it with extract.wav2vec --per-layer."
            )
        if "layers" not in dset.attrs:
            raise SystemExit(
                f"{src_dir} has no 'layers' attribute, so the middle axis is "
                f"unlabelled. Re-extract with the current extract.wav2vec."
            )
        return np.asarray(dset.attrs["layers"]).astype(int)


def load_store(src_dir: Path, stories: Sequence[str], needed: Sequence[int],
               available: Sequence[int]) -> Dict[str, np.ndarray]:
    """Load only the layers any configuration will use, once, into memory.

    Restricting to the union of requested layers is what keeps this affordable:
    a 24-layer store over 84 stories is ~2.9 GB, but a coarse sweep touches a
    third of the layers and therefore a third of the memory.
    """
    index = {int(layer): i for i, layer in enumerate(available)}
    take = [index[i] for i in needed]
    out = {}
    for story in stories:
        with h5py.File(src_dir / f"{story}.hf5", "r") as f:
            out[story] = f["data"][:, take, :].astype(np.float32)
    return out


def band_from_store(store: Dict[str, np.ndarray], needed: Sequence[int],
                    layers: Sequence[int]) -> Dict[str, np.ndarray]:
    """Average the chosen layers, from the already-loaded subset."""
    pos = {layer: i for i, layer in enumerate(needed)}
    take = [pos[i] for i in layers]
    return {story: arr[:, take, :].mean(axis=1) for story, arr in store.items()}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    data = p.add_argument_group("data")
    data.add_argument("--subjects", default="UTS01",
                      help="'all', or a comma-separated list")
    data.add_argument("--source", required=True,
                      help="per-layer directory under data/features")
    data.add_argument("--configs", required=True,
                      help="space-separated layer configurations, each a "
                           "single index ('20'), a range to average "
                           "('18-23'), or 'all'")
    data.add_argument("--stories-json", default="all_stories.json")
    data.add_argument("--held-out-story", default=HELD_OUT_STORY)
    data.add_argument("--baseline-features", default=None,
                      help="a flat (n_TRs, n_dim) feature directory scored "
                           "alongside the layers under identical folds and "
                           "mask — 'opensmile' gives the eGeMAPS reference "
                           "line, i.e. the target the fine-tuned models were "
                           "trained to predict")
    data.add_argument("--max-stories", type=int, default=None,
                      help="use only the first N training stories. The sweep "
                           "needs the relative ordering of layers, not a "
                           "publishable number, so a subset is usually enough "
                           "and is much cheaper.")

    design = p.add_argument_group("design")
    design.add_argument("--trim", type=int, default=5)
    design.add_argument("--ndelays", type=int, default=4)
    design.add_argument("--with-text", action="store_true",
                        help="add the semantic band as a covariate and score "
                             "the audio band's split correlation instead of a "
                             "plain audio-only correlation")
    design.add_argument("--text-features", default="gpt2_mean",
                        help="semantic band, only used with --with-text")

    model = p.add_argument_group("model")
    model.add_argument("--eval", default="cv", choices=["cv", "holdout"],
                       help="cv = cross-validated within training stories "
                            "(default; the only honest basis for choosing a "
                            "layer). holdout = score on the repeated story.")
    model.add_argument("--alpha-min", type=float, default=0.0)
    model.add_argument("--alpha-max", type=float, default=12.0)
    model.add_argument("--num-alphas", type=int, default=13)
    model.add_argument("--n-splits", type=int, default=5,
                       help="CV folds. The default is 5, not leave-one-story-"
                            "out: fit_banded_cv refits inside every outer "
                            "fold, so 83 folds means 83 nested fits per "
                            "configuration. A sweep needs the ranking of "
                            "layers, and 5 folds gives that at a sixteenth of "
                            "the cost. Pass None-like values only for a final "
                            "single-configuration fit.")
    model.add_argument("--min-ev", type=float, default=0.1,
                       help="fit only voxels with explainable variance above "
                            "this (needs the repeated story for EV)")

    solver = p.add_argument_group("solver")
    solver.add_argument("--solver", default="random_search",
                        choices=["random_search", "hyper_gradient"])
    solver.add_argument("--n-iter", type=int, default=20)
    solver.add_argument("--n-targets-batch", type=int, default=200)
    solver.add_argument("--n-alphas-batch", type=int, default=5)
    solver.add_argument("--himalaya-backend", default="torch_cuda",
                        choices=["torch_cuda", "torch", "numpy", "cupy"])

    out = p.add_argument_group("output")
    out.add_argument("--out", default=None, help="default: results/encoding")
    out.add_argument("--tag", default=None)
    return p


def resolve_subjects(spec: str, stories_json: Path) -> List[str]:
    if spec.strip().lower() == "all":
        return list(SUBJECTS)
    return [s.strip() for s in spec.split(",") if s.strip()]


def run_subject(subject: str, args, src_dir: Path, configs: List[str],
                out_root: Path) -> List[dict]:
    t0 = time.time()
    stories_json = Path(ENCODING_SPLIT_DIR) / args.stories_json

    stories = stories_for_subject(subject, stories_json)
    stories = [s for s in stories if subject_has_story(subject, s)]
    held = args.held_out_story
    train_stories = [s for s in stories if not s.startswith(held)]
    if args.max_stories:
        train_stories = train_stories[: args.max_stories]
    held_out = held if (held in stories and subject_has_story(subject, held)) else None

    if args.eval == "holdout" and held_out is None:
        raise RuntimeError(
            f"{subject}: --eval holdout needs '{held}', which this subject "
            f"does not have. Use --eval cv."
        )
    if args.min_ev > 0 and held_out is None:
        raise RuntimeError(
            f"{subject}: --min-ev needs the repeated story '{held}' to "
            f"estimate explainable variance. Pass --min-ev 0 or use a subject "
            f"that has it."
        )

    all_stories = train_stories + ([held_out] if held_out else [])
    log.info(f"[{subject}] {len(train_stories)} training stories; "
             f"held-out = {held_out or 'NONE'}")

    # ---- layer configurations ------------------------------------------
    available = store_layers(src_dir, all_stories[0])
    parsed = {spec: parse_config(spec, available) for spec in configs}
    needed = sorted({i for layers in parsed.values() for i in layers})
    log.info(f"  store holds layers {sorted(available.tolist())}; "
             f"loading {len(needed)} of them: {needed}")

    store = load_store(src_dir, all_stories, needed, available)
    n_trs = {s: arr.shape[0] for s, arr in store.items()}

    text_all = None
    if args.with_text:
        log.info(f"  loading text band '{args.text_features}' as covariate")
        text_all = load_features(args.text_features, all_stories)
        for s in all_stories:
            if text_all[s].shape[0] != n_trs[s]:
                raise ValueError(
                    f"{s}: text band has {text_all[s].shape[0]} TRs but the "
                    f"audio store has {n_trs[s]} — not the same TR grid."
                )

    # ---- responses, EV, mask, folds: computed once, shared by every config
    #
    # The mask is derived from the held-out story, so it is available before
    # any training response is read — and applying it per story as they load
    # is the difference between a 0.4 GB matrix and an 18 GB one for 83
    # stories at 81k voxels. The numbers are identical either way: both
    # z-scoring and mean-centring act per voxel, so dropping columns first
    # cannot change the columns that remain.
    Y_test_fit = voxel_mask = None
    n_voxels = None

    if held_out is not None:
        repeats = load_response_repeats(held_out, subject)
        trimmed = np.stack([trim_response(r, n_trs[held_out], args.trim)
                            for r in repeats])
        ev = explainable_variance(trimmed)
        n_voxels = int(ev.size)
        Y_test_full = prepare_responses(trimmed.mean(axis=0))
        del trimmed, repeats
        log.info(f"  EV>{args.min_ev} in {(ev > args.min_ev).sum():,}/"
                 f"{n_voxels:,} voxels")
        if args.min_ev > 0:
            voxel_mask = ev > args.min_ev
        Y_test_fit = (Y_test_full[:, voxel_mask] if voxel_mask is not None
                      else Y_test_full)
        del Y_test_full

    blocks = []
    for s in train_stories:
        block = trim_response(load_response([s], subject), n_trs[s], args.trim)
        if n_voxels is None:
            n_voxels = int(block.shape[1])
        blocks.append(block[:, voxel_mask] if voxel_mask is not None else block)
    Y_train_fit = prepare_responses(np.vstack(blocks))
    del blocks
    log.info(f"  Y_train {Y_train_fit.shape} "
             f"({'masked' if voxel_mask is not None else 'all voxels'})")

    alphas = np.logspace(args.alpha_min, args.alpha_max, args.num_alphas)
    solver_params = default_solver_params(
        n_iter=args.n_iter, n_targets_batch=args.n_targets_batch,
        n_alphas_batch=args.n_alphas_batch,
    )
    set_himalaya_backend(args.himalaya_backend)

    save_dir = out_root / subject
    ensure_dirs(save_dir)
    rows = []

    splits = None

    def score_band(spec: str, label: str, layers: Sequence[int],
                   audio: Dict[str, np.ndarray]) -> dict:
        """Fit one prosodic band and summarise it. Identical folds and mask
        for every caller, which is what makes the rows comparable."""
        nonlocal splits

        spaces = {"audio": {s: audio[s] for s in train_stories}}
        if args.with_text:
            spaces = {"text": {s: text_all[s] for s in train_stories}, **spaces}

        design = build_design(train_stories, spaces, trim=args.trim,
                              ndelays=args.ndelays)
        if splits is None:
            # Depends only on the story layout, which never changes here, so
            # every configuration is scored on identical folds.
            splits = story_folds(design.story_ids, args.n_splits)
            log.info(f"  {len(splits)} CV folds (shared by every configuration)")

        if Y_train_fit.shape[0] != design.X.shape[0]:
            raise ValueError(
                f"{subject}/{label}: design has {design.X.shape[0]} TRs but "
                f"the response has {Y_train_fit.shape[0]} — alignment failed."
            )

        log.info(f"  [{label}] layers={layers} X={design.X.shape}")
        if args.eval == "holdout":
            test_spaces = {"audio": {held_out: audio[held_out]}}
            if args.with_text:
                test_spaces = {"text": {held_out: text_all[held_out]},
                               **test_spaces}
            design_test = build_design([held_out], test_spaces, trim=args.trim,
                                       ndelays=args.ndelays,
                                       fitted_pca=design.fitted_pca)
            result = fit_banded(
                X_train=design.X, Y_train=Y_train_fit,
                X_test=design_test.X, Y_test=Y_test_fit,
                bands=design.bands, splits=splits, alphas=alphas,
                solver=args.solver, solver_params=solver_params,
            )
        else:
            result = fit_banded_cv(
                X=design.X, Y=Y_train_fit, bands=design.bands,
                story_ids=design.story_ids, outer_splits=splits, alphas=alphas,
                solver=args.solver, solver_params=solver_params, logger=None,
            )

        # With a covariate, the prosody answer is the audio band's split score,
        # not the whole model's correlation.
        if args.with_text and result.split_corrs is not None:
            idx = result.band_names.index("audio")
            score = result.split_corrs[idx]
            score_kind = "audio_split_r"
        else:
            score = result.corrs
            score_kind = "audio_r"

        full = np.full(n_voxels, np.nan, dtype=np.float64)
        if voxel_mask is not None:
            full[voxel_mask] = score
        else:
            full[:] = score
        np.save(save_dir / f"{label}_{score_kind}.npy", full)

        row = {
            "subject": subject,
            "config": spec,
            "label": label,
            "layers": layers,
            "n_layers": len(layers),
            "score_kind": score_kind,
            "n_voxels_fit": int(score.size),
            "mean_r": float(np.nanmean(score)),
            "median_r": float(np.nanmedian(score)),
            "max_r": float(np.nanmax(score)),
            "n_above_0.1": int((score > 0.1).sum()),
            "top1pct_mean_r": float(np.nanmean(
                np.sort(score)[-max(1, score.size // 100):])),
        }
        log.info(f"  [{label}] mean {score_kind}={row['mean_r']:+.4f} "
                 f"median={row['median_r']:+.4f} max={row['max_r']:.4f} "
                 f"r>0.1 in {row['n_above_0.1']:,}")
        return row

    # A flat reference band — eGeMAPS is the obvious one — scored under exactly
    # the same folds, mask and alphas. Without it the layer profile is a curve
    # with no zero line: you can see which layer wins, but not whether any of
    # them beats the handcrafted descriptors the models were distilled from.
    if args.baseline_features:
        log.info(f"  baseline band '{args.baseline_features}'")
        flat = load_features(args.baseline_features, all_stories)
        for st in all_stories:
            if flat[st].shape[0] != n_trs[st]:
                raise ValueError(
                    f"{st}: baseline band has {flat[st].shape[0]} TRs but the "
                    f"audio store has {n_trs[st]} — not the same TR grid."
                )
        rows.append(score_band(args.baseline_features, args.baseline_features,
                               [], flat))

    for spec in configs:
        layers = parsed[spec]
        rows.append(score_band(spec, config_label(spec, layers), layers,
                               band_from_store(store, needed, layers)))

    log.info(f"[{subject}] {len(rows)} configurations in "
             f"{(time.time() - t0) / 60:.1f} min")
    return rows


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    src_dir = Path(FEATURES_DIR) / args.source
    if not src_dir.is_dir():
        raise SystemExit(f"No such per-layer store: {src_dir}")

    configs = [c for c in args.configs.split() if c.strip()]
    if not configs:
        raise SystemExit("--configs is empty")

    name = args.source + ("__withtext" if args.with_text else "")
    if args.tag:
        name += f"__{args.tag}"
    out_root = Path(args.out or ENCODING_OUT) / "prosody_sweep" / args.eval / name
    ensure_dirs(out_root)

    stories_json = Path(ENCODING_SPLIT_DIR) / args.stories_json
    subjects = resolve_subjects(args.subjects, stories_json)

    log.info(f"Source     : {src_dir}")
    log.info(f"Configs    : {configs}")
    log.info(f"Subjects   : {', '.join(subjects)}")
    log.info(f"Eval       : {args.eval}"
             f"{' (+ text covariate)' if args.with_text else ' (prosody only)'}")
    log.info(f"Output     : {out_root}")

    t0 = time.time()
    rows = []
    for subject in subjects:
        rows.extend(run_subject(subject, args, src_dir, configs, out_root))

    with open(out_root / "sweep.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)

    header = ["subject", "config", "n_layers", "score_kind", "mean_r",
              "median_r", "max_r", "n_above_0.1", "top1pct_mean_r"]
    with open(out_root / "sweep.csv", "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in header) + "\n")

    log.info(f"\n{'config':>10s} {'n_lay':>5s} {'mean_r':>9s} {'median_r':>9s} "
             f"{'max_r':>8s} {'r>0.1':>8s}")
    for r in sorted(rows, key=lambda r: -r["mean_r"]):
        log.info(f"{r['config']:>10s} {r['n_layers']:5d} {r['mean_r']:+9.4f} "
                 f"{r['median_r']:+9.4f} {r['max_r']:8.4f} "
                 f"{r['n_above_0.1']:8,d}")

    log.info(f"\nBest by mean_r: {max(rows, key=lambda r: r['mean_r'])['config']}")
    log.info(f"All configurations finished in {(time.time() - t0) / 60:.1f} min")
    log.info(f"Summary written to {out_root / 'sweep.csv'}")


if __name__ == "__main__":
    main()
