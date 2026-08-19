"""
Fit voxelwise encoding models on the LeBel (ds003020) dataset.

Three models are fit from the *same* design matrix and the *same* CV folds:

    text   — semantics only   (e.g. GPT-2 embeddings)
    audio  — prosody only     (e.g. eGeMAPS, or fine-tuned wav2vec2)
    joint  — both bands

which is what makes the downstream contrasts interpretable:

    delta      = r_joint - max(r_text, r_audio)     -> integration
    preference = r_text - r_audio                   -> semantic vs prosodic
    split      = per-band r inside the joint model  -> banded ridge only

Examples
--------
Banded ridge, held-out story, all subjects, prosody = eGeMAPS::

    python -m encoding.run_encoding --subjects all \\
        --text-features gpt2_mean --audio-features opensmile \\
        --backend banded --eval holdout --min-ev 0.1

Same, but with the fine-tuned wav2vec2 features and both backends::

    python -m encoding.run_encoding --subjects UTS01,UTS02 \\
        --text-features gpt2_mean --audio-features wav2vec_mean_layers18to23 \\
        --backend both --eval holdout
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (ENCODING_SPLIT_DIR, ENCODING_OUT, HELD_OUT_STORY, SUBJECTS,
                    ensure_dirs)
from common.io import (load_features, load_response, load_response_repeats,
                       save_results, stories_for_subject, subject_has_story)
from .banded import (default_solver_params, fit_banded, fit_banded_cv,
                     set_himalaya_backend)
from .cv import explainable_variance, story_folds
from .huth_ridge import fit_huth
from .preprocess import build_design, prepare_responses, trim_response

log = logging.getLogger("encoding")

#: Which bands each model is allowed to see.
MODEL_BANDS: Dict[str, List[str]] = {
    "text":  ["text"],
    "audio": ["audio"],
    "joint": ["text", "audio"],
}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    data = p.add_argument_group("data")
    data.add_argument("--subjects", default="all",
                      help="'all', or a comma-separated list (UTS01,UTS02)")
    data.add_argument("--text-features", default="gpt2_mean",
                      help="feature directory under data/features for the semantic band")
    data.add_argument("--audio-features", default="opensmile",
                      help="feature directory under data/features for the prosodic band")
    data.add_argument("--stories-json", default="all_stories.json",
                      help="story list in data/derivative (subject -> stories)")
    data.add_argument("--held-out-story", default=HELD_OUT_STORY,
                      help="repeated story reserved for testing; never trained on")
    data.add_argument("--max-stories", type=int, default=None,
                      help="use only the first N training stories (smoke tests)")

    design = p.add_argument_group("design")
    design.add_argument("--trim", type=int, default=5,
                        help="TRs trimmed from each story (see preprocess.trim_story)")
    design.add_argument("--ndelays", type=int, default=4,
                        help="number of FIR delays (1..ndelays TRs)")
    design.add_argument("--use-pca", action="store_true",
                        help="reduce each band with PCA before delaying")
    design.add_argument("--n-comps", type=float, default=0.90,
                        help="PCA components: <=1 means explained variance")

    model = p.add_argument_group("model")
    model.add_argument("--models", nargs="+", default=["text", "audio", "joint"],
                       choices=sorted(MODEL_BANDS))
    model.add_argument("--backend", default="banded",
                       choices=["banded", "huth", "both"],
                       help="banded = himalaya per-band alphas (primary); "
                            "huth = single shared alpha (conservative check)")
    model.add_argument("--eval", default="holdout", choices=["holdout", "cv"],
                       help="holdout = score on the repeated held-out story; "
                            "cv = nested cross-validation over training stories")
    model.add_argument("--alpha-min", type=float, default=1.0,
                       help="log10 of the smallest alpha")
    model.add_argument("--alpha-max", type=float, default=20.0,
                       help="log10 of the largest alpha")
    model.add_argument("--num-alphas", type=int, default=20)
    model.add_argument("--n-splits", type=int, default=None,
                       help="CV folds; default = leave-one-story-out")
    model.add_argument("--min-ev", type=float, default=0.0,
                       help="fit only voxels with explainable variance above "
                            "this (holdout eval only). 0.1 is a good default "
                            "and cuts runtime a lot.")

    solver = p.add_argument_group("solver")
    solver.add_argument("--solver", default="random_search",
                        choices=["random_search", "hyper_gradient"])
    solver.add_argument("--n-iter", type=int, default=20,
                        help="random_search iterations over band weightings")
    solver.add_argument("--n-targets-batch", type=int, default=200)
    solver.add_argument("--n-alphas-batch", type=int, default=5)
    solver.add_argument("--himalaya-backend", default="torch_cuda",
                        choices=["torch_cuda", "torch", "numpy", "cupy"])
    solver.add_argument("--n-jobs", type=int, default=1,
                        help="parallel jobs for the huth backend")

    out = p.add_argument_group("output")
    out.add_argument("--out", default=None,
                     help="output root (default: results/encoding)")
    out.add_argument("--tag", default=None,
                     help="extra label appended to the output directory")
    out.add_argument("--overwrite", action="store_true",
                     help="refit even if results already exist")

    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------

def resolve_subjects(spec: str, stories_json: Path) -> List[str]:
    if spec != "all":
        return [s.strip() for s in spec.split(",") if s.strip()]
    with open(stories_json, encoding="utf-8") as f:
        data = json.load(f)
    info = data.get("dataset_info", {})
    if "participants" in info:
        return list(info["participants"])
    return sorted(data.get("participants", {})) or list(SUBJECTS)


def resolve_stories(subject: str, args, stories_json: Path
                    ) -> Tuple[List[str], Optional[str]]:
    """Training stories for `subject`, plus the held-out story if available."""
    stories = stories_for_subject(subject, stories_json)

    held = args.held_out_story
    # Drop the held-out story and any repeat-suffixed variant of it, otherwise
    # the test story leaks into training under a different name.
    train = [s for s in stories if not s.startswith(held)]

    if args.max_stories:
        train = train[: args.max_stories]

    have_held = held in stories and subject_has_story(subject, held)
    return train, (held if have_held else None)


def load_bands(args, stories: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    log.info(f"  loading text band  '{args.text_features}'")
    text = load_features(args.text_features, stories)
    log.info(f"  loading audio band '{args.audio_features}'")
    audio = load_features(args.audio_features, stories)
    return {"text": text, "audio": audio}


def load_aligned_response(subject: str, stories: List[str],
                          feature_lengths: Dict[str, int], trim: int
                          ) -> np.ndarray:
    """Concatenated response, trimmed onto the same grid as the design."""
    blocks, offsets = [], set()
    for story in stories:
        resp = load_response([story], subject)
        offsets.add(resp.shape[0] - feature_lengths[story])
        blocks.append(trim_response(resp, feature_lengths[story], trim))
    log.info(f"  response/feature grid offset(s): {sorted(offsets)} "
             f"(0 = padded grid, 5 = raw acquisition grid)")
    return np.vstack(blocks)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def _subset_bands(bands: Dict[str, slice], names: List[str]) -> Dict[str, slice]:
    return {name: bands[name] for name in names}


def fit_one_model(model_name: str, backend: str, args, design, Y_train,
                  design_test=None, Y_test=None, splits=None) -> Dict[str, object]:
    """Fit `model_name` with `backend`; returns arrays ready to save."""
    band_subset = _subset_bands(design.bands, MODEL_BANDS[model_name])
    n_cols = sum(s.stop - s.start for s in band_subset.values())
    log.info(f"    [{backend}] {model_name}: bands={list(band_subset)} "
             f"({n_cols} columns)")

    alphas = np.logspace(args.alpha_min, args.alpha_max, args.num_alphas)

    if backend == "banded":
        solver_params = default_solver_params(
            n_iter=args.n_iter,
            n_targets_batch=args.n_targets_batch,
            n_alphas_batch=args.n_alphas_batch,
        )
        if args.eval == "holdout":
            result = fit_banded(
                X_train=design.X, Y_train=Y_train,
                X_test=design_test.X, Y_test=Y_test,
                bands=band_subset, splits=splits, alphas=alphas,
                solver=args.solver, solver_params=solver_params,
            )
        else:
            result = fit_banded_cv(
                X=design.X, Y=Y_train, bands=band_subset,
                story_ids=design.story_ids, outer_splits=splits, alphas=alphas,
                solver=args.solver, solver_params=solver_params, logger=log,
            )
        return result.as_dict()

    # huth: single shared alpha, so the design must be physically sliced
    columns = np.concatenate([np.arange(s.start, s.stop)
                              for s in band_subset.values()])
    X_train = design.X[:, columns]
    X_test = design_test.X[:, columns] if design_test is not None else None

    result = fit_huth(
        X_train=X_train, Y_train=Y_train, story_ids=design.story_ids,
        alphas=alphas, X_test=X_test, Y_test=Y_test,
        final_test=(args.eval == "holdout"),
        nsplits=args.n_splits, n_jobs=args.n_jobs, logger=log,
    )
    return result.as_dict()


def run_subject(subject: str, args, out_root: Path) -> None:
    t0 = time.time()
    stories_json = Path(ENCODING_SPLIT_DIR) / args.stories_json
    train_stories, held_out = resolve_stories(subject, args, stories_json)

    log.info(f"[{subject}] {len(train_stories)} training stories; "
             f"held-out = {held_out or 'NONE'}")

    if args.eval == "holdout" and held_out is None:
        raise RuntimeError(
            f"{subject}: --eval holdout needs the repeated story "
            f"'{args.held_out_story}', which this subject does not have. "
            f"Use --eval cv instead."
        )

    all_stories = train_stories + ([held_out] if held_out else [])
    features = load_bands(args, all_stories)
    feature_lengths = {s: features["text"][s].shape[0] for s in all_stories}

    train_features = {
        band: {s: arr for s, arr in feats.items() if s in train_stories}
        for band, feats in features.items()
    }
    design = build_design(
        train_stories, train_features, trim=args.trim, ndelays=args.ndelays,
        use_pca=args.use_pca, n_comps=args.n_comps,
    )
    log.info(f"  train {design}")

    Y_train = prepare_responses(
        load_aligned_response(subject, train_stories, feature_lengths, args.trim)
    )
    if Y_train.shape[0] != design.X.shape[0]:
        raise ValueError(
            f"{subject}: design has {design.X.shape[0]} TRs but the response "
            f"has {Y_train.shape[0]} after trimming — alignment failed."
        )
    log.info(f"  Y_train {Y_train.shape}")

    design_test, Y_test, ev = None, None, None
    voxel_mask = None

    if args.eval == "holdout":
        test_features = {
            band: {held_out: feats[held_out]} for band, feats in features.items()
        }
        design_test = build_design(
            [held_out], test_features, trim=args.trim, ndelays=args.ndelays,
            use_pca=args.use_pca, n_comps=args.n_comps,
            fitted_pca=design.fitted_pca,       # never refit on the test story
        )
        repeats = load_response_repeats(held_out, subject)
        trimmed = np.stack([
            trim_response(rep, feature_lengths[held_out], args.trim)
            for rep in repeats
        ])
        ev = explainable_variance(trimmed)
        Y_test = prepare_responses(trimmed.mean(axis=0))
        log.info(f"  test {design_test} | Y_test {Y_test.shape} | "
                 f"EV>0.1 in {(ev > 0.1).sum():,}/{ev.size:,} voxels")

        if args.min_ev > 0:
            voxel_mask = ev > args.min_ev
            log.info(f"  fitting {voxel_mask.sum():,} voxels with EV > {args.min_ev}")

    splits = story_folds(design.story_ids, args.n_splits)
    log.info(f"  {len(splits)} CV folds (shared by every model)")

    n_voxels = Y_train.shape[1]
    Y_train_fit = Y_train[:, voxel_mask] if voxel_mask is not None else Y_train
    Y_test_fit = (Y_test[:, voxel_mask] if (voxel_mask is not None and
                                            Y_test is not None) else Y_test)

    backends = ["banded", "huth"] if args.backend == "both" else [args.backend]

    for backend in backends:
        if backend == "banded":
            set_himalaya_backend(args.himalaya_backend)

        for model_name in args.models:
            save_dir = out_root / backend / args.eval / subject
            marker = save_dir / f"{model_name}_corrs.npy"
            if marker.exists() and not args.overwrite:
                log.info(f"    [{backend}] {model_name}: exists, skipping "
                         f"(--overwrite to refit)")
                continue

            arrays = fit_one_model(
                model_name, backend, args, design, Y_train_fit,
                design_test=design_test, Y_test=Y_test_fit, splits=splits,
            )

            # Scatter masked results back into full voxel space so every saved
            # map has the same length and can be compared voxel by voxel.
            payload = {}
            for key, value in arrays.items():
                if key == "band_names":
                    payload[f"{model_name}_{key}"] = value
                    continue
                value = np.asarray(value)
                if voxel_mask is not None and value.shape[-1] == voxel_mask.sum():
                    full = np.zeros(value.shape[:-1] + (n_voxels,), dtype=float)
                    full[..., voxel_mask] = value
                    value = full
                payload[f"{model_name}_{key}"] = value

            save_results(save_dir, payload)
            corrs = payload[f"{model_name}_corrs"]
            scored = corrs[voxel_mask] if voxel_mask is not None else corrs
            log.info(f"    [{backend}] {model_name}: mean r={scored.mean():.4f}, "
                     f"max r={scored.max():.4f}, "
                     f"r>0.1 in {(scored > 0.1).sum():,} voxels")

    meta = {
        "subject": subject,
        "text_features": args.text_features,
        "audio_features": args.audio_features,
        "train_stories": train_stories,
        "held_out_story": held_out,
        "n_train_TRs": int(design.X.shape[0]),
        "n_voxels": int(n_voxels),
        "bands": {k: [v.start, v.stop] for k, v in design.bands.items()},
        "trim": args.trim,
        "ndelays": args.ndelays,
        "use_pca": args.use_pca,
        "n_comps": args.n_comps,
        "eval": args.eval,
        "n_folds": len(splits),
        "alphas": [args.alpha_min, args.alpha_max, args.num_alphas],
        "min_ev": args.min_ev,
        "n_voxels_fit": int(voxel_mask.sum()) if voxel_mask is not None else int(n_voxels),
    }
    for backend in backends:
        save_dir = out_root / backend / args.eval / subject
        extras = {"meta": meta}
        if ev is not None:
            extras["ev"] = ev
            if voxel_mask is not None:
                extras["voxel_mask"] = voxel_mask
        save_results(save_dir, extras)

    log.info(f"[{subject}] done in {(time.time() - t0) / 60:.1f} min")


# --------------------------------------------------------------------------

def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    stories_json = Path(ENCODING_SPLIT_DIR) / args.stories_json
    subjects = resolve_subjects(args.subjects, stories_json)

    name = f"{args.text_features}__{args.audio_features}"
    if args.tag:
        name = f"{name}__{args.tag}"
    out_root = Path(args.out or ENCODING_OUT) / name
    ensure_dirs(out_root)

    log.info(f"Subjects : {', '.join(subjects)}")
    log.info(f"Models   : {', '.join(args.models)}")
    log.info(f"Backend  : {args.backend} | eval: {args.eval}")
    log.info(f"Output   : {out_root}")

    started = time.time()
    for subject in subjects:
        try:
            run_subject(subject, args, out_root)
        except (FileNotFoundError, KeyError, RuntimeError) as exc:
            # One subject missing data should not abandon the whole run.
            log.error(f"[{subject}] skipped: {type(exc).__name__}: {exc}")

    log.info(f"All subjects finished in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
