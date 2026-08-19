"""
Run the block-permutation test for one or more subjects.

Each model is fit once on the training stories and predicts the held-out
story; the null is then built by block-shuffling the observed responses, with
the same shuffle applied to all three models on every iteration so that the
delta null is valid (see `stats/permutation.py`).

Example
-------
    python -m stats.run_permutation --subjects all \\
        --text-features gpt2_mean --audio-features opensmile \\
        --n-perms 1000 --blocklen 10 --min-ev 0.1
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from config import ENCODING_OUT, HELD_OUT_STORY, ensure_dirs
from common.io import load_response_repeats, save_results
from encoding.banded import (default_solver_params, fit_banded,
                             set_himalaya_backend)
from encoding.cv import explainable_variance, story_folds
from encoding.preprocess import build_design, prepare_responses, trim_response
from encoding.run_encoding import (MODEL_BANDS, load_aligned_response,
                                   load_bands, resolve_stories,
                                   resolve_subjects, _subset_bands)
from .permutation import permutation_null, summarize

log = logging.getLogger("permutation")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--subjects", default="all")
    p.add_argument("--text-features", default="gpt2_mean")
    p.add_argument("--audio-features", default="opensmile")
    p.add_argument("--stories-json", default="all_stories.json")
    p.add_argument("--held-out-story", default=HELD_OUT_STORY)
    p.add_argument("--max-stories", type=int, default=None)

    p.add_argument("--trim", type=int, default=5)
    p.add_argument("--ndelays", type=int, default=4)
    p.add_argument("--use-pca", action="store_true")
    p.add_argument("--n-comps", type=float, default=0.90)

    p.add_argument("--n-perms", type=int, default=1000)
    p.add_argument("--blocklen", type=int, default=10,
                   help="permutation block length in TRs (20 s at TR=2 s)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05, help="FDR level")
    p.add_argument("--min-ev", type=float, default=0.1,
                   help="fit and test only voxels above this explainable variance")

    p.add_argument("--alpha-min", type=float, default=1.0)
    p.add_argument("--alpha-max", type=float, default=20.0)
    p.add_argument("--num-alphas", type=int, default=20)
    p.add_argument("--solver", default="random_search")
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--n-targets-batch", type=int, default=200)
    p.add_argument("--n-alphas-batch", type=int, default=5)
    p.add_argument("--himalaya-backend", default="torch_cuda")

    p.add_argument("--save-null", action="store_true",
                   help="also save the full (n_perms, n_voxels) null arrays; "
                        "these are large")
    p.add_argument("--out", default=None)
    p.add_argument("--tag", default=None)
    return p.parse_args(argv)


def run_subject(subject: str, args, out_root: Path) -> None:
    t0 = time.time()
    stories_json = Path(args.stories_json)
    if not stories_json.is_absolute():
        from config import ENCODING_SPLIT_DIR
        stories_json = Path(ENCODING_SPLIT_DIR) / args.stories_json

    train_stories, held_out = resolve_stories(subject, args, stories_json)
    if held_out is None:
        raise RuntimeError(
            f"{subject}: no repeated story '{args.held_out_story}' — the "
            f"permutation test needs the held-out story."
        )
    log.info(f"[{subject}] {len(train_stories)} training stories, "
             f"held-out '{held_out}'")

    all_stories = train_stories + [held_out]
    features = load_bands(args, all_stories)
    feature_lengths = {s: features["text"][s].shape[0] for s in all_stories}

    train_features = {
        band: {s: a for s, a in feats.items() if s in train_stories}
        for band, feats in features.items()
    }
    design = build_design(train_stories, train_features, trim=args.trim,
                          ndelays=args.ndelays, use_pca=args.use_pca,
                          n_comps=args.n_comps)
    design_test = build_design(
        [held_out], {b: {held_out: f[held_out]} for b, f in features.items()},
        trim=args.trim, ndelays=args.ndelays, use_pca=args.use_pca,
        n_comps=args.n_comps, fitted_pca=design.fitted_pca,
    )

    Y_train = prepare_responses(
        load_aligned_response(subject, train_stories, feature_lengths, args.trim)
    )
    repeats = load_response_repeats(held_out, subject)
    trimmed = np.stack([trim_response(r, feature_lengths[held_out], args.trim)
                        for r in repeats])
    ev = explainable_variance(trimmed)
    Y_test = prepare_responses(trimmed.mean(axis=0))

    mask = ev > args.min_ev if args.min_ev > 0 else np.ones(ev.shape, bool)
    log.info(f"  testing {mask.sum():,}/{ev.size:,} voxels (EV > {args.min_ev})")
    n_voxels = ev.size

    Y_train_fit, Y_test_fit = Y_train[:, mask], Y_test[:, mask]
    splits = story_folds(design.story_ids)
    alphas = np.logspace(args.alpha_min, args.alpha_max, args.num_alphas)
    solver_params = default_solver_params(
        n_iter=args.n_iter, n_targets_batch=args.n_targets_batch,
        n_alphas_batch=args.n_alphas_batch,
    )

    set_himalaya_backend(args.himalaya_backend)

    observed, predictions = {}, {}
    for model_name, band_names in MODEL_BANDS.items():
        log.info(f"  fitting {model_name}")
        result = fit_banded(
            X_train=design.X, Y_train=Y_train_fit,
            X_test=design_test.X, Y_test=Y_test_fit,
            bands=_subset_bands(design.bands, band_names),
            splits=splits, alphas=alphas, solver=args.solver,
            solver_params=solver_params, compute_splits=False,
            return_predictions=True,
        )
        observed[model_name] = result.corrs
        predictions[model_name] = result.predictions

    observed["delta"] = (observed["joint"]
                         - np.maximum(observed["text"], observed["audio"]))

    log.info(f"  {args.n_perms} block permutations (blocklen={args.blocklen})")
    null = permutation_null(
        Y_test_fit, predictions, n_perms=args.n_perms, blocklen=args.blocklen,
        seed=args.seed, logger=log, progress_every=max(1, args.n_perms // 10),
    )

    save_dir = out_root / subject
    payload, report = {}, {}
    for name, obs in observed.items():
        stats = summarize(obs, null[name], name, alpha=args.alpha)

        def _scatter(values, fill=0.0, dtype=float):
            full = np.full(n_voxels, fill, dtype=dtype)
            full[mask] = values
            return full

        payload[f"{name}_observed"] = _scatter(obs)
        payload[f"{name}_pvals"] = _scatter(stats["pvals"], fill=1.0)
        payload[f"{name}_pvals_fdr"] = _scatter(stats["pvals_fdr"], fill=1.0)
        payload[f"{name}_significant"] = _scatter(stats["reject"], fill=False,
                                                  dtype=bool)
        if args.save_null:
            payload[f"{name}_null"] = null[name].astype(np.float32)

        report[name] = {
            "n_significant": stats["n_significant"],
            "n_tested": stats["n_tested"],
            "mean": stats["mean"],
            "max": stats["max"],
        }
        log.info(f"    {name:6s} mean={stats['mean']:+.4f} "
                 f"max={stats['max']:+.4f} "
                 f"FDR q<{args.alpha}: {stats['n_significant']:,} voxels")

    payload["ev"] = ev
    payload["voxel_mask"] = mask
    payload["report"] = {
        "subject": subject, "n_perms": args.n_perms, "blocklen": args.blocklen,
        "min_ev": args.min_ev, "fdr_alpha": args.alpha,
        "held_out_story": held_out, "n_train_stories": len(train_stories),
        "results": report,
    }
    save_results(save_dir, payload)
    log.info(f"[{subject}] done in {(time.time() - t0) / 60:.1f} min "
             f"-> {save_dir}")


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    from config import ENCODING_SPLIT_DIR
    stories_json = Path(ENCODING_SPLIT_DIR) / args.stories_json
    subjects = resolve_subjects(args.subjects, stories_json)

    name = f"{args.text_features}__{args.audio_features}"
    if args.tag:
        name = f"{name}__{args.tag}"
    out_root = Path(args.out or (Path(ENCODING_OUT) / name / "permutation"))
    ensure_dirs(out_root)
    log.info(f"Subjects: {', '.join(subjects)} | output: {out_root}")

    for subject in subjects:
        try:
            run_subject(subject, args, out_root)
        except (FileNotFoundError, KeyError, RuntimeError) as exc:
            log.error(f"[{subject}] skipped: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
