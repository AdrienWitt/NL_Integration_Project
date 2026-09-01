"""
Context-length sweep for the semantic band: how much preceding context does a
language model need before its representation stops helping predict the brain?

Fits a *text-only* encoding model, once per candidate feature store, and
reports a score per store. The stores are the `extract.context_lm` bands —
`gpt2_k0`, `gpt2_k1`, `gpt2_k4`, ... — which differ only in how many preceding
words each word was embedded with.

Why this is not the prosody sweep
---------------------------------
`run_prosody_sweep` sweeps *layers inside one* 3-D per-layer store. Context
length cannot be stored that way: each k needs its own forward pass, so it is
one 2-D store per configuration. Everything else is deliberately identical —
the same responses, EV mask, folds and alphas across configurations, computed
once — because that is what makes the rows comparable.

Why context length is a result, not a hyperparameter
----------------------------------------------------
Jain & Huth (2018) found that cortical areas differ in the context length they
prefer, and TRIBE (2025, fig. 6c) reports whole-brain encoding still rising at
1024 words with no plateau. The profile is the finding; the argmax is a
by-product. Per-configuration score maps are written to disk so the
context-length preference can be read voxelwise, not just as a mean.

Choose on cv, report on holdout
-------------------------------
Same rule as the prosody sweep: `--eval cv` never touches the repeated story,
so picking a winner there costs nothing. `--eval holdout` is for reporting a
choice already made.

Examples
--------
    python -m encoding.run_semantic_sweep --subjects UTS01 \\
        --sources "gpt2_k0 gpt2_k1 gpt2_k4 gpt2_k16 gpt2_k64 gpt2_k256" \\
        --stories-json common_stories_all9.json --eval cv

    # the stricter reading: what does context explain beyond the prosody band?
    python -m encoding.run_semantic_sweep --subjects UTS01 \\
        --sources "gpt2_k0 gpt2_k256" --with-audio opensmile --eval cv
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from config import (ENCODING_SPLIT_DIR, ENCODING_OUT, FEATURES_DIR,
                    HELD_OUT_STORY, SUBJECTS, ensure_dirs)
from common.io import (load_features, load_response, load_response_repeats,
                       stories_for_subject, subject_has_story)
from .banded import (default_solver_params, fit_banded, fit_banded_cv,
                     set_himalaya_backend)
from .cv import explainable_variance, story_folds
from .preprocess import build_design, prepare_responses, trim_response

log = logging.getLogger("semantic_sweep")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    data = p.add_argument_group("data")
    data.add_argument("--subjects", default="UTS01",
                      help="comma-separated, or 'all'")
    data.add_argument("--sources", required=True,
                      help="space-separated text feature stores under "
                           "FEATURES_DIR, e.g. \"gpt2_k0 gpt2_k16 gpt2_k256\"")
    data.add_argument("--stories-json", default="all_stories.json")
    data.add_argument("--held-out-story", default=HELD_OUT_STORY)
    data.add_argument("--baseline-features", default=None,
                      help="a flat reference band scored under identical folds "
                           "and mask, e.g. the existing gpt2_mean. Without it "
                           "the profile has no zero line.")
    data.add_argument("--max-stories", type=int, default=None)

    design = p.add_argument_group("design")
    design.add_argument("--trim", type=int, default=5)
    design.add_argument("--ndelays", type=int, default=4)
    design.add_argument("--with-audio", default=None,
                        help="add this audio store as a covariate and report "
                             "the text band's SPLIT score — 'what does the "
                             "semantic band explain beyond prosody'")

    model = p.add_argument_group("model")
    model.add_argument("--eval", default="cv", choices=["cv", "holdout"])
    model.add_argument("--alpha-min", type=float, default=0.0)
    model.add_argument("--alpha-max", type=float, default=12.0)
    model.add_argument("--num-alphas", type=int, default=13)
    model.add_argument("--n-splits", type=int, default=5)
    model.add_argument("--min-ev", type=float, default=0.1)

    solver = p.add_argument_group("solver")
    solver.add_argument("--solver", default="random_search")
    solver.add_argument("--n-iter", type=int, default=20)
    solver.add_argument("--n-targets-batch", type=int, default=200)
    solver.add_argument("--n-alphas-batch", type=int, default=5)
    solver.add_argument("--himalaya-backend", default="torch_cuda")

    out = p.add_argument_group("output")
    out.add_argument("--out", default=None)
    out.add_argument("--tag", default=None)
    return p


def resolve_subjects(spec: str) -> List[str]:
    if spec.strip().lower() == "all":
        return list(SUBJECTS)
    return [s.strip() for s in spec.split(",") if s.strip()]


def run_subject(subject: str, args, sources: List[str],
                out_root: Path) -> List[dict]:
    t0 = time.time()
    stories_json = Path(ENCODING_SPLIT_DIR) / args.stories_json

    stories = stories_for_subject(subject, stories_json)
    stories = [s for s in stories if subject_has_story(subject, s)]
    held = args.held_out_story
    train_stories = [s for s in stories if not s.startswith(held)]
    if args.max_stories:
        train_stories = train_stories[: args.max_stories]
    held_out = held if (held in stories and subject_has_story(subject, held)) \
        else None

    if args.eval == "holdout" and held_out is None:
        raise RuntimeError(f"{subject}: --eval holdout needs '{held}'.")
    if args.min_ev > 0 and held_out is None:
        raise RuntimeError(
            f"{subject}: --min-ev needs the repeated story '{held}' to "
            f"estimate explainable variance. Pass --min-ev 0.")

    all_stories = train_stories + ([held_out] if held_out else [])
    log.info(f"[{subject}] {len(train_stories)} training stories; "
             f"held-out = {held_out or 'NONE'}")

    # The TR grid comes from the first store; every other store must match it
    # exactly, or the design and the responses are not describing the same
    # seconds of the stimulus.
    reference = load_features(sources[0], all_stories)
    n_trs = {s: arr.shape[0] for s, arr in reference.items()}

    audio_all = None
    if args.with_audio:
        log.info(f"  loading audio band '{args.with_audio}' as covariate")
        audio_all = load_features(args.with_audio, all_stories)
        for s in all_stories:
            if audio_all[s].shape[0] != n_trs[s]:
                raise ValueError(
                    f"{s}: audio band has {audio_all[s].shape[0]} TRs but the "
                    f"text store has {n_trs[s]} — not the same TR grid.")

    # ---- responses, EV, mask, folds: computed once, shared by every config
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
    log.info(f"  Y_train {Y_train_fit.shape}")

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

    def score_band(label: str, text: Dict[str, np.ndarray]) -> dict:
        """Fit one semantic band. Identical folds and mask for every caller."""
        nonlocal splits

        for s in all_stories:
            if text[s].shape[0] != n_trs[s]:
                raise ValueError(
                    f"{s}: store '{label}' has {text[s].shape[0]} TRs but the "
                    f"reference store has {n_trs[s]} — not the same TR grid.")

        spaces = {"text": {s: text[s] for s in train_stories}}
        if args.with_audio:
            spaces["audio"] = {s: audio_all[s] for s in train_stories}

        design = build_design(train_stories, spaces, trim=args.trim,
                              ndelays=args.ndelays)
        if splits is None:
            splits = story_folds(design.story_ids, args.n_splits)
            log.info(f"  {len(splits)} CV folds (shared by every configuration)")

        if Y_train_fit.shape[0] != design.X.shape[0]:
            raise ValueError(
                f"{subject}/{label}: design has {design.X.shape[0]} TRs but "
                f"the response has {Y_train_fit.shape[0]} — alignment failed.")

        log.info(f"  [{label}] X={design.X.shape}")
        if args.eval == "holdout":
            test_spaces = {"text": {held_out: text[held_out]}}
            if args.with_audio:
                test_spaces["audio"] = {held_out: audio_all[held_out]}
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
                solver=args.solver, solver_params=solver_params,
                inner_n_splits=args.n_splits, logger=None,
            )

        if args.with_audio and result.split_corrs is not None:
            idx = result.band_names.index("text")
            score = result.split_corrs[idx]
            score_kind = "text_split_r"
        else:
            score = result.corrs
            score_kind = "text_r"

        full = np.full(n_voxels, np.nan, dtype=np.float64)
        if voxel_mask is not None:
            full[voxel_mask] = score
        else:
            full[:] = score
        np.save(save_dir / f"{label}_{score_kind}.npy", full)

        row = {
            "subject": subject,
            "config": label,
            "n_dim": int(next(iter(text.values())).shape[1]),
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

    if args.baseline_features:
        log.info(f"  baseline band '{args.baseline_features}'")
        rows.append(score_band(args.baseline_features,
                               load_features(args.baseline_features,
                                             all_stories)))

    for source in sources:
        # One store at a time: each is only ~40 MB over the common stories, and
        # holding all of them would buy nothing since none is reused.
        text = reference if source == sources[0] else load_features(
            source, all_stories)
        rows.append(score_band(source, text))

    log.info(f"[{subject}] {len(rows)} configurations in "
             f"{(time.time() - t0) / 60:.1f} min")
    return rows


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    sources = [s for s in args.sources.split() if s.strip()]
    if not sources:
        raise SystemExit("--sources is empty")
    missing = [s for s in sources if not (Path(FEATURES_DIR) / s).is_dir()]
    if missing:
        raise SystemExit(f"No such feature store(s) under {FEATURES_DIR}: "
                         f"{missing}")

    name = "semantic" + ("__withaudio" if args.with_audio else "")
    if args.tag:
        name += f"__{args.tag}"
    out_root = Path(args.out or ENCODING_OUT) / "semantic_sweep" / args.eval / name
    ensure_dirs(out_root)

    subjects = resolve_subjects(args.subjects)
    log.info(f"Sources  : {sources}")
    log.info(f"Subjects : {', '.join(subjects)}")
    log.info(f"Eval     : {args.eval}"
             f"{' (+ audio covariate)' if args.with_audio else ' (text only)'}")
    log.info(f"Output   : {out_root}")

    t0 = time.time()
    rows = []
    for subject in subjects:
        rows.extend(run_subject(subject, args, sources, out_root))

    with open(out_root / "sweep.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)

    header = ["subject", "config", "n_dim", "score_kind", "mean_r",
              "median_r", "max_r", "n_above_0.1", "top1pct_mean_r"]
    with open(out_root / "sweep.csv", "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in header) + "\n")

    log.info(f"\n{'config':>16s} {'n_dim':>6s} {'mean_r':>9s} {'median_r':>9s} "
             f"{'max_r':>8s} {'r>0.1':>8s}")
    for r in sorted(rows, key=lambda r: -r["mean_r"]):
        log.info(f"{r['config']:>16s} {r['n_dim']:6d} {r['mean_r']:+9.4f} "
                 f"{r['median_r']:+9.4f} {r['max_r']:8.4f} "
                 f"{r['n_above_0.1']:8,d}")

    log.info(f"\nBest by mean_r: {max(rows, key=lambda r: r['mean_r'])['config']}")
    log.info(f"Finished in {(time.time() - t0) / 60:.1f} min")
    log.info(f"Summary written to {out_root / 'sweep.csv'}")


if __name__ == "__main__":
    main()
