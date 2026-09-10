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

Same, but with fine-tuned wav2vec2 features and both backends. The audio band
is materialised from a per-layer store first, so the layer choice is explicit
and reproducible (see `extract.build_band`)::

    python -m extract.build_band --source perlayer_ft_robust \\
        --layers 18-23 --out-name ft_robust_18to23

    python -m encoding.run_encoding --subjects UTS01,UTS02 \\
        --text-features gpt2_mean --audio-features ft_robust_18to23 \\
        --backend both --eval holdout

To choose that layer range in the first place rather than assume it, sweep it
with `encoding.run_prosody_sweep` on the training stories.

More than two bands
-------------------
`--band NAME=STORE` (repeatable) replaces the text/audio pair with an
arbitrary named band set, which is what a decomposition *inside* one modality
needs. Each band still gets its own alpha, and `split_corrs` still reports each
band's share of the joint prediction::

    python -m encoding.run_encoding --subjects all \\
        --band arousal=emotion_arousal \\
        --band dominance=emotion_dominance \\
        --band valence=emotion_valence \\
        --models arousal dominance valence arousal+valence joint \\
        --eval holdout --min-ev 0.1

`arousal+valence` fits an explicit subset, so `r_joint - r_arousal+valence` is
the unique contribution of dominance beyond the other two — which is how the
"is the third dimension worth keeping" question gets answered per voxel rather
than argued. Note the AVD dimensions are far from orthogonal: arousal and
dominance correlate ~0.95 in this model, so read that contrast together with a
cross-subject replication count, never as a single-subject map.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import (ENCODING_SPLIT_DIR, ENCODING_OUT, HELD_OUT_STORY, SUBJECTS,
                    ensure_dirs)
from common.io import (load_features, load_response, load_response_repeats,
                       save_results, stories_for_subject, subject_has_story)
from .banded import (default_solver_params, fit_banded, fit_banded_cv,
                     set_himalaya_backend)
from .cv import explainable_variance, plan_folds
from .huth_ridge import fit_huth
from .preprocess import build_design, prepare_responses, trim_response

log = logging.getLogger("encoding")

#: Which bands each model sees in the default two-band (text/audio) setup.
#: `resolve_model_bands` builds the equivalent for a `--band` run; this stays
#: the name `stats.run_permutation` imports, since permutation is text/audio.
MODEL_BANDS: Dict[str, List[str]] = {
    "text":  ["text"],
    "audio": ["audio"],
    "joint": ["text", "audio"],
}


def resolve_band_stores(args) -> Dict[str, str]:
    """``{band name: feature store}`` for this run.

    Without `--band` this is the historical two-band setup — `text` and
    `audio` from `--text-features` / `--audio-features` — so every existing
    script, sbatch and both sweeps keep working untouched.

    With `--band NAME=STORE` (repeatable) the band set is whatever was named.
    That is what a *within*-modality decomposition needs: three affective
    dimensions are three bands of one modality, not a text band and an audio
    band, and banded ridge gives each its own alpha exactly as it does for
    text vs audio.
    """
    # getattr, not args.band: `stats.run_permutation` imports `load_bands` and
    # defines no --band flag of its own, being text/audio by construction. A
    # bare attribute access here breaks it at run time, which is exactly what
    # happened when this function was introduced.
    if not getattr(args, "band", None):
        return {"text": args.text_features, "audio": args.audio_features}

    bands: Dict[str, str] = {}
    for item in args.band:
        name, sep, store = item.partition("=")
        name, store = name.strip(), store.strip()
        if not sep or not name or not store:
            raise ValueError(f"--band {item!r}: write it as --band name=store")
        if name == "joint":
            raise ValueError(
                "--band joint: 'joint' already names every band at once, so "
                "it cannot also be one of them.")
        if name in bands:
            raise ValueError(f"--band {name!r} given twice")
        bands[name] = store
    return bands


def resolve_model_bands(bands: Dict[str, str],
                        models: Sequence[str]) -> Dict[str, List[str]]:
    """``{model name: [bands it sees]}``, validating every requested model.

    A model is a band name, ``joint`` (every band), or a ``+``-joined subset.
    The subset form is how a leave-one-band-out contribution is asked for:
    with bands arousal/dominance/valence, ``--models arousal+valence joint``
    fits both, and ``r_joint - r_arousal+valence`` is the unique contribution
    of dominance beyond the other two — the same conditional-contribution
    logic the text/audio permutations already use, one level down.
    """
    model_bands: Dict[str, List[str]] = {name: [name] for name in bands}
    model_bands["joint"] = list(bands)

    for model in models:
        if model in model_bands:
            continue
        parts = [q.strip() for q in model.split("+") if q.strip()]
        unknown = [q for q in parts if q not in bands]
        if len(parts) < 2 or unknown:
            raise ValueError(
                f"--models {model!r}: expected a band name "
                f"({', '.join(bands)}), 'joint', or a '+'-joined subset of "
                f"bands" + (f"; unknown: {', '.join(unknown)}" if unknown
                            else " (a single '+' subset needs >= 2 bands)"))
        if set(parts) == set(bands):
            # A subset that is every band IS `joint`, so r_joint - r_subset is
            # identically zero. Read as a leave-one-band-out contribution that
            # says "the omitted band adds nothing" -- a wrong conclusion drawn
            # from a band the caller simply forgot to declare.
            raise ValueError(
                f"--models {model!r} names every band, so it is the joint "
                f"model under another name and `r_joint - r_{model}` is zero "
                f"by construction. Did you mean to leave one band out, or to "
                f"add a --band that is missing?")
        model_bands[model] = parts
    return model_bands


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
    data.add_argument("--band", action="append", metavar="NAME=STORE",
                      default=None,
                      help="add a named band, repeatable; replaces the "
                           "text/audio pair. e.g. --band arousal=emotion_arousal "
                           "--band valence=emotion_valence")
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
    model.add_argument("--models", nargs="+", default=None,
                       help="band names, 'joint', or '+'-joined subsets "
                            "(arousal+valence). Default: every band alone, "
                            "then joint.")
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
    model.add_argument("--alpha-n-splits", "--inner-n-splits",
                       dest="alpha_n_splits", type=int, default=None,
                       help="folds used to CHOOSE alphas, in either eval mode. "
                            "Default None = leave-one-story-out, right for a "
                            "final single-configuration model. The sweeps bound "
                            "it to --n-splits because the fit count is "
                            "n_splits * alpha_n_splits *per configuration*. Set "
                            "it to match a sweep when comparing against one.")
    model.add_argument("--n-splits", type=int, default=None,
                       help="CV folds; default = leave-one-story-out")
    model.add_argument("--max-repeats", type=int, default=5,
                       help="use only the first N repeats of the held-out story, "
                            "for every subject. UTS01-03 have 10 and the rest "
                            "have 5, which gives the first three a cleaner target "
                            "(ceiling 0.808 vs 0.696) and a differently-precise EV "
                            "mask (1,776 voxels from 10 repeats vs 6,555 from five, "
                            "on UTS01 as its own control). Capping at 5 puts every "
                            "subject on one footing. Pass 0 to use all of them.")
    model.add_argument("--min-ev", type=float, default=0.0,
                       help="fit only voxels with explainable variance above "
                            "this (holdout eval only). 0.1 is a good default "
                            "and cuts runtime a lot.")

    solver = p.add_argument_group("solver")
    solver.add_argument("--primal", action="store_true",
                        help="fit banded ridge in the primal (GroupRidgeCV) "
                             "instead of the dual. Right whenever p is small: "
                             "a linear kernel from p features has rank <= p, "
                             "so a 12-column design leaves ~8,700 zero "
                             "eigenvalues, eigh fails and the svd fallback "
                             "costs ~42x. Measured x53 faster at 12 columns, "
                             "agreeing to |diff| < 0.002 per voxel.")
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
    """Load every band's store, keyed by band name and in declared order.

    `band_stores` is resolved here when the caller has not already stashed it,
    so any namespace with `--text-features`/`--audio-features` works without
    having to know that `run_encoding.main` normally does it.
    """
    stores = getattr(args, "band_stores", None) or resolve_band_stores(args)
    features: Dict[str, Dict[str, np.ndarray]] = {}
    width = max(len(name) for name in stores)
    for name, store in stores.items():
        log.info(f"  loading band {name:<{width}} <- '{store}'")
        features[name] = load_features(store, stories)
    return features


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
                  design_test=None, Y_test=None, plan=None) -> Dict[str, object]:
    """Fit `model_name` with `backend`; returns arrays ready to save."""
    band_subset = _subset_bands(design.bands, args.model_bands[model_name])
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
                bands=band_subset, splits=plan.alpha_search, alphas=alphas,
                primal=args.primal,
                solver=args.solver, solver_params=solver_params,
            )
        else:
            result = fit_banded_cv(
                X=design.X, Y=Y_train, bands=band_subset,
                story_ids=design.story_ids, outer_splits=plan.evaluation,
                alphas=alphas,
                solver=args.solver, solver_params=solver_params, logger=log,
                inner_n_splits=plan.alpha_n_splits, primal=args.primal,
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
    # Any band would do -- build_design already checks that every band agrees
    # on every story's length -- so take the first declared one.
    ref_band = next(iter(features))
    feature_lengths = {s: features[ref_band][s].shape[0] for s in all_stories}

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
    n_repeats = None

    if held_out is not None and (args.eval == "holdout" or args.min_ev > 0):
        # EV is a function of Y alone, so it can be computed whenever the
        # repeated story is available -- including under --eval cv, where this
        # used to be skipped. That skip made `--min-ev` a silent no-op on cv
        # while both sweeps honoured it, so run_encoding's cv scores covered
        # ~81,126 voxels and the sweeps' ~1,776, with nothing saying so.
        #
        # It does not compromise the cv fit: the repeated story is never in
        # `train_stories` (resolve_stories drops it and every repeat-suffixed
        # variant), so nothing loaded here enters the design or the folds. Only
        # the mask is taken, and a mask built from Y alone leaves the
        # permutation exactly valid -- see CLAUDE.md, "The EV mask: what it
        # does and does not break".
        repeats = load_response_repeats(
            held_out, subject, max_repeats=args.max_repeats, logger=log)
        # Recorded in meta: the target is the MEAN of these, so the noise
        # ceiling depends on how many there were -- 10 for UTS01-03, 5 for the
        # rest. Without it downstream cannot compute a correct ceiling.
        n_repeats = len(repeats)
        trimmed = np.stack([
            trim_response(rep, feature_lengths[held_out], args.trim)
            for rep in repeats
        ])
        ev = explainable_variance(trimmed)
        log.info(f"  EV>{args.min_ev} in {(ev > args.min_ev).sum():,}/"
                 f"{ev.size:,} voxels")
        if args.min_ev > 0:
            voxel_mask = ev > args.min_ev
            log.info(f"  fitting {voxel_mask.sum():,} voxels with EV > {args.min_ev}")

    if args.eval == "holdout":
        test_features = {
            band: {held_out: feats[held_out]} for band, feats in features.items()
        }
        design_test = build_design(
            [held_out], test_features, trim=args.trim, ndelays=args.ndelays,
            use_pca=args.use_pca, n_comps=args.n_comps,
            fitted_pca=design.fitted_pca,       # never refit on the test story
            fitted_scalers=design.fitted_scalers,  # ...nor re-standardise on it
        )
        Y_test = prepare_responses(trimmed.mean(axis=0))
        log.info(f"  test {design_test} | Y_test {Y_test.shape}")

    if args.min_ev > 0 and voxel_mask is None:
        log.warning(
            f"--min-ev {args.min_ev} could not be applied: {subject} has no "
            f"repeats of '{args.held_out_story}', so there is no explainable "
            f"variance to threshold. These scores are whole-brain, unlike the "
            f"sweeps' masked ones. Do not compare them.")

    # One object, two named fold sets, so neither can be read as the other.
    # See encoding.cv.FoldPlan for the two bugs that motivated it.
    plan = plan_folds(design.story_ids, args.eval,
                      n_splits=args.n_splits,
                      alpha_n_splits=args.alpha_n_splits)
    log.info(f"  folds: {plan.describe()} (shared by every model)")
    if args.eval == "holdout" and args.n_splits is not None:
        log.warning(
            "--n-splits is ignored under --eval holdout: there is no outer "
            "loop to split, the score comes from the held-out story. The "
            "alpha search is --alpha-n-splits (default leave-one-story-out).")

    n_voxels = Y_train.shape[1]
    Y_train_fit = Y_train[:, voxel_mask] if voxel_mask is not None else Y_train
    Y_test_fit = (Y_test[:, voxel_mask] if (voxel_mask is not None and
                                            Y_test is not None) else Y_test)

    backends = ["banded", "huth"] if args.backend == "both" else [args.backend]

    if args.eval == "cv" and "huth" in backends:
        # ridge_cv picks per-voxel alphas by LOO over ALL training stories and
        # then reuses them inside the CV it reports, so its cross-validated r
        # is optimistically biased; fit_banded_cv re-runs its inner search
        # inside each outer fold and is not. CLAUDE.md calls huth a
        # "conservative lower bound", which is true on holdout and backwards
        # here.
        log.warning(
            "--backend huth with --eval cv: the huth solver selects alphas "
            "over all training stories and reuses them inside the reported "
            "CV, so its cv scores are optimistically biased while the banded "
            "ones are not. A huth-vs-banded comparison on cv is not a "
            "conservative check. Use --eval holdout for that contrast.")

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
                design_test=design_test, Y_test=Y_test_fit, plan=plan,
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
        # The band set, whatever its shape. `text_features`/`audio_features`
        # stay for readers written against the two-band runs, but only when
        # this run actually had those bands -- writing them from the unused
        # defaults under --band would be a lie a downstream script would act on.
        "band_stores": dict(args.band_stores),
        "model_bands": {k: list(v) for k, v in args.model_bands.items()},
        **({} if args.band else {"text_features": args.text_features,
                                 "audio_features": args.audio_features}),
        "train_stories": train_stories,
        "held_out_story": held_out,
        "n_repeats": n_repeats,          # the number actually used
        "max_repeats": args.max_repeats,  # the cap asked for, 0 = no cap
        "n_train_TRs": int(design.X.shape[0]),
        "n_voxels": int(n_voxels),
        "bands": {k: [v.start, v.stop] for k, v in design.bands.items()},
        "trim": args.trim,
        "ndelays": args.ndelays,
        "use_pca": args.use_pca,
        "n_comps": args.n_comps,
        "eval": args.eval,
        "n_folds": None if plan.evaluation is None else len(plan.evaluation),
        # The inner loop is part of the estimator, not of the design, so two
        # runs that disagree here are not strictly comparable even with
        # identical folds. Recorded so a later reader can tell.
        # Primal `deltas` weight feature groups, dual ones weight kernels, so
        # anything reusing band weights has to know which produced them.
        "solver_form": "primal" if args.primal else "dual",
        "alpha_n_splits": plan.alpha_n_splits,   # None = leave-one-story-out
        "folds": plan.describe(),
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

    args.band_stores = resolve_band_stores(args)
    if args.models is None:
        args.models = ([*args.band_stores, "joint"] if args.band
                       else list(MODEL_BANDS))
    args.model_bands = resolve_model_bands(args.band_stores, args.models)

    # `store:9-11` is a legal band name but an awkward directory name, so the
    # colon becomes an L: perlayer_gpt2_k16:8 -> perlayer_gpt2_k16L8. The band
    # names themselves go into meta.json unchanged, which is what any later
    # reader should key on.
    def _dirsafe(band: str) -> str:
        return band.replace(":", "L")

    if args.band:
        # Band *names* here, not stores: with three affective dimensions the
        # store names are near-identical and the directory would be unreadable.
        name = "bands_" + "-".join(args.band_stores)
    else:
        name = f"{_dirsafe(args.text_features)}__{_dirsafe(args.audio_features)}"
    if args.tag:
        name = f"{name}__{args.tag}"
    out_root = Path(args.out or ENCODING_OUT) / name
    ensure_dirs(out_root)

    log.info(f"Subjects : {', '.join(subjects)}")
    log.info("Bands    : " + ", ".join(f"{n} <- {s}"
                                       for n, s in args.band_stores.items()))
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
