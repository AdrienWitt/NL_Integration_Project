"""
Banded ridge regression (himalaya) — the primary encoding backend.

Why banded, for this project specifically
-----------------------------------------
The headline statistic here is

    delta = r_joint - max(r_text, r_audio)

i.e. "does combining semantics and prosody predict a voxel better than the
better single modality does". That comparison is only meaningful if the joint
model is not handicapped relative to its competitors.

With one shared alpha over a concatenated ``[GPT-2 768-1536d | eGeMAPS 88d]``
design, that single alpha must compromise between two bands with very
different dimensionality and effective SNR — while `r_text` and `r_audio` each
get their own optimal alpha. The joint model then loses for a purely
methodological reason, and delta is biased downward; it can even go negative
in voxels where both modalities genuinely contribute.

Banded ridge gives every band its own regularisation, so the joint model
properly *nests* the unimodal ones (it can shrink a band's contribution toward
zero and recover the single-modality fit). Delta then reflects complementary
information rather than a regularisation artifact.

Two consequences worth remembering when reading the results:

* Under banded ridge delta is >= 0 almost by construction, up to CV noise.
  The question is never "is delta positive" but "is it significantly greater
  than the null", which is what `stats/permutation.py` tests.
* The unimodal models are fit here as *single-band* banded ridge, with the
  same solver, the same alpha grid, and the same CV folds as the joint model.
  Anything else reintroduces the asymmetry this backend exists to remove.

The per-band split scores are a bonus second readout: they say how much each
band contributes *inside* the joint model, which is a different question from
whether joining helped at all.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("encoding.banded")


@dataclass
class BandedResult:
    """Outcome of one banded-ridge fit."""

    corrs: np.ndarray                       #: (n_voxels,) joint model r
    split_corrs: Optional[np.ndarray]       #: (n_bands, n_voxels) per-band r
    band_names: List[str]
    deltas: Optional[np.ndarray] = None     #: (n_bands, n_voxels) log kernel weights
    best_alphas: Optional[np.ndarray] = None
    n_folds: Optional[int] = None
    predictions: Optional[np.ndarray] = None  #: (n_test_TRs, n_voxels), if asked

    def as_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {"corrs": self.corrs, "band_names": self.band_names}
        if self.split_corrs is not None:
            out["split_corrs"] = self.split_corrs
        if self.deltas is not None:
            out["deltas"] = self.deltas
        if self.best_alphas is not None:
            out["best_alphas"] = self.best_alphas
        return out


def set_himalaya_backend(name: str = "torch_cuda"):
    """Select the himalaya compute backend, falling back to numpy on failure."""
    from himalaya.backend import set_backend
    return set_backend(name, on_error="warn")


def _band_kernelizer(bands: Dict[str, slice]):
    """One linear kernel per band, centred but not rescaled.

    The features were already z-scored per column in `preprocess.build_design`,
    so rescaling here would undo the relative weighting the delays introduce.
    Shared by every fitting path so that a permutation refit and the observed
    fit cannot drift apart in their preprocessing.
    """
    from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    per_band = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        Kernelizer(kernel="linear"),
    )
    return ColumnKernelizer(
        [(name, per_band, columns) for name, columns in bands.items()]
    )


def _build_pipeline(bands: Dict[str, slice], splits, alphas: np.ndarray,
                    solver: str, solver_params: dict):
    from himalaya.kernel_ridge import MultipleKernelRidgeCV
    from sklearn.pipeline import make_pipeline

    kernelizer = _band_kernelizer(bands)

    params = dict(solver_params)
    params["alphas"] = alphas

    # With a single band the gamma simplex is degenerate — every random draw is
    # [1.0] — so one iteration explores exactly as much as a hundred would.
    if len(bands) == 1 and solver == "random_search":
        params["n_iter"] = 1

    model = MultipleKernelRidgeCV(
        kernels="precomputed", solver=solver, solver_params=params, cv=splits
    )
    return make_pipeline(kernelizer, model), model


def _fit_primal(X_train, Y_train, X_test, Y_test, bands, splits,
                solver_params, compute_splits):
    """Banded ridge in the primal, via `himalaya.ridge.GroupRidgeCV`.

    The dual builds an n x n kernel, and a linear kernel from p features has
    rank at most p. The AVD design is 12 columns against ~8,700 training TRs,
    so its Gram matrix carries ~8,700 zero eigenvalues, LAPACK's eigh gives up,
    and `fit_banded` falls back to svd at ~42x the cost. **Small bands are the
    worst case for the dual, not the safest** -- the opposite of what this file
    used to assume.

    In the primal there is nothing to go wrong: X^T X is 12x12 and full rank.
    Measured on an AVD-shaped problem (12 columns, 3 bands, 2,400 samples,
    numpy backend, without even triggering the svd fallback):

        dual    186.5 s    mean r 0.7428
        primal    3.5 s    mean r 0.7428     x53

    per-voxel |dual - primal| mean 0.00012, max 0.0017 -- inside the
    hyperparameter search's own noise -- and the per-band split predictions
    come back in the same shape.

    `deltas_` here weights *feature groups*, not kernels, so it is NOT the
    array `fit_banded_fixed` wants. Anything reusing band weights must know
    which form produced them; `run_encoding` records it in meta as
    `solver_form`.
    """
    from himalaya.backend import get_backend
    from himalaya.ridge import GroupRidgeCV
    from himalaya.scoring import correlation_score, correlation_score_split

    backend = get_backend()
    groups = np.zeros(X_train.shape[1], dtype=int)
    for index, band_slice in enumerate(bands.values()):
        groups[band_slice] = index

    params = dict(solver_params or {})
    params.pop("diagonalize_method", None)      # dual-only
    model = GroupRidgeCV(groups=groups, cv=splits, solver_params=params)
    model.fit(X_train, Y_train)

    corrs = backend.to_numpy(correlation_score(Y_test, model.predict(X_test)))
    split_corrs = None
    if compute_splits and len(bands) > 1:
        split_corrs = np.asarray(backend.to_numpy(correlation_score_split(
            Y_test, model.predict(X_test, split=True))), dtype=float)

    return BandedResult(
        corrs=np.asarray(corrs, dtype=float),
        split_corrs=split_corrs,
        band_names=list(bands),
        deltas=np.asarray(backend.to_numpy(model.deltas_), dtype=float),
        best_alphas=np.asarray(backend.to_numpy(model.best_alphas_), dtype=float),
    )


def fit_banded(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    bands: Dict[str, slice],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    alphas: np.ndarray,
    solver: str = "random_search",
    solver_params: Optional[dict] = None,
    compute_splits: bool = True,
    return_predictions: bool = False,
    primal: bool = False,
) -> BandedResult:
    """Fit on (X_train, Y_train) and score correlations on (X_test, Y_test).

    `bands` maps a band name to its contiguous column block of X — exactly the
    `Design.bands` produced by `preprocess.build_design`. A single-entry
    `bands` gives an ordinary kernel-ridge model with its own alpha search,
    which is how the unimodal baselines are fit.
    """
    from himalaya.backend import get_backend
    from himalaya.scoring import correlation_score, correlation_score_split

    backend = get_backend()
    solver_params = solver_params or {}

    pipeline, model = _build_pipeline(bands, splits, alphas, solver, solver_params)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    Y_train = np.asarray(Y_train, dtype=np.float32)
    Y_test = np.asarray(Y_test, dtype=np.float32)

    # eigh is ~30x faster than svd (see `default_solver_params`) but it does not
    # always converge here, and the reason is structural rather than unlucky: a
    # linear kernel built from p features has rank at most p, so with p=4096
    # columns and n=9,461 training TRs the Gram matrix carries ~5,000 zero
    # eigenvalues. "Too many repeated eigenvalues" is precisely what LAPACK
    # then reports. It bit four subjects out of nine — the ones with fewer
    # stories, where the shortfall is worst.
    #
    # So: take eigh when it works, fall back to svd when it does not, per fit.
    # Paying svd's cost only on the fits that need it keeps the speed on the
    # ~80% that don't. (The primal formulation would sidestep this entirely,
    # since X^T X is 4096x4096 and full rank — worth revisiting if the fallback
    # starts firing on most fits rather than a minority.)
    #
    # It says so when it fires. This used to be silent, and the comment above
    # asks the reader to revisit the design "if the fallback starts firing on
    # most fits" -- a condition nothing could observe. A 42x slowdown that
    # leaves no trace is indistinguishable from a hung job, and a fit that
    # takes 4 minutes on four folds and hours on the fifth is exactly what it
    # looks like from the outside.
    #
    # Small bands are the worst case, not the safest. A linear kernel from p
    # features has rank at most p, so the 12-column AVD design leaves ~8,700
    # zero eigenvalues against 8,700 training TRs -- far more degenerate than
    # the p=4096 case that motivated this fallback.
    if primal:
        if return_predictions:
            raise ValueError(
                "primal=True does not return test predictions; the only caller "
                "that wants them is the prediction-shuffle null, which is dual.")
        return _fit_primal(X_train, Y_train, X_test, Y_test, bands, splits,
                           solver_params, compute_splits)

    n_cols = X_train.shape[1]
    try:
        pipeline.fit(X_train, Y_train)
    except RuntimeError as exc:
        if "eigenvalues decomposition failed" not in str(exc):
            raise
        log.warning(
            "eigh failed at p=%d, n=%d (a linear kernel from %d features has "
            "rank <= %d, so the Gram matrix carries ~%d zero eigenvalues); "
            "retrying this fit with diagonalize_method='svd', which is ~42x "
            "slower. If this fires on most fits, use the primal solver.",
            n_cols, X_train.shape[0], n_cols, n_cols,
            max(0, X_train.shape[0] - n_cols))
        retry_params = dict(solver_params)
        retry_params["diagonalize_method"] = "svd"
        pipeline, model = _build_pipeline(bands, splits, alphas, solver,
                                          retry_params)
        pipeline.fit(X_train, Y_train)

    Y_pred = pipeline.predict(X_test)
    corrs = backend.to_numpy(correlation_score(Y_test, Y_pred))

    split_corrs = None
    if compute_splits and len(bands) > 1:
        Y_pred_split = pipeline.predict(X_test, split=True)
        split_corrs = backend.to_numpy(
            correlation_score_split(Y_test, Y_pred_split)
        )

    deltas = getattr(model, "deltas_", None)
    if deltas is not None:
        deltas = backend.to_numpy(deltas)
    best_alphas = getattr(model, "best_alphas_", None)
    if best_alphas is not None:
        best_alphas = backend.to_numpy(best_alphas)

    return BandedResult(
        corrs=np.asarray(corrs, dtype=np.float64),
        split_corrs=None if split_corrs is None else np.asarray(split_corrs,
                                                                dtype=np.float64),
        band_names=list(bands.keys()),
        deltas=deltas,
        best_alphas=best_alphas,
        predictions=(backend.to_numpy(Y_pred).astype(np.float32)
                     if return_predictions else None),
    )


def fit_banded_fixed(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    bands: Dict[str, slice],
    deltas: np.ndarray,
    solver: str = "conjugate_gradient",
    solver_params: Optional[dict] = None,
) -> np.ndarray:
    """Refit with the band weights held fixed; return test correlations.

    This is the permutation workhorse. `deltas` is the `(n_bands, n_targets)`
    log kernel-weight array a `MultipleKernelRidgeCV` already chose on the
    *observed* data, and himalaya's own guidance for reusing them is
    `WeightedKernelRidge(alpha=1, deltas=model.deltas_)` — alpha is redundant
    once the deltas carry the scale, since the effective weights are
    `exp(deltas) / alpha`.

    Holding them fixed is not an optimisation, it is what makes the null
    correct. Re-running the hyperparameter search on every shuffle would let
    the model discover that the shuffled band is now useless and shrink it
    away, and the null would then describe a *well-tuned* model on noise
    rather than the same model the observed statistic came from. Fixing them
    keeps the regularisation regime — and therefore the capacity to overfit
    the shuffled band — identical on both sides of the comparison.

    It is also the only affordable option: the alpha search is the expensive
    part of a fit, and a thousand permutations of it per subject per direction
    is not a computation anyone runs.
    """
    from himalaya.backend import get_backend
    from himalaya.kernel_ridge import WeightedKernelRidge
    from himalaya.scoring import correlation_score
    from sklearn.pipeline import make_pipeline

    backend = get_backend()
    kernelizer = _band_kernelizer(bands)
    model = WeightedKernelRidge(
        alpha=1, deltas=deltas, kernels="precomputed",
        solver=solver, solver_params=dict(solver_params or {}),
    )
    pipeline = make_pipeline(kernelizer, model)

    pipeline.fit(np.asarray(X_train, dtype=np.float32),
                 np.asarray(Y_train, dtype=np.float32))
    Y_pred = pipeline.predict(np.asarray(X_test, dtype=np.float32))
    corrs = backend.to_numpy(
        correlation_score(np.asarray(Y_test, dtype=np.float32), Y_pred)
    )
    return np.asarray(corrs, dtype=np.float64)


def fit_banded_cv(
    X: np.ndarray,
    Y: np.ndarray,
    bands: Dict[str, slice],
    story_ids: np.ndarray,
    outer_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    alphas: np.ndarray,
    solver: str = "random_search",
    solver_params: Optional[dict] = None,
    compute_splits: bool = True,
    inner_n_splits: Optional[int] = None,
    primal: bool = False,
    logger=None,
) -> BandedResult:
    """Nested CV: hyperparameters are chosen inside each outer training set.

    For every outer fold the model re-runs its own inner CV on that fold's
    training stories only, so no held-out story ever influences the alphas or
    the band weights that are used to predict it. This is slower than picking
    alphas once on all stories, and it is the reason the CV correlations here
    can be trusted as out-of-sample.

    `inner_n_splits` bounds that inner loop. It matters far more than it looks:
    the total number of ridge fits is ``len(outer_splits) * inner_n_splits``,
    and leaving the inner loop at its default of leave-one-story-out means 32
    inner fits per outer fold on a 40-story sweep — 160 fits per configuration
    where 25 would rank the layers just as well. Leave it None for a final,
    single-configuration model where the extra folds are worth the hours.
    """
    from .cv import story_folds

    fold_corrs, fold_splits = [], []

    for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        if logger:
            held = sorted(np.unique(story_ids[test_idx]).tolist())
            logger.info(f"    outer fold {fold}/{len(outer_splits)} "
                        f"(held-out story index {held})")

        inner_splits = story_folds(story_ids[train_idx],
                                   n_splits=inner_n_splits)

        result = fit_banded(
            X_train=X[train_idx], Y_train=Y[train_idx],
            X_test=X[test_idx], Y_test=Y[test_idx],
            bands=bands, splits=inner_splits, alphas=alphas,
            solver=solver, solver_params=solver_params, primal=primal,
            compute_splits=compute_splits,
        )
        fold_corrs.append(result.corrs)
        if result.split_corrs is not None:
            fold_splits.append(result.split_corrs)

    corrs = np.mean(np.stack(fold_corrs), axis=0)
    split_corrs = np.mean(np.stack(fold_splits), axis=0) if fold_splits else None

    return BandedResult(
        corrs=corrs,
        split_corrs=split_corrs,
        band_names=list(bands.keys()),
        n_folds=len(outer_splits),
    )


def default_solver_params(n_iter: int = 20, n_targets_batch: int = 200,
                          n_alphas_batch: int = 5,
                          n_targets_batch_refit: int = 200,
                          diagonalize_method: str = "eigh",
                          progress_bar: bool = False) -> dict:
    """Solver settings for `random_search`; batch sizes bound GPU memory.

    `diagonalize_method` decides how the kernel is factorised once per CV fold
    and then reused across the whole alpha grid — it is the single most
    expensive thing in a fit, and the difference between the two options is not
    subtle. Measured on an A100 80GB at the sweep's real shapes
    (n=13,329, p=352, 1,776 targets, 5 folds, `scripts/bench_solvers.py`):

        eigh   9.1 s
        svd    > 280 s

    This file used to pass ``"svd"``, which is why a configuration that should
    cost seconds was costing half an hour. The kernel here is a linear Gram
    matrix — symmetric positive semi-definite by construction — which is
    exactly what `eigh` is for, and it is himalaya's own default. `svd` is the
    fallback for kernels that are not, and we never build one.

    `progress_bar` is off because himalaya's bar redraws a line on stdout
    thousands of times per fit. Under SLURM stdout is a file on the shared
    BeeGFS filesystem, and with an array of 36 tasks writing at that rate it
    fails outright:

        OSError: [Errno 121] Remote I/O error
          ... himalaya/progress_bar.py line 120, in update
              sys.stdout.write(bar)

    Half the first full-array run died that way, not one of them from the
    regression itself. The bar also drowns the log lines that say what the
    model actually scored.
    """
    return dict(
        n_iter=n_iter,
        n_targets_batch=n_targets_batch,
        n_alphas_batch=n_alphas_batch,
        n_targets_batch_refit=n_targets_batch_refit,
        diagonalize_method=diagonalize_method,
        progress_bar=progress_bar,
    )
