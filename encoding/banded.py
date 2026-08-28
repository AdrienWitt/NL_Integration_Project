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

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


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


def _build_pipeline(bands: Dict[str, slice], splits, alphas: np.ndarray,
                    solver: str, solver_params: dict):
    from himalaya.kernel_ridge import (ColumnKernelizer, Kernelizer,
                                       MultipleKernelRidgeCV)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Centre each band but leave the scale alone: the features were already
    # z-scored per column in preprocess.build_design, and rescaling here would
    # undo the relative weighting the delays introduce.
    per_band = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        Kernelizer(kernel="linear"),
    )
    kernelizer = ColumnKernelizer(
        [(name, per_band, columns) for name, columns in bands.items()]
    )

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
            solver=solver, solver_params=solver_params,
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
                          n_targets_batch_refit: int = 200) -> dict:
    """Solver settings for `random_search`; batch sizes bound GPU memory."""
    return dict(
        n_iter=n_iter,
        n_targets_batch=n_targets_batch,
        n_alphas_batch=n_alphas_batch,
        n_targets_batch_refit=n_targets_batch_refit,
        diagonalize_method="svd",
    )
