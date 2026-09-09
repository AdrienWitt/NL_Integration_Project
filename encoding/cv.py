"""
Cross-validation folds and the noise ceiling.

Folds are grouped by story and built once, then handed to *every* model
(text-only, audio-only, joint) and to both ridge backends. Sharing the exact
same splits is what makes

    delta = r_joint - max(r_text, r_audio)

a comparison between models rather than between fold assignments.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.stats
from sklearn.model_selection import GroupKFold

Split = Tuple[np.ndarray, np.ndarray]


def story_folds(story_ids: np.ndarray, n_splits: Optional[int] = None
                ) -> List[Split]:
    """Grouped CV folds: every TR of a story lands on the same side.

    Parameters
    ----------
    story_ids : (n_TRs,) array
        Story index per TR, from `Design.story_ids`.
    n_splits : int, optional
        Number of folds. Defaults to the number of stories, i.e.
        leave-one-story-out. Capped at the number of stories.

    Returns
    -------
    list of (train_idx, test_idx)
        Materialised (not a generator) so the identical folds can be reused
        across models and backends.
    """
    story_ids = np.asarray(story_ids)
    n_stories = len(np.unique(story_ids))
    if n_stories < 2:
        raise ValueError(f"Need >= 2 stories to cross-validate, got {n_stories}")

    n_splits = n_stories if n_splits is None else min(n_splits, n_stories)
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros((len(story_ids), 1))
    return [(tr, te) for tr, te in splitter.split(dummy, groups=story_ids)]


def leave_one_run_out(n_samples: int, run_onsets: Sequence[int]) -> List[Split]:
    """Leave-one-run-out splits from run onset indices (himalaya convention)."""
    run_onsets = np.asarray(run_onsets)
    all_samples = np.arange(n_samples)
    runs = np.split(all_samples, run_onsets[1:])
    if any(len(run) == 0 for run in runs):
        raise ValueError("Some runs are empty — check run_onsets for duplicates")

    splits = []
    for held in range(len(runs)):
        val = runs[held]
        train = np.hstack([runs[i] for i in range(len(runs)) if i != held])
        splits.append((train, val))
    return splits


def explainable_variance(repeats: np.ndarray, bias_correction: bool = True,
                         do_zscore: bool = True) -> np.ndarray:
    """Per-voxel explainable variance from repeated presentations.

    Parameters
    ----------
    repeats : (n_repeats, n_TRs, n_voxels) array
        Responses to the same story presented several times.

    Returns
    -------
    (n_voxels,) array
        Fraction of a voxel's variance that is stimulus-locked, i.e. the
        ceiling any encoding model could reach. Voxels with EV near zero carry
        no reproducible signal and are usually masked out before modelling.
    """
    repeats = np.asarray(repeats, dtype=np.float64)
    if repeats.ndim != 3:
        raise ValueError(
            f"Expected (n_repeats, n_TRs, n_voxels), got {repeats.shape}"
        )
    if do_zscore:
        repeats = scipy.stats.zscore(repeats, axis=1)

    mean_var = repeats.var(axis=1, dtype=np.float64, ddof=1).mean(axis=0)
    var_mean = repeats.mean(axis=0).var(axis=0, dtype=np.float64, ddof=1)
    ev = var_mean / mean_var

    if bias_correction:
        n_repeats = repeats.shape[0]
        ev = ev - (1 - ev) / (n_repeats - 1)
    return ev


def noise_ceiling(ev: np.ndarray, n_repeats: Optional[int] = None) -> np.ndarray:
    """Highest correlation an ideal model could reach, given EV.

    `ev` (bias-corrected) estimates the *single-presentation* reliability rho.
    `sqrt(rho)` is therefore the ceiling for predicting ONE presentation --
    but the encoding models are scored against the **mean of `n_repeats`
    presentations**, which is a cleaner target. Spearman-Brown gives that
    target's reliability,

        rho_mean = n * rho / (1 + (n - 1) * rho)

    and the ceiling is its square root. Pass `n_repeats` to get it.

    Why this matters here rather than being a detail: UTS01-03 have 10 repeats
    of the held-out story and UTS04-09 have 5, so at the observed mean
    rho = 0.158 the attainable ceilings are 0.808 and 0.696 -- a 16% advantage
    to the first three, for a reason that is purely measurement. Returning
    `sqrt(rho)` = 0.398 for both understates the ceiling roughly twofold AND
    leaves that gap uncorrected, which is the opposite of what a normalisation
    is for. `n_repeats=None` keeps the old single-presentation meaning, and is
    only right if you are predicting a single presentation.

    Negative EV, which bias correction can produce for pure-noise voxels, is
    clipped to 0.
    """
    rho = np.clip(ev, 0.0, None)
    if n_repeats is not None:
        if n_repeats < 1:
            raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
        rho = n_repeats * rho / (1.0 + (n_repeats - 1) * rho)
    return np.sqrt(rho)


def normalize_by_ceiling(corrs: np.ndarray, ev: np.ndarray,
                         min_ev: float = 0.01,
                         n_repeats: Optional[int] = None) -> np.ndarray:
    """Express `corrs` as a fraction of the noise ceiling.

    Pass `n_repeats` -- the number of presentations averaged into the target
    the correlations were scored against. Without it the ceiling is the
    single-presentation one, which is too small and, worse, identical for
    subjects with different repeat counts; see `noise_ceiling`.

    Voxels whose ceiling is below `min_ev` are returned as NaN rather than
    divided by ~0, which would manufacture enormous normalised scores in
    exactly the voxels that carry no signal.
    """
    ceiling = noise_ceiling(ev, n_repeats=n_repeats)
    out = np.full_like(np.asarray(corrs, dtype=np.float64), np.nan)
    usable = ceiling > np.sqrt(min_ev)
    out[usable] = corrs[usable] / ceiling[usable]
    return out
