"""
Cross-validation folds and the noise ceiling.

Folds are grouped by story and built once, then handed to *every* model
(text-only, audio-only, joint) and to both ridge backends. Sharing the exact
same splits is what makes

    delta = r_joint - max(r_text, r_audio)

a comparison between models rather than between fold assignments.
"""

from dataclasses import dataclass
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


@dataclass(frozen=True)
class FoldPlan:
    """The two fold sets a fit needs, each with exactly one meaning.

    `story_folds` alone is ambiguous, and the ambiguity has caused two bugs.
    The same call builds *evaluation* folds under ``--eval cv`` and the
    *alpha-search* folds under ``--eval holdout``, so one flag named
    `--n-splits` silently meant two different things depending on a mode set
    elsewhere. Consequences, both real and both already paid for:

    * `run_encoding --eval holdout` bounded its alpha search to 5 folds while
      the cv half of the same job searched over 23, so one run reported two
      eval modes produced by two estimators;
    * `run_permutation` had to keep a comment asking a human to keep its alpha
      search equal to whatever run_encoding was doing, which is a coupling that
      breaks the moment run_encoding changes -- and it did.

    Here `n_splits` always means "folds a cross-validated score is reported
    over" and `alpha_n_splits` always means "folds used to choose alphas",
    whatever the eval mode. Under holdout there is no evaluation fold set --
    the score comes from the held-out story -- so `evaluation` is None and
    nothing can accidentally read it as the alpha search.
    """

    evaluation: Optional[List[Split]]
    alpha_search: Optional[List[Split]]
    alpha_n_splits: Optional[int]

    def describe(self) -> str:
        n_alpha = ("leave-one-story-out" if self.alpha_n_splits is None
                   else f"{self.alpha_n_splits}-fold")
        if self.evaluation is None:
            return f"holdout scoring; alphas by {n_alpha} CV over the training stories"
        return (f"{len(self.evaluation)} evaluation folds; alphas by {n_alpha} "
                f"CV inside each")


def plan_folds(story_ids: np.ndarray, eval_mode: str,
               n_splits: Optional[int] = None,
               alpha_n_splits: Optional[int] = None) -> FoldPlan:
    """Build both fold sets for `eval_mode`, which is 'cv' or 'holdout'.

    `alpha_n_splits` defaults to None, i.e. leave-one-story-out, which is right
    for a final single-configuration model. A sweep must pass it explicitly:
    the fit count is ``n_splits * alpha_n_splits`` *per configuration*, so
    leaving it unbounded over ~96 configurations is the difference between 25
    and 115 fits each. Passing it at the call site is the point -- the cost is
    then visible where the decision is made instead of inherited from a default.
    """
    if eval_mode not in ("cv", "holdout"):
        raise ValueError(f"eval_mode must be 'cv' or 'holdout', got {eval_mode!r}")

    if eval_mode == "holdout":
        # No outer loop: the score comes from the repeated story. `n_splits`
        # has nothing to name here, so it is not consulted -- a caller that
        # passes it is saying something about an evaluation that does not exist.
        return FoldPlan(evaluation=None,
                        alpha_search=story_folds(story_ids, alpha_n_splits),
                        alpha_n_splits=alpha_n_splits)

    # cv: the alpha search is nested inside each evaluation fold and is built
    # there from that fold's training stories, so only its size travels.
    return FoldPlan(evaluation=story_folds(story_ids, n_splits),
                    alpha_search=None,
                    alpha_n_splits=alpha_n_splits)


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
