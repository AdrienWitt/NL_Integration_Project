"""
Single-alpha ridge (Huth lab) — the secondary encoding backend.

This is the solver the earlier NL_Project runs used: one regularisation
strength per voxel, applied to the whole design matrix at once, with alphas
chosen by leave-one-story-out and performance estimated by grouped K-fold.

It is kept for one reason: to show that a positive
``delta = r_joint - max(r_text, r_audio)`` is not an artifact of the solver.
Because a joint model here must serve both bands with a single alpha while the
unimodal models each get their own, this backend *understates* delta. Treat it
as a conservative lower bound, and `banded.py` as the primary estimate. If a
voxel shows integration under both, the finding is solver-independent.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from common.ridge_utils.ridge import ridge_cv


@dataclass
class HuthResult:
    corrs: np.ndarray                    #: (n_voxels,)
    valphas: np.ndarray                  #: (n_voxels,) selected alpha per voxel
    fold_corrs: Optional[np.ndarray]     #: (n_voxels, n_folds) when CV-scored

    def as_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {"corrs": self.corrs, "valphas": self.valphas}
        if self.fold_corrs is not None and len(np.asarray(self.fold_corrs)):
            out["fold_corrs"] = self.fold_corrs
        return out


def fit_huth(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    story_ids: np.ndarray,
    alphas: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    Y_test: Optional[np.ndarray] = None,
    final_test: bool = False,
    nboots: Optional[int] = None,
    nsplits: Optional[int] = None,
    singcutoff: float = 1e-10,
    normalpha: bool = False,
    use_corr: bool = True,
    normalize_stim: bool = False,
    normalize_resp: bool = True,
    n_jobs: int = 1,
    valphas: Optional[np.ndarray] = None,
    logger=None,
) -> HuthResult:
    """Run `ridge_cv` and package its output.

    `final_test=True` fits on all training stories and scores the held-out
    story; otherwise performance comes from grouped K-fold over the training
    stories. `valphas` skips the alpha search and reuses a previous selection.
    """
    n_stories = len(np.unique(story_ids))
    nboots = n_stories if nboots is None else min(nboots, n_stories)
    nsplits = n_stories if nsplits is None else min(nsplits, n_stories)

    if final_test and (X_test is None or Y_test is None):
        raise ValueError("final_test=True requires X_test and Y_test")

    _, corrs, valphas_used, fold_corrs, _, _ = ridge_cv(
        stim=X_train,
        resp=Y_train,
        stim_test=X_test,
        resp_test=Y_test,
        alphas=alphas,
        story_ids=story_ids,
        nboots=nboots,
        nsplits=nsplits,
        singcutoff=singcutoff,
        normalpha=normalpha,
        use_corr=use_corr,
        return_wt=False,
        normalize_stim=normalize_stim,
        normalize_resp=normalize_resp,
        n_jobs=n_jobs,
        optimize_alpha=valphas is None,
        valphas=valphas,
        final_test=final_test,
        logger=logger,
    )

    fold_corrs = np.asarray(fold_corrs) if len(np.asarray(fold_corrs)) else None
    return HuthResult(
        corrs=np.asarray(corrs, dtype=np.float64),
        valphas=np.asarray(valphas_used),
        fold_corrs=fold_corrs,
    )
