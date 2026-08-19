"""
Block-permutation significance testing for r and for delta.

Why blocks
----------
fMRI time courses are strongly autocorrelated (the HRF alone smears a response
over ~6 s). Shuffling single TRs would destroy that autocorrelation and build a
null far narrower than the real one, so every model would look significant.
Permuting contiguous blocks of TRs preserves within-block temporal structure
while destroying the correspondence between stimulus and response, which is
exactly the null of interest.

Why the same permutation index for every model
----------------------------------------------
The statistic of interest is

    delta = r_joint - max(r_text, r_audio)

a *difference* between models. Its null distribution is only valid if all three
models are evaluated under the identical shuffle on each iteration — otherwise
the max() term is taken over independently-noisy quantities and the null is
biased. `permutation_null` therefore shuffles once per iteration and scores all
models against that one shuffle.

Why predictions rather than refits
----------------------------------
Refitting three banded-ridge models a thousand times is not affordable. Each
model is fit once; the permutation then acts on the *observed* test responses
while the predictions stay fixed. This tests the same null — that a model's
prediction has no temporal correspondence with the response — at a fraction of
the cost.

p-values use the (b + 1) / (m + 1) form (Phipson & Smyth, 2010), which counts
the observed statistic as one realisation under the null and so never returns
an impossible p = 0.
"""

from typing import Dict, List

import numpy as np


def block_permutation_index(n_samples: int, blocklen: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Row index that reorders `n_samples` rows in contiguous blocks.

    Trailing samples that do not fill a whole block are dropped, so the
    returned index can be shorter than `n_samples`; score the observed data on
    the same index to keep the comparison fair.
    """
    if blocklen < 1:
        raise ValueError("blocklen must be >= 1")
    n_blocks = n_samples // blocklen
    if n_blocks < 2:
        raise ValueError(
            f"{n_samples} samples with blocklen={blocklen} gives {n_blocks} "
            f"blocks — too few to permute. Use a shorter blocklen."
        )
    order = rng.permutation(n_blocks)
    return np.concatenate([
        np.arange(b * blocklen, (b + 1) * blocklen) for b in order
    ])


def _columnwise_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson r between matching columns of A and B."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    denom = np.sqrt((A ** 2).sum(0) * (B ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (A * B).sum(0) / denom, 0.0)


def permutation_null(
    Y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    n_perms: int = 1000,
    blocklen: int = 10,
    seed: int = 42,
    include_delta: bool = True,
    progress_every: int = 100,
    logger=None,
) -> Dict[str, np.ndarray]:
    """Null distributions of r (per model) and of delta.

    Parameters
    ----------
    Y_true : (n_TRs, n_voxels)
        Observed test responses.
    predictions : dict
        ``{model_name: (n_TRs, n_voxels)}``. To get a delta null, include the
        keys ``"text"``, ``"audio"`` and ``"joint"``.
    n_perms : int
        Number of block shuffles.
    blocklen : int
        Block length in TRs. Should comfortably exceed the HRF width; 10 TRs
        (20 s at TR=2 s) is a reasonable default.

    Returns
    -------
    dict of (n_perms, n_voxels) arrays
        One entry per model, plus ``"delta"`` when the three modality models
        are present.
    """
    rng = np.random.default_rng(seed)
    names = list(predictions)
    n_samples = Y_true.shape[0]

    null: Dict[str, List[np.ndarray]] = {name: [] for name in names}
    can_delta = include_delta and {"text", "audio", "joint"} <= set(names)
    if can_delta:
        null["delta"] = []
    elif include_delta and logger:
        logger.warning(
            "delta null skipped: needs predictions for text, audio and joint"
        )

    for i in range(n_perms):
        index = block_permutation_index(n_samples, blocklen, rng)
        Y_shuffled = Y_true[index]

        this_perm = {}
        for name in names:
            # Predictions stay put; only the response is reordered.
            r = _columnwise_corr(Y_shuffled, predictions[name][: len(index)])
            this_perm[name] = r
            null[name].append(r)

        if can_delta:
            null["delta"].append(
                this_perm["joint"]
                - np.maximum(this_perm["text"], this_perm["audio"])
            )

        if logger and progress_every and (i + 1) % progress_every == 0:
            logger.info(f"  permutation {i + 1}/{n_perms}")

    return {name: np.stack(vals) for name, vals in null.items()}


def permutation_pvalues(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """One-tailed p-values, (b + 1) / (m + 1) (Phipson & Smyth, 2010).

    One-tailed because every hypothesis here is directional: a model predicts
    better than chance, or joining modalities helps. A two-tailed test would
    also flag voxels predicted *worse* than chance, which is not of interest.
    """
    observed = np.asarray(observed)
    null = np.asarray(null)
    if null.shape[1:] != observed.shape:
        raise ValueError(
            f"null {null.shape} does not match observed {observed.shape}"
        )
    n_perms = null.shape[0]
    exceed = (null >= observed[np.newaxis, ...]).sum(axis=0)
    return (exceed + 1) / (n_perms + 1)


def fdr_correct(pvals: np.ndarray, alpha: float = 0.05):
    """Benjamini-Hochberg FDR. Returns (reject, pvals_corrected)."""
    from statsmodels.stats.multitest import fdrcorrection
    return fdrcorrection(np.asarray(pvals).ravel(), alpha=alpha)


def summarize(observed: np.ndarray, null: np.ndarray, label: str,
              alpha: float = 0.05, positive_only: bool = True) -> dict:
    """p-values, FDR and a short summary for one statistic."""
    pvals = permutation_pvalues(observed, null)
    reject, pvals_fdr = fdr_correct(pvals, alpha=alpha)
    if positive_only:
        reject = reject & (observed > 0)
    return {
        "label": label,
        "pvals": pvals,
        "pvals_fdr": pvals_fdr,
        "reject": reject,
        "n_significant": int(reject.sum()),
        "n_tested": int(observed.size),
        "max": float(np.nanmax(observed)),
        "mean": float(np.nanmean(observed)),
    }
