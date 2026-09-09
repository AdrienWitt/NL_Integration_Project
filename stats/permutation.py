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

Two different nulls live here, for two different questions
---------------------------------------------------------
**`permutation_null` — "does this model predict at all?"** Each model is fit
once and the shuffle acts on the observed test responses while the predictions
stay fixed, which is cheap and tests exactly H0 "this model's prediction has
no temporal correspondence with the response". All models share one shuffle
per iteration so that differences between them stay comparable. Use it for
`r_text`, `r_audio`, `r_joint`.

**`draper_stoneman_null` — "does this band add beyond the other one?"** This is
the null `delta` needs, and the one above cannot supply it. Shuffling every
band drives all three correlations to ~0, whereas the observed

    delta = r_joint - max(r_text, r_audio)

is a difference of two *large* correlations that banded ridge makes positive
almost by construction — the joint model nests both unimodal ones. Tested
against a shuffle-everything null, a voxel driven purely by semantics is
"significant for integration". The fix is to shuffle **one band** in time,
leave the other band and the response aligned, and refit the joint model with
its observed band weights so it keeps the same capacity to overfit the
shuffled columns. That is H0 "this band contributes nothing beyond the other",
with the nesting advantage already priced in.

Testing delta through those two conditional nulls is exact, not an
approximation, because

    r_joint - max(r_text, r_audio) == min(r_joint - r_text, r_joint - r_audio)

so `delta > 0` holds iff both conditional contributions are positive. The
conjunction of the two one-sided tests is therefore an intersection-union test
of H0: delta <= 0 — valid at level alpha with no correction between the two
components (Berger, 1982), and conservative. See `conjunction_pvalues`.

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

    .. deprecated::
       Nothing in the pipeline uses this any more. Dropping rows silently made
       the null and the observed statistic disagree on sample size (290 vs 291
       TRs on the held-out story). Use `block_shuffle_index`, which keeps the
       tail in place and returns a full-length index.
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


def block_shuffle_index(n_samples: int, blocklen: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Full-length row index that permutes contiguous blocks in place.

    Differs from `block_permutation_index` in that nothing is dropped: the
    trailing samples that do not fill a whole block stay where they are, so the
    result can index a design matrix whose length is fixed by the response.
    Losing rows is fine when the whole design is being reordered together; it
    is not fine when only *one band* is reordered and the rest of the design
    and the response must stay aligned.
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
    idx = np.concatenate([
        np.arange(b * blocklen, (b + 1) * blocklen) for b in order
    ])
    tail = np.arange(n_blocks * blocklen, n_samples)
    return np.concatenate([idx, tail]) if tail.size else idx


def draper_stoneman_null(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    bands: Dict[str, slice],
    shuffled_band: str,
    deltas: np.ndarray,
    n_perms: int = 1000,
    blocklen: int = 10,
    seed: int = 42,
    solver_params: Dict = None,
    progress_every: int = 100,
    logger=None,
) -> Dict[str, np.ndarray]:
    """Null of the JOINT model's r when one band carries no aligned signal.

    Returns ``{"null": (n_perms, n_voxels), "observed": (n_voxels,)}``.

    **The observed value is returned from here, not computed by the caller.**
    A permutation test compares one number against a distribution, and both
    sides must come out of the *same estimator* — the observed fit uses
    `MultipleKernelRidgeCV`, the null uses `WeightedKernelRidge`, and while
    they agree to ~1e-6 on identical input (see `stats/validate_null.py`,
    gate 1), "agree closely" is not "are the same". Nothing about the caller
    would look wrong if it mixed them, and the resulting bias would be
    invisible, so the invariant is enforced here instead of documented.

    This is the null that `delta` actually needs, and the reason the
    shuffle-everything null in `permutation_null` cannot serve. Shuffling every
    band tests "is there any signal at all": under it all three correlations
    collapse toward zero, while the observed delta is a difference of two
    *large* correlations that is positive almost by construction, because
    banded ridge lets the joint model nest both unimodal ones. A voxel driven
    purely by semantics clears that null easily and gets called integrative.

    Here instead one band's rows are block-shuffled in time while the other
    band, the response, and the test set all stay exactly as observed. The
    joint model is refit — keeping its full dimensionality and its observed
    band weights — so the null says: *how well does the joint model score when
    this band's timing is destroyed but its columns are still there to overfit
    with?* That is H0 "this band adds nothing beyond the other one", with the
    nesting advantage already priced in.

    Only the training design is shuffled. The test data is never touched, so
    the comparison is between two models scored on identical held-out data,
    and the only thing that varies is what the model was able to learn.

    Blocks, not rows: BOLD is heavily autocorrelated, and a free row shuffle
    would destroy that structure, narrow the null and make the test
    anticonservative. `blocklen` should comfortably exceed the HRF width.

    The shuffle acts on the *delayed* design rather than on the raw features.
    CLAUDE.md argues the opposite, and this is a deliberate reversal: moving a
    whole row keeps that row's own FIR delay bundle internally consistent, so
    only the block boundaries are artificial, whereas shuffling raw features
    and rebuilding delays corrupts the first `ndelays` rows of *every* block.
    With blocklen=10 and ndelays=4 that is 4 clean rows traded for 40% of rows
    carrying mixed-provenance delays.

    Blocks also ignore story boundaries in the concatenated training design, so
    a block can straddle two stories. At blocklen=10 over ~24 stories that is
    ~23 straddling blocks out of ~868, and it only ever *reduces* the shuffled
    band's coherence, which makes the null slightly wider and the test slightly
    conservative. Calibration was measured, not assumed: on synthetic data with
    known ground truth the H0-true voxels reject at 5.0% against a nominal 5%
    (`stats/validate_null.py`).
    """
    from encoding.banded import fit_banded_fixed

    if shuffled_band not in bands:
        raise ValueError(
            f"shuffled_band {shuffled_band!r} is not one of {list(bands)}"
        )
    if len(bands) < 2:
        raise ValueError(
            "A Draper-Stoneman null needs the joint model: shuffling the only "
            "band in the design just gives the null for r > 0, which "
            "permutation_null already provides more cheaply."
        )

    rng = np.random.default_rng(seed)
    columns = bands[shuffled_band]
    X_train = np.asarray(X_train, dtype=np.float32)
    n_train = X_train.shape[0]

    observed = fit_banded_fixed(X_train, Y_train, X_test, Y_test, bands,
                                deltas, solver_params=solver_params)

    null = []
    for i in range(n_perms):
        idx = block_shuffle_index(n_train, blocklen, rng)
        X_perm = X_train.copy()
        X_perm[:, columns] = X_train[idx, columns]
        null.append(fit_banded_fixed(
            X_perm, Y_train, X_test, Y_test, bands, deltas,
            solver_params=solver_params,
        ))
        if logger and progress_every and (i + 1) % progress_every == 0:
            logger.info(f"  DS[{shuffled_band}] permutation {i + 1}/{n_perms}")

    return {"null": np.stack(null), "observed": observed}


def conjunction_pvalues(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """Intersection-union p-value for `delta > 0`: the larger of the two.

    `delta = r_joint - max(r_text, r_audio)` is exactly
    `min(r_joint - r_text, r_joint - r_audio)`, i.e. the smaller of the two
    conditional contributions. So `delta > 0` holds iff BOTH are positive, and
    H0 is the *union* of the two component nulls. Rejecting a union hypothesis
    means rejecting every component, so the test is "both significant" and its
    p-value is `max(p_a, p_b)`.

    This is an intersection-union test (Berger, 1982): it is valid at level
    alpha with **no multiplicity correction between the two components**, and
    it is conservative — which is the right direction for a claim about
    integration. Correction across *voxels* is still required, and is applied
    to these max-p values.
    """
    return np.maximum(np.asarray(p_a), np.asarray(p_b))


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
        # block_shuffle_index, not block_permutation_index: the latter drops the
        # trailing partial block, so on a 291-TR story the null was scored on
        # 290 TRs while the observed r it is compared against was scored on 291.
        # A permutation test may not change the sample size between its two
        # sides; keeping the tail in place costs 1 TR of exchangeability and
        # removes the mismatch.
        index = block_shuffle_index(n_samples, blocklen, rng)
        Y_shuffled = Y_true[index]

        this_perm = {}
        for name in names:
            # Predictions stay put; only the response is reordered.
            r = _columnwise_corr(Y_shuffled, predictions[name])
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
