"""
Ground-truth check on the two nulls, on synthetic data where the answer is known.

Two gates, both of which must pass before any of this is run on real data.

GATE 1 -- refit fidelity. Every Draper-Stoneman null value comes from
`fit_banded_fixed`, which refits with the band weights the observed
`MultipleKernelRidgeCV` chose. If that refit does not reproduce the observed
correlations when handed the SAME, UNSHUFFLED design, then the null sits on a
different scale from the statistic it is compared against and every p-value is
wrong -- silently, since nothing would crash. So it is measured, not assumed.
It also checks that the band ORDER of the deltas is honoured: swapping the rows
must change the answer, or a transposition somewhere would be undetectable.

GATE 2 -- ground truth. The claim being tested is not "the code runs". It is
that the prediction-shuffle null CANNOT test integration and the
Draper-Stoneman conjunction CAN — so the
script builds voxels of known type and counts how each null labels them:

    text-only   voxel: delta must NOT be significant
    audio-only  voxel: delta must NOT be significant
    integrated  voxel: delta SHOULD be significant
    pure noise  voxel: nothing should be significant

A run that shows the naive null flagging unimodal voxels, and the conjunction
not flagging them, is the whole argument for the rewrite in one table.

    python -m stats.validate_null --n-train 900 --per-kind 30 --n-perms 200
    python -m stats.validate_null --n-perms 500    # tighter p-values
    python -m stats.validate_null --refit-only     # gate 1 alone, seconds
"""

import argparse
import logging

import numpy as np

from encoding.banded import (default_solver_params, fit_banded,
                             set_himalaya_backend)
from .permutation import (conjunction_pvalues, draper_stoneman_null,
                          fdr_correct, permutation_null, permutation_pvalues)

log = logging.getLogger("validate_null")

KINDS = ["text_only", "audio_only", "integrated", "noise"]


def ar1(n, p, rho, rng):
    """Autocorrelated features — a white design would flatter the block null."""
    x = rng.standard_normal((n, p))
    for t in range(1, n):
        x[t] = rho * x[t - 1] + np.sqrt(1 - rho ** 2) * x[t]
    return x


def make_data(n_train=1400, n_test=400, p_text=24, p_audio=12,
              per_kind=40, snr=1.0, rho=0.8, seed=0):
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    text, audio = ar1(n, p_text, rho, rng), ar1(n, p_audio, rho, rng)

    n_vox = per_kind * len(KINDS)
    kind = np.repeat(np.arange(len(KINDS)), per_kind)
    W_t = rng.standard_normal((p_text, n_vox))
    W_a = rng.standard_normal((p_audio, n_vox))
    use_t = np.isin(kind, [0, 2]).astype(float)      # text_only, integrated
    use_a = np.isin(kind, [1, 2]).astype(float)      # audio_only, integrated

    signal = (text @ (W_t * use_t) + audio @ (W_a * use_a))
    sd = signal.std(0)
    signal = np.divide(signal, np.where(sd > 0, sd, 1.0))
    Y = signal * snr + ar1(n, n_vox, 0.4, rng)

    z = lambda a: (a - a.mean(0)) / np.where(a.std(0) > 0, a.std(0), 1.0)
    X = np.hstack([z(text), z(audio)])
    bands = {"text": slice(0, p_text), "audio": slice(p_text, p_text + p_audio)}
    return (X[:n_train], z(Y)[:n_train], X[n_train:], z(Y)[n_train:],
            bands, kind)


def check_refit_fidelity(X_tr, Y_tr, X_te, Y_te, bands, fit, sp, max_iters=(50, 200)):
    """GATE 1: does the fixed-weight refit reproduce the observed fit exactly?"""
    from encoding.banded import fit_banded_fixed

    log.info("\ngate 1 -- refit fidelity (unshuffled design must reproduce the fit)")
    log.info(f"  observed bands={fit.band_names} deltas={None if fit.deltas is None else fit.deltas.shape}"
             f" mean r={fit.corrs.mean():+.6f}")
    ok = True
    for mi in max_iters:
        ref = fit_banded_fixed(X_tr, Y_tr, X_te, Y_te, bands, fit.deltas,
                               solver_params={"max_iter": mi, "tol": 1e-8})
        diff = np.abs(ref - fit.corrs).max()
        log.info(f"  max_iter={mi:<5d} mean r={ref.mean():+.6f}  max|diff|={diff:.2e}")
        if diff > 1e-3:
            ok = False
    # Band order: swapping the delta rows MUST change the result, or the order
    # is being ignored and a transposition would be invisible.
    sw = fit_banded_fixed(X_tr, Y_tr, X_te, Y_te, bands, fit.deltas[::-1],
                          solver_params={"max_iter": 200, "tol": 1e-8})
    swapped_differs = not np.allclose(sw, fit.corrs, atol=1e-6)
    log.info(f"  delta rows swapped -> max|diff|={np.abs(sw - fit.corrs).max():.2e} "
             f"({'band order honoured' if swapped_differs else 'ORDER IGNORED -- BUG'})")
    return ok and swapped_differs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-perms", type=int, default=200)
    ap.add_argument("--n-train", type=int, default=1400)
    ap.add_argument("--per-kind", type=int, default=40,
                    help="voxels of each of the four kinds")
    ap.add_argument("--ds-max-iter", type=int, default=100)
    ap.add_argument("--blocklen", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--snr", type=float, default=1.0)
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refit-only", action="store_true",
                    help="run gate 1 only and stop -- seconds, no permutations")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    X_tr, Y_tr, X_te, Y_te, bands, kind = make_data(
        n_train=args.n_train, per_kind=args.per_kind, snr=args.snr,
        seed=args.seed)
    log.info(f"X_train {X_tr.shape}  Y_train {Y_tr.shape}  bands {bands}")

    set_himalaya_backend(args.backend)
    alphas = np.logspace(0, 8, 9)
    n_stories = 10
    story_ids = np.repeat(np.arange(n_stories), len(X_tr) // n_stories)
    story_ids = np.pad(story_ids, (0, len(X_tr) - len(story_ids)), mode="edge")
    splits = [(np.where(story_ids != f)[0], np.where(story_ids == f)[0])
              for f in range(5)]
    sp = default_solver_params(n_iter=10, n_targets_batch=200,
                               n_alphas_batch=3)

    fits = {}
    for name, keep in [("text", ["text"]), ("audio", ["audio"]),
                       ("joint", ["text", "audio"])]:
        fits[name] = fit_banded(
            X_train=X_tr, Y_train=Y_tr, X_test=X_te, Y_test=Y_te,
            bands={b: bands[b] for b in keep}, splits=splits, alphas=alphas,
            solver="random_search", solver_params=sp, compute_splits=False,
            return_predictions=True,
        )
    r = {k: v.corrs for k, v in fits.items()}
    delta = r["joint"] - np.maximum(r["text"], r["audio"])

    gate1 = check_refit_fidelity(X_tr, Y_tr, X_te, Y_te, bands, fits["joint"], sp)
    if not gate1:
        log.info("\nVERDICT: gate 1 FAILED -- the fixed refit does not reproduce the "
                 "observed fit, so no Draper-Stoneman p-value can be trusted.")
        return 1
    if args.refit_only:
        log.info("\nVERDICT: gate 1 passed (--refit-only, gate 2 not run)")
        return 0

    log.info("\nobserved r by voxel kind (mean):")
    for i, k in enumerate(KINDS):
        m = kind == i
        log.info(f"  {k:11s} text={r['text'][m].mean():+.3f} "
                 f"audio={r['audio'][m].mean():+.3f} "
                 f"joint={r['joint'][m].mean():+.3f} "
                 f"delta={delta[m].mean():+.3f}")

    # ---- naive null: shuffle everything ---------------------------------
    null = permutation_null(
        Y_te, {k: v.predictions for k, v in fits.items()},
        n_perms=args.n_perms, blocklen=args.blocklen, seed=args.seed,
        progress_every=0,
    )
    p_naive = permutation_pvalues(delta, null["delta"])
    rej_naive, _ = fdr_correct(p_naive, alpha=args.alpha)
    rej_naive = rej_naive & (delta > 0)

    # ---- Draper-Stoneman conjunction ------------------------------------
    ds_p = {}
    for band, kept in [("audio", "text"), ("text", "audio")]:
        ds = draper_stoneman_null(
            X_train=X_tr, Y_train=Y_tr, X_test=X_te, Y_test=Y_te,
            bands=bands, shuffled_band=band, deltas=fits["joint"].deltas,
            n_perms=args.n_perms, blocklen=args.blocklen, seed=args.seed,
            solver_params={"max_iter": args.ds_max_iter}, progress_every=0,
        )
        ds_p[band] = permutation_pvalues(ds["observed"] - r[kept],
                                         ds["null"] - r[kept][np.newaxis, :])
        h0 = ds_p[band][kind == (0 if band == "audio" else 1)]
        log.info(f"  DS[{band}] calibration on voxels where H0 is true: "
                 f"p<0.05 in {(h0 < 0.05).mean():.1%} (want ~5%), "
                 f"median p={np.median(h0):.3f} (want ~0.5)")
    p_conj = conjunction_pvalues(ds_p["audio"], ds_p["text"])
    rej_conj, _ = fdr_correct(p_conj, alpha=args.alpha)
    rej_conj = rej_conj & (delta > 0)

    log.info(f"\ndelta significant at FDR q<{args.alpha}, by voxel kind:")
    log.info(f"  {'kind':11s} {'naive (shuffle all)':>21s} "
             f"{'DS conjunction':>16s}")
    ok = True
    for i, k in enumerate(KINDS):
        m = kind == i
        a, b, n = rej_naive[m].sum(), rej_conj[m].sum(), m.sum()
        log.info(f"  {k:11s} {a:>13d}/{n:<7d} {b:>10d}/{n:<5d}")
        if k in ("text_only", "audio_only", "noise") and b > 0.1 * n:
            ok = False
        if k == "integrated" and b < 0.5 * n:
            ok = False

    log.info("\nexpected: unimodal and noise voxels NOT significant for delta, "
             "integrated ones significant")
    log.info("VERDICT: " + ("both gates passed -- conjunction behaves correctly"
                            if ok else "gate 2 FAILED -- conjunction mislabels voxels"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
