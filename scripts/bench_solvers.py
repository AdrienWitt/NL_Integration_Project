"""
Primal versus dual ridge at this project's actual shapes.

The Gallant voxelwise tutorials make a point we have so far ignored: ridge
regression has two equivalent formulations, and which one is cheap depends on
the shape of the design, not on the science.

    primal (RidgeCV / GroupRidgeCV)         cost ~ O(n p^2 + p^3)
    dual   (KernelRidgeCV / MultipleKRCV)   cost ~ O(n^2 p + n^3)

With n samples and p features, the dual wins when p > n — which is the usual
case in fMRI encoding, and why the tutorials reach for kernel ridge by default.
Our prosodic sweep is the other case:

    opensmile band   p =   88 * 4 delays =   352   vs  n = 13,329
    one layer band   p = 1024 * 4 delays = 4,096   vs  n = 13,329

so every fit in the sweep is being done in the expensive direction. This script
measures how expensive rather than arguing from exponents, and checks that the
two paths agree on the scores before we act on the timing.

Run it where a GPU is: the ordering can change between CPU and GPU because the
dual path's cost is dominated by an eigendecomposition that GPUs like.

    sbatch scripts/bench_solvers.sbatch
"""

import time

import numpy as np


def make_data(n_samples, n_features, n_targets, seed=0):
    """Correlated design and targets with real (if arbitrary) structure.

    White noise would make every alpha equivalent and the CV search trivial,
    which is not the regime we care about timing.
    """
    rng = np.random.RandomState(seed)
    latent = rng.randn(n_samples, 20).astype(np.float32)
    mixing = rng.randn(20, n_features).astype(np.float32)
    X = latent @ mixing + 0.5 * rng.randn(n_samples, n_features).astype(np.float32)
    weights = rng.randn(n_features, n_targets).astype(np.float32) / np.sqrt(n_features)
    Y = X @ weights + 2.0 * rng.randn(n_samples, n_targets).astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)
    return X.astype(np.float32), Y.astype(np.float32)


def timed(label, fn):
    t0 = time.time()
    try:
        score = fn()
    except Exception as exc:                       # noqa: BLE001
        print(f"  {label:<24s} FAILED  {type(exc).__name__}: {exc}")
        return None, None
    dt = time.time() - t0
    print(f"  {label:<24s} {dt:8.1f} s   mean r = {score:+.4f}")
    return dt, score


def main():
    from himalaya.backend import set_backend
    backend = set_backend("torch_cuda", on_error="warn")
    print(f"backend: {backend.name}\n")

    from himalaya.kernel_ridge import (ColumnKernelizer, Kernelizer,
                                       MultipleKernelRidgeCV)
    from himalaya.ridge import GroupRidgeCV, RidgeCV
    from himalaya.scoring import correlation_score
    from sklearn.model_selection import check_cv, KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_samples, n_targets = 13329, 1776
    alphas = np.logspace(0, 12, 13)

    for n_features in (352, 4096):
        print(f"n_samples={n_samples}  n_features={n_features}  "
              f"n_targets={n_targets}")
        X, Y = make_data(n_samples, n_features, n_targets)
        n_test = n_samples // 5
        X_tr, Y_tr = X[:-n_test], Y[:-n_test]
        X_te, Y_te = X[-n_test:], Y[-n_test:]
        splits = list(KFold(n_splits=5).split(X_tr))

        def score_of(model, kernelize):
            model.fit(X_tr, Y_tr)
            pred = model.predict(X_te)
            return float(backend.to_numpy(
                correlation_score(backend.asarray(Y_te), pred)).mean())

        # Dual — what encoding/banded.py does today, single band.
        def run_dual():
            per_band = make_pipeline(
                StandardScaler(with_mean=True, with_std=False),
                Kernelizer(kernel="linear"))
            kern = ColumnKernelizer([("audio", per_band, slice(0, n_features))])
            model = MultipleKernelRidgeCV(
                kernels="precomputed", solver="random_search",
                solver_params=dict(alphas=alphas, n_iter=1, n_targets_batch=200,
                                   n_alphas_batch=5),
                cv=splits)
            return score_of(make_pipeline(kern, model), True)

        # Primal banded — same random_search semantics, feature-space weights.
        def run_primal_grouped():
            model = GroupRidgeCV(
                groups=np.zeros(n_features, dtype=int),
                solver="random_search",
                solver_params=dict(alphas=alphas, n_iter=1, n_targets_batch=200,
                                   n_alphas_batch=5),
                cv=splits)
            return score_of(model, False)

        # Primal single-alpha — the cheapest thing that could work.
        def run_primal_svd():
            model = RidgeCV(alphas=alphas, cv=splits,
                            solver_params=dict(n_targets_batch=200))
            return score_of(model, False)

        # The real sweep does not use the params above: default_solver_params
        # adds diagonalize_method="svd" and n_targets_batch_refit=200. The
        # first calibration run spent 420 s where this benchmark spent 9 s at
        # identical shapes, and those two settings are the whole difference —
        # so time them explicitly rather than guess which one costs.
        def run_dual_with(diag, refit):
            per_band = make_pipeline(
                StandardScaler(with_mean=True, with_std=False),
                Kernelizer(kernel="linear"))
            kern = ColumnKernelizer([("audio", per_band, slice(0, n_features))])
            params = dict(alphas=alphas, n_iter=1, n_targets_batch=200,
                          n_alphas_batch=5)
            if diag is not None:
                params["diagonalize_method"] = diag
            if refit is not None:
                params["n_targets_batch_refit"] = refit
            model = MultipleKernelRidgeCV(
                kernels="precomputed", solver="random_search",
                solver_params=params, cv=splits)
            return score_of(make_pipeline(kern, model), True)

        timed("dual (bench params)", run_dual)
        timed("dual +diag=svd", lambda: run_dual_with("svd", None))
        timed("dual +refit=200", lambda: run_dual_with(None, 200))
        timed("dual (sweep params)", lambda: run_dual_with("svd", 200))
        timed("GroupRidgeCV (primal)", run_primal_grouped)
        timed("RidgeCV (primal, svd)", run_primal_svd)
        print()


if __name__ == "__main__":
    main()
