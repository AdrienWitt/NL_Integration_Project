"""
Is surprisal already in the audio? The stimulus-side check before any
``audio x surprisal`` interaction band is built.

The theoretical case for that interaction is precision weighting: acoustic
input should be encoded more strongly when the word was not predicted. But
speakers also *mark* unpredictable words acoustically -- they accent and
lengthen them (Aylett & Turk 2004, smooth signal redundancy) -- so surprisal
may be partly a function of the audio itself. If it is, ``audio x surprisal``
is partly ``audio x g(audio)``: a nonlinearity of the prosodic band wearing a
predictive label, and no brain weighting is needed to explain it.

Two measurements, both on the stimuli alone (no fMRI):

1. **Named prominence.** Pooled r between surprisal and each of the 88
   eGeMAPS functionals, raw and after partialling word rate out of both --
   surprisal correlates with word rate at +0.74 at k=256, and so do most
   rate-sensitive acoustic functionals, so the raw r is mostly rate.
2. **Decodability.** Cross-validated ridge from openSMILE (88d) and from the
   audio band actually carried (``perlayer_base_emotion:11``, 1024d) to
   surprisal, entropy and word rate, the predictive targets residualised on
   word rate. Folds are by story; alpha is chosen on an inner story split of
   the training fold only, so the reported r is out of sample.

Rows are ``[TR_PAD + trim : -trim]`` per story, the design slice.

    PYTHONPATH=. python3 scripts/surprisal_prominence.py --k 256
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from config import FEATURES_DIR, TR_PAD

ALPHAS = np.logspace(-1, 5, 13)


def load(store, story, layer=None):
    with h5py.File(Path(FEATURES_DIR) / store / f"{story}.hf5", "r") as f:
        d = f["data"]
        if layer is None:
            return d[:]
        layers = list(d.attrs["layers"])
        return d[:, layers.index(layer), :]


def resid(y, x):
    X = np.column_stack([np.ones_like(x), x])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def zfit(X):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    return mu, sd


def ridge_path(Xtr, ytr, Xte):
    """Predictions on Xte for every alpha, via one SVD of Xtr."""
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    Uty = U.T @ ytr
    return [Xte @ (Vt.T @ (s / (s ** 2 + a) * Uty)) for a in ALPHAS]


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def cv_decode(X, y, groups, n_outer=5, seed=0):
    rng = np.random.default_rng(seed)
    stories = rng.permutation(np.unique(groups))
    folds = np.array_split(stories, n_outer)
    pred = np.empty_like(y)
    chosen = []
    for test in folds:
        te = np.isin(groups, test)
        tr_st = np.setdiff1d(stories, test)
        inner = np.isin(groups, rng.choice(tr_st, len(tr_st) // 5,
                                           replace=False))
        fit, val = ~te & ~inner, ~te & inner
        mu, sd = zfit(X[fit])
        scores = [corr(p, y[val]) for p in
                  ridge_path((X[fit] - mu) / sd, y[fit] - y[fit].mean(),
                             (X[val] - mu) / sd)]
        a_idx = int(np.argmax(scores))
        chosen.append(float(ALPHAS[a_idx]))
        mu, sd = zfit(X[~te])
        pred[te] = ridge_path((X[~te] - mu) / sd, y[~te] - y[~te].mean(),
                              (X[te] - mu) / sd)[a_idx] + y[~te].mean()
    return corr(pred, y), chosen


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=256)
    p.add_argument("--audio", default="perlayer_base_emotion:11")
    p.add_argument("--trim", type=int, default=5)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    surp_store = f"surprisal_gpt2_k{args.k}"
    a_store, a_layer = args.audio.split(":")
    stories = sorted(q.stem for q in (Path(FEATURES_DIR) / surp_store).glob("*.hf5"))
    names = (Path(FEATURES_DIR) / "opensmile" / "feature_names.txt").read_text().split()

    S, O, A, G = [], [], [], []
    for st in stories:
        s, o, a = load(surp_store, st), load("opensmile", st), load(a_store, st, int(a_layer))
        assert len(s) == len(o) == len(a), (st, len(s), len(o), len(a))
        sl = slice(TR_PAD + args.trim, len(s) - args.trim)
        S.append(s[sl]); O.append(o[sl]); A.append(a[sl]); G += [st] * len(s[sl])
    S, O, A = (np.concatenate(x).astype(np.float64) for x in (S, O, A))
    G = np.array(G)
    surp, ent, wr = S[:, 0], S[:, 1], S[:, 2]
    print(f"{len(stories)} stories, {len(G)} TRs, k={args.k}")

    surp_r, ent_r = resid(surp, wr), resid(ent, wr)
    rows = []
    for j, n in enumerate(names):
        o = O[:, j]
        rows.append(dict(feature=n, r_surp=corr(surp, o),
                         r_surp_partial=corr(surp_r, resid(o, wr)),
                         r_wordrate=corr(wr, o)))
    print("\n1. surprisal x eGeMAPS, top 12 by |partial r| (word rate removed from both)")
    print(f"   {'feature':52s} {'raw':>7s} {'partial':>8s} {'r(wr)':>7s}")
    for r in sorted(rows, key=lambda r: -abs(r["r_surp_partial"]))[:12]:
        print(f"   {r['feature']:52s} {r['r_surp']:+7.3f} {r['r_surp_partial']:+8.3f} {r['r_wordrate']:+7.3f}")
    pr = np.array([r["r_surp_partial"] for r in rows])
    print(f"   |partial r| median {np.median(np.abs(pr)):.3f}, max {np.abs(pr).max():.3f}, "
          f"{(np.abs(pr) > 0.1).sum()}/88 above 0.1")

    print("\n2. cross-validated decoding (folds by story), out-of-sample r")
    targets = {"word_rate": wr, "surprisal": surp, "surprisal | wr": surp_r,
               "entropy | wr": ent_r}
    preds = {"opensmile 88d": O, f"{args.audio} 1024d": A}
    dec = {}
    print(f"   {'target':16s}" + "".join(f"{k:>30s}" for k in preds))
    for tn, y in targets.items():
        line = f"   {tn:16s}"
        for pn, X in preds.items():
            r, alphas = cv_decode(X, y, G)
            dec[f"{pn} -> {tn}"] = dict(r=r, r2=r * r, alphas=alphas)
            line += f"{r:>+22.3f} (R2 {r * r:.3f})"
        print(line)

    if args.out:
        Path(args.out).write_text(json.dumps(dict(k=args.k, n_TRs=int(len(G)),
                                                  correlations=rows, decoding=dec), indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
