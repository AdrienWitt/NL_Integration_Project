"""
Calibration diagnostics for a permutation run, on the real data.

`stats.validate_null` is the other half of this and answers a different
question: on SYNTHETIC voxels of known type, can the conjunction separate an
integrated voxel from a unimodal one. It cannot say whether the null is
correctly calibrated on this dataset, because on real data nobody knows which
voxels are null.

This script does, via one construction that needs no such knowledge.

The self-calibration, which is the panel that matters
-----------------------------------------------------
Every saved null draw is, by hypothesis, a sample from H0. So take draw *i*,
treat it as if it were the observed statistic, and compute its p-value against
the other m-1 draws with the estimator the real test uses. Under
exchangeability that p is *exactly* uniform on {1/m, ..., 1}, and the fraction
of draws falling below 0.05 is the test's empirical false-positive rate. If it
comes back at 0.05 the null is calibrated; if it comes back at 0.09 the test is
anticonservative and every p-value in the paper is optimistic.

With the codebase's own estimator, `p = (b + 1) / (m + 1)` (Phipson & Smyth
2010), the leave-one-out form reduces to

    p_i = (m + 1 - rank_min(x)_i) / m

with `rank_min` the ascending rank taking the smallest value among ties, so
ties are handled exactly rather than by a convention that would bias the count.

What this does and does not establish
-------------------------------------
It tests INTERNAL calibration: that the p-value machinery is correct with
respect to its own null. It catches an off-by-one in `(b + 1) / (m + 1)`, a
tie-handling convention that biases the count, an observed statistic taken
from a different estimator than the null, a misaligned mask. Those are the
failures that produce plausible-looking p-values and no traceback.

It does NOT establish that the null is the right reference for the true H0,
and it CANNOT be used to choose `--blocklen`. Every draw comes from the same
block-shuffle procedure, so the draws are exchangeable with each other **by
construction, at any blocklen** -- if the procedure narrows the null by
destroying temporal autocorrelation, both sides of this comparison carry the
identical error and it cancels. A blocklen far too short would still return a
nominal rate here.

External validity needs a different measurement, and the cheap one is a
convergence check: refit the null at several blocklens and watch the null's
standard deviation. If it stops widening past some blocklen, the
autocorrelation is absorbed; if it is still climbing at the value in use, the
test is anticonservative. `stats.validate_null` covers the third leg, on
synthetic data where H0 is known by construction.

Panels A-F follow `../Clean_Irony/diagnostic_permutations.py`, the methods
reference for this project, so the two are readable side by side.

    python -m stats.diagnose_permutation --perm-dir /root/prosody_nulls/k256_conditional
    python -m stats.diagnose_permutation --perm-dir <dir> --stats delta --alpha 0.05
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("diagnose_permutation")

#: Default statistics, matching a `--shuffle-block conditional` run.
DEFAULT_STATS = ["delta_audio_given_text", "delta_text_given_audio", "delta"]

LABELS = {
    "delta_audio_given_text": r"$\Delta r_{audio|text}$",
    "delta_text_given_audio": r"$\Delta r_{text|audio}$",
    "delta": r"$\delta$ (conjunction)",
}


def rank_min(x: np.ndarray) -> np.ndarray:
    """Ascending rank along axis 0, smallest rank among ties (1-indexed).

    scipy's `rankdata` grew an `axis` argument only in 1.10, and this runs
    against whatever the cluster env pinned, so it is done by hand: sort, then
    walk back the positions where the value repeats.
    """
    m = x.shape[0]
    order = np.argsort(x, axis=0, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, m + 1)[:, None], axis=0)
    srt = np.take_along_axis(x, order, axis=0)
    # For each sorted position, the rank of the first element equal to it.
    first = np.arange(1, m + 1)[:, None] * np.ones((1, x.shape[1]), dtype=int)
    same = srt[1:] == srt[:-1]
    for i in range(1, m):                       # m is 1000; this is cheap
        first[i] = np.where(same[i - 1], first[i - 1], first[i])
    out = np.empty_like(ranks)
    np.put_along_axis(out, order, first, axis=0)
    return out


def self_pvalues(null: np.ndarray) -> np.ndarray:
    """Leave-one-out p-value of every null draw against the others.

    `(m, n_voxels)` in, same shape out. Uniform on {1/m, ..., 1} under H0.
    """
    m = null.shape[0]
    return (m + 1 - rank_min(null)) / m


#: `delta` has no permutation of its own, and that is not an omission. Its
#: p-value is the intersection-union rule `max(p_audio|text, p_text|audio)`
#: (Berger 1982), which is valid with no null for the min statistic and no
#: correction between the two components. So there is no `delta_null.npy` to
#: load, and the diagnostics build what each panel actually needs:
#:
#:   panels B/D/F  a DISPLAY null, `min(null_a, null_b)` paired by draw index,
#:                 because `delta = min(...)` elementwise. Descriptive only --
#:                 it is not the null any p-value came from.
#:   panel G       the real rule, `max(self_p_a, self_p_b)`. Under the global
#:                 null the max of two p-values is stochastically LARGER than
#:                 uniform, so the empirical rate must come out BELOW nominal.
#:                 That is the conservativeness of an IU test, measured.
IU_COMPONENTS = ("delta_audio_given_text", "delta_text_given_audio")


def load_subject(d: Path, stat: str):
    """`(null, observed_roi, p_roi)` for one subject, restricted to the ROI.

    The saved `*_observed.npy` and `*_pvals.npy` are full-brain, padded outside
    the EV mask (p = 1 there), while `*_null.npy` holds only the fitted voxels.
    Diagnosing the padded array would put tens of thousands of structural
    p = 1 into the histogram and make any null look conservative.
    """
    mask = np.load(d / "voxel_mask.npy")
    if (d / f"{stat}_null.npy").exists():
        null = np.load(d / f"{stat}_null.npy")
    elif stat == "delta":
        a, b = (np.load(d / f"{c}_null.npy") for c in IU_COMPONENTS)
        null = np.minimum(a, b)          # display only; see IU_COMPONENTS
    else:
        raise FileNotFoundError(d / f"{stat}_null.npy")
    obs = np.load(d / f"{stat}_observed.npy")[mask]
    pv = np.load(d / f"{stat}_pvals.npy")[mask]
    if null.shape[1] != obs.size:
        raise ValueError(f"{d.name}/{stat}: null has {null.shape[1]} voxels, "
                         f"mask selects {obs.size}")
    return null, obs, pv


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--perm-dir", required=True,
                   help="directory of per-subject permutation outputs")
    p.add_argument("--stats", nargs="+", default=DEFAULT_STATS)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-draws", type=int, default=None,
                   help="subsample the null draws used for self-calibration; "
                        "the default uses all of them")
    p.add_argument("--out", default=None,
                   help="output directory (default: <perm-dir>/diagnostics)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(args.perm_dir)
    out_dir = Path(args.out) if args.out else root / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted(q.name for q in root.iterdir()
                      if q.is_dir() and (q / "voxel_mask.npy").exists())
    if not subjects:
        raise SystemExit(f"no subject directories with voxel_mask.npy under {root}")
    log.info(f"{len(subjects)} subjects under {root.name}: {', '.join(subjects)}")

    summary = {"perm_dir": str(root), "alpha": args.alpha, "stats": {}}
    n_rows, n_cols = len(args.stats), 7
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 3.1),
                             squeeze=False)

    for r, stat in enumerate(args.stats):
        pooled_p, pooled_obs, pooled_nullmean, pooled_nullsd = [], [], [], []
        pooled_self, per_subject = [], {}

        for s in subjects:
            d = root / s
            null, obs, pv = load_subject(d, stat)
            if args.max_draws and null.shape[0] > args.max_draws:
                null = null[:args.max_draws]

            if stat == "delta":
                # The rule the test actually uses, not the display null.
                sp = np.maximum(*(self_pvalues(np.load(d / f"{c}_null.npy"))
                                  for c in IU_COMPONENTS))
            else:
                sp = self_pvalues(null)
            fpr = float((sp <= args.alpha).mean())
            per_subject[s] = {
                "n_voxels": int(obs.size), "n_draws": int(null.shape[0]),
                "empirical_fpr": fpr,
                "observed_mean": float(obs.mean()),
                "null_mean": float(null.mean()),
                "frac_p_below_alpha": float((pv <= args.alpha).mean()),
            }
            pooled_p.append(pv)
            pooled_obs.append(obs)
            pooled_nullmean.append(null.mean(0))
            pooled_nullsd.append(null.std(0))
            # One column per voxel is plenty for the uniformity check and keeps
            # the pooled array from reaching 9 x 1000 x 16k.
            pooled_self.append(sp[:, ::max(1, sp.shape[1] // 200)].ravel())
            log.info(f"  {stat:24} {s}  empirical FPR {fpr:.4f}  "
                     f"({obs.size} voxels)")

        P = np.concatenate(pooled_p)
        O = np.concatenate(pooled_obs)
        NM = np.concatenate(pooled_nullmean)
        NS = np.concatenate(pooled_nullsd)
        SP = np.concatenate(pooled_self)

        pooled_fpr = float((SP <= args.alpha).mean())
        # Max deviation of the empirical CDF from the discrete uniform it must
        # follow -- a KS statistic, computed against the discrete reference so
        # the grid of attainable p-values is not itself read as misfit.
        grid = np.linspace(0, 1, 101)
        dev = float(np.max(np.abs(np.searchsorted(np.sort(SP), grid, "right")
                                  / SP.size - grid)))
        summary["stats"][stat] = {
            "pooled_empirical_fpr": pooled_fpr,
            "ks_max_deviation": dev,
            "pooled_frac_p_below_alpha": float((P <= args.alpha).mean()),
            "per_subject": per_subject,
        }

        lab = LABELS.get(stat, stat)
        sig = P <= args.alpha

        ax = axes[r][0]
        ax.hist(P, bins=40, color="#4878a8", edgecolor="white", linewidth=.4)
        ax.axhline(P.size / 40, color="#c44", ls="--", lw=1,
                   label="uniform")
        ax.set_xlabel("p"); ax.set_ylabel(f"{lab}\n\nvoxels")
        ax.set_title("A — p histogram", fontsize=9, fontweight="bold")
        ax.legend(fontsize=6)

        ax = axes[r][1]
        ax.scatter(NM, O, s=1, alpha=.15, color="#4878a8", rasterized=True)
        lim = [min(NM.min(), O.min()), max(NM.max(), O.max())]
        ax.plot(lim, lim, color="#c44", ls="--", lw=1)
        ax.set_xlabel("null mean"); ax.set_ylabel("observed")
        ax.set_title("B — obs vs null", fontsize=9, fontweight="bold")

        ax = axes[r][2]
        exp = -np.log10(np.linspace(1 / P.size, 1, P.size))
        ax.plot(exp, -np.log10(np.sort(P)), lw=1, color="#4878a8")
        ax.plot(exp, exp, color="#c44", ls="--", lw=1)
        ax.set_xlabel(r"expected $-\log_{10}p$")
        ax.set_ylabel(r"observed $-\log_{10}p$")
        ax.set_title("C — QQ (observed)", fontsize=9, fontweight="bold")

        ax = axes[r][3]
        d0 = root / subjects[0]
        null0, obs0, pv0 = load_subject(d0, stat)
        picks = [int(np.argmin(pv0)), int(np.argsort(pv0)[pv0.size // 2]),
                 int(np.argmax(pv0))]
        for v, c in zip(picks, ["#c44", "#888", "#4878a8"]):
            ax.hist(null0[:, v], bins=30, alpha=.45, color=c,
                    label=f"p={pv0[v]:.3f}")
            ax.axvline(obs0[v], color=c, lw=1.2)
        ax.set_xlabel(lab); ax.set_ylabel("draws")
        ax.set_title(f"D — nulls ({subjects[0]})", fontsize=9, fontweight="bold")
        ax.legend(fontsize=6)

        ax = axes[r][4]
        ax.hist(O[~sig], bins=40, alpha=.6, color="#888", label="n.s.")
        ax.hist(O[sig], bins=40, alpha=.6, color="#c44", label=f"p<={args.alpha}")
        ax.set_xlabel(lab); ax.set_ylabel("voxels")
        ax.set_title("E — effect sizes", fontsize=9, fontweight="bold")
        ax.legend(fontsize=6)

        ax = axes[r][5]
        ax.scatter(O, NS, s=1, alpha=.15, color="#4878a8", rasterized=True)
        ax.set_xlabel("observed"); ax.set_ylabel("null sd")
        ax.set_title("F — null homogeneity", fontsize=9, fontweight="bold")

        ax = axes[r][6]
        ax.hist(SP, bins=40, color="#3a8f5a", edgecolor="white", linewidth=.4)
        ax.axhline(SP.size / 40, color="#c44", ls="--", lw=1)
        ax.set_xlabel("p of a null draw vs the others")
        ax.set_ylabel("draws x voxels")
        note = " (IU, conservative)" if stat == "delta" else ""
        ax.set_title(f"G — calibration{note}\n"
                     f"FPR {pooled_fpr:.4f} (nom. {args.alpha})",
                     fontsize=9, fontweight="bold")

    fig.suptitle(f"Permutation diagnostics — {root.name}  "
                 f"({len(subjects)} subjects)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = out_dir / "diagnostic_permutations.png"
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    (out_dir / "calibration.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'statistic':26} {'FPR':>8} {'nominal':>9} {'KS dev':>8} "
          f"{'p<=a':>8}")
    for stat, v in summary["stats"].items():
        print(f"{stat:26} {v['pooled_empirical_fpr']:8.4f} {args.alpha:9.2f} "
              f"{v['ks_max_deviation']:8.4f} "
              f"{v['pooled_frac_p_below_alpha']:8.4f}")
    print(f"\nfigure  {fig_path}")
    print(f"numbers {out_dir / 'calibration.json'}")


if __name__ == "__main__":
    main()
