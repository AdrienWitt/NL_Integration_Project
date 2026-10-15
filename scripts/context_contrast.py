#!/usr/bin/env python3
"""
Does integration change with context length? A paired permutation test between
two k, on fsaverage, with vertex, TFCE and cluster FWE.

    results/group/k<A>.../<stat>__centred_stack.npy   (9 x 327,684)
    results/group/k<B>.../<stat>__centred_stack.npy
        --> results/group/context_contrast/k<A>_vs_k<B>/

Why this permutation and not the existing nulls. Each run's null answers "does
audio (or text) add anything at this k". The difference of two such nulls is
not a null for "k makes no difference": under that H0 both contributions can be
large and equal. What *is* exchangeable under it is the condition label within
a subject -- if k does not matter, which of a subject's two maps is called k=A
and which k=B is arbitrary. Swapping the labels negates that subject's
difference, so the exact test is a sign-flip over all 2^9 = 512 relabelings of
the per-subject difference maps. Random effects, like `p_signflip_fwe`, and it
needs no new model fits.

Differences are taken between the *null-centred* per-subject maps. Adding a
shuffled band's columns moves r by an amount that depends on the band, and the
text band changes with k, so the raw difference would carry a change in that
nesting bias as well as in the effect. The centred maps are each run's
excess over its own null. Per-subject integration is the min of its two centred
components, the per-subject version of the intersection-union `delta`.

Two-sided, reported as two one-sided maps (`up` = larger at k=B, `down` =
smaller), each at alpha/2. The flip set is closed under negation, so one set of
positive-tail null maxima serves both directions.

Statistics per flip, all max-statistic FWE over the family:
    vertex   moderated t (sd floored at its 25th percentile, as group_maps)
    tfce     TFCE of t, E=1 H=2, step fixed from the observed |t|
    cluster  extent (mm2) and mass (sum of area * (t - thr)) at t > 2.90 and
             t > 4.50 (one-sided p < .01 and < .001 at df = 8)
Family: vertices reached by >= 7 subjects (coverage is identical across k,
checked); a missing subject is filled with 0 -- no difference -- the same
under every flip, so exact and conservative.

Usage
-----
    PYTHONPATH=. python3 scripts/context_contrast.py --pairs 4:256 16:256 256:512

The same test serves every difference between two accuracies in the paper
(2026-10-15, one procedure for all): `--pattern` takes any label in place of k
(e.g. the audio-context series, `...emoctx_r0p5_w{k}L11__conditional` with
`--pairs 0:16`), and `--diff NAME=DIR:B-A` tests two statistics of one group
directory against each other (modality preference `text-audio`, valence against
arousal `valence-arousal`), on their null-centred per-subject stacks.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
import group_maps as gm  # noqa: E402

log = logging.getLogger("context_contrast")

COMPONENTS = ("delta_audio_given_text", "delta_text_given_audio")
T_THRESHOLDS = (2.90, 4.50)

_W = {}


def _tstat(signed: np.ndarray) -> np.ndarray:
    sd = np.maximum(signed.std(axis=0, ddof=1), _W["sd_floor"])
    return signed.mean(axis=0) / (sd / np.sqrt(signed.shape[0]))


def _clusters(t: np.ndarray, thr: float):
    idx = np.flatnonzero(t > thr)
    if idx.size == 0:
        return idx, np.zeros(0, int), np.zeros(0), np.zeros(0)
    _, lab = connected_components(_W["adj"][idx][:, idx], directed=False)
    w = _W["area"][idx]
    return (idx, lab, np.bincount(lab, weights=w),
            np.bincount(lab, weights=w * (t[idx] - thr)))


def _flip_max(signs) -> np.ndarray:
    """[max t, max tfce, (max extent, max mass) per threshold], positive tail."""
    t = _tstat(_W["data"] * np.asarray(signs)[:, None])
    out = [t.max(), gm.tfce(t, _W["adj"], _W["area"], _W["dh"]).max()]
    for thr in T_THRESHOLDS:
        _, _, ext, mass = _clusters(t, thr)
        out += [ext.max() if ext.size else 0.0, mass.max() if mass.size else 0.0]
    return np.asarray(out)


def contrast(d: np.ndarray, family: np.ndarray, adj, area, jobs: int) -> dict:
    """Sign-flip FWE on per-subject differences `d` (n_subj x n_vertices)."""
    data = np.nan_to_num(d[:, family], nan=0.0).astype(np.float64)
    sd_obs = data.std(axis=0, ddof=1)
    _W.update(data=data, adj=adj[family][:, family], area=area[family],
              sd_floor=np.percentile(sd_obs[sd_obs > 0], 25.0))
    t_obs = _tstat(data)
    _W["dh"] = max(float(np.abs(t_obs).max()), 1e-6) / 50

    flips = list(product((1.0, -1.0), repeat=data.shape[0]))
    with mp.get_context("fork").Pool(jobs) as pool:
        maxima = np.asarray(pool.map(_flip_max, flips, chunksize=4))

    def full(v, fill=np.nan):
        out = np.full(family.size, fill, dtype=np.float64)
        out[family] = v
        return out

    res = {"t": full(t_obs),
           "mean_diff": full(data.mean(axis=0)),
           "n_up": full((data > 0).sum(axis=0), 0)}
    for direction, sign in (("up", 1.0), ("down", -1.0)):
        t = sign * t_obs
        res[f"p_vertex_{direction}"] = full(
            (maxima[None, :, 0] >= t[:, None]).mean(axis=1))
        tf = gm.tfce(t, _W["adj"], _W["area"], _W["dh"])
        res[f"p_tfce_{direction}"] = full(
            (maxima[None, :, 1] >= tf[:, None]).mean(axis=1))
        for j, thr in enumerate(T_THRESHOLDS):
            idx, lab, ext, mass = _clusters(t, thr)
            for kind, col, stat in (("extent", 2 + 2 * j, ext),
                                    ("mass", 3 + 2 * j, mass)):
                p = np.ones(t.size)
                for c in range(stat.size):
                    p[idx[lab == c]] = np.mean(maxima[:, col] >= stat[c])
                res[f"p_t{thr:g}_{kind}_{direction}"] = full(p)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+", default=["4:256", "16:256", "256:512"],
                    help="A:B, tested as k=B minus k=A")
    ap.add_argument("--pattern",
                    default="results/group/k{k}L8__base_emotionL11__conditional")
    ap.add_argument("--out", type=Path,
                    default=Path("results/group/context_contrast"))
    ap.add_argument("--diff", nargs="+", default=None, metavar="NAME=DIR:B-A",
                    help="test B minus A within one group directory instead of --pairs")
    ap.add_argument("--prefix", default="k", help="output dir label prefix for --pairs")
    ap.add_argument("--min-subjects", type=int, default=7)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    adj, area = gm.fsaverage_mesh()
    stacks = {}

    def load(k):
        if k not in stacks:
            root = Path(args.pattern.format(k=k))
            comps = {c: np.load(root / f"{c}__centred_stack.npy")
                     for c in COMPONENTS}
            comps["delta"] = np.fmin(*comps.values())
            count = np.load(root / f"{COMPONENTS[0]}__n_subjects.npy")
            stacks[k] = (comps, count)
        return stacks[k]

    summary = {}
    if args.diff:
        for spec in args.diff:
            name, _, rest = spec.partition("=")
            root, _, stats = rest.rpartition(":")
            sb_name, _, sa_name = stats.partition("-")
            root = Path(root)
            sb = np.load(root / f"{sb_name}__centred_stack.npy")
            sa = np.load(root / f"{sa_name}__centred_stack.npy")
            count = np.load(root / f"{sb_name}__n_subjects.npy")
            if not np.array_equal(count, np.load(root / f"{sa_name}__n_subjects.npy")):
                raise SystemExit(f"coverage differs between {sb_name} and {sa_name}")
            family = count >= args.min_subjects
            out = args.out / name
            out.mkdir(parents=True, exist_ok=True)
            log.info(f"{name}: {sb_name} - {sa_name}, {int(family.sum()):,} vertices, 512 flips")
            res = contrast(sb - sa, family, adj, area, args.jobs)
            entry = {"mean_diff": float(np.nanmean(res["mean_diff"]))}
            for key, arr in res.items():
                np.save(out / f"{key}.npy", arr.astype(np.float32))
                if key.startswith("p_"):
                    entry[key] = int(np.nansum(arr < args.alpha / 2))
            np.save(out / "family.npy", family)
            summary[name] = entry
            log.info("    " + ", ".join(f"{k} {v}" for k, v in entry.items() if k.startswith("p_")))
        summary["_meta"] = {"alpha_per_direction": args.alpha / 2,
                            "family_min_subjects": args.min_subjects, "n_flips": 512}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
        log.info(f"-> {args.out}")
        return
    for pair in args.pairs:
        a, b = pair.split(":")
        (sa, ca), (sb, cb) = load(a), load(b)
        if not np.array_equal(ca, cb):
            raise SystemExit(f"coverage differs between {a} and {b}")
        family = ca >= args.min_subjects
        out = args.out / f"{args.prefix}{a}_vs_{args.prefix}{b}"
        out.mkdir(parents=True, exist_ok=True)
        summary[pair] = {}
        for stat in ("delta",) + COMPONENTS:
            log.info(f"{b} - {a}  {stat}: {int(family.sum()):,} vertices, "
                     f"512 flips")
            res = contrast(sb[stat] - sa[stat], family, adj, area, args.jobs)
            entry = {"mean_diff": float(np.nanmean(res["mean_diff"]))}
            for key, arr in res.items():
                np.save(out / f"{stat}__{key}.npy", arr.astype(np.float32))
                if key.startswith("p_"):
                    entry[key] = int(np.nansum(arr < args.alpha / 2))
            summary[pair][stat] = entry
            log.info("    " + ", ".join(f"{k} {v}" for k, v in entry.items()
                                        if k.startswith("p_")))
        np.save(out / "family.npy", family)
    summary["_meta"] = {"alpha_per_direction": args.alpha / 2,
                        "t_thresholds": T_THRESHOLDS,
                        "family_min_subjects": args.min_subjects,
                        "n_flips": 512}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    log.info(f"-> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
