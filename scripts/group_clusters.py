#!/usr/bin/env python3
"""
Cluster and TFCE inference on the group maps, against the group permutation
null, with a Glasser HCP-MMP1 table of the integration clusters.

    per-subject nulls --project--> group mean + 1000 group-null maps
                      --z-------->  z against each vertex's own null
                      --cluster--> extent FWE          (the table rows)
                      --TFCE----->  TFCE FWE             (the map)

Why this null. `group_maps.py` offers three group tests; the sign-flip family is
the one that TFCE was first built on, and it assumes each subject's centred
statistic is symmetric under H0 -- which the real nulls do not guarantee
(`check_tfce_null.py`). The group *permutation* null makes no such assumption:
the nine subjects share `--seed`, so draw *i* is the identical block shuffle of
time for all of them and the per-draw group mean is a sample of the observed
group mean under H0. Taking the max cluster extent (or max TFCE) over each draw
turns that into an FWE null over the whole family, the usual way. Fixed effects,
like `p_perm`: about these nine people and this stimulus.

z, not p, for the cluster-forming threshold. With 1000 draws the smallest p is
1/1001, so "p < 0.001" would mean "beats every draw" -- a threshold with no
resolution. Instead each vertex is standardised by its own null,
z = (x - mean) / sd, with the mean and sd taken over the observed map AND the
1000 draws together. Pooling the observed map in keeps all 1001 maps
exchangeable under H0, so every one of them is thresholded by the identical
rule and the test stays exact. `--z 3.09` is the p < 0.001 equivalent; run
`--z 2.33` (p < 0.01) as the sensitivity row.

Clusters give the table and TFCE gives the map. TFCE has no discrete units, so a
table still needs clusters; clusters have an arbitrary threshold, which TFCE
removes. Both come from the same z stack and the same null, so they can be read
side by side. TFCE is E=1, H=2 (surface defaults) with a fixed step in z, so
every map -- observed or null -- is integrated on the same grid.

Integration is the conjunction, as everywhere else in this project: a vertex is
integrative when it lies in a significant `audio|text` cluster AND a significant
`text|audio` cluster, i.e. the per-vertex max of the two cluster p-values
(intersection-union, Berger 1982). Cluster-level inference does not localise
to single vertices, so read "vertex v is significant" as "v belongs to a
significant region" for both families. Integration clusters are the connected
components of that conjunction, ranked by mean `delta`.

Two cluster statistics, both from the same components: extent (area, mm2) and
mass (sum of area * (z - thr)). Mass also rewards a small cluster that is very
strong; extent only a large one.

Any other statistic -- a marginal map (`text`, `audio`), an N-band
leave-one-out (`delta_emotion_given_opensmile`), a single AVD dimension -- goes
through `--stats NAME ...`: the same z / cluster / TFCE inference per statistic,
one cluster table each, and no conjunction (there is no pair to intersect).
Default without `--stats` is the text/audio conditional run, unchanged.

Outputs, under `<group-dir>/clusters/`, method = tfce | z<thr>_{extent,mass}:
    <stat>__z.npy, __tfce_perm.npy, __p_tfce_perm_fwe.npy   (327,684 long)
    <stat>__<method>__p_fwe.npy, <stat>__z<thr>__cluster_id.npy
    delta__<method>__p_fwe.npy, delta__<method>__cluster_id.npy
    <stat>__<method>__clusters.csv       significant component clusters (for
                                         tfce: components of the significant
                                         vertices, descriptive)
    integration__<method>.csv            integration clusters, ranked by delta
    summary.json
Then `scripts/visualisation/group_report.py --root <group-dir>` for the HTML.

Projection is the expensive part (~10 min per statistic); the covered-vertex
group null is cached under `--cache-dir`, so re-running with another `--z`
costs only the cluster/TFCE passes.

Usage
-----
    PYTHONPATH=. python3 scripts/group_clusters.py \\
        --perm-dir /root/prosody_nulls/k256_conditional \\
        --group-dir results/group/k256L8__base_emotionL11__conditional \\
        --marginal-dir results/group/k256L8__base_emotionL11__marginalfixed
    PYTHONPATH=. python3 scripts/group_clusters.py \\
        --perm-dir /root/prosody_nulls/emo_vs_osm \\
        --group-dir results/group/emotionL11_vs_opensmile__each \\
        --stats delta_emotion_given_opensmile delta_opensmile_given_emotion
    # thresholds default to 2.33 and 3.09 (p < .01, p < .001)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
import group_maps as gm  # noqa: E402

log = logging.getLogger("group_clusters")

COMPONENTS = ("delta_audio_given_text", "delta_text_given_audio")
CONJ = "delta"
ATLAS = "HCPMMP1"


# --------------------------------------------------------------------------
# Projection: the observed group mean and every group-null draw.
# --------------------------------------------------------------------------

def project_group(name: str, subject_dirs: list[Path], p2f, alpha: float):
    """Same accumulation as `group_maps.run_statistic`, but the group null is
    returned instead of reduced to a p-value. Also returns each subject's
    surface significance (p <= alpha against its own projected null)."""
    obs_sum = null_sum = count = None
    per_subject_sig = []
    for d in subject_dirs:
        maps = gm.SubjectMaps(p2f.Projector(d.name), d)
        log.info(f"    {d.name}: projecting {name}")
        obs = maps.observed(name)
        null = maps.null(name)
        if obs_sum is None:
            obs_sum = np.zeros(obs.size, dtype=np.float64)
            null_sum = np.zeros(null.shape, dtype=np.float32)
            count = np.zeros(obs.size, dtype=np.int16)
        r = maps.reached
        obs_sum[r] += obs[r]
        null_sum[:, r] += null[:, r]
        count[r] += 1
        p_subj = ((null >= obs[None, :]).sum(axis=0) + 1) / (null.shape[0] + 1)
        per_subject_sig.append(np.where(r, p_subj <= alpha, False))
        del null
    n = np.maximum(count, 1)
    null_sum /= n[None, :].astype(np.float32)
    return obs_sum / n, null_sum, count, np.asarray(per_subject_sig)


def load_or_project(name, subject_dirs, p2f, alpha, cache: Path,
                    covered_floor: int):
    """Covered-vertex group null, cached (~250 MB instead of 1.3 GB)."""
    f = cache / f"{name}__group.npz"
    if f.exists():
        z = np.load(f)
        log.info(f"  {name}: group null from cache {f}")
        return z["obs"], z["null"], z["count"], z["sig"], z["covered"]
    obs, null, count, sig = project_group(name, subject_dirs, p2f, alpha)
    covered = count >= covered_floor
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(f, obs=obs[covered].astype(np.float32),
             null=null[:, covered], count=count, sig=sig[:, covered],
             covered=covered)
    return obs[covered], null[:, covered], count, sig[:, covered], covered


# --------------------------------------------------------------------------
# Cluster extent and TFCE over the observed map and every null draw.
# --------------------------------------------------------------------------

_W = {}  # worker globals, inherited by fork


def _components(above: np.ndarray, z: np.ndarray, thr: float):
    """Connected suprathreshold components: (indices, labels, extent, mass).

    extent = cortical area in mm2; mass = sum over members of area * (z - thr),
    i.e. the area under the suprathreshold part of the z surface. Mass rewards a
    small cluster that is very strong as well as a large one that is barely over
    the line; extent only the latter (Bullmore et al. 1999).
    """
    idx = np.flatnonzero(above)
    if idx.size == 0:
        return idx, np.zeros(0, int), np.zeros(0), np.zeros(0)
    _, lab = connected_components(_W["adj"][idx][:, idx], directed=False)
    w = _W["area"][idx]
    return (idx, lab, np.bincount(lab, weights=w),
            np.bincount(lab, weights=w * (z[idx] - thr)))


def _max_cluster(i: int) -> np.ndarray:
    """(n_thresholds, 2): max extent and max mass of map i at each threshold."""
    out = np.zeros((len(_W["thrs"]), 2))
    for j, thr in enumerate(_W["thrs"]):
        _, _, ext, mass = _components(_W["Z"][i] > thr, _W["Z"][i], thr)
        if ext.size:
            out[j] = ext.max(), mass.max()
    return out


def _max_tfce(i: int) -> float:
    t = gm.tfce(_W["Z"][i], _W["adj"], _W["area"], _W["dh"])
    return float(t.max()) if t.size else 0.0


def _map(fn, n: int, jobs: int) -> np.ndarray:
    if jobs <= 1:
        return np.array([fn(i) for i in range(n)])
    with mp.get_context("fork").Pool(jobs) as pool:
        return np.array(pool.map(fn, range(n), chunksize=8))


def infer(obs, null, adj, area, thrs, dh, jobs):
    """z-standardise, then extent- and mass-FWE clusters at every threshold
    and TFCE-FWE, all on the family.

    Row 0 of the stack is the observed map. Its own maximum stays in the null
    (the identity permutation), so p = mean(max_i >= s), no +1 needed.
    """
    stack = np.vstack([obs[None, :], null]).astype(np.float32)
    mu = stack.mean(axis=0)
    sd = stack.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.where(sd > 0, (stack - mu) / sd, 0.0).astype(np.float32)
    del stack
    _W.update(Z=Z, adj=adj, area=area, thrs=list(thrs), dh=dh)

    maxima = _map(_max_cluster, Z.shape[0], jobs)  # (P+1, n_thr, 2)
    out = {"z": Z[0], "clusters": {}}
    for j, thr in enumerate(thrs):
        idx, lab, ext, mass = _components(Z[0] > thr, Z[0], thr)
        cid = np.zeros(obs.size, dtype=np.int32)
        p_ext = np.ones(obs.size)
        p_mass = np.ones(obs.size)
        for c in range(ext.size):
            members = idx[lab == c]
            cid[members] = c + 1
            p_ext[members] = np.mean(maxima[:, j, 0] >= ext[c])
            p_mass[members] = np.mean(maxima[:, j, 1] >= mass[c])
        out["clusters"][thr] = {
            "cluster_id": cid, "p_extent": p_ext, "p_mass": p_mass,
            "extent": ext, "mass": mass,
            "null_extent": maxima[1:, j, 0], "null_mass": maxima[1:, j, 1]}

    max_tf = _map(_max_tfce, Z.shape[0], jobs)
    tf_obs = gm.tfce(Z[0], adj, area, dh)
    out.update(tfce=tf_obs,
               p_tfce=(max_tf[None, :] >= tf_obs[:, None]).mean(axis=1),
               null_tfce=max_tf[1:])
    return out


# --------------------------------------------------------------------------
# Atlas
# --------------------------------------------------------------------------

def load_glasser(fsdir: Path):
    """Per-vertex (area, section) for lh-then-rh; area names without the
    `L_`/`R_` prefix and `_ROI` suffix, hemisphere carried separately."""
    import nibabel.freesurfer.io as fsio
    sections = json.loads(
        (fsdir / "label" / f"{ATLAS}_sections.json").read_text())["sections"]
    areas = []
    for hemi in ("lh", "rh"):
        labels, _, names = fsio.read_annot(fsdir / "label" /
                                           f"{hemi}.{ATLAS}.annot")
        names = [n.decode() for n in names]
        short = np.array(["???" if n.startswith("???") else n.split("_")[1]
                          for n in names] + ["???"], dtype=object)
        areas.append(short[labels])  # labels == -1 -> the trailing "???"
    area = np.concatenate(areas)
    section = np.array([sections.get(a, "unlabelled") for a in area],
                       dtype=object)
    return area, section


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def _top(labels: np.ndarray, k: int = 3) -> str:
    n = labels.size
    return "; ".join(f"{a} {c / n:.0%}" for a, c in
                     Counter(labels.tolist()).most_common(k))


def describe(members, hemi_split, area_mm2, glasser, maps, extra=None):
    area, section = glasser
    hemi = "lh" if members[0] < hemi_split else "rh"
    row = {
        "hemi": hemi,
        "n_vertices": members.size,
        "area_mm2": round(float(area_mm2[members].sum()), 1),
        "glasser_areas": _top(area[members], 4),
        "glasser_sections": _top(section[members], 2),
    }
    for key, m in maps.items():
        row[key] = round(float(np.nanmean(m[members])), 4)
    if extra:
        row.update(extra)
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def integration_table(p_conj, adj, area_mm2, hemi_split, glasser, maps,
                      p_tfce_conj, k_both, n_subj, alpha, min_area):
    """Connected components of the conjunction, ranked by mean delta.
    Returns (rows, per-vertex rank id, n fragments dropped).

    Components under `min_area` mm2 are dropped from the table. They are not
    regions but slivers where two component maps barely touch -- a single
    vertex has no area to speak of and, ranked by its mean, would head the
    table on the strength of one noisy value."""
    sig = np.nan_to_num(p_conj, nan=1.0) < alpha
    idx = np.flatnonzero(sig)
    conj_id = np.zeros(sig.size, dtype=np.int32)
    rows, dropped = [], 0
    if idx.size:
        _, lab = connected_components(adj[idx][:, idx], directed=False)
        for c in range(lab.max() + 1):
            members = idx[lab == c]
            if area_mm2[members].sum() < min_area:
                dropped += 1
                continue
            peak = members[np.nanargmax(maps["mean_delta"][members])]
            rows.append((members, describe(
                members, hemi_split, area_mm2, glasser, maps, {
                    "peak_area": glasser[0][peak],
                    "peak_section": glasser[1][peak],
                    "peak_delta": round(float(maps["mean_delta"][peak]), 4),
                    "p_fwe": round(float(np.nanmax(p_conj[members])), 4),
                    "pct_tfce_sig": round(float(np.mean(
                        p_tfce_conj[members] < alpha)) * 100, 1),
                    "mean_subjects_sig_both": round(float(
                        k_both[members].mean()), 2),
                    "mean_n_subjects": round(float(
                        n_subj[members].mean()), 2)})))
    rows.sort(key=lambda t: -t[1]["mean_delta"])
    table = []
    for rank, (members, row) in enumerate(rows, 1):
        conj_id[members] = rank
        table.append(dict(rank=rank, **row))
    return table, conj_id, dropped


def tfce_table(p_tfce, z, adj, area_mm2, hemi_split, glasser, maps, alpha,
               min_area):
    """Connected components of the TFCE-significant vertices of one map.

    TFCE gives a p-value per vertex, not per cluster, so these components are
    a description of the thresholded map (what the figures show), not a
    further test: p_fwe is the smallest vertex p in the component, and the
    peak is the vertex of largest z. Components under `min_area` mm2 are left
    out, as for integration."""
    sig = np.nan_to_num(p_tfce, nan=1.0) < alpha
    idx = np.flatnonzero(sig)
    rows = []
    if idx.size:
        _, lab = connected_components(adj[idx][:, idx], directed=False)
        for c in range(lab.max() + 1):
            members = idx[lab == c]
            if area_mm2[members].sum() < min_area:
                continue
            peak = members[np.nanargmax(z[members])]
            rows.append(describe(members, hemi_split, area_mm2, glasser, maps, {
                "p_fwe": round(float(np.nanmin(p_tfce[members])), 4),
                "peak_z": round(float(z[peak]), 2),
                "peak_area": glasser[0][peak]}))
    rows.sort(key=lambda row: -row["area_mm2"])
    return [dict(cluster=i + 1, **row) for i, row in enumerate(rows)]


def tag(thr: float, kind: str) -> str:
    return f"z{thr:g}_{kind}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perm-dir", type=Path, default=None,
                    help="subject subdirectories with *_null.npy")
    ap.add_argument("--group-dir", type=Path, required=True,
                    help="the matching results/group/<run> (conditional)")
    ap.add_argument("--marginal-dir", type=Path, default=None,
                    help="results/group/<run> of the marginal fit, for the "
                         "r_text / r_audio / preference columns")
    ap.add_argument("--z", type=float, nargs="+", default=[2.33, 3.09],
                    help="cluster-forming thresholds in null-sd units "
                         "(3.09 ~ p<0.001, 2.33 ~ p<0.01)")
    ap.add_argument("--tfce-dh", type=float, default=0.1,
                    help="TFCE step in z units, fixed for every map")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-subjects", type=int, default=5)
    ap.add_argument("--min-area", type=float, default=20.0,
                    help="integration clusters smaller than this (mm2) are "
                         "left out of the table, and counted in summary.json")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="default: <perm-dir>/_group_cache")
    ap.add_argument("--stats", nargs="+", default=None,
                    help="cluster these statistics, each on its own, with no "
                         "conjunction (default: the text/audio components "
                         "and their conjunction)")
    ap.add_argument("--tfce-tables-only", action="store_true",
                    help="no inference: rewrite <stat>__tfce__clusters.csv "
                         "from the z and TFCE p maps already in clusters/")
    ap.add_argument("--jobs", type=int, default=max(1, mp.cpu_count() - 2))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out = args.group_dir / "clusters"
    out.mkdir(parents=True, exist_ok=True)
    adj, area_mm2 = gm.fsaverage_mesh()
    hemi_split = area_mm2.size // 2
    sys.path.insert(0, str(Path(__file__).resolve().parent / "visualisation"))
    from plot_group_inflated import fsaverage_dir
    glasser = load_glasser(fsaverage_dir())

    def full(covered, values, fill=np.nan, dtype=np.float32):
        arr = np.full(covered.size, fill, dtype=dtype)
        arr[covered] = values
        return arr

    stats = tuple(args.stats) if args.stats else COMPONENTS
    if args.tfce_tables_only:
        g = args.group_dir
        for name in stats:
            maps = {f"mean_{name}": np.load(g / f"{name}__mean.npy")}
            if args.marginal_dir:
                for key, f in (("r_text", "text__mean.npy"),
                               ("r_audio", "audio__mean.npy"),
                               ("preference", "preference__mean.npy")):
                    if (args.marginal_dir / f).exists():
                        maps[key] = np.load(args.marginal_dir / f)
            rows = tfce_table(np.load(out / f"{name}__p_tfce_perm_fwe.npy"),
                              np.load(out / f"{name}__z.npy"), adj, area_mm2,
                              hemi_split, glasser, maps, args.alpha,
                              args.min_area)
            write_csv(rows, out / f"{name}__tfce__clusters.csv")
            log.info(f"  {name}: {len(rows)} TFCE components")
        return 0

    if args.perm_dir is None:
        ap.error("--perm-dir is required unless --tfce-tables-only")
    cache = args.cache_dir or args.perm_dir / "_group_cache"
    subject_dirs = sorted(d for d in args.perm_dir.iterdir()
                          if d.is_dir() and (d / "voxel_mask.npy").exists())
    log.info(f"{len(subject_dirs)} subjects, z > {args.z}, "
             f"family n >= {args.min_subjects}")
    p2f = gm._load_projector_module()
    res = {}
    summary = {"z_thresholds": args.z, "alpha": args.alpha,
               "tfce": {"E": 1.0, "H": 2.0, "dh_z": args.tfce_dh},
               "family_min_subjects": args.min_subjects,
               "min_area_mm2": args.min_area, "methods": {}}
    stats = tuple(args.stats) if args.stats else COMPONENTS
    conjunction = args.stats is None
    for name in stats:
        obs, null, count, sig, covered = load_or_project(
            name, subject_dirs, p2f, args.alpha, cache, args.min_subjects)
        saved = np.load(args.group_dir / f"{name}__n_subjects.npy")
        if not np.array_equal(saved, count):
            raise SystemExit(f"{name}: coverage differs from {args.group_dir} "
                             f"-- wrong --perm-dir for this group directory?")
        log.info(f"  {name}: {int(covered.sum()):,} vertices, "
                 f"{null.shape[0]} draws")
        r = infer(obs, null, adj[covered][:, covered], area_mm2[covered],
                  args.z, args.tfce_dh, args.jobs)
        del null
        summary["n_draws"] = int(r["null_tfce"].size)
        summary["family_vertices"] = int(covered.sum())
        R = {"covered": covered,
             "z": full(covered, r["z"]),
             "p": {"tfce": full(covered, r["p_tfce"])},
             "cid": {},
             "sig": np.zeros((sig.shape[0], covered.size), bool)}
        R["sig"][:, covered] = sig
        np.save(out / f"{name}__z.npy", R["z"])
        np.save(out / f"{name}__tfce_perm.npy", full(covered, r["tfce"]))
        np.save(out / f"{name}__p_tfce_perm_fwe.npy", R["p"]["tfce"])
        np.save(out / f"{name}__null_max_tfce.npy", r["null_tfce"])
        for thr, c in r["clusters"].items():
            cid = full(covered, c["cluster_id"], 0, np.int32)
            for kind in ("extent", "mass"):
                t = tag(thr, kind)
                R["p"][t] = full(covered, c[f"p_{kind}"])
                R["cid"][t] = cid
                np.save(out / f"{name}__{t}__p_fwe.npy", R["p"][t])
                np.save(out / f"{name}__{t}__null_max.npy", c[f"null_{kind}"])
                crit = np.quantile(c[f"null_{kind}"], 1 - args.alpha)
                log.info(f"    {t}: {c[kind].size} clusters, critical {kind} "
                         f"{crit:,.1f}; {int(np.sum(R['p'][t] < args.alpha)):,}"
                         f" vertices significant")
            np.save(out / f"{name}__z{thr:g}__cluster_id.npy", cid)
        log.info(f"    tfce: {int(np.nansum(R['p']['tfce'] < args.alpha)):,} "
                 f"vertices significant")
        res[name] = R

    # ---- context maps, all on the same 327,684-vertex grid --------------
    g = args.group_dir
    if conjunction:
        maps = {"mean_delta": np.load(g / "delta__mean.npy"),
                "mean_audio_given_text": np.load(g / "delta_audio_given_text__mean.npy"),
                "mean_text_given_audio": np.load(g / "delta_text_given_audio__mean.npy")}
    else:
        maps = {f"mean_{n}": np.load(g / f"{n}__mean.npy") for n in stats}
    if args.marginal_dir:
        m = args.marginal_dir
        for key, f in (("r_text", "text__mean.npy"),
                       ("r_audio", "audio__mean.npy"),
                       ("preference", "preference__mean.npy")):
            if (m / f).exists():
                maps[key] = np.load(m / f)
    if conjunction:
        n_subj = np.load(g / "delta__n_subjects.npy")
        a, b = (res[n] for n in COMPONENTS)
        # Per vertex, subjects individually significant in BOTH components.
        k_both = (a["sig"] & b["sig"]).sum(axis=0)
        p_tfce_conj = np.fmax(a["p"]["tfce"], b["p"]["tfce"])
        np.save(out / f"{CONJ}__p_tfce_perm_fwe.npy", p_tfce_conj)

    methods = ["tfce"] + [tag(t, k) for t in args.z for k in ("extent", "mass")]
    for method in methods:
        entry = {}
        if method == "tfce":
            for name in stats:
                write_csv(tfce_table(res[name]["p"]["tfce"], res[name]["z"],
                                     adj, area_mm2, hemi_split, glasser, maps,
                                     args.alpha, args.min_area),
                          out / f"{name}__tfce__clusters.csv")
        else:
            for name in stats:
                R, rows = res[name], []
                cid, p = R["cid"][method], R["p"][method]
                for c in range(1, int(cid.max()) + 1):
                    members = np.flatnonzero(cid == c)
                    rows.append(describe(members, hemi_split, area_mm2,
                                         glasser, maps, {
                        "p_fwe": round(float(p[members[0]]), 4),
                        "peak_z": round(float(np.nanmax(R["z"][members])), 2),
                        "peak_area": glasser[0][members[np.nanargmax(
                            R["z"][members])]],
                        "pct_tfce_sig": round(float(np.mean(
                            R["p"]["tfce"][members] < args.alpha)) * 100, 1)}))
                rows = [r for r in rows if r["p_fwe"] < args.alpha]
                rows.sort(key=lambda row: -row["area_mm2"])
                write_csv([dict(cluster=i + 1, **row)
                           for i, row in enumerate(rows)],
                          out / f"{name}__{method}__clusters.csv")
        for name in stats:
            entry[name] = int(np.nansum(res[name]["p"][method] < args.alpha))
        if not conjunction:
            summary["methods"][method] = entry
            log.info(f"{method}: " + ", ".join(
                f"{n} {v:,}" for n, v in entry.items()))
            continue

        p_conj = np.fmax(a["p"][method], b["p"][method])
        np.save(out / f"{CONJ}__{method}__p_fwe.npy", p_conj)
        table, conj_id, dropped = integration_table(
            p_conj, adj, area_mm2, hemi_split, glasser, maps, p_tfce_conj,
            k_both, n_subj, args.alpha, args.min_area)
        np.save(out / f"{CONJ}__{method}__cluster_id.npy", conj_id)
        write_csv(table, out / f"integration__{method}.csv")
        entry[CONJ] = int(np.nansum(p_conj < args.alpha))
        entry["n_integration_clusters"] = len(table)
        entry[f"fragments_under_{args.min_area:g}mm2"] = dropped
        summary["methods"][method] = entry
        log.info(f"\n{method}: {len(table)} integration clusters, "
                 f"{entry[CONJ]:,} vertices")
        for row in table[:8]:
            log.info(f"  {row['rank']:2d} {row['hemi']} {row['area_mm2']:8.0f} "
                     f"mm2 delta {row['mean_delta']:+.4f} p {row['p_fwe']:.3f} "
                     f"{row['glasser_areas']}")

    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log.info(json.dumps(summary["methods"], indent=1))
    log.info(f"-> {out}")


if __name__ == "__main__":
    sys.exit(main())
