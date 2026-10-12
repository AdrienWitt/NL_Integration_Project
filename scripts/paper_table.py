#!/usr/bin/env python3
"""
The paper's score table: one row per map, in the order the paper presents them.

For each map, from the group directories that already exist (nothing is
re-tested here):
    mean       group mean over vertices covered by >= --min-subjects subjects
    perm FDR   vertices with group-permutation p < .05 after BH
    TFCE FWE   vertices with group-permutation TFCE p < .05 (group_clusters.py)
    clusters   significant z>3.09 extent clusters, and the largest few with
               their peak Glasser area

Integration rows use the conjunction: `delta` for mean/perm FDR, the
conjunction TFCE map, and the integration cluster table.

    PYTHONPATH=. python3 scripts/paper_table.py [--out results/group/paper_table]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

G = Path("results/group")
EMO = "llama8bk1024L13__base_emotionL11"
OSM = "llama8bk1024L13__opensmile"

# (section, label, group dir, statistic, kind) -- kind 'stat' or 'integration'
ROWS = [
    ("1 audio bands alone", "openSMILE (88d)", f"{OSM}__marginalfixed", "audio", "stat"),
    ("1 audio bands alone", "emotion wav2vec2 L11 (1024d)", f"{EMO}__marginalfixed", "audio", "stat"),
    ("1 audio bands alone", "AVD (3d)", "avd__marginalfixed", "joint", "stat"),
    ("2 affective dimensions alone", "arousal", "avd__marginalfixed", "arousal", "stat"),
    ("2 affective dimensions alone", "dominance", "avd__marginalfixed", "dominance", "stat"),
    ("2 affective dimensions alone", "valence", "avd__marginalfixed", "valence", "stat"),
    ("3 unique contribution, AVD", "valence | arousal+dominance", "avd__conditional", "delta_valence_given_arousal+dominance", "stat"),
    ("3 unique contribution, AVD", "arousal | dominance+valence", "avd__conditional", "delta_arousal_given_dominance+valence", "stat"),
    ("3 unique contribution, AVD", "dominance | arousal+valence", "avd__conditional", "delta_dominance_given_arousal+valence", "stat"),
    ("4 audio bands against each other", "emotion | openSMILE", "emotionL11_vs_opensmile__each", "delta_emotion_given_opensmile", "stat"),
    ("4 audio bands against each other", "openSMILE | emotion", "emotionL11_vs_opensmile__each", "delta_opensmile_given_emotion", "stat"),
    ("4 audio bands against each other", "openSMILE | AVD", "opensmile_vs_avd__each", "delta_opensmile_given_avd", "stat"),
    ("4 audio bands against each other", "AVD | openSMILE", "opensmile_vs_avd__each", "delta_avd_given_opensmile", "stat"),
    ("5 text alone", "Llama-3.1-8B L13, k=1024", f"{EMO}__marginalfixed", "text", "stat"),
    ("6 integration, text x emotion", "emotion | text", f"{EMO}__conditional", "delta_audio_given_text", "stat"),
    ("6 integration, text x emotion", "text | emotion", f"{EMO}__conditional", "delta_text_given_audio", "stat"),
    ("6 integration, text x emotion", "integration (conjunction)", f"{EMO}__conditional", "delta", "integration"),
    ("7 integration, text x openSMILE", "openSMILE | text", f"{OSM}__conditional", "delta_audio_given_text", "stat"),
    ("7 integration, text x openSMILE", "text | openSMILE", f"{OSM}__conditional", "delta_text_given_audio", "stat"),
    ("7 integration, text x openSMILE", "integration (conjunction)", f"{OSM}__conditional", "delta", "integration"),
]


def load(path):
    return np.load(path) if path.exists() else None


def top_clusters(csv, n):
    if not csv.exists():
        return 0, ""
    df = pd.read_csv(csv)
    if "p_fwe" in df:
        df = df[df["p_fwe"] < 0.05]
    df = df.sort_values("n_vertices", ascending=False)
    # tables written before --stats (2026-10-12) have no peak_area: fall back
    # to the cluster's largest Glasser area
    area = (df["peak_area"] if "peak_area" in df
            else df["glasser_areas"].str.split(" ").str[0])
    desc = "; ".join(f"{h} {ar} ({int(nv)})" for h, ar, nv in
                     zip(df["hemi"].head(n), area.head(n), df["n_vertices"].head(n)))
    return len(df), desc


def row(section, label, d, stat, kind, min_subj, n_top):
    gd = G / d
    mean = load(gd / f"{stat}__mean.npy")
    nsub = load(gd / f"{stat}__n_subjects.npy")
    fdr = load(gd / f"{stat}__p_perm_fdr.npy")
    if mean is None:
        raise FileNotFoundError(gd / f"{stat}__mean.npy")
    cov = (nsub >= min_subj) & np.isfinite(mean)
    tfce = load(gd / "clusters" / f"{stat}__p_tfce_perm_fwe.npy")
    if kind == "integration":
        csv = gd / "clusters" / "integration__z3.09_extent.csv"
    else:
        csv = gd / "clusters" / f"{stat}__z3.09_extent__clusters.csv"
    n_cl, desc = top_clusters(csv, n_top)
    sig = cov & (fdr < 0.05) if fdr is not None else None
    return {
        "section": section, "map": label, "group_dir": d, "statistic": stat,
        "n_covered": int(cov.sum()),
        "mean": round(float(np.nanmean(mean[cov])), 4),
        "mean_where_sig": (round(float(np.nanmean(mean[sig])), 4)
                           if sig is not None and sig.any() else np.nan),
        "perm_fdr": int(sig.sum()) if sig is not None else np.nan,
        "tfce_fwe": int((cov & (tfce < 0.05)).sum()) if tfce is not None else np.nan,
        "z3.09_clusters": n_cl,
        "largest_clusters": desc,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=G / "paper_table")
    ap.add_argument("--min-subjects", type=int, default=5)
    ap.add_argument("--n-top", type=int, default=4)
    a = ap.parse_args(argv)
    df = pd.DataFrame([row(*r, a.min_subjects, a.n_top) for r in ROWS])
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "paper_table.csv", index=False)
    cols = ["map", "mean", "mean_where_sig", "perm_fdr", "tfce_fwe",
            "z3.09_clusters", "largest_clusters"]
    with open(a.out / "paper_table.md", "w") as f:
        for sec, sub in df.groupby("section", sort=False):
            f.write(f"\n### {sec}\n\n" + sub[cols].to_markdown(index=False) + "\n")
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        print(df[["section"] + cols].to_string(index=False))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
