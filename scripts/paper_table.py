#!/usr/bin/env python3
"""
The paper's score table: one row per map, in the order the paper presents them.

For each map, from the group directories that already exist (nothing is
re-tested here):
    mean       group mean over vertices covered by >= --min-subjects subjects
    perm FDR   vertices with group-permutation p < .05 after BH
    TFCE FWE   vertices with group-permutation TFCE p < .05 (group_clusters.py)
    clusters   connected components of the TFCE-significant vertices (what
               the figures show), and the largest few with their peak Glasser
               area
    extent     vertices significant by cluster-extent FWE at z > 3.09, the
               other reading of the same permutation null (reported only)

Integration rows use the conjunction: `delta` for mean/perm FDR, the
conjunction TFCE map, and the integration cluster table.

    PYTHONPATH=. python3 scripts/paper_table.py [--out results/group/paper_table]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

G = Path("results/group")
# selected bands (2026-10-16 refit): emotion 16 s, eGeMAPS 2 s, text k512
EMO = "llama8bk512L13__emoctx16L11"
OSM = "llama8bk512L13__osm2"

# (section, label, group dir, statistic, kind) -- kind 'stat' or 'integration'
ROWS = [
    ("1 audio bands alone", "openSMILE (88d)", f"{OSM}__marginalfixed", "audio", "stat"),
    ("1 audio bands alone", "emotion wav2vec2 L12 (1024d)", f"{EMO}__marginalfixed", "audio", "stat"),
    ("1 audio bands alone", "AVD (3d)", "avd16__marginalfixed", "joint", "stat"),
    ("2 affective dimensions alone", "arousal", "avd16__marginalfixed", "arousal", "stat"),
    ("2 affective dimensions alone", "dominance", "avd16__marginalfixed", "dominance", "stat"),
    ("2 affective dimensions alone", "valence", "avd16__marginalfixed", "valence", "stat"),
    ("2b arousal + valence model", "arousal alone", "av16__marginalfixed", "arousal", "stat"),
    ("2b arousal + valence model", "valence alone", "av16__marginalfixed", "valence", "stat"),
    ("2b arousal + valence model", "arousal + valence", "av16__marginalfixed", "joint", "stat"),
    ("2b arousal + valence model", "valence | arousal", "av16__conditional", "delta_valence_given_arousal", "stat"),
    ("2b arousal + valence model", "arousal | valence", "av16__conditional", "delta_arousal_given_valence", "stat"),
    ("3 unique contribution, AVD", "valence | arousal+dominance", "avd16__conditional", "delta_valence_given_arousal+dominance", "stat"),
    ("3 unique contribution, AVD", "arousal | dominance+valence", "avd16__conditional", "delta_arousal_given_dominance+valence", "stat"),
    ("3 unique contribution, AVD", "dominance | arousal+valence", "avd16__conditional", "delta_dominance_given_arousal+valence", "stat"),
    ("4 audio bands against each other", "emotion | openSMILE", "emotion16_vs_osm2__each", "delta_emotion_given_opensmile", "stat"),
    ("4 audio bands against each other", "openSMILE | emotion", "emotion16_vs_osm2__each", "delta_opensmile_given_emotion", "stat"),
    ("4 audio bands against each other", "openSMILE | AVD", "osm2_vs_avd16__each", "delta_opensmile_given_avd", "stat"),
    ("4 audio bands against each other", "AVD | openSMILE", "osm2_vs_avd16__each", "delta_avd_given_opensmile", "stat"),
    ("5 text alone", "Llama-3.1-8B L13, k=512", f"{EMO}__marginalfixed", "text", "stat"),
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
        csv = gd / "clusters" / "integration__tfce.csv"
    else:
        csv = gd / "clusters" / f"{stat}__tfce__clusters.csv"
    ext = load(gd / "clusters" / f"{stat}__z3.09_extent__p_fwe.npy")
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
        "clusters": n_cl,
        "largest_clusters": desc,
        "extent_fwe": int((cov & (ext < 0.05)).sum()) if ext is not None else np.nan,
    }


# Within-subject scores: mean r over each subject's EV mask, from the encoding
# fits at the selected configuration (2026-10-15: emotion 16 s, eGeMAPS 2 s,
# text k512; `within_subject_scores.csv`, one row per run/eval/subject/model, pulled
# from Baobab's results/encoding/<run>/banded/<eval>/<SUBJ>/<model>_corrs.npy).
LL = "llama8b_k512L13__"
EM = LL + "perlayer_emoctx_r0p5_w16L11"
EO, OA, AV = "bands_emotion-opensmile__ctx", "bands_opensmile-avd__ctx", "bands_arousal-valence__ctx"
SCORES = [  # (label, run, model)
    ("eGeMAPS", EO, "opensmile"),
    ("emotion L12", EO, "emotion"),
    ("AVD", OA, "avd"),
    ("arousal", AV, "arousal"),
    ("dominance", "bands_arousal-dominance-valence__ctx", "dominance"),
    ("valence", AV, "valence"),
    ("Llama text", LL + "opensmile_r0p5_w2", "text"),
    ("Llama + emotion (joint)", EM, "joint"),
    ("Llama + eGeMAPS (joint)", LL + "opensmile_r0p5_w2", "joint"),
    ("Llama + AVD (joint)", LL + "perlayer_emoctx_r0p5_w16_avd", "joint"),
]
CONTRASTS = [  # (label, (run, model) - (run, model))
    ("emotion - eGeMAPS", (EO, "emotion"), (EO, "opensmile")),
    ("emotion | eGeMAPS", (EO, "joint"), (EO, "opensmile")),
    ("eGeMAPS | emotion", (EO, "joint"), (EO, "emotion")),
    ("AVD - eGeMAPS", (OA, "avd"), (OA, "opensmile")),
    ("AVD | eGeMAPS", (OA, "joint"), (OA, "opensmile")),
    ("eGeMAPS | AVD", (OA, "joint"), (OA, "avd")),
    ("valence | arousal", (AV, "joint"), (AV, "arousal")),
    ("arousal | valence", (AV, "joint"), (AV, "valence")),
    ("valence - arousal", (AV, "valence"), (AV, "arousal")),
    ("text - emotion (preference)", (EM, "text"), (EM, "audio")),
    ("emotion | text  (joint - text)", (EM, "joint"), (EM, "text")),
    ("text | emotion  (joint - audio)", (EM, "joint"), (EM, "audio")),
    ("eGeMAPS | text", (LL + "opensmile_r0p5_w2", "joint"), (LL + "opensmile_r0p5_w2", "text")),
    ("text | eGeMAPS", (LL + "opensmile_r0p5_w2", "joint"), (LL + "opensmile_r0p5_w2", "audio")),
    ("AVD | text", (LL + "perlayer_emoctx_r0p5_w16_avd", "joint"), (LL + "perlayer_emoctx_r0p5_w16_avd", "text")),
]


def within_subject(path):
    d = pd.read_csv(path)
    p = d.pivot_table(index=["run", "model", "subject"], columns="eval",
                      values="mean_r")
    rows = []
    for label, run, model in SCORES:
        x = p.loc[(run, model)]
        rows.append({"band": label, "n_subjects": len(x),
                     "cv": round(x["cv"].mean(), 4),
                     "holdout": round(x["holdout"].mean(), 4)})
    scores = pd.DataFrame(rows)
    rows = []
    for label, a, b in CONTRASTS:
        diff = p.loc[a] - p.loc[b]
        rows.append({"contrast": label,
                     "cv": round(diff["cv"].mean(), 4),
                     "cv_n_pos": f"{int((diff['cv'] > 0).sum())}/{len(diff)}",
                     "holdout": round(diff["holdout"].mean(), 4),
                     "holdout_n_pos": f"{int((diff['holdout'] > 0).sum())}/{len(diff)}"})
    return scores, pd.DataFrame(rows)


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
            "clusters", "largest_clusters", "extent_fwe"]
    with open(a.out / "paper_table.md", "w") as f:
        for sec, sub in df.groupby("section", sort=False):
            f.write(f"\n### {sec}\n\n" + sub[cols].to_markdown(index=False) + "\n")
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        print(df[["section"] + cols].to_string(index=False))
    ws = a.out / "within_subject_scores.csv"
    if ws.exists():
        scores, contrasts = within_subject(ws)
        scores.to_csv(a.out / "within_subject_bands.csv", index=False)
        contrasts.to_csv(a.out / "within_subject_contrasts.csv", index=False)
        with open(a.out / "paper_table.md", "a") as f:
            f.write("\n### Within subject: mean r over the EV mask, 9 subjects\n\n"
                    + scores.to_markdown(index=False) + "\n\n"
                    + contrasts.to_markdown(index=False) + "\n")
        print(scores.to_string(index=False))
        print(contrasts.to_string(index=False))
    else:
        print(f"no {ws}: within-subject section skipped")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
