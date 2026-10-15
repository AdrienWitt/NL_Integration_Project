#!/usr/bin/env python3
"""
Every table of the paper, main and supplementary, from outputs that already
exist. Nothing is re-tested here; run `scripts/paper_table.py` first (it
writes paper_table.csv and the within-subject CSVs this script reads).

    Table 1    within-subject scores and contrasts (cv / holdout, n of 9)
    Table 2    group inference per map: mean, TFCE FWE vertices, TFCE components
               clusters, and the Desikan regions where the TFCE-significant
               effect is largest (instead of cluster peak labels, which name
               one Glasser area for clusters spanning most of a lobe)
    S1         band selection: every configuration, cv and holdout
    S2         per-subject scores
    S3         full group table (adds perm FDR and mean where significant)
    S4         every component of the TFCE-significant vertices of every paper map
    S5         Desikan parcels for the integration maps (emotion, openSMILE)
    S6         integration x context: TFCE vertices per k, paired k contrasts
    S7         null calibration of the two final integration models
    S8         negative controls: prediction, word rate, interaction

    PYTHONPATH=. python3 scripts/paper_tables.py
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from scripts.paper_table import ROWS, EMO, OSM, G

OUT = G / "paper_table" / "tables"
PT = G / "paper_table"
FS = Path("/usr/local/freesurfer/7.4.1/subjects/fsaverage/label")
NV = 163_842
MIN_SUBJ = 5

SHORT = {
    "superiortemporal": "STG", "transversetemporal": "Heschl",
    "bankssts": "banks STS", "middletemporal": "MTG",
    "inferiortemporal": "ITG", "supramarginal": "supramarginal",
    "parsopercularis": "IFGoper", "parstriangularis": "IFGtri",
    "parsorbitalis": "IFGorb", "precentral": "precentral",
    "postcentral": "postcentral", "superiorfrontal": "SFG",
    "caudalmiddlefrontal": "caudal MFG", "rostralmiddlefrontal": "rostral MFG",
    "lateralorbitofrontal": "lat. OFC", "medialorbitofrontal": "med. OFC",
    "precuneus": "precuneus", "inferiorparietal": "IPL",
    "superiorparietal": "SPL", "temporalpole": "temporal pole",
    "insula": "insula", "posteriorcingulate": "PCC",
    "isthmuscingulate": "isthmus cing.", "parahippocampal": "parahippocampal",
    "fusiform": "fusiform", "frontalpole": "frontal pole",
}


def mdt(df, **kw):
    """to_markdown with '|' in labels escaped, so 'emotion | text' does not
    split a cell."""
    e = df.rename(columns=lambda c: str(c).replace("|", "\\|"))
    for c in e.columns:
        if e[c].dtype == object:
            e[c] = e[c].astype(str).str.replace("|", "\\|", regex=False)
    return e.to_markdown(index=False, **kw)


def md(df, path, title, note=""):
    df.to_csv(path.with_suffix(".csv"), index=False)
    with open(path.with_suffix(".md"), "w") as f:
        f.write(f"# {title}\n\n")
        if note:
            f.write(note.strip() + "\n\n")
        f.write(mdt(df) + "\n")


def desikan():
    labs, names = [], None
    for h in ("lh", "rh"):
        lab, _, nm = nib.freesurfer.read_annot(FS / f"{h}.aparc.annot")
        labs.append(lab)
        names = [n.decode() for n in nm]
    return np.concatenate(labs), names


def regions(mean, sig, lab, names, n=4, min_sig=50):
    """Desikan parcels ranked by mean effect over their TFCE-significant
    vertices, among parcels with at least `min_sig` such vertices."""
    hemi = np.r_[np.zeros(NV, int), np.ones(NV, int)]
    out = []
    for h in (0, 1):
        for i, nm in enumerate(names):
            m = sig & (lab == i) & (hemi == h)
            if m.sum() >= min_sig and nm not in ("unknown", "corpuscallosum"):
                out.append((float(np.nanmean(mean[m])), int(m.sum()),
                            f"{'lr'[h]}h {SHORT.get(nm, nm)}"))
    out.sort(reverse=True)
    return "; ".join(f"{r} ({v:.3f})" for v, _, r in out[:n])


def table2():
    pt = pd.read_csv(PT / "paper_table.csv")
    lab, names = desikan()
    rows = []
    for (sec, label, d, stat, kind), (_, p) in zip(ROWS, pt.iterrows()):
        gd = G / d
        mean = np.load(gd / f"{stat}__mean.npy")
        nsub = np.load(gd / f"{stat}__n_subjects.npy")
        tf = np.load(gd / "clusters" / f"{stat}__p_tfce_perm_fwe.npy")
        cov = (nsub >= MIN_SUBJ) & np.isfinite(mean)
        sig = cov & (tf < 0.05)
        rows.append({
            "section": sec, "map": label, "mean": p["mean"],
            "tfce_fwe": int(sig.sum()),
            "pct_covered": round(100 * sig.sum() / cov.sum(), 1),
            "clusters": p["clusters"],
            "regions (mean effect where significant)":
                regions(mean, sig, lab, names) or "-",
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "table2_group.csv", index=False)
    with open(OUT / "table2_group.md", "w") as f:
        f.write("# Table 2. Group inference per map\n\n"
                "n = 9, fsaverage, 61,168 vertices covered by >= 5 subjects. "
                "mean: group mean over covered vertices (r for bands alone, "
                "delta r for 'beyond' maps). tfce_fwe: vertices with TFCE "
                "group-permutation FWE p < .05 (1000 permutations; "
                "Draper-Stoneman conditional nulls for 'X | Y'; integration "
                "= intersection-union of the two conditional tests). "
                "clusters: connected components of the TFCE-significant "
                "vertices. extent_fwe: vertices significant by cluster-extent "
                "FWE p < .05 at z > 3.09, same null. Regions: Desikan "
                "parcels with >= 50 TFCE-significant vertices, ranked by mean "
                "effect over those vertices (top 4).\n\n"
                "Preference r_text - r_emotion (no null of its own): mean "
                "+0.0674, text ahead at 85.8% of covered vertices, sign-flip "
                "FWE 13 vertices (n>=9 family, 3,337). Preference r_text - "
                "r_openSMILE: +0.256, text ahead 96.3%, FWE 2,278.\n")
        for sec, sub in df.groupby("section", sort=False):
            f.write(f"\n### {sec}\n\n"
                    + mdt(sub.drop(columns="section")) + "\n")
    return df


def table1():
    b = pd.read_csv(PT / "within_subject_bands.csv")
    c = pd.read_csv(PT / "within_subject_contrasts.csv")
    note = ("Mean r over each subject's explainable-variance mask (EV > 0.1, "
            "5 repeats), averaged over 9 subjects. cv: 5-fold "
            "cross-validation over the 24 training stories (selection); "
            "holdout: the held-out story (wheretheressmoke, 291 TRs, "
            "reporting). Contrasts: n_pos = subjects with a positive "
            "difference; 9/9 is sign-test p = 0.002.")
    b.to_csv(OUT / "table1_within_subject_bands.csv", index=False)
    c.to_csv(OUT / "table1_within_subject_contrasts.csv", index=False)
    with open(OUT / "table1_within_subject.md", "w") as f:
        f.write("# Table 1. Within-subject encoding scores\n\n" + note + "\n\n"
                + mdt(b) + "\n\n"
                + mdt(c) + "\n")


def s1_selection():
    parts = []
    for ev in ("cv", "holdout"):
        d = pd.read_csv(PT / "selection" / f"selection_{ev}.csv")
        g = d.groupby(["panel", "series", "x"], sort=False)["r"]
        parts.append(g.agg(["mean", "count"]).rename(
            columns={"mean": ev, "count": f"n_{ev}"}))
    df = pd.concat(parts, axis=1).reset_index()
    df["chosen_on_cv"] = ""
    for (pa, se), sub in df.groupby(["panel", "series"]):
        df.loc[sub["cv"].idxmax(), "chosen_on_cv"] = "max"
    df[["cv", "holdout"]] = df[["cv", "holdout"]].round(4)
    # panels of plot_selection_curves (2026-10-15): A emotion layer per seconds of
    # preceding audio, B audio context, C Llama-8B layer per context, D text context
    isA, isC = df["panel"] == "A", df["panel"] == "C"
    df.loc[isA, "series"] = "emotion, " + df.loc[isA, "series"]
    df.loc[isC, "series"] = "Llama-8B, " + df.loc[isC, "series"]
    df["x"] = df["x"].astype(int)
    df["panel"] = df["panel"].map({"A": "audio depth (layer)",
                                   "B": "audio context (s)",
                                   "C": "text depth (layer)",
                                   "D": "text context (words)"})
    md(df, OUT / "S1_selection",
       "Supplementary Table 1. Band selection",
       "Mean r over the EV mask, averaged over subjects, for every "
       "configuration in the selection sweeps (x = layer for depth, seconds of "
       "preceding audio or words of context for context). "
       "Bands were chosen on cv; holdout is reported, never used to choose. "
       "'max' marks the cv argmax within each series.")


def s2_subjects():
    d = pd.read_csv(PT / "within_subject_scores.csv")
    d["model"] = d["run"].str.replace("llama8b_k512L13__", "text x ").str.replace("__ctx", "") \
        + " : " + d["model"]
    df = d.pivot_table(index=["model", "eval"], columns="subject",
                       values="mean_r").round(4).reset_index()
    md(df, OUT / "S2_per_subject",
       "Supplementary Table 2. Per-subject scores",
       "Mean r over each subject's EV mask, for every model of every fit "
       "used in the paper.")


def s3_full():
    pt = pd.read_csv(PT / "paper_table.csv")
    cols = ["section", "map", "n_covered", "mean", "mean_where_sig",
            "perm_fdr", "tfce_fwe", "clusters", "largest_clusters", "extent_fwe"]
    md(pt[cols], OUT / "S3_group_full",
       "Supplementary Table 3. Group inference, all tests",
       "perm_fdr: group permutation, BH-FDR q < .05 per vertex. tfce_fwe: "
       "TFCE FWE p < .05 (the threshold of the figures). clusters / "
       "largest_clusters: connected components of the TFCE-significant "
       "vertices, with each component's size and peak Glasser area. "
       "extent_fwe: vertices significant by cluster-extent FWE p < .05 at "
       "z > 3.09 against the same null. Large components span most of "
       "the temporal lobe, so the peak label names one area of a much larger "
       "territory; see S4 for every area each cluster covers.")


def s4_clusters():
    out = []
    for sec, label, d, stat, kind in ROWS:
        cd = G / d / "clusters"
        csv = (cd / "integration__tfce.csv" if kind == "integration"
               else cd / f"{stat}__tfce__clusters.csv")
        if not csv.exists():
            continue
        c = pd.read_csv(csv)
        if "p_fwe" in c:
            c = c[c["p_fwe"] < 0.05]
        col = ("mean_delta" if kind == "integration"
               else f"mean_{stat}".replace("mean_delta_", "mean_", 1)
               if f"mean_{stat}" not in c else f"mean_{stat}")
        c = c.assign(mean_effect=c[col] if col in c else np.nan)
        # tables written before peak_area existed: largest Glasser area instead
        first = c["glasser_areas"].str.split(" ").str[0]
        c["peak_area"] = c["peak_area"].fillna(first) if "peak_area" in c else first
        c = c[[k for k in ("cluster", "hemi", "n_vertices", "area_mm2",
                           "p_fwe", "peak_z", "peak_area", "mean_effect",
                           "glasser_areas", "glasser_sections") if k in c]]
        c.insert(0, "map", label)
        c.insert(0, "section", sec)
        out.append(c.sort_values("n_vertices", ascending=False))
    df = pd.concat(out, ignore_index=True)
    md(df, OUT / "S4_clusters",
       "Supplementary Table 4. Clusters of the TFCE maps",
       "Every connected component of TFCE-significant vertices (FWE "
       "p < .05, at least 20 mm2) of every paper map, largest first within "
       "map; p_fwe is the smallest vertex p in the component. "
       "Glasser (HCP-MMP1) areas.")


def s5_parcels():
    out = []
    for run, band in ((EMO, "emotion"), (OSM, "openSMILE")):
        for stat, lab in (("delta_audio_given_text", f"{band} | text"),
                          ("delta_text_given_audio", f"text | {band}"),
                          ("delta", "integration")):
            p = pd.read_csv(G / f"{run}__conditional" / "tables"
                            / f"{stat}__parcels_aparc.csv")
            p.insert(0, "map", lab)
            out.append(p)
    df = pd.concat(out, ignore_index=True)
    md(df, OUT / "S5_parcels_integration",
       "Supplementary Table 5. Desikan parcels, integration maps",
       "Per parcel: covered vertices (>= 5 subjects), mean effect over them, "
       "and % of covered vertices significant (group permutation FDR). "
       "Low pct_covered means 'not covered', not 'no effect'.")


def s6_context():
    rows = []
    for k in (4, 16, 64, 256, 512, 1024):
        gd = G / f"llama8bk{k}L13__base_emotionL11__conditional"
        r = {"k (words)": k}
        for stat, lab in (("delta_audio_given_text", "emotion | text"),
                          ("delta_text_given_audio", "text | emotion"),
                          ("delta", "integration")):
            mean = np.load(gd / f"{stat}__mean.npy")
            nsub = np.load(gd / f"{stat}__n_subjects.npy")
            tf = np.load(gd / "clusters" / f"{stat}__p_tfce_perm_fwe.npy")
            cov = (nsub >= MIN_SUBJ) & np.isfinite(mean)
            r[f"{lab} mean"] = round(float(np.nanmean(mean[cov])), 4)
            r[f"{lab} TFCE"] = int((cov & (tf < 0.05)).sum())
        rows.append(r)
    a = pd.DataFrame(rows)
    s = json.load(open(G / "context_contrast_llama" / "summary.json"))
    rows = []
    for pair, v in s.items():
        if pair.startswith("_"):
            continue
        for stat, lab in (("delta", "integration"),
                          ("delta_audio_given_text", "emotion | text"),
                          ("delta_text_given_audio", "text | emotion")):
            rows.append({"k pair": pair.replace(":", " vs "), "map": lab,
                         "mean difference": round(v[stat]["mean_diff"], 4),
                         "TFCE up": v[stat]["p_tfce_up"],
                         "TFCE down": v[stat]["p_tfce_down"]})
    b = pd.DataFrame(rows)
    c = pd.read_csv(G / "roi_context_curve_llama.csv")
    a.to_csv(OUT / "S6_context_by_k.csv", index=False)
    b.to_csv(OUT / "S6_context_contrasts.csv", index=False)
    c.to_csv(OUT / "S6_context_roi_curve.csv", index=False)
    with open(OUT / "S6_context.md", "w") as f:
        f.write("# Supplementary Table 6. Integration x context\n\n"
                "A. Per context length (Llama-3.1-8B L13 x emotion L11): group "
                "mean and TFCE FWE vertices.\n\n" + mdt(a)
                + "\n\nB. Paired sign-flip contrasts between context lengths "
                "(TFCE FWE p < .05 vertices; 'up' = larger at the longer "
                "context). Only regions where integration rises against the "
                "erosion of emotion | text are interpretable.\n\n"
                + mdt(b)
                + "\n\nC. ROI curve (Desikan, per hemisphere and k): "
                "`S6_context_roi_curve.csv` (the data of Fig 7d).\n")


def s7_calibration():
    rows = []
    for run, lab in ((EMO, "text x emotion"), (OSM, "text x openSMILE")):
        c = json.load(open(G / f"{run}__conditional" / "diagnostics"
                           / "calibration.json"))
        for stat, v in c["stats"].items():
            per = [s["empirical_fpr"] for s in v.get("per_subject", {}).values()]
            rows.append({"model": lab, "statistic": stat,
                         "empirical FPR (nominal .05)": round(v["pooled_empirical_fpr"], 4),
                         "per-subject range": (f"{min(per):.4f}-{max(per):.4f}"
                                               if per else ""),
                         "KS max deviation": round(v["ks_max_deviation"], 4)})
    md(pd.DataFrame(rows), OUT / "S7_calibration",
       "Supplementary Table 7. Null calibration",
       "Each saved null draw is treated as an observed statistic and tested "
       "against the other 999 with the same estimator. The intersection-union "
       "rule (delta) is conservative by construction. This validates the "
       "estimator, ties and IU rule; the block length (10 TRs) was checked "
       "separately: blocklen 1 narrows the null by 20-25%, 10/20/40 agree "
       "within 1-3%.")


def s8_controls():
    ctrl = [
        ("prediction (surprisal + entropy) | text + emotion + word rate",
         "llama8bk1024L13__base_emotionL11__pred_given_rest",
         "delta_pred_given_text+audio+wordrate",
         "0 FDR voxels in 9/9; means -0.0020 to +0.0007",
         "group count is a shifted null (raw delta r ~0 where significant): "
         "surprisal adds nothing measurable"),
        ("text | emotion + word rate",
         "llama8bk1024L13__base_emotionL11__text_given_audio_wordrate",
         "delta_text_given_audio+wordrate",
         "cv +0.0589 vs +0.0582 without word rate (ratio 1.01); "
         "FDR-significant in 65-85% of each mask",
         "word rate absorbs none of the text contribution, incl. Heschl"),
        ("text x emotion interaction (20x20 PC products) | text + emotion",
         "llama8bk1024L13__base_emotionL11__inter_pc20",
         "delta_inter_given_text+audio",
         "0 FDR voxels in 9/9; means -0.0004 to +0.0027",
         "combination is additive"),
    ]
    rows = []
    for lab, d, stat, within, concl in ctrl:
        gd = G / d
        mean = np.load(gd / f"{stat}__mean.npy")
        nsub = np.load(gd / f"{stat}__n_subjects.npy")
        cov = (nsub >= MIN_SUBJ) & np.isfinite(mean)
        fdr = np.load(gd / f"{stat}__p_perm_fdr.npy")
        fwe = np.load(gd / f"{stat}__p_signflip_fwe.npy")
        sig = cov & (fdr < 0.05)
        rows.append({"control": lab, "within subject": within,
                     "group mean": round(float(np.nanmean(mean[cov])), 4),
                     "perm FDR vertices": int(sig.sum()),
                     "raw mean where sig": (round(float(np.nanmean(mean[sig])), 4)
                                            if sig.any() else np.nan),
                     "sign-flip FWE (n>=9)": int((fwe < 0.05).sum()),
                     "conclusion": concl})
    md(pd.DataFrame(rows), OUT / "S8_controls",
       "Supplementary Table 8. Negative controls",
       "Each row is a Draper-Stoneman conditional test (1000 permutations) "
       "of one extra band given the carried bands, on the final Llama x "
       "emotion model.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table1()
    t2 = table2()
    print(t2.drop(columns="section").to_string(index=False))
    s1_selection()
    s2_subjects()
    s3_full()
    s4_clusters()
    s5_parcels()
    s6_context()
    s7_calibration()
    s8_controls()
    for p in sorted(OUT.iterdir()):
        print("->", p)


if __name__ == "__main__":
    main()
