#!/usr/bin/env python3
"""
Integration x context by ROI: how audio|text, text|audio and delta move with
the text band's context length, per hemisphere.

Reads `results/group/roi_context_curve_llama.csv` (written by
`scripts/roi_context_curve.py`): Desikan-based ROIs, group mean over covered
vertices, fraction of covered vertices significant (group permutation FDR and TFCE FWE).
Filled markers: >= 50% of the ROI's covered vertices significant (TFCE FWE
p < .05, the paper's threshold); open markers: fewer.

Read with the confound in mind: audio|text erodes as the text band strengthens,
so only ROIs where delta GROWS with k (Heschl, STG, banks STS, precentral) are
interpretable; where it falls (IFG, MTG) the min is bound by the audio term.

    python scripts/visualisation/plot_roi_context.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# categorical slots 1-6 of the reference palette, fixed order
ROIS = [("HeschlTransverse", "Heschl", "#2a78d6"),
        ("STG", "STG", "#eb6834"),
        ("STS_banks", "banks STS", "#1baf7a"),
        ("precentral", "precentral", "#eda100"),
        ("IFG", "IFG", "#e87ba4"),
        ("MTG", "MTG", "#008300")]
STATS = [("delta_audio_given_text", "Emotion beyond text"),
         ("delta_text_given_audio", "Text beyond emotion"),
         ("delta", "Integration (δ)")]
K = ["k4", "k16", "k64", "k256", "k512", "k1024"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
HEMI = {"lh": "left", "rh": "right"}
# print sizes (2026-10-15: the 8-10 pt labels were unreadable once the figure is scaled to a page)
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 14, 13, 12, 12


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", type=Path,
                    default=Path("results/group/roi_context_curve_llama.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/group/paper_table/roi_context"))
    a = ap.parse_args(argv)
    d = pd.read_csv(a.csv)
    a.out.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 3, figsize=(13, 7.6), sharex=True,
                            constrained_layout=True)
    x = range(len(K))
    for i, hemi in enumerate(["lh", "rh"]):
        for j, (stat, title) in enumerate(STATS):
            ax = axs[i, j]
            for roi, label, c in ROIS:
                s = (d[(d.stat == stat) & (d.hemi == hemi) & (d.roi == roi)]
                     .set_index("run").reindex(K))
                ax.plot(x, s.mean_r, color=c, lw=2.2, label=label)
                for xi, (r, f) in enumerate(zip(s.mean_r, s.frac_sig_tfce)):
                    ax.plot(xi, r, "o", ms=7, color=c, mec=c, mew=1.4,
                            mfc=c if f >= 0.5 else "white")
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_title(f"{title}, {HEMI[hemi]}", loc="left", fontsize=FS_TITLE,
                         color=INK, fontweight="bold")
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=FS_TICK)
            ax.set_xticks(list(x), [k[1:] for k in K])
            if i == 1:
                ax.set_xlabel("words of context (text band)",
                              fontsize=FS_LABEL, color=MUTED)
            if j == 0:
                ax.set_ylabel("group mean Δr", fontsize=FS_LABEL, color=MUTED)
    h, l = axs[0, 0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], ls="", marker="o", ms=7, mfc="white",
                        mec=MUTED, mew=1.4))
    l.append("open marker: < 50% of covered vertices significant (TFCE FWE)")
    fig.legend(h, l, loc="outside upper center", ncol=4, frameon=False,
               fontsize=FS_LEGEND)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(a.out / f"roi_context.{ext}", dpi=200, facecolor="white")
    print(f"-> {a.out}/roi_context.png")


if __name__ == "__main__":
    main()
