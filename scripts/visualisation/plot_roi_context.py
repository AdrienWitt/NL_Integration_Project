#!/usr/bin/env python3
"""
Integration x context by ROI: how audio|text, text|audio and delta move with
the text band's context length, per hemisphere.

Reads `results/group/roi_context_curve_llama.csv` (written by
`scripts/roi_context_curve.py`): Desikan-based ROIs, group mean over covered
vertices, fraction of covered vertices significant (group permutation FDR).
Filled markers: >= 50% of the ROI's covered vertices significant; open
markers: fewer.

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
STATS = [("delta_audio_given_text", "Audio beyond text"),
         ("delta_text_given_audio", "Text beyond audio"),
         ("delta", "Integration (delta)")]
K = ["k4", "k16", "k256", "k1024"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", type=Path,
                    default=Path("results/group/roi_context_curve_llama.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/group/paper_table/roi_context"))
    a = ap.parse_args(argv)
    d = pd.read_csv(a.csv)
    a.out.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True,
                            constrained_layout=True)
    x = range(len(K))
    for i, hemi in enumerate(["lh", "rh"]):
        for j, (stat, title) in enumerate(STATS):
            ax = axs[i, j]
            for roi, label, c in ROIS:
                s = (d[(d.stat == stat) & (d.hemi == hemi) & (d.roi == roi)]
                     .set_index("run").reindex(K))
                ax.plot(x, s.mean_r, color=c, lw=2, label=label)
                for xi, (r, f) in enumerate(zip(s.mean_r, s.frac_sig_perm)):
                    ax.plot(xi, r, "o", ms=6, color=c, mec=c, mew=1.4,
                            mfc=c if f >= 0.5 else "white")
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_title(f"{title}, {hemi}", loc="left", fontsize=10,
                         color=INK, fontweight="bold")
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.set_xticks(list(x), [k[1:] for k in K])
            if i == 1:
                ax.set_xlabel("words of context (k), Llama-8B L13",
                              fontsize=9, color=MUTED)
            if j == 0:
                ax.set_ylabel("group mean delta r", fontsize=9, color=MUTED)
    h, l = axs[0, 0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], ls="", marker="o", ms=6, mfc="white",
                        mec=MUTED, mew=1.4))
    l.append("open marker: < 50% of covered vertices significant (perm FDR)")
    fig.legend(h, l, loc="outside upper center", ncol=7, frameon=False,
               fontsize=8)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(a.out / f"roi_context.{ext}", dpi=200, facecolor="white")
    print(f"-> {a.out}/roi_context.png")


if __name__ == "__main__":
    main()
