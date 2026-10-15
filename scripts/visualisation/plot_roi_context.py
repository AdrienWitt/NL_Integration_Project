#!/usr/bin/env python3
"""
Integration x context by ROI, along the two context axes that cross at the
selected models (text: Llama-8B L13, 512 words; audio: emotion L11, 16 s).

    main   roi_context.png               delta only, 2 x 2:
                                         rows = left / right hemisphere,
                                         columns = text context (audio at 16 s),
                                                   audio context (text at 512 words)
    SI     roi_context_beyond_text.png   the two conditional terms along the text axis
           roi_context_beyond_audio.png  the two conditional terms along the audio axis

delta is the smaller of the two conditional contributions, so it can only be read
with them: where delta falls (IFG, MTG along the text axis) the emotion term is
eroding as the text band strengthens; where it rises (Heschl) the text term is
entering. The SI figures carry that decomposition (decision 2026-10-15: delta in
the main text, the conditional terms in the SI).

Inputs are tidy tables from `scripts/roi_context_curve.py` (Desikan ROIs, group
mean over covered vertices, fraction of covered vertices significant). Run labels
are `k<words>` for the text axis and `w<seconds>` for the audio axis. Filled
markers: >= 50% of the ROI's covered vertices significant (TFCE FWE p < .05);
open markers: fewer. The selected configuration is marked on both axes.

    python scripts/visualisation/plot_roi_context.py \\
        --text-csv results/group/roi_context_curve_llama.csv \\
        --audio-csv results/group/roi_context_curve_audio.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROIS = [("HeschlTransverse", "Heschl", "#2a78d6"),
        ("STG", "STG", "#eb6834"),
        ("STS_banks", "banks STS", "#1baf7a"),
        ("precentral", "precentral", "#eda100"),
        # the three IFG parts drawn separately (2026-10-15): pars orbitalis is the
        # region of the previous study's context effect, and merging it with
        # opercularis/triangularis hid a different curve
        ("IFG_operc", "IFG opercularis", "#e87ba4"),
        ("IFG_triang", "IFG triangularis", "#8e5ad6"),
        ("IFG_orbitalis", "IFG orbitalis", "#a0522d"),
        ("MTG", "MTG", "#008300")]
BEYOND = [("delta_audio_given_text", "Emotion beyond text"),
          ("delta_text_given_audio", "Text beyond emotion")]
AXES = {"text": dict(runs=["k4", "k16", "k64", "k256", "k512", "k1024"], selected="k512",
                     xlabel="words of context (text band)",
                     title="Integration (δ) × text context"),
        "audio": dict(runs=["w0", "w2", "w4", "w8", "w16", "w32"], selected="w16",
                      xlabel="seconds of preceding audio (emotion band)",
                      title="Integration (δ) × audio context")}
HEMI = {"lh": "left", "rh": "right"}
INK, MUTED, GRID, SEL = "#1f1f1e", "#6b6a64", "#e4e3de", "#f1efe8"
# print sizes (2026-10-15: the 8-10 pt labels were unreadable once scaled to a page)
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 14, 13, 12, 12


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=FS_TITLE, color=INK, fontweight="bold")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=FS_TICK)


def panel(ax, d, stat, hemi, axis):
    """One statistic for one hemisphere along one context axis; False if no data yet."""
    spec = AXES[axis]
    runs = [r for r in spec["runs"] if d is not None and r in set(d.run)]
    ax.set_xticks(range(len(spec["runs"])), [r[1:] for r in spec["runs"]])
    ax.set_xlim(-0.4, len(spec["runs"]) - 0.6)
    ax.axvspan(spec["runs"].index(spec["selected"]) - 0.3,
               spec["runs"].index(spec["selected"]) + 0.3, color=SEL, zorder=0)
    if not runs:
        ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center",
                va="center", fontsize=FS_LABEL, color=MUTED)
        return False
    for roi, label, c in ROIS:
        s = (d[(d.stat == stat) & (d.hemi == hemi) & (d.roi == roi)]
             .set_index("run").reindex(spec["runs"]))
        x = range(len(spec["runs"]))
        ax.plot(x, s.mean_r, color=c, lw=2.2, label=label)
        for xi, (r, f) in zip(x, zip(s.mean_r, s.frac_sig_tfce)):
            if pd.notna(r):
                ax.plot(xi, r, "o", ms=7, color=c, mec=c, mew=1.4,
                        mfc=c if f >= 0.5 else "white")
    return True


def legend(fig, ax):
    h, l = ax.get_legend_handles_labels()
    h.append(plt.Line2D([], [], ls="", marker="o", ms=7, mfc="white", mec=MUTED, mew=1.4))
    l.append("open: < 50% of vertices significant")
    h.append(plt.Rectangle((0, 0), 1, 1, color=SEL))
    l.append("selected configuration")
    fig.legend(h, l, loc="outside upper center", ncol=5, frameon=False, fontsize=FS_LEGEND,
               handlelength=1.8, columnspacing=1.2)


def save(fig, out, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"-> {out}/{name}.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--text-csv", type=Path,
                    default=Path("results/group/roi_context_curve_llama.csv"))
    ap.add_argument("--audio-csv", type=Path,
                    default=Path("results/group/roi_context_curve_audio.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/group/paper_table/roi_context"))
    a = ap.parse_args(argv)
    data = {ax: (pd.read_csv(p) if p.exists() else None)
            for ax, p in (("text", a.text_csv), ("audio", a.audio_csv))}
    a.out.mkdir(parents=True, exist_ok=True)

    # main: delta, rows = hemisphere, columns = context axis, one y scale per row
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.4), sharey="row", constrained_layout=True)
    for i, hemi in enumerate(HEMI):
        for j, axis in enumerate(AXES):
            ax = axs[i, j]
            panel(ax, data[axis], "delta", hemi, axis)
            style(ax, f"{AXES[axis]['title']}, {HEMI[hemi]}")
            if i == 1:
                ax.set_xlabel(AXES[axis]["xlabel"], fontsize=FS_LABEL, color=MUTED)
            if j == 0:
                ax.set_ylabel("group mean δ", fontsize=FS_LABEL, color=MUTED)
    legend(fig, axs[0, 0])
    save(fig, a.out, "roi_context")

    # SI: the two conditional terms, one figure per axis
    for axis in AXES:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8.4), sharey="row", constrained_layout=True)
        for i, hemi in enumerate(HEMI):
            for j, (stat, title) in enumerate(BEYOND):
                ax = axs[i, j]
                panel(ax, data[axis], stat, hemi, axis)
                style(ax, f"{title}, {HEMI[hemi]}")
                if i == 1:
                    ax.set_xlabel(AXES[axis]["xlabel"], fontsize=FS_LABEL, color=MUTED)
                if j == 0:
                    ax.set_ylabel("group mean Δr", fontsize=FS_LABEL, color=MUTED)
        legend(fig, axs[0, 0])
        save(fig, a.out, f"roi_context_beyond_{axis}")


if __name__ == "__main__":
    main()
