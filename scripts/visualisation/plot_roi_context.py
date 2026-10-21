#!/usr/bin/env python3
"""
Integration x context by ROI, along the two context axes that cross at the
selected models (text: Llama-8B L13, 512 words; audio: emotion layer 12, 8 s).

    main   roi_context.png               delta only, 2 x 2:
                                         rows = left / right hemisphere,
                                         columns = text context (audio at 8 s),
                                                   audio context (text at 512 words)
    SI     roi_context_beyond_text.png   the two conditional terms along the text axis
           roi_context_beyond_audio.png  the two conditional terms along the audio axis

delta is the smaller of the two conditional contributions, so it can only be read
with them: where delta falls (IFG, MTG along the text axis) the emotion term is
eroding as the text band strengthens; where it rises (Heschl) the text term is
entering. The SI figures carry that decomposition (decision 2026-10-15: delta in
the main text, the conditional terms in the SI).

Inputs are tidy tables from `scripts/roi_context_curve.py` (psn atlas ROIs, group
mean over covered vertices, fraction of covered vertices significant). Run labels
are `k<words>` for the text axis and `w<seconds>` for the audio axis. Filled
markers: >= 50% of the ROI's covered vertices significant (TFCE FWE p < .05);
open markers: fewer. The selected configuration is shaded on both axes.

Style (2026-10-15 restyle): ROIs in the reference categorical order,
validated (adjacent CVD dE >= 9.2, normal-vision >= 15.6) before aSTS was
added in a lighter tint of pSTS's green (2026-10-19, psn atlas); temporal ROIs solid
with circles, parietal dotted with triangles, frontal dashed with squares, so identity never rests on hue
alone; legend grouped by lobe; recessive grid, no frame; one y scale per row.

    python scripts/visualisation/plot_roi_context.py \\
        --text-csv results/group/roi_context_curve_text8.csv \\
        --audio-csv results/group/roi_context_curve_audio.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# (key, label, colour, lobe); the IFG parts are drawn separately (pars orbitalis is
# the region of the previous study's context effect, and merging it with
# opercularis/triangularis hid a different curve)
ROIS = [("early_auditory", "early auditory", "#2a78d6", "temporal"),
        ("STG", "STG", "#eb6834", "temporal"),
        ("aSTS", "aSTS", "#7fd4b4", "temporal"),
        ("pSTS", "pSTS", "#1baf7a", "temporal"),
        ("MTG", "MTG", "#008300", "temporal"),
        ("angular", "angular", "#b2367d", "parietal"),
        ("supramarginal", "supramarginal", "#8a6d3b", "parietal"),
        ("precentral", "precentral", "#eda100", "frontal"),
        ("IFG_oper", "IFG opercularis", "#e87ba4", "frontal"),
        ("IFG_tri", "IFG triangularis", "#6250d6", "frontal"),
        ("IFG_orb", "IFG orbitalis", "#e34948", "frontal")]
LOBE = {"temporal": dict(ls="-", marker="o"), "parietal": dict(ls=(0, (1, 1.5)), marker="^"),
        "frontal": dict(ls=(0, (4, 2)), marker="s")}
BEYOND = [("delta_audio_given_text", "Emotion beyond text"),
          ("delta_text_given_audio", "Text beyond emotion")]
STAT_LABEL = {"delta": "Integration δ"} | dict(BEYOND)
AXES = {"text": dict(runs=["k4", "k16", "k64", "k256", "k512", "k1024"], selected="k512",
                     xlabel="words of context (text band)", name="text context",
                     fixed="audio context fixed at 8 s"),
        "audio": dict(runs=["w0", "w2", "w4", "w8", "w16"], selected="w8",
                      xlabel="seconds of preceding audio (emotion band)", name="audio context",
                      fixed="text context fixed at 512 words")}
HEMI = {"lh": ("L", "Left hemisphere"), "rh": ("R", "Right hemisphere")}
SURFACE, INK, INK2, MUTED, GRID, SEL = "#ffffff", "#1f1f1e", "#52514e", "#8a8983", "#ebeae5", "#f3f1ea"
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 15, 13, 12, 12
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlepad": 10})


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=INK2, labelsize=FS_TICK, length=0, pad=6)
    ax.axhline(0, color=MUTED, lw=1.1, zorder=1)


def panel(ax, d, stat, hemi, axis):
    """One statistic for one hemisphere along one context axis."""
    spec = AXES[axis]
    n = len(spec["runs"])
    ax.set_xticks(range(n), [r[1:] for r in spec["runs"]])
    ax.set_xlim(-0.4, n - 0.6)
    s_i = spec["runs"].index(spec["selected"])
    ax.axvspan(s_i - 0.32, s_i + 0.32, color=SEL, zorder=0, lw=0)
    if d is None or not set(spec["runs"]) & set(d.run):
        ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center",
                va="center", fontsize=FS_LABEL, color=MUTED)
        return
    for roi, label, c, lobe in ROIS:
        s = (d[(d.stat == stat) & (d.hemi == hemi) & (d.roi == roi)]
             .set_index("run").reindex(spec["runs"]))
        x = range(n)
        ax.plot(x, s.mean_r, color=c, lw=2, ls=LOBE[lobe]["ls"], zorder=2,
                solid_capstyle="round", dash_capstyle="round")
        for xi, (r, f) in zip(x, zip(s.mean_r, s.frac_sig_tfce)):
            if pd.isna(r):
                continue
            filled = f >= 0.5
            ax.plot(xi, r, LOBE[lobe]["marker"], ms=8, zorder=3,
                    mfc=c if filled else SURFACE, mec=SURFACE if filled else c,
                    mew=1.6 if filled else 2)


def legend(fig, ax_leg):
    ax_leg.axis("off")
    groups = (("temporal", "Temporal", 0.0, 2), ("parietal", "Parietal", 0.27, 1),
              ("frontal", "Frontal", 0.43, 2))
    for lobe, title, x, ncol in groups:
        h = [Line2D([], [], color=c, lw=2, ls=LOBE[lobe]["ls"], marker=LOBE[lobe]["marker"],
                    ms=8, mfc=c, mec=SURFACE, mew=1.6)
             for _, _, c, lb in ROIS if lb == lobe]
        l = [label for _, label, _, lb in ROIS if lb == lobe]
        lg = ax_leg.legend(h, l, title=title, loc="upper left", bbox_to_anchor=(x, 1.0),
                           ncol=ncol, frameon=False, fontsize=FS_LEGEND, title_fontsize=FS_LEGEND,
                           alignment="left", handlelength=2.6, columnspacing=1.4)
        lg.get_title().set_color(INK2)
        lg.get_title().set_fontweight("bold")
        ax_leg.add_artist(lg)
    # filled vs open, shown in every lobe shape (circle, triangle, square)
    shapes = [LOBE[g[0]]["marker"] for g in groups]
    key = [tuple(Line2D([], [], ls="", marker=m, ms=8, mfc=INK2, mec=SURFACE, mew=1.6)
                 for m in shapes),
           tuple(Line2D([], [], ls="", marker=m, ms=8, mfc=SURFACE, mec=INK2, mew=2)
                 for m in shapes),
           Patch(color=SEL)]
    lg = ax_leg.legend(key, ["≥ 50% significant", "< 50% significant", "selected model"],
                       handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
                       title="Markers", loc="upper left", bbox_to_anchor=(0.78, 1.0),
                       frameon=False, fontsize=FS_LEGEND, title_fontsize=FS_LEGEND, alignment="left")
    lg.get_title().set_color(INK2)
    lg.get_title().set_fontweight("bold")


# Label of each region on the lateral view of the LEFT hemisphere (front to the
# left), in image fractions; outside [0, 1] is the margin around the brain. The
# right hemisphere is drawn mirrored, so its labels take x -> 1 - x.
BRAIN_LABELS = {"precentral": (0.32, -0.12), "IFG_oper": (-0.04, 0.22),
                "IFG_tri": (-0.06, 0.48), "IFG_orb": (0.0, 0.88),
                "early_auditory": (0.68, -0.12), "supramarginal": (1.02, 0.06),
                "angular": (1.14, 0.30), "pSTS": (1.14, 0.54),
                "STG": (1.12, 0.72), "MTG": (0.82, 1.14), "aSTS": (0.30, 1.14)}
BRAIN_XLIM, BRAIN_YLIM = (-0.42, 1.42), (1.24, -0.2)


def brain(ax, hemi, atlas_dir):
    """The regions in their curve colours on the CONN lateral view of one
    hemisphere (plot_roi_atlas_panel.py), each labelled with a leader line."""
    import json
    ax.axis("off")
    img = plt.imread(atlas_dir / f"roi_atlas_{hemi}.png")
    ax.imshow(img, extent=(0, 1, 1, 0))
    ax.set_xlim(*BRAIN_XLIM)
    ax.set_ylim(*BRAIN_YLIM)
    ax.set_aspect(img.shape[0] / img.shape[1], adjustable="datalim")
    meta = json.load(open(atlas_dir / "roi_atlas_centroids.json"))
    cents = meta["centroids"][hemi]
    on_disc = set(meta.get("on_disc", {}).get(hemi, []))
    # the posterior labels stacked in the margin take their rows in the
    # vertical order of their discs, so that the leader lines never cross
    slots = {roi: BRAIN_LABELS[roi] for roi in BRAIN_LABELS}
    stack = [r for r in ("angular", "pSTS", "STG") if r in cents]
    for roi, pos in zip(sorted(stack, key=lambda r: cents[r][1]),
                        sorted((BRAIN_LABELS[r] for r in stack), key=lambda p: p[1])):
        slots[roi] = pos
    for roi, label, c, _ in ROIS:
        if roi not in cents:
            continue
        lx, ly = slots[roi]
        if hemi == "rh":
            lx = 1 - lx
        ha = "right" if lx < 0.5 else "left"
        ax.annotate(label, xy=cents[roi], xytext=(lx, ly), fontsize=FS_TICK - 1, color=INK,
                    ha=ha if not 0 <= lx <= 1 else "center", va="center",
                    arrowprops=dict(arrowstyle="-", color=c, lw=1.6, shrinkA=2,
                                    shrinkB=5.5 if roi in on_disc else 0))
        # a ring around the 6-mm disc (the curve's vertices), leaving it visible;
        # a small dot where the disc is hidden in this view or absent
        if roi in on_disc:
            ax.plot(*cents[roi], "o", ms=11, mfc="none", mec=INK, mew=1.1)
        else:
            ax.plot(*cents[roi], "o", ms=4, mfc=c, mec=SURFACE, mew=1)


def grid_figure(data, stats_cols, axis_of_col, ylabel, out, name, atlas_dir=None):
    """Columns = (statistic, context axis); rows = hemispheres, each named by a bold
    horizontal row header in the left margin (L / R), spelled out in the y label.
    With `atlas_dir`, a bottom row shows where each region is, left and right
    hemisphere side by side, so the curves keep the full width."""
    brains = atlas_dir is not None and (atlas_dir / "roi_atlas_lh.png").exists()
    fig = plt.figure(figsize=(15.5, 16.5 if brains else 10.6), facecolor=SURFACE)
    # the legend sits right under the curves it explains, the brains below it
    gs = fig.add_gridspec(4 if brains else 2, 2, hspace=0.24, wspace=0.08,
                          height_ratios=[1, 1, 0.3, 0.95] if brains else None,
                          left=0.115, right=0.985, top=0.94, bottom=0.01 if brains else 0.2)
    legend(fig, fig.add_subplot(gs[2, :]) if brains
           else fig.add_axes([0.115, 0.005, 0.87, 0.115]))
    axs = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)]
    for i in range(2):
        axs[i][1].sharey(axs[i][0])
    brain_axes = []
    for i, hemi in enumerate(HEMI):
        for j, stat in enumerate(stats_cols):
            ax = axs[i][j]
            spec = AXES[axis_of_col[j]]
            style(ax)
            panel(ax, data[axis_of_col[j]], stat, hemi, axis_of_col[j])
            if i == 0:
                ax.set_title(f"{STAT_LABEL[stat]} across {spec['name']}", loc="left",
                             fontsize=FS_TITLE, color=INK, fontweight="bold", pad=26)
                ax.text(0, 1.035, spec["fixed"], transform=ax.transAxes, ha="left",
                        va="bottom", fontsize=FS_LABEL, color=INK2)
            # x labels on both rows: the brain row below breaks the column
            ax.set_xlabel(spec["xlabel"], fontsize=FS_LABEL, color=INK2, labelpad=8)
            if i == 0 and not brains:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(f"{HEMI[hemi][1]}: {ylabel}", fontsize=FS_LABEL, color=INK2, labelpad=8)
            else:
                ax.tick_params(labelleft=False)
        bb = axs[i][0].get_position()
        fig.text(0.015, (bb.y0 + bb.y1) / 2, HEMI[hemi][0], ha="left", va="center",
                 fontsize=FS_TITLE + 2, color=INK, fontweight="bold")
        if brains:
            ax = fig.add_subplot(gs[3, i])
            brain_axes.append(ax)
            brain(ax, hemi, atlas_dir)
            ax.set_title(f"Regions, {HEMI[hemi][1].lower()}", loc="left", fontsize=FS_TITLE,
                         color=INK, fontweight="bold", pad=4)
    # panel letters, read row by row: curves a-d (left hemisphere first), brains e-f
    panels = [axs[0][0], axs[0][1], axs[1][0], axs[1][1]] + (brain_axes if brains else [])
    for letter, ax in zip("abcdef", panels):
        bb = ax.get_position()
        fig.text(bb.x0 - 0.035, bb.y1 + (0.035 if ax in axs[0] else 0.012), letter,
                 ha="left", va="bottom", fontsize=FS_TITLE + 4, color=INK, fontweight="bold")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {out}/{name}.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--text-csv", type=Path,
                    default=Path("results/group/roi_context_curve_text8_disc.csv"))
    ap.add_argument("--audio-csv", type=Path,
                    default=Path("results/group/roi_context_curve_audio8_disc.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/group/paper_table/roi_context"))
    ap.add_argument("--atlas-dir", type=Path, default=Path("results/group/roi_atlas"),
                    help="plot_roi_atlas_panel.py output; adds the region column "
                         "to the main figure when present")
    a = ap.parse_args(argv)
    data = {ax: (pd.read_csv(p) if p.exists() else None)
            for ax, p in (("text", a.text_csv), ("audio", a.audio_csv))}
    a.out.mkdir(parents=True, exist_ok=True)

    # main: delta along the two context axes
    grid_figure(data, ["delta", "delta"], ["text", "audio"], "integration δ", a.out, "roi_context",
                atlas_dir=a.atlas_dir)
    # SI: the two conditional terms along each axis
    for axis in AXES:
        grid_figure(data, [s for s, _ in BEYOND], [axis, axis], "Δr", a.out,
                    f"roi_context_beyond_{axis}")


if __name__ == "__main__":
    main()
