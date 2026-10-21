#!/usr/bin/env python3
"""
Band-selection figure: audio depth, text depth, text context.

    A  audio layer profile   emotion model (12 blocks), one curve per seconds
                             of preceding audio W (0.5-s readout, 2026-10-15);
                             selected eGeMAPS as the reference line
    B  audio context         emotion L12 and eGeMAPS against W (0.5-s readout);
                             eGeMAPS summarises [t-W, t+0.5], the emotion model
                             sees W s before the pooled 0.5 s
    C  text layer profile    Llama-3.1-8B, every layer, one curve per context
                             length k = 4..1024 (per-layer sweeps, 2026-10-14)
    D  text context          Llama-3.1-8B L13 (k=4..1024), read from the same
                             per-layer sweeps as panel C

GPT-2 is no longer read (not part of the paper's comparison, 2026-10-15).

Mean r over each subject's EV>0.1 mask, mean over the nine subjects, error bars
one SEM across subjects. The carried band is ringed. Layers (A, C) are chosen by
the highest mean, and "n/9" is the number of subjects whose own score is highest
there. Context (B, D) is lengthened while the longer one improves in a majority
of subjects (2026-10-20 rule), and "next n/9" is the number of subjects who gain
from the next, rejected, step.
Selection is on cv (the project rule), so cv is the default; `--eval holdout`
draws the same figure at the same carried layers.

These are *selection* curves, not inference: each point is an EV-mask mean.
The per-vertex permutation maps of the carried bands are the `marginalfixed`
group maps.

Inputs are the sweep.csv files from Baobab's results/encoding/{prosody,semantic}
_sweep/<eval>/ (`--sweeps`). Until 2026-10-14 the Llama k4/k16 points of panel C
came from the joint fits (`--llama-k`), because only k256/k1024 had per-layer
sweeps; every k has one now, so B and C read the same numbers. `--llama-k` is
still accepted and ignored.
The tidy table actually drawn is written next to the figure.

    python scripts/visualisation/plot_selection_curves.py --sweeps <dir> \\
        --llama-k <csv> --out results/group/paper_table/selection
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUBJECTS = [f"UTS0{i}" for i in range(1, 10)]
# categorical slots 1-3 of the reference palette, fixed order
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
LLAMA_KS = [4, 16, 64, 256, 512, 1024]
AUDIO_WS = [0, 2, 4, 8, 16, 32]
W_EMO, W_OSM, L_EMO, K_TEXT, L_TEXT = 8, 2, 12, 512, 13  # carried configs
# Emotion blocks are numbered 1-12 like Llama layers (hidden_states[0] is the CNN
# output, so sweep config "11" is block 12, the last; 2026-10-15 fix).
# ordered context lengths -> one sequential ramp (light = short), the carried k darkest
SEQ = ["#b9d3f2", "#8db7ea", "#5f99df", "#3a7dd0", "#2563b3", "#173f78"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def read_sweep(path, kind):
    return [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if r["score_kind"] == kind]


def tidy(sweeps: Path, llama_k: Path, ev: str) -> pd.DataFrame:
    rows = []
    add = lambda panel, series, x, subj, r: rows.append(
        dict(panel=panel, series=series, x=x, subject=subj, r=float(r)))
    for w in AUDIO_WS:
        for s in SUBJECTS:
            for r in read_sweep(sweeps / f"prosody_sweep/{ev}/perlayer_emoctx_r0p5_w{w}__{s}/sweep.csv",
                                "audio_r"):
                if r["config"].isdigit():
                    block = int(r["config"]) + 1
                    add("A", f"W = {w} s", block, s, r["mean_r"])
                    if block == L_EMO:
                        add("B", f"emotion L{L_EMO}", w, s, r["mean_r"])
            for r in read_sweep(sweeps / f"prosody_sweep/{ev}/perlayer_base_emotion__{s}_opensmile_r0p5_w{w}/sweep.csv",
                                "audio_r"):
                if r["config"] == f"opensmile_r0p5_w{w}":
                    add("B", "eGeMAPS", w, s, r["mean_r"])
    for k in LLAMA_KS:
        for s in SUBJECTS:
            for r in read_sweep(sweeps / f"semantic_sweep/{ev}/semantic__{s}_llama8b_k{k}/sweep.csv",
                                "text_r"):
                m = re.fullmatch(r"perlayer_llama8b_k(\d+)_L(\d+)", r["config"])
                if m:
                    add("C", f"k = {k}", int(m.group(2)), s, r["mean_r"])
                    if int(m.group(2)) == L_TEXT:
                        add("D", f"Llama-8B L{L_TEXT}", k, s, r["mean_r"])
    return pd.DataFrame(rows)


def wins(df, series, x_best):
    """Subjects whose own argmax over this series is at x_best."""
    p = df[df.series == series].pivot(index="subject", columns="x", values="r")
    return int((p.idxmax(axis=1) == x_best).sum()), len(p)


def next_gain(df, series, x_best):
    """Subjects who gain from the step after x_best (None at the last point)."""
    p = df[df.series == series].pivot(index="subject", columns="x", values="r")
    xs = list(p.columns)
    i = xs.index(x_best)
    if i + 1 == len(xs):
        return None
    return int((p[xs[i + 1]] > p[x_best]).sum()), len(p)


def style(ax, title, xlabel):
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def curve(ax, df, series, color, best=None, xs=None, band=True, step=False):
    g = df[df.series == series].groupby("x")["r"]
    m, se = g.mean(), g.sem()
    x = m.index.values if xs is None else xs(m.index.values)
    if band:
        ax.fill_between(x, m - se, m + se, color=color, alpha=0.15, lw=0)
    ax.plot(x, m.values, color=color, lw=2 if band else 1.5, marker="o",
            ms=4 if band else 3, label=series, mec="white", mew=0.8)
    if best is not None:
        i = list(m.index).index(best)
        ax.plot(x[i], m.iloc[i], "o", ms=11, mfc="none", mec=INK, mew=1.4)
        if step and next_gain(df, series, best):
            k, n = next_gain(df, series, best)
            note = f"next {k}/{n}"
        else:
            k, n = wins(df, series, best)
            note = f"{k}/{n}"
        ax.annotate(f"{m.iloc[i]:.3f}\n{note}", (x[i], m.iloc[i]),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=7.5, color=INK)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sweeps", type=Path, required=True)
    ap.add_argument("--llama-k", type=Path, default=None, help="ignored (see docstring)")
    ap.add_argument("--out", type=Path, default=Path("results/group/paper_table/selection"))
    ap.add_argument("--eval", default="cv", choices=["cv", "holdout"])
    a = ap.parse_args(argv)
    df = tidy(a.sweeps, a.llama_k, a.eval)
    cnt = df.groupby(["panel", "series", "x"]).size()
    if (cnt != 9).any():
        raise SystemExit(f"not 9 subjects everywhere:\n{cnt[cnt != 9]}")
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / f"selection_{a.eval}.csv", index=False)

    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.6), constrained_layout=True)
    axs = axs.ravel()
    A = df[df.panel == "A"]
    B = df[df.panel == "B"]
    osm = B[(B.series == "eGeMAPS") & (B.x == W_OSM)]["r"].mean()
    axs[0].axhline(osm, color=MUTED, lw=1.2, ls="--",
                   label=f"eGeMAPS (W = {W_OSM} s)")
    for w, col in zip(AUDIO_WS, SEQ):
        curve(axs[0], A, f"W = {w} s", col, best=L_EMO if w == W_EMO else None,
              band=w == W_EMO)
    style(axs[0], "A  Audio: layer (emotion model)", "layer")

    # durations double from 2 s on: evenly spaced, as the log axis of panel D
    pos = lambda v: np.array([AUDIO_WS.index(int(w)) for w in v])
    curve(axs[1], B, f"emotion L{L_EMO}", C1, best=W_EMO, xs=pos, step=True)
    curve(axs[1], B, "eGeMAPS", C2, best=W_OSM, xs=pos, step=True)
    axs[1].set_xticks(range(len(AUDIO_WS)), [str(w) for w in AUDIO_WS])
    style(axs[1], "B  Audio: context", "seconds of preceding audio (W)")

    C = df[df.panel == "C"]
    # carried k drawn last, on top: k1024 sits within 0.0006 of it at every layer and,
    # drawn over it, looked like the winner
    for k, col in sorted(zip(LLAMA_KS, SEQ), key=lambda kc: kc[0] == K_TEXT):
        curve(axs[2], C, f"k = {k}", col, best=L_TEXT if k == K_TEXT else None,
              band=k == K_TEXT)
    h, l = axs[2].get_legend_handles_labels()
    order = sorted(range(len(l)), key=lambda i: int(l[i].split("= ")[1]))
    axs[2].legend([h[i] for i in order], [l[i] for i in order], frameon=False, fontsize=7.5,
                  loc="lower right", ncol=2, title="words of context", title_fontsize=7.5)
    style(axs[2], "C  Text: layer (Llama-3.1-8B)", "layer")

    D = df[df.panel == "D"]
    sym = lambda v: np.where(v == 0, 1, v)  # k=0 drawn at 1 on the log axis
    curve(axs[3], D, f"Llama-8B L{L_TEXT}", C1, best=K_TEXT, xs=sym, step=True)
    axs[3].set_xscale("log")
    ticks = [4, 16, 64, 256, 512, 1024]
    axs[3].set_xticks(ticks, [str(t) for t in ticks])
    axs[3].minorticks_off()
    style(axs[3], "D  Text: context", "words of preceding context (k)")

    for i in (0, 2):
        axs[i].set_ylabel(f"mean r over EV mask ({a.eval})", fontsize=9, color=MUTED)
    axs[0].legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2,
                  title="seconds of preceding audio", title_fontsize=7.5)
    axs[1].legend(frameon=False, fontsize=8, loc="lower left")
    axs[3].legend(frameon=False, fontsize=8, loc="lower right")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(a.out / f"selection_curves_{a.eval}.{ext}", dpi=200,
                    facecolor="white")
    print(f"-> {a.out}/selection_curves_{a.eval}.png")


if __name__ == "__main__":
    main()
