#!/usr/bin/env python3
"""
Band-selection figure: audio depth, text depth, text context.

    A  audio layer profile   base_emotion (12 blocks), eGeMAPS as the
                             reference line (base_robust dropped from the paper)
    B  text layer profile    Llama-3.1-8B, every layer, one curve per context
                             length k = 4..1024 (per-layer sweeps, 2026-10-14)
    C  context length        Llama-3.1-8B L13 (k=4..1024), read from the same
                             per-layer sweeps as panel B

GPT-2 is still read into the tidy table (S Table 1) but no longer drawn.

Mean r over each subject's EV>0.1 mask, mean over the nine subjects, error bars
one SEM across subjects. The carried band is ringed and its "n/9" is the number
of subjects whose own score is highest there among the shown configurations.
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
    for store, label in [("perlayer_base_emotion", "emotion (12 blocks)")]:
        for s in SUBJECTS:
            for r in read_sweep(sweeps / f"prosody_sweep/{ev}/{store}__{s}/sweep.csv",
                                "audio_r"):
                if r["config"].isdigit():
                    add("A", label, int(r["config"]), s, r["mean_r"])
                elif r["config"] == "opensmile" and store == "perlayer_base_emotion":
                    add("A", "eGeMAPS", -1, s, r["mean_r"])
    for k in LLAMA_KS:
        for s in SUBJECTS:
            for r in read_sweep(sweeps / f"semantic_sweep/{ev}/semantic__{s}_llama8b_k{k}/sweep.csv",
                                "text_r"):
                m = re.fullmatch(r"perlayer_llama8b_k(\d+)_L(\d+)", r["config"])
                if m:
                    add("B", f"k = {k}", int(m.group(2)), s, r["mean_r"])
                    if int(m.group(2)) == 13:
                        add("C", "Llama-8B L13", k, s, r["mean_r"])
    for tag, k in [("k0layers", 0), ("k4layers", 4), ("layers", 16),
                   ("k64layers", 64), ("k256layers", 256), ("k512layers", 512)]:
        for s in SUBJECTS:
            for r in read_sweep(sweeps / f"semantic_sweep/{ev}/semantic__{s}_{tag}/sweep.csv",
                                "text_r"):
                if r["config"] == f"perlayer_gpt2_k{k}_L8":
                    add("C", "GPT-2 L8", k, s, r["mean_r"])
    return pd.DataFrame(rows)


def wins(df, series, x_best):
    """Subjects whose own argmax over this series is at x_best."""
    p = df[df.series == series].pivot(index="subject", columns="x", values="r")
    return int((p.idxmax(axis=1) == x_best).sum()), len(p)


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


def curve(ax, df, series, color, best=None, xs=None, band=True):
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
        k, n = wins(df, series, best)
        ax.annotate(f"{m.iloc[i]:.3f}\n{k}/{n}", (x[i], m.iloc[i]),
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

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    A = df[df.panel == "A"]
    osm = A[A.series == "eGeMAPS"]["r"].mean()
    axs[0].axhline(osm, color=MUTED, lw=1.2, ls="--",
                   label="eGeMAPS (88 features, no layers)")
    curve(axs[0], A, "emotion (12 blocks)", C1, best=11)
    style(axs[0], "A  Audio: layer", "transformer block")

    B = df[df.panel == "B"]
    for k, col in zip(LLAMA_KS, SEQ):
        curve(axs[1], B, f"k = {k}", col, best=13 if k == 1024 else None, band=k == 1024)
    style(axs[1], "B  Text: layer (Llama-3.1-8B)", "layer")

    C = df[df.panel == "C"]
    sym = lambda v: np.where(v == 0, 1, v)  # k=0 drawn at 1 on the log axis
    curve(axs[2], C, "Llama-8B L13", C1, best=1024, xs=sym)
    axs[2].set_xscale("log")
    ticks = [4, 16, 64, 256, 512, 1024]
    axs[2].set_xticks(ticks, [str(t) for t in ticks])
    axs[2].minorticks_off()
    style(axs[2], "C  Text: context", "words of preceding context (k)")

    axs[0].set_ylabel(f"mean r over EV mask ({a.eval})", fontsize=9, color=MUTED)
    axs[0].legend(frameon=False, fontsize=8, loc="upper left")
    axs[1].legend(frameon=False, fontsize=7.5, loc="lower right", ncol=2,
                  title="words of context", title_fontsize=7.5)
    axs[2].legend(frameon=False, fontsize=8, loc="lower right")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(a.out / f"selection_curves_{a.eval}.{ext}", dpi=200,
                    facecolor="white")
    print(f"-> {a.out}/selection_curves_{a.eval}.png")


if __name__ == "__main__":
    main()
