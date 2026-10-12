#!/usr/bin/env python3
"""
Band-selection figure: audio depth, text depth, text context.

    A  audio layer profile   base_emotion (12 blocks), base_robust (24 blocks),
                             openSMILE as the reference line
    B  text layer profile    Llama-3.1-8B at k=256 and k=1024, GPT-2 k=256,
                             x = relative depth so 12- and 32-block models align
    C  context length        Llama-3.1-8B L13 (k=4..1024) and GPT-2 L8 (k=0..512)

Mean r over each subject's EV>0.1 mask, mean over the nine subjects, error bars
one SEM across subjects. The carried band is ringed and its "n/9" is the number
of subjects whose own score is highest there among the shown configurations.
Selection is on cv (the project rule), so cv is the default; `--eval holdout`
draws the same figure at the same carried layers.

These are *selection* curves, not inference: each point is an EV-mask mean.
The per-vertex permutation maps of the carried bands are the `marginalfixed`
group maps.

Inputs are the sweep.csv files from Baobab's results/encoding/{prosody,semantic}
_sweep/<eval>/ (`--sweeps`), plus the Llama k4/k16 text scores, which exist only
inside the joint fits (`--llama-k`, rows of within_subject_scores.csv format).
Panel C's Llama points therefore come from the joint runner, whose single-band
text fits differ from the sweep's L13 by <= 0.0003 (a separate alpha search).
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
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def read_sweep(path, kind):
    return [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if r["score_kind"] == kind]


def tidy(sweeps: Path, llama_k: Path, ev: str) -> pd.DataFrame:
    rows = []
    add = lambda panel, series, x, subj, r: rows.append(
        dict(panel=panel, series=series, x=x, subject=subj, r=float(r)))
    for store, label in [("perlayer_base_emotion", "emotion (12 blocks)"),
                         ("perlayer_base_robust", "robust (24 blocks)")]:
        for s in SUBJECTS:
            for r in read_sweep(sweeps / f"prosody_sweep/{ev}/{store}__{s}/sweep.csv",
                                "audio_r"):
                if r["config"].isdigit():
                    add("A", label, int(r["config"]), s, r["mean_r"])
                elif r["config"] == "opensmile" and store == "perlayer_base_emotion":
                    add("A", "openSMILE", -1, s, r["mean_r"])
    for tag, label, n in [("llama8b_k1024", "Llama-8B k=1024", 32),
                          ("llama8b_k256", "Llama-8B k=256", 32),
                          ("k256layers", "GPT-2 k=256", 12)]:
        for s in SUBJECTS:
            for r in read_sweep(sweeps / f"semantic_sweep/{ev}/semantic__{s}_{tag}/sweep.csv",
                                "text_r"):
                m = re.fullmatch(r"perlayer_(\w+?)_k(\d+)_L(\d+)", r["config"])
                if m and (tag.startswith("k") == (m.group(1) == "gpt2")):
                    add("B", label, int(m.group(3)) / n, s, r["mean_r"])
    for tag, k in [("k0layers", 0), ("k4layers", 4), ("layers", 16),
                   ("k64layers", 64), ("k256layers", 256), ("k512layers", 512)]:
        for s in SUBJECTS:
            for r in read_sweep(sweeps / f"semantic_sweep/{ev}/semantic__{s}_{tag}/sweep.csv",
                                "text_r"):
                if r["config"] == f"perlayer_gpt2_k{k}_L8":
                    add("C", "GPT-2 L8", k, s, r["mean_r"])
    d = pd.read_csv(llama_k)
    d = d[(d["model"] == "text") & (d["eval"] == ev)]
    for run, sub in d.groupby("run"):
        k = int(re.search(r"_k(\d+)L13", run).group(1))
        for r in sub.itertuples():
            add("C", "Llama-8B L13", k, r.subject, r.mean_r)
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


def curve(ax, df, series, color, best=None, xs=None):
    g = df[df.series == series].groupby("x")["r"]
    m, se = g.mean(), g.sem()
    x = m.index.values if xs is None else xs(m.index.values)
    ax.fill_between(x, m - se, m + se, color=color, alpha=0.15, lw=0)
    ax.plot(x, m.values, color=color, lw=2, marker="o", ms=4, label=series,
            mec="white", mew=0.8)
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
    ap.add_argument("--llama-k", type=Path, required=True)
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
    osm = A[A.series == "openSMILE"]["r"].mean()
    axs[0].axhline(osm, color=MUTED, lw=1.2, ls="--")
    axs[0].text(23, osm, " openSMILE", va="bottom", ha="right", fontsize=8, color=MUTED)
    curve(axs[0], A, "emotion (12 blocks)", C1, best=11)
    curve(axs[0], A, "robust (24 blocks)", C2)
    style(axs[0], "A  Audio: layer", "transformer block")

    B = df[df.panel == "B"]
    curve(axs[1], B, "Llama-8B k=1024", C1, best=13 / 32)
    curve(axs[1], B, "Llama-8B k=256", C3)
    curve(axs[1], B, "GPT-2 k=256", C2)
    style(axs[1], "B  Text: layer", "relative depth (layer / blocks)")

    C = df[df.panel == "C"]
    sym = lambda v: np.where(v == 0, 1, v)  # k=0 drawn at 1 on the log axis
    curve(axs[2], C, "Llama-8B L13", C1, best=1024, xs=sym)
    curve(axs[2], C, "GPT-2 L8", C2, xs=sym)
    axs[2].set_xscale("log")
    ticks = [1, 4, 16, 64, 256, 1024]
    axs[2].set_xticks(ticks, ["0", "4", "16", "64", "256", "1024"])
    axs[2].minorticks_off()
    style(axs[2], "C  Text: context", "words of preceding context (k)")

    axs[0].set_ylabel(f"mean r over EV mask ({a.eval})", fontsize=9, color=MUTED)
    axs[0].legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axs[1:]:
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(a.out / f"selection_curves_{a.eval}.{ext}", dpi=200,
                    facecolor="white")
    print(f"-> {a.out}/selection_curves_{a.eval}.png")


if __name__ == "__main__":
    main()
