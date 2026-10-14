#!/usr/bin/env python3
"""
Readout resolution and acoustic context for the two prosody bands.

    A  gain over the paper's grid (R = 2 s, W = 0) as a function of the seconds
       of preceding audio W, for the emotion model (best block at each W) and
       eGeMAPS, at R = 2 s and at R = 0.5 s resampled with Lanczos. The number
       next to each point is how many of the nine subjects gain.
    B  emotion layer profile at R = 0.5 s, one curve per W.

R and W are defined in `extract.emotion_context` and `extract.opensmile_context`.
Mean r over each subject's EV>0.1 mask, mean over subjects, band one SEM of the
per-subject gain. Inputs are the sweep.csv files of
`scripts/cluster/{emoctx,osmctx}_sweep.sbatch` under <sweeps>/prosody_sweep/<eval>/.

    python scripts/visualisation/plot_acoustic_context.py --sweeps results/encoding \\
        --out results/group/paper_table/selection
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SUBJECTS = [f"UTS0{i}" for i in range(1, 10)]
WS = [0, 2, 4, 8, 16]
RS = {"2": "R = 2 s", "0p5": "R = 0.5 s"}
C1, C2 = "#2a78d6", "#eb6834"
SEQ = ["#b9d3f2", "#8db7ea", "#5f99df", "#2563b3", "#173f78"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def tidy(sweeps: Path, ev: str) -> pd.DataFrame:
    rows, root = [], sweeps / "prosody_sweep" / ev
    for r_tag in RS:
        for w in WS:
            for s in SUBJECTS:
                for r in csv.DictReader(open(root / f"perlayer_emoctx_r{r_tag}_w{w}__{s}/sweep.csv")):
                    if r["config"].isdigit():
                        rows.append(dict(band="emotion", R=r_tag, W=w, layer=int(r["config"]),
                                         subject=s, r=float(r["mean_r"])))
                for r in csv.DictReader(open(root / f"perlayer_base_emotion__{s}_opensmile_r{r_tag}_w{w}/sweep.csv")):
                    if r["config"].startswith("opensmile"):
                        rows.append(dict(band="eGeMAPS", R=r_tag, W=w, layer=-1,
                                         subject=s, r=float(r["mean_r"])))
    return pd.DataFrame(rows)


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sweeps", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/group/paper_table/selection"))
    ap.add_argument("--eval", default="cv", choices=["cv", "holdout"])
    a = ap.parse_args(argv)
    df = tidy(a.sweeps, a.eval)
    cnt = df.groupby(["band", "R", "W", "layer"]).size()
    if (cnt != 9).any():
        raise SystemExit(f"not 9 subjects everywhere:\n{cnt[cnt != 9]}")
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / f"acoustic_context_{a.eval}.csv", index=False)

    # emotion: block chosen on the group mean at each (R, W); eGeMAPS has none
    m = df.groupby(["band", "R", "W", "layer"])["r"].mean()
    best = m.groupby(["band", "R", "W"]).idxmax()
    pick = pd.concat([df.set_index(["band", "R", "W", "layer"]).loc[[i]] for i in best])
    P = pick.reset_index().pivot_table(index=["band", "R", "W"], columns="subject", values="r")

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for band, col in (("emotion", C1), ("eGeMAPS", C2)):
        ref = P.loc[(band, "2", 0)]
        for r_tag, ls in (("2", "-"), ("0p5", "--")):
            g = P.loc[(band, r_tag)].sub(ref, axis=1)
            mu, se = g.mean(1), g.sem(1)
            axs[0].fill_between(WS, mu - se, mu + se, color=col, alpha=0.12, lw=0)
            axs[0].plot(WS, mu.values, color=col, ls=ls, marker="o", ms=4, mec="white",
                        mew=0.8, label=f"{band}, {RS[r_tag]}")
            for w in WS:
                if (band, r_tag, w) == (band, "2", 0):
                    continue
                axs[0].annotate(f"{int((g.loc[w] > 0).sum())}", (w, mu.loc[w]),
                                textcoords="offset points", xytext=(0, 5), ha="center",
                                fontsize=6.5, color=col)
    axs[0].axhline(0, color=MUTED, lw=0.8)
    axs[0].set_xticks(WS)
    axs[0].set_ylabel(f"gain in mean r over R = 2 s, W = 0 ({a.eval})", fontsize=9, color=MUTED)
    axs[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    style(axs[0], "A  Readout and acoustic context", "seconds of preceding audio (W)")

    E = df[(df.band == "emotion") & (df.R == "0p5")].groupby(["W", "layer"])["r"]
    for w, col in zip(WS, SEQ):
        mu, se = E.mean().loc[w], E.sem().loc[w]
        axs[1].plot(mu.index, mu.values, color=col, marker="o", ms=3, mec="white",
                    mew=0.6, label=f"W = {w} s")
    axs[1].set_ylabel(f"mean r over EV mask ({a.eval})", fontsize=9, color=MUTED)
    axs[1].legend(frameon=False, fontsize=7.5, loc="upper left")
    style(axs[1], "B  Emotion model, R = 0.5 s: layer", "transformer block")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(a.out / f"acoustic_context_{a.eval}.{ext}", dpi=200, facecolor="white")
    print(f"-> {a.out}/acoustic_context_{a.eval}.png")


if __name__ == "__main__":
    main()
