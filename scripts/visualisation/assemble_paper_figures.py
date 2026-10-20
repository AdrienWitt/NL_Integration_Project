#!/usr/bin/env python3
"""
Paper figures assembled from the CONN renders already on disk -- no MATLAB.

Reads the untitled CONN prints `results/group/<run>/export/conn/fig_mosaic8/
<stat>__<test>.surf.png` (written by conn_figures.py stage 1, with their
`.range.txt`), cuts four views out of each (left lateral, left medial, right
medial, right lateral), and lays one map per row with a letter, a short label
and one colorbar per group of rows that share a scale. The colorbar is sampled
from CONN's own bar in the same print, so surface and bar cannot disagree.

Writes to `results/group/paper_figures/`:
    <fig>.{png,pdf,svg}        assembled figures (text editable in the svg)
    panels/<run>__<stat>__<test>.png
                               the 4-view strip of every map used, no text,
                               to place freely in Inkscape / Illustrator

    PYTHONPATH=. python3 scripts/visualisation/assemble_paper_figures.py
    ... --test z3.09_extent__p_fwe     # same figures, cluster-extent threshold
    ... --only fig3 fig7
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conn_figures import colorbar_label  # noqa: E402

G = Path("results/group")
# selected bands (2026-10-16 refit): emotion 16 s, eGeMAPS 2 s, text k512
EMO = "llama8bk512L13__emoctx16L11"
OSM = "llama8bk512L13__osm2"
T = "TEST"            # replaced by --test for maps that have a null

# (x0, y0, x1, y1) in a 3275x2514 mosaic8 print (conn_figures.REF_VIEWS)
VIEWS = {"L lat": (87, 65, 992, 697), "L med": (65, 832, 970, 1464),
         "R med": (1902, 826, 2802, 1462), "R lat": (1873, 59, 2773, 695)}
ORDER = ["L lat", "L med", "R med", "R lat"]
BAR_X, BAR_Y = 3230, (1550, 2480)       # CONN's own colorbar in the print
SIZE = (3275, 2514)

# figure -> rows; a row is (label, run, stat, test) or ("img", path)
FIGS = {
    "fig3_prosody_bands": [
        ("eGeMAPS (88 features)", f"{OSM}__marginalfixed", "audio", T),
        ("wav2vec2 emotion, layer 12 (1024 d)", f"{EMO}__marginalfixed", "audio", T),
        ("Arousal + dominance + valence (3 d)", "avd16__marginalfixed", "joint", T),
    ],
    "fig4_audio_bands_compared": [
        ("Emotion beyond eGeMAPS", "emotion16_vs_osm2__each", "delta_emotion_given_opensmile", T),
        ("eGeMAPS beyond emotion", "emotion16_vs_osm2__each", "delta_opensmile_given_emotion", T),
        ("AVD beyond eGeMAPS", "osm2_vs_avd16__each", "delta_avd_given_opensmile", T),
        ("eGeMAPS beyond AVD", "osm2_vs_avd16__each", "delta_opensmile_given_avd", T),
    ],
    "fig5_affective_dimensions": [
        ("Valence beyond arousal", "av16__conditional", "delta_valence_given_arousal", T),
        ("Arousal beyond valence", "av16__conditional", "delta_arousal_given_valence", T),
    ],
    "fig6_semantics": [
        ("Text: Llama-3.1-8B, layer 13, 512 words", f"{EMO}__marginalfixed", "text", T),
    ],
    "fig7_integration": [
        ("Emotion beyond text", f"{EMO}__conditional", "delta_audio_given_text", T),
        ("Text beyond emotion", f"{EMO}__conditional", "delta_text_given_audio", T),
        ("Integration (conjunction)", f"{EMO}__conditional", "delta", T),
    ],  # the ROI x context curves are their own main figure (roi_context.png), 2026-10-13
    # ---- supplementary
    "figS1_affective_alone": [
        ("Arousal", "avd16__marginalfixed", "arousal", T),
        ("Dominance", "avd16__marginalfixed", "dominance", T),
        ("Valence", "avd16__marginalfixed", "valence", T),
    ],
    "figS2_dominance_unique": [
        ("Dominance beyond arousal + valence", "avd16__conditional", "delta_dominance_given_arousal+valence", T),
    ],
    # modality preference moved to the SI (2026-10-20): it tests no hypothesis,
    # the conditional contributions of Fig 7 answer "does prosody add to meaning"
    "figS_preference": [
        ("Preference r text − r emotion (effect map)", "context_contrast_diff/preference", "preference_ctraw2m20", "unthresholded"),
    ],
    "figS4_integration_opensmile": [
        ("eGeMAPS beyond text", f"{OSM}__conditional", "delta_audio_given_text", T),
        ("Text beyond eGeMAPS", f"{OSM}__conditional", "delta_text_given_audio", T),
        ("Integration (conjunction)", f"{OSM}__conditional", "delta", T),
    ],
    # One figure per statistic, one row per context length (k = 512 repeats Fig 7).
    **{f"figS_context_{name}": [
        (f"{lab}, k = {k}", f"llama8bk{k}L13__emoctx16L11__conditional", st, T)
        for k in (4, 16, 64, 256, 512, 1024)]
       for name, st, lab in [("emotion_beyond_text", "delta_audio_given_text", "Emotion beyond text"),
                             ("text_beyond_emotion", "delta_text_given_audio", "Text beyond emotion"),
                             ("integration", "delta", "Integration")]},
    # unique_text / unique_audio / integration in the partition dir are Fig 7a-c
    # (same means, same vertex counts), so S6 is the one map that is new; the
    # partition counts go in its legend (partition.json)
    "figS6_partition": [
        ("Preference r text − r emotion, ≥ 8/9 subjects agree", f"{EMO}__partition", "preference_consistent", "unthresholded"),
    ],
}

TEST_NOTE = {"p_tfce_perm_fwe": "TFCE, group permutation FWE p < .05",
             "z3.09_extent__p_fwe": "z > 3.09, cluster-extent FWE p < .05"}
INK, MUTED = "#1f1f1e", "#6b6a64"
SHORT = {"r of the winner (+ valence, \u2212 arousal)": "r winner (+V, \u2212A)",
         "\u0394r (valence \u2212 arousal)": "\u0394r (V \u2212 A)"}


def mosaic(run, stat, test):
    p = G / run / "export/conn/fig_mosaic8" / f"{stat}__{test}.surf.png"
    if not p.exists():
        raise FileNotFoundError(p)
    im = Image.open(p).convert("RGB")
    if im.size != SIZE:
        raise ValueError(f"{p}: {im.size}, expected {SIZE}; re-run "
                         "conn_figures.py --skip-print to normalise it")
    return p, np.asarray(im)


def strip(a):
    """The four views side by side, cropped to one common height."""
    views = [a[y0:y1, x0:x1] for x0, y0, x1, y1 in (VIEWS[v] for v in ORDER)]
    h = min(v.shape[0] for v in views)
    gap = np.full((h, 30, 3), 255, np.uint8)
    out = []
    for v in views:
        out += [v[:h], gap]
    return np.concatenate(out[:-1], axis=1)


def bar(a, range_txt):
    """CONN's colorbar from the print, and its end values from range.txt."""
    col = a[BAR_Y[0]:BAR_Y[1], BAR_X - 3:BAR_X + 4].mean(1)
    painted = np.where(col.sum(1) < 735)[0]
    # whole painted span: a diverging bar is near-white in the middle
    r = (painted[0], painted[-1])
    colours = col[r[0] + 3:r[-1] - 2][::-1] / 255.0       # bottom -> top
    # A contoured map (contour_maps.py) carries its outline colours in the
    # first and last colormap rows, so CONN's bar ends in them; take the bar
    # from the colormap file instead, without those two rows.
    stat = range_txt.name.split("__")[0]
    cm = range_txt.parent.parent / f"{stat}.cmap.txt"
    if re.search(r"_ctr[a-z]w\d", stat) and cm.exists():
        colours = np.loadtxt(cm)[1:-1]
    ticks = None
    for line in range_txt.read_text().splitlines():
        k, _, v = line.partition("\t")
        if k == "colorbar_ticklabels":
            ticks = [float(t) for t in v.split("|")]
    return ListedColormap(colours), ticks


def build(name, rows, test, out, panels):
    rows = [(r[0], r[1], r[2], test if r[3] == T else r[3]) if r[0] != "img"
            else r for r in rows]
    ims = []
    for r in rows:
        if r[0] == "img":
            ims.append(np.asarray(Image.open(r[1]).convert("RGB")))
        else:
            ims.append(strip(mosaic(*r[1:])[1]))
    W, ax_w = 7.2, 0.85                         # inches; axes fraction of W
    heights = [W * ax_w * im.shape[0] / im.shape[1] for im in ims]
    title_h = 0.22                              # inches per row for the label
    H = sum(heights) + title_h * len(rows) + 0.1
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(len(rows), 2, width_ratios=[1, 0.025],
                          height_ratios=heights, hspace=title_h * len(rows) / sum(heights),
                          wspace=0.03, left=0.04, right=0.04 + ax_w + 0.02,
                          top=1 - 0.2 / H, bottom=0.05 / H)
    groups = []                                   # (first row, last row, cmap, ticks, label)
    for i, r in enumerate(rows):
        letter = "abcdefghijklmnop"[i]
        ax = fig.add_subplot(gs[i, 0])
        ax.axis("off")
        if r[0] == "img":
            ax.imshow(ims[i])
        else:
            label, run, stat, tst = r
            p, a = mosaic(run, stat, tst)
            s = ims[i]
            ax.imshow(s)
            Image.fromarray(s).save(panels / f"{run}__{stat}__{tst}.png")
            cmap, ticks = bar(a, p.parent / (p.name[:-4] + ".range.txt"))
            note = "" if tst in (test, "unthresholded") or tst not in TEST_NOTE else f" ({TEST_NOTE[tst]})"
            ax.set_title(label + note, loc="left", fontsize=7.5, color=INK, pad=2)
            key = (tuple(ticks), SHORT.get(colorbar_label(stat, run), colorbar_label(stat, run)))
            if groups and tuple(groups[-1][3:]) == key and groups[-1][1] == i - 1:
                groups[-1][1] = i
            else:
                groups.append([i, i, cmap, *key])
        if len(rows) > 1:  # a single-map figure needs no panel letter
            ax.text(-0.03, 1.0, letter, transform=ax.transAxes, fontsize=10,
                    fontweight="bold", va="bottom", ha="right", color=INK)
    for first, last, cmap, ticks, lab in groups:
        cax = fig.add_subplot(gs[first:last + 1, 1])
        span = last - first + 1
        pad = 0.18 if span == 1 else 0.08
        pos = cax.get_position()
        cax.set_position([pos.x0, pos.y0 + pos.height * pad, pos.width,
                          pos.height * (1 - 2 * pad)])
        lo, hi = ticks[0], ticks[-1]
        cax.imshow(cmap(np.linspace(0, 1, 256))[:, None, :3], aspect="auto",
                   origin="lower", extent=[0, 1, lo, hi])
        cax.set_xticks([])
        cax.yaxis.tick_right()
        cax.set_yticks(ticks, [f"{t:g}" for t in ticks])
        cax.tick_params(labelsize=6.5, colors=MUTED, length=2)
        cax.set_ylabel(lab, fontsize=7, color=MUTED, labelpad=1)
        cax.yaxis.set_label_position("right")
        for sp in cax.spines.values():           # no outline around the bar
            sp.set_visible(False)
    suffix = "" if test == "p_tfce_perm_fwe" else "__" + test
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}{suffix}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"-> {out / (name + suffix)}.png  ({len(rows)} rows, {len(groups)} colorbar(s))")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--test", default="p_tfce_perm_fwe",
                    choices=["p_tfce_perm_fwe", "z3.09_extent__p_fwe"])
    ap.add_argument("--only", nargs="*", help="figure name prefixes")
    ap.add_argument("--out", type=Path, default=G / "paper_figures")
    a = ap.parse_args(argv)
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42,
                         "font.family": "DejaVu Sans"})
    panels = a.out / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    for name, rows in FIGS.items():
        if a.only and not any(name.startswith(o) for o in a.only):
            continue
        if a.test != "p_tfce_perm_fwe" and all(r[0] == "img" or r[3] != T for r in rows):
            continue                               # nothing thresholded to redo
        build(name, rows, a.test, a.out, panels)


if __name__ == "__main__":
    main()
