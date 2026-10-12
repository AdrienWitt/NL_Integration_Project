#!/usr/bin/env python3
"""
Integration (`delta`) as a grid: inference method x context length, one page.

    rows     every method of group_report.methods() -- uncorrected, FDR,
             sign-flip FWE, TFCE, cluster extent/mass at each threshold
    columns  k = 4, 16, 256, 512
        --> results/group/integration_grid/index.html

Unlike the per-k reports, every cell shares ONE colour scale (taken over all
k), so a colour difference between columns is a difference in effect size and
not in scaling. Figures are cached in integration_grid/figures/.

The paired k-contrast maps from `scripts/context_contrast.py` (TFCE) close the
page, since the grid shows each k tested against its own null and cannot by
itself say that two k differ.

Usage
-----
    python scripts/visualisation/integration_grid.py
"""
from __future__ import annotations

import argparse
import html
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE))

from group_report import CSS, img_tag, methods, pmap  # noqa: E402
from plot_group_inflated import fsaverage_dir, load_surfaces, render  # noqa: E402
from plot_group_maps import limits  # noqa: E402

_R = {}


def _render(job):
    values, vlim, title, f = job
    render("delta", values, _R["surfaces"], False, vlim, title, f, 70,
           ("lateral", "medial"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks", nargs="+", type=int, default=[4, 16, 256, 512])
    ap.add_argument("--pattern",
                    default="results/group/k{k}L8__base_emotionL11__conditional")
    ap.add_argument("--contrast-dir", type=Path,
                    default=Path("results/group/context_contrast"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/group/integration_grid"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--rerender", action="store_true")
    args = ap.parse_args(argv)

    roots = {k: Path(args.pattern.format(k=k)) for k in args.ks}
    summ = {k: json.loads((r / "clusters" / "summary.json").read_text())
            for k, r in roots.items()}
    ms = methods(summ[args.ks[0]]["z_thresholds"])
    means = {k: np.load(r / "delta__mean.npy").astype(float)
             for k, r in roots.items()}
    covered = {k: np.isfinite(m) for k, m in means.items()}
    vlim = limits(np.concatenate([m[np.isfinite(m)] for m in means.values()]),
                  False)

    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    todo, cells = [], {}
    for m in ms:
        for k in args.ks:
            p = pmap(m, roots[k], "delta")
            sig = np.isfinite(p) & (p < m["alpha"]) & covered[k]
            f = fig_dir / f"{m['key']}__k{k}.png"
            n_clu = summ[k]["methods"].get(m["key"], {}).get(
                "n_integration_clusters")
            cells[m["key"], k] = (f, int(sig.sum()), int(covered[k].sum()),
                                  n_clu)
            if args.rerender or not f.exists():
                todo.append((np.where(sig, np.nan_to_num(means[k]), 0.0), vlim,
                             f"k = {k} | {m['label']}", f))

    contrast = []
    cdir = args.contrast_dir
    if (cdir / "summary.json").exists():
        cs = json.loads((cdir / "summary.json").read_text())
        a2 = cs.pop("_meta")["alpha_per_direction"]
        for pair, stats in cs.items():
            a, b = pair.split(":")
            d = cdir / f"k{a}_vs_k{b}"
            diff = np.load(d / "delta__mean_diff.npy").astype(float)
            sig = ((np.load(d / "delta__p_tfce_up.npy") < a2) |
                   (np.load(d / "delta__p_tfce_down.npy") < a2))
            f = fig_dir / f"contrast_k{a}_vs_k{b}.png"
            contrast.append((f"k{b} − k{a}", f, stats["delta"]))
            if args.rerender or not f.exists():
                todo.append(("signed", np.where(sig, np.nan_to_num(diff), 0.0),
                             limits(diff[np.isfinite(diff)], True),
                             f"Integration k{b} − k{a} | paired TFCE", f))

    _R["surfaces"] = load_surfaces(fsaverage_dir(), "inflated")
    plain = [t for t in todo if not isinstance(t[0], str)]
    signed = [t[1:] for t in todo if isinstance(t[0], str)]
    if plain:
        print(f"rendering {len(plain)} figures")
        with mp.get_context("fork").Pool(args.jobs) as pool:
            pool.map(_render, plain)
    for values, lim, title, f in signed:
        render("delta", values, _R["surfaces"], True, lim, title, f, 70,
               ("lateral", "medial"))

    ks = args.ks
    out = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           "<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Integration by method and context</title><style>{CSS}"
           ".grid4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));"
           "gap:10px}@media (max-width:900px){.grid4{grid-template-columns:"
           "repeat(2,minmax(0,1fr))}}</style></head><body><main>"
           "<h1>Integration by method, threshold and context length</h1>"
           "<p class='muted'>delta = min(audio | text, text | audio) · GPT-2 "
           "L8 against emotion L11 · nine subjects · one colour scale for "
           f"every map (0 to {vlim[1]:.3f}) · α = 0.05 unless stated</p>"]
    out.append("<nav><a href='#counts'>Counts</a>" + "".join(
        f"<a href='#{m['key']}'>{html.escape(m['label'])}</a>" for m in ms)
        + "<a href='#contrast'>k contrasts</a></nav>")

    out.append("<h2 id='counts'>Significant integration vertices</h2>"
               "<div class='scroll'><table><thead><tr><th>Method</th>")
    out += [f"<th class='num'>k = {k}</th>" for k in ks]
    out.append("</tr></thead><tbody>")
    for m in ms:
        out.append(f"<tr><td><a href='#{m['key']}'>{html.escape(m['label'])}"
                   "</a></td>")
        for k in ks:
            _, n, cov, n_clu = cells[m["key"], k]
            extra = "" if n_clu is None else f" · {n_clu} cl."
            out.append(f"<td class='num'>{n:,} ({n / cov:.0%}){extra}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div><p class='muted'>Percent of the "
               "covered family (61,168 vertices; the sign-flip row can only "
               "reach the 3,337-vertex core). “cl.” = integration clusters "
               "≥ 20 mm². Each k is tested against its own null; whether two "
               "k differ is the paired test at the bottom.</p>")

    group = None
    for m in ms:
        if m["group"] != group:
            group = m["group"]
            out.append(f"<h2>{html.escape(group)}</h2>")
        out.append(f"<h3 id='{m['key']}'>{html.escape(m['label'])}</h3>"
                   "<div class='grid4'>")
        for k in ks:
            f, n, cov, n_clu = cells[m["key"], k]
            extra = "" if n_clu is None else f" · {n_clu} clusters"
            out.append(f"<div class='cell'>{img_tag(f)}<div class='cap'>"
                       f"<b>k = {k}</b> · {n:,} vertices ({n / cov:.0%})"
                       f"{extra}</div></div>")
        out.append("</div>")

    if contrast:
        out.append("<h2 id='contrast'>Does integration differ between k? "
                   "Paired sign-flip, TFCE</h2><div class='note'>Within "
                   "subject, the k labels are exchangeable if context does "
                   "not matter; swapping them negates that subject's "
                   "difference, so the exact null is all 512 sign flips. "
                   "Red = larger at the longer context, blue = smaller; "
                   "each direction at α = 0.025.</div><div class='grid4'>")
        for label, f, e in contrast:
            out.append(f"<div class='cell'>{img_tag(f)}<div class='cap'>"
                       f"<b>{label}</b> · ↑ {e['p_tfce_up']:,} / ↓ "
                       f"{e['p_tfce_down']:,} vertices · mean Δ "
                       f"{e['mean_diff']:+.4f}</div></div>")
        out.append("</div>")

    out.append("</main></body></html>")
    page = args.out / "index.html"
    page.write_text("".join(out))
    print(f"-> {page} ({page.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
