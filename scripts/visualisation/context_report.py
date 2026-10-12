#!/usr/bin/env python3
"""
Integration across context lengths, one page: how the cluster/TFCE results of
`group_clusters.py` move with k.

    results/group/k<K>L8__base_emotionL11__conditional/clusters/   (per k)
        --> results/group/context_clusters_report.html

Reads the per-k `summary.json`, the p-maps and the group means; renders
nothing, so it takes seconds. The per-k `clusters/report.html` pages hold the
figures.

Why a per-section table with all three statistics, not only integration:
`delta` is the min of two conditional terms, and as the text band gets
stronger with k the audio term erodes mechanically ("a stronger text band
absorbs prosody"). Growth in `delta` is only interpretable where the min is
bound by the text term -- i.e. where text|audio rises with k -- and falling
`delta` is not interpretable where audio|text is what fell. The table puts
both terms beside `delta` so each row can be read that way.

Usage
-----
    python scripts/visualisation/context_report.py            # k = 4 16 256 512
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import html
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE))

from group_report import CSS, glasser_parcels, img_tag, methods  # noqa: E402
from plot_group_inflated import fsaverage_dir, load_surfaces, render  # noqa: E402
from plot_group_maps import limits  # noqa: E402

STATS = {"delta": "Integration",
         "delta_audio_given_text": "Audio | text",
         "delta_text_given_audio": "Text | audio"}


CONTRAST_TESTS = [("vertex", "Vertex FWE"), ("tfce", "TFCE"),
                  ("t2.9_extent", "Cluster extent t>2.90"),
                  ("t2.9_mass", "Cluster mass t>2.90"),
                  ("t4.5_extent", "Cluster extent t>4.50"),
                  ("t4.5_mass", "Cluster mass t>4.50")]

_R = {}


def _render(job):
    stat, values, vlim, title, f = job
    render(stat, values, _R["surfaces"], True, vlim, title, f, 80,
           ("lateral", "medial"))


def contrast_section(cdir: Path, sections_of, lab_sec, rerender: bool,
                     jobs: int) -> list[str]:
    """Paired sign-flip contrasts between context lengths
    (`scripts/context_contrast.py`)."""
    summary = json.loads((cdir / "summary.json").read_text())
    meta = summary.pop("_meta")
    a2 = meta["alpha_per_direction"]
    out = ["<h2 id='contrast'>Does integration change with k? "
           "Paired sign-flip test</h2>",
           f"""<div class='note'>Per subject, the null-centred map at the
longer context minus the one at the shorter. If k does not matter, the two
labels are exchangeable within a subject, so swapping them (negating that
subject's difference) gives an exact null over all {meta['n_flips']}
relabelings. Random effects, FWE over the {meta['family_min_subjects']}+-subject
family, two-sided as two one-sided maps at α = {a2:g} each. Per-subject
integration is the min of its two centred components.
<b>↑</b> larger at the longer context, <b>↓</b> smaller.</div>"""]
    # counts
    out.append("<div class='scroll'><table><thead><tr><th>Contrast</th>"
               "<th>Statistic</th><th class='num'>mean Δ</th>")
    out += [f"<th class='num'>{lab} ↑ / ↓</th>" for _, lab in CONTRAST_TESTS]
    out.append("</tr></thead><tbody>")
    for pair, stats in summary.items():
        a, b = pair.split(":")
        for stat, e in stats.items():
            out.append(f"<tr><td>k{b} − k{a}</td><td>{STATS[stat]}</td>"
                       f"<td class='num'>{e['mean_diff']:+.4f}</td>")
            for key, _ in CONTRAST_TESTS:
                out.append(f"<td class='num'>{e[f'p_{key}_up']:,} / "
                           f"{e[f'p_{key}_down']:,}</td>")
            out.append("</tr>")
    out.append("</tbody></table></div><p class='muted'>Vertices significant "
               "in each direction.</p>")

    # figures: mean difference where TFCE rejects in either direction
    surfaces = load_surfaces(fsaverage_dir(), "inflated")
    fig_dir = cdir / "figures"
    fig_dir.mkdir(exist_ok=True)
    todo, figs = [], {}
    for pair in summary:
        a, b = pair.split(":")
        d = cdir / f"k{a}_vs_k{b}"
        for stat in STATS:
            diff = np.load(d / f"{stat}__mean_diff.npy").astype(float)
            sig = ((np.load(d / f"{stat}__p_tfce_up.npy") < a2) |
                   (np.load(d / f"{stat}__p_tfce_down.npy") < a2))
            f = fig_dir / f"k{a}_vs_k{b}__{stat}.png"
            figs[pair, stat] = f
            if rerender or not f.exists():
                vals = np.where(sig, np.nan_to_num(diff), 0.0)
                vlim = limits(diff[np.isfinite(diff)], True)
                todo.append((stat, vals, vlim,
                             f"{STATS[stat]}: k{b} − k{a} | TFCE", f))
    _R["surfaces"] = surfaces
    if todo:
        with mp.get_context("fork").Pool(jobs) as pool:
            pool.map(_render, todo)

    lab, sec = lab_sec
    n_half = lab.size // 2
    for pair in summary:
        a, b = pair.split(":")
        d = cdir / f"k{a}_vs_k{b}"
        out.append(f"<h3>k{b} − k{a}</h3><div class='grid'>")
        for stat in STATS:
            n_up = summary[pair][stat]["p_tfce_up"]
            n_dn = summary[pair][stat]["p_tfce_down"]
            out.append(f"<div class='cell'>{img_tag(figs[pair, stat])}"
                       f"<div class='cap'><b>{STATS[stat]}</b> · TFCE ↑ "
                       f"{n_up:,} / ↓ {n_dn:,}</div></div>")
        out.append("</div>")
        fam = np.load(d / "family.npy")
        maps = {(stat, k): np.load(d / f"{stat}__{k}.npy") for stat in STATS
                for k in ("mean_diff", "p_tfce_up", "p_tfce_down")}
        out.append("<details><summary>By Glasser section (TFCE)</summary>"
                   "<div class='scroll'><table><thead><tr><th>hemi</th>"
                   "<th>Section</th><th class='num'>family vertices</th>")
        for stat in STATS:
            out.append(f"<th class='num'>{STATS[stat]}<br>mean Δ</th>"
                       f"<th class='num'>{STATS[stat]}<br>% ↑ / % ↓</th>")
        out.append("</tr></thead><tbody>")
        for hemi, sl in (("lh", slice(0, n_half)), ("rh", slice(n_half, None))):
            for section in sections_of:
                inside = np.zeros(lab.size, bool)
                inside[sl] = sec[sl] == section
                c = inside & fam
                if c.sum() < 50:
                    continue
                out.append(f"<tr><td>{hemi}</td><td>{html.escape(section)}</td>"
                           f"<td class='num'>{int(c.sum()):,}</td>")
                for stat in STATS:
                    md = float(np.nanmean(maps[stat, "mean_diff"][c]))
                    up = float(np.mean(maps[stat, "p_tfce_up"][c] < a2))
                    dn = float(np.mean(maps[stat, "p_tfce_down"][c] < a2))
                    cls = "up" if up > 0.1 else "down" if dn > 0.1 else ""
                    out.append(f"<td class='num {cls}'>{md:+.4f}</td>"
                               f"<td class='num {cls}'>{up:.0%} / {dn:.0%}</td>")
                out.append("</tr>")
        out.append("</tbody></table></div><p class='muted'>Sections with at "
                   "least 50 family vertices. Green / red: more than 10% of "
                   "the section significantly up / down.</p></details>")
    return out


def pmap_path(root: Path, method: str, stat: str) -> Path:
    c = root / "clusters"
    if method == "tfce":
        return c / f"{stat}__p_tfce_perm_fwe.npy"
    if method == "fdr0.05":
        return root / f"{stat}__p_perm_fdr.npy"
    return c / f"{stat}__{method}__p_fwe.npy"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks", nargs="+", type=int, default=[4, 16, 256, 512])
    ap.add_argument("--pattern", default="results/group/k{k}L8__base_emotionL11__conditional")
    ap.add_argument("--primary", nargs="+",
                    default=["z3.09_extent", "tfce", "fdr0.05"],
                    help="methods given a per-section table")
    ap.add_argument("--out", type=Path,
                    default=Path("results/group/context_clusters_report.html"))
    ap.add_argument("--contrast-dir", type=Path,
                    default=Path("results/group/context_contrast"))
    ap.add_argument("--rerender", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args(argv)

    roots = {k: Path(args.pattern.format(k=k)) for k in args.ks}
    roots = {k: r for k, r in roots.items()
             if (r / "clusters" / "summary.json").exists()}
    if not roots:
        raise SystemExit("no clusters/summary.json for any k")
    ks = sorted(roots)
    summ = {k: json.loads((roots[k] / "clusters" / "summary.json").read_text())
            for k in ks}
    ms = [m for m in methods(summ[ks[0]]["z_thresholds"])
          if m["key"] in ("fdr0.05", "tfce") or m.get("table")]
    lab, sec = glasser_parcels(fsaverage_dir())
    sections = sorted({s for s in sec if s})

    means = {(k, s): np.load(roots[k] / f"{s}__mean.npy") for k in ks
             for s in STATS}

    def sig(k, method, s):
        p = np.load(pmap_path(roots[k], method, s))
        return np.isfinite(p) & (p < 0.05) & np.isfinite(means[k, s])

    out = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           "<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Integration across context</title><style>{CSS}"
           "td.up{color:#1a7f37}td.down{color:#9b2226}</style></head>"
           "<body><main><h1>Integration across context length</h1>"
           f"<p class='muted'>Semantic band GPT-2 L8 at k = "
           f"{', '.join(map(str, ks))} words, against emotion L11 · group "
           "permutation null, 1000 draws · Glasser HCP-MMP1 sections · "
           "<a href='#contrast'>jump to the k contrasts</a></p>"]
    out.append("""<div class='note'><b>Reading the section tables.</b>
Integration is the smaller of the two conditional terms. As k grows the text
band strengthens and <i>audio | text</i> erodes mechanically, so a
<b>fall</b> in integration where audio | text falls is not interpretable. A
<b>rise</b> in integration where text | audio rises is. Mean values are over
each section's covered vertices; the per-k pages
(<code>k*/clusters/report.html</code>) hold the figures and cluster
tables.</div>""")

    # ---- summary: vertices and clusters per method per k ----------------
    out.append("<h2>Significant vertices and integration clusters</h2>"
               "<div class='scroll'><table><thead><tr><th>Method</th>"
               "<th>Statistic</th>")
    out += [f"<th class='num'>k = {k}</th>" for k in ks]
    out.append("</tr></thead><tbody>")
    for m in ms:
        for s, lab_s in STATS.items():
            out.append(f"<tr><td>{html.escape(m['label'])}</td><td>{lab_s}</td>")
            for k in ks:
                n = int(sig(k, m["key"], s).sum())
                out.append(f"<td class='num'>{n:,}</td>")
            out.append("</tr>")
        if m.get("table"):
            out.append(f"<tr><td>{html.escape(m['label'])}</td>"
                       "<td>integration clusters ≥ "
                       f"{summ[ks[0]].get('min_area_mm2', 0):g} mm²</td>")
            for k in ks:
                n = summ[k]["methods"][m["key"]]["n_integration_clusters"]
                out.append(f"<td class='num'>{n}</td>")
            out.append("</tr>")
    out.append("</tbody></table></div>")

    # ---- per-section tables ----------------------------------------------
    for key in args.primary:
        m = next((x for x in ms if x["key"] == key), None)
        if m is None:
            continue
        out.append(f"<h2>By Glasser section: {html.escape(m['label'])}</h2>"
                   "<div class='scroll'><table><thead><tr><th>hemi</th>"
                   "<th>Section</th>")
        for s, lab_s in STATS.items():
            out += [f"<th class='num'>{lab_s}<br>k{k}</th>" for k in ks]
        out += ["<th class='num'>% sig integr.<br>k" + str(k) + "</th>"
                for k in ks]
        out.append("</tr></thead><tbody>")
        sigs = {k: sig(k, key, "delta") for k in ks}
        n_half = lab.size // 2
        for hemi, sl in (("lh", slice(0, n_half)), ("rh", slice(n_half, None))):
            for section in sections:
                inside = np.zeros(lab.size, bool)
                inside[sl] = sec[sl] == section
                cov = inside & np.isfinite(means[ks[0], "delta"])
                if cov.sum() < 0.3 * inside.sum() or cov.sum() == 0:
                    continue
                out.append(f"<tr><td>{hemi}</td><td>{html.escape(section)}</td>")
                for s in STATS:
                    vals = [float(np.nanmean(means[k, s][cov])) for k in ks]
                    trend = ("up" if vals[-1] > vals[0] + 0.002 else
                             "down" if vals[-1] < vals[0] - 0.002 else "")
                    out += [f"<td class='num {trend}'>{v:+.4f}</td>"
                            for v in vals]
                out += [f"<td class='num'>{sigs[k][cov].mean():.0%}</td>"
                        for k in ks]
                out.append("</tr>")
        out.append("</tbody></table></div><p class='muted'>Sections listed "
                   "when at least 30% of their vertices are covered. Green / "
                   "red: the mean moves by more than 0.002 from the smallest "
                   "to the largest k.</p>")

    if (args.contrast_dir / "summary.json").exists():
        out += contrast_section(args.contrast_dir, sections, (lab, sec),
                                args.rerender, args.jobs)
    out.append("</main></body></html>")
    args.out.write_text("".join(out))
    print(f"-> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
