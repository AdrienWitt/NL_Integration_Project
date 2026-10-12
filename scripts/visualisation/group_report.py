#!/usr/bin/env python3
"""
One self-contained HTML page comparing every inference method on a group run.

    results/group/<run>/                 vertex-level p-maps (group_maps.py)
    results/group/<run>/clusters/        cluster + TFCE maps (group_clusters.py)
        --> results/group/<run>/clusters/report.html

Rows are methods, columns the three statistics of the conditional analysis
(audio beyond text, text beyond audio, and their conjunction `delta`). Each cell
is the group mean drawn on the inflated surface where the method rejects, with
one colour scale per statistic across all methods, so a difference between two
cells is a difference in what survived and never in how it was drawn.

Methods, all at alpha = 0.05 unless the row says otherwise:

    vertex, uncorrected p_perm < .05 / .01 / .001     no multiplicity control
    vertex, permutation FDR q < .05 / .01             p_perm_fdr
    vertex, sign-flip FWE (n = 9 core)                p_signflip_fwe
    TFCE, permutation FWE                             p_tfce_perm_fwe
    cluster extent, z > 2.33 / 3.09                   z<thr>_extent
    cluster mass,   z > 2.33 / 3.09                   z<thr>_mass

Usage
-----
    python scripts/visualisation/group_report.py \\
        --root results/group/k256L8__base_emotionL11__conditional
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import logging
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE))

from plot_group_inflated import (  # noqa: E402
    fsaverage_dir, load_surfaces, render,
)
from plot_group_maps import limits  # noqa: E402

log = logging.getLogger("group_report")

STATS = {
    "delta_audio_given_text": "Audio beyond text",
    "delta_text_given_audio": "Text beyond audio",
    "delta": "Integration (conjunction)",
}


def methods(z_thresholds: list[float]) -> list[dict]:
    """(key, label, family, how to load p for a statistic)."""
    out = []
    for a in (0.05, 0.01, 0.001):
        out.append(dict(key=f"unc{a:g}", group="Vertex-level",
                        label=f"Uncorrected p_perm < {a:g}",
                        file="{root}/{stat}__p_perm.npy", alpha=a))
    for q in (0.05, 0.01):
        out.append(dict(key=f"fdr{q:g}", group="Vertex-level",
                        label=f"Permutation FDR q < {q:g}",
                        file="{root}/{stat}__p_perm_fdr.npy", alpha=q))
    out.append(dict(key="signflip", group="Vertex-level",
                    label="Sign-flip max-t FWE (n = 9 core only)",
                    file="{root}/{stat}__p_signflip_fwe.npy", alpha=0.05))
    out.append(dict(key="tfce", group="TFCE",
                    label="TFCE, permutation FWE",
                    file="{root}/clusters/{stat}__p_tfce_perm_fwe.npy",
                    conj="{root}/clusters/delta__p_tfce_perm_fwe.npy",
                    alpha=0.05))
    for kind in ("extent", "mass"):
        for z in z_thresholds:
            t = f"z{z:g}_{kind}"
            p_equiv = {2.33: "0.01", 3.09: "0.001"}.get(z, "")
            out.append(dict(
                key=t, group=f"Cluster {kind}", table=t,
                label=f"Cluster {kind}, z > {z:g}"
                      + (f" (p < {p_equiv})" if p_equiv else ""),
                file="{root}/clusters/{stat}__" + t + "__p_fwe.npy",
                conj="{root}/clusters/delta__" + t + "__p_fwe.npy",
                alpha=0.05))
    return out


def pmap(m: dict, root: Path, stat: str) -> np.ndarray | None:
    pattern = m["conj"] if stat == "delta" and "conj" in m else m["file"]
    f = Path(pattern.format(root=root, stat=stat))
    return np.load(f).astype(float) if f.exists() else None


_R = {}  # render globals, inherited by fork


def _render(job) -> None:
    stat, shown, vlim, title, f = job
    render(stat, shown, _R["surfaces"], False, vlim, title, f, _R["dpi"],
           ("lateral", "medial"))


def img_tag(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img loading="lazy" src="data:image/png;base64,{data}" alt="">'


def csv_table(path: Path, cols: list[str], limit: int) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return "<p class='muted'>No significant clusters.</p>"
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    cols = [c for c in cols if c in rows[0]]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(r[c])}</td>" for c in cols)
        + "</tr>" for r in rows[:limit])
    more = (f"<p class='muted'>{len(rows) - limit} more rows in "
            f"<code>{path.name}</code>.</p>" if len(rows) > limit else "")
    return (f"<div class='scroll'><table><thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>{more}")


def glasser_parcels(fsdir: Path):
    """Per-vertex Glasser label (e.g. `L_A1`) and section, lh-then-rh."""
    import nibabel.freesurfer.io as fsio
    sections = json.loads(
        (fsdir / "label" / "HCPMMP1_sections.json").read_text())["sections"]
    labels = []
    for hemi, pre in (("lh", "L"), ("rh", "R")):
        lab, _, names = fsio.read_annot(fsdir / "label" /
                                        f"{hemi}.HCPMMP1.annot")
        short = [n.decode().split("_")[1] if not n.startswith(b"???")
                 else "" for n in names] + [""]
        labels.append(np.array([f"{pre}_{a}" if a else "" for a in short],
                               dtype=object)[lab])
    lab = np.concatenate(labels)
    sec = np.array([sections.get(x[2:], "") for x in lab], dtype=object)
    return lab, sec


def parcel_table_html(method: str, sigs: dict, means: dict, covered: dict,
                      parcels, min_cov: float, limit: int) -> str:
    """Glasser areas ranked by % of covered vertices significant for the
    conjunction, then by mean delta. Areas under `min_cov` coverage are left
    out: a parcel the family barely reaches has a percentage of almost
    nothing."""
    lab, sec = parcels
    cov = covered["delta"]
    rows = []
    for name in np.unique(lab[lab != ""]):
        inside = lab == name
        c = inside & cov
        if c.sum() == 0 or c.sum() / inside.sum() < min_cov:
            continue
        row = {"area": name, "section": sec[inside][0],
               "covered": f"{c.sum() / inside.sum():.0%}"}
        for s, short in (("delta", "sig integr."),
                         ("delta_audio_given_text", "sig audio|text"),
                         ("delta_text_given_audio", "sig text|audio")):
            sig = sigs.get((method, s))
            row[short] = "—" if sig is None else f"{sig[c].mean():.0%}"
        for s, short in (("delta", "mean delta"),
                         ("delta_audio_given_text", "mean audio|text"),
                         ("delta_text_given_audio", "mean text|audio")):
            row[short] = f"{np.nanmean(means[s][c]):+.4f}"
        sig_d = sigs.get((method, "delta"))
        row["_k"] = (0.0 if sig_d is None else float(sig_d[c].mean()),
                     float(np.nanmean(means["delta"][c])))
        rows.append(row)
    rows.sort(key=lambda r: r["_k"], reverse=True)
    rows = [r for r in rows if r["_k"][0] > 0][:limit]
    if not rows:
        return "<p class='muted'>No Glasser area has significant vertices.</p>"
    cols = [k for k in rows[0] if k != "_k"]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(
        f"<td>{html.escape(str(r[c]))}</td>" for c in cols) + "</tr>"
        for r in rows)
    return (f"<div class='scroll'><table><thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


INTEGRATION_COLS = [
    "rank", "hemi", "area_mm2", "peak_area", "peak_section", "glasser_areas",
    "p_fwe", "mean_delta", "mean_audio_given_text", "mean_text_given_audio",
    "r_text", "r_audio", "preference", "mean_subjects_sig_both",
    "mean_n_subjects", "pct_tfce_sig"]
COMPONENT_COLS = [
    "cluster", "hemi", "area_mm2", "glasser_sections", "glasser_areas",
    "p_fwe", "peak_z", "mean_audio_given_text", "mean_text_given_audio",
    "pct_tfce_sig"]

CSS = """
:root{--bg:#fbfbf9;--fg:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--card:#fff;
--accent:#9b2226;--head:#f1f0ec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#161615;--fg:#ecebe7;--muted:#a09f99;--line:#34332f;--card:#1f1f1d;
--accent:#e07a5f;--head:#262624}}
:root[data-theme="dark"]{--bg:#161615;--fg:#ecebe7;--muted:#a09f99;
--line:#34332f;--card:#1f1f1d;--accent:#e07a5f;--head:#262624}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1400px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.6rem;margin:0 0 4px}h2{font-size:1.25rem;margin:40px 0 8px;
border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:1rem;margin:20px 0 6px}
.muted{color:var(--muted)}code{font-size:.9em}
nav{position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:2;
border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:6px 14px;
font-size:.9rem}nav a{color:var(--accent);text-decoration:none}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
@media (max-width:900px){.grid{grid-template-columns:1fr}}
.cell{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:8px}.cell img{width:100%;height:auto;display:block;border-radius:4px;
background:#fff}.cell .cap{font-size:.85rem;margin-top:4px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;font-size:.82rem;width:100%}
th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:left;
white-space:nowrap}th{background:var(--head);position:sticky;top:0}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.note{background:var(--card);border-left:3px solid var(--accent);
padding:8px 12px;margin:12px 0;font-size:.9rem}
details summary{cursor:pointer;color:var(--accent);margin:8px 0}
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=80)
    ap.add_argument("--table-rows", type=int, default=25)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--rerender", action="store_true",
                    help="redraw figures even if report_figures/ has them")
    ap.add_argument("--min-coverage", type=float, default=0.3,
                    help="parcel tables list Glasser areas at least this "
                         "fraction covered by the family")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = args.root
    summary = json.loads((root / "clusters" / "summary.json").read_text())
    ms = methods(summary["z_thresholds"])
    surfaces = load_surfaces(fsaverage_dir(), "inflated")

    means, vlims, covered = {}, {}, {}
    for stat in STATS:
        m = np.load(root / f"{stat}__mean.npy").astype(float)
        means[stat], covered[stat] = m, np.isfinite(m)
        vlims[stat] = limits(m[np.isfinite(m)], False)

    counts = {}  # (method, stat) -> n significant
    images = {}
    fig_dir = root / "clusters" / "report_figures"
    fig_dir.mkdir(exist_ok=True)
    sigs = {}  # (method, stat) -> boolean map, for the parcel tables
    todo = []
    for m in ms:
        for stat, label in STATS.items():
            p = pmap(m, root, stat)
            if p is None:
                continue
            sig = np.isfinite(p) & (p < m["alpha"]) & covered[stat]
            counts[m["key"], stat] = int(sig.sum())
            sigs[m["key"], stat] = sig
            f = fig_dir / f"{m['key']}__{stat}.png"
            if args.rerender or not f.exists():
                todo.append((stat, np.where(sig, np.nan_to_num(means[stat]),
                                            0.0),
                             vlims[stat], f"{label} | {m['label']}", f))
            images[m["key"], stat] = f
    log.info(f"rendering {len(todo)} figures on {args.jobs} processes")
    _R.update(surfaces=surfaces, dpi=args.dpi)
    if todo:
        with mp.get_context("fork").Pool(args.jobs) as pool:
            pool.map(_render, todo)
    images = {k: img_tag(f) for k, f in images.items()}
    parcels = glasser_parcels(fsaverage_dir())

    n_cov = {s: int(covered[s].sum()) for s in STATS}
    run = root.name
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Integration inference methods</title><style>{CSS}</style>"
             f"</head><body><main>"]
    parts.append(
        f"<h1>Integration inference methods</h1>"
        f"<p class='muted'><code>{html.escape(run)}</code> · nine subjects · "
        f"{summary.get('n_draws', '?')} permutation draws · family: vertices "
        f"reached by ≥{summary['family_min_subjects']} subjects "
        f"({n_cov['delta']:,}) · atlas: Glasser HCP-MMP1</p>")
    nav = ["<a href='#summary'>Summary</a>"] + [
        f"<a href='#{m['key']}'>{html.escape(m['label'])}</a>" for m in ms]
    parts.append("<nav>" + "".join(nav) + "</nav>")

    parts.append("""<div class='note'>
<b>How to read this.</b> Every cell is the same group-mean map, coloured only
where the row's method rejects (α = 0.05 unless stated), on one colour scale
per column. <b>Integration</b> is the intersection–union conjunction: a vertex
counts only if it is significant for audio beyond text <i>and</i> text beyond
audio under the same method, so its p is the larger of the two.
The cluster and TFCE rows use the <b>group permutation null</b>: all nine
subjects share a seed, so each of the 1000 draws is the same shuffle of time
for everyone and the per-draw group map is an exact null. Each vertex is
z-scored against its own null (observed map pooled in, so all 1001 maps are
exchangeable). Cluster FWE compares each cluster with the largest cluster in
every draw. <b>Extent</b> is the cluster's area (mm²). <b>Mass</b> is
Σ area·(z − threshold), which also rewards small, strong clusters. TFCE uses
E = 1, H = 2 with a fixed step of 0.1 z. Cluster-level significance means
"part of a significant region", not that single vertex. The sign-flip row
is random effects but only covers the 3,337 vertices all nine subjects
reach.</div>""")

    # Summary table
    parts.append("<h2 id='summary'>Summary: significant vertices</h2>"
                 "<div class='scroll'><table><thead><tr><th>Method</th>")
    for s, lab in STATS.items():
        parts.append(f"<th class='num'>{lab}</th>")
    parts.append("<th class='num'>Integration clusters</th></tr></thead><tbody>")
    for m in ms:
        parts.append(f"<tr><td><a href='#{m['key']}'>"
                     f"{html.escape(m['label'])}</a></td>")
        for s in STATS:
            c = counts.get((m["key"], s))
            parts.append("<td class='num'>—</td>" if c is None else
                         f"<td class='num'>{c:,} ({c / n_cov[s]:.1%})</td>")
        nclu = summary["methods"].get(m["key"], {}).get(
            "n_integration_clusters")
        parts.append(f"<td class='num'>{'—' if nclu is None else nclu}</td></tr>")
    parts.append("</tbody></table></div><p class='muted'>Percentages are "
                 "of the covered family. The sign-flip row is limited to the "
                 "n = 9 core by construction.</p>")

    group = None
    for m in ms:
        if m["group"] != group:
            group = m["group"]
            parts.append(f"<h2>{html.escape(group)}</h2>")
        parts.append(f"<h3 id='{m['key']}'>{html.escape(m['label'])}</h3>"
                     "<div class='grid'>")
        for s, lab in STATS.items():
            if (m["key"], s) not in images:
                continue
            c = counts[m["key"], s]
            parts.append(f"<div class='cell'>{images[m['key'], s]}"
                         f"<div class='cap'><b>{lab}</b> · {c:,} vertices "
                         f"({c / n_cov[s]:.1%})</div></div>")
        parts.append("</div>")
        parts.append(f"<details{' open' if m['group'] == 'Vertex-level' else ''}>"
                     "<summary>Glasser areas, ranked by % significant for "
                     "integration</summary>")
        parts.append(parcel_table_html(m["key"], sigs, means, covered,
                                       parcels, args.min_coverage,
                                       args.table_rows))
        parts.append("</details>")
        key = m.get("table") or ("tfce" if m["key"] == "tfce" else None)
        if key:
            parts.append("<h3>Integration clusters, ranked by mean delta</h3>")
            parts.append(csv_table(root / "clusters" / f"integration__{key}.csv",
                                   INTEGRATION_COLS, args.table_rows))
        if m.get("table"):
            for s in ("delta_audio_given_text", "delta_text_given_audio"):
                parts.append(f"<details><summary>{STATS[s]}: significant "
                             f"clusters</summary>")
                parts.append(csv_table(
                    root / "clusters" / f"{s}__{m['table']}__clusters.csv",
                    COMPONENT_COLS, args.table_rows))
                parts.append("</details>")

    parts.append("</main></body></html>")
    out = args.out or root / "clusters" / "report.html"
    out.write_text("".join(parts))
    log.info(f"-> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
