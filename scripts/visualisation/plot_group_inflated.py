#!/usr/bin/env python3
"""
Anatomically readable group figures: fsaverage inflated views + a parcel table.

    results/group/<run>/<stat>__mean.npy  --nilearn--> figures_inflated/<stat>.png
                                          --aparc---> tables/<stat>__parcels.csv

`plot_group_maps.py` renders the same maps as pycortex flatmaps. A flatmap
shows every vertex at once, which is what you want for extent, and it shows no
recognisable anatomy, which is what you do not want in a talk. This script is
the complement, not a replacement: the *same* group maps and the *same* p-maps,
drawn on the inflated surface in conventional lateral/medial views, plus a
table naming the regions.

Nothing is re-projected and nothing is re-tested. The maps are already on
fsaverage (`project_to_fsaverage.py`) and already carry per-vertex p-values
(`group_maps.py`); this only draws them.

Why not the pycortex ROI outlines instead: the fsaverage `overlays.svg` in this
database is the Huth-lab retinotopy set (V1-V4, hMT, FFA, PPA, IPS0-5, ...)
plus a handful of sulci. For an auditory/language result those outlines name
almost nothing that matters, and they need Inkscape on PATH besides. The
FreeSurfer parcellations shipped with fsaverage do name it.

Usage
-----
    python scripts/visualisation/plot_group_inflated.py --root results/group/<run>
    python scripts/visualisation/plot_group_inflated.py --root results/group/<run> \\
        --stats delta preference --atlas a2009s
    python scripts/visualisation/plot_group_inflated.py --root results/group/<run> --test none

`--atlas aparc` is Desikan-Killiany (34 regions/hemi, coarse and legible),
`a2009s` is Destrieux (75, gyri and sulci separately), `Yeo2011_7Networks_N1000`
the Yeo networks. Any `<hemi>.<name>.annot` under fsaverage/label works.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nibabel.freesurfer.io as fsio
from nilearn import plotting as nlp

_HERE = Path(__file__).resolve().parent
#: scripts/visualisation/ -> the project root is two levels up; `_HERE` itself
#: carries plot_group_maps.py, which these scripts import to stay in step with
#: its colormaps, labels and default tests.
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE))

import config  # noqa: E402
from plot_group_maps import (  # noqa: E402
    DEFAULT_TEST, DIVERGING, FALLBACK_TEST, LABELS, SEQUENTIAL, SIGNED,
    discover, limits, p_path,
)

log = logging.getLogger("plot_group_inflated")

#: fsaverage has 163,842 vertices per hemisphere and the group maps are
#: lh-then-rh, as `project_to_fsaverage.Projector.to_fsaverage` concatenates
#: them. Asserted against the surface rather than hard-coded.
HEMIS = ("lh", "rh")
VIEWS = ("lateral", "medial")
NILEARN_HEMI = {"lh": "left", "rh": "right"}


def fsaverage_dir() -> Path:
    """The project's own fsaverage, falling back to the FreeSurfer install."""
    local = Path(config.FREESURFER_DIR) / "fsaverage"
    if (local / "surf" / "lh.inflated").exists():
        return local
    home = Path(config.__dict__.get("FREESURFER_HOME", "")
                or "/usr/local/freesurfer/7.4.1")
    fallback = home / "subjects" / "fsaverage"
    if (fallback / "surf" / "lh.inflated").exists():
        return fallback
    raise SystemExit(
        f"no fsaverage surfaces under {local} or {fallback}. Set FREESURFER_DIR.")


def load_surfaces(fsdir: Path, surf: str) -> dict:
    """Geometry and a sulcal-depth background per hemisphere."""
    out = {}
    for hemi in HEMIS:
        coords, faces = fsio.read_geometry(fsdir / "surf" / f"{hemi}.{surf}")
        sulc = fsio.read_morph_data(fsdir / "surf" / f"{hemi}.sulc")
        out[hemi] = {"mesh": (coords, faces), "bg": sulc, "n": coords.shape[0]}
    return out


def split_hemis(values: np.ndarray, surfaces: dict) -> dict:
    n_lh = surfaces["lh"]["n"]
    expected = n_lh + surfaces["rh"]["n"]
    if values.shape[0] != expected:
        raise SystemExit(
            f"map has {values.shape[0]} vertices, fsaverage has {expected}")
    return {"lh": values[:n_lh], "rh": values[n_lh:]}


def masked_map(root: Path, stat: str, test: str, alpha: float):
    """The group mean, zeroed where it does not survive `test`.

    Zero rather than NaN because nilearn's `threshold` makes sub-threshold
    vertices transparent and NaN renders as a hole. Vertices below
    `--min-subjects` are already NaN in the saved mean and stay masked.
    """
    mean = np.load(root / f"{stat}__mean.npy").astype(float)
    covered = np.isfinite(mean)
    shown = np.where(covered, mean, 0.0)
    if test == "none":
        return shown, covered, np.zeros(0, dtype=bool)
    pmap = p_path(root, stat, test)
    if not pmap.exists():
        raise SystemExit(f"{pmap.name} not in {root}")
    p = np.load(pmap)
    sig = np.isfinite(p) & (p < alpha) & covered
    return np.where(sig, shown, 0.0), covered, sig


def render(stat: str, values: np.ndarray, surfaces: dict, signed: bool,
           vlim: tuple[float, float], title: str, out: Path, dpi: int,
           views: tuple[str, ...]) -> None:
    per_hemi = split_hemis(values, surfaces)
    cmap = "text_vs_audio" if signed else SEQUENTIAL
    # A magnitude map starts at 0 and a signed one is symmetric; `limits`
    # already enforces that, so vlim is used as given.
    vmin, vmax = vlim
    eps = max(abs(vmax), abs(vmin)) * 1e-6 or 1e-9

    fig, axes = plt.subplots(
        len(views), 2, figsize=(9.5, 3.4 * len(views)),
        subplot_kw={"projection": "3d"})
    axes = np.atleast_2d(axes)
    for row, view in enumerate(views):
        for col, hemi in enumerate(HEMIS):
            nlp.plot_surf(
                surfaces[hemi]["mesh"], per_hemi[hemi],
                bg_map=surfaces[hemi]["bg"], hemi=NILEARN_HEMI[hemi],
                view=view, cmap=cmap, threshold=eps, vmin=vmin, vmax=vmax,
                bg_on_data=False, darkness=0.55, colorbar=False,
                axes=axes[row, col], figure=fig)
            # A 3-d axes leaves a wide empty margin around a surface; zoom
            # fills it so four panels read as one figure rather than four
            # stamps. Purely a camera setting -- no data is cropped.
            axes[row, col].set_box_aspect(None, zoom=1.35)
            axes[row, col].set_title(f"{hemi} {view}", fontsize=9, y=0.94)

    # One colorbar for the figure, from a throwaway mappable, so the panels
    # keep their full width instead of each carrying its own bar.
    sm = plt.cm.ScalarMappable(
        cmap=matplotlib.colormaps[cmap],
        norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
    cax = fig.add_axes([0.32, 0.055, 0.36, 0.016])
    fig.colorbar(sm, cax=cax, orientation="horizontal")
    cax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=11, y=0.985)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.09,
                        wspace=-0.08, hspace=-0.12)
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)
    log.info("wrote %s", out)


def parcel_table(stat: str, root: Path, surfaces: dict, fsdir: Path,
                 atlas: str, test: str, alpha: float, min_vertices: int,
                 out: Path) -> list[dict]:
    """Mean effect and significant extent per parcel, over covered vertices.

    Covered means the vertex survived `--min-subjects` at group time, so a
    parcel's mean is over the cortex this dataset actually reaches, and
    `n_covered / n_parcel` says how much of it that was.
    """
    mean = np.load(root / f"{stat}__mean.npy").astype(float)
    covered = np.isfinite(mean)
    if test == "none":
        sig = np.zeros_like(covered)
    else:
        p = np.load(p_path(root, stat, test))
        sig = np.isfinite(p) & (p < alpha) & covered

    rows = []
    offset = 0
    for hemi in HEMIS:
        n = surfaces[hemi]["n"]
        labels, _, names = fsio.read_annot(
            fsdir / "label" / f"{hemi}.{atlas}.annot")
        sl = slice(offset, offset + n)
        m, cov, sg = mean[sl], covered[sl], sig[sl]
        for idx, raw in enumerate(names):
            name = raw.decode() if isinstance(raw, bytes) else str(raw)
            if name in ("unknown", "Unknown", "???", "corpuscallosum",
                        "Medial_wall", "Background+FreeSurfer_Defined_Medial_Wall"):
                continue
            in_parcel = labels == idx
            n_cov = int((in_parcel & cov).sum())
            if n_cov < min_vertices:
                continue
            rows.append({
                "hemi": hemi,
                "parcel": name,
                "n_vertices": int(in_parcel.sum()),
                "n_covered": n_cov,
                "pct_covered": round(100 * n_cov / int(in_parcel.sum()), 1),
                "mean": round(float(np.nanmean(m[in_parcel & cov])), 5),
                "n_significant": int((in_parcel & sg).sum()),
                "pct_significant": round(
                    100 * int((in_parcel & sg).sum()) / n_cov, 1),
            })
        offset += n

    rows.sort(key=lambda r: r["mean"], reverse=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s (%d parcels)", out, len(rows))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="a results/group/<run> directory")
    ap.add_argument("--stats", nargs="*", default=None,
                    help="statistics to draw (default: all in --root)")
    ap.add_argument("--test", default=None,
                    help="p-map to threshold with, or 'none'. Default: "
                         "p_signflip_fwe for preference, p_perm_fdr otherwise")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--atlas", default="aparc",
                    help="fsaverage annot: aparc | a2009s | "
                         "Yeo2011_7Networks_N1000 | ...")
    ap.add_argument("--surf", default="inflated",
                    help="fsaverage surface: inflated | pial | white")
    ap.add_argument("--views", nargs="*", default=list(VIEWS))
    ap.add_argument("--min-parcel-vertices", type=int, default=100,
                    help="drop parcels with fewer covered vertices than this")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where the figures go (default: <root>/figures_inflated)")
    ap.add_argument("--table-dir", type=Path, default=None,
                    help="where the parcel CSVs go (default: <root>/tables)")
    ap.add_argument("--no-table", action="store_true")
    ap.add_argument("--top", type=int, default=10,
                    help="parcels to print per statistic")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f"no such directory: {root}")

    fsdir = fsaverage_dir()
    log.info("fsaverage: %s", fsdir)
    surfaces = load_surfaces(fsdir, args.surf)
    out_dir = args.out_dir or (root / "figures_inflated")
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir = args.table_dir or (root / "tables")
    if not args.no_table:
        table_dir.mkdir(parents=True, exist_ok=True)

    for stat in discover(root, args.stats):
        test = args.test or DEFAULT_TEST.get(stat, FALLBACK_TEST)
        if test != "none" and not p_path(root, stat, test).exists():
            log.warning("%s: no %s, drawing unthresholded", stat, test)
            test = "none"
        values, covered, sig = masked_map(root, stat, test, args.alpha)
        signed = stat in SIGNED
        vlim = limits(np.load(root / f"{stat}__mean.npy").astype(float), signed)
        label = LABELS.get(stat, stat)
        suffix = "unthresholded" if test == "none" else test
        title = (f"{label}  |  {suffix}"
                 + ("" if test == "none"
                    else f" < {args.alpha}: {int(sig.sum()):,} vertices"))
        render(stat, values, surfaces, signed, vlim, title,
               out_dir / f"{stat}__{suffix}__inflated.png", args.dpi,
               tuple(args.views))

        if args.no_table:
            continue
        rows = parcel_table(stat, root, surfaces, fsdir, args.atlas, test,
                            args.alpha, args.min_parcel_vertices,
                            table_dir / f"{stat}__parcels_{args.atlas}.csv")
        log.info("  top %d parcels by mean (%s):", args.top, stat)
        for r in rows[:args.top]:
            log.info("    %-4s %-28s mean %+.4f  %5.1f%% sig  n=%d",
                     r["hemi"], r["parcel"], r["mean"], r["pct_significant"],
                     r["n_covered"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
