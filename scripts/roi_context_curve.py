"""
Where integration sits as a function of context length, per anatomical ROI.

The group maps say *how much* integration there is and `plot_group_maps.py`
shows *where*, but neither turns "left IFG is the principal locus once context
is included" into a number that can be compared across context lengths. This
does: one row per (run, statistic, hemisphere, ROI), so the claim becomes a
curve over k rather than a reading of four flatmaps side by side.

Usage
-----
    python scripts/roi_context_curve.py \
        --run k4=results/group/k4L8__base_emotionL11__conditional \
        --run k16=results/group/k16L8__base_emotionL11__conditional \
        --run k256=results/group/k256L8__base_emotionL11__conditional \
        --run k512=results/group/k512L8__base_emotionL11__conditional \
        --out results/group/roi_context_curve.csv

`--run` is repeatable and its label is what lands in the `run` column, so the
ordering of the curve is the order they are passed.

Peak discs (`--peak-disc 6`, the paper's figure)
------------------------------------------------
A region average mixes the integrative part of a gyrus with the rest (the
temporo-parietal junction areas reach delta = +0.03 while the dorsal angular
gyrus is near zero). With `--peak-disc R`, each region is summarised instead
by a geodesic disc of radius R mm on the fsaverage midthickness, centred on
the region's peak of delta AVERAGED over every `--select-run` (all context
lengths of both axes), and restricted to the region and to vertices covered
in all of those runs. Selecting on the average over conditions does not favour
any of them, so the differences between context lengths along a curve stay
unbiased; only the overall level is optimistic. The same disc is used at every
point of both curves. `--discs-out` writes the centres and sizes.

What it does NOT do
-------------------
**It does not correct anything a second time.** `group_maps.py` already
FDR-corrects within each statistic's family, and `run_permutation` before it
corrects within each subject's EV ROI and pads to full brain with p = 1. This
script only *counts* vertices that already cleared a threshold. Re-running BH
over an ROI's vertices would be correcting twice and would be wrong in the
anticonservative direction.

**It does not compare ROIs on significance counts alone.** A count scales with
the size of the ROI and with how many subjects reach it, and coverage is the
binding constraint on these maps — only 3,337 of 327,684 vertices are reached by
all nine subjects. Every row therefore carries `n_roi`, `n_covered` and
`coverage` beside the count, and a row whose coverage is low is a row about
five subjects' periphery, not about the ROI.

Reading the result
------------------
The trap this analysis has to survive is that a longer k makes the *text* band
strictly better (+0.0154 mean r from k16 to k256, 9/9 subjects), while the
conjunction that defines integration is limited by the *audio* term — audio
clears FDR in ~18% of voxels against text's ~75%. So:

  - an integration ROI that GROWS with k is a conservative result, obtained
    against the direction of the bias;
  - one that SHRINKS is confounded, because "context absorbs prosody" and "a
    stronger text band absorbs prosody" predict the same thing.

Which is why the default statistic set is not `delta` alone. Run the two
conditional components beside it: `delta_text_given_audio` rising while
`delta_audio_given_text` falls is the signature of the confound, whereas both
moving together is not.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("roi_context_curve")

FREESURFER_HOME = os.environ.get(
    "FREESURFER_HOME", "/usr/local/freesurfer/7.4.1")
N_VERTS_PER_HEMI = 163842          # fsaverage, ico7

# Desikan-Killiany parcels, grouped into the regions this question is about.
# IFG proper is pars opercularis + pars triangularis; pars orbitalis is kept
# apart rather than folded in, because it is the part most often excluded from
# "IFG" in the irony/pragmatics literature and merging it would quietly change
# what the number means.
ROI_GROUPS: dict[str, tuple[str, ...]] = {
    "IFG":              ("parsopercularis", "parstriangularis"),
    "IFG_orbitalis":    ("parsorbitalis",),
    "IFG_operc":        ("parsopercularis",),
    "IFG_triang":       ("parstriangularis",),
    "STG":              ("superiortemporal",),
    "STS_banks":        ("bankssts",),
    "HeschlTransverse": ("transversetemporal",),
    "MTG":              ("middletemporal",),
    "supramarginal":    ("supramarginal",),
    "inferiorparietal": ("inferiorparietal",),
    "precentral":       ("precentral",),
    "rostralmidfrontal": ("rostralmiddlefrontal",),
}

# The paper's atlas (scripts/atlas/build_psn_atlas.py): HCP-MMP1 areas grouped
# into regions, one parcel per region, so each ROI is its own parcel.
ROI_GROUPS_PSN: dict[str, tuple[str, ...]] = {
    r: (r,) for r in ("early_auditory", "STG", "aSTS", "pSTS", "MTG",
                      "angular", "supramarginal", "precentral", "IFG_oper",
                      "IFG_tri", "IFG_orb")}

DEFAULT_STATS = (
    "delta",
    "delta_audio_given_text",
    "delta_text_given_audio",
)


def load_parcels(annot: str = "aparc") -> dict[str, np.ndarray]:
    """Vertex -> parcel-name masks for both hemispheres, in [lh; rh] order.

    The concatenation order is not a convention this script gets to pick: it
    has to match `project_to_fsaverage.py`, which builds every group map as
    `concatenate([lh, rh])`. Getting it backwards would silently relabel a
    left-IFG result as right.
    """
    import nibabel.freesurfer as fsio

    label_dir = Path(FREESURFER_HOME) / "subjects" / "fsaverage" / "label"
    masks: dict[str, np.ndarray] = {}
    for h_i, hemi in enumerate(("lh", "rh")):
        path = label_dir / f"{hemi}.{annot}.annot"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Set FREESURFER_HOME, or install the "
                f"fsaverage subject.")
        labels, _, names = fsio.read_annot(str(path))
        if labels.shape[0] != N_VERTS_PER_HEMI:
            raise ValueError(
                f"{path} has {labels.shape[0]} vertices, expected "
                f"{N_VERTS_PER_HEMI} (fsaverage ico7). The group maps are on "
                f"fsaverage; a different surface cannot be indexed into them.")
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        offset = h_i * N_VERTS_PER_HEMI
        for idx, name in enumerate(names):
            full = np.zeros(2 * N_VERTS_PER_HEMI, dtype=bool)
            full[offset:offset + N_VERTS_PER_HEMI] = labels == idx
            masks[f"{hemi}.{name}"] = full
    return masks


def roi_mask(parcels: dict[str, np.ndarray], hemi: str,
             parts: tuple[str, ...]) -> np.ndarray:
    out = np.zeros(2 * N_VERTS_PER_HEMI, dtype=bool)
    for part in parts:
        key = f"{hemi}.{part}"
        if key not in parcels:
            raise KeyError(f"{key} is not a parcel in this annotation")
        out |= parcels[key]
    return out


def _mesh_graph():
    """fsaverage midthickness (mean of white and pial) as a sparse edge graph
    in mm, both hemispheres stacked [lh; rh] like the group maps."""
    import nibabel.freesurfer as fsio
    from scipy.sparse import coo_matrix

    surf_dir = Path(FREESURFER_HOME) / "subjects" / "fsaverage" / "surf"
    rows, cols, vals = [], [], []
    for h_i, hemi in enumerate(("lh", "rh")):
        w, faces = fsio.read_geometry(str(surf_dir / f"{hemi}.white"))
        pv, _ = fsio.read_geometry(str(surf_dir / f"{hemi}.pial"))
        xyz = (w + pv) / 2
        off = h_i * N_VERTS_PER_HEMI
        for a, b in ((0, 1), (1, 2), (2, 0)):
            i, j = faces[:, a], faces[:, b]
            d = np.linalg.norm(xyz[i] - xyz[j], axis=1)
            rows += [i + off, j + off]
            cols += [j + off, i + off]
            vals += [d, d]
    n = 2 * N_VERTS_PER_HEMI
    rows, cols, vals = map(np.concatenate, (rows, cols, vals))
    # every edge is listed by both of its faces; a sparse matrix would SUM the
    # duplicates and double every length, so keep one entry per edge
    _, keep = np.unique(rows.astype(np.int64) * n + cols, return_index=True)
    return coo_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, n)).tocsr()


def peak_discs(select_dirs: list[Path], parcels: dict[str, np.ndarray],
               radius: float, stat: str = "delta",
               min_n: int = 100, min_n_sparse: int = 50) -> dict[tuple[str, str], dict]:
    """One disc per (hemi, region): peak of `stat` averaged over the selection
    runs, within the region and the vertices covered in every run."""
    from scipy.sparse.csgraph import dijkstra

    maps = [np.load(d / f"{stat}__mean.npy") for d in select_dirs]
    covered = np.all([np.isfinite(m) for m in maps], axis=0)
    avg = np.where(covered, np.mean([np.nan_to_num(m) for m in maps], axis=0), np.nan)
    graph = _mesh_graph()

    def best_disc(region, allowed, min_n=min_n):
        """Centre = vertex whose disc has the highest mean, among the region's
        top decile and outside `allowed`; a disc keeps at least min_n vertices."""
        idx = np.flatnonzero(region & allowed)
        if not idx.size:
            return None
        cand = idx[avg[idx] >= np.quantile(avg[idx], 0.9)]
        best = None
        for v in cand:
            disc = region & (dijkstra(graph, indices=int(v), limit=radius) <= radius)
            if disc.sum() < min_n:
                continue
            score = float(avg[disc].mean())
            if best is None or score > best[0]:
                best = (score, int(v), disc)
        return best

    # A hot spot that straddles a region border would give both regions their
    # peak at that border, and two curves measuring one spot. Regions are
    # placed in order of their best disc, and a later disc is centred at least
    # two radii from every disc already placed in that hemisphere.
    discs = {}
    for hemi in ("lh", "rh"):
        regions = {roi: roi_mask(parcels, hemi, parts) & covered
                   for roi, parts in ROI_GROUPS.items()}
        free = {roi: best_disc(m, np.ones_like(m)) for roi, m in regions.items()}
        order = sorted((roi for roi in free if free[roi]), key=lambda k: -free[k][0])
        allowed = np.ones(2 * N_VERTS_PER_HEMI, dtype=bool)
        # A region too sparsely covered for a disc of min_n vertices (right
        # pars orbitalis) gets one of min_n_sparse, placed after all the others
        # so that it cannot displace them.
        sparse = [roi for roi in regions if roi not in order]
        for roi in order + sparse:
            best = best_disc(regions[roi], allowed,
                             min_n_sparse if roi in sparse else min_n)
            if best is None:
                log.warning("%s.%s: no disc", hemi, roi)
                continue
            score, peak, disc = best
            allowed &= dijkstra(graph, indices=peak, limit=2 * radius) > 2 * radius
            discs[(hemi, roi)] = {"peak": peak, "mask": disc,
                                  "n": int(disc.sum()), "disc_mean": score}
    return discs


def _load(run_dir: Path, stat: str, suffix: str) -> np.ndarray | None:
    path = run_dir / f"{stat}__{suffix}.npy"
    if not path.exists():
        return None
    return np.load(path)


def summarise_run(run_dir: Path, label: str, stats: tuple[str, ...],
                  parcels: dict[str, np.ndarray],
                  alpha: float, discs: dict | None = None) -> list[dict]:
    rows: list[dict] = []
    for stat in stats:
        mean = _load(run_dir, stat, "mean")
        if mean is None:
            log.warning("[%s] %s__mean.npy missing, skipping", label, stat)
            continue
        n_subj = _load(run_dir, stat, "n_subjects")
        p_perm = _load(run_dir, stat, "p_perm_fdr")
        p_prev = _load(run_dir, stat, "p_prevalence_fdr")
        p_fwe = _load(run_dir, stat, "p_signflip_fwe")
        # TFCE group-permutation FWE, the threshold of every paper figure
        # (written by group_clusters.py under clusters/).
        p_tfce = _load(run_dir / "clusters", stat, "p_tfce_perm_fwe")

        # A vertex is "covered" if the group mean is defined there. That is
        # exactly the --min-subjects family group_maps.py wrote; recomputing it
        # from n_subjects would re-impose a floor this script did not choose.
        covered = np.isfinite(mean)

        for hemi in ("lh", "rh"):
            for roi, parts in ROI_GROUPS.items():
                if discs is not None:
                    if (hemi, roi) not in discs:
                        continue
                    m = discs[(hemi, roi)]["mask"]
                else:
                    m = roi_mask(parcels, hemi, parts)
                in_roi = m & covered
                n_roi = int(m.sum())
                n_cov = int(in_roi.sum())
                row = {
                    "run": label,
                    "stat": stat,
                    "hemi": hemi,
                    "roi": roi,
                    "n_roi": n_roi,
                    "n_covered": n_cov,
                    "coverage": n_cov / n_roi if n_roi else np.nan,
                    "mean_r": float(np.nanmean(mean[in_roi])) if n_cov else np.nan,
                    "median_r": float(np.nanmedian(mean[in_roi])) if n_cov else np.nan,
                }
                if n_subj is not None and n_cov:
                    row["mean_n_subjects"] = float(np.nanmean(n_subj[in_roi]))
                for name, pmap in (("perm", p_perm), ("prevalence", p_prev),
                                   ("fwe", p_fwe), ("tfce", p_tfce)):
                    if pmap is None or not n_cov:
                        row[f"n_sig_{name}"] = np.nan
                        row[f"frac_sig_{name}"] = np.nan
                        continue
                    sig = in_roi & np.isfinite(pmap) & (pmap < alpha)
                    row[f"n_sig_{name}"] = int(sig.sum())
                    row[f"frac_sig_{name}"] = float(sig.sum() / n_cov)
                    if name == "perm":
                        row["mean_r_sig"] = (float(np.nanmean(mean[sig]))
                                             if sig.any() else np.nan)
                rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="Per-ROI summary of group integration maps across runs.")
    p.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                   help="A group-map directory and the label it gets in the "
                        "output. Repeatable; order is preserved.")
    p.add_argument("--stats", nargs="+", default=list(DEFAULT_STATS),
                   help="Statistics to summarise (default: delta and its two "
                        "conditional components).")
    p.add_argument("--annot", default="psn",
                   help="FreeSurfer annotation: psn (the paper's HCP-MMP1 "
                        "regions) or aparc (Desikan-Killiany).")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Threshold applied to the ALREADY-corrected p maps.")
    p.add_argument("--rois", nargs="+", default=None,
                   help="Restrict to these ROI keys (default: all).")
    p.add_argument("--out", default=None, help="Write the tidy table here.")
    p.add_argument("--peak-disc", type=float, default=None, metavar="MM",
                   help="Summarise each region by a geodesic disc of this radius "
                        "at its peak of delta averaged over --select-run.")
    p.add_argument("--select-run", action="append", default=None, metavar="DIR",
                   help="Group-map directories whose delta, averaged, picks the "
                        "peaks (default: the --run directories).")
    p.add_argument("--discs-out", default=None,
                   help="Write the disc centres and sizes (JSON) here.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.annot == "psn":
        ROI_GROUPS.clear()
        ROI_GROUPS.update(ROI_GROUPS_PSN)

    if args.rois:
        unknown = set(args.rois) - set(ROI_GROUPS)
        if unknown:
            raise SystemExit(f"unknown ROI keys: {sorted(unknown)}")
        for key in list(ROI_GROUPS):
            if key not in args.rois:
                ROI_GROUPS.pop(key)

    parcels = load_parcels(args.annot)

    discs = None
    if args.peak_disc:
        sel = [Path(d) for d in (args.select_run or
                                 [r.partition("=")[2] for r in args.run])]
        discs = peak_discs(sel, parcels, args.peak_disc)
        if args.discs_out:
            import json
            Path(args.discs_out).parent.mkdir(parents=True, exist_ok=True)
            json.dump({f"{h}.{r}": {**{k: v for k, v in d.items() if k != "mask"},
                                     "radius_mm": args.peak_disc,
                                     "vertices": np.flatnonzero(d["mask"]).tolist()}
                       for (h, r), d in discs.items()},
                      open(args.discs_out, "w"), indent=1)

    rows: list[dict] = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run wants LABEL=DIR, got {spec!r}")
        label, _, d = spec.partition("=")
        run_dir = Path(d)
        if not run_dir.is_dir():
            raise SystemExit(f"{run_dir} is not a directory")
        rows += summarise_run(run_dir, label, tuple(args.stats), parcels,
                              args.alpha, discs)

    df = pd.DataFrame(rows)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        log.info("wrote %s (%d rows)", args.out, len(df))

    # The console view is the curve itself: ROIs down, runs across, so a
    # left-IFG effect that grows with context is visible as a row that rises.
    for stat in args.stats:
        sub = df[(df["stat"] == stat)]
        if sub.empty:
            continue
        for value, title in (("mean_r", "mean r"),
                             ("frac_sig_perm", "fraction significant (perm FDR)"),
                             ("frac_sig_tfce", "fraction significant (TFCE FWE)")):
            pivot = sub.pivot_table(index=["hemi", "roi"], columns="run",
                                    values=value, sort=False)
            # Preserve the order --run was given in.
            order = [s.partition("=")[0] for s in args.run]
            pivot = pivot[[c for c in order if c in pivot.columns]]
            print(f"\n=== {stat} — {title} ===")
            print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
