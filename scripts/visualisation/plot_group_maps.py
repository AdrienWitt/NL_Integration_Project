#!/usr/bin/env python3
"""
Brain figures for the group maps written by `scripts/group_maps.py`.

    results/group/<run>/<stat>__mean.npy  --pycortex quickflat--> figures/<stat>.png

One figure per statistic: the group mean effect on an fsaverage flatmap, masked
to the vertices that survive a chosen p-map, over a curvature background with
ROI outlines.

Colour follows the job the map does, not taste:

- **Magnitude** (`r_text`, `r_audio`, `r_joint`, every `delta`) is one hue,
  light to dark. These maps are non-negative by construction, so a diverging
  scale would spend half its range on values that cannot occur and imply a
  meaningful midpoint where there is none.
- **Polarity** (`preference = r_text - r_audio`) is two hues with a **neutral
  grey** midpoint, symmetric about zero, so that "neither band is ahead" reads
  as absence of colour rather than as a third category. The limits are forced
  symmetric: an asymmetric diverging scale silently makes one modality look
  stronger than it is.
- Never a rainbow, and never a hue at the diverging midpoint.

Unthresholded maps are written alongside, because the threshold is a decision a
reader should be able to audit and a figure that only ever appears masked hides
where the effect was merely subthreshold.

Usage
-----
    python scripts/visualisation/plot_group_maps.py --root results/group/<run>
    python scripts/visualisation/plot_group_maps.py --root results/group/<run> --test none
    python scripts/visualisation/plot_group_maps.py --root results/group/<run> --stats delta \\
        --test p_signflip_fwe
    python scripts/visualisation/plot_group_maps.py --root results/group/<run> \\
        --test p_tfce_fwe        # after `group_maps.py --tfce`
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

log = logging.getLogger("plot_group_maps")

_HERE = Path(__file__).resolve().parent

#: One hue, light -> dark -- but the pale end of a full Reds/Blues ramp is the
#: same lightness as the curvature background these maps sit on, so the weakest
#: suprathreshold vertices read as "no data" rather than "small effect". Both
#: ramps are therefore truncated to start at a visibly tinted step. This keeps
#: the single-hue ordering intact and only removes the range that the
#: background already occupies.
def _truncate(name: str, start: float, new_name: str):
    base = matplotlib.colormaps[name]
    cmap = LinearSegmentedColormap.from_list(
        new_name, base(np.linspace(start, 1.0, 256)))
    try:
        matplotlib.colormaps.register(cmap)
    except (ValueError, AttributeError):
        pass
    return new_name


SEQUENTIAL = _truncate("Reds", 0.22, "mag_red")
COVERAGE = _truncate("Blues", 0.22, "cov_blue")

#: Two hues, neutral grey midpoint. Built explicitly rather than taken from
#: RdBu/coolwarm, whose midpoints are near-white and vanish into the curvature
#: background that every one of these figures sits on.
DIVERGING = LinearSegmentedColormap.from_list(
    "text_vs_audio", ["#1A5D99", "#9A9A9A", "#B2182B"])
try:
    matplotlib.colormaps.register(DIVERGING)
except (ValueError, AttributeError):
    pass

#: Statistics whose values are signed; everything else is a magnitude.
SIGNED = ("preference",)

#: Default p-map per statistic. `preference` has no permutation of its own and
#: its sign test is blocked by the 1/512 floor, so its default is the sign-flip
#: FWE -- see the preference section in CLAUDE.md.
DEFAULT_TEST = {"preference": "p_signflip_fwe"}
FALLBACK_TEST = "p_perm_fdr"

LABELS = {
    "text": "Semantic band (GPT-2 k256 L8), r",
    "audio": "Prosodic band (emotion L11), r",
    "joint": "Joint model, r",
    "preference": "preference = r(text) - r(audio)",
    "delta": "Integration: delta = min of both conditional contributions",
    "delta_audio_given_text": "Audio beyond text, delta r",
    "delta_text_given_audio": "Text beyond audio, delta r",
    "arousal": "Arousal band, r",
    "dominance": "Dominance band, r",
    "valence": "Valence band, r",
    "delta_arousal_given_dominance+valence":
        "Arousal beyond dominance+valence, delta r",
    "delta_dominance_given_arousal+valence":
        "Dominance beyond arousal+valence, delta r",
    "delta_valence_given_arousal+dominance":
        "Valence beyond arousal+dominance, delta r",
}


def discover(root: Path, stats: list[str] | None) -> list[str]:
    names = sorted({p.name.split("__")[0] for p in root.glob("*__mean.npy")})
    if stats:
        missing = [s for s in stats if s not in names]
        if missing:
            raise SystemExit(f"not in {root}: {', '.join(missing)}")
        names = [s for s in stats if s in names]
    return names


def limits(values: np.ndarray, signed: bool) -> tuple[float, float]:
    """Robust limits; symmetric about zero when the map is signed."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    if signed:
        hi = float(np.percentile(np.abs(finite), 99)) or 1.0
        return -hi, hi
    lo = 0.0
    hi = float(np.percentile(finite, 99))
    return lo, (hi if hi > lo else float(finite.max()) or 1.0)


def make_one(cortex, root: Path, stat: str, test: str, alpha: float,
             out_dir: Path, dpi: int, with_rois: bool) -> None:
    mean = np.load(root / f"{stat}__mean.npy").astype(float)
    signed = stat in SIGNED
    cmap = "text_vs_audio" if signed else SEQUENTIAL

    variants = [("unthresholded", mean, None)]
    if test != "none":
        p_path = root / f"{stat}__{test}.npy"
        if p_path.exists():
            p = np.load(p_path)
            masked = np.where(np.isfinite(p) & (p < alpha), mean, np.nan)
            variants.append((test, masked, int(np.isfinite(masked).sum())))
        else:
            log.warning(f"  {stat}: no {test}.npy, unthresholded only")

    vmin, vmax = limits(mean, signed)
    for suffix, data, n_sig in variants:
        vx = cortex.Vertex(data, "fsaverage", vmin=vmin, vmax=vmax, cmap=cmap)
        fig = cortex.quickflat.make_figure(
            vx, with_rois=with_rois, with_curvature=True, with_colorbar=True,
            with_labels=with_rois)
        title = LABELS.get(stat, stat)
        if n_sig is not None:
            title += f"  |  {test} < {alpha}: {n_sig:,} vertices"
        fig.suptitle(title, fontsize=11, y=0.99)
        out = out_dir / f"{stat}__{suffix}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        log.info(f"  -> {out.name}")


def make_coverage(cortex, root: Path, out_dir: Path, dpi: int,
                  with_rois: bool) -> None:
    """How many subjects reach each vertex - the map that bounds every other."""
    candidates = sorted(root.glob("*__n_subjects.npy"))
    if not candidates:
        return
    n = np.load(candidates[0]).astype(float)
    n[n < 1] = np.nan
    vx = cortex.Vertex(n, "fsaverage", vmin=1, vmax=9, cmap=COVERAGE)
    fig = cortex.quickflat.make_figure(
        vx, with_rois=with_rois, with_curvature=True, with_colorbar=True,
        with_labels=with_rois)
    fig.suptitle("Subjects reaching each vertex (1-9)", fontsize=11, y=0.99)
    out = out_dir / "coverage__n_subjects.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  -> {out.name}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="a results/group/<run> directory")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <root>/figures")
    ap.add_argument("--stats", nargs="*", default=None)
    ap.add_argument("--test", default="auto",
                    help="p-map to threshold with, or 'none'. 'auto' uses "
                         "p_perm_fdr, and p_signflip_fwe for preference.")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--no-rois", action="store_true",
                    help="drop ROI outlines and labels")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if "FREESURFER_HOME" in os.environ:
        fs_bin = Path(os.environ["FREESURFER_HOME"]) / "bin"
        os.environ["PATH"] = f"{os.environ['PATH']}:{fs_bin}"

    # Reuse project_to_fsaverage's setup rather than reimplementing it: pycortex
    # caches its subject list on the first `cortex.db` lookup, so the filestore
    # has to be set *and the cache cleared* before anything touches the db.
    # Assigning `default_filestore` afterwards silently keeps the old store and
    # fails later with KeyError: 'fsaverage'.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "project_to_fsaverage", _HERE.parent / "project_to_fsaverage.py")
    p2f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p2f)
    cortex = p2f.setup_pycortex()

    db = Path(p2f.PYCORTEX_DB)
    if not (db / "fsaverage" / "surfaces" / "flat_lh.gii").exists():
        raise SystemExit(
            f"no fsaverage flat surface under {db}. quickflat needs one; "
            f"the LeBel pycortex database ships it.")

    # pycortex rasterises the ROI overlay by shelling out to Inkscape, and
    # raises from deep inside quickflat when it is absent. Degrade to a figure
    # without ROI outlines rather than producing nothing.
    with_rois = not args.no_rois
    if with_rois and shutil.which("inkscape") is None:
        log.warning("inkscape not on PATH -- drawing without ROI outlines. "
                    "`apt-get install inkscape` restores them.")
        with_rois = False

    names = discover(args.root, args.stats)
    if not names:
        raise SystemExit(f"no *__mean.npy under {args.root}")
    out_dir = args.out or (args.root / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"{len(names)} statistics -> {out_dir}")

    for stat in names:
        test = (DEFAULT_TEST.get(stat, FALLBACK_TEST)
                if args.test == "auto" else args.test)
        log.info(f"{stat} (threshold: {test})")
        make_one(cortex, args.root, stat, test, args.alpha, out_dir,
                 args.dpi, with_rois)
    make_coverage(cortex, args.root, out_dir, args.dpi, with_rois)
    log.info(f"\n-> {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
