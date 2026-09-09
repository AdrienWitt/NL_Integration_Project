"""
Turning per-subject r maps into the contrasts the project is about.

Contrasts
---------
delta      = r_joint - max(r_text, r_audio)
    Integration. Positive only where using both modalities beats the better
    single one. Under banded ridge this is >= 0 up to CV noise, so read it
    together with the permutation test, never on its own.

preference = r_text - r_audio
    Which modality drives a voxel. Positive = semantics, negative = prosody.
    Only meaningful where at least one model actually predicts the voxel, so
    it is masked by `min_r`; elsewhere it is the difference of two noise
    estimates and will look like structured nonsense on a brain map.

split_frac = split_r_band / sum(split_r)
    Banded ridge only: each band's share of the joint model's prediction.
    This is the "normalised variance contribution" readout. It answers a
    different question from delta — how the joint model divides its work,
    rather than whether joining helped — so the two can disagree, and both
    are worth reporting.

All maps can additionally be divided by the noise ceiling sqrt(EV), which puts
subjects with different data quality on a comparable scale.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import ENCODING_OUT, STATS_OUT, ensure_dirs
from encoding.cv import normalize_by_ceiling

log = logging.getLogger("analysis")

MODELS = ["text", "audio", "joint"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def subject_dirs(results_dir: Path) -> List[Path]:
    return sorted(d for d in Path(results_dir).iterdir()
                  if d.is_dir() and (d / "joint_corrs.npy").exists())


def load_subject(subject_dir: Path,
                 permutation_dir: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """Load every saved map for one subject."""
    out: Dict[str, np.ndarray] = {}
    for model in MODELS:
        path = subject_dir / f"{model}_corrs.npy"
        if path.exists():
            out[model] = np.load(path)

    for extra in ["joint_split_corrs", "joint_deltas", "ev", "voxel_mask"]:
        path = subject_dir / f"{extra}.npy"
        if path.exists():
            out[extra] = np.load(path)

    # The conjunction test's verdict, if it has been run. It lives under the
    # permutation output rather than beside the fits, so it has to be pointed
    # at; without it `integration_map` can only fall back to an unthresholded
    # delta > 0, which is not a test.
    if permutation_dir is not None:
        sig = Path(permutation_dir) / subject_dir.name / "delta_significant.npy"
        if sig.exists():
            out["delta_significant"] = np.load(sig)
        else:
            log.warning(f"{subject_dir.name}: no delta_significant.npy under "
                        f"{permutation_dir}")

    meta_path = subject_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            out["meta"] = json.load(f)
    return out


# --------------------------------------------------------------------------
# Contrasts
# --------------------------------------------------------------------------

def compute_contrasts(maps: Dict[str, np.ndarray], min_r: float = 0.05,
                      normalize: bool = False,
                      selection: str = "ev") -> Dict[str, np.ndarray]:
    """Derive delta, preference and split fractions from one subject's maps.

    `selection` picks the voxels every mean and every count below is taken
    over, and the choice is a statistical one, not a convenience:

    ``"ev"`` (default)
        the saved `voxel_mask` -- explainable variance from the repeats of the
        held-out story. It is computed from the RESPONSE alone and never sees
        a model, so selecting on it cannot bias a comparison between models.

    ``"min_r"`` (legacy)
        `max(r_text, r_audio, r_joint) > min_r`, which selects on the very
        maps the contrasts are then computed from. That is double dipping:
        conditioning on the maximum keeps voxels where the winning band's
        noise happened to be positive, so within the selected set the winner's
        r is biased up relative to the loser's and |preference| is inflated --
        and the semantic-vs-prosodic counts tilt toward whichever band has the
        larger sampling variance. `min_r=0.05` is also below the per-voxel SE
        on a 291-TR story, so the gate is mostly noise. Available only to
        reproduce older numbers; do not report from it.
    """
    missing = [m for m in MODELS if m not in maps]
    if missing:
        raise KeyError(f"Missing model maps: {missing}")

    r_text, r_audio, r_joint = maps["text"], maps["audio"], maps["joint"]

    if normalize:
        if "ev" not in maps:
            raise KeyError("normalize=True needs an explainable-variance map")
        ev = maps["ev"]
        # The scores were computed against the MEAN of n repeats, so the
        # ceiling is the mean's, not a single presentation's. Without
        # n_repeats the ceiling is ~2x too small and identical for the
        # 10-repeat and 5-repeat subjects, i.e. it leaves in place exactly the
        # imbalance normalising is supposed to remove.
        n_repeats = (maps.get("meta") or {}).get("n_repeats")
        if n_repeats is None:
            log.warning(
                "normalize=True but meta.json has no 'n_repeats': falling back "
                "to the single-presentation ceiling, which understates the "
                "ceiling and does NOT put 10-repeat and 5-repeat subjects on "
                "a comparable scale. Re-run the fit to record it.")
        r_text = normalize_by_ceiling(r_text, ev, n_repeats=n_repeats)
        r_audio = normalize_by_ceiling(r_audio, ev, n_repeats=n_repeats)
        r_joint = normalize_by_ceiling(r_joint, ev, n_repeats=n_repeats)

    best_unimodal = np.maximum(r_text, r_audio)
    delta = r_joint - best_unimodal

    if selection == "ev":
        if "voxel_mask" not in maps:
            raise KeyError(
                "selection='ev' needs voxel_mask.npy, which run_encoding saves "
                "whenever --min-ev > 0. Re-run with --min-ev, or pass "
                "selection='min_r' and accept the double dipping it documents.")
        predicted = np.asarray(maps["voxel_mask"], dtype=bool)
    elif selection == "min_r":
        log.warning("selection='min_r' selects voxels using the same maps the "
                    "contrasts are computed from; |preference| and the winner "
                    "counts are biased. Reporting from this is not defensible.")
        predicted = np.maximum(best_unimodal, r_joint) > min_r
    else:
        raise ValueError(f"selection must be 'ev' or 'min_r', got {selection!r}")

    preference = np.where(predicted, r_text - r_audio, np.nan)

    out = {
        "r_text": r_text,
        "r_audio": r_audio,
        "r_joint": r_joint,
        "delta": delta,
        "preference": preference,
        "predicted_mask": predicted,
    }

    if "joint_split_corrs" in maps:
        split = maps["joint_split_corrs"]          # (n_bands, n_voxels)
        # Shares only make sense where the split scores are positive and add
        # up to something; negative split scores mean a band actively hurt.
        positive = np.clip(split, 0, None)
        total = positive.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(total > 0, positive / total, np.nan)
        out["split_text"] = split[0]
        out["split_audio"] = split[1] if split.shape[0] > 1 else np.zeros_like(split[0])
        out["split_frac_text"] = frac[0]
        out["split_frac_audio"] = frac[1] if frac.shape[0] > 1 else np.zeros_like(frac[0])

    return out


def winner_map(contrasts: Dict[str, np.ndarray]) -> np.ndarray:
    """Label each voxel 0=none, 1=semantic, 2=prosodic.

    Preference only. Integration used to be folded in here as a fourth label
    that overwrote the other two wherever `delta > 0`, which mixed two
    orthogonal axes -- *which modality wins* and *whether joining helped* --
    into one map. Since banded ridge makes `delta >= 0` almost by
    construction, that turned `winner.npy` into a near-constant map of 3s and
    left `n_semantic`/`n_prosodic` counting only the voxels where CV noise
    pushed delta below zero. Integration is now `integration_map`.
    """
    labels = np.zeros(contrasts["r_text"].shape, dtype=np.int8)
    predicted = contrasts["predicted_mask"]
    labels[predicted & (contrasts["r_text"] >= contrasts["r_audio"])] = 1
    labels[predicted & (contrasts["r_audio"] > contrasts["r_text"])] = 2
    return labels


def integration_map(contrasts: Dict[str, np.ndarray],
                    significant: Optional[np.ndarray] = None,
                    delta_min: float = 0.0) -> np.ndarray:
    """Where joining the modalities helped. Boolean, over predicted voxels.

    Pass `significant` -- `delta_significant.npy` from
    `stats/run_permutation.py --shuffle-block conditional` -- and that is the
    answer: the conjunction of the two Draper-Stoneman conditional tests,
    FDR-corrected. Without it this falls back to an unthresholded
    `delta > delta_min`, which is the exact call the permutation machinery
    exists to replace: under banded ridge the joint model nests both unimodal
    ones, so `delta > 0` is nearly everywhere and means almost nothing.
    """
    predicted = contrasts["predicted_mask"]
    if significant is not None:
        return predicted & np.asarray(significant, dtype=bool)
    log.warning(
        "integration_map: no delta_significant map given, falling back to an "
        "unthresholded delta > %g. Under banded ridge delta is >= 0 almost by "
        "construction, so this is not a test. Run stats.run_permutation "
        "--shuffle-block conditional and pass its delta_significant.npy.",
        delta_min)
    return predicted & (contrasts["delta"] > delta_min)


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------

def subject_row(subject: str, contrasts: Dict[str, np.ndarray],
                maps: Dict[str, np.ndarray], min_r: float) -> dict:
    predicted = contrasts["predicted_mask"]
    n_pred = int(predicted.sum())
    labels = winner_map(contrasts)
    integrative = integration_map(contrasts, maps.get("delta_significant"))

    def _mean(key, mask=None):
        values = contrasts[key]
        values = values[mask] if mask is not None else values
        values = values[np.isfinite(values)]
        return float(values.mean()) if values.size else float("nan")

    row = {
        "subject": subject,
        "n_voxels": int(contrasts["r_text"].size),
        "n_selected": n_pred,
        "mean_r_text": _mean("r_text", predicted),
        "mean_r_audio": _mean("r_audio", predicted),
        "mean_r_joint": _mean("r_joint", predicted),
        "mean_delta": _mean("delta", predicted),
        "max_delta": float(np.nanmax(contrasts["delta"])),
        "n_semantic": int((labels == 1).sum()),
        "n_prosodic": int((labels == 2).sum()),
        "n_integrative": int(integrative.sum()),
        "integration_tested": "delta_significant" in maps,
    }
    if "split_frac_text" in contrasts:
        row["mean_split_frac_text"] = _mean("split_frac_text", predicted)
        row["mean_split_frac_audio"] = _mean("split_frac_audio", predicted)
    if "ev" in maps:
        row["mean_ev"] = float(np.nanmean(maps["ev"]))
    return row


def group_summary(results_dir: Path, min_r: float = 0.05,
                  normalize: bool = False, save: bool = True,
                  selection: str = "ev",
                  permutation_dir: Optional[Path] = None) -> pd.DataFrame:
    """Per-subject contrast table for one results directory."""
    results_dir = Path(results_dir)
    dirs = subject_dirs(results_dir)
    if not dirs:
        raise FileNotFoundError(f"No subject results under {results_dir}")

    rows, per_subject = [], {}
    for subject_dir in dirs:
        subject = subject_dir.name
        maps = load_subject(subject_dir, permutation_dir=permutation_dir)
        try:
            contrasts = compute_contrasts(maps, min_r=min_r,
                                          normalize=normalize,
                                          selection=selection)
        except KeyError as exc:
            log.warning(f"{subject}: skipped ({exc})")
            continue
        per_subject[subject] = contrasts
        rows.append(subject_row(subject, contrasts, maps, min_r))

        if save:
            out_dir = subject_dir / "contrasts"
            out_dir.mkdir(exist_ok=True)
            for name, arr in contrasts.items():
                np.save(out_dir / f"{name}.npy", arr)
            np.save(out_dir / "winner.npy", winner_map(contrasts))
            np.save(out_dir / "integration.npy",
                    integration_map(contrasts, maps.get("delta_significant")))

    table = pd.DataFrame(rows).set_index("subject")

    # A native-space group mean only means anything if every subject is in the
    # same voxel space, and they are not: the nine subjects run 81,126 to
    # 109,469 voxels. np.stack would raise on a real nine-subject directory and
    # silently write a one-subject "group" map on a per-subject one. And the
    # maps are padded outside the EV mask -- with 0.0 by run_encoding, NaN by
    # the sweeps -- so a plain mean averages the padding in either way.
    # scripts/project_to_fsaverage.py is the supported route: it resamples to a
    # common surface and normalises by the mask's own interpolation weight.
    sizes = {k: v["r_text"].size for k, v in per_subject.items()}
    if len(set(sizes.values())) > 1:
        log.warning(
            "skipping the native-space group mean: subjects are in different "
            "voxel spaces (%s). Use scripts/project_to_fsaverage.py.",
            ", ".join(f"{k}={v:,}" for k, v in sorted(sizes.items())))
        per_subject = {}
    elif len(per_subject) == 1:
        log.warning("only one subject under %s -- not writing a 'group' mean "
                    "for it.", results_dir)
        per_subject = {}

    if per_subject:
        # NaN outside each subject's selection, so the mean is over voxels that
        # were actually fitted rather than over the zero padding around them.
        def _masked(c, key):
            arr = np.asarray(c[key], dtype=np.float64).copy()
            arr[~c["predicted_mask"]] = np.nan
            return arr
        stacked = {
            key: np.nanmean(
                np.stack([_masked(c, key) for c in per_subject.values()]),
                axis=0)
            for key in ["r_text", "r_audio", "r_joint", "delta"]
        }
        if save:
            group_dir = results_dir / "group"
            group_dir.mkdir(exist_ok=True)
            for name, arr in stacked.items():
                np.save(group_dir / f"mean_{name}.npy", arr)
            table.to_csv(group_dir / "subject_summary.csv")
            log.info(f"Group maps and table written to {group_dir}")

    return table


# --------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default=None,
                   help="e.g. results/encoding/gpt2_mean__opensmile/banded/holdout")
    p.add_argument("--features", default="gpt2_mean__opensmile",
                   help="used to build the default results dir")
    p.add_argument("--backend", default="banded", choices=["banded", "huth"])
    p.add_argument("--eval", default="holdout", choices=["holdout", "cv"])
    p.add_argument("--min-r", type=float, default=0.05,
                   help="voxels below this r are treated as unpredicted")
    p.add_argument("--normalize", action="store_true",
                   help="divide r by the noise ceiling first (uses n_repeats "
                        "from meta.json; without it the ceiling is wrong)")
    p.add_argument("--selection", default="ev", choices=["ev", "min_r"],
                   help="voxels to report over. 'ev' = the model-independent "
                        "explainable-variance mask (default). 'min_r' selects "
                        "on the same maps it then reports, which is double "
                        "dipping and is kept only to reproduce old numbers")
    p.add_argument("--permutation-dir", default=None,
                   help="stats/run_permutation output root, so integration can "
                        "use the conjunction test instead of delta > 0")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    results_dir = Path(args.results_dir) if args.results_dir else (
        Path(ENCODING_OUT) / args.features / args.backend / args.eval
    )
    log.info(f"Reading {results_dir}")

    table = group_summary(
        results_dir, min_r=args.min_r, normalize=args.normalize,
        selection=args.selection,
        permutation_dir=Path(args.permutation_dir) if args.permutation_dir
        else None)

    ensure_dirs(STATS_OUT)
    out_csv = Path(STATS_OUT) / f"summary_{args.features}_{args.backend}_{args.eval}.csv"
    table.to_csv(out_csv)

    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print()
        print(table.round(4).to_string())
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
