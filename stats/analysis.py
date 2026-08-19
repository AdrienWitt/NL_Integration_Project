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
from typing import Dict, List

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


def load_subject(subject_dir: Path) -> Dict[str, np.ndarray]:
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

    meta_path = subject_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            out["meta"] = json.load(f)
    return out


# --------------------------------------------------------------------------
# Contrasts
# --------------------------------------------------------------------------

def compute_contrasts(maps: Dict[str, np.ndarray], min_r: float = 0.05,
                      normalize: bool = False) -> Dict[str, np.ndarray]:
    """Derive delta, preference and split fractions from one subject's maps."""
    missing = [m for m in MODELS if m not in maps]
    if missing:
        raise KeyError(f"Missing model maps: {missing}")

    r_text, r_audio, r_joint = maps["text"], maps["audio"], maps["joint"]

    if normalize:
        if "ev" not in maps:
            raise KeyError("normalize=True needs an explainable-variance map")
        ev = maps["ev"]
        r_text = normalize_by_ceiling(r_text, ev)
        r_audio = normalize_by_ceiling(r_audio, ev)
        r_joint = normalize_by_ceiling(r_joint, ev)

    best_unimodal = np.maximum(r_text, r_audio)
    delta = r_joint - best_unimodal

    # Preference is only interpretable where something is predicted at all.
    predicted = np.maximum(best_unimodal, r_joint) > min_r
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


def winner_map(contrasts: Dict[str, np.ndarray], delta_min: float = 0.0
               ) -> np.ndarray:
    """Label each voxel 0=none, 1=semantic, 2=prosodic, 3=integrative."""
    labels = np.zeros(contrasts["r_text"].shape, dtype=np.int8)
    predicted = contrasts["predicted_mask"]

    semantic = predicted & (contrasts["r_text"] >= contrasts["r_audio"])
    prosodic = predicted & (contrasts["r_audio"] > contrasts["r_text"])
    labels[semantic] = 1
    labels[prosodic] = 2

    integrative = predicted & (contrasts["delta"] > delta_min)
    labels[integrative] = 3
    return labels


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------

def subject_row(subject: str, contrasts: Dict[str, np.ndarray],
                maps: Dict[str, np.ndarray], min_r: float) -> dict:
    predicted = contrasts["predicted_mask"]
    n_pred = int(predicted.sum())
    labels = winner_map(contrasts)

    def _mean(key, mask=None):
        values = contrasts[key]
        values = values[mask] if mask is not None else values
        values = values[np.isfinite(values)]
        return float(values.mean()) if values.size else float("nan")

    row = {
        "subject": subject,
        "n_voxels": int(contrasts["r_text"].size),
        f"n_predicted(r>{min_r})": n_pred,
        "mean_r_text": _mean("r_text", predicted),
        "mean_r_audio": _mean("r_audio", predicted),
        "mean_r_joint": _mean("r_joint", predicted),
        "mean_delta": _mean("delta", predicted),
        "max_delta": float(np.nanmax(contrasts["delta"])),
        "n_semantic": int((labels == 1).sum()),
        "n_prosodic": int((labels == 2).sum()),
        "n_integrative": int((labels == 3).sum()),
    }
    if "split_frac_text" in contrasts:
        row["mean_split_frac_text"] = _mean("split_frac_text", predicted)
        row["mean_split_frac_audio"] = _mean("split_frac_audio", predicted)
    if "ev" in maps:
        row["mean_ev"] = float(np.nanmean(maps["ev"]))
    return row


def group_summary(results_dir: Path, min_r: float = 0.05,
                  normalize: bool = False, save: bool = True
                  ) -> pd.DataFrame:
    """Per-subject contrast table for one results directory."""
    results_dir = Path(results_dir)
    dirs = subject_dirs(results_dir)
    if not dirs:
        raise FileNotFoundError(f"No subject results under {results_dir}")

    rows, per_subject = [], {}
    for subject_dir in dirs:
        subject = subject_dir.name
        maps = load_subject(subject_dir)
        try:
            contrasts = compute_contrasts(maps, min_r=min_r, normalize=normalize)
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

    table = pd.DataFrame(rows).set_index("subject")

    if per_subject:
        stacked = {
            key: np.nanmean(
                np.stack([c[key] for c in per_subject.values()]), axis=0
            )
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
                   help="divide r by the noise ceiling sqrt(EV) first")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    results_dir = Path(args.results_dir) if args.results_dir else (
        Path(ENCODING_OUT) / args.features / args.backend / args.eval
    )
    log.info(f"Reading {results_dir}")

    table = group_summary(results_dir, min_r=args.min_r, normalize=args.normalize)

    ensure_dirs(STATS_OUT)
    out_csv = Path(STATS_OUT) / f"summary_{args.features}_{args.backend}_{args.eval}.csv"
    table.to_csv(out_csv)

    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print()
        print(table.round(4).to_string())
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
