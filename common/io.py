"""
Loading stimulus features, fMRI responses and story lists; saving results.

Feature convention
------------------
One HDF5 file per story, named ``<story>.hf5`` (``.h5`` also accepted), holding
a single dataset of shape ``(n_TRs, n_dim)``. `load_features` reads whichever
dataset the file contains, so files written with a key other than "data" still
load.

Response convention
-------------------
``<FMRI_DIR>/<subject>/<story>.hf5`` with dataset ``data`` of shape
``(n_TRs, n_voxels)``. The repeated story additionally carries
``individual_repeats`` of shape ``(n_repeats, n_TRs, n_voxels)``, which is what
`load_response_repeats` returns and what the explainable-variance ceiling needs.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np

from config import FEATURES_DIR, FMRI_DIR

_FEATURE_SUFFIXES = (".hf5", ".h5")


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def _read_single_dataset(path) -> np.ndarray:
    """Read the one array stored in an HDF5 file, whatever its key is."""
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        if not keys:
            raise ValueError(f"{path} contains no datasets")
        if "data" in keys:
            return np.asarray(f["data"])
        if len(keys) > 1:
            raise ValueError(
                f"{path} holds several datasets {keys} and none is named "
                f"'data' — cannot decide which to load"
            )
        return np.asarray(f[keys[0]])


def load_features(feature_name_or_dir, stories: Optional[Sequence[str]] = None
                  ) -> Dict[str, np.ndarray]:
    """Load a feature space as ``{story: (n_TRs, n_dim)}``.

    Parameters
    ----------
    feature_name_or_dir : str or Path
        Either a directory, or a name resolved under `config.FEATURES_DIR`
        (e.g. ``"gpt2_mean"`` -> ``data/features/gpt2_mean``).
    stories : sequence of str, optional
        Load only these stories. Missing ones are reported, not skipped
        silently, because a silently absent story changes the design matrix.
    """
    folder = Path(feature_name_or_dir)
    if not folder.is_dir():
        folder = Path(FEATURES_DIR) / str(feature_name_or_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Feature directory not found: {folder}")

    # A story can appear as both .h5 and .hf5; .hf5 wins to match the rest of
    # the pipeline, and each story is loaded once.
    by_story: Dict[str, Path] = {}
    for suffix in reversed(_FEATURE_SUFFIXES):   # .h5 first, .hf5 overwrites
        for path in sorted(folder.glob(f"*{suffix}")):
            by_story[path.name[: -len(suffix)]] = path

    if stories is not None:
        missing = [s for s in stories if s not in by_story]
        if missing:
            raise FileNotFoundError(
                f"{folder.name}: no feature file for {missing}"
            )
        by_story = {s: by_story[s] for s in stories}

    return {story: _read_single_dataset(path) for story, path in by_story.items()}


def feature_dim(features: Dict[str, np.ndarray]) -> int:
    """Number of columns of a feature dict, checking every story agrees."""
    dims = {arr.shape[1] for arr in features.values()}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent feature dimensions across stories: {dims}")
    return dims.pop()


# --------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------

def load_response(stories: Sequence[str], subject: str,
                  fmri_dir=None) -> np.ndarray:
    """Concatenate ``(n_TRs, n_voxels)`` responses over `stories`, in order."""
    fmri_dir = Path(fmri_dir or FMRI_DIR) / subject
    blocks = []
    for story in stories:
        path = fmri_dir / f"{story}.hf5"
        if not path.exists():
            raise FileNotFoundError(f"No response for {subject}/{story}: {path}")
        with h5py.File(path, "r") as f:
            if "data" not in f:
                raise KeyError(
                    f"{path} has no 'data' dataset (keys: {list(f.keys())}). "
                    f"For the repeated story use load_response_repeats()."
                )
            blocks.append(np.asarray(f["data"]))
    return np.vstack(blocks)


def load_response_repeats(story: str, subject: str, fmri_dir=None) -> np.ndarray:
    """Return ``(n_repeats, n_TRs, n_voxels)`` for a story presented repeatedly.

    Only the held-out story has repeats; they are what makes the explainable-
    variance noise ceiling computable.
    """
    path = Path(fmri_dir or FMRI_DIR) / subject / f"{story}.hf5"
    if not path.exists():
        raise FileNotFoundError(f"No response for {subject}/{story}: {path}")
    with h5py.File(path, "r") as f:
        if "individual_repeats" not in f:
            raise KeyError(
                f"{path} has no 'individual_repeats' dataset "
                f"(keys: {list(f.keys())}) — this story was not repeated."
            )
        return np.asarray(f["individual_repeats"])


def subject_has_story(subject: str, story: str, fmri_dir=None) -> bool:
    return (Path(fmri_dir or FMRI_DIR) / subject / f"{story}.hf5").exists()


# --------------------------------------------------------------------------
# Story lists
# --------------------------------------------------------------------------

def load_story_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stories_for_subject(subject: str, json_path) -> List[str]:
    """Stories heard by `subject`, from an ``all_stories.json``-style file."""
    data = load_story_json(json_path)
    participants = data.get("participants", {})
    if subject not in participants:
        raise KeyError(
            f"{subject} not in {json_path} (have: {sorted(participants)})"
        )
    entry = participants[subject]
    return list(entry["stories"] if isinstance(entry, dict) else entry)


def train_test_stories(json_path) -> tuple:
    """(train, test) story lists from a ``train_test_split_*.json`` file."""
    data = load_story_json(json_path)
    return list(data["train"]["stories"]), list(data["test"]["stories"])


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------

def save_results(save_dir, results: Dict[str, object]) -> None:
    """Save each entry as ``<save_dir>/<name>.npy``; dicts/lists go to JSON."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, value in results.items():
        if isinstance(value, (dict, list, str)):
            with open(save_dir / f"{name}.json", "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2)
        else:
            np.save(save_dir / f"{name}.npy", np.asarray(value))
