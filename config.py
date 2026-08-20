"""
Single source of truth for every path in the project.

Every path can be overridden with an environment variable of the same name,
so the exact same code runs on the laptop, on an external drive, or on a
cluster without editing this file:

    export FMRI_DIR=/scratch/$USER/ds003020/derivative/preprocessed_data
    python -m encoding.run_encoding --subjects all

Only `FMRI_DIR` normally needs overriding: the full LeBel preprocessed
responses are large and often live outside the repo (e.g. E:/NL/...), while
this repo ships only the subjects you have copied locally.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Root
# --------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent


def _env(name: str, default: Path) -> Path:
    """Return $name as a Path if set, else `default`."""
    value = os.environ.get(name)
    return Path(value) if value else default


# --------------------------------------------------------------------------
# Data (LeBel et al. 2023, ds003020)
# --------------------------------------------------------------------------

DATA_DIR = _env("DATA_DIR", PROJECT_DIR / "data")

DS003020_DIR   = _env("DS003020_DIR", DATA_DIR / "ds003020")
DERIVATIVE_DIR = _env("DERIVATIVE_DIR", DS003020_DIR / "derivative")

#: Raw stimulus wavs shipped with ds003020 (native sample rate).
STIMULI_DIR = _env("STIMULI_DIR", DS003020_DIR / "stimuli")

#: Same stimuli resampled to 16 kHz mono — what wav2vec2/HuBERT/WavLM expect.
STIMULI_16K_DIR = _env("STIMULI_16K_DIR", DATA_DIR / "stimuli_16k")

#: Voxelwise preprocessed fMRI: <FMRI_DIR>/<subject>/<story>.hf5
FMRI_DIR = _env("FMRI_DIR", DERIVATIVE_DIR / "preprocessed_data")

#: Optional fsaverage-projected responses: <FSAVERAGE_DIR>/<subject>/<story>.npy
FSAVERAGE_DIR = _env("FSAVERAGE_DIR", DERIVATIVE_DIR / "fsaverage_brain")

TEXTGRID_DIR  = _env("TEXTGRID_DIR", DERIVATIVE_DIR / "TextGrids")
RESPDICT_PATH = _env("RESPDICT_PATH", DERIVATIVE_DIR / "respdict.json")

# --------------------------------------------------------------------------
# Stimulus features (one .hf5 per story, dataset "data", shape (n_TRs, n_dim))
# --------------------------------------------------------------------------

FEATURES_DIR = _env("FEATURES_DIR", DATA_DIR / "features")

#: TR-aligned eGeMAPS windows, one JSON per story, for fine-tuning.
PROSODY_DIR         = FEATURES_DIR / "prosody" / "opensmile"
FINETUNE_TARGET_DIR = _env("FINETUNE_TARGET_DIR",
                           FEATURES_DIR / "prosody" / "finetune_targets")

# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

SPLIT_DIR = _env("SPLIT_DIR", DATA_DIR / "splits")

#: Fine-tuning story split (train / val / held-out test).
FINETUNE_SPLIT = SPLIT_DIR / "stories_split.json"

#: Encoding story lists. `all_stories.json` maps subject -> stories heard.
ENCODING_SPLIT_DIR = _env("ENCODING_SPLIT_DIR", DATA_DIR / "derivative")

# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

RESULTS_DIR    = _env("RESULTS_DIR", PROJECT_DIR / "results")
ENCODING_OUT   = RESULTS_DIR / "encoding"
FINETUNE_OUT   = RESULTS_DIR / "finetune"
STATS_OUT      = RESULTS_DIR / "stats"

# --------------------------------------------------------------------------
# Acquisition / stimulus constants
# --------------------------------------------------------------------------

TR              = 2.0     # seconds
SAMPLING_RATE   = 16000   # Hz, wav2vec2 input rate
WINDOW_SIZE_SEC = 2.0     # audio window per TR — matches the TR
TR_PAD          = 5       # TRs dropped by load_simulated_trfiles
TR_START_TIME   = 10.0    # sound onset relative to first trigger, seconds

# --------------------------------------------------------------------------
# Prosody target set
# --------------------------------------------------------------------------

#: openSMILE feature set used for every prosody target and for the eGeMAPS
#: audio band. eGeMAPSv02 Functionals is 88 features: F0, jitter, shimmer,
#: loudness, HNR, spectral slopes, formants and MFCCs, each summarised over
#: the window by its functionals.
EGEMAPS_FEATURE_SET = "eGeMAPSv02"
EGEMAPS_N_FUNCTIONALS = 88

#: Story dropped from every training set and reserved as the final test story.
#: It is the one story with 10 repeats, which is what makes an explainable-
#: variance ceiling (and therefore a noise-ceiling-normalised score) possible.
HELD_OUT_STORY = "wheretheressmoke"

SUBJECTS = ["UTS01", "UTS02", "UTS03", "UTS04", "UTS05",
            "UTS06", "UTS07", "UTS08", "UTS09"]


def ensure_dirs(*paths) -> None:
    """mkdir -p for each argument. Accepts str or Path."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """Human-readable path dump — handy as a first sanity check."""
    rows = [
        ("PROJECT_DIR",         PROJECT_DIR),
        ("DATA_DIR",            DATA_DIR),
        ("STIMULI_DIR",         STIMULI_DIR),
        ("STIMULI_16K_DIR",     STIMULI_16K_DIR),
        ("FMRI_DIR",            FMRI_DIR),
        ("FSAVERAGE_DIR",       FSAVERAGE_DIR),
        ("TEXTGRID_DIR",        TEXTGRID_DIR),
        ("RESPDICT_PATH",       RESPDICT_PATH),
        ("FEATURES_DIR",        FEATURES_DIR),
        ("PROSODY_DIR",         PROSODY_DIR),
        ("FINETUNE_TARGET_DIR", FINETUNE_TARGET_DIR),
        ("SPLIT_DIR",           SPLIT_DIR),
        ("ENCODING_SPLIT_DIR",  ENCODING_SPLIT_DIR),
        ("RESULTS_DIR",         RESULTS_DIR),
    ]
    width = max(len(name) for name, _ in rows)
    lines = []
    for name, path in rows:
        mark = "ok " if Path(path).exists() else "MISSING"
        lines.append(f"  [{mark}] {name:<{width}}  {path}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("Resolved project paths:\n")
    print(describe())
