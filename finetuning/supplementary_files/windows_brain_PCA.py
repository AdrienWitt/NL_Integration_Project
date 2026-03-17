"""
Combined OpenSMILE + Brain PCA extraction script – TR-aligned only.
Pipeline:
  1. Extract TR-aligned OpenSMILE features (window starts at TR onset, no sliding shift)
  2. Load brain fMRI responses (already z-scored per story, temporally untouched),
     mask to top-N% voxels — NO temporal trimming applied to brain
  3. Trim audio features with [5+TRIM:-TRIM] to align with brain length
  4. Concatenate across ALL stories → assert lengths match (hard stop if not)
  5. Fit global PCA on concatenated brain data (no re-scaling — already z-scored)
  6. Save per-story JSON with nested structure:
       {
         "audio_features": { "tr_aligned": { "data": ..., "description": ... } },
         "brain_targets":  { "pca": { "data": ..., "explained_variance_ratio": ..., ... } },
         "feature_names": [...],
         "metadata": {...}
       }
     + aggregated per-subject file
Note: audio features saved RAW (not normalized) — z-score them in your dataset class.
"""

import os
import sys
import json
import logging
import numpy as np
import librosa
import opensmile
import h5py
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import REPO_DIR, DER_DIR
from utils.prosody_utils import load_trfiles, RESPDICT_PATH

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = r"E:\NL\clean_nl_preproc\ds003020\derivative\preprocessed_data"
TRIM = 5                    # TRs to trim from each side
PERCENTILE = 95             # voxel-selection percentile
N_PCA = 3                   # brain PCA components
SHIFTS = [0.0]              # ← only TR-aligned (no 1s shift)
WINDOW_SIZE = 2.0           # seconds – matches TR

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenSMILE
# ---------------------------------------------------------------------------
def load_smile():
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

def extract_opensmile_story(story: str, audio_dir: str, tr_times: np.ndarray,
                            smile) -> tuple[dict, list]:
    """
    Extract OpenSMILE features for each TR (only shift=0).
    Returns:
      - dict {0.0: np.ndarray (n_TRs, n_features)}
      - list of feature names
    """
    audio_path = os.path.join(audio_dir, f"{story}.wav")
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    num_samples = len(y)
    window_samples = int(WINDOW_SIZE * sr)

    result = {0.0: []}
    feature_names = None

    for tr_time in tr_times:
        start_time = tr_time + 0.0
        start_sample = int(start_time * sr)
        end_sample = start_sample + window_samples

        if start_sample >= num_samples:
            window = np.zeros(window_samples, dtype=np.float32)
        elif end_sample <= num_samples:
            window = y[start_sample:end_sample]
        else:
            window = y[start_sample:]
            pad_len = window_samples - len(window)
            window = np.pad(window, (0, pad_len), mode='constant')

        feats = smile.process_signal(window, sr)
        if feature_names is None:
            feature_names = list(feats.columns)

        result[0.0].append(feats.values.reshape(1, -1))

    return {0.0: np.vstack(result[0.0])}, feature_names

# ---------------------------------------------------------------------------
# Brain loading & masking
# ---------------------------------------------------------------------------
def load_encoding_scores(subject: str) -> np.ndarray:
    path = os.path.join(REPO_DIR, "encoding/results/opensmile_all_stories",
                        subject, "corrs.npy")
    return np.load(path)

def get_top_voxel_mask(subject: str, percentile: int = PERCENTILE):
    scores = load_encoding_scores(subject).ravel()
    threshold = np.percentile(scores, percentile)
    mask = scores >= threshold
    log.info(f" Voxel mask: {mask.sum():,} / {len(scores):,} "
             f"({mask.sum()/len(scores)*100:.1f}%) | threshold r = {threshold:.4f}")
    return mask, float(threshold)

def load_brain_story(subject: str, story: str, mask: np.ndarray) -> np.ndarray:
    path = os.path.join(DATA_DIR, subject, f"{story}.hf5")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing brain file: {path}")
    with h5py.File(path, "r") as hf:
        data = hf["data"][:]
    # Ensure (n_vox, n_TRs) → then transpose to (n_TRs, n_vox)
    if data.shape[0] < data.shape[1]:
        data = data.T
    if mask is not None:
        data = data[mask, :]
    return data.T  # (n_TRs, n_voxels masked)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 70)
    log.info("COMBINED EXTRACTION: OpenSMILE (TR-aligned) + Brain PCA")
    log.info(f" Voxel percentile     : {PERCENTILE}")
    log.info(f" PCA components       : {N_PCA}")
    log.info(f" Trim (TRs each side) : {TRIM} → slice [5+{TRIM}:-{TRIM}]")
    log.info(f" Audio shifts         : {SHIFTS}  ← only aligned")
    log.info(f" Audio loading        : librosa native sample rate")
    log.info(f" Audio features       : saved RAW — z-score later")
    log.info(f" Brain                : top voxels, z-scored per story, no trim")
    log.info("=" * 70)

    # Load story metadata
    json_path = os.path.join(DER_DIR, "all_stories.json")
    with open(json_path, encoding="utf-8") as f:
        all_stories_data = json.load(f)
    subject_to_stories: dict = all_stories_data["participants"]

    trfiles = load_trfiles(RESPDICT_PATH, tr=2.0, pad=5, start_time=10)
    audio_dir = os.path.join(REPO_DIR, "ds003020", "stimuli")
    smile = load_smile()

    out_dir = Path(REPO_DIR) / "features" / "prosody" / "brain_targets_finetuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_names = None

    for subject, stories in subject_to_stories.items():
        log.info(f"\n{'=' * 70}")
        log.info(f"Processing subject: {subject} ({len(stories)} stories)")
        log.info(f"{'=' * 70}")

        mask, threshold = get_top_voxel_mask(subject)
        audio_arrays = {}
        brain_arrays = {}
        valid_stories = []
        subject_story_data = {}

        for story in stories:
            if not os.path.exists(os.path.join(audio_dir, f"{story}.wav")):
                log.warning(f"Audio missing: {story} → skipping")
                continue
            if not os.path.exists(os.path.join(DATA_DIR, subject, f"{story}.hf5")):
                log.warning(f"Brain file missing: {story} → skipping")
                continue
            if story not in trfiles:
                log.warning(f"No TR timings: {story} → skipping")
                continue

            tr_times = (trfiles[story][0].get_reltriggertimes() +
                        trfiles[story][0].soundstarttime)

            brain = load_brain_story(subject, story, mask)
            log.info(f"Extracting OpenSMILE for {story} ...")

            raw_audio, story_feature_names = extract_opensmile_story(
                story, audio_dir, tr_times, smile
            )

            if feature_names is None and story_feature_names:
                feature_names = story_feature_names
                log.info(f"Feature names captured ({len(feature_names)}) from {story}")

            # Trim audio to match brain length
            audio_trimmed = {
                shift: arr[5 + TRIM : -TRIM] for shift, arr in raw_audio.items()
            }

            n_audio = audio_trimmed[0.0].shape[0]
            n_brain = brain.shape[0]
            if n_audio != n_brain:
                raise ValueError(
                    f"TR mismatch — {subject}/{story} | audio: {n_audio}, brain: {n_brain}"
                )

            log.info(f" ✓ {story}: {n_brain} TRs aligned")
            audio_arrays[story] = audio_trimmed
            brain_arrays[story] = brain
            valid_stories.append(story)

        if not valid_stories:
            log.warning(f"No valid stories for {subject} — skipping")
            continue

        if feature_names is None:
            log.error(f"No feature names captured for {subject}")
            continue

        # Fit global PCA
        log.info(f"Fitting global PCA on {len(valid_stories)} stories ...")
        brain_concat = np.vstack([brain_arrays[s] for s in valid_stories])
        log.info(f"Concat brain shape: {brain_concat.shape} (TRs × voxels)")

        if np.isnan(brain_concat).any():
            log.warning("NaNs in brain data → replacing with 0")
            brain_concat = np.nan_to_num(brain_concat, nan=0.0)

        n_pca_actual = min(N_PCA, brain_concat.shape[1])
        pca = PCA(n_components=n_pca_actual)
        brain_pcs_concat = pca.fit_transform(brain_concat)

        var_exp = pca.explained_variance_ratio_.tolist()
        total_ve = float(np.sum(var_exp))
        log.info(f"PCA: {n_pca_actual} components explain {total_ve*100:.1f}% variance")
        for i, ve in enumerate(var_exp, 1):
            log.info(f" PC{i}: {ve*100:.1f}%")

        # Save per story
        cursor = 0
        for story in valid_stories:
            n = brain_arrays[story].shape[0]
            story_brain_pcs = brain_pcs_concat[cursor : cursor + n]
            cursor += n

            story_data = {
                "subject": subject,
                "story": story,
                "n_TRs": int(n),
                "tr_length_sec": 2.0,
                "audio_window_sec": 2.0,

                "feature_names": feature_names,

                "audio_features": {
                    "tr_aligned": {
                        "description": "eGeMAPSv02 functionals — window starts exactly at TR onset",
                        "data": audio_arrays[story][0.0].tolist()
                    }
                },

                "brain_targets": {
                    "pca": {
                        "type": "global_per_subject",
                        "n_components": n_pca_actual,
                        "explained_variance_ratio": var_exp,
                        "total_variance_explained": total_ve,
                        "data": story_brain_pcs.tolist()
                    }
                },

                "metadata": {
                    "n_voxels_selected": int(brain_arrays[story].shape[1]),
                    "encoding_threshold_r": float(threshold),
                    "voxel_selection": f"top {100 - PERCENTILE}% by encoding r",
                    "brain_preproc": "z-scored per story, no temporal trim",
                    "audio_preproc": f"trimmed [{5+TRIM}:-{TRIM}] TRs; raw values (z-score later)",
                }
            }

            # Save per-story
            story_safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in story).replace(" ", "_").strip("_")
            per_story_path = out_dir / f"sub-{subject}_{story_safe}_prosody+brain-pca.json"
            with open(per_story_path, "w", encoding="utf-8") as f:
                json.dump(story_data, f, indent=2)
            log.info(f" Saved: {per_story_path.name}")

            subject_story_data[story] = story_data

        # Save per-subject aggregate
        per_subject_data = {
            "subject": subject,
            "feature_names": feature_names,
            "pca_global": {
                "n_components": n_pca_actual,
                "explained_variance_ratio": var_exp,
                "total_variance_explained": total_ve,
            },
            "stories": subject_story_data
        }

        per_subject_path = out_dir / f"sub-{subject}_all-stories_prosody+brain-pca.json"
        with open(per_subject_path, "w", encoding="utf-8") as f:
            json.dump(per_subject_data, f, indent=2)
        log.info(f" Saved per-subject: {per_subject_path.name}")

        log.info(f"Completed {subject}: {len(valid_stories)} stories | {cursor} TRs")

    log.info("\n" + "=" * 70)
    log.info("Extraction finished.")
    log.info(f"Files saved in: {out_dir}")
    log.info("=" * 70)

if __name__ == "__main__":
    main()