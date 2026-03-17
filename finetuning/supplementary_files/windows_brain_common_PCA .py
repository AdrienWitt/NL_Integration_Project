"""
Combined OpenSMILE + Brain PCA extraction script – TR-aligned, averaged across subjects.
Pipeline:
  1. Build common fsaverage vertex mask (top PERCENTILE% by mean encoding r)
  2. Extract TR-aligned OpenSMILE features per story
  3. Load fsaverage brain .npy files, apply common mask
  4. Average brain responses across subjects per story per TR
  5. Fit single global PCA on averaged brain data (stories concatenated)
  6. Save one JSON per story to averaged/ folder:
       {story_safe}_prosody+brain-pca-avg.json
  7. Save diagnostic plots

Note: audio features saved RAW — z-score in dataset class.
      brain targets = averaged across subjects → PCA on clean stimulus-driven signal.
"""

import os
import sys
import json
import logging
import numpy as np
import librosa
import opensmile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import REPO_DIR, DER_DIR
from utils.prosody_utils import load_trfiles, RESPDICT_PATH

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FSAVERAGE_BRAIN_DIR = r"E:\NL\clean_nl_preproc\ds003020\derivative\fsaverage_brain"
CORRS_DIR           = r"C:\Users\wittmann\OneDrive - unige.ch\Documents\Sarcasm_experiment\NL_Project\encoding\results\opensmile_all_stories"
TRIM                = 5
PERCENTILE          = 95
N_PCA               = 3
WINDOW_SIZE         = 2.0

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SUBJECTS = ["UTS01", "UTS02", "UTS03", "UTS04", "UTS05",
            "UTS06", "UTS07", "UTS08", "UTS09"]

# ---------------------------------------------------------------------------
# OpenSMILE
# ---------------------------------------------------------------------------
def load_smile():
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def extract_opensmile_story(story, audio_dir, tr_times, smile):
    audio_path     = os.path.join(audio_dir, f"{story}.wav")
    y, sr          = librosa.load(audio_path, sr=None, mono=True)
    num_samples    = len(y)
    window_samples = int(WINDOW_SIZE * sr)
    result         = []
    feature_names  = None

    for tr_time in tr_times:
        start_sample = int(tr_time * sr)
        end_sample   = start_sample + window_samples
        if start_sample >= num_samples:
            window = np.zeros(window_samples, dtype=np.float32)
        elif end_sample <= num_samples:
            window = y[start_sample:end_sample]
        else:
            window = np.pad(y[start_sample:], (0, end_sample - num_samples),
                            mode='constant')

        feats = smile.process_signal(window, sr)
        if feature_names is None:
            feature_names = list(feats.columns)
        result.append(feats.values.reshape(1, -1))

    return np.vstack(result), feature_names

# ---------------------------------------------------------------------------
# Common mask
# ---------------------------------------------------------------------------
def build_common_mask():
    log.info("Building common fsaverage vertex mask ...")
    all_corrs = []
    for subject in SUBJECTS:
        path = os.path.join(CORRS_DIR, subject, "corrs_fsaverage.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path} — run project_corrs.py first")
        corrs = np.load(path).ravel()
        all_corrs.append(corrs)
        log.info(f"  {subject}: {corrs.shape[0]:,} vertices, "
                 f"max r={corrs.max():.3f}, mean r={corrs.mean():.3f}")

    all_corrs  = np.stack(all_corrs)
    mean_corrs = np.mean(all_corrs, axis=0)
    threshold  = np.percentile(mean_corrs, PERCENTILE)
    mask       = mean_corrs >= threshold

    log.info(f"Mask: {mask.sum():,} / {len(mask):,} vertices "
             f"(top {100-PERCENTILE}%, threshold r={threshold:.4f})")
    return mask, mean_corrs, all_corrs, float(threshold)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 70)
    log.info("EXTRACTION: OpenSMILE + Brain PCA (averaged across subjects)")
    log.info(f"  Vertex percentile : top {100-PERCENTILE}%")
    log.info(f"  PCA components    : {N_PCA}")
    log.info(f"  Trim (each side)  : {TRIM} TRs → slice [{5+TRIM}:-{TRIM}]")
    log.info("=" * 70)

    # Load metadata
    json_path = os.path.join(DER_DIR, "all_stories.json")
    with open(json_path, encoding="utf-8") as f:
        subject_to_stories = json.load(f)["participants"]

    trfiles   = load_trfiles(RESPDICT_PATH, tr=2.0, pad=5, start_time=10)
    audio_dir = os.path.join(REPO_DIR, "ds003020", "stimuli")
    smile     = load_smile()

    out_dir = Path(REPO_DIR) / "features" / "prosody" / "brain_targets_finetuning"
    avg_dir = out_dir / "averaged"
    out_dir.mkdir(parents=True, exist_ok=True)
    avg_dir.mkdir(parents=True, exist_ok=True)

    feature_names = None

    # ── Step 1: build common mask ─────────────────────────────────────────
    common_mask, mean_corrs, all_corrs, threshold_global = build_common_mask()
    n_common = int(common_mask.sum())

    # ── Step 2: load all data ─────────────────────────────────────────────
    log.info("\nStep 2: loading brain + audio data ...")

    story_brain_collect = {}  # {story: [brain_subj1, brain_subj2, ...]}
    story_audio         = {}  # {story: audio_arr} — identical across subjects
    story_n_subjects    = {}  # {story: n_subjects_with_data}

    for subject in SUBJECTS:
        stories = subject_to_stories.get(subject, [])
        log.info(f"\n  {subject} ({len(stories)} stories)")

        for story in stories:
            audio_path = os.path.join(audio_dir, f"{story}.wav")
            brain_path = os.path.join(FSAVERAGE_BRAIN_DIR, subject, f"{story}.npy")

            if not os.path.exists(audio_path):
                log.warning(f"    Audio missing: {story} → skip")
                continue
            if not os.path.exists(brain_path):
                log.warning(f"    Brain .npy missing: {story} → skip")
                continue
            if story not in trfiles:
                log.warning(f"    No TR timings: {story} → skip")
                continue

            tr_times = (trfiles[story][0].get_reltriggertimes() +
                        trfiles[story][0].soundstarttime)

            # Load brain + apply common mask
            brain_fs = np.load(brain_path)
            brain    = brain_fs[:, common_mask].astype(np.float32)
            if np.isnan(brain).any():
                brain = np.nan_to_num(brain, nan=0.0)

            # Extract audio once per story
            if story not in story_audio:
                raw_audio, story_feature_names = extract_opensmile_story(
                    story, audio_dir, tr_times, smile
                )
                if feature_names is None:
                    feature_names = story_feature_names
                    log.info(f"    Feature names: {len(feature_names)} from {story}")

                audio_trimmed = raw_audio[5 + TRIM: -TRIM]
                n_audio = audio_trimmed.shape[0]
                n_brain = brain.shape[0]

                if n_audio != n_brain:
                    log.warning(f"    TR mismatch {subject}/{story}: "
                                f"audio={n_audio}, brain={n_brain} → skip")
                    continue

                story_audio[story] = audio_trimmed
            else:
                # Validate brain length matches stored audio
                n_audio = story_audio[story].shape[0]
                n_brain = brain.shape[0]
                if n_audio != n_brain:
                    log.warning(f"    TR mismatch {subject}/{story}: "
                                f"audio={n_audio}, brain={n_brain} → skip")
                    continue

            story_brain_collect.setdefault(story, []).append(brain)
            log.info(f"    ✓ {story}: {brain.shape[0]} TRs "
                     f"(subject {len(story_brain_collect[story])})")

    if not story_brain_collect:
        raise RuntimeError("No valid data found — check paths")

    log.info(f"\n  Total stories collected: {len(story_brain_collect)}")
    for story, arrays in sorted(story_brain_collect.items()):
        log.info(f"    {story}: {len(arrays)} subjects")

    # ── Step 3: average brain per story → fit global PCA ─────────────────
    log.info("\nStep 3: averaging brain responses across subjects per story ...")

    story_order     = sorted(story_brain_collect.keys())
    story_avg_brain = {}
    story_n_trs     = {}
    brain_rows      = []

    for story in story_order:
        arrays    = story_brain_collect[story]
        avg_brain = np.mean(np.stack(arrays), axis=0)  # (n_trs, n_common)
        story_avg_brain[story]  = avg_brain
        story_n_trs[story]      = avg_brain.shape[0]
        story_n_subjects[story] = len(arrays)
        brain_rows.append(avg_brain)
        log.info(f"  {story}: {len(arrays)} subjects averaged → {avg_brain.shape}")

    log.info("\nStep 3b: fitting global PCA on averaged brain responses ...")
    brain_all  = np.vstack(brain_rows)
    total_trs  = brain_all.shape[0]
    log.info(f"  Averaged brain matrix: {brain_all.shape}")

    n_pca_actual  = min(N_PCA, brain_all.shape[1])
    pca           = PCA(n_components=n_pca_actual)
    brain_pcs_all = pca.fit_transform(brain_all)

    var_exp  = pca.explained_variance_ratio_.tolist()
    total_ve = float(np.sum(var_exp))
    log.info(f"  {n_pca_actual} components → {total_ve*100:.1f}% variance explained")
    for i, ve in enumerate(var_exp, 1):
        log.info(f"    PC{i}: {ve*100:.2f}%")


    # ── Step 5: slice PCA and save one JSON per story ─────────────────────
    log.info("\nStep 5: saving averaged JSON files ...")
    cursor = 0

    for story in story_order:
        n       = story_n_trs[story]
        avg_pcs = brain_pcs_all[cursor: cursor + n]
        cursor += n

        story_safe = "".join(
            c if c.isalnum() or c in " _-" else "_" for c in story
        ).replace(" ", "_").strip("_")

        story_data = {
            "story":            story,
            "n_TRs":            int(n),
            "n_subjects":       story_n_subjects[story],
            "tr_length_sec":    2.0,
            "audio_window_sec": 2.0,
            "feature_names":    feature_names,

            "audio_features": {
                "tr_aligned": {
                    "description": "eGeMAPSv02 functionals — window starts at TR onset",
                    "data": story_audio[story].tolist(),
                }
            },

            "brain_targets": {
                "pca": {
                    "type":                     "global_fsaverage_averaged_then_pca",
                    "n_components":             n_pca_actual,
                    "explained_variance_ratio": var_exp,
                    "total_variance_explained": total_ve,
                    "n_common_vertices":        n_common,
                    "vertex_threshold_r":       threshold_global,
                    "n_subjects_averaged":      story_n_subjects[story],
                    "data":                     avg_pcs.tolist(),
                }
            },

            "metadata": {
                "brain_space":     "fsaverage",
                "voxel_selection": f"top {100-PERCENTILE}% fsaverage vertices "
                                   f"by mean encoding r across all subjects",
                "brain_preproc":   "z-scored per story, projected to fsaverage, "
                                   "averaged across subjects, then global PCA",
                "audio_preproc":   f"trimmed [{5+TRIM}:-{TRIM}] TRs; "
                                   f"raw values (z-score in dataset)",
            }
        }

        out_path = avg_dir / f"{story_safe}_prosody+brain-pca-avg.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(story_data, f, indent=2)
        log.info(f"  Saved: {out_path.name}")

    assert cursor == total_trs, f"Cursor mismatch: {cursor} != {total_trs}"

    log.info("\n" + "=" * 70)
    log.info(f"Done. {len(story_order)} stories | {total_trs} TRs total")
    log.info(f"Files saved in: {avg_dir}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()