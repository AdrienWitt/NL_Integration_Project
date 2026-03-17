import os
import sys
import json
import numpy as np
import librosa
import opensmile
from tqdm import tqdm
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import h5py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import DATA_DIR, REPO_DIR, DER_DIR
from utils.prosody_utils import load_simulated_trfiles

# Initialize OpenSMILE
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

# Constants from audio script
WINDOW_SIZE = 2.0
RESPDICT_PATH = Path(REPO_DIR) / "ds003020" / "derivative" / "respdict.json"

# Brain extraction functions (adapted)
def get_masked_responses(stories, subject, mask):
    subject_dir = os.path.join(DATA_DIR, subject)
    resp = {}
    for story in stories:
        path = os.path.join(subject_dir, f"{story}.hf5")
        if not os.path.isfile(path):
            print(f" ⚠️ Missing: {story}.hf5 → skipping")
            continue
        print(f" 📂 Loading {story}.hf5...", end=" ")
        with h5py.File(path, "r") as hf:
            data = hf["data"][:]
        if data.shape[0] < data.shape[1]:  # likely already (n_TRs, n_vox)
            data = data.T  # → (n_vox, n_TRs)
        if mask is not None:
            data = data[mask, :]
        resp[story] = data.T  # Return as (n_TRs, n_voxels)
        print(f"✓ ({data.T.shape[0]} TRs, {data.T.shape[1]} voxels)")
    return resp

def load_encoding_scores(subject):
    subject_dir = os.path.join(REPO_DIR, "encoding/results/opensmile_all_stories", subject)
    scores_path = os.path.join(subject_dir, "corrs.npy")
    scores = np.load(scores_path)
    return scores

def get_top_voxel_mask(subject, percentile=95):
    scores = load_encoding_scores(subject)
    voxel_scores = scores.ravel()
    threshold = np.percentile(voxel_scores, percentile)
    mask = voxel_scores >= threshold
    n_voxels = len(voxel_scores)
    n_selected = mask.sum()
    print(f" 📊 Selected: {n_selected:,} / {n_voxels:,} voxels "
          f"({n_selected/n_voxels*100:.1f}%) | threshold r = {threshold:.4f}")
    return mask, threshold

def extract_pca_targets(percentile=95, n_pca=3):
    json_path = os.path.join(DER_DIR, "all_stories.json")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    subject_to_stories = data.get("participants", {})
    output = {}
    total_subjects = len(subject_to_stories)
    for subj_idx, (subject, stories) in enumerate(subject_to_stories.items(), 1):
        print(f"\n{'='*70}")
        print(f"[{subj_idx}/{total_subjects}] Processing subject: {subject}")
        print(f"{'='*70}")
        mask, threshold = get_top_voxel_mask(subject, percentile=percentile)
        print(f"\n 📖 Loading {len(stories)} stories for {subject}:")
        responses = get_masked_responses(stories, subject, mask)
        if not responses:
            print(f" ⚠️ No valid stories for {subject}, skipping")
            continue
        print(f"\n 🔬 Fitting GLOBAL PCA on all {len(responses)} stories...")
        all_data = []
        story_names = []
        story_lengths = []
        for story, arr in responses.items():
            if arr.size > 0:
                all_data.append(arr)  # (n_TRs, n_voxels)
                story_names.append(story)
                story_lengths.append(arr.shape[0])
        if not all_data:
            print(f" ⚠️ No valid data for {subject}, skipping")
            continue
        concat_data = np.vstack(all_data)
        print(f" Concatenated shape: {concat_data.shape} (total TRs × voxels)")
        scaler = StandardScaler()
        concat_scaled = scaler.fit_transform(concat_data)
        if concat_data.shape[1] < n_pca:
            print(f" ⚠️ Only {concat_data.shape[1]} voxels, reducing to {concat_data.shape[1]} components")
            n_pca_actual = concat_data.shape[1]
        else:
            n_pca_actual = n_pca
        pca = PCA(n_components=n_pca_actual)
        pca.fit(concat_scaled)
        var_explained = pca.explained_variance_ratio_
        cumsum_var = np.cumsum(var_explained)
        print(f" ✓ PCA fitted: PC1-{n_pca_actual} explain {cumsum_var[-1]*100:.1f}% variance")
        for i in range(n_pca_actual):
            print(f" PC{i+1}: {var_explained[i]*100:.1f}% (cumulative: {cumsum_var[i]*100:.1f}%)")
        print(f"\n 📊 Projecting each story onto global PCA components:")
        output[subject] = {}
        start_idx = 0
        for story, length in zip(story_names, story_lengths):
            print(f" • {story}...", end=" ")
            story_data_scaled = concat_scaled[start_idx:start_idx + length, :]
            components = pca.transform(story_data_scaled)  # (n_TRs, n_pca)
            story_output = {}
            for i in range(n_pca_actual):
                story_output[f'PC{i+1}'] = components[:, i].tolist()
            story_output['explained_variance_ratio'] = var_explained.tolist()
            story_output['metadata'] = {
                'n_voxels_total': int(concat_data.shape[1]),
                'n_TRs': int(length),
                'threshold_r': float(threshold),
                'n_components': n_pca_actual,
                'total_variance_explained': float(cumsum_var[-1]),
                'pca_type': 'global'
            }
            output[subject][story] = story_output
            start_idx += length
            print(f"✓ ({length} TRs)")
        print(f"\n ✅ Completed {subject}: {len(story_names)} stories processed")
    return output

# Audio extraction function (adapted)
def extract_tr_aligned_features(audio_path: str, story_name: str, trfiles: dict) -> list:
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr
    print(f"\nProcessing: {audio_path} ({duration:.1f}s @ {sr}Hz)")
    window_samples = int(WINDOW_SIZE * sr)
    features_list = []
    if story_name not in trfiles:
        print(f"No TR times found for {story_name}")
        return []
    tr_info = trfiles[story_name][0]
    tr_times = tr_info.get_reltriggertimes() + tr_info.soundstarttime
    for idx, tr_time in enumerate(tr_times):
        for shift in [0.0, 1.0]:
            start_time = tr_time + shift
            end_time = start_time + WINDOW_SIZE
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            if start_sample >= len(y):
                continue
            if end_sample <= len(y):
                window_audio = y[start_sample:end_sample]
            else:
                window_audio = y[start_sample:]
                pad_len = end_sample - len(y)
                window_audio = np.pad(window_audio, (0, pad_len), mode='constant')
            try:
                feats = smile.process_signal(window_audio, sr)
                if not feats.empty:
                    feature_dict = {
                        'window_start_time': float(start_time),
                        'window_end_time': float(end_time),
                        'tr_time': float(tr_time),
                        'shift': float(shift),
                        'features': {col: float(val) for col, val in zip(feats.columns, feats.iloc[0].values)}
                    }
                    features_list.append(feature_dict)
                    if idx == 0 and shift == 0.0:
                        print(f"Extracted {len(feats.columns)} features (eGeMAPSv02 functionals)")
                        print(f"Example feature names: {list(feats.columns)[:5]}...")
                    print(f" TR {idx:3d} shift {shift:.1f}: {start_time:6.1f} – {end_time:6.1f}s")
            except Exception as e:
                print(f" Warning: Failed TR {idx} shift {shift} at {start_time:.1f}s → {e}")
    print(f"→ {len(features_list)} TR-aligned windows extracted (approx 2× original)")
    return features_list, tr_times

# Combined main function
def main():
    print("="*70)
    print("COMBINED EXTRACTION: OPENSMILE FEATURES + BRAIN PCA TARGETS")
    print("Using GLOBAL PCA (fit on all stories per subject)")
    print("="*70)
    print("Configuration:")
    print(" • Percentile: 95th (top 5% of voxels)")
    print(" • PCA components: 3")
    print(" • PCA mode: GLOBAL (same components across all stories)")
    print(" • Audio windows: 2s with 0s and 1s shifts per TR (repeating PCA per TR)")
    print("="*70)

    # Extract brain PCA
    brain_data = extract_pca_targets(percentile=95, n_pca=3)

    # Save brain targets as in original
    out_dir = Path(REPO_DIR) / "features" / "prosody" / "brain_targets_finetuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    brain_outfile = out_dir / "top95pct_globalPCA3_targets.json"
    with open(brain_outfile, "w", encoding="utf-8") as f:
        json.dump(brain_data, f, indent=2)
    print(f"\n💾 Brain targets saved: {brain_outfile}")

    # Load trfiles
    with open(RESPDICT_PATH, "r") as f:
        respdict = json.load(f)
    trfiles = load_simulated_trfiles(respdict, tr=2.0, pad=5, start_time=10)

    # Process audio and combine per subject/story
    stimuli_dir = Path(REPO_DIR) / "ds003020" / "stimuli"
    audio_files = sorted([f for f in os.listdir(stimuli_dir) if f.endswith('.wav')])
    print(f"Found {len(audio_files)} audio stories")

    combined_output = {}

    for subject in brain_data:
        combined_output[subject] = {}
        for story in tqdm(brain_data[subject], desc=f"Combining for {subject}"):
            audio_file = f"{story}.wav"
            audio_path = stimuli_dir / audio_file
            if not audio_path.exists():
                print(f" ⚠️ Audio not found for {story}, skipping")
                continue

            windows, tr_times = extract_tr_aligned_features(str(audio_path), story, trfiles)
            if not windows:
                print(f" ⚠️ No windows for {story}, skipping")
                continue

            brain_story = brain_data[subject][story]
            pcs = {k: brain_story[k] for k in brain_story if k.startswith('PC')}
            n_trs = brain_story['metadata']['n_TRs']
            metadata = brain_story['metadata']
            explained_variance = brain_story['explained_variance_ratio']

            # Ensure lengths match (n_trs should == len(tr_times))
            if n_trs != len(tr_times):
                print(f" ⚠️ Mismatch: {n_trs} brain TRs vs {len(tr_times)} audio TRs for {story}, skipping")
                continue

            combined_windows = []
            window_idx = 0
            for tr_idx, tr_time in enumerate(tr_times):
                for shift in [0.0, 1.0]:
                    if window_idx >= len(windows):
                        break
                    window = windows[window_idx]
                    if abs(window['tr_time'] - tr_time) > 1e-6 or window['shift'] != shift:
                        print(f" ⚠️ Mismatch at TR {tr_idx} shift {shift}, skipping window")
                        continue

                    combined_window = window.copy()
                    brain_pcs = {}
                    for pc_key in pcs:
                        brain_pcs[pc_key] = pcs[pc_key][tr_idx]  # Repeat the same PCA value for both shifts

                    combined_window['brain_pcs'] = brain_pcs
                    combined_window['metadata'] = metadata  # Add per-story metadata
                    combined_window['explained_variance_ratio'] = explained_variance  # Global per subject

                    combined_windows.append(combined_window)
                    window_idx += 1

            if combined_windows:
                combined_output[subject][story] = combined_windows
            else:
                print(f" ⚠️ No combined windows for {story}")

    # Save combined JSON
    combined_outfile = out_dir / "combined_opensmile_pca_top95pct_globalPCA3.json"
    with open(combined_outfile, "w", encoding="utf-8") as f:
        json.dump(combined_output, f, indent=2)
    print(f"\n💾 Combined results saved: {combined_outfile}")

    # Final summary (adapted from brain script)
    total_subjects = len(combined_output)
    total_stories = sum(len(stories) for stories in combined_output.values())
    print(f"\n{'='*70}")
    print(f"✅ PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"📁 Saved: {combined_outfile}")
    print(f"📊 Targets per window: OpenSmile features + 3 GLOBAL PCA components (repeated per TR for 1s shifts)")
    print(f"👥 Subjects processed: {total_subjects}")
    print(f"📖 Total stories processed: {total_stories}")
    print(f"\n💡 Note: PCA was fit GLOBALLY per subject; repeated for each 1s-shifted window")
    print(f" → Allows choice of 1s or 2s windows in finetuning")
    print(f"{'='*70}")

    # Validation (adapted)
    if combined_output:
        print(f"\n{'='*70}")
        print(f"VALIDATION: EXAMPLE OUTPUT")
        print(f"{'='*70}")
        first_sub = next(iter(combined_output))
        stories_list = list(combined_output[first_sub].keys())
        print(f"\n📌 Subject: {first_sub}")
        print(f" Stories: {len(stories_list)}")
        print(f"{'-'*70}")
        if stories_list:
            first_story = stories_list[0]
            data = combined_output[first_sub][first_story]
            print(f"\n🎯 Example story: {first_story}")
            print(f"Number of windows: {len(data)} (approx 2x TRs)")
            # Show first window
            first_window = data[0]
            print(f"\nExample window (first):")
            print(f" - window_start_time: {first_window['window_start_time']:.1f}")
            print(f" - tr_time: {first_window['tr_time']:.1f}")
            print(f" - shift: {first_window['shift']:.1f}")
            print(f" - features: {len(first_window['features'])} OpenSmile features")
            print(f"   Example: {list(first_window['features'].keys())[:3]}...")
            print(f" - brain_pcs: {list(first_window['brain_pcs'].keys())}")
            print(f"   PC1 preview: {first_window['brain_pcs']['PC1']:.4f}")
            if 'explained_variance_ratio' in first_window:
                print(f"\n📊 Variance explained:")
                var_ratios = first_window['explained_variance_ratio']
                cumsum = np.cumsum(var_ratios)
                for i, (var, cum) in enumerate(zip(var_ratios, cumsum), 1):
                    print(f" PC{i}: {var*100:5.2f}% (cumulative: {cum*100:5.2f}%)")
            print(f"\n📋 Metadata:")
            for k, v in sorted(first_window['metadata'].items()):
                print(f" {k:30s}: {v}")
        print(f"\n{'='*70}")
        print("✅ All done! Ready for wav2vec finetuning with combined targets.")
        print(f"{'='*70}")

if __name__ == "__main__":
    main()