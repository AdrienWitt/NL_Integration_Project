import os
import json
import numpy as np
import librosa
import opensmile
from tqdm import tqdm
from pathlib import Path
from .config import REPO_DIR

from prosody_utils import load_simulated_trfiles

# Initialize OpenSMILE
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

# Constants
WINDOW_SIZE = 2.0
RESPDICT_PATH = Path(REPO_DIR) / "ds003020" / "derivative" / "respdict.json"


def extract_tr_aligned_features(audio_path: str, story_name: str, trfiles: dict) -> list:
    """Extract OpenSMILE features for TR-aligned 2-second windows + 1s shifted version."""
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
        # Generate two windows per TR: at tr_time and tr_time + 1s
        for shift in [0.0, 1.0]:
            start_time = tr_time + shift
            end_time = start_time + WINDOW_SIZE

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            # Skip if start is before audio or fully beyond
            if start_sample >= len(y):
                continue

            # Extract window (pad at end if needed)
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
                        'tr_time': float(tr_time),          # reference TR
                        'shift': float(shift),              # 0 or 1
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
    return features_list


def main():
    stimuli_dir = Path(REPO_DIR) / "ds003020" / "stimuli"
    output_dir = Path(REPO_DIR) / "features" / "prosody" / "opensmile"
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted([f for f in os.listdir(stimuli_dir) if f.endswith('.wav')])
    print(f"Found {len(audio_files)} stories")

    with open(RESPDICT_PATH, "r") as f:
        respdict = json.load(f)
    trfiles = load_simulated_trfiles(respdict, tr=2.0, pad=5, start_time=10)

    for audio_file in tqdm(audio_files, desc="Processing stories"):
        story_name = audio_file[:-4]
        audio_path = stimuli_dir / audio_file

        windows = extract_tr_aligned_features(
            audio_path=str(audio_path),
            story_name=story_name,
            trfiles=trfiles
        )

        if windows:
            output_path = output_dir / f"{story_name}_opensmile_tr_aligned.json"
            with open(output_path, 'w') as f:
                json.dump(windows, f, indent=2)
            print(f"Saved → {output_path.name}")
        else:
            print(f"No windows saved for {story_name}")


if __name__ == "__main__":
    main()