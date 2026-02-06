import os
import json
from typing import Dict, List, Optional, Union

import torch
import numpy as np
import torchaudio
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from transformers import AutoFeatureExtractor

from .prosody_utils import SAMPLING_RATE, WINDOW_SIZE_SEC, RESPDICT_PATH, load_trfiles


class ProsodyDataset(Dataset):
    """
    Dataset that loads TR-aligned 2-second audio windows and corresponding OpenSMILE prosody features.
    - One sample per valid TR (non-overlapping, starting at TR onset).
    - Labels are taken from pre-computed JSON files.
    - Supports global normalization and optional PCA.
    """

    def __init__(
        self,
        audio_dir: str,
        prosody_dir: str,
        processor: AutoFeatureExtractor,
        story_names: Optional[List[str]] = None,
        use_pca: bool = False,
        pca_threshold: float = 0.90,
        pca: Optional[PCA] = None,
        scalers: Optional[Dict[str, StandardScaler]] = None,
        respdict_path: str = RESPDICT_PATH,
        tr: float = 2.0,
        pad: int = 5,
        sampling_rate: int = SAMPLING_RATE,
        window_size_sec: float = WINDOW_SIZE_SEC,
    ):
        self.audio_dir = audio_dir
        self.prosody_dir = prosody_dir
        self.processor = processor
        self.story_names_filter = story_names or []
        self.use_pca = use_pca
        self.pca_threshold = pca_threshold
        self.pca = pca
        self.scalers = scalers

        self.sampling_rate = sampling_rate
        self.max_length = int(window_size_sec * sampling_rate)
        self.window_size_sec = window_size_sec

        self.resampler = torchaudio.transforms.Resample(
            orig_freq=sampling_rate, new_freq=sampling_rate
        )

        # Load TR timing information once
        self.trfiles = load_trfiles(respdict_path, tr=tr, pad=pad, start_time=10)

        # Discover valid stories (audio + JSON + TR info)
        self.available_stories = sorted(
            f[:-4] for f in os.listdir(audio_dir) if f.endswith(".wav")
        )
        self.valid_stories = [
            story
            for story in self.available_stories
            if story in self.trfiles
            and os.path.exists(os.path.join(prosody_dir, f"{story}_opensmile_tr_aligned.json"))
        ]
        
        if story_names:
            requested = set(story_names)
            self.valid_stories = [
                s for s in self.valid_stories if s in requested
            ]
            if not self.valid_stories:
                raise ValueError(
                    f"None of the requested stories {story_names} "
                    f"are valid. Available valid: {self.valid_stories}"
                )

        if not self.valid_stories:
            raise ValueError("No valid stories found (audio + JSON + TR timing)")

        # Pre-process all data
        self.preprocessed_data = self._preprocess_data()

        # Apply optional story name filter
        if self.story_names_filter:
            self.preprocessed_data = [
                item for item in self.preprocessed_data
                if item["story_name"] in self.story_names_filter
            ]

        # Set feature names from first sample
        self.feature_names = (
            self.preprocessed_data[0]["feature_names"] if self.preprocessed_data else []
        )

    def _preprocess_data(self) -> List[Dict]:
        preprocessed = []
        all_raw_features = {}  # Collect for global normalization

        # 1. Collect all raw feature values across all stories
        for story in self.valid_stories:
            json_path = os.path.join(self.prosody_dir, f"{story}_opensmile_tr_aligned.json")
            with open(json_path, "r") as f:
                windows = json.load(f) or []

            for window in windows:
                for feat, val in window.get("features", {}).items():
                    all_raw_features.setdefault(feat, []).append(val)

        # Fit scalers if not provided
        if self.scalers is None:
            self.scalers = {
                feat: StandardScaler().fit(np.array(values).reshape(-1, 1))
                for feat, values in all_raw_features.items()
            }

        # 2. Process each story and generate samples
        for story in self.valid_stories:
            # Load full waveform
            audio_path = os.path.join(self.audio_dir, f"{story}.wav")
            waveform, sr = torchaudio.load(audio_path)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if sr != self.sampling_rate:
                waveform = self.resampler(waveform)

            # Get TR times
            tr_info = self.trfiles[story][0]
            tr_times = tr_info.get_reltriggertimes() + tr_info.soundstarttime

            # Load corresponding OpenSMILE windows (assume order matches)
            json_path = os.path.join(self.prosody_dir, f"{story}_opensmile_tr_aligned.json")
            with open(json_path, "r") as f:
                opensmile_windows = json.load(f) or []

            if len(opensmile_windows) != len(tr_times):
                print(
                    f"Warning: {story} - {len(opensmile_windows)} JSON windows vs "
                    f"{len(tr_times)} TR times → mismatch!"
                )

            for idx, (tr_time, osm_window) in enumerate(zip(tr_times, opensmile_windows)):
                for shift in [0.0, 1.0]:
                    start_time = float(tr_time)
                    end_time = start_time + self.window_size_sec
    
                    start_sample = int(start_time * self.sampling_rate)
                    end_sample = min(
                        int(end_time * self.sampling_rate), waveform.shape[1]
                    )

                    window_audio_np = waveform[0, start_sample:end_sample].numpy()

                    # Process with Wav2Vec2 processor
                    inputs = self.processor(
                        window_audio_np,
                        sampling_rate=self.sampling_rate,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=self.max_length,
                        truncation=True,
                    )
                    input_values = inputs.input_values.squeeze(0)  # [max_length]

                    # Normalize labels
                    normalized = {}
                    for feat, raw_val in osm_window.get("features", {}).items():
                        normalized[feat] = float(
                            self.scalers[feat].transform([[raw_val]])[0][0]
                        )

                    feat_names = sorted(normalized.keys())
                    labels_tensor = torch.tensor(
                        [normalized[fn] for fn in feat_names], dtype=torch.float32
                    )

                    preprocessed.append(
                        {
                            "input_values": input_values,
                            "labels": labels_tensor,
                            "story_name": story,
                            "window_time": f"{start_time:.2f}-{end_time:.2f}",
                            "feature_names": feat_names,
                            "tr_index": idx,
                            "tr_time": float(tr_time),
                        }
                    )

        # Optional PCA
        if self.use_pca and preprocessed:
            X = np.array([item["labels"].numpy() for item in preprocessed])

            if self.pca is None:
                self.pca = PCA(n_components=self.pca_threshold)
                X_pca = self.pca.fit_transform(X)
                print(
                    f"PCA → {self.pca.n_components_} components, "
                    f"explained variance: {sum(self.pca.explained_variance_ratio_):.3f}"
                )
            else:
                X_pca = self.pca.transform(X)

            for i, item in enumerate(preprocessed):
                item["labels"] = torch.tensor(X_pca[i], dtype=torch.float32)
                item["feature_names"] = [f"PC_{j+1}" for j in range(X_pca.shape[1])]

            self.pca_info = {
                "n_components": self.pca.n_components_,
                "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
                "cumulative_explained_variance": np.cumsum(
                    self.pca.explained_variance_ratio_
                ).tolist(),
            }

        return preprocessed

    def __len__(self) -> int:
        return len(self.preprocessed_data)

    def __getitem__(self, idx: int) -> Dict:
        return self.preprocessed_data[idx]
