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
    

    
class ProsodyBrainDataset(Dataset):
    """
    TR-aligned dataset for fine-tuning Wav2Vec2 on prosody or brain targets.

    Parameters
    ----------
    audio_dir : str
        Directory containing ``<story>.wav`` files.
    prosody_brain_dir : str
        Directory with combined JSON files produced by the extraction pipeline.
        Expected naming: ``sub-{subject}_{story}_prosody+brain-pca.json``
    processor : AutoFeatureExtractor
        Wav2Vec2 (or compatible) feature extractor.
    use_brain_pca : bool
        If True  → labels are brain PCA components (long format: all subjects).
        If False → labels are z-scored OpenSMILE features (each story once).
    story_names : list[str] | None
        Optional whitelist of story names to include.
    subject_names : list[str] | None
        Optional whitelist of subject IDs to include (only relevant when
        ``use_brain_pca=True``).
    scalers : dict | None
        Pre-fitted ``{feature_name: StandardScaler}`` for audio features.
        If None, scalers are fitted on the loaded data (prosody-only mode) or
        on the union of all subjects' data (brain mode).
    respdict_path : str
        Path to the TR timing respdict (forwarded to ``load_trfiles``).
    tr : float
        TR length in seconds.
    pad : int
        Padding passed to ``load_trfiles``.
    sampling_rate : int
        Target audio sampling rate.
    window_size_sec : float
        Duration of each audio window (should equal TR).
    """

    def __init__(
        self,
        audio_dir: str,
        prosody_brain_dir: str,
        processor: AutoFeatureExtractor,
        use_brain_pca: bool = False,
        story_names: Optional[List[str]] = None,
        subject_names: Optional[List[str]] = None,
        scalers: Optional[Dict[str, StandardScaler]] = None,
        respdict_path: str = RESPDICT_PATH,
        tr: float = 2.0,
        pad: int = 5,
        sampling_rate: int = SAMPLING_RATE,
        window_size_sec: float = WINDOW_SIZE_SEC,
    ):
        self.audio_dir = audio_dir
        self.prosody_brain_dir = prosody_brain_dir
        self.processor = processor
        self.use_brain_pca = use_brain_pca
        self.story_filter = set(story_names) if story_names else None
        self.subject_filter = set(subject_names) if subject_names else None
        self.scalers = scalers

        self.sampling_rate = sampling_rate
        self.max_length = int(window_size_sec * sampling_rate)
        self.window_size_sec = window_size_sec

        self.trfiles = load_trfiles(respdict_path, tr=tr, pad=pad, start_time=10)

        # Discover JSON files and decide which to load
        self._json_records = self._discover_jsons()
        if not self._json_records:
            raise ValueError(
                "No valid JSON records found. Check prosody_brain_dir and filters."
            )

        # Build samples
        self.samples: List[Dict] = []
        self.feature_names: List[str] = []
        self._build_samples()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_jsons(self) -> List[Dict]:
        """
        Scan prosody_brain_dir for per-story JSON files.

        Returns a list of dicts: {"subject": ..., "story": ..., "path": ...}

        In prosody-only mode each story is added only once (first subject found).
        In brain mode every subject × story combination is included.
        """
        records = []
        seen_stories: set = set()  # used only in prosody-only mode

        for fname in sorted(os.listdir(self.prosody_brain_dir)):
            # Match per-story files (skip per-subject aggregate files)
            if not fname.endswith("_prosody+brain-pca.json"):
                continue
            if "_all-stories_" in fname:
                continue

            # Parse subject and story from filename:
            # sub-{subject}_{story_safe}_prosody+brain-pca.json
            stem = fname.replace("_prosody+brain-pca.json", "")
            if not stem.startswith("sub-"):
                continue
            # subject id is everything between "sub-" and the first "_"
            rest = stem[4:]  # strip "sub-"
            parts = rest.split("_", 1)
            if len(parts) < 2:
                continue
            subject, story_safe = parts[0], parts[1]

            # Reverse the safe-story name (underscores may replace spaces/special chars
            # — we keep the safe name as the story identifier to match filenames)
            story = story_safe

            # Apply filters
            if self.subject_filter and subject not in self.subject_filter:
                continue
            if self.story_filter and story not in self.story_filter:
                continue
            if story not in self.trfiles:
                # Try to find the original story name by checking audio dir
                audio_match = self._resolve_story_name(story)
                if audio_match is None:
                    continue
                story = audio_match

            full_path = os.path.join(self.prosody_brain_dir, fname)

            if not self.use_brain_pca:
                # Prosody-only: deduplicate by story
                if story in seen_stories:
                    continue
                seen_stories.add(story)

            records.append({"subject": subject, "story": story, "path": full_path})

        return records

    def _resolve_story_name(self, story_safe: str) -> Optional[str]:
        """
        Try to match a safe story name back to an actual story present in
        self.trfiles, by checking for a corresponding .wav file.
        """
        # Direct match first
        if story_safe in self.trfiles:
            return story_safe
        # Check if a wav file exists with that name
        wav_path = os.path.join(self.audio_dir, f"{story_safe}.wav")
        if os.path.exists(wav_path) and story_safe in self.trfiles:
            return story_safe
        return None

    # ------------------------------------------------------------------
    # Sample building
    # ------------------------------------------------------------------

    def _build_samples(self):
        """
        Load JSON records, z-score audio features (fit scalers if needed),
        extract audio windows, and populate self.samples.
        """
        # ---- Pass 1: collect all raw audio features to fit scalers ----
        raw_story_data: List[Dict] = []  # lightweight cache of loaded JSONs

        for record in self._json_records:
            with open(record["path"], "r", encoding="utf-8") as f:
                jdata = json.load(f)
            raw_story_data.append({"record": record, "jdata": jdata})

        # Fit scalers on audio features if not provided
        if self.scalers is None:
            all_raw: Dict[str, List[float]] = {}
            for entry in raw_story_data:
                feat_names = entry["jdata"]["feature_names"]
                audio_matrix = np.array(
                    entry["jdata"]["audio_features"]["tr_aligned"]["data"],
                    dtype=np.float32,
                )  # (n_TRs, n_feats)
                for fi, fname in enumerate(feat_names):
                    all_raw.setdefault(fname, []).extend(audio_matrix[:, fi].tolist())

            self.scalers = {
                feat: StandardScaler().fit(np.array(vals).reshape(-1, 1))
                for feat, vals in all_raw.items()
            }

        # ---- Pass 2: build actual samples ----
        for entry in raw_story_data:
            record = entry["record"]
            jdata = entry["jdata"]
            story = record["story"]
            subject = record["subject"]

            feat_names: List[str] = jdata["feature_names"]
            audio_matrix = np.array(
                jdata["audio_features"]["tr_aligned"]["data"], dtype=np.float32
            )  # (n_TRs, n_feats)

            # z-score audio features
            audio_scaled = np.column_stack([
                self.scalers[fn].transform(audio_matrix[:, fi].reshape(-1, 1)).ravel()
                for fi, fn in enumerate(feat_names)
            ])  # (n_TRs, n_feats)

            # Brain PCA targets (if applicable)
            if self.use_brain_pca:
                brain_matrix = np.array(
                    jdata["brain_targets"]["pca"]["data"], dtype=np.float32
                )  # (n_TRs, n_pca)
                pca_info = {
                    "n_components": jdata["brain_targets"]["pca"]["n_components"],
                    "explained_variance_ratio": jdata["brain_targets"]["pca"][
                        "explained_variance_ratio"
                    ],
                }
            else:
                brain_matrix = None
                pca_info = None

            n_trs = audio_scaled.shape[0]

            # Load waveform (cached per story to avoid repeated disk reads)
            waveform = self._load_waveform(story)

            # TR times
            tr_info = self.trfiles[story][0]
            tr_times = tr_info.get_reltriggertimes() + tr_info.soundstarttime

            # The audio matrix is already trimmed; align TR times accordingly.
            # Extraction used slice [5+TRIM:-TRIM] — we infer the offset from
            # the difference between len(tr_times) and n_trs.
            offset = (len(tr_times) - n_trs) // 2  # should equal 5+TRIM
            tr_times_trimmed = tr_times[offset: offset + n_trs]

            if len(tr_times_trimmed) != n_trs:
                print(
                    f"Warning: {subject}/{story} — TR time slice ({len(tr_times_trimmed)}) "
                    f"!= audio rows ({n_trs}). Skipping."
                )
                continue

            # Set feature_names once
            if not self.feature_names:
                self.feature_names = feat_names

            for i in range(n_trs):
                tr_time = float(tr_times_trimmed[i])
                start_sample = int(tr_time * self.sampling_rate)
                end_sample = min(
                    start_sample + self.max_length, waveform.shape[1]
                )

                window_np = waveform[0, start_sample:end_sample].numpy()

                inputs = self.processor(
                    window_np,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                )
                input_values = inputs.input_values.squeeze(0)  # (max_length,)

                audio_label = torch.tensor(audio_scaled[i], dtype=torch.float32)

                # Always create both audio_labels and labels for consistency
                if self.use_brain_pca:
                    brain_label = torch.tensor(
                        brain_matrix[i], dtype=torch.float32
                    )
                    combined_labels = torch.cat([audio_label, brain_label], dim=0)
                    sample = {
                        "input_values": input_values,
                        "audio_labels": audio_label,
                        "brain_labels": brain_label,
                        "labels": combined_labels,
                        "story_name": story,
                        "subject": subject,
                        "tr_index": i,
                        "tr_time": tr_time,
                        "feature_names": feat_names,
                        "pca_info": pca_info,
                    }
                else:
                    sample = {
                        "input_values": input_values,
                        "audio_labels": audio_label,
                        "labels": audio_label,
                        "story_name": story,
                        "subject": subject,
                        "tr_index": i,
                        "tr_time": tr_time,
                        "feature_names": feat_names,
                    }

                self.samples.append(sample)

        if not self.samples:
            raise ValueError("Dataset is empty after building samples.")

        # Update feature_names to match actual label dimension
        if self.use_brain_pca and self.samples:
            n_brain_components = self.samples[0]["brain_labels"].shape[0]
            # feature_names already contains OpenSMILE names; append brain PC names
            brain_pc_names = [f"PC_{i+1}" for i in range(n_brain_components)]
            self.feature_names = self.feature_names + brain_pc_names

        mode = "brain PCA" if self.use_brain_pca else "prosody-only"
        n_stories = len({s["story_name"] for s in self.samples})
        n_subjects = len({s["subject"] for s in self.samples})
        print(
            f"ProsodyDataset [{mode}] — "
            f"{len(self.samples)} samples | "
            f"{n_stories} stories | "
            f"{n_subjects} subject(s)"
        )

    # ------------------------------------------------------------------
    # Audio loading helper
    # ------------------------------------------------------------------

    def _load_waveform(self, story: str) -> torch.Tensor:
        """Load, mono-mix and resample a story waveform. Not cached — call once per story."""
        audio_path = os.path.join(self.audio_dir, f"{story}.wav")
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sampling_rate
            )
            waveform = resampler(waveform)
        return waveform

    # ------------------------------------------------------------------
    # Standard Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]
        # Build return dict, including only keys that are actually present in the sample
        # This prevents the HF Trainer from filtering out None values
        result = {
            "input_values": item["input_values"],
        }
        
        # Always include labels and audio_labels if they exist
        if "labels" in item:
            result["labels"] = item["labels"]
        if "audio_labels" in item:
            result["audio_labels"] = item["audio_labels"]
        if "brain_labels" in item:
            result["brain_labels"] = item["brain_labels"]
        
        # Include metadata
        result.update({
            "story_name": item.get("story_name"),
            "subject": item.get("subject"),
            "tr_index": item.get("tr_index"),
            "tr_time": item.get("tr_time"),
            "feature_names": item.get("feature_names"),
            "pca_info": item.get("pca_info"),
        })
        
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def label_dim(self) -> int:
        """Dimensionality of the primary label vector."""
        return self.samples[0]["labels"].shape[0] if self.samples else 0

    @property
    def audio_label_dim(self) -> int:
        return self.samples[0]["audio_labels"].shape[0] if self.samples else 0

    @property
    def brain_label_dim(self) -> int:
        if self.use_brain_pca and self.samples:
            return self.samples[0]["brain_labels"].shape[0]
        return 0

    def get_label_names(self) -> List[str]:
        if self.use_brain_pca:
            dim = self.brain_label_dim
            return [f"PC_{i+1}" for i in range(dim)]
        return list(self.feature_names)

    def get_fitted_scalers(self) -> Dict[str, StandardScaler]:
        """Return the fitted audio-feature scalers (useful for train/val sharing)."""
        return self.scalers
