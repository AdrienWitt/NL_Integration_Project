"""
TR-aligned windows of audio paired with prosody targets.

One sample = one TR: the `WINDOW_SIZE_SEC` of audio starting at that TR's
onset, labelled with the eGeMAPS functionals of that same window.

Targets come from the per-story JSONs `prep/make_finetune_targets.py` writes,
so this class never touches the fMRI files. Older JSONs also carry a
``brain_targets`` block from the removed multi-task path; it is ignored.

Scaling
-------
Prosody features are z-scored with scalers fitted on the *training* stories
only, and `get_fitted_scalers()` hands them to the validation set. Refitting on
validation would leak the validation distribution into the model's targets and
quietly inflate the reported metrics.

Alignment
---------
The extraction script slices its audio matrix to ``[TR_PAD + trim : -trim]``
while the TR onset grid from `load_trfiles` starts at 0. The onsets are
therefore advanced by ``TR_PAD + trim`` before windows are cut. Getting this
offset wrong shifts every label by a fixed number of TRs, which shows up as
uniformly poor but not obviously broken training — hence the explicit check.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torchaudio
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor

from config import (EGEMAPS_N_FUNCTIONALS, SAMPLING_RATE, TR, TR_PAD,
                    WINDOW_SIZE_SEC)
from common.tr_alignment import load_trfiles, tr_onsets

#: Filename suffix written by prep/make_finetune_targets.py. The "+brain-pca"
#: is historical — the JSONs on disk still carry that name and an unused
#: brain_targets block; only the audio half is read.
TARGET_SUFFIX = "_prosody+brain-pca-avg.json"


class ProsodyDataset(Dataset):
    """Audio windows with per-TR eGeMAPS prosody targets.

    Parameters
    ----------
    audio_dir : str
        Directory of ``<story>.wav`` at `SAMPLING_RATE`.
    target_dir : str
        Output of `prep/make_finetune_targets.py`; its ``averaged/`` subfolder
        holds one JSON per story.
    processor : AutoFeatureExtractor
        Feature extractor of the base speech model.
    story_names : list of str, optional
        Restrict to these stories (the train or val side of the split).
    scalers : dict, optional
        Pre-fitted ``{feature_name: StandardScaler}``. Pass the training set's
        scalers when building the validation set.
    trim : int
        Must match the `TRIM` used during target extraction.
    expect_n_features : int or None
        Fail fast unless the targets have exactly this many prosody features.
        Defaults to the 88 eGeMAPSv02 functionals. Pass None to skip the check
        when deliberately training on a reduced feature set.
    """

    def __init__(
        self,
        audio_dir: str,
        target_dir: str,
        processor: AutoFeatureExtractor,
        story_names: Optional[List[str]] = None,
        scalers: Optional[Dict[str, StandardScaler]] = None,
        trim: int = 5,
        expect_n_features: Optional[int] = EGEMAPS_N_FUNCTIONALS,
        tr: float = TR,
        sampling_rate: int = SAMPLING_RATE,
        window_size_sec: float = WINDOW_SIZE_SEC,
    ):
        self.audio_dir = audio_dir
        self.target_dir = target_dir
        self.processor = processor
        self.story_filter = set(story_names) if story_names else None
        self.scalers = scalers
        self.trim = trim
        self.expect_n_features = expect_n_features
        self.sampling_rate = sampling_rate
        self.window_size_sec = window_size_sec
        self.max_length = int(window_size_sec * sampling_rate)

        self.trfiles = load_trfiles(tr=tr, pad=TR_PAD)

        self.records = self._discover()
        if not self.records:
            raise ValueError(
                f"No target JSONs matched under {target_dir}/averaged "
                f"(story filter: {sorted(self.story_filter) if self.story_filter else 'none'})"
            )

        self.feature_names: List[str] = []
        self.samples: List[Dict] = []
        self._build()

    # -- discovery ----------------------------------------------------------

    def _discover(self) -> List[Dict[str, str]]:
        avg_dir = os.path.join(self.target_dir, "averaged")
        if not os.path.isdir(avg_dir):
            raise FileNotFoundError(
                f"{avg_dir} not found — run prep/make_finetune_targets.py first"
            )

        records = []
        for fname in sorted(os.listdir(avg_dir)):
            if not fname.endswith(TARGET_SUFFIX):
                continue
            story = fname[: -len(TARGET_SUFFIX)]
            if self.story_filter and story not in self.story_filter:
                continue
            if story not in self.trfiles:
                print(f"  skipping {story}: no TR timing")
                continue
            records.append({"story": story, "path": os.path.join(avg_dir, fname)})
        return records

    # -- building -----------------------------------------------------------

    def _load_waveform(self, story: str) -> torch.Tensor:
        path = os.path.join(self.audio_dir, f"{story}.wav")
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sampling_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sampling_rate
            )(waveform)
        return waveform

    def _fit_scalers(self, loaded: List[Dict]) -> Dict[str, StandardScaler]:
        columns: Dict[str, List[float]] = {}
        for entry in loaded:
            names = entry["data"]["feature_names"]
            matrix = np.asarray(
                entry["data"]["audio_features"]["tr_aligned"]["data"],
                dtype=np.float32,
            )
            for i, name in enumerate(names):
                columns.setdefault(name, []).extend(matrix[:, i].tolist())
        return {
            name: StandardScaler().fit(np.asarray(values).reshape(-1, 1))
            for name, values in columns.items()
        }

    def _build(self) -> None:
        loaded = []
        for record in self.records:
            with open(record["path"], encoding="utf-8") as f:
                loaded.append({"record": record, "data": json.load(f)})

        if self.scalers is None:
            self.scalers = self._fit_scalers(loaded)

        offset = TR_PAD + self.trim

        for entry in loaded:
            data = entry["data"]
            story = entry["record"]["story"]

            names = data["feature_names"]
            audio = np.asarray(
                data["audio_features"]["tr_aligned"]["data"], dtype=np.float32
            )
            n_trs = audio.shape[0]

            missing = [n for n in names if n not in self.scalers]
            if missing:
                raise KeyError(
                    f"{story}: no scaler for {missing[:5]} — the training set "
                    f"was built from targets with different feature names."
                )

            audio_z = np.column_stack([
                self.scalers[name].transform(audio[:, i].reshape(-1, 1)).ravel()
                for i, name in enumerate(names)
            ])

            onsets = tr_onsets(story, self.trfiles)[offset: offset + n_trs]
            if len(onsets) != n_trs:
                # Raise rather than skip. A wrong --trim shifts every label by
                # a fixed number of TRs and drops whole stories, which used to
                # print and continue — leaving a model trained on whatever
                # survived, with the evidence scrolled off the screen.
                raise ValueError(
                    f"{story}: only {len(onsets)} TR onsets available after "
                    f"offset {offset} (TR_PAD={TR_PAD} + trim={self.trim}), "
                    f"but the targets have {n_trs} TRs. --trim must match the "
                    f"TRIM used by prep/make_finetune_targets.py."
                )

            if not self.feature_names:
                self.feature_names = list(names)
                if (self.expect_n_features is not None
                        and len(names) != self.expect_n_features):
                    raise ValueError(
                        f"{story}: targets carry {len(names)} prosody features "
                        f"but {self.expect_n_features} were expected "
                        f"(eGeMAPSv02 functionals). Re-run "
                        f"prep/make_finetune_targets.py, or pass "
                        f"expect_n_features=None to train on a reduced set."
                    )
            elif list(names) != self.feature_names:
                # Otherwise this surfaces much later as a torch.stack failure
                # inside the collator, with nothing pointing at the story.
                raise ValueError(
                    f"{story}: feature names differ from the first story "
                    f"({len(names)} vs {len(self.feature_names)} features). "
                    f"All target JSONs must come from one "
                    f"prep/make_finetune_targets.py run."
                )

            waveform = self._load_waveform(story)

            for i in range(n_trs):
                start = int(float(onsets[i]) * self.sampling_rate)
                end = min(start + self.max_length, waveform.shape[1])
                window = waveform[0, start:end].numpy()

                inputs = self.processor(
                    window, sampling_rate=self.sampling_rate,
                    return_tensors="pt", padding="max_length",
                    max_length=self.max_length, truncation=True,
                )
                self.samples.append({
                    "input_values": inputs.input_values.squeeze(0),
                    "labels": torch.tensor(audio_z[i], dtype=torch.float32),
                    "story_name": story,
                    "tr_index": i,
                    "tr_time": float(onsets[i]),
                })

        if not self.samples:
            raise ValueError("Dataset is empty after building samples")

        self.label_names = list(self.feature_names)

        n_stories = len({s["story_name"] for s in self.samples})
        if n_stories != len(self.records):
            raise ValueError(
                f"built windows for {n_stories} stories but {len(self.records)} "
                f"target files matched — stories were dropped silently"
            )
        print(f"  ProsodyDataset: {len(self.samples):,} windows "
              f"from {n_stories} stories, label dim {self.label_dim}")

    # -- Dataset interface --------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]

    # -- introspection ------------------------------------------------------

    @property
    def label_dim(self) -> int:
        return self.samples[0]["labels"].shape[0] if self.samples else 0

    def get_fitted_scalers(self) -> Dict[str, StandardScaler]:
        """Training scalers, to be reused for validation without refitting."""
        return self.scalers
