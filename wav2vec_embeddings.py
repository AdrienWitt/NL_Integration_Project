# -*- coding: utf-8 -*-
"""
FINAL CLEAN SCRIPT – prosody-aware wav2vec2 embeddings (NO ATTENTION)

Extracted features:
  • Mean-pooled LAST layer (baseline)
  • Mean-pooled LAYERS 18–23 (prosody-rich stack)

Optimized for your fine-tuned wav2vec2-large (24 layers).
"""

import os
import json
import numpy as np
import torch
import torchaudio
import h5py
import logging
from os.path import join

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from encoding.ridge_utils.stimulus_utils import load_simulated_trfiles

# ============================= CONFIG =============================
SAMPLING_RATE = 16000
WINDOW_SIZE = 2.0
WINDOW_SAMPLES = int(WINDOW_SIZE * SAMPLING_RATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = "features_new"
OUT_LAST = join(BASE_DIR, "wav2vec_mean")
OUT_18_23 = join(BASE_DIR, "wav2vec_mean_layers18to23")

os.makedirs(OUT_LAST, exist_ok=True)
os.makedirs(OUT_18_23, exist_ok=True)

MODEL_PATH = "finetuning_results/layers_frozen_6/final_model"
PROCESSOR_NAME = "facebook/wav2vec2-large-960h"

BEST_LAYER_INDICES = list(range(18, 24))  # 18–23

# ================================================================

def load_model_and_processor():
    processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_PATH)
    model.to(device).eval()

    n_layers = model.config.num_hidden_layers
    logging.info(f"Loaded fine-tuned wav2vec2 with {n_layers} transformer layers")
    return model, processor


# ------------------------------------------------
# Feature extractors (MEAN ONLY)
# ------------------------------------------------

def mean_last_layer(audio_window, processor, model):
    inputs = processor(
        audio_window.squeeze(0).cpu().numpy(),
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=WINDOW_SAMPLES
    ).to(device)

    with torch.no_grad():
        out = model(**inputs)

    return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)


def mean_layers_18_to_23(audio_window, processor, model):
    inputs = processor(
        audio_window.squeeze(0).cpu().numpy(),
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=WINDOW_SAMPLES
    ).to(device)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    layer_means = []
    for idx in BEST_LAYER_INDICES:
        hidden = out.hidden_states[idx + 1]   # skip CNN output
        layer_means.append(hidden.mean(dim=1).squeeze(0))

    # 🔑 AVERAGE ACROSS LAYERS
    avg_embedding = torch.stack(layer_means).mean(dim=0)

    return avg_embedding.cpu().numpy().astype(np.float32)  # (1024,)


# ------------------------------------------------
# Story processing
# ------------------------------------------------

def process_story(waveform, tr_times, processor, model):
    last_layer_feats = []
    multi_layer_feats = []

    for tr in tr_times:
        start = int(tr * SAMPLING_RATE)
        end = start + WINDOW_SAMPLES

        if end <= waveform.shape[1]:
            win = waveform[:, start:end]
        else:
            win = torch.nn.functional.pad(
                waveform[:, start:], (0, WINDOW_SAMPLES - waveform.shape[1] + start)
            )

        win = win.to(device)

        last_layer_feats.append(mean_last_layer(win, processor, model))
        multi_layer_feats.append(mean_layers_18_to_23(win, processor, model))

    return np.stack(last_layer_feats), np.stack(multi_layer_feats)


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():
    model, processor = load_model_and_processor()

    with open("ds003020/derivative/respdict.json") as f:
        respdict = json.load(f)

    trfiles = load_simulated_trfiles(respdict, tr=2.0, start_time=10.0, pad=5)

    textgrid_dir = "ds003020/derivative/TextGrids"
    audio_dir = "stimuli_16k"
    stories = sorted(f[:-9] for f in os.listdir(textgrid_dir) if f.endswith(".TextGrid"))

    for story in stories:
        logging.info(f"Processing {story}")

        waveform, sr = torchaudio.load(join(audio_dir, f"{story}.wav"))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != SAMPLING_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLING_RATE)(waveform)

        tr_times = trfiles[story][0].get_reltriggertimes() + trfiles[story][0].soundstarttime

        last_feats, multi_feats = process_story(waveform, tr_times, processor, model)

        with h5py.File(join(OUT_LAST, f"{story}.hf5"), "w") as f:
            f.create_dataset("data", data=last_feats)

        with h5py.File(join(OUT_18_23, f"{story}.hf5"), "w") as f:
            f.create_dataset("data", data=multi_feats)

        logging.info(f"Saved {story}: last {last_feats.shape}, layers18–23 {multi_feats.shape}")

    logging.info("DONE")


if __name__ == "__main__":
    main()
