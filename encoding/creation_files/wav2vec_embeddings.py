# -*- coding: utf-8 -*-
"""
Clean & modular script for extracting prosody-aware wav2vec2 embeddings
- Mean-pooled last layer (baseline)
- Mean-pooled average over a user-defined set of layers (e.g. fine-tuned layers)

Supports wav2vec2-large (24 transformer layers, indices 0–23)
"""

import os
from os.path import dirname, join
import json
import logging
import numpy as np
import torch
import torchaudio
import h5py

# Make sure repo root is in path
REPO_DIR = join(dirname(dirname(dirname(os.path.abspath(__file__)))))
import sys
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from encoding.utils.ridge_utils.stimulus_utils import load_simulated_trfiles

# ============================= CONFIGURATION =============================

SAMPLING_RATE    = 16000
WINDOW_SIZE_SEC  = 2.0
WINDOW_SAMPLES   = int(WINDOW_SIZE_SEC * SAMPLING_RATE)

BASE_OUTPUT_DIR  = "features"

# === Choose which layers to average ===
# Common choices:
# - Fine-tuned layers only:   range(6, 24)   → layers 6–23
# - Late layers (prosody):    range(18, 24)  → layers 18–23
# - Middle + late:            range(9, 24)
# - All transformer layers:   range(0, 24)

LAYER_RANGE_TO_AVERAGE = range(6, 24)          # ← edit this line to choose layers

# Derived names (used in folders & logs)
layer_str = f"layers{LAYER_RANGE_TO_AVERAGE.start}to{LAYER_RANGE_TO_AVERAGE.stop-1}"
print(f"Selected layers to average: {layer_str} ({len(LAYER_RANGE_TO_AVERAGE)} layers)")

MODEL_PATH       = "finetuning/model_output/layers_frozen_18/final_model"
PROCESSOR_NAME   = "facebook/wav2vec2-large-960h"

# Output subdirectories
OUT_LAST_LAYER   = join(BASE_OUTPUT_DIR, "wav2vec_mean")
OUT_SELECTED     = join(BASE_OUTPUT_DIR, f"wav2vec_mean_{layer_str}")
# -------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_model_and_processor():
    """Load fine-tuned model and matching processor"""
    processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_PATH)
    model.to(device).eval()
    
    n_layers = model.config.num_hidden_layers
    logging.info(f"Loaded fine-tuned wav2vec2-large – {n_layers} transformer layers")
    logging.info(f"Device: {device}")
    
    return model, processor


def extract_mean_last_layer(audio_chunk: torch.Tensor, processor, model) -> np.ndarray:
    """Mean-pool the final hidden state (classic wav2vec2 baseline)"""
    inputs = processor(
        audio_chunk.squeeze(0).cpu().numpy(),
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
        padding="max_length",
        max_length=WINDOW_SAMPLES,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
    return embedding.cpu().numpy().astype(np.float32)


def extract_mean_selected_layers(audio_chunk: torch.Tensor, processor, model,
                                layer_indices: range) -> np.ndarray:
    """Average mean-pooled hidden states from the selected transformer layers"""
    inputs = processor(
        audio_chunk.squeeze(0).cpu().numpy(),
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
        padding="max_length",
        max_length=WINDOW_SAMPLES,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    layer_embeddings = []
    for layer_idx in layer_indices:
        # hidden_states[0]  = CNN output
        # hidden_states[1]  = after transformer layer 0
        # hidden_states[24] = after transformer layer 23
        hidden = outputs.hidden_states[layer_idx + 1]
        mean_pooled = hidden.mean(dim=1).squeeze(0)          # [1024]
        layer_embeddings.append(mean_pooled)

    # Average across chosen layers
    final_emb = torch.stack(layer_embeddings).mean(dim=0)
    return final_emb.cpu().numpy().astype(np.float32)


def extract_features_from_window(audio_chunk: torch.Tensor, processor, model,
                                selected_layers: range) -> tuple[np.ndarray, np.ndarray]:
    """Extract both feature types from one 2-second window"""
    last_emb   = extract_mean_last_layer(audio_chunk, processor, model)
    select_emb = extract_mean_selected_layers(audio_chunk, processor, model, selected_layers)
    return last_emb, select_emb


def extract_story_features(waveform: torch.Tensor, tr_times: list[float],
                          processor, model, selected_layers: range) -> tuple[np.ndarray, np.ndarray]:
    """Process entire story – one embedding vector per TR"""
    last_features   = []
    select_features = []

    for tr_start_sec in tr_times:
        start_sample = int(tr_start_sec * SAMPLING_RATE)
        end_sample   = start_sample + WINDOW_SAMPLES

        if end_sample <= waveform.shape[1]:
            chunk = waveform[:, start_sample:end_sample]
        else:
            # pad if story ends before full window
            to_pad = WINDOW_SAMPLES - (waveform.shape[1] - start_sample)
            chunk = torch.nn.functional.pad(
                waveform[:, start_sample:], (0, to_pad)
            )

        chunk = chunk.to(device)

        last_emb, select_emb = extract_features_from_window(
            chunk, processor, model, selected_layers
        )

        last_features.append(last_emb)
        select_features.append(select_emb)

    return np.stack(last_features), np.stack(select_features)


def main():
    model, processor = load_model_and_processor()

    # Load timing info
    with open("ds003020/derivative/respdict.json") as f:
        respdict = json.load(f)

    trfiles = load_simulated_trfiles(respdict, tr=2.0, start_time=10.0, pad=5)

    # Discover stories from TextGrids (assuming naming convention)
    textgrid_dir = "ds003020/derivative/TextGrids"
    audio_dir    = "stimuli_16k"

    stories = sorted(
        f[:-9] for f in os.listdir(textgrid_dir) if f.endswith(".TextGrid")
    )

    os.makedirs(OUT_LAST_LAYER, exist_ok=True)
    os.makedirs(OUT_SELECTED,   exist_ok=True)

    for story in stories:
        logging.info(f"Processing story: {story}")

        audio_path = join(audio_dir, f"{story}.wav")
        waveform, sr = torchaudio.load(audio_path)

        # mono + resample if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != SAMPLING_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLING_RATE)(waveform)

        # Get TR onsets
        if story not in trfiles:
            logging.warning(f"Skipping {story} – no TR timing info")
            continue

        tr_info = trfiles[story][0]
        tr_times = tr_info.get_reltriggertimes() + tr_info.soundstarttime

        # Extract features
        last_feats, selected_feats = extract_story_features(
            waveform, tr_times, processor, model, LAYER_RANGE_TO_AVERAGE
        )

        # Save
        with h5py.File(join(OUT_LAST_LAYER, f"{story}.h5"), "w") as f:
            f.create_dataset("data", data=last_feats)

        with h5py.File(join(OUT_SELECTED, f"{story}.h5"), "w") as f:
            f.create_dataset("data", data=selected_feats)

        logging.info(
            f"Saved {story} | "
            f"last: {last_feats.shape} | "
            f"{layer_str}: {selected_feats.shape}"
        )

    logging.info("All stories processed. Done.")


if __name__ == "__main__":
    main()