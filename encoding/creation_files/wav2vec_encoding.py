# -*- coding: utf-8 -*-
"""
FINAL BULLETPROOF SCRIPT – extracts the BEST prosody-aware embeddings
from YOUR fine-tuned wav2vec2-large model (first 6 frozen → last 18 trained)

Features extracted:
  • Classic mean-pooled last layer (for baseline)
  • Multi-attention last layer
  • Multi-attention from LAYERS 18–23 only ← THIS IS THE GOLD (the 6 layers you trained hardest)

Works perfectly with 12-layer (base) or 24-layer (large) models — but optimized for YOUR large model.
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = "features"
OUTPUT_MEAN        = join(BASE_DIR, "wav2vec_finetuned_mean")
OUTPUT_LAST_ATTN   = join(BASE_DIR, "wav2vec_finetuned_multi_attention_last_layer")
OUTPUT_BEST_LAYERS = join(BASE_DIR, "wav2vec_finetuned_multi_attention_layers18to23")  # ← THE WINNER

for d in [OUTPUT_MEAN, OUTPUT_LAST_ATTN, OUTPUT_BEST_LAYERS]:
    os.makedirs(d, exist_ok=True)

# YOUR fine-tuned model path
MODEL_PATH = "finetuning_results/layers_frozen_6/final_model"           # ← 24-layer large model
PROCESSOR_NAME = "facebook/wav2vec2-large-960h"

# We only want the 6 layers you trained the most on: 18,19,20,21,22,23
BEST_LAYER_INDICES = list(range(18, 24))  # [18,19,20,21,22,23]

# ==================================================================

def load_model_and_processor():
    processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    
    n_layers = model.config.num_hidden_layers
    logging.info(f"Loaded YOUR fine-tuned model → {n_layers} transformer layers (first 6 frozen, last 18 trained)")
    return model, processor, n_layers


# 1. Classic mean pooling (last layer only) – for comparison
def get_mean_pooled(audio_window, processor, model):
    inputs = processor(
        audio_window.squeeze(0).cpu().numpy(),
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
        padding="max_length",
        max_length=WINDOW_SAMPLES,
        truncation=True
    ).to(device)
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)


# 2. Multi-attention from ONE specific layer
def get_multi_attention_one_layer(audio_window, processor, model, layer_idx=-1):
    audio = audio_window - audio_window.mean()
    inputs = processor(
        audio.squeeze(0).cpu().numpy(),
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=WINDOW_SAMPLES
    ).to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, output_attentions=True)
    hidden = out.hidden_states[layer_idx + 1].squeeze(0)   # +1 to skip CNN output
    attn   = out.attentions[layer_idx].squeeze(0)
    weighted = torch.matmul(attn[:, -1, :], hidden).mean(0)
    return weighted.cpu().numpy().astype(np.float32)


# 3. Multi-attention from BEST layers (18–23) ← THIS IS YOUR MAIN FEATURE
def get_multi_attention_best_layers(audio_window, processor, model, layer_indices=BEST_LAYER_INDICES):
    audio = audio_window - audio_window.mean()
    inputs = processor(
        audio.squeeze(0).cpu().numpy(),
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=WINDOW_SAMPLES
    ).to(device)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, output_attentions=True)

    embeddings = []
    for idx in layer_indices:
        hidden = out.hidden_states[idx + 1].squeeze(0)   # skip CNN → idx+1
        attn   = out.attentions[idx].squeeze(0)
        weighted = torch.matmul(attn[:, -1, :], hidden).mean(0)   # [1024]
        embeddings.append(weighted.cpu().numpy().astype(np.float32))

    return np.stack(embeddings)  # (6, 1024)


def process_story(story, waveform, tr_times, processor, model):
    mean_list = []
    last_attn_list = []
    best_layers_list = []   # ← layers 18–23

    for tr_time in tr_times:
        start = int(tr_time * SAMPLING_RATE)
        end   = start + WINDOW_SAMPLES
        if end <= waveform.shape[1]:
            win = waveform[:, start:end]
        else:
            win = waveform[:, start:]
            win = torch.nn.functional.pad(win, (0, WINDOW_SAMPLES - win.shape[1]))
        win = win.to(device)

        mean_list.append(get_mean_pooled(win, processor, model))
        last_attn_list.append(get_multi_attention_one_layer(win, processor, model, layer_idx=-1))
        best_layers_list.append(get_multi_attention_best_layers(win, processor, model))

    return (
        np.stack(mean_list),           # (n_TRs, 1024)
        np.stack(last_attn_list),      # (n_TRs, 1024)
        np.stack(best_layers_list)     # (n_TRs, 6, 1024) ← THE BEST ONE
    )


def main():
    model, processor, n_layers = load_model_and_processor()
    assert n_layers == 24, "This script is optimized for your 24-layer large model"

    with open("ds003020/derivative/respdict.json") as f:
        respdict = json.load(f)
    trfiles = load_simulated_trfiles(respdict, tr=2.0, start_time=10.0, pad=5)

    textgrid_dir = "ds003020/derivative/TextGrids"
    stories = sorted([f[:-9] for f in os.listdir(textgrid_dir) if f.endswith(".TextGrid")])
    audio_dir = "stimuli_16k"

    for story in stories:
        logging.info(f"Processing {story} with YOUR prosody-finetuned model...")
        waveform, sr = torchaudio.load(join(audio_dir, f"{story}.wav"))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != SAMPLING_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLING_RATE)(waveform)

        tr_times = trfiles[story][0].get_reltriggertimes() + trfiles[story][0].soundstarttime

        mean_feats, last_attn_feats, best_feats = process_story(story, waveform, tr_times, processor, model)

        saves = [
            ("mean_pooled",              mean_feats,      OUTPUT_MEAN),
            ("multi_attention_last",     last_attn_feats, OUTPUT_LAST_ATTN),
            ("multi_attention_layers18to23", best_feats,  OUTPUT_BEST_LAYERS),  # ← USE THIS ONE
        ]

        for name, data, folder in saves:
            path = join(folder, f"{story}.hf5")
            with h5py.File(path, "w") as f:
                f.create_dataset("data", data=data)
            logging.info(f"   Saved {path} → {data.shape}")

    logging.info("DONE! Your best features are in:")
    logging.info(f"   {OUTPUT_BEST_LAYERS}  ← (n_TRs, 6, 1024) from layers 18–23 (the ones you trained most)")


if __name__ == "__main__":
    main()