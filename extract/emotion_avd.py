"""
Extract arousal / dominance / valence per TR from the audEERING emotion model.

Gives a very low-dimensional, highly interpretable prosody band: 3 numbers per
TR describing the affective state of the speech, from a model trained on
MSP-Podcast. Same TR-aligned windows and same output format as every other
band, so it drops straight into the encoding models.

Why this band is worth having
-----------------------------
It is the opposite extreme from the 1024-d wav2vec2 band. If 3 affective
dimensions predict a voxel about as well as 1024 learned features do, that
voxel is tracking something close to affective prosody rather than fine
acoustic detail — a much stronger claim than "the audio band predicts it".
With banded ridge the dimensionality difference is handled properly, so the
comparison is fair.

`--output` choices:
  ``avd``    3-d  arousal, dominance, valence  (the regression head's output)
  ``hidden`` 1024-d mean-pooled encoder states (what the head reads)
  ``both``   1027-d concatenation

Usage
-----
    python -m extract.emotion_avd --output avd --out-name emotion_avd
    python -m extract.emotion_avd --output hidden --out-name emotion_hidden
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from config import FEATURES_DIR, SAMPLING_RATE, STIMULI_16K_DIR, ensure_dirs
from common.tr_alignment import load_trfiles, tr_onsets
from finetune import EMOTION_DIMENSIONS, EMOTION_MODEL

log = logging.getLogger("extract.emotion")


class RegressionHead(nn.Module):
    """audEERING's head: dense -> tanh -> out_proj, matching the checkpoint."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


def build_emotion_model(model_name: str):
    """Load the checkpoint with its regression head attached.

    The weights are laid out as ``wav2vec2.*`` plus ``classifier.*``. Declaring
    `base_model_prefix = "wav2vec2"` is what lets `from_pretrained` route the
    encoder weights correctly; without it the encoder loads randomly
    initialised and the predicted AVD values are meaningless.
    """
    from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class EmotionModel(Wav2Vec2PreTrainedModel):
        base_model_prefix = "wav2vec2"

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.init_weights()

        def forward(self, input_values, attention_mask=None):
            outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
            pooled = torch.mean(outputs[0], dim=1)
            return pooled, self.classifier(pooled)

    model = EmotionModel.from_pretrained(model_name)
    return model


@torch.no_grad()
def embed_story(waveform: torch.Tensor, onsets: np.ndarray, processor, model,
                device, window_samples: int, output: str) -> np.ndarray:
    rows = []
    for onset in onsets:
        start = int(onset * SAMPLING_RATE)
        end = start + window_samples
        if end <= waveform.shape[1]:
            chunk = waveform[:, start:end]
        else:
            pad = window_samples - max(0, waveform.shape[1] - start)
            chunk = torch.nn.functional.pad(waveform[:, start:], (0, pad))

        inputs = processor(
            chunk.squeeze(0).numpy(), sampling_rate=SAMPLING_RATE,
            return_tensors="pt", padding="max_length",
            max_length=window_samples, truncation=True,
        )
        input_values = inputs.input_values.to(device)
        pooled, logits = model(input_values)

        pooled = pooled.squeeze(0).cpu().numpy()
        logits = logits.squeeze(0).cpu().numpy()

        if output == "avd":
            rows.append(logits)
        elif output == "hidden":
            rows.append(pooled)
        else:
            rows.append(np.concatenate([logits, pooled]))

    return np.stack(rows).astype(np.float32)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default=EMOTION_MODEL,
                   help="emotion checkpoint (default: the audEERING AVD model)")
    p.add_argument("--output", default="avd", choices=["avd", "hidden", "both"])
    p.add_argument("--audio-dir", default=str(STIMULI_16K_DIR))
    p.add_argument("--out-name", default=None, help="default: emotion_<output>")
    p.add_argument("--stories", default=None)
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    from transformers import AutoFeatureExtractor
    from config import WINDOW_SIZE_SEC

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_emotion_model(args.model).to(device).eval()
    processor = AutoFeatureExtractor.from_pretrained(args.model)

    n_labels = model.config.num_labels
    if args.output in ("avd", "both") and n_labels != len(EMOTION_DIMENSIONS):
        raise ValueError(
            f"{args.model} predicts {n_labels} labels, expected "
            f"{len(EMOTION_DIMENSIONS)} ({', '.join(EMOTION_DIMENSIONS)}). "
            f"Use --output hidden, or point --model at the AVD checkpoint."
        )

    out_dir = Path(FEATURES_DIR) / (args.out_name or f"emotion_{args.output}")
    ensure_dirs(out_dir)

    log.info(f"Model  : {args.model}")
    log.info(f"Encoder: {model.config.num_hidden_layers} layers, "
             f"hidden {model.config.hidden_size}")
    log.info(f"Output : {args.output} "
             f"({'/'.join(EMOTION_DIMENSIONS)} in this order)")
    log.info(f"Target : {out_dir}")

    trfiles = load_trfiles()
    audio_dir = Path(args.audio_dir)
    window_samples = int(WINDOW_SIZE_SEC * SAMPLING_RATE)

    if args.stories:
        stories = [s.strip() for s in args.stories.split(",") if s.strip()]
    else:
        stories = sorted(q.stem for q in audio_dir.glob("*.wav"))

    import torchaudio
    done = skipped = 0
    for story in stories:
        out_path = out_dir / f"{story}.hf5"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        if story not in trfiles or not (audio_dir / f"{story}.wav").exists():
            log.warning(f"  {story}: no TR timing or wav, skipping")
            skipped += 1
            continue

        waveform, sr = torchaudio.load(str(audio_dir / f"{story}.wav"))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != SAMPLING_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLING_RATE)(waveform)

        features = embed_story(waveform, tr_onsets(story, trfiles), processor,
                               model, device, window_samples, args.output)
        with h5py.File(out_path, "w") as f:
            f.create_dataset("data", data=features)
        log.info(f"  {story}: {features.shape} -> {out_path.name}")
        done += 1

    if args.output in ("avd", "both"):
        names = list(EMOTION_DIMENSIONS)
        if args.output == "both":
            names += [f"hidden_{i}" for i in range(model.config.hidden_size)]
        with open(out_dir / "feature_names.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(names))

    log.info(f"Done: {done} written, {skipped} skipped.")


if __name__ == "__main__":
    main()
