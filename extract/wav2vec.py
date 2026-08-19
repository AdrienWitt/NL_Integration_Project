"""
Extract TR-aligned wav2vec2 / HuBERT / WavLM features from the stimuli.

One vector per TR: the `WINDOW_SIZE_SEC` window starting at that TR's onset is
passed through the encoder and pooled over time. Output is one
``<story>.hf5`` per story with a ``data`` dataset of shape (n_TRs, hidden_size)
— the format `common.io.load_features` expects.

Checkpoint handling
-------------------
A directory saved by `finetune.training.train_model` holds an
`AudioEncoderForProsody`, whose weights are nested under an ``encoder.``
prefix. Loading such a directory straight into
`Wav2Vec2Model` does *not* line those keys up: the fine-tuned weights are
dropped and randomly initialised ones silently take their place, so the
"fine-tuned" features are nothing of the sort. This script detects our own
checkpoints from their config and unwraps `.encoder` explicitly. Pass
``--strict`` to abort on any unexpected/missing key instead of warning.

Layer choice
------------
`--layers last` takes the final hidden state. `--layers 18-23` averages the
mean-pooled activations of transformer layers 18..23. Later layers of a speech
model are more phonetic/prosodic and less acoustic; which range transfers best
is empirical, so the flag exists to be swept.

Examples
--------
    python -m extract.wav2vec --layers last --out-name wav2vec_pretrained
    python -m extract.wav2vec \\
        --model-path results/finetune/wav2vec2_robust_frozen_12_lr3e-05_seed42/final_model \\
        --layers 18-23 --out-name wav2vec_ft_robust_18to23
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch
import torchaudio

from config import (FEATURES_DIR, SAMPLING_RATE, STIMULI_16K_DIR,
                    WINDOW_SIZE_SEC, ensure_dirs)
from common.tr_alignment import load_trfiles, tr_onsets
from finetune import REGISTRY, format_registry, resolve_model

log = logging.getLogger("extract.wav2vec")


def parse_layers(spec: str) -> Optional[List[int]]:
    """'last' -> None ; '18-23' -> [18..23] ; '6,12,18' -> [6,12,18]."""
    spec = spec.strip().lower()
    if spec in ("last", "final"):
        return None
    if spec == "auto":
        return "auto"
    if "-" in spec:
        start, stop = spec.split("-")
        return list(range(int(start), int(stop) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def load_encoder(model_path: str, strict: bool = False):
    """Return a bare speech encoder, unwrapping our fine-tuned wrappers."""
    from transformers import AutoModel

    config_file = Path(model_path) / "config.json"
    is_local_checkpoint = config_file.exists()
    # "num_prosody_features" is the removed multi-task wrapper; older
    # checkpoints are still recognised and unwrapped the same way.
    wrapper_keys = {"num_features", "num_prosody_features"}

    if is_local_checkpoint:
        with open(config_file, encoding="utf-8") as f:
            raw_config = json.load(f)

        if wrapper_keys & set(raw_config):
            from finetune.models import AudioEncoderForProsody
            cls = AudioEncoderForProsody
            log.info(f"Fine-tuned checkpoint detected -> {cls.__name__}; "
                     f"unwrapping .encoder")
            base_name = raw_config.get("base_model_name")
            if base_name is None:
                raise ValueError(
                    f"{config_file} has no 'base_model_name'; cannot rebuild "
                    f"the wrapper. Re-save the model with finetune.training."
                )
            n_features = raw_config.get("num_features") or raw_config.get(
                "num_prosody_features")
            wrapper = cls.from_pretrained(model_path, base_model_name=base_name,
                                          num_features=n_features)
            return wrapper.encoder, base_name

    log.info(f"Loading plain speech model: {model_path}")
    model = AutoModel.from_pretrained(model_path)
    return model, model_path


@torch.no_grad()
def embed_window(window: np.ndarray, processor, model, device,
                 layers: Optional[List[int]], max_length: int) -> np.ndarray:
    inputs = processor(
        window, sampling_rate=SAMPLING_RATE, return_tensors="pt",
        padding="max_length", max_length=max_length, truncation=True,
    ).to(device)

    if layers is None:
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

    outputs = model(**inputs, output_hidden_states=True)
    # hidden_states[0] is the CNN output; transformer layer i is at i + 1.
    pooled = [outputs.hidden_states[i + 1].mean(dim=1).squeeze(0) for i in layers]
    return torch.stack(pooled).mean(dim=0).cpu().numpy()


def load_waveform(path: Path) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLING_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLING_RATE)(waveform)
    return waveform


def extract_story(waveform: torch.Tensor, onsets: np.ndarray, processor, model,
                  device, layers: Optional[List[int]]) -> np.ndarray:
    window_samples = int(WINDOW_SIZE_SEC * SAMPLING_RATE)
    vectors = []
    for onset in onsets:
        start = int(onset * SAMPLING_RATE)
        end = start + window_samples
        if end <= waveform.shape[1]:
            chunk = waveform[:, start:end]
        else:
            # The last windows can run past the end of the audio; pad with
            # silence rather than emitting a shorter, differently-scaled vector.
            pad = window_samples - max(0, waveform.shape[1] - start)
            chunk = torch.nn.functional.pad(waveform[:, start:], (0, pad))
        vectors.append(
            embed_window(chunk.squeeze(0).numpy(), processor, model, device,
                         layers, window_samples)
        )
    return np.stack(vectors).astype(np.float32)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", "--model-path", dest="model_path",
                   default="wav2vec2", metavar="NAME",
                   help="registry key (" + ", ".join(sorted(REGISTRY)) + "), a "
                        "Hugging Face id, or a directory saved by "
                        "finetune.training. --list-models shows the registry.")
    p.add_argument("--list-models", action="store_true",
                   help="print the model registry and exit")
    p.add_argument("--processor", default=None,
                   help="feature extractor to use (default: the base model)")
    p.add_argument("--layers", default="auto",
                   help="'auto' (the model's registry default), 'last', a "
                        "range like '18-23', or a list like '6,12,18'")
    p.add_argument("--audio-dir", default=str(STIMULI_16K_DIR))
    p.add_argument("--out-name", default=None,
                   help="subdirectory under data/features (default: derived "
                        "from the model and layer choice)")
    p.add_argument("--stories", default=None,
                   help="comma-separated subset; default is every wav found")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")

    if args.list_models:
        print("Available models\n")
        print(format_registry())
        print("\nAny other value is a Hugging Face id or a fine-tuned "
              "directory from finetune.training.")
        return

    # A registry key resolves to its checkpoint; anything else (a hub id or a
    # fine-tuned directory) passes straight through.
    spec = resolve_model(args.model_path)
    model_path = spec.checkpoint

    layers = parse_layers(args.layers)
    if layers == "auto":
        if spec.default_layers is None:
            raise SystemExit(
                f"--layers auto needs a registry entry, and {args.model_path!r} "
                f"is not one. Pass an explicit range, e.g. --layers 18-23, or "
                f"--layers last. Fine-tuned directories always need this."
            )
        layers = parse_layers(spec.default_layers)
        log.info(f"--layers auto -> {spec.default_layers} (default for "
                 f"{spec.key})")

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    model, base_name = load_encoder(model_path)
    model.to(device).eval()

    # Layer indices must exist. The emotion checkpoint is pruned to 12
    # transformer layers, so a range like 18-23 that is valid for
    # wav2vec2-large silently addresses nothing there.
    n_layers = getattr(model.config, "num_hidden_layers", None)
    if layers is not None and n_layers is not None:
        bad = [i for i in layers if i < 0 or i >= n_layers]
        if bad:
            raise ValueError(
                f"--layers {args.layers} requests layer(s) {bad}, but this "
                f"encoder has {n_layers} transformer layers (valid 0..{n_layers - 1}). "
                f"audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim is pruned "
                f"to 12 layers; try --layers 6-11 or --layers last."
            )
    log.info(f"Encoder   : {n_layers} transformer layers, "
             f"hidden size {getattr(model.config, 'hidden_size', '?')}")

    from transformers import AutoFeatureExtractor
    processor_name = args.processor or base_name
    processor = AutoFeatureExtractor.from_pretrained(processor_name)

    if args.out_name:
        out_name = args.out_name
    else:
        stem = (spec.key if spec.key in REGISTRY
                else Path(model_path).name).replace("-", "_")
        suffix = "last" if layers is None else f"layers{layers[0]}to{layers[-1]}"
        out_name = f"wav2vec_{stem}_{suffix}"
    out_dir = Path(FEATURES_DIR) / out_name
    ensure_dirs(out_dir)

    log.info(f"Model     : {model_path}")
    log.info(f"Processor : {processor_name}")
    log.info(f"Layers    : {'last hidden state' if layers is None else layers}")
    log.info(f"Device    : {device}")
    log.info(f"Output    : {out_dir}")

    trfiles = load_trfiles()
    audio_dir = Path(args.audio_dir)

    if args.stories:
        stories = [s.strip() for s in args.stories.split(",") if s.strip()]
    else:
        stories = sorted(p.stem for p in audio_dir.glob("*.wav"))
    log.info(f"{len(stories)} stories to process")

    done = skipped = 0
    for story in stories:
        out_path = out_dir / f"{story}.hf5"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        if story not in trfiles:
            log.warning(f"  {story}: no TR timing, skipping")
            skipped += 1
            continue

        wav_path = audio_dir / f"{story}.wav"
        if not wav_path.exists():
            log.warning(f"  {story}: {wav_path} missing, skipping")
            skipped += 1
            continue

        waveform = load_waveform(wav_path)
        onsets = tr_onsets(story, trfiles)
        features = extract_story(waveform, onsets, processor, model, device, layers)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("data", data=features)
        log.info(f"  {story}: {features.shape} -> {out_path.name}")
        done += 1

    log.info(f"Done: {done} written, {skipped} skipped. Features in {out_dir}")


if __name__ == "__main__":
    main()
