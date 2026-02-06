# Finetuning module (refactored)

This folder contains refactored modules from the previous `model_prosody.py` file.

Layout (data-science friendly):

- `prosody_dataset.py`: `ProsodyDataset` and preprocessing (scalers, PCA, audio/JSON loading).
- `models.py`: `AudioEncoderForProsody` and model utilities (freeze/unfreeze, checkpointing helpers).
- `training.py`: `train_model` (TrainingArguments, Trainer setup, save/metrics logic).
- `metrics_callbacks.py`: `compute_metrics_with_names` and `MetricsCallback` for evaluation logging.
- `prosody_utils.py`: constants and small helpers (e.g. `load_trfiles`).
- `model_prosody.py`: thin compatibility wrapper re-exporting the main symbols.

Quick usage

Run training (example):

```bash
python fine_tuning.py --audio_dir /path/to/wavs --prosody_dir /path/to/jsons --output_dir /path/to/out
```

Notes
- The refactor keeps the public API intact via `model_prosody.py` so existing imports continue to work.
- If you run the scripts as top-level modules, Python needs to import the local modules directly (the code uses absolute imports for that purpose).
