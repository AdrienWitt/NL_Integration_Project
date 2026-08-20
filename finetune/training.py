"""
Trainer setup for the prosody fine-tuning runs.

Thin wrapper over the HF `Trainer`: a collator that pads the audio and stacks
the label tensor, plus optional layer-wise learning-rate decay from `optim.py`.

The multi-task (prosody + brain PCA) machinery that used to live here —
`MultiTaskDataCollator`, `MultiTaskTrainer`, and the `use_brain_pca` branches —
was removed on 2026-08-19. Fine-tuning the encoder on brain responses and then
using its features for voxelwise encoding is circular. See
`trash/brain_pca_multitask/`.
"""

import json
import os
from typing import Dict, List, Optional, Union

import torch
from transformers import (AutoConfig, AutoFeatureExtractor,
                          EarlyStoppingCallback, Trainer, TrainingArguments,
                          set_seed)

from config import FINETUNE_OUT
from . import resolve_base_model
from .metrics import MetricsCallback, compute_prosody_metrics
from .models import AudioEncoderForProsody
from .optim import LLRDTrainerMixin


# --------------------------------------------------------------------------
# Collators
# --------------------------------------------------------------------------

class ProsodyDataCollator:
    """Pad the audio and stack the single label tensor."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: List[Dict]) -> Dict:
        encoded = self.processor.pad(
            {"input_values": [item["input_values"] for item in batch]},
            padding=True, return_tensors="pt",
        )
        encoded["labels"] = torch.stack([item["labels"] for item in batch])
        return encoded


class ProsodyTrainer(LLRDTrainerMixin, Trainer):
    """Standard Trainer plus optional layer-wise learning-rate decay."""


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

#: Metrics where a higher value is better, for `--metric-for-best`.
_GREATER_IS_BETTER = {"eval_mean_r", "eval_mean_r2"}

#: bf16 needs Ampere or newer. `torch.cuda.is_available()` alone made the
#: Trainer raise on V100/T4.
_bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

def run_name(model_type: str, base_model_name: str, num_layers_to_freeze,
             truncate_layers: Optional[int] = None,
             llrd: Optional[float] = None,
             learning_rate: Optional[float] = None,
             seed: Optional[int] = None) -> str:
    """Directory name that records the run's defining settings.

    Every setting that changes the result is in the name. It used to key on
    model / truncation / freezing only, so a sweep over `--llrd`,
    `--learning-rate` or `--seed` wrote every arm into the same directory and
    each run silently overwrote the last.
    """
    # Registry keys are already short and stable, so they make better folder
    # names than the checkpoint path they resolve to.
    from .registry import REGISTRY
    spec_checkpoint = (REGISTRY[model_type].checkpoint
                       if model_type in REGISTRY else None)
    if spec_checkpoint and base_model_name in (None, spec_checkpoint):
        # The caller now always resolves base_model_name, so "did the user
        # override it?" is a comparison against the registry, not a None check.
        model_id = model_type.lower().replace("-", "_")
    else:
        model_id = (base_model_name or model_type).rstrip("/").split("/")[-1] \
            .replace("-", "_").lower()

    if num_layers_to_freeze is None:
        layers = "no_frozen"
    elif isinstance(num_layers_to_freeze, int):
        layers = f"frozen_{num_layers_to_freeze}"
    else:
        layers = f"frozen_{'_'.join(str(i) for i in num_layers_to_freeze)}"

    parts = [model_id]
    if truncate_layers:
        parts.append(f"trunc{truncate_layers}")
    parts.append(layers)
    if llrd is not None:
        parts.append(f"llrd{llrd:g}")
    if learning_rate is not None:
        parts.append(f"lr{learning_rate:g}")
    if seed is not None:
        parts.append(f"seed{seed}")
    return "_".join(parts)


def train_model(
    train_dataset,
    val_dataset,
    output_dir: Optional[str] = None,
    model_type: str = "wav2vec2",
    base_model_name: Optional[str] = None,
    num_layers_to_freeze: Union[int, List[int], None] = 8,
    truncate_layers: Optional[int] = None,
    learning_rate: float = 3e-5,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 10,
    patience: int = 3,
    save_total_limit: int = 3,
    resume_from_checkpoint: Optional[str] = None,
    llrd: Optional[float] = None,
    metric_for_best: str = "eval_loss",
    dataloader_workers: int = 4,
    torch_compile: bool = False,
    seed: int = 42,
) -> Dict:
    """Fine-tune a speech encoder on per-TR eGeMAPS prosody targets."""
    # Before anything is constructed: the Trainer only calls set_seed in its
    # own __init__, which is long after the regressor head and the pooling
    # layer have already drawn their weights from an unseeded RNG. Without
    # this, two runs at the same --seed differ, and the arms this project
    # compares are meant to differ only in the backbone.
    set_seed(seed)

    base_model_name = resolve_base_model(model_type, base_model_name)

    name = run_name(model_type, base_model_name, num_layers_to_freeze,
                    truncate_layers=truncate_layers, llrd=llrd,
                    learning_rate=learning_rate, seed=seed)
    root = output_dir or FINETUNE_OUT
    output_dir = os.path.join(str(root), name)
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    num_prosody = len(train_dataset.feature_names)

    print(f"Backbone : {base_model_name}")
    print(f"Run      : {name}")
    print(f"Output   : {output_dir}")
    print(f"Targets  : {num_prosody} prosody features")

    # -- model ------------------------------------------------------------
    if resume_from_checkpoint and os.path.isdir(resume_from_checkpoint):
        print(f"Resuming from {resume_from_checkpoint}")
        # Pass freeze/truncate through rather than letting the constructor
        # default apply: the checkpoint's own config carries num_hidden_layers,
        # so truncation is already baked in, but freezing is not, and
        # freeze_base_model is now authoritative rather than additive.
        model = AudioEncoderForProsody.from_pretrained(
            resume_from_checkpoint, num_features=num_prosody,
            base_model_name=base_model_name,
            freeze_layers=num_layers_to_freeze,
        )
    else:
        config = AutoConfig.from_pretrained(base_model_name)
        model = AudioEncoderForProsody(
            config=config, num_features=num_prosody,
            base_model_name=base_model_name,
            freeze_layers=num_layers_to_freeze,
            truncate_layers=truncate_layers,
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} parameters "
          f"({100 * trainable / total:.1f}%)")

    processor = AutoFeatureExtractor.from_pretrained(base_model_name)

    # -- training arguments -----------------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.05,
        # A ratio, not a fixed 500 steps: at ~24k windows and an effective
        # batch of 32 an epoch is ~750 steps, so 500 spent two thirds of the
        # first epoch warming up.
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        # bf16 needs Ampere or newer; on older cards it is a hard error, so
        # fall back to fp16 rather than refusing to run.
        bf16=_bf16_ok,
        fp16=torch.cuda.is_available() and not _bf16_ok,
        gradient_checkpointing=True,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        dataloader_num_workers=dataloader_workers,
        dataloader_pin_memory=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=metric_for_best in _GREATER_IS_BETTER,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to=["tensorboard"],
        torch_compile=torch_compile,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        seed=seed,
    )

    # Windows uses spawn-based multiprocessing, which cannot pickle the
    # dataset's open handles; Linux forks and is fine with workers.
    if os.name == "nt":
        training_args.dataloader_num_workers = 0

    # -- collator, trainer, metrics ---------------------------------------
    data_collator = ProsodyDataCollator(processor)
    feature_names = list(train_dataset.feature_names)

    def metrics_fn(eval_pred):
        return compute_prosody_metrics(eval_pred, feature_names)

    trainer = ProsodyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=metrics_fn,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=patience),
            MetricsCallback(metrics_dir),
        ],
    )

    trainer.llrd_decay = llrd
    if llrd is not None:
        print(f"Layer-wise LR decay enabled (decay={llrd})")

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Order matters under DDP. Every collective must be reached by every rank,
    # and every plain file write must happen on exactly one rank. Doing the
    # writes first deadlocked: ranks 1..N raced past save_model (a no-op for
    # them) into evaluate() and sat in an ALLGATHER for the full 30-minute NCCL
    # timeout while rank 0 was still writing, and the four ranks writing the
    # same paths concurrently left a 0-byte feature_scalers.joblib behind.
    final_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(final_dir)      # internally rank-guarded by the Trainer
    final_metrics = trainer.evaluate()  # collective: all ranks, before any I/O

    is_main = trainer.is_world_process_zero()
    if is_main:
        processor.save_pretrained(final_dir)

        if getattr(train_dataset, "scalers", None):
            # Needed to un-standardise predictions, and to score any later
            # evaluation on the same scale the model was trained on. joblib,
            # not torch.save: these are sklearn estimators, and torch.load
            # defaults to weights_only=True since torch 2.6, which refuses to
            # unpickle them.
            import joblib
            joblib.dump(train_dataset.scalers,
                        os.path.join(final_dir, "feature_scalers.joblib"))

    info = {
        "run_name": name,
        "model_type": model_type,
        "base_model": base_model_name,
        "num_prosody_features": num_prosody,
        "features": list(train_dataset.feature_names),
        "num_layers_frozen": num_layers_to_freeze,
        "truncate_layers": truncate_layers,
        "encoder_layers": getattr(model.config, "num_hidden_layers", None),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "learning_rate": learning_rate,
        "llrd": llrd,
        "batch_size": batch_size,
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "epochs_requested": num_epochs,
        "epochs_completed": trainer.state.epoch,
        "seed": seed,
        "metric_for_best": metric_for_best,
        "best_metric": trainer.state.best_metric,
        # Epoch is a float and lands a hair under the target, so compare with
        # a tolerance rather than reporting every completed run as stopped.
        "early_stopped": (trainer.state.epoch or 0) < num_epochs - 0.5,
        "n_train_windows": len(train_dataset),
        "n_val_windows": len(val_dataset),
        "final_metrics": final_metrics,
        "training_time_sec": train_result.metrics.get("train_runtime"),
        "gpus": [torch.cuda.get_device_name(i)
                 for i in range(torch.cuda.device_count())],
    }
    if is_main:
        with open(os.path.join(final_dir, "training_info.json"), "w",
                  encoding="utf-8") as f:
            json.dump(info, f, indent=2, default=str)
        with open(os.path.join(metrics_dir, "final_metrics.json"), "w",
                  encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2, default=str)

    # Without this the ranks exit with a live process group and NCCL warns
    # about leaked resources.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    print(f"\nSaved fine-tuned model to {final_dir}")
    print(f"  Point extract/wav2vec.py at this directory with "
          f"--model {final_dir}")
    return info
