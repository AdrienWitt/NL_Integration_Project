import os
import json
import torch
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from .models import AudioEncoderForProsody
from .metrics_callbacks import compute_metrics_with_names, MetricsCallback


def train_model(
    train_dataset,
    val_dataset,
    output_dir: str,
    model_type: str = "wav2vec2",
    base_model_name: str = None,
    num_layers_to_freeze: int = 8,
    learning_rate: float = 3e-5,
    batch_size: int = 8,
    num_epochs: int = 10,
    patience: int = 3,
    save_total_limit: int = 3,
    resume_from_checkpoint: str = None,
):
    MODEL_MAP = {
        "wav2vec2": "facebook/wav2vec2-large-960h",
        "hubert":   "facebook/hubert-large-ll60k",
        "wavlm":    "microsoft/wavlm-large",
    }

    if base_model_name is None:
        base_model_name = MODEL_MAP.get(model_type.lower())
        if base_model_name is None:
            raise ValueError(f"Unknown model_type '{model_type}'. Provide base_model_name.")

    print(f"Training with {model_type} backbone: {base_model_name}")

    model_id = model_type.lower()
    if base_model_name and base_model_name not in MODEL_MAP.get(model_type.lower(), ""):
        model_id = base_model_name.split("/")[-1].replace("-", "_").lower()

    layers_str = f"layers_frozen_{num_layers_to_freeze}" if num_layers_to_freeze is not None else "no_layers_frozen"
    if hasattr(train_dataset, "use_pca") and train_dataset.use_pca:
        pca_str = f"PCA_{train_dataset.pca_threshold:.2f}_ncomp_{train_dataset.pca.n_components_}"
        layers_str = f"{layers_str}_{pca_str}"

    subfolder_name = f"{model_id}_{layers_str}"
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(SCRIPT_DIR, output_dir, subfolder_name)
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    num_features = len(train_dataset.feature_names)
    print(f"Predicting {num_features} features")

    if resume_from_checkpoint and os.path.isdir(resume_from_checkpoint):
        print(f"Resuming from {resume_from_checkpoint}")
        config = AutoConfig.from_pretrained(resume_from_checkpoint)
        if hasattr(config, 'num_features'):
            if num_features != config.num_features:
                print(f"Warning: num_features mismatch. Using {config.num_features} from checkpoint config.")
                num_features = config.num_features

        model = AudioEncoderForProsody.from_pretrained(
            resume_from_checkpoint,
            num_features=num_features,
            base_model_name=base_model_name,
        )
        model.freeze_base_model(num_layers_to_freeze)
    else:
        config = AutoConfig.from_pretrained(base_model_name)
        model = AudioEncoderForProsody(
            config=config,
            num_features=num_features,
            base_model_name=base_model_name,
        )
        model.freeze_base_model(num_layers_to_freeze)

    processor = AutoFeatureExtractor.from_pretrained(base_model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.05,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to=[],
        torch_compile=True,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda p: compute_metrics_with_names(p, train_dataset.feature_names),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=patience),
            MetricsCallback(metrics_dir),
        ],
    )

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(final_dir)

    if hasattr(train_dataset, "scalers"):
        torch.save(train_dataset.scalers, os.path.join(final_dir, "feature_scalers.pt"))

    final_metrics = trainer.evaluate()

    training_info = {
        "model_type": model_type,
        "base_model": base_model_name,
        "features": train_dataset.feature_names,
        "num_layers_frozen": num_layers_to_freeze,
        "best_eval_loss": trainer.state.best_metric,
        "num_epochs_completed": trainer.state.epoch,
        "early_stopped": trainer.state.global_step < num_epochs * len(train_dataset) // batch_size,
        "final_metrics": final_metrics,
        "training_time": train_result.metrics.get("train_runtime"),
        "gpu_info": {
            "num_gpus": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None
        }
    }

    with open(os.path.join(final_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)

    with open(os.path.join(metrics_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    if hasattr(train_dataset, "pca_info"):
        with open(os.path.join(final_dir, "pca_info.json"), "w") as f:
            json.dump(train_dataset.pca_info, f, indent=2)

    return training_info
