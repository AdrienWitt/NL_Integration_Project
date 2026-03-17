import os
import json
import torch
from utils.config import MAIN_DIR
import numpy as np
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)

from .models import AudioEncoderForProsody, AudioEncoderMultiTask
from .metrics_callbacks import compute_metrics_with_names, compute_multitask_metrics, MetricsCallback


class MultiTaskTrainer(Trainer):

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        prosody_labels = inputs.pop("prosody_labels", None)
        brain_labels   = inputs.pop("brain_labels", None)

        outputs = model(
            input_values=inputs["input_values"],
            attention_mask=inputs.get("attention_mask"),
            prosody_labels=prosody_labels,
            brain_labels=brain_labels,
        )

        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def _prepare_inputs(self, inputs):
        prosody_labels = inputs.pop("prosody_labels", None)
        brain_labels   = inputs.pop("brain_labels", None)
        inputs = super()._prepare_inputs(inputs)
        if prosody_labels is not None:
            inputs["prosody_labels"] = prosody_labels.to(self.args.device)
        if brain_labels is not None:
            inputs["brain_labels"] = brain_labels.to(self.args.device)
        return inputs

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override to:
        - pass prosody/brain labels through the custom forward
        - return concatenated [prosody | brain] logits and labels
          so compute_metrics can split them at num_prosody_features
        """
        has_labels = "prosody_labels" in inputs and "brain_labels" in inputs

        with torch.no_grad():
            prosody_labels = inputs.get("prosody_labels")
            brain_labels   = inputs.get("brain_labels")

            # Forward (compute_loss path handles label popping; call model directly here)
            inputs_for_model = {
                "input_values":   inputs["input_values"],
                "attention_mask": inputs.get("attention_mask"),
                "prosody_labels": prosody_labels,
                "brain_labels":   brain_labels,
            }
            outputs = model(**inputs_for_model)

        loss = outputs["loss"] if has_labels else None

        if prediction_loss_only:
            return (loss, None, None)

        # Concatenate logits: [B, num_prosody + num_brain]
        prosody_logits = outputs["prosody_logits"]
        brain_logits   = outputs["brain_logits"]
        logits = torch.cat([prosody_logits, brain_logits], dim=1)

        # Concatenate labels in same order
        if has_labels:
            combined_labels = torch.cat([prosody_labels, brain_labels], dim=1)
        else:
            combined_labels = None

        return (loss, logits, combined_labels)


class MultiTaskDataCollator:
    """
    Custom collator for multi-task learning that separates audio and brain labels.
    Reconstructs audio_labels from the concatenated labels if necessary.
    """
    def __init__(self, processor):
        self.processor = processor
        self.call_count = 0
    
    def __call__(self, batch):
        self.call_count += 1
        
        # Extract input_values and pad them
        input_values = [item["input_values"] for item in batch]
        
        # Use processor to pad audio inputs
        batch_encoded = self.processor.pad(
            {"input_values": input_values},
            padding=True,
            return_tensors="pt",
        )
        
        # Extract prosody and brain labels
        prosody_labels_list = []
        brain_labels_list = []
        
        for item_idx, item in enumerate(batch):
            # Get brain_labels (should always be present)
            if "brain_labels" not in item:
                print(f"DEBUG: Call {self.call_count}, Item {item_idx}, Keys: {list(item.keys())}")
                raise KeyError(
                    f"Missing 'brain_labels' in batch item {item_idx}. "
                    f"Available keys: {list(item.keys())}"
                )
            
            brain_label = item["brain_labels"]
            
            # Get audio_labels - try multiple sources
            audio_label = None
            if "audio_labels" in item:
                audio_label = item["audio_labels"]
            elif "labels" in item:
                # Extract audio portion from concatenated labels
                # labels = [audio_features + brain_features]
                labels = item["labels"]
                if labels is not None:
                    brain_dim = brain_label.shape[0]
                    audio_label = labels[:-brain_dim]
            
            if audio_label is None:
                print(f"DEBUG: Call {self.call_count}, Item {item_idx}")
                print(f"  Keys: {list(item.keys())}")
                if "labels" in item:
                    print(f"  labels shape: {item['labels'].shape if item['labels'] is not None else None}")
                print(f"  brain_labels shape: {brain_label.shape}")
                raise KeyError(
                    f"Cannot extract audio labels from item {item_idx}. "
                    f"Available keys: {list(item.keys())}"
                )
            
            prosody_labels_list.append(audio_label)
            brain_labels_list.append(brain_label)
        
        # Stack labels
        batch_encoded["prosody_labels"] = torch.stack(prosody_labels_list)
        batch_encoded["brain_labels"] = torch.stack(brain_labels_list)
        
        return batch_encoded


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
    brain_weight: float = 0.5,
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

    # Check if multi-task (brain PCA) mode
    use_brain_pca = getattr(train_dataset, "use_brain_pca", False)
    if use_brain_pca:
        layers_str = f"{layers_str}_multitask"

    subfolder_name = f"{model_id}_{layers_str}"
    
    output_dir = os.path.join(MAIN_DIR, output_dir, subfolder_name)
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Get feature dimensions
    num_prosody_features = train_dataset.audio_label_dim if use_brain_pca else len(train_dataset.feature_names)
    num_brain_features = train_dataset.brain_label_dim if use_brain_pca else None
    
    if use_brain_pca:
        print(f"Multi-task mode: predicting {num_prosody_features} prosody features + {num_brain_features} brain PCA components")
    else:
        print(f"Prosody-only mode: predicting {num_prosody_features} features")

    if resume_from_checkpoint and os.path.isdir(resume_from_checkpoint):
        print(f"Resuming from {resume_from_checkpoint}")
        config = AutoConfig.from_pretrained(resume_from_checkpoint)
        
        if use_brain_pca:
            model = AudioEncoderMultiTask.from_pretrained(
                resume_from_checkpoint,
                num_prosody_features=num_prosody_features,
                num_brain_features=num_brain_features,
                base_model_name=base_model_name,
                brain_weight=brain_weight,
            )
        else:
            model = AudioEncoderForProsody.from_pretrained(
                resume_from_checkpoint,
                num_features=num_prosody_features,
                base_model_name=base_model_name,
            )
        model.freeze_base_model(num_layers_to_freeze)
    else:
        config = AutoConfig.from_pretrained(base_model_name)
        if use_brain_pca:
            model = AudioEncoderMultiTask(
                config=config,
                num_prosody_features=num_prosody_features,
                num_brain_features=num_brain_features,
                base_model_name=base_model_name,
                freeze_layers=num_layers_to_freeze,
                brain_weight=brain_weight,
            )
        else:
            model = AudioEncoderForProsody(
                config=config,
                num_features=num_prosody_features,
                base_model_name=base_model_name,
                freeze_layers=num_layers_to_freeze,
            )

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
        remove_unused_columns=False
    )

    # Select appropriate data collator and trainer
    if use_brain_pca:
        data_collator = MultiTaskDataCollator(processor)
        trainer_class = MultiTaskTrainer
        # Disable num_workers to avoid serialization issues with multi-task labels
        training_args.dataloader_num_workers = 0
    else:
        class ProsodyDataCollator:
            def __init__(self, processor):
                self.processor = processor

            def __call__(self, batch):
                input_values = [item["input_values"] for item in batch]
                labels       = [item["labels"] for item in batch]
                batch_encoded = self.processor.pad(
                    {"input_values": input_values},
                    padding=True,
                    return_tensors="pt",
                )
                batch_encoded["labels"] = torch.stack(labels)
                return batch_encoded

        data_collator = ProsodyDataCollator(processor)
        trainer_class = Trainer

    if use_brain_pca:
        prosody_feature_names = train_dataset.feature_names[:num_prosody_features]

        def multitask_compute_metrics(p):
            predictions, labels = p
            all_preds  = np.array(predictions[0] if isinstance(predictions, (tuple, list)) else predictions)
            all_labels = np.concatenate([np.array(labels[0]), np.array(labels[1])], axis=1) \
                        if isinstance(labels, (tuple, list)) else np.array(labels)
            return compute_multitask_metrics(all_preds, all_labels, prosody_feature_names, num_prosody_features)

        metrics_fn = multitask_compute_metrics
    else:
        metrics_fn = lambda p: compute_metrics_with_names(p, train_dataset.feature_names)

    trainer = trainer_class(
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


   