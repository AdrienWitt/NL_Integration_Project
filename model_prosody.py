import os
import torch
import numpy as np
import pandas as pd
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torchaudio
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoFeatureExtractor,
    Wav2Vec2FeatureExtractor,
    PreTrainedModel,
    Wav2Vec2PreTrainedModel,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback
)
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from encoding.config import REPO_DIR
from encoding.ridge_utils.stimulus_utils import load_simulated_trfiles


# Constants
SAMPLING_RATE = 16000
WINDOW_SIZE_SEC = 2.0
RESPDICT_PATH = os.path.join(REPO_DIR, "ds003020/derivative/respdict.json")


class ProsodyDataset(Dataset):
    """
    Dataset that loads TR-aligned 2-second audio windows and corresponding OpenSMILE prosody features.
    - One sample per valid TR (non-overlapping, starting at TR onset).
    - Labels are taken from pre-computed JSON files.
    - Supports global normalization and optional PCA.
    """

    def __init__(
        self,
        audio_dir: str,
        prosody_dir: str,
        processor: AutoFeatureExtractor,
        story_names: Optional[List[str]] = None,
        use_pca: bool = False,
        pca_threshold: float = 0.90,
        pca: Optional[PCA] = None,
        scalers: Optional[Dict[str, StandardScaler]] = None,
        respdict_path: str = "ds003020/derivative/respdict.json",
        tr: float = 2.0,
        pad: int = 5,
        sampling_rate: int = 16000,
        window_size_sec: float = 2.0,
    ):
        self.audio_dir = audio_dir
        self.prosody_dir = prosody_dir
        self.processor = processor
        self.story_names_filter = story_names or []
        self.use_pca = use_pca
        self.pca_threshold = pca_threshold
        self.pca = pca
        self.scalers = scalers

        self.sampling_rate = sampling_rate
        self.max_length = int(window_size_sec * sampling_rate)
        self.window_size_sec = window_size_sec

        self.resampler = torchaudio.transforms.Resample(
            orig_freq=sampling_rate, new_freq=sampling_rate
        )

        # Load TR timing information once
        with open(respdict_path, "r") as f:
            respdict = json.load(f)
        self.trfiles = load_simulated_trfiles(respdict, tr=tr, pad=pad, start_time=10)

        # Discover valid stories (audio + JSON + TR info)
        self.available_stories = sorted(
            f[:-4] for f in os.listdir(audio_dir) if f.endswith(".wav")
        )
        self.valid_stories = [
            story
            for story in self.available_stories
            if story in self.trfiles
            and os.path.exists(os.path.join(prosody_dir, f"{story}_opensmile_tr_aligned.json"))
        ]
        
        if story_names:
            requested = set(story_names)
            self.valid_stories = [
                s for s in self.valid_stories if s in requested
            ]
            if not self.valid_stories:
                raise ValueError(
                    f"None of the requested stories {story_names} "
                    f"are valid. Available valid: {self.valid_stories}"
                )

              
        if not self.valid_stories:
            raise ValueError("No valid stories found (audio + JSON + TR timing)")

        # Pre-process all data
        self.preprocessed_data = self._preprocess_data()

        # Apply optional story name filter
        if self.story_names_filter:
            self.preprocessed_data = [
                item for item in self.preprocessed_data
                if item["story_name"] in self.story_names_filter
            ]

        # Set feature names from first sample
        self.feature_names = (
            self.preprocessed_data[0]["feature_names"] if self.preprocessed_data else []
        )

    def _preprocess_data(self) -> List[Dict]:
        preprocessed = []
        all_raw_features = {}  # Collect for global normalization

        # 1. Collect all raw feature values across all stories
        for story in self.valid_stories:
            json_path = os.path.join(self.prosody_dir, f"{story}_opensmile_tr_aligned.json")
            with open(json_path, "r") as f:
                windows = json.load(f) or []

            for window in windows:
                for feat, val in window.get("features", {}).items():
                    all_raw_features.setdefault(feat, []).append(val)

        # Fit scalers if not provided
        if self.scalers is None:
            self.scalers = {
                feat: StandardScaler().fit(np.array(values).reshape(-1, 1))
                for feat, values in all_raw_features.items()
            }

        # 2. Process each story and generate samples
        for story in self.valid_stories:
            # Load full waveform
            audio_path = os.path.join(self.audio_dir, f"{story}.wav")
            waveform, sr = torchaudio.load(audio_path)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if sr != self.sampling_rate:
                waveform = self.resampler(waveform)

            # Get TR times
            tr_info = self.trfiles[story][0]
            tr_times = tr_info.get_reltriggertimes() + tr_info.soundstarttime

            # Load corresponding OpenSMILE windows (assume order matches)
            json_path = os.path.join(self.prosody_dir, f"{story}_opensmile_tr_aligned.json")
            with open(json_path, "r") as f:
                opensmile_windows = json.load(f) or []

            if len(opensmile_windows) != len(tr_times):
                print(
                    f"Warning: {story} - {len(opensmile_windows)} JSON windows vs "
                    f"{len(tr_times)} TR times → mismatch!"
                )

            for idx, (tr_time, osm_window) in enumerate(zip(tr_times, opensmile_windows)):
                for shift in [0.0, 1.0]:
                    start_time = float(tr_time)
                    end_time = start_time + self.window_size_sec
    
                    start_sample = int(start_time * self.sampling_rate)
                    end_sample = min(
                        int(end_time * self.sampling_rate), waveform.shape[1]
                    )
    
                    window_audio_np = waveform[0, start_sample:end_sample].numpy()
    
                    # Process with Wav2Vec2 processor
                    inputs = self.processor(
                        window_audio_np,
                        sampling_rate=self.sampling_rate,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=self.max_length,
                        truncation=True,
                    )
                    input_values = inputs.input_values.squeeze(0)  # [max_length]
    
                    # Normalize labels
                    normalized = {}
                    for feat, raw_val in osm_window.get("features", {}).items():
                        normalized[feat] = float(
                            self.scalers[feat].transform([[raw_val]])[0][0]
                        )
    
                    feat_names = sorted(normalized.keys())
                    labels_tensor = torch.tensor(
                        [normalized[fn] for fn in feat_names], dtype=torch.float32
                    )
    
                    preprocessed.append(
                        {
                            "input_values": input_values,
                            "labels": labels_tensor,
                            "story_name": story,
                            "window_time": f"{start_time:.2f}-{end_time:.2f}",
                            "feature_names": feat_names,
                            "tr_index": idx,
                            "tr_time": float(tr_time),
                        }
                    )

        # Optional PCA
        if self.use_pca and preprocessed:
            X = np.array([item["labels"].numpy() for item in preprocessed])

            if self.pca is None:
                self.pca = PCA(n_components=self.pca_threshold)
                X_pca = self.pca.fit_transform(X)
                print(
                    f"PCA → {self.pca.n_components_} components, "
                    f"explained variance: {sum(self.pca.explained_variance_ratio_):.3f}"
                )
            else:
                X_pca = self.pca.transform(X)

            for i, item in enumerate(preprocessed):
                item["labels"] = torch.tensor(X_pca[i], dtype=torch.float32)
                item["feature_names"] = [f"PC_{j+1}" for j in range(X_pca.shape[1])]

            self.pca_info = {
                "n_components": self.pca.n_components_,
                "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
                "cumulative_explained_variance": np.cumsum(
                    self.pca.explained_variance_ratio_
                ).tolist(),
            }

        return preprocessed

    def __len__(self) -> int:
        return len(self.preprocessed_data)

    def __getitem__(self, idx: int) -> Dict:
        return self.preprocessed_data[idx]



class AudioEncoderForProsody(PreTrainedModel):
    config_class = AutoConfig

    def __init__(
        self,
        config,                               # ← first positional = config (required!)
        num_features: int,
        base_model_name: Optional[str] = None,
        freeze_layers: Union[int, List[int]] = 8,
        dropout_p: float = 0.1,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        # Determine base_model_name: from argument → from config → error
        self.base_model_name = base_model_name
        if self.base_model_name is None:
            self.base_model_name = getattr(config, "base_model_name", None)
            if self.base_model_name is None:
                raise ValueError(
                    "base_model_name is required when creating a new model.\n"
                    "Either pass it explicitly or make sure it's stored in the saved config."
                )

        print(f"Initializing with backbone: {self.base_model_name}")
        self.encoder = AutoModel.from_pretrained(self.base_model_name)

        self.hidden_size = config.hidden_size
        print(f"Hidden size: {self.hidden_size}")

        self.dropout = nn.Dropout(dropout_p)
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, num_features),
        )
        self.loss_fct = nn.MSELoss()

        # Freeze after initialization
        self.freeze_base_model(freeze_layers)

        # Save important attributes to config for future loading
        self.config.base_model_name = self.base_model_name
        self.config.num_features = num_features
        self.config.dropout_p = dropout_p

    @property
    def gradient_checkpointing(self):
        return getattr(self.encoder, 'gradient_checkpointing', False)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        else:
            if hasattr(self.encoder, "gradient_checkpointing"):
                self.encoder.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        if hasattr(self.encoder, "gradient_checkpointing_disable"):
            self.encoder.gradient_checkpointing_disable()
        else:
            if hasattr(self.encoder, "gradient_checkpointing"):
                self.encoder.gradient_checkpointing = False

    def freeze_base_model(self, layers_to_freeze: Union[int, List[int]] = None):
        # Freeze conv / projection layers if present
        for name, module in self.encoder.named_modules():
            if "feature_extractor" in name or "feature_projection" in name:
                for p in module.parameters():
                    p.requires_grad = False

        if layers_to_freeze is not None:
            if isinstance(layers_to_freeze, int):
                layers_to_freeze = list(range(layers_to_freeze))

            # Find transformer layers (common patterns)
            if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layers"):
                layers = self.encoder.encoder.layers
            elif hasattr(self.encoder, "layers"):
                layers = self.encoder.layers
            else:
                print("Warning: Could not find transformer layers to freeze — skipping layer freezing")
                return

            for i in layers_to_freeze:
                if i >= len(layers):
                    print(f"Warning: Cannot freeze layer {i} (max {len(layers)-1})")
                    continue
                print(f"Freezing layer {i}")
                for p in layers[i].parameters():
                    p.requires_grad = False

    def unfreeze_all_transformer_layers(self):
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layers"):
            layers = self.encoder.encoder.layers
        elif hasattr(self.encoder, "layers"):
            layers = self.encoder.layers
        else:
            return

        for layer in layers:
            for p in layer.parameters():
                p.requires_grad = True
        print("Unfroze all transformer layers")

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        outputs = self.encoder(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state  # [B, T, D]
        pooled = torch.mean(hidden_states, dim=1)   # mean over time
        pooled = self.dropout(pooled)
        logits = self.regressor(pooled)

        loss = self.loss_fct(logits, labels) if labels is not None else None

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

    @torch.no_grad()
    def get_hidden_states(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
    ):
        was_training = self.training
        self.eval()

        outputs = self.encoder(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        self.train(was_training)
        return outputs



def compute_metrics_with_names(eval_pred, feature_names):
    predictions, labels = eval_pred
    metrics = {"eval_loss": mean_squared_error(labels, predictions)}

    for i, feat in enumerate(feature_names):
        p, t = predictions[:, i], labels[:, i]
        metrics.update({
            f"{feat}_mse":  mean_squared_error(t, p),
            f"{feat}_rmse": np.sqrt(mean_squared_error(t, p)),
            f"{feat}_mae":  mean_absolute_error(t, p),
            f"{feat}_r2":   r2_score(t, p),
        })

    return metrics


class MetricsCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics_history = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metrics["epoch"] = state.epoch
        self.metrics_history.append(metrics)

        pd.DataFrame(self.metrics_history).to_csv(
            os.path.join(self.output_dir, "metrics_history.csv"), index=False
        )

        with open(os.path.join(self.output_dir, f"metrics_epoch_{state.epoch:.1f}.json"), "w") as f:
            json.dump(metrics, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
#   Main training function 
# ──────────────────────────────────────────────────────────────────────────────

def train_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: str,
    model_type: str = "wav2vec2",
    base_model_name: Optional[str] = None,
    num_layers_to_freeze: Optional[int] = 8,
    learning_rate: float = 3e-5,
    batch_size: int = 8,
    num_epochs: int = 10,
    patience: int = 3,
    save_total_limit: int = 3,
    resume_from_checkpoint: Optional[str] = None,
):
    """
    Flexible training for prosody regression.
    model_type: "wav2vec2", "hubert", "wavlm"
    """
    MODEL_MAP = {
        "wav2vec2": "facebook/wav2vec2-large-960h",
        "hubert":   "facebook/hubert-large-ll60k",          # or -ls960-ft
        "wavlm":    "microsoft/wavlm-large",
    }

    if base_model_name is None:
        base_model_name = MODEL_MAP.get(model_type.lower())
        if base_model_name is None:
            raise ValueError(f"Unknown model_type '{model_type}'. Provide base_model_name.")

    print(f"Training with {model_type} backbone: {base_model_name}")

    # Subdirectory naming — now includes model identifier
    model_id = model_type.lower()
    if base_model_name and base_model_name not in MODEL_MAP.get(model_type.lower(), ""):
        # If custom checkpoint name, use a short sanitized version
        model_id = base_model_name.split("/")[-1].replace("-", "_").lower()

    layers_str = f"layers_frozen_{num_layers_to_freeze}" if num_layers_to_freeze is not None else "no_layers_frozen"
    if hasattr(train_dataset, "use_pca") and train_dataset.use_pca:
        pca_str = f"PCA_{train_dataset.pca_threshold:.2f}_ncomp_{train_dataset.pca.n_components_}"
        layers_str = f"{layers_str}_{pca_str}"

    # Final subfolder name example: hubert_layers_frozen_10_PCA_0.90_ncomp_15
    #               or: wavlm_layers_frozen_6
    subfolder_name = f"{model_id}_{layers_str}"

    output_dir = os.path.join(output_dir, subfolder_name)
    
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    num_features = len(train_dataset.feature_names)
    print(f"Predicting {num_features} features")

    # Load model
    if resume_from_checkpoint and os.path.isdir(resume_from_checkpoint):
        print(f"Resuming from {resume_from_checkpoint}")
        model = AudioEncoderForProsody.from_pretrained(
            resume_from_checkpoint,
            base_model_name=base_model_name,
            num_features=num_features,            
        )
        model.freeze_base_model(num_layers_to_freeze)
    
    else:
        model = AudioEncoderForProsody(
            base_model_name=base_model_name,
            num_features=num_features,
        )
        model.freeze_base_model(num_layers_to_freeze)

        
        
    # Processor should already be set in datasets — but ensure consistency
    
    processor = AutoFeatureExtractor.from_pretrained(base_model_name)
    
    # Training args (unchanged — good defaults)
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

    # Save final model & artifacts
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