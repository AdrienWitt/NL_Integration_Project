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
    Wav2Vec2Model,
    Wav2Vec2Processor,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Config,
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
        processor: Wav2Vec2Processor,
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

class Wav2Vec2ForProsody(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config, num_features: int, freeze_layers: Union[int, List[int]] = 6):
        """
        Args:
            config: Wav2Vec2Config configuration
            num_features: Number of OpenSMILE features to predict
            freeze_layers: Number of layers to freeze (default: 4) or list of specific layers to freeze
        """
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(0.1)
        
        # Determine the hidden size from config, with fallback to detect at runtime
        self.hidden_size = getattr(config, "hidden_size", 1024)
        
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),  
            nn.Dropout(0.1),
            nn.Linear(512, num_features) 
        )
        self.loss_fct = nn.MSELoss()
        self.init_weights()
        
        # Freeze layers by default
        self.freeze_base_model(freeze_layers)

    def freeze_feature_encoder(self):
        """Freeze the feature encoder layers"""
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False

    def freeze_base_model(self, layers_to_freeze: Union[int, List[int]] = None):
        """Freeze transformer layers.
        
        Args:
            layers_to_freeze: If int, freezes layers 0 to layers_to_freeze-1.
                            If list, freezes only the specified layer indices.
                            If None, no layers are frozen.
        """
        # Always freeze feature encoder
        self.freeze_feature_encoder()
        
        if layers_to_freeze is not None:
            if isinstance(layers_to_freeze, int):
                # Freeze first N layers
                layers_to_freeze = list(range(layers_to_freeze))
            
            # Freeze specified layers
            for i in layers_to_freeze:
                print(f"Freezing layer {i}")
                for param in self.wav2vec2.encoder.layers[i].parameters():
                    param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all layers except feature encoder."""
        # Unfreeze all transformer layers
        for i in range(len(self.wav2vec2.encoder.layers)):
            for param in self.wav2vec2.encoder.layers[i].parameters():
                param.requires_grad = True
        print("Unfroze all transformer layers")

    def forward(
        self,
        input_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]  # Get the last hidden state
        
        # Pool the output (mean pooling over time dimension)
        pooled_output = torch.mean(hidden_states, dim=1)
        
        pooled_output = self.dropout(pooled_output)
        
        # Predict prosody features
        logits = self.regressor(pooled_output)
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

    def get_hidden_states(
        self,
        input_values: torch.Tensor,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ):
        """Extract hidden states (embeddings) from different layers of the model.
        
        Args:
            input_values: Audio input tensor
            output_hidden_states: Whether to return hidden states from all layers
            return_dict: Whether to return output as a dict or tuple
            
        Returns:
            Dictionary containing:
                - last_hidden_state: Final layer hidden states [batch, seq_len, hidden_size]
                - hidden_states: Tuple of hidden states from all layers (if output_hidden_states=True)
        """
        # Set model to eval mode during embedding extraction
        was_training = self.training
        self.eval()
        
        # Run model with output_hidden_states=True to get embeddings from all layers
        with torch.no_grad():
            outputs = self.wav2vec2(
                input_values,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        
        # Return to original training state
        self.train(was_training)
        
        return outputs

def compute_metrics(eval_pred):
    """Compute metrics for each prosody feature."""
    predictions, labels = eval_pred
    
    # Initialize metrics dictionary
    metrics = {
        'eval_loss': 0.0,  # Overall MSE loss
        'feature_metrics': {}
    }
    
    # Compute overall MSE loss
    mse = mean_squared_error(labels, predictions)
    metrics['eval_loss'] = mse
    
    # Get feature names from the dataset
    feature_names = train_dataset.feature_names
    
    # Compute metrics for each feature
    for i, feature in enumerate(feature_names):
        feature_pred = predictions[:, i]
        feature_true = labels[:, i]
        
        feature_metrics = {
            'mse': mean_squared_error(feature_true, feature_pred),
            'rmse': np.sqrt(mean_squared_error(feature_true, feature_pred)),
            'mae': mean_absolute_error(feature_true, feature_pred),
            'r2': r2_score(feature_true, feature_pred)
        }
        
        # Add feature-specific metrics to the main metrics dict
        for metric_name, value in feature_metrics.items():
            metrics[f'{feature}_{metric_name}'] = value
        
        metrics['feature_metrics'][feature] = feature_metrics
    
    return metrics

class MetricsCallback(TrainerCallback):
    """Custom callback to save detailed metrics history."""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.metrics_history = []
        
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Add epoch number to metrics
        metrics['epoch'] = state.epoch
        self.metrics_history.append(metrics)
        
        # Save metrics history to CSV
        metrics_df = pd.DataFrame(self.metrics_history)
        metrics_df.to_csv(os.path.join(self.output_dir, 'metrics_history.csv'), index=False)
        
        # Save detailed metrics for current epoch
        epoch_metrics_file = os.path.join(self.output_dir, f'metrics_epoch_{state.epoch:.1f}.json')
        with open(epoch_metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

def train_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: str,
    num_layers_to_freeze: int = None,
    learning_rate: float = 3e-5,
    batch_size: int = 8,
    num_epochs: int = 10,
    patience: int = 3,
    save_total_limit: int = 3,
    resume_from_checkpoint: Optional[str] = None
):
    """Train the wav2vec model for prosody prediction, with support for resuming from checkpoint."""
    
    # Create a subdirectory for the number of layers frozen and PCA information
    layers_frozen_dir = f"layers_frozen_{num_layers_to_freeze}" if num_layers_to_freeze is not None else "no_layers_frozen"
    
    # Add PCA information to directory name if PCA is used
    if hasattr(train_dataset, 'use_pca') and train_dataset.use_pca:
        pca_info = f"PCA_{train_dataset.pca_threshold:.2f}_ncomp_{train_dataset.pca.n_components_}"
        layers_frozen_dir = f"{layers_frozen_dir}_{pca_info}"
    
    output_dir = os.path.join(output_dir, layers_frozen_dir)
    
    # Create metrics directory
    metrics_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Number of features to predict
    num_features = len(train_dataset.feature_names)
    print(f"Number of features (model output size): {num_features}")
    
    # Load model
    if resume_from_checkpoint and os.path.isdir(resume_from_checkpoint):
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        # Load model from checkpoint instead of base pretrained weights
        model = Wav2Vec2ForProsody.from_pretrained(resume_from_checkpoint, num_features=num_features)
        # Ensure feature encoder and layers are frozen as in original training
        model.freeze_base_model(num_layers_to_freeze)
    else:
        print("Starting fresh training from pretrained weights")
        model = Wav2Vec2ForProsody.from_pretrained(
            "facebook/wav2vec2-large-960h",
            num_features=num_features
        )
        model.freeze_base_model(num_layers_to_freeze)
    
    # Create a compute_metrics function with access to feature names
    def compute_metrics_with_names(eval_pred):
        predictions, labels = eval_pred
        metrics = {
            'eval_loss': 0.0,
            'feature_metrics': {}
        }
        
        mse = mean_squared_error(labels, predictions)
        metrics['eval_loss'] = mse
        
        for i, feature in enumerate(train_dataset.feature_names):
            feature_pred = predictions[:, i]
            feature_true = labels[:, i]
            
            feature_metrics = {
                'mse': mean_squared_error(feature_true, feature_pred),
                'rmse': np.sqrt(mean_squared_error(feature_true, feature_pred)),
                'mae': mean_absolute_error(feature_true, feature_pred),
                'r2': r2_score(feature_true, feature_pred)
            }
            
            for metric_name, value in feature_metrics.items():
                metrics[f'{feature}_{metric_name}'] = value
            
            metrics['feature_metrics'][feature] = feature_metrics
        
        return metrics

    # Check for available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    if num_gpus > 0:
        print(f"Using GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
    
    # Define training arguments with multi-GPU support
    training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=4,
    learning_rate=3e-5,              
    weight_decay=0.05,                
    warmup_steps=500,                 
    lr_scheduler_type="cosine",       
    bf16=True,                        
    fp16=False,
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
    # Optional but recommended
    torch_compile=True,
    ddp_find_unused_parameters=False,
)
    
    # Initialize trainer with callbacks
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_with_names,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=patience),
            MetricsCallback(metrics_dir)
        ]
    )
    
    # Train the model
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save the final model
    final_output_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(final_output_dir)
    
    # Save feature scalers - FIX: access scalers from story_data
    if hasattr(train_dataset, 'scalers'):
        torch.save(train_dataset.scalers, os.path.join(final_output_dir, "feature_scalers.pt"))
    else:
        print("Warning: Scalers not found in dataset. Add them to ProsodyDataset if needed.")
    
    # Compute final metrics on validation set
    final_metrics = trainer.evaluate()
    
    # Save training info with detailed metrics
    training_info = {
        "features": train_dataset.feature_names,  # Use actual feature names
        "num_layers_frozen": num_layers_to_freeze,
        "best_eval_loss": trainer.state.best_metric,
        "num_epochs_completed": trainer.state.epoch,
        "early_stopped": trainer.state.global_step < num_epochs * len(train_dataset) // batch_size,
        "final_metrics": final_metrics,
        "training_time": train_result.metrics.get("train_runtime", None),
        "total_steps": train_result.metrics.get("train_steps", None),
        "gpu_info": {
            "num_gpus": num_gpus,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(num_gpus)] if num_gpus > 0 else None
        }
    }
    
    # Save training info
    with open(os.path.join(final_output_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)
    
    # Save final metrics separately
    with open(os.path.join(metrics_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    # Save PCA information if available
    if hasattr(train_dataset, 'pca_info'):
        with open(os.path.join(final_output_dir, "pca_info.json"), "w") as f:
            json.dump(train_dataset.pca_info, f, indent=2)
    
    return training_info
   