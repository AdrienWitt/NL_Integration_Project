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

# Constants
SAMPLING_RATE = 16000

class ProsodyDataset(Dataset):
    def __init__(self, audio_dir, prosody_dir, processor, story_names: List[str] = None):
        """
        Args:
            audio_dir: Path to directory containing .wav files
            prosody_dir: Path to directory containing OpenSMILE JSON feature files
            processor: Wav2Vec2Processor for audio processing
            story_names: List of story names to filter dataset
        """
        self.audio_dir = audio_dir
        self.prosody_dir = prosody_dir
        self.processor = processor
        self.story_names = story_names
        
        # Get sorted list of audio and prosody files
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
        self.prosody_files = sorted([f for f in os.listdir(prosody_dir) if f.endswith("_opensmile_windows.json")])
        
        if len(self.audio_files) != len(self.prosody_files):
            raise ValueError(f"Mismatch: {len(self.audio_files)} audio files vs {len(self.prosody_files)} prosody files.")

        # Set maximum audio length for 2-second windows at 16kHz
        self.max_length = int(2.0 * SAMPLING_RATE)  # 2 seconds * 16000 Hz = 32000 samples
        
        # Define resampler if needed
        self.resampler = torchaudio.transforms.Resample(orig_freq=SAMPLING_RATE, new_freq=SAMPLING_RATE)

        # Pre-process audio and features
        self.preprocessed_data = self._preprocess_data()

        # Initialize feature names
        if self.preprocessed_data:
            self.feature_names = self.preprocessed_data[0]["feature_names"]
        else:
            self.feature_names = []

    def _preprocess_data(self):
        """Pre-process audio and features into tensors with normalization based on all stories."""
        preprocessed_data = []
        all_features = {}

        # Collect all feature values across all stories for normalization
        for file in self.prosody_files:
            story_name = file.replace('_opensmile_windows.json', '')
            
            prosody_path = os.path.join(self.prosody_dir, file)
            with open(prosody_path, "r") as f:
                windows = json.load(f)
                for window in windows:
                    for feat_name, feat_value in window['features'].items():
                        if feat_name not in all_features:
                            all_features[feat_name] = []
                        all_features[feat_name].append(feat_value)

        # Create scalers for each feature using data from all stories
        self.scalers = {
            feat: StandardScaler().fit(np.array(values).reshape(-1, 1))
            for feat, values in all_features.items()
        }

        # Normalize features and process data
        for file in self.prosody_files:
            story_name = file.replace('_opensmile_windows.json', '')
            
            prosody_path = os.path.join(self.prosody_dir, file)
            with open(prosody_path, "r") as f:
                windows = json.load(f)
                
                # Load audio
                audio_path = os.path.join(self.audio_dir, f"{story_name}.wav")
                waveform, sr = torchaudio.load(audio_path)
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                # Resample if needed
                if sr != SAMPLING_RATE:
                    waveform = self.resampler(waveform)
                
                for window in windows:
                    # Extract the window segment using start and end times
                    start_time = float(window['window_start_time'])
                    end_time = float(window['window_end_time'])
                    start_sample = int(start_time * SAMPLING_RATE)
                    end_sample = int(end_time * SAMPLING_RATE)
                    
                    # Ensure end_sample doesn't exceed audio length
                    end_sample = min(end_sample, waveform.shape[1])
                    
                    # Extract window audio
                    window_audio = waveform[0, start_sample:end_sample].numpy()
                    
                    # Process audio
                    inputs = self.processor(
                        window_audio, 
                        sampling_rate=SAMPLING_RATE, 
                        return_tensors="pt",
                        padding="max_length",
                        max_length=self.max_length
                    )
                    input_values = inputs.input_values.squeeze()
                    
                    # Normalize features
                    normalized_features = {}
                    for feat_name, feat_value in window['features'].items():
                        normalized_features[feat_name] = float(
                            self.scalers[feat_name].transform([[feat_value]])[0][0]
                        )
                    
                    # Get all normalized features in a fixed order
                    feature_names = sorted(normalized_features.keys())  # Sort for consistent order
                    normalized_features_list = [normalized_features[feat] for feat in feature_names]
                    
                    preprocessed_data.append({
                        "input_values": input_values,
                        "labels": torch.tensor(normalized_features_list, dtype=torch.float32),
                        "story_name": story_name,
                        "window_time": f"{start_time:.2f}-{end_time:.2f}",
                        "feature_names": feature_names
                    })
        
        return preprocessed_data

    def __len__(self):
        """Returns total number of windows across all stories."""
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        """Returns a single window's pre-processed audio and features."""
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
    feature_names = train_dataset.feature_names  # We'll need to pass this somehow
    
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
    learning_rate: float = 1e-4,
    batch_size: int = 8,
    num_epochs: int = 10,
    patience: int = 3,
    save_total_limit: int = 3,
    resume_from_checkpoint: Optional[str] = None
):
    """Train the wav2vec model for prosody prediction, with support for resuming from checkpoint."""
    
    # Create a subdirectory for the number of layers frozen
    layers_frozen_dir = f"layers_frozen_{num_layers_to_freeze}" if num_layers_to_freeze is not None else "no_layers_frozen"
    output_dir = os.path.join(output_dir, layers_frozen_dir)
    
    # Create metrics directory
    metrics_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Number of features to predict
    num_features = len(train_dataset.feature_names)
    
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
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to=[],
        warmup_steps=100,
        weight_decay=0.01,
        # Multi-GPU settings
        local_rank=-1,  # For distributed training
        ddp_find_unused_parameters=False,  # Optimize DDP performance
        # Mixed precision settings
        fp16=True,  # Enable mixed precision training
        fp16_opt_level="O1",  # Use O1 for better stability
        # Gradient settings
        gradient_accumulation_steps=4,  # Accumulate gradients for larger effective batch size
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        # Performance optimizations
        dataloader_num_workers=4,  # Parallel data loading
        dataloader_pin_memory=True,  # Pin memory for faster data transfer to GPU
        # Optional: Use 8-bit Adam optimizer for memory efficiency
        optim="adamw_torch_fused" if num_gpus > 0 else "adamw_torch",
        # Resume from checkpoint
        resume_from_checkpoint=resume_from_checkpoint
        
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
    
    return training_info
   