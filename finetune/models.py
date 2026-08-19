"""
Regression head on top of a self-supervised speech encoder.

`AudioEncoderForProsody`
    One head predicting the eGeMAPS feature vector of a TR from that TR's
    audio window.

Pooling over time uses a learned attention weighting rather than a plain mean:
within a 2 s window the prosodically informative frames are a minority, and
mean pooling dilutes them.

A brain-PCA multi-task variant used to live here and was removed on
2026-08-19 — training the encoder on brain responses and then using its
features for voxelwise encoding is circular. See `trash/brain_pca_multitask/`.
"""

from typing import List, Optional, Union

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel


class _SpeechRegressorBase(PreTrainedModel):
    """Shared encoder handling: freezing, checkpointing, attention pooling."""

    config_class = AutoConfig

    def __init__(self, config, base_model_name: Optional[str] = None,
                 freeze_layers: Union[int, List[int], None] = 6,
                 truncate_layers: Optional[int] = None, **kwargs):
        super().__init__(config, **kwargs)

        if base_model_name is None:
            base_model_name = getattr(config, "base_model_name", None)
            if base_model_name is None:
                raise ValueError(
                    "base_model_name is required when creating a new model or "
                    "loading from a checkpoint that does not record it."
                )
        self.base_model_name = base_model_name

        self.encoder = AutoModel.from_pretrained(base_model_name, config=config)
        if truncate_layers is not None:
            self.truncate_encoder(truncate_layers)
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(0.1)

        # Learned temporal attention pooling: one scalar score per frame.
        self.temporal_attn = nn.Linear(self.hidden_size, 1)

        self.loss_fct = nn.MSELoss()
        # Truncation first, so freeze indices refer to the final stack.
        self.freeze_base_model(freeze_layers)
        self.config.base_model_name = base_model_name
        self.config.truncate_layers = truncate_layers

    # -- encoder plumbing ---------------------------------------------------

    @property
    def gradient_checkpointing(self):
        return getattr(self.encoder, "gradient_checkpointing", False)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        elif hasattr(self.encoder, "gradient_checkpointing"):
            self.encoder.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        if hasattr(self.encoder, "gradient_checkpointing_disable"):
            self.encoder.gradient_checkpointing_disable()
        elif hasattr(self.encoder, "gradient_checkpointing"):
            self.encoder.gradient_checkpointing = False

    def _transformer_layers(self):
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layers"):
            return self.encoder.encoder.layers
        if hasattr(self.encoder, "layers"):
            return self.encoder.layers
        return None

    def truncate_encoder(self, n_layers: int):
        """Keep only the first `n_layers` transformer layers.

        Used to build a depth-matched control: the audEERING emotion model is
        pruned to 12 layers, so comparing it against a 24-layer backbone
        confounds emotion pretraining with capacity. Truncating the control to
        the same depth removes that confound.
        """
        layers = self._transformer_layers()
        if layers is None:
            raise ValueError("Cannot truncate: no transformer layers found")
        if n_layers > len(layers):
            raise ValueError(
                f"Cannot truncate to {n_layers} layers: the encoder has only "
                f"{len(layers)}."
            )
        if n_layers == len(layers):
            return

        kept = nn.ModuleList(list(layers)[:n_layers])
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layers"):
            self.encoder.encoder.layers = kept
        else:
            self.encoder.layers = kept

        # Keep the configs honest — feature extraction reads num_hidden_layers
        # to validate layer indices.
        self.encoder.config.num_hidden_layers = n_layers
        self.config.num_hidden_layers = n_layers
        print(f"Truncated encoder to the first {n_layers} transformer layers")

    def freeze_base_model(self, layers_to_freeze: Union[int, List[int], None] = None):
        """Freeze the CNN front end always, plus exactly the requested layers.

        The convolutional feature extractor is frozen unconditionally: it
        encodes low-level acoustics that the pretraining objective already
        fixed, and fine-tuning it on a few hours of audio destabilises training.

        Note the `named_modules` loop rather than `encoder.freeze_feature_encoder()`:
        the HF helper also clears `Wav2Vec2FeatureEncoder._requires_grad`, which
        is the flag whose `forward` uses to force `hidden_states.requires_grad`.
        Under gradient checkpointing that is what keeps the graph connected
        across the frozen bottom of the stack — clear it and the trainable upper
        layers silently receive no gradient at all.

        This method is *authoritative*, not additive: every transformer layer
        not named here is explicitly unfrozen. Being additive made the resume
        path wrong, because the constructor's default had already frozen the
        bottom half by the time this was called with the real value.
        """
        for name, module in self.encoder.named_modules():
            if "feature_extractor" in name or "feature_projection" in name:
                for p in module.parameters():
                    p.requires_grad = False

        layers = self._transformer_layers()
        if layers is None:
            if layers_to_freeze is not None:
                print("Warning: no transformer layers found — skipping freezing")
            return

        if layers_to_freeze is None:
            layers_to_freeze = []
        elif isinstance(layers_to_freeze, int):
            layers_to_freeze = list(range(layers_to_freeze))
        else:
            layers_to_freeze = list(layers_to_freeze)

        out_of_range = [i for i in layers_to_freeze if i >= len(layers)]
        if out_of_range:
            raise ValueError(
                f"Cannot freeze layers {out_of_range}: this encoder has only "
                f"{len(layers)} transformer layers (valid 0..{len(layers) - 1}). "
                f"The emotion checkpoint is pruned to 12 layers, so a "
                f"--freeze-layers value tuned for a 24-layer model is too large."
            )

        frozen = set(layers_to_freeze)
        for i, layer in enumerate(layers):
            for p in layer.parameters():
                p.requires_grad = i not in frozen

        if frozen:
            print(f"Froze transformer layers {sorted(frozen)} of {len(layers)}")
        else:
            print(f"All {len(layers)} transformer layers trainable "
                  f"(CNN front end still frozen)")

    def unfreeze_all_transformer_layers(self):
        layers = self._transformer_layers()
        if layers is None:
            return
        for layer in layers:
            for p in layer.parameters():
                p.requires_grad = True
        print("Unfroze all transformer layers")

    # -- forward helpers ----------------------------------------------------

    def pool(self, input_values, attention_mask=None):
        """Encode a batch of waveforms into one attention-pooled vector each."""
        outputs = self.encoder(input_values, attention_mask=attention_mask,
                               output_hidden_states=False)
        hidden = outputs.last_hidden_state                       # [B, T, D]
        weights = torch.softmax(self.temporal_attn(hidden), dim=1)  # [B, T, 1]
        pooled = (hidden * weights).sum(dim=1)                   # [B, D]
        return self.dropout(pooled)

    @torch.no_grad()
    def get_hidden_states(self, input_values, attention_mask=None,
                          output_hidden_states: bool = True):
        """All layer activations — what `extract/wav2vec.py` reads."""
        was_training = self.training
        self.eval()
        outputs = self.encoder(input_values, attention_mask=attention_mask,
                               output_hidden_states=output_hidden_states,
                               return_dict=True)
        self.train(was_training)
        return outputs


class AudioEncoderForProsody(_SpeechRegressorBase):
    """Predict a TR's prosody feature vector from its audio window."""

    def __init__(self, config, num_features: Optional[int] = None,
                 base_model_name: Optional[str] = None,
                 freeze_layers: Union[int, List[int], None] = 6,
                 truncate_layers: Optional[int] = None, **kwargs):
        super().__init__(config, base_model_name=base_model_name,
                         freeze_layers=freeze_layers,
                         truncate_layers=truncate_layers, **kwargs)

        if num_features is None:
            num_features = getattr(config, "num_features", None)
            if num_features is None:
                raise ValueError("num_features must be given or stored in config")
        self.num_features = num_features

        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_features),
        )
        self.config.num_features = num_features

    def forward(self, input_values, attention_mask=None, labels=None):
        logits = self.regressor(self.pool(input_values, attention_mask))
        if labels is None:
            return {"logits": logits}
        return {"loss": self.loss_fct(logits, labels), "logits": logits}

