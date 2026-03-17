import torch
from torch import nn
from transformers import AutoModel, AutoConfig, PreTrainedModel
from typing import Optional, Union, List


class AudioEncoderForProsody(PreTrainedModel):
    config_class = AutoConfig

    def __init__(
        self,
        config,
        num_features: Optional[int] = None,
        base_model_name: Optional[str] = None,
        freeze_layers: Union[int, List[int]] = 6,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        if num_features is None:
            num_features = getattr(config, "num_features", None)
            if num_features is None:
                raise ValueError(
                    "num_features must be provided either as an argument "
                    "or stored in the config."
                )
        self.num_features = num_features

        if base_model_name is None:
            base_model_name = getattr(config, "base_model_name", None)
            if base_model_name is None:
                raise ValueError(
                    "base_model_name is required when creating a new model "
                    "or loading from checkpoint."
                )
        self.base_model_name = base_model_name

        self.encoder = AutoModel.from_pretrained(
            self.base_model_name,
            config=config,
        )

        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(0.1)

        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_features),
        )

        self.loss_fct = nn.MSELoss()

        self.freeze_base_model(freeze_layers)

        self.config.base_model_name = self.base_model_name
        self.config.num_features = self.num_features

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
        for name, module in self.encoder.named_modules():
            if "feature_extractor" in name or "feature_projection" in name:
                for p in module.parameters():
                    p.requires_grad = False

        if layers_to_freeze is not None:
            if isinstance(layers_to_freeze, int):
                layers_to_freeze = list(range(layers_to_freeze))

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

class AudioEncoderMultiTask(PreTrainedModel):
    """
    Multi-task audio encoder for joint prosody + brain PCA prediction.
    
    The encoder learns shared representations that benefit both tasks.
    Separate prediction heads allow independent task supervision.
    """
    config_class = AutoConfig

    def __init__(
        self,
        config,
        num_prosody_features: Optional[int] = None,
        num_brain_features: Optional[int] = None,
        base_model_name: Optional[str] = None,
        freeze_layers: Union[int, List[int]] = 6,
        brain_weight: float = 0.5,
        shared_hidden_dim: int = 512,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        if num_prosody_features is None:
            num_prosody_features = getattr(config, "num_prosody_features", None)
            if num_prosody_features is None:
                raise ValueError(
                    "num_prosody_features must be provided either as an argument "
                    "or stored in the config."
                )
        self.num_prosody_features = num_prosody_features

        if num_brain_features is None:
            num_brain_features = getattr(config, "num_brain_features", None)
            if num_brain_features is None:
                raise ValueError(
                    "num_brain_features must be provided either as an argument "
                    "or stored in the config."
                )
        self.num_brain_features = num_brain_features

        if base_model_name is None:
            base_model_name = getattr(config, "base_model_name", None)
            if base_model_name is None:
                raise ValueError(
                    "base_model_name is required when creating a new model "
                    "or loading from checkpoint."
                )
        self.base_model_name = base_model_name
        self.brain_weight = brain_weight
        self.shared_hidden_dim = shared_hidden_dim

        self.encoder = AutoModel.from_pretrained(
            self.base_model_name,
            config=config,
        )

        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(0.1)

        # Shared layers before task-specific heads
        self.shared_fc = nn.Sequential(
            nn.Linear(self.hidden_size, shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Prosody prediction head
        self.prosody_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_prosody_features),
        )

        # Brain PCA prediction head
        self.brain_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_brain_features),
        )

        self.loss_fct = nn.MSELoss()

        self.freeze_base_model(freeze_layers)

        self.config.base_model_name = self.base_model_name
        self.config.num_prosody_features = self.num_prosody_features
        self.config.num_brain_features = self.num_brain_features
        self.config.brain_weight = self.brain_weight

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
        for name, module in self.encoder.named_modules():
            if "feature_extractor" in name or "feature_projection" in name:
                for p in module.parameters():
                    p.requires_grad = False

        if layers_to_freeze is not None:
            if isinstance(layers_to_freeze, int):
                layers_to_freeze = list(range(layers_to_freeze))

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
        prosody_labels: Optional[torch.Tensor] = None,
        brain_labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for multi-task learning.
        
        Parameters
        ----------
        input_values : torch.Tensor
            Audio features [B, T]
        attention_mask : Optional[torch.Tensor]
            Attention mask [B, T]
        prosody_labels : Optional[torch.Tensor]
            Prosody target labels [B, num_prosody_features]
        brain_labels : Optional[torch.Tensor]
            Brain PCA target labels [B, num_brain_features]
        
        Returns
        -------
        dict
            Contains: loss, prosody_logits, brain_logits
        """
        outputs = self.encoder(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state  # [B, T, D]
        pooled = torch.mean(hidden_states, dim=1)   # mean over time
        pooled = self.dropout(pooled)

        # Shared representation
        shared = self.shared_fc(pooled)  # [B, shared_hidden_dim]

        # Task-specific predictions
        prosody_logits = self.prosody_head(shared)  # [B, num_prosody_features]
        brain_logits = self.brain_head(shared)      # [B, num_brain_features]

        # Compute weighted multi-task loss
        loss = None
        if prosody_labels is not None and brain_labels is not None:
            prosody_loss = self.loss_fct(prosody_logits, prosody_labels)
            brain_loss = self.loss_fct(brain_logits, brain_labels)
            loss = prosody_loss + self.brain_weight * brain_loss

        return {
        "loss": loss,
        "logits": torch.cat([prosody_logits, brain_logits], dim=1),  # Trainer collects this
        "prosody_logits": prosody_logits,
        "brain_logits": brain_logits,
    }

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