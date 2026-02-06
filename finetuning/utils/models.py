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
