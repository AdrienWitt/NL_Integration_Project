"""
Regression head on top of a self-supervised speech encoder.

`AudioEncoderForProsody`
    One head predicting the eGeMAPS feature vector of a TR from that TR's
    audio window.

Pooling over time uses a learned attention weighting rather than a plain mean:
within a 2 s window the prosodically informative frames are a minority, and
mean pooling dilutes them.

Two levers exist to stop the eGeMAPS objective from eating the representation
------------------------------------------------------------------------------
Fine-tuning on the 88 functionals measurably *degrades* brain prediction, and
the damage has a shape: it grows monotonically with depth into the trainable
region (emotion, cv, 9/9 subjects: -0.004 at L6, -0.056 at L11) and it stops at
openSMILE's own encoding score (ft L11 = 0.094, openSMILE = 0.087). Where a base
layer already scored *below* openSMILE the same objective made it better. So the
loss transports every trainable layer toward one fixed destination -- a
representation sufficient for 88 numbers -- and the only free variable is how far
along that path training travels.

`l2sp` shortens the path. Weight decay pulls toward zero, which is not where the
pretrained solution is; L2-SP (Xuhong et al. 2018) penalises the distance to the
*pretrained* weights instead, so lambda is a direct, sweepable bound on
``||theta - theta_0||``. Freezing is the same idea with only two settings per
layer, which is why it cannot express "let L11 move a little".

`pool_layers="weighted"` changes *where* the pull lands. With the head on
`last_hidden_state` every gradient enters at the top and the extracted layer
takes the full force. A learned softmax over all hidden states lets the model
satisfy the target from wherever the information already is -- and eGeMAPS is
low-level, so that is the early, frozen layers. Read the learned weights after
training: they say which depth the objective actually wanted.

Both default to off, so an unflagged run reproduces the earlier checkpoints.

A brain-PCA multi-task variant used to live here and was removed on
2026-08-19 — training the encoder on brain responses and then using its
features for voxelwise encoding is circular. See `trash/brain_pca_multitask/`.
"""

from typing import List, Optional, Union

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel


def _layer_of(name: str) -> Union[int, str]:
    """Transformer layer index in a parameter name, else a coarse group."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return "other"


class _SpeechRegressorBase(PreTrainedModel):
    """Shared encoder handling: freezing, checkpointing, attention pooling."""

    config_class = AutoConfig

    def __init__(self, config, base_model_name: Optional[str] = None,
                 freeze_layers: Union[int, List[int], None] = 6,
                 truncate_layers: Optional[int] = None,
                 pool_layers: Optional[str] = None,
                 l2sp: Optional[float] = None, **kwargs):
        super().__init__(config, **kwargs)

        # Both settings change the module tree (`layer_weights` exists or does
        # not) or the loss, so a checkpoint has to be rebuilt with the values it
        # was trained under. Falling back to the config the way `num_features`
        # and `base_model_name` already do means `from_pretrained(dir)` with no
        # keywords reconstructs the right model instead of dropping a key.
        if pool_layers is None:
            pool_layers = getattr(config, "pool_layers", "last")
        if l2sp is None:
            l2sp = getattr(config, "l2sp", 0.0)

        if base_model_name is None:
            base_model_name = getattr(config, "base_model_name", None)
            if base_model_name is None:
                raise ValueError(
                    "base_model_name is required when creating a new model or "
                    "loading from a checkpoint that does not record it."
                )
        self.base_model_name = base_model_name

        # LayerDrop randomly skips whole transformer layers during training.
        # Under DDP that is fatal: the skipped layer's 16 parameters receive no
        # gradient, and the reducer aborts with "Expected to have finished
        # reduction in the prior iteration". It also makes every forward
        # stochastic in *which* layers ran, which is noise in a comparison whose
        # entire point is contrasting backbones. Both checkpoints ship 0.1.
        config.layerdrop = 0.0

        # SpecAugment masks input time steps. The target here is the eGeMAPS
        # functionals of that exact window, so masking removes the very signal
        # the label describes — and an unmasked batch leaves masked_spec_embed
        # without a gradient, which trips the same DDP check.
        config.apply_spec_augment = False

        self.encoder = AutoModel.from_pretrained(base_model_name, config=config)
        if truncate_layers is not None:
            self.truncate_encoder(truncate_layers)
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(0.1)

        # Learned temporal attention pooling: one scalar score per frame.
        self.temporal_attn = nn.Linear(self.hidden_size, 1)

        if pool_layers not in ("last", "weighted"):
            raise ValueError(
                f"pool_layers must be 'last' or 'weighted', got {pool_layers!r}")
        self.pool_layers = pool_layers
        if pool_layers == "weighted":
            # One logit per hidden state, and there are num_hidden_layers + 1
            # of those: index 0 is the CNN output, block i is at i + 1. Read
            # after truncation so the count matches the final stack.
            n_states = self.encoder.config.num_hidden_layers + 1
            self.layer_weights = nn.Parameter(torch.zeros(n_states))

        self.loss_fct = nn.MSELoss()
        # Truncation first, so freeze indices refer to the final stack.
        self.freeze_base_model(freeze_layers)

        # Anchors must be taken *after* freezing: only trainable parameters can
        # drift, so only those need a reference, and snapshotting the frozen
        # bottom would waste hundreds of MB on tensors that never move.
        self.l2sp = float(l2sp)
        self._anchor_names: dict = {}
        if self.l2sp > 0:
            self._register_anchors()

        self.config.base_model_name = base_model_name
        self.config.truncate_layers = truncate_layers
        self.config.pool_layers = pool_layers
        self.config.l2sp = self.l2sp

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

        # masked_spec_embed is only ever read when SpecAugment masks something,
        # and we disable SpecAugment. Left trainable it would sit in the DDP
        # reduction never receiving a gradient, which aborts the step exactly
        # the way a LayerDrop-skipped layer does.
        spec_embed = getattr(self.encoder, "masked_spec_embed", None)
        if spec_embed is not None and not getattr(
                self.encoder.config, "apply_spec_augment", False):
            spec_embed.requires_grad = False

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

    # -- drift control ------------------------------------------------------

    def _register_anchors(self):
        """Snapshot the pretrained value of every trainable encoder parameter.

        Registered as buffers so they follow the model through `.to(device)`
        and AMP, but with ``persistent=False`` so they stay out of the
        checkpoint: they are recoverable from the base model at any time, and
        saving them would double every checkpoint on disk.
        """
        n = 0
        for name, param in self.encoder.named_parameters():
            if not param.requires_grad:
                continue
            buf = "_anchor_" + name.replace(".", "__")
            self.register_buffer(buf, param.detach().clone(), persistent=False)
            self._anchor_names[name] = buf
            n += param.numel()
        print(f"L2-SP: anchored {n:,} trainable encoder parameters "
              f"({n * 4 / 1e6:.0f} MB)")

    def drift_penalty(self) -> Optional[torch.Tensor]:
        """``sum ||theta - theta_0||^2`` over the trainable encoder weights."""
        if self.l2sp <= 0 or not self._anchor_names:
            return None
        total = None
        for name, param in self.encoder.named_parameters():
            buf = self._anchor_names.get(name)
            if buf is None or not param.requires_grad:
                continue
            term = ((param - getattr(self, buf)) ** 2).sum()
            total = term if total is None else total + term
        return total

    @torch.no_grad()
    def drift_report(self) -> dict:
        """Relative distance from the pretrained weights, per transformer layer.

        The quantity the whole intervention is about, so it is logged rather
        than inferred: `l2sp` is a knob whose units are meaningless on their
        own, and this turns it into "layer 11 moved 3.2% of its norm".
        """
        by_layer: dict = {}
        for name, param in self.encoder.named_parameters():
            buf = self._anchor_names.get(name)
            if buf is None:
                continue
            anchor = getattr(self, buf)
            key = _layer_of(name)
            num, den = by_layer.get(key, (0.0, 0.0))
            by_layer[key] = (num + float(((param - anchor) ** 2).sum()),
                             den + float((anchor ** 2).sum()))
        return {k: (num / den) ** 0.5 if den > 0 else 0.0
                for k, (num, den) in sorted(by_layer.items(), key=str)}

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
        weighted = self.pool_layers == "weighted"
        outputs = self.encoder(input_values, attention_mask=attention_mask,
                               output_hidden_states=weighted)
        if weighted:
            states = torch.stack(outputs.hidden_states, dim=0)   # [L+1,B,T,D]
            w = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
            hidden = (states.to(w.dtype) * w).sum(dim=0)         # [B, T, D]
        else:
            hidden = outputs.last_hidden_state                   # [B, T, D]
        weights = torch.softmax(self.temporal_attn(hidden), dim=1)  # [B, T, 1]
        pooled = (hidden * weights).sum(dim=1)                   # [B, D]
        return self.dropout(pooled)

    @torch.no_grad()
    def layer_weight_profile(self) -> Optional[List[float]]:
        """The learned softmax over hidden states, or None if not in use."""
        if self.pool_layers != "weighted":
            return None
        return torch.softmax(self.layer_weights, dim=0).tolist()

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
                 truncate_layers: Optional[int] = None,
                 pool_layers: Optional[str] = None,
                 l2sp: Optional[float] = None, **kwargs):
        super().__init__(config, base_model_name=base_model_name,
                         freeze_layers=freeze_layers,
                         truncate_layers=truncate_layers,
                         pool_layers=pool_layers, l2sp=l2sp, **kwargs)

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
        loss = self.loss_fct(logits, labels)
        penalty = self.drift_penalty()
        if penalty is not None:
            # 1/2 lambda ||theta - theta_0||^2, the usual parameterisation, so
            # lambda is comparable to a weight-decay coefficient.
            loss = loss + 0.5 * self.l2sp * penalty
        return {"loss": loss, "logits": logits}

