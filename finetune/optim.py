"""
Layer-wise learning-rate decay (LLRD) for the speech encoder.

Hard freezing is a blunt instrument: a layer is either fully plastic or fully
fixed. LLRD is the softer, usually better-behaved version of the same idea —
every layer trains, but lower layers get exponentially smaller learning rates:

    lr(layer i) = base_lr * decay ** (n_layers - 1 - i)

so the top layer trains at `base_lr` and, with `decay=0.9` over 24 layers, the
bottom trains ~10x slower.

Why it matters for this project specifically
--------------------------------------------
The fine-tuning target is 88 eGeMAPS functionals — F0, loudness, jitter,
shimmer, spectral slopes. These are *low-level acoustic* descriptors, and the
information needed to predict them already sits in the early layers. Unfreeze
too much and the network can take the shortcut: reshape its whole representation
to emit those 88 numbers, discarding the richer structure that made the
pretrained features worth using for brain prediction in the first place. The
result fits eGeMAPS beautifully and encodes the brain *worse* than the frozen
baseline.

LLRD keeps the lower layers close to their pretrained solution while letting the
upper layers — the ones feature extraction reads from — adapt. Combine with
`--freeze-layers` or use it instead; freezing wins where both apply, since a
frozen parameter has no gradient regardless of its learning rate.
"""

from typing import List, Optional

from torch import nn

#: Parameters that conventionally receive no weight decay.
NO_DECAY = ("bias", "LayerNorm.weight", "layer_norm.weight")


def _no_decay(name: str) -> bool:
    return any(token in name for token in NO_DECAY)


def _layer_index(name: str) -> Optional[int]:
    """Transformer layer index encoded in a parameter name, if any."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def build_llrd_param_groups(model: nn.Module, base_lr: float, decay: float,
                            weight_decay: float = 0.0) -> List[dict]:
    """Optimizer parameter groups implementing layer-wise LR decay.

    Parameters
    ----------
    model : nn.Module
        An `AudioEncoderForProsody`.
    base_lr : float
        Learning rate for the heads and the topmost encoder layer.
    decay : float
        Per-layer multiplier, in (0, 1]. 1.0 disables decay.
    weight_decay : float
        Applied to everything except biases and LayerNorm weights.

    Returns
    -------
    list of dict
        Ready to hand to `torch.optim.AdamW`. Frozen parameters are omitted.
    """
    if not 0 < decay <= 1:
        raise ValueError(f"llrd decay must be in (0, 1], got {decay}")

    layers = None
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layers"):
            layers = encoder.encoder.layers
        elif hasattr(encoder, "layers"):
            layers = encoder.layers
    n_layers = len(layers) if layers is not None else 0

    groups: dict = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        index = _layer_index(name) if name.startswith("encoder.") else None
        if index is not None and n_layers:
            # Deeper layer -> smaller exponent -> larger lr.
            lr = base_lr * (decay ** (n_layers - 1 - index))
            tag = f"layer{index}"
        elif name.startswith("encoder."):
            # Anything else inside the encoder (embeddings, final norm) sits
            # at the bottom of the stack and gets the smallest rate.
            lr = base_lr * (decay ** n_layers)
            tag = "encoder_other"
        else:
            # Heads and pooling train at the full rate: they are new.
            lr = base_lr
            tag = "head"

        decayed = not _no_decay(name)
        key = (tag, round(lr, 12), decayed)
        groups.setdefault(
            key,
            {"params": [], "lr": lr,
             "weight_decay": weight_decay if decayed else 0.0,
             "name": f"{tag}{'' if decayed else '_nodecay'}"},
        )["params"].append(param)

    return sorted(groups.values(), key=lambda g: -g["lr"])


def summarize_param_groups(groups: List[dict]) -> str:
    lines = []
    for g in groups:
        n = sum(p.numel() for p in g["params"])
        lines.append(f"    {g['name']:16s} lr={g['lr']:.3e}  "
                     f"wd={g['weight_decay']:.3g}  {n:,} params")
    return "\n".join(lines)


class LLRDTrainerMixin:
    """Mixin giving a HF `Trainer` layer-wise LR decay.

    Set `trainer.llrd_decay` before training; when it is None the base
    implementation is used unchanged.
    """

    llrd_decay: Optional[float] = None

    def create_optimizer(self):
        if self.llrd_decay is None or self.optimizer is not None:
            return super().create_optimizer()

        groups = build_llrd_param_groups(
            self.model,
            base_lr=self.args.learning_rate,
            decay=self.llrd_decay,
            weight_decay=self.args.weight_decay,
        )
        print(f"  LLRD decay={self.llrd_decay}: "
              f"{len(groups)} parameter groups")
        print(summarize_param_groups(groups[:3]))
        print(f"    ... lowest lr = {groups[-1]['lr']:.3e}")

        optimizer_cls, kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        kwargs.pop("lr", None)
        kwargs.pop("weight_decay", None)
        self.optimizer = optimizer_cls(groups, lr=self.args.learning_rate,
                                       **kwargs)
        return self.optimizer
