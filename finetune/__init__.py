"""Fine-tuning self-supervised speech models on prosody and brain targets."""

from .registry import (EMOTION_DIMENSIONS, EMOTION_MODEL, MODEL_MAP, REGISTRY,
                       SpeechModelSpec, format_registry, resolve_base_model,
                       resolve_model)

__all__ = [
    "EMOTION_DIMENSIONS", "EMOTION_MODEL", "MODEL_MAP", "REGISTRY",
    "SpeechModelSpec", "format_registry", "resolve_base_model", "resolve_model",
]
