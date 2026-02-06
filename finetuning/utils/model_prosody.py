"""Compatibility wrapper re-exporting split modules.

This file previously contained dataset, model and training code. That logic
has been moved into smaller modules for clearer data-science workflows.

Keep this file so existing imports that reference `model_prosody` continue to work.
"""

from .prosody_dataset import ProsodyDataset
from .models import AudioEncoderForProsody
from .training import train_model
from .metrics_callbacks import compute_metrics_with_names, MetricsCallback

__all__ = [
    "ProsodyDataset",
    "AudioEncoderForProsody",
    "train_model",
    "compute_metrics_with_names",
    "MetricsCallback",
]