"""
Evaluation metrics and per-epoch metric logging.

Per-target r is the metric that matters here. MSE alone hides the failure mode
these models actually have: predicting close to the mean of every feature,
which yields a respectable MSE and an r of ~0. Reporting r per feature makes
that immediately visible.

Note on `eval_loss`
-------------------
`compute_prosody_metrics` reports `mean_mse` but deliberately does *not* try to
set `eval_loss`. It used to, on the assumption that the Trainer keeps a key
already carrying the `eval_` prefix — it does not: `Trainer.evaluation_loop`
overwrites `metrics["eval_loss"]` with the model's own averaged loss *after*
`compute_metrics` returns (transformers 4.50, `trainer.py:4435`). The two
values were near-identical anyway, since the training loss is that same MSE.

Select on `mean_r` (via `--metric-for-best`) when you care about the failure
mode above: MSE is minimised by predicting each feature's mean, and `mean_r`
is not.
"""

import json
import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import TrainerCallback


def _per_target_metrics(preds: np.ndarray, labels: np.ndarray,
                        names: Sequence[str], prefix: str = "") -> Dict:
    """Per-column mse/rmse/mae/r2/r, plus the lists needed for the averages."""
    metrics: Dict[str, float] = {}
    r2s, rs, mses = [], [], []

    for i, name in enumerate(names):
        p, t = preds[:, i], labels[:, i]
        mse = float(mean_squared_error(t, p))
        r2 = float(r2_score(t, p))
        # pearsonr is undefined for a constant vector; a collapsed prediction
        # is a real result (r = 0), not an error.
        r = float(pearsonr(p, t)[0]) if np.std(p) > 0 and np.std(t) > 0 else 0.0

        key = f"{prefix}{name}"
        metrics[f"{key}_mse"] = mse
        metrics[f"{key}_rmse"] = float(np.sqrt(mse))
        metrics[f"{key}_mae"] = float(mean_absolute_error(t, p))
        metrics[f"{key}_r2"] = r2
        metrics[f"{key}_r"] = r

        r2s.append(r2)
        rs.append(r)
        mses.append(mse)

    return {"metrics": metrics, "r2": r2s, "r": rs, "mse": mses}


def compute_prosody_metrics(eval_pred, feature_names: Sequence[str]) -> Dict:
    """Single-task metrics over the prosody features."""
    predictions, labels = eval_pred
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    per = _per_target_metrics(predictions, labels, feature_names)
    metrics = per["metrics"]
    metrics["mean_r2"] = float(np.mean(per["r2"]))
    metrics["mean_r"] = float(np.mean(per["r"]))
    metrics["mean_mse"] = float(np.mean(per["mse"]))
    return metrics


class MetricsCallback(TrainerCallback):
    """Append every evaluation to a CSV and dump a JSON per epoch."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.history: List[Dict] = []
        os.makedirs(output_dir, exist_ok=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        record = dict(metrics)
        record["epoch"] = state.epoch
        self.history.append(record)

        pd.DataFrame(self.history).to_csv(
            os.path.join(self.output_dir, "metrics_history.csv"), index=False
        )
        epoch_file = os.path.join(
            self.output_dir, f"metrics_epoch_{state.epoch:.1f}.json"
        )
        with open(epoch_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
