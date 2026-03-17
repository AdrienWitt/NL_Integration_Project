import os
import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import TrainerCallback


def compute_metrics_with_names(eval_pred, feature_names):
    """Single-task prosody metrics — naming consistent with multitask mode."""
    predictions, labels = eval_pred

    r2s, rs, mses = [], [], []
    metrics = {}

    for i, feat in enumerate(feature_names):
        p, t = predictions[:, i], labels[:, i]
        mse = float(mean_squared_error(t, p))
        r2  = float(r2_score(t, p))
        r   = float(pearsonr(p, t)[0]) if np.std(p) > 0 and np.std(t) > 0 else 0.0
        metrics[f"{feat}_mse"]  = mse
        metrics[f"{feat}_rmse"] = float(np.sqrt(mse))
        metrics[f"{feat}_mae"]  = float(mean_absolute_error(t, p))
        metrics[f"{feat}_r2"]   = r2
        metrics[f"{feat}_r"]    = r
        r2s.append(r2)
        rs.append(r)
        mses.append(mse)

    metrics["mean_r2"]  = float(np.mean(r2s))
    metrics["mean_r"]   = float(np.mean(rs))
    metrics["mean_mse"] = float(np.mean(mses))
    metrics["eval_loss"]        = float(np.mean(mses))  # ← used by EarlyStopping + best model

    return metrics


def compute_multitask_metrics(predictions, labels, prosody_feature_names, num_prosody_features):
    """
    Dual-task metrics for joint prosody + brain PCA prediction.
    Expects predictions and labels both shaped [N, num_prosody + num_brain].
    """
    prosody_preds  = predictions[:, :num_prosody_features]
    brain_preds    = predictions[:, num_prosody_features:]
    prosody_labels = labels[:, :num_prosody_features]
    brain_labels   = labels[:, num_prosody_features:]

    metrics = {}

    # ── Prosody ──────────────────────────────────────────────────────────
    prosody_r2s, prosody_rs, prosody_mses = [], [], []
    for i, fname in enumerate(prosody_feature_names):
        p_i, t_i = prosody_preds[:, i], prosody_labels[:, i]
        mse = float(mean_squared_error(t_i, p_i))
        r2  = float(r2_score(t_i, p_i))
        r   = float(pearsonr(p_i, t_i)[0]) if np.std(p_i) > 0 and np.std(t_i) > 0 else 0.0
        metrics[f"{fname}_mse"]  = mse        # ← removed prosody_
        metrics[f"{fname}_rmse"] = float(np.sqrt(mse))
        metrics[f"{fname}_mae"]  = float(mean_absolute_error(t_i, p_i))
        metrics[f"{fname}_r2"]   = r2
        metrics[f"{fname}_r"]    = r
        prosody_r2s.append(r2)
        prosody_rs.append(r)
        prosody_mses.append(mse)

    metrics["mean_r2"]  = float(np.mean(prosody_r2s))   # ← removed prosody_
    metrics["mean_r"]   = float(np.mean(prosody_rs))
    metrics["mean_mse"] = float(np.mean(prosody_mses))
    metrics["eval_loss"] = float(np.mean(prosody_mses))

    # ── Brain PCA ────────────────────────────────────────────────────────
    if brain_labels.shape[1] > 0:
        brain_r2s, brain_rs, brain_mses = [], [], []
        for i in range(brain_labels.shape[1]):
            p_i, t_i = brain_preds[:, i], brain_labels[:, i]
            mse = float(mean_squared_error(t_i, p_i))
            r2  = float(r2_score(t_i, p_i))
            r   = float(pearsonr(p_i, t_i)[0]) if np.std(p_i) > 0 and np.std(t_i) > 0 else 0.0
            metrics[f"brain_PC{i+1}_mse"]  = mse
            metrics[f"brain_PC{i+1}_rmse"] = float(np.sqrt(mse))
            metrics[f"brain_PC{i+1}_mae"]  = float(mean_absolute_error(t_i, p_i))
            metrics[f"brain_PC{i+1}_r2"]   = r2
            metrics[f"brain_PC{i+1}_r"]    = r
            brain_r2s.append(r2)
            brain_rs.append(r)
            brain_mses.append(mse)

        metrics["brain_mean_r2"]  = float(np.mean(brain_r2s))
        metrics["brain_mean_r"]   = float(np.mean(brain_rs))
        metrics["brain_mean_mse"] = float(np.mean(brain_mses))

    return metrics


class MetricsCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics_history = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metrics["epoch"] = state.epoch
        self.metrics_history.append(metrics)

        pd.DataFrame(self.metrics_history).to_csv(
            os.path.join(self.output_dir, "metrics_history.csv"), index=False
        )
        with open(os.path.join(self.output_dir, f"metrics_epoch_{state.epoch:.1f}.json"), "w") as f:
            json.dump(metrics, f, indent=2)