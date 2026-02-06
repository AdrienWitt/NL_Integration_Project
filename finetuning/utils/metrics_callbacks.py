import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import TrainerCallback


def compute_metrics_with_names(eval_pred, feature_names):
    predictions, labels = eval_pred
    metrics = {"eval_loss": mean_squared_error(labels, predictions)}

    for i, feat in enumerate(feature_names):
        p, t = predictions[:, i], labels[:, i]
        metrics.update({
            f"{feat}_mse":  mean_squared_error(t, p),
            f"{feat}_rmse": np.sqrt(mean_squared_error(t, p)),
            f"{feat}_mae":  mean_absolute_error(t, p),
            f"{feat}_r2":   r2_score(t, p),
        })

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
