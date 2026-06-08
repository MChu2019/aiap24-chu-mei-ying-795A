import json
from pathlib import Path

import numpy as np
import pandas as pd


def classification_metrics(y_true, y_pred, y_score):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(y_score)) + 1
    n_pos = max((y_true == 1).sum(), 1)
    n_neg = max((y_true == 0).sum(), 1)
    auc = (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def regression_metrics(y_true, y_pred):
    error = y_pred - y_true
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    ss_res = np.sum(error**2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_predictions(ids, y_true, y_pred, path, y_score=None):
    frame = pd.DataFrame({"delivery_id": ids, "y_true": y_true, "y_pred": y_pred})
    if y_score is not None:
        frame["y_score"] = y_score
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
