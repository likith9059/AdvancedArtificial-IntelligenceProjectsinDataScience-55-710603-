"""Week 7 evaluation utilities for freshly trained model."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
)

def collect_labels(dataset):
    return np.concatenate([labels.numpy().reshape(-1) for _, labels in dataset]).astype(int)

def collect_probabilities(model, dataset):
    return model.predict(dataset, verbose=0).reshape(-1)

def select_threshold(y_true, probabilities):
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.01):
        preds = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_true, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
            "defect_precision": float(precision_score(y_true, preds, pos_label=1, zero_division=0)),
            "defect_recall": float(recall_score(y_true, preds, pos_label=1, zero_division=0)),
            "f1_defect": float(f1_score(y_true, preds, pos_label=1, zero_division=0)),
            "f2_defect": float(fbeta_score(y_true, preds, beta=2, pos_label=1, zero_division=0)),
        })
    df = pd.DataFrame(rows).sort_values(["f2_defect", "defect_recall", "balanced_accuracy"], ascending=False)
    return float(df.iloc[0]["threshold"]), df
