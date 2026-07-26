"""Week 4 evaluation utilities for Xception transfer learning."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

def collect_labels(dataset):
    return np.concatenate([labels.numpy().reshape(-1) for _, labels in dataset]).astype(int)

def collect_probabilities(model, dataset):
    return model.predict(dataset, verbose=0).reshape(-1)

def evaluate_binary_model(model, dataset, threshold=0.5):
    y_true = collect_labels(dataset)
    probabilities = collect_probabilities(model, dataset)
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "defect_precision": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "defect_recall": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "f1_defect": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }

    report_df = pd.DataFrame(
        classification_report(
            y_true,
            predictions,
            target_names=["ok_front", "def_front"],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual ok_front", "Actual def_front"],
        columns=["Pred ok_front", "Pred def_front"],
    )

    return metrics, report_df, cm_df
