"""Week 5 model evaluation utilities."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def collect_labels(dataset):
    return np.concatenate([labels.numpy().reshape(-1) for _, labels in dataset]).astype(int)

def collect_probabilities(model, dataset):
    return model.predict(dataset, verbose=0).reshape(-1)

def evaluate_models(trained_models, validation_ds, training_times):
    y_val = collect_labels(validation_ds)
    rows = []
    for name, model in trained_models.items():
        p = collect_probabilities(model, validation_ds)
        pred = (p >= 0.5).astype(int)
        rows.append({"model": name, "accuracy": float(accuracy_score(y_val, pred)), "defect_precision": float(precision_score(y_val, pred, pos_label=1, zero_division=0)), "defect_recall": float(recall_score(y_val, pred, pos_label=1, zero_division=0)), "f1_defect": float(f1_score(y_val, pred, pos_label=1, zero_division=0)), "roc_auc": float(roc_auc_score(y_val, p)), "pr_auc": float(average_precision_score(y_val, p)), "training_minutes": float(training_times.get(name, 0))})
    return pd.DataFrame(rows).sort_values(["pr_auc", "defect_recall", "f1_defect"], ascending=False)
