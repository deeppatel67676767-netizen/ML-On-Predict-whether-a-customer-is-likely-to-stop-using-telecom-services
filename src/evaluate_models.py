"""
Evaluate trained models and return metrics + best model selection.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


def evaluate_model(model, X_test, y_test) -> dict:
    """Return a metrics dict for one model."""
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1 Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }

    # ROC-AUC requires probability estimates
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics["ROC-AUC"] = round(roc_auc_score(y_test, y_prob), 4)
    else:
        metrics["ROC-AUC"] = None

    return metrics


def evaluate_all_models(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate all models and return a comparison DataFrame."""
    rows = []
    for name, model in trained_models.items():
        m = evaluate_model(model, X_test, y_test)
        m["Model"] = name
        rows.append(m)

    df = pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"])
    return df


def select_best_model(metrics_df: pd.DataFrame, trained_models: dict):
    """
    Select the best model by F1 Score. Break ties using ROC-AUC then Recall.
    Returns (best_model_name, best_model_object).
    """
    df = metrics_df.copy()

    # Fill None ROC-AUC with 0 for sorting
    df["ROC-AUC_sort"] = df["ROC-AUC"].fillna(0)

    df_sorted = df.sort_values(
        by=["F1 Score", "ROC-AUC_sort", "Recall"],
        ascending=False
    )

    best_name = df_sorted.iloc[0]["Model"]
    return best_name, trained_models[best_name]


def get_confusion_matrix(model, X_test, y_test) -> np.ndarray:
    """Return confusion matrix as numpy array."""
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)
