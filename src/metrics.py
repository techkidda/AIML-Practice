from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRIC_NAMES = ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")


def evaluate_predictions(
    estimator: Any, features: pd.DataFrame, target: pd.Series
) -> tuple[dict[str, float | None], np.ndarray, list[Any]]:
    """Return classification metrics and a labeled confusion matrix."""
    predictions = estimator.predict(features)
    labels = list(estimator.classes_)
    scores = None

    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(features)
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(features)

    auc: float | None = None
    try:
        if scores is not None and len(labels) == 2:
            positive_scores = scores[:, 1] if np.asarray(scores).ndim == 2 else scores
            auc = float(roc_auc_score(target, positive_scores))
        elif scores is not None and len(labels) > 2 and np.asarray(scores).ndim == 2:
            auc = float(
                roc_auc_score(
                    target,
                    scores,
                    labels=labels,
                    multi_class="ovr",
                    average="weighted",
                )
            )
    except ValueError:
        auc = None

    metrics: dict[str, float | None] = {
        "Accuracy": float(accuracy_score(target, predictions)),
        "AUC": auc,
        "Precision": float(
            precision_score(target, predictions, average="weighted", zero_division=0)
        ),
        "Recall": float(
            recall_score(target, predictions, average="weighted", zero_division=0)
        ),
        "F1": float(f1_score(target, predictions, average="weighted", zero_division=0)),
        "MCC": float(matthews_corrcoef(target, predictions)),
    }
    matrix = confusion_matrix(target, predictions, labels=labels)
    return metrics, matrix, labels
