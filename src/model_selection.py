from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from src.preprocessing import build_pipeline


MODEL_PARAMETER_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logistic_regression": {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
    },
    "decision_tree": {
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_leaf": [1, 3, 5],
        "classifier__class_weight": [None, "balanced"],
    },
    "knn": {
        "classifier__n_neighbors": [3, 5, 7, 11],
        "classifier__weights": ["uniform", "distance"],
    },
    "support_vector_machine": {
        "classifier__estimator__C": [0.1, 1.0, 10.0],
        "classifier__estimator__gamma": ["scale", "auto"],
        "classifier__estimator__class_weight": [None, "balanced"],
    },
}


def fit_with_cross_validation(
    slug: str, classifier: Any, features: pd.DataFrame, target: pd.Series
) -> tuple[Any, dict[str, dict[str, float]], dict[str, Any]]:
    """Tune on stratified training folds and refit the selected pipeline."""
    minimum_class_size = int(target.value_counts().min())
    split_count = min(5, minimum_class_size)
    if split_count < 2:
        raise ValueError("Cross-validation requires at least two training rows per class.")

    scoring: dict[str, Any] = {
        "Accuracy": "accuracy",
        "AUC": make_scorer(
            roc_auc_score,
            response_method=("predict_proba", "decision_function"),
            multi_class="ovr",
            average="weighted",
        ),
        "Precision": make_scorer(
            precision_score, average="macro", zero_division=0
        ),
        "Recall": make_scorer(recall_score, average="macro", zero_division=0),
        "F1": make_scorer(f1_score, average="macro", zero_division=0),
        "MCC": make_scorer(matthews_corrcoef),
    }
    malignant_recall = (
        len(target.unique()) == 2 and "M" in set(target.astype(str).unique())
    )
    if malignant_recall:
        scoring["Malignant Recall (M)"] = make_scorer(
            recall_score, pos_label="M", zero_division=0
        )

    selection_metric = "Malignant Recall (M)" if malignant_recall else "F1"
    cross_validation = StratifiedKFold(
        n_splits=split_count, shuffle=True, random_state=42
    )
    parameter_grid = dict(MODEL_PARAMETER_GRIDS.get(slug, {}))
    if slug == "support_vector_machine" and minimum_class_size < 4:
        classifier = SVC(kernel="rbf", random_state=69)
        parameter_grid = {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__gamma": ["scale", "auto"],
            "classifier__class_weight": [None, "balanced"],
        }
    if slug == "knn":
        maximum_neighbors = len(target) - (len(target) + split_count - 1) // split_count
        parameter_grid["classifier__n_neighbors"] = [
            neighbors
            for neighbors in parameter_grid["classifier__n_neighbors"]
            if neighbors <= maximum_neighbors
        ] or [1]

    search = GridSearchCV(
        build_pipeline(classifier, features),
        param_grid=parameter_grid,
        scoring=scoring,
        refit=selection_metric,
        cv=cross_validation,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(features, target)

    best_index = search.best_index_
    summary = {
        metric: {
            "mean": float(search.cv_results_[f"mean_test_{metric}"][best_index]),
            "std": float(search.cv_results_[f"std_test_{metric}"][best_index]),
        }
        for metric in scoring
    }
    return search.best_estimator_, summary, search.best_params_
