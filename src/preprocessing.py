from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(estimator: Any, features: pd.DataFrame) -> Pipeline:
    """Build a training-only preprocessing and estimator pipeline."""
    numeric_features = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [name for name in features.columns if name not in numeric_features]

    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers, remainder="drop")),
            ("classifier", estimator),
        ]
    )
