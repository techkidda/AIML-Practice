from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "breast_cancer_wisconsin_diagnostic.csv"
TARGET_COLUMN = "Diagnosis"
IGNORED_COLUMNS = ("ID",)
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_reference_dataset() -> pd.DataFrame:
    dataset = pd.read_csv(DATASET_PATH)
    dataset = dataset.drop(columns=list(IGNORED_COLUMNS), errors="ignore")
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Dataset is missing the target column {TARGET_COLUMN!r}.")
    return dataset


def split_reference_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = load_reference_dataset()
    features = dataset.drop(columns=TARGET_COLUMN)
    target = dataset[TARGET_COLUMN]
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
