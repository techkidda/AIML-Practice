from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


MINIMUM_TARGET_CLASS_SAMPLES = 3
MINIMUM_DATASET_ROWS = 10
HOLDOUT_FRACTION = 0.2


@dataclass(frozen=True)
class TargetColumnAssessment:
    column: str
    classes: int
    minimum_class_samples: int | None
    missing_values: int
    eligible: bool
    reason: str


def assess_target_columns(
    dataset: pd.DataFrame,
) -> list[TargetColumnAssessment]:
    """Assess each column as a stratified classification target."""
    assessments = []
    row_count = len(dataset)
    test_row_count = math.ceil(row_count * HOLDOUT_FRACTION)
    train_row_count = row_count - test_row_count

    for column in dataset.columns:
        target = dataset[column]
        missing_values = int(target.isna().sum())
        counts = target.value_counts(dropna=True)
        class_count = len(counts)
        minimum_samples = int(counts.min()) if class_count else None
        reason = ""

        if row_count < MINIMUM_DATASET_ROWS:
            reason = f"Dataset has fewer than {MINIMUM_DATASET_ROWS} rows."
        elif missing_values:
            reason = f"Contains {missing_values:,} missing target values."
        elif class_count < 2:
            reason = "Needs at least two distinct classes."
        elif minimum_samples is not None and minimum_samples < MINIMUM_TARGET_CLASS_SAMPLES:
            reason = (
                f"At least {MINIMUM_TARGET_CLASS_SAMPLES} rows are needed for every class; "
                f"the smallest class has {minimum_samples}."
            )
        elif class_count > min(train_row_count, test_row_count):
            reason = (
                "There are too many classes for the stratified train/holdout split "
                f"({class_count} classes, {test_row_count} holdout rows)."
            )

        assessments.append(
            TargetColumnAssessment(
                column=str(column),
                classes=class_count,
                minimum_class_samples=minimum_samples,
                missing_values=missing_values,
                eligible=not reason,
                reason=reason,
            )
        )

    return assessments
