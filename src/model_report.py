from __future__ import annotations

import re

import pandas as pd


_CROSS_VALIDATION_SCORE = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*±\s*(-?\d+(?:\.\d+)?)\s*$"
)


def build_model_observations(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str]:
    """Summarize URL experiment results and choose by training CV macro-F1."""
    required_columns = {
        "ML Model Name",
        "Accuracy",
        "F1",
        "MCC",
        "CV F1 (mean ± std)",
    }
    missing_columns = required_columns - set(results.columns)
    if missing_columns:
        raise ValueError(
            "Cannot summarize model results; missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    if results.empty:
        raise ValueError("Cannot summarize an empty model results table.")

    scored_models = []
    for _, row in results.iterrows():
        match = _CROSS_VALIDATION_SCORE.fullmatch(str(row["CV F1 (mean ± std)"]))
        if match is None:
            raise ValueError(
                f"Invalid cross-validation F1 value for {row['ML Model Name']!r}."
            )
        mean_score, standard_deviation = map(float, match.groups())
        scored_models.append(
            {
                "name": str(row["ML Model Name"]),
                "mean": mean_score,
                "std": standard_deviation,
                "accuracy": float(row["Accuracy"]),
                "f1": float(row["F1"]),
                "mcc": float(row["MCC"]),
            }
        )

    winner = max(scored_models, key=lambda model: (model["mean"], -model["std"]))
    observations = []
    for model in scored_models:
        description = (
            f"Training CV macro-F1: {model['mean']:.3f} ± {model['std']:.3f}; "
            f"holdout macro-F1: {model['f1']:.3f}, accuracy: "
            f"{model['accuracy']:.3f}, MCC: {model['mcc']:.3f}."
        )
        if model["name"] == winner["name"]:
            description = "Preferred by training CV macro-F1. " + description
        observations.append(
            {"ML Model Name": model["name"], "Observation about model performance": description}
        )

    winner_summary = (
        f"{winner['name']} is the preferred model because it had the highest "
        f"training cross-validation macro-F1 ({winner['mean']:.3f} ± "
        f"{winner['std']:.3f}). Its untouched holdout macro-F1 was "
        f"{winner['f1']:.3f} (accuracy {winner['accuracy']:.3f}, "
        f"MCC {winner['mcc']:.3f}). This is a practical selection criterion, "
        "not proof of statistically significant superiority."
    )
    return pd.DataFrame(observations), winner["name"], winner_summary
