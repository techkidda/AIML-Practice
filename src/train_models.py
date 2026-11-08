from __future__ import annotations

import argparse
import importlib
import json
import logging
from typing import Any

import joblib

from src.dataset import PROJECT_ROOT, split_reference_dataset
from src.logging_config import configure_logging
from src.metrics import evaluate_predictions
from src.model_registry import MODEL_BY_SLUG, MODEL_DEFINITIONS
from src.preprocessing import build_pipeline

LOGGER = logging.getLogger("ml_practice.training")


def train_and_save(slug: str) -> dict[str, float | None]:
    if slug not in MODEL_BY_SLUG:
        raise ValueError(f"Unknown model {slug!r}. Choose from: {', '.join(MODEL_BY_SLUG)}.")
    definition = MODEL_BY_SLUG[slug]
    model_module = importlib.import_module(definition.module)
    model_class: Any = getattr(model_module, definition.class_name)
    classifier = model_class().build()

    x_train, x_test, y_train, y_test = split_reference_dataset()
    pipeline = build_pipeline(classifier, x_train)
    pipeline.fit(x_train, y_train)

    artifact_directory = PROJECT_ROOT / "models"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_directory / f"{slug}.joblib"
    joblib.dump(pipeline, artifact_path)

    metrics, _, _ = evaluate_predictions(pipeline, x_test, y_test)
    LOGGER.info("Trained %s and saved artifact to %s", definition.display_name, artifact_path)
    return metrics


def export_test_data() -> Path:
    _, x_test, _, y_test = split_reference_dataset()
    test_data = x_test.copy()
    test_data["Diagnosis"] = y_test
    destination = PROJECT_ROOT / "test_data.csv"
    test_data.to_csv(destination, index=False)
    LOGGER.info("Exported %d held-out rows to %s", len(test_data), destination)
    return destination


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Train the five reference classifiers.")
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_BY_SLUG),
        help="Train one model only; omit to train every model.",
    )
    arguments = parser.parse_args()
    export_test_data()

    definitions = (
        [MODEL_BY_SLUG[arguments.model]]
        if arguments.model
        else list(MODEL_DEFINITIONS)
    )
    results: dict[str, dict[str, float | None]] = {}
    for definition in definitions:
        results[definition.display_name] = train_and_save(definition.slug)

    results_path = PROJECT_ROOT / "metrics.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
