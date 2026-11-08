import json
import tempfile
import unittest
from pathlib import Path

from src.train_models import persist_metrics


class TrainingMetricsPersistenceTests(unittest.TestCase):
    def test_single_model_run_preserves_other_model_results(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_path = Path(temporary_directory) / "metrics.json"
            original = {
                "Logistic Regression": {"Accuracy": 0.9},
                "Decision Tree": {"Accuracy": 0.8},
            }
            results_path.write_text(json.dumps(original), encoding="utf-8")

            updated = persist_metrics(
                results_path,
                {"Logistic Regression": {"Accuracy": 0.95}},
                merge_existing=True,
            )

            self.assertEqual(updated["Decision Tree"], {"Accuracy": 0.8})
            self.assertEqual(updated["Logistic Regression"], {"Accuracy": 0.95})

    def test_full_training_run_replaces_stale_results(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_path = Path(temporary_directory) / "metrics.json"
            results_path.write_text(
                json.dumps({"Obsolete model": {"Accuracy": 0.1}}),
                encoding="utf-8",
            )

            updated = persist_metrics(
                results_path,
                {"Naive Bayes": {"Accuracy": 0.7}},
                merge_existing=False,
            )

            self.assertEqual(updated, {"Naive Bayes": {"Accuracy": 0.7}})


if __name__ == "__main__":
    unittest.main()
