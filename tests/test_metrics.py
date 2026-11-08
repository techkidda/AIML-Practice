import unittest

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.metrics import METRIC_NAMES, evaluate_predictions


class EvaluationMetricsTests(unittest.TestCase):
    def test_evaluate_predictions_returns_all_required_metrics(self):
        features = pd.DataFrame({"signal": [0, 1, 2, 3, 4, 5]})
        target = pd.Series(
            ["benign", "benign", "benign", "malignant", "malignant", "malignant"]
        )
        estimator = make_pipeline(StandardScaler(), LogisticRegression()).fit(
            features, target
        )

        metrics, matrix, labels = evaluate_predictions(estimator, features, target)

        self.assertTrue(set(METRIC_NAMES).issubset(metrics))
        self.assertTrue(all(metrics[name] is not None for name in METRIC_NAMES))
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(set(labels), {"benign", "malignant"})
        self.assertEqual(metrics["Accuracy"], 1.0)

    def test_binary_metrics_report_macro_recall_and_malignant_recall(self):
        features = pd.DataFrame({"signal": range(10)})
        target = pd.Series(["B"] * 9 + ["M"])
        estimator = DummyClassifier(strategy="most_frequent").fit(features, target)

        metrics, _, _ = evaluate_predictions(estimator, features, target)

        self.assertEqual(metrics["Accuracy"], 0.9)
        self.assertEqual(metrics["Recall"], 0.5)
        self.assertEqual(metrics["Malignant Recall (M)"], 0.0)


if __name__ == "__main__":
    unittest.main()
