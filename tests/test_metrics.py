import unittest

import pandas as pd
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

        self.assertEqual(tuple(metrics), METRIC_NAMES)
        self.assertTrue(all(value is not None for value in metrics.values()))
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(set(labels), {"benign", "malignant"})
        self.assertEqual(metrics["Accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
