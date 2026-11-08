import unittest

import pandas as pd

from src.model_report import build_model_observations


class ModelObservationTests(unittest.TestCase):
    def test_selects_winner_by_cv_f1_and_describes_each_model(self):
        results = pd.DataFrame(
            [
                {
                    "ML Model Name": "Logistic Regression",
                    "Accuracy": 0.8,
                    "F1": 0.79,
                    "MCC": 0.6,
                    "CV F1 (mean ± std)": "0.810 ± 0.020",
                },
                {
                    "ML Model Name": "Decision Tree",
                    "Accuracy": 0.9,
                    "F1": 0.88,
                    "MCC": 0.8,
                    "CV F1 (mean ± std)": "0.800 ± 0.050",
                },
            ]
        )

        observations, winner_name, winner_summary = build_model_observations(results)

        self.assertEqual(winner_name, "Logistic Regression")
        self.assertEqual(len(observations), 2)
        self.assertIn("0.810 ± 0.020", observations.iloc[0, 1])
        self.assertIn("highest training cross-validation macro-F1", winner_summary)
        self.assertIn("not proof of statistically significant superiority", winner_summary)

    def test_rejects_missing_cv_selection_metric(self):
        results = pd.DataFrame(
            [{"ML Model Name": "KNN", "Accuracy": 0.8, "F1": 0.7, "MCC": 0.6}]
        )

        with self.assertRaisesRegex(ValueError, "CV F1"):
            build_model_observations(results)


if __name__ == "__main__":
    unittest.main()
