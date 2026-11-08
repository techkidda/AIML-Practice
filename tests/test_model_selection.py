import unittest

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.model_selection import fit_with_cross_validation
from src.models.support_vector_machine import SupportVectorMachineModel


class ModelSelectionTests(unittest.TestCase):
    def test_grid_search_returns_refitted_pipeline_parameters_and_cv_spread(self):
        features = pd.DataFrame(
            {
                "signal": list(range(40)),
                "noise": [value % 3 for value in range(40)],
            }
        )
        target = pd.Series(["low"] * 20 + ["high"] * 20)

        model, cross_validation, parameters = fit_with_cross_validation(
            "logistic_regression",
            LogisticRegression(max_iter=1500, random_state=69),
            features,
            target,
        )

        self.assertIn("classifier__C", parameters)
        self.assertIn("Accuracy", cross_validation)
        self.assertGreaterEqual(cross_validation["Accuracy"]["mean"], 0.0)
        self.assertGreaterEqual(cross_validation["Accuracy"]["std"], 0.0)
        self.assertEqual(model.predict(features).shape, target.shape)

    def test_svm_handles_three_sample_classes_after_the_holdout_split(self):
        features = pd.DataFrame({"signal": range(8)})
        target = pd.Series(["M"] * 2 + ["B"] * 6)

        model, cross_validation, parameters = fit_with_cross_validation(
            "support_vector_machine",
            SupportVectorMachineModel.build(),
            features,
            target,
        )

        self.assertIn("classifier__C", parameters)
        self.assertIn("Malignant Recall (M)", cross_validation)
        self.assertEqual(model.predict(features).shape, target.shape)

    def test_multiclass_grid_search_reports_cross_validation_auc(self):
        features = pd.DataFrame(
            {
                "signal": list(range(60)),
                "cycle": [value % 4 for value in range(60)],
            }
        )
        target = pd.Series(["class_a"] * 20 + ["class_b"] * 20 + ["class_c"] * 20)

        model, cross_validation, _ = fit_with_cross_validation(
            "logistic_regression",
            LogisticRegression(max_iter=1500, random_state=69),
            features,
            target,
        )

        self.assertIn("AUC", cross_validation)
        self.assertGreaterEqual(cross_validation["AUC"]["mean"], 0.0)
        self.assertEqual(model.predict(features).shape, target.shape)


if __name__ == "__main__":
    unittest.main()
