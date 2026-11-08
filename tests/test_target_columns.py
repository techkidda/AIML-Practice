import unittest

import pandas as pd

from src.target_columns import assess_target_columns


class TargetColumnAssessmentTests(unittest.TestCase):
    def test_only_complete_targets_with_enough_samples_per_class_are_eligible(self):
        dataset = pd.DataFrame(
            {
                "good_target": ["a"] * 5 + ["b"] * 5,
                "missing_target": ["a"] * 9 + [None],
                "rare_target": ["a"] * 8 + ["b", "c"],
                "identifier": list(range(10)),
            }
        )

        assessments = {
            assessment.column: assessment
            for assessment in assess_target_columns(dataset)
        }

        self.assertTrue(assessments["good_target"].eligible)
        self.assertEqual(assessments["good_target"].classes, 2)
        self.assertEqual(assessments["good_target"].minimum_class_samples, 5)
        self.assertFalse(assessments["missing_target"].eligible)
        self.assertEqual(assessments["missing_target"].missing_values, 1)
        self.assertIn("missing target values", assessments["missing_target"].reason)
        self.assertFalse(assessments["rare_target"].eligible)
        self.assertIn("smallest class has 1", assessments["rare_target"].reason)
        self.assertFalse(assessments["identifier"].eligible)
        self.assertIn("smallest class has 1", assessments["identifier"].reason)

    def test_target_is_rejected_if_holdout_cannot_represent_all_classes(self):
        dataset = pd.DataFrame(
            {
                "target": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"],
                "feature": range(10),
            }
        )

        assessments = {
            assessment.column: assessment
            for assessment in assess_target_columns(dataset)
        }

        self.assertFalse(assessments["target"].eligible)
        self.assertIn("too many classes", assessments["target"].reason)


if __name__ == "__main__":
    unittest.main()
