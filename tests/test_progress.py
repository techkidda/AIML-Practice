import unittest

from src.progress import estimate_remaining_seconds, format_duration


class ProgressEstimateTests(unittest.TestCase):
    def test_estimate_uses_average_completed_model_time(self):
        self.assertEqual(estimate_remaining_seconds([4.0, 6.0], 3), 15.0)

    def test_estimate_is_unavailable_until_a_model_finishes(self):
        self.assertIsNone(estimate_remaining_seconds([], 5))

    def test_estimate_never_goes_below_zero(self):
        self.assertEqual(estimate_remaining_seconds([3.0], 0), 0.0)
        self.assertEqual(estimate_remaining_seconds([3.0], -2), 0.0)

    def test_duration_format_is_readable_for_seconds_and_minutes(self):
        self.assertEqual(format_duration(18.4), "18s")
        self.assertEqual(format_duration(125), "2m 5s")
        self.assertEqual(format_duration(-1), "0s")


if __name__ == "__main__":
    unittest.main()
