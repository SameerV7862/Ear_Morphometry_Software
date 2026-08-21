from unittest import TestCase

from src.earid.cli import _select_open_set_threshold


class OpenSetThresholdTests(TestCase):
    def test_threshold_meets_zero_false_accept_target(self):
        threshold = _select_open_set_threshold(
            known_scores=[0.9, 0.8, 0.4],
            known_predictions=["a", "b", "c"],
            known_labels=["a", "b", "x"],
            unknown_scores=[0.7, 0.6],
            target_far=0.0,
        )

        self.assertGreater(threshold, 0.7)
        self.assertLess(threshold, 0.8)

    def test_requires_unknown_validation_samples(self):
        with self.assertRaises(ValueError):
            _select_open_set_threshold([0.9], ["a"], ["a"], [], 0.01)
