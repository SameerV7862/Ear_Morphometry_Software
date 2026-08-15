from unittest import TestCase

from src.earid.metrics import identification_metrics


class IdentificationMetricsTests(TestCase):
    def test_top_k_calibration_and_risk_coverage(self):
        metrics = identification_metrics(
            [[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]],
            [0, 1],
            num_bins=5,
        )

        self.assertEqual(0.5, metrics["top_1_accuracy"])
        self.assertEqual(1.0, metrics["top_5_accuracy"])
        self.assertEqual(5, len(metrics["calibration_bins"]))
        self.assertEqual(10, len(metrics["risk_coverage"]))

    def test_rejects_target_outside_class_range(self):
        with self.assertRaises(ValueError):
            identification_metrics([[0.5, 0.5]], [2])
