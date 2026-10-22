"""Unit tests for Random Forest Service Time Predictor and G/G/k moment estimation."""

import unittest
import numpy as np

from ml.rf_service_time import (
    RFServiceTimePredictor,
    generate_synthetic_workload,
    train_rf_predictor,
)


class TestRFServiceTime(unittest.TestCase):
    def test_synthetic_data_generation(self):
        """Verify synthetic data generator structure and value ranges."""
        df = generate_synthetic_workload(n_samples=200, noise_level=2.0, seed=123)
        self.assertEqual(len(df), 200)
        self.assertIn("CPU", df.columns)
        self.assertIn("RAM", df.columns)
        self.assertIn("Type", df.columns)
        self.assertIn("Actual_Time", df.columns)
        self.assertTrue(np.all(df["Actual_Time"] > 0))

    def test_rf_fit_predict(self):
        """Verify training and prediction accuracy."""
        predictor, metrics = train_rf_predictor(n_samples=300, noise_level=1.0, seed=42)
        self.assertTrue(predictor.is_fitted)
        self.assertIn("R2", metrics)
        self.assertIn("MAE", metrics)
        self.assertGreater(metrics["R2"], 0.70)

        # Test single and batch prediction
        test_inputs = np.array([[8, 16, 2], [1, 2, 1]])
        preds = predictor.predict(test_inputs)
        self.assertEqual(len(preds), 2)
        # Larger resource request should have longer execution time
        self.assertGreater(preds[0], preds[1])

    def test_estimate_service_moments_for_ggk(self):
        """Verify mu and Cs2 moment estimation for G/G/k queueing."""
        predictor, _ = train_rf_predictor(n_samples=300, noise_level=2.0, seed=42)

        sample_tasks = np.array([
            [4, 8, 1],
            [8, 16, 2],
            [12, 32, 3],
            [16, 64, 3],
        ])

        mu, mean_s, var_s, cs2 = predictor.estimate_service_moments(sample_tasks)

        self.assertGreater(mean_s, 0.0)
        self.assertAlmostEqual(mu, 1.0 / mean_s)
        self.assertGreaterEqual(var_s, 0.0)
        self.assertGreaterEqual(cs2, 0.0)
        self.assertAlmostEqual(cs2, var_s / (mean_s ** 2))

    def test_unfitted_model_error(self):
        """Predict before fit must raise RuntimeError."""
        p = RFServiceTimePredictor()
        with self.assertRaises(RuntimeError):
            p.predict([[4, 8, 1]])


if __name__ == "__main__":
    unittest.main()
