"""Unit tests for workload characterizer."""

import unittest
import numpy as np

from workload.characterizer import (
    compute_arrival_metrics,
    compute_service_metrics,
    compute_workload_parameters,
)


class TestWorkloadCharacterizer(unittest.TestCase):
    def test_deterministic_workload(self):
        """A deterministic workload has variance = 0 and C^2 = 0."""
        inter_arrivals = [2.0, 2.0, 2.0, 2.0, 2.0]
        service_times = [0.5, 0.5, 0.5, 0.5, 0.5]

        metrics = compute_workload_parameters(inter_arrivals, service_times)

        self.assertAlmostEqual(metrics.mean_arrival, 2.0)
        self.assertAlmostEqual(metrics.lambda_rate, 0.5)
        self.assertAlmostEqual(metrics.var_arrival, 0.0)
        self.assertAlmostEqual(metrics.ca2, 0.0)

        self.assertAlmostEqual(metrics.mean_service, 0.5)
        self.assertAlmostEqual(metrics.mu_rate, 2.0)
        self.assertAlmostEqual(metrics.var_service, 0.0)
        self.assertAlmostEqual(metrics.cs2, 0.0)

    def test_known_statistics(self):
        """Test with manually verified sample values."""
        inter_arrivals = [1.0, 2.0, 3.0]  # mean = 2, sample var = 1.0 (ddof=1)
        service_times = [2.0, 4.0, 6.0]   # mean = 4, sample var = 4.0 (ddof=1)

        metrics = compute_workload_parameters(inter_arrivals, service_times, ddof=1)

        # Arrival checks
        self.assertAlmostEqual(metrics.mean_arrival, 2.0)
        self.assertAlmostEqual(metrics.lambda_rate, 0.5)
        self.assertAlmostEqual(metrics.var_arrival, 1.0)
        # Ca^2 = 1.0 / (2.0^2) = 0.25
        self.assertAlmostEqual(metrics.ca2, 0.25)

        # Service checks
        self.assertAlmostEqual(metrics.mean_service, 4.0)
        self.assertAlmostEqual(metrics.mu_rate, 0.25)
        self.assertAlmostEqual(metrics.var_service, 4.0)
        # Cs^2 = 4.0 / (4.0^2) = 0.25
        self.assertAlmostEqual(metrics.cs2, 0.25)

    def test_exponential_rv_variability(self):
        """For an exponential distribution, Var = E^2 so C^2 should be close to 1.0."""
        np.random.seed(42)
        rate_lambda = 4.0
        # Exponential sample with mean 1/4 = 0.25
        samples = np.random.exponential(scale=1.0 / rate_lambda, size=100000)

        lam, mean_a, var_a, ca2 = compute_arrival_metrics(samples)

        self.assertAlmostEqual(mean_a, 0.25, delta=0.01)
        self.assertAlmostEqual(lam, 4.0, delta=0.1)
        # C_a^2 should be approximately 1.0
        self.assertAlmostEqual(ca2, 1.0, delta=0.05)

    def test_error_handling_empty(self):
        """Should raise ValueError on empty sequences."""
        with self.assertRaises(ValueError):
            compute_arrival_metrics([])

    def test_error_handling_negative(self):
        """Should raise ValueError if any value is <= 0."""
        with self.assertRaises(ValueError):
            compute_service_metrics([1.0, -2.0, 3.0])
        with self.assertRaises(ValueError):
            compute_arrival_metrics([0.0, 1.0, 2.0])

    def test_error_handling_nan_inf(self):
        """Should raise ValueError for NaN or Inf."""
        with self.assertRaises(ValueError):
            compute_arrival_metrics([1.0, float("nan"), 2.0])
        with self.assertRaises(ValueError):
            compute_service_metrics([1.0, float("inf"), 2.0])


if __name__ == "__main__":
    unittest.main()
