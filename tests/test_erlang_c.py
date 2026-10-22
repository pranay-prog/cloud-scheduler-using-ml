"""Unit tests for Erlang-C (M/M/k) queueing module."""

import unittest

from queueing.erlang_c import (
    check_stability,
    compute_erlang_c_metrics,
    erlang_c_probability,
    expected_waiting_time,
    is_stable,
    traffic_intensity,
)


class TestErlangC(unittest.TestCase):
    def test_mm1_equivalence(self):
        """For k=1, Erlang-C should exactly match standard M/M/1 closed-form."""
        arrival_rate = 2.0
        service_rate = 5.0
        k = 1

        rho = arrival_rate / (k * service_rate)  # 2/5 = 0.4
        expected_pw = rho                         # P_W = rho for M/M/1
        # W_q = lambda / (mu * (mu - lambda)) = 2 / (5 * 3) = 2/15
        expected_wq = arrival_rate / (service_rate * (service_rate - arrival_rate))

        pw = erlang_c_probability(arrival_rate, service_rate, k)
        wq = expected_waiting_time(arrival_rate, service_rate, k)

        self.assertAlmostEqual(pw, expected_pw)
        self.assertAlmostEqual(wq, expected_wq)
        self.assertAlmostEqual(wq, 2.0 / 15.0)

    def test_mm2_analytical_values(self):
        """Test against known analytical solution for k=2."""
        # lambda = 1.0, mu = 1.0, k = 2 -> a = 1.0, rho = 0.5
        # P_W = 1/3, W_q = (1/3) / (2 - 1) = 1/3
        arrival_rate = 1.0
        service_rate = 1.0
        k = 2

        pw = erlang_c_probability(arrival_rate, service_rate, k)
        wq = expected_waiting_time(arrival_rate, service_rate, k)

        self.assertAlmostEqual(pw, 1.0 / 3.0)
        self.assertAlmostEqual(wq, 1.0 / 3.0)

    def test_stability_boundary_and_exceptions(self):
        """System must be unstable when rho >= 1.0."""
        # rho = 1.0 (boundary)
        self.assertFalse(is_stable(arrival_rate=2.0, service_rate=1.0, k=2))
        with self.assertRaises(ValueError):
            check_stability(arrival_rate=2.0, service_rate=1.0, k=2)
        with self.assertRaises(ValueError):
            erlang_c_probability(arrival_rate=2.0, service_rate=1.0, k=2)
        with self.assertRaises(ValueError):
            expected_waiting_time(arrival_rate=2.0, service_rate=1.0, k=2)

        # rho > 1.0
        self.assertFalse(is_stable(arrival_rate=5.0, service_rate=1.0, k=2))
        with self.assertRaises(ValueError):
            expected_waiting_time(arrival_rate=5.0, service_rate=1.0, k=2)

        # rho < 1.0 (stable)
        self.assertTrue(is_stable(arrival_rate=1.8, service_rate=1.0, k=2))

    def test_invalid_parameters(self):
        """Check validation for non-positive rates and non-integer/invalid k."""
        with self.assertRaises(ValueError):
            traffic_intensity(arrival_rate=-1.0, service_rate=1.0, k=1)
        with self.assertRaises(ValueError):
            traffic_intensity(arrival_rate=1.0, service_rate=0.0, k=1)
        with self.assertRaises(ValueError):
            traffic_intensity(arrival_rate=1.0, service_rate=1.0, k=0)

    def test_full_metrics_suite(self):
        """Test compute_erlang_c_metrics consistency and Little's Law."""
        arrival_rate = 3.0
        service_rate = 2.0
        k = 3
        # a = 1.5, rho = 1.5 / 3 = 0.5 < 1.0

        m = compute_erlang_c_metrics(arrival_rate, service_rate, k)

        self.assertAlmostEqual(m.offered_load, 1.5)
        self.assertAlmostEqual(m.traffic_intensity, 0.5)
        # Little's Law: L_q = lambda * W_q
        self.assertAlmostEqual(m.l_q, arrival_rate * m.w_q)
        # System time: W = W_q + 1/mu
        self.assertAlmostEqual(m.w_sys, m.w_q + (1.0 / service_rate))
        # Little's Law: L = lambda * W
        self.assertAlmostEqual(m.l_sys, arrival_rate * m.w_sys)


if __name__ == "__main__":
    unittest.main()
