"""Unit tests for Kingman / Allen-Cunneen G/G/k approximation module."""

import unittest

from queueing.erlang_c import expected_waiting_time
from queueing.kingman import (
    compute_ggk_metrics,
    expected_waiting_time_ggk,
    variability_factor,
)


class TestKingmanGGk(unittest.TestCase):
    def test_variability_factor_calculation(self):
        """V = (C_a^2 + C_s^2) / 2."""
        self.assertAlmostEqual(variability_factor(1.0, 1.0), 1.0)
        self.assertAlmostEqual(variability_factor(0.0, 0.0), 0.0)
        self.assertAlmostEqual(variability_factor(0.5, 1.5), 1.0)
        self.assertAlmostEqual(variability_factor(2.0, 4.0), 3.0)

    def test_variability_factor_negative_rejected(self):
        """Negative variability values must raise ValueError."""
        with self.assertRaises(ValueError):
            variability_factor(-0.1, 1.0)
        with self.assertRaises(ValueError):
            variability_factor(1.0, -0.5)

    def test_exact_match_with_erlang_c_when_ca2_cs2_are_one(self):
        """If C_a^2 = C_s^2 = 1 (Poisson arrivals, exponential service),

        G/G/k formula must exactly match Erlang-C M/M/k.
        """
        test_cases = [
            # (lambda, mu, k)
            (1.0, 2.0, 1),
            (2.0, 3.0, 1),
            (1.5, 1.0, 2),
            (4.0, 2.0, 3),
            (10.0, 3.0, 5),
        ]

        for lam, mu, k in test_cases:
            with self.subTest(lam=lam, mu=mu, k=k):
                wq_mmk = expected_waiting_time(lam, mu, k)
                wq_ggk = expected_waiting_time_ggk(lam, mu, k, ca2=1.0, cs2=1.0)

                self.assertAlmostEqual(
                    wq_ggk,
                    wq_mmk,
                    places=10,
                    msg=f"Failed for lambda={lam}, mu={mu}, k={k}",
                )

    def test_linear_scaling_with_variability(self):
        """W_q^(G/G/k) must scale strictly linearly with V."""
        lam = 2.0
        mu = 1.5
        k = 2

        wq_mmk = expected_waiting_time(lam, mu, k)

        # V = (3.0 + 1.0) / 2 = 2.0 -> W_q should be exactly 2 * wq_mmk
        wq_double = expected_waiting_time_ggk(lam, mu, k, ca2=3.0, cs2=1.0)
        self.assertAlmostEqual(wq_double, 2.0 * wq_mmk)

        # Deterministic: V = 0 -> W_q = 0
        wq_det = expected_waiting_time_ggk(lam, mu, k, ca2=0.0, cs2=0.0)
        self.assertAlmostEqual(wq_det, 0.0)

    def test_kingman_gg1_special_case(self):
        """For k=1, verify Kingman's formula: W_q ~ ((Ca2 + Cs2)/2) * (rho / (1 - rho)) * (1/mu)."""
        lam = 3.0
        mu = 5.0
        k = 1
        ca2 = 0.8
        cs2 = 1.2
        rho = lam / mu  # 0.6

        # Standard Kingman formula for G/G/1:
        v = (ca2 + cs2) / 2.0  # 1.0
        expected_kingman = v * (rho / (1.0 - rho)) * (1.0 / mu)

        wq_ggk = expected_waiting_time_ggk(lam, mu, k, ca2=ca2, cs2=cs2)
        self.assertAlmostEqual(wq_ggk, expected_kingman)

    def test_stability_enforcement(self):
        """Unstable systems (rho >= 1) must raise ValueError."""
        with self.assertRaises(ValueError):
            expected_waiting_time_ggk(arrival_rate=4.0, service_rate=2.0, k=2, ca2=1.0, cs2=1.0)

    def test_ggk_metrics_suite(self):
        """Test complete G/G/k metrics data object."""
        m = compute_ggk_metrics(arrival_rate=2.0, service_rate=2.0, k=2, ca2=1.5, cs2=0.5)

        self.assertAlmostEqual(m.variability_factor, 1.0)
        self.assertAlmostEqual(m.traffic_intensity, 0.5)
        self.assertAlmostEqual(m.w_q_ggk, m.w_q_mmk)
        self.assertAlmostEqual(m.l_q_ggk, 2.0 * m.w_q_ggk)
        self.assertAlmostEqual(m.w_sys_ggk, m.w_q_ggk + 0.5)


if __name__ == "__main__":
    unittest.main()
