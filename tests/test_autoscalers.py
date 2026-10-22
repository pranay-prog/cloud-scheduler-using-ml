"""Unit tests for storage buffer, processing queue, and autoscalers."""

import unittest

from system.storage_buffer import StorageBuffer
from system.processing_queue import ProcessingQueue
from autoscaler.threshold import ThresholdAutoscaler
from autoscaler.variability_aware import (
    VariabilityAwareAutoscaler,
    AutoscalerStats,
)


class TestSystemState(unittest.TestCase):
    def test_storage_buffer_rejection(self):
        """Buffer must accept items up to capacity and reject upon overflow."""
        buf = StorageBuffer(capacity=3)
        self.assertTrue(buf.push("req_1"))
        self.assertTrue(buf.push("req_2"))
        self.assertTrue(buf.push("req_3"))
        self.assertFalse(buf.push("req_4"))  # Rejected!

        stats = buf.get_stats()
        self.assertEqual(stats.backlog, 3)
        self.assertEqual(stats.admitted, 3)
        self.assertEqual(stats.rejected, 1)
        self.assertEqual(stats.total_arrivals, 4)
        self.assertAlmostEqual(stats.rejection_rate, 0.25)

        # Pop item and push again
        item = buf.pop()
        self.assertEqual(item, "req_1")
        self.assertEqual(buf.backlog, 2)
        self.assertTrue(buf.push("req_5"))
        self.assertEqual(buf.backlog, 3)

    def test_processing_queue_tracking(self):
        """Processing queue must track backlog and peak length correctly."""
        pq = ProcessingQueue(capacity=5)
        pq.enqueue("task_1")
        pq.enqueue("task_2")
        self.assertEqual(pq.backlog, 2)
        self.assertEqual(pq.peak_backlog, 2)

        t1 = pq.dequeue()
        self.assertEqual(t1, "task_1")
        self.assertEqual(pq.backlog, 1)
        self.assertEqual(pq.peak_backlog, 2)


class TestAutoscalers(unittest.TestCase):
    def test_threshold_autoscaler_transitions(self):
        """Threshold autoscaler scales up on high Q, down on low Q, and holds within band."""
        ts = ThresholdAutoscaler(high_threshold=10, low_threshold=3, k_min=1, k_max=10)

        # Q within deadband [3, 10]: hold
        self.assertEqual(ts.decide({"q_len": 6}, current_k=4), 4)

        # Q > high_threshold: scale up
        self.assertEqual(ts.decide({"q_len": 12}, current_k=4), 5)

        # Q < low_threshold: scale down
        self.assertEqual(ts.decide({"q_len": 2}, current_k=4), 3)

    def test_variability_boost_safety_margin(self):
        """Verify variability boost factors."""
        # Bursty (Ca2 > 1.5): alpha = 0.8
        self.assertAlmostEqual(VariabilityAwareAutoscaler.compute_safety_margin(3.0), 0.80)
        self.assertAlmostEqual(VariabilityAwareAutoscaler.compute_safety_margin(1.6), 0.80)

        # Smooth (Ca2 <= 0.5): alpha = 1.0
        self.assertAlmostEqual(VariabilityAwareAutoscaler.compute_safety_margin(0.3), 1.00)
        self.assertAlmostEqual(VariabilityAwareAutoscaler.compute_safety_margin(0.5), 1.00)

        # Moderate (0.5 < Ca2 <= 1.5): intermediate
        self.assertAlmostEqual(VariabilityAwareAutoscaler.compute_safety_margin(1.0), 0.90)

    def test_bursty_workload_chooses_higher_k_than_threshold(self):
        """CORE NOVELTY TEST:

        For identical arrival rate lambda and queue backlog Q(t),
        the bursty workload causes the VariabilityAwareAutoscaler to choose
        a HIGHER k than the ThresholdAutoscaler.
        """
        # Common scenario:
        # lambda = 3.6 req/s, mu = 1.0 req/s, current_k = 4, Q(t) = 8
        arrival_rate = 3.6
        service_rate = 1.0
        current_k = 4
        q_len = 8

        # 1. Baseline Threshold Autoscaler:
        # Static thresholds: high = 12, low = 3
        # Since Q(t) = 8 <= 12, the threshold autoscaler does NOT scale up.
        threshold_as = ThresholdAutoscaler(
            high_threshold=12,
            low_threshold=3,
            k_min=1,
            k_max=15,
        )
        k_threshold = threshold_as.decide({"q_len": q_len}, current_k=current_k)
        self.assertEqual(k_threshold, current_k)  # Holds at 4

        # 2. Variability-Aware Autoscaler with a BURSTY workload:
        # Ca2 = 4.5 (high arrival variability), Cs2 = 2.5, SLA target = 2.0s
        va_as = VariabilityAwareAutoscaler(
            sla_target=2.0,
            k_min=1,
            k_max=15,
        )
        bursty_stats = AutoscalerStats(
            q_len=q_len,
            arrival_rate=arrival_rate,
            service_rate=service_rate,
            ca2=4.5,  # Bursty!
            cs2=2.5,
        )
        k_bursty = va_as.decide(bursty_stats, current_k=current_k)

        # The bursty workload must demand a higher k than threshold autoscaler!
        self.assertGreater(
            k_bursty,
            k_threshold,
            f"Expected k_bursty ({k_bursty}) > k_threshold ({k_threshold}) for bursty workload.",
        )

        # 3. Conversely, with a SMOOTH workload (Ca2 = 0.2, Cs2 = 0.2):
        smooth_stats = AutoscalerStats(
            q_len=q_len,
            arrival_rate=arrival_rate,
            service_rate=service_rate,
            ca2=0.2,  # Smooth!
            cs2=0.2,
        )
        k_smooth = va_as.decide(smooth_stats, current_k=current_k)

        # Bursty workload must also allocate strictly more capacity than smooth workload
        self.assertGreater(
            k_bursty,
            k_smooth,
            f"Expected k_bursty ({k_bursty}) > k_smooth ({k_smooth}) due to variability awareness.",
        )


if __name__ == "__main__":
    unittest.main()
