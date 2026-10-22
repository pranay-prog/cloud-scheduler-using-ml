"""Integration and unit tests for simulation runner, traces, and metrics."""

import unittest
from workload.generator import get_canonical_benchmark_traces, generate_trace
from simulation.metrics import SimulationMetrics
from simulation.runner import SimulationRunner
from autoscaler.threshold import ThresholdAutoscaler
from autoscaler.variability_aware import VariabilityAwareAutoscaler


class TestSimulationIntegration(unittest.TestCase):
    def test_canonical_traces_generation(self):
        """Verify that canonical traces A1, A2, and A3 generate correctly."""
        traces = get_canonical_benchmark_traces(num_requests=50, target_lambda=4.0, target_mu=1.0)
        self.assertIn("A1", traces)
        self.assertIn("A2", traces)
        self.assertIn("A3", traces)

        self.assertEqual(len(traces["A1"].requests), 50)
        self.assertEqual(len(traces["A2"].requests), 50)
        self.assertEqual(len(traces["A3"].requests), 50)

        df = traces["A1"].to_dataframe()
        self.assertEqual(len(df), 50)
        self.assertIn("inter_arrival", df.columns)
        self.assertIn("service_time", df.columns)

    def test_metrics_tracker_calculations(self):
        """Test SimulationMetrics percentile, SLA violation, and VM-hours tracking."""
        m = SimulationMetrics(sla_target=2.0)
        m.record_initial_capacity(4, current_time=0.0)

        # 4 servers running for 100 seconds = 400 VM-seconds
        m.record_capacity_change(6, current_time=100.0)
        # 6 servers running for 200 seconds = 1200 VM-seconds (total 1600 VM-seconds)

        m.record_arrival()
        m.record_arrival()
        m.record_buffer_overflow()
        m.record_completion(wait_time=0.5, total_latency=1.5)  # Under SLA
        m.record_completion(wait_time=2.0, total_latency=3.0)  # Exceeds SLA (2.0s)

        res = m.finalize(end_time=300.0, policy_name="TestPolicy", trace_name="TestTrace")

        self.assertEqual(res.total_arrivals, 2)
        self.assertEqual(res.rejected_requests, 1)
        self.assertEqual(res.completed_requests, 2)
        self.assertAlmostEqual(res.buffer_overflow_rate, 0.5)
        self.assertAlmostEqual(res.sla_violation_rate, 0.5)
        # 1600 VM-seconds / 3600 = 0.4444 VM-hours
        self.assertAlmostEqual(res.vm_hours, 1600.0 / 3600.0, places=3)
        self.assertAlmostEqual(res.avg_latency, 2.25)
        self.assertGreater(res.p95_latency, 0.0)

    def test_end_to_end_simulation_run(self):
        """Execute short discrete-event simulation run with VariabilityAwareAutoscaler."""
        trace = generate_trace(
            name="TestShort",
            num_requests=30,
            target_lambda=5.0,
            target_mu=2.0,
            target_ca2=2.0,
            seed=10,
        )
        as_policy = VariabilityAwareAutoscaler(sla_target=1.5, k_min=2, k_max=10)
        runner = SimulationRunner(
            trace=trace,
            autoscaler=as_policy,
            policy_name="VA-Test",
            initial_k=3,
            boot_delay=1.0,
            buffer_capacity=100,
            control_interval=1.0,
            sla_target=1.5,
        )

        res = runner.run()
        self.assertEqual(res.total_arrivals, 30)
        self.assertEqual(res.completed_requests + res.rejected_requests, 30)
        self.assertGreater(res.vm_hours, 0.0)


if __name__ == "__main__":
    unittest.main()
