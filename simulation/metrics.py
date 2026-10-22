"""Simulation performance metrics tracking.

Monitors and computes:
- Average latency
- P95 and P99 latency percentiles
- Resource cost in VM-hours (integral of active servers over simulation duration)
- SLA violation rate (fraction of requests exceeding SLA delay target)
- Buffer overflow rate (fraction of incoming requests dropped at B(t))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class SimulationResult:
    """Consolidated benchmark outcome for a policy and workload trace."""

    policy_name: str
    trace_name: str
    total_arrivals: int
    completed_requests: int
    rejected_requests: int
    avg_latency: float            # Mean wait/latency (s)
    p95_latency: float            # 95th percentile latency (s)
    p99_latency: float            # 99th percentile latency (s)
    vm_hours: float               # Integrated VM-hours
    sla_violation_rate: float     # Fraction exceeding SLA
    buffer_overflow_rate: float   # Fraction rejected by buffer
    sla_target: float

    def as_dict(self) -> Dict[str, float]:
        """Export summary metrics as dict."""
        return {
            "policy": self.policy_name,
            "trace": self.trace_name,
            "total_arrivals": self.total_arrivals,
            "completed": self.completed_requests,
            "rejected": self.rejected_requests,
            "avg_latency": self.avg_latency,
            "p95_latency": self.p95_latency,
            "p99_latency": self.p99_latency,
            "vm_hours": self.vm_hours,
            "sla_violation_rate": self.sla_violation_rate,
            "buffer_overflow_rate": self.buffer_overflow_rate,
        }


class SimulationMetrics:
    """Dynamic collector of event latencies, capacity history, and overflow counters."""

    def __init__(self, sla_target: float = 2.0):
        self.sla_target: float = float(sla_target)
        self.latencies: List[float] = []
        self.wait_times: List[float] = []
        self.total_arrivals: int = 0
        self.buffer_overflows: int = 0

        # VM-hour tracking
        self._vm_seconds: float = 0.0
        self._current_k: int = 0
        self._last_time: float = 0.0

    def record_initial_capacity(self, k: int, current_time: float = 0.0) -> None:
        """Register starting server capacity."""
        self._current_k = int(k)
        self._last_time = float(current_time)

    def record_capacity_change(self, new_k: int, current_time: float) -> None:
        """Integrate VM capacity over elapsed time interval."""
        dt = max(0.0, float(current_time) - self._last_time)
        self._vm_seconds += self._current_k * dt
        self._current_k = int(new_k)
        self._last_time = float(current_time)

    def record_arrival(self) -> None:
        """Increment arrival counter."""
        self.total_arrivals += 1

    def record_buffer_overflow(self) -> None:
        """Increment overflow counter."""
        self.buffer_overflows += 1

    def record_completion(self, wait_time: float, total_latency: float) -> None:
        """Record completed task wait and end-to-end response times."""
        self.wait_times.append(float(wait_time))
        self.latencies.append(float(total_latency))

    def finalize(self, end_time: float, policy_name: str, trace_name: str) -> SimulationResult:
        """Finalize integrals and generate comprehensive SimulationResult."""
        # Final interval integration
        dt = max(0.0, float(end_time) - self._last_time)
        self._vm_seconds += self._current_k * dt
        self._last_time = float(end_time)

        vm_hours = self._vm_seconds / 3600.0

        if self.latencies:
            lat_arr = np.array(self.latencies, dtype=float)
            avg_lat = float(np.mean(lat_arr))
            p95_lat = float(np.percentile(lat_arr, 95))
            p99_lat = float(np.percentile(lat_arr, 99))
            sla_violations = int(np.sum(lat_arr > self.sla_target))
            sla_viol_rate = float(sla_violations / len(lat_arr))
        else:
            avg_lat = 0.0
            p95_lat = 0.0
            p99_lat = 0.0
            sla_viol_rate = 0.0

        overflow_rate = (
            float(self.buffer_overflows / self.total_arrivals)
            if self.total_arrivals > 0
            else 0.0
        )

        return SimulationResult(
            policy_name=policy_name,
            trace_name=trace_name,
            total_arrivals=self.total_arrivals,
            completed_requests=len(self.latencies),
            rejected_requests=self.buffer_overflows,
            avg_latency=avg_lat,
            p95_latency=p95_lat,
            p99_latency=p99_lat,
            vm_hours=vm_hours,
            sla_violation_rate=sla_viol_rate,
            buffer_overflow_rate=overflow_rate,
            sla_target=self.sla_target,
        )
