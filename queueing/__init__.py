"""Queueing models package for cloud scheduler and autoscaling."""

from .erlang_c import (
    ErlangCMetrics,
    erlang_c_probability,
    expected_waiting_time,
    is_stable,
    check_stability,
    traffic_intensity,
    compute_erlang_c_metrics,
)
from .kingman import (
    variability_factor,
    expected_waiting_time_ggk,
    GGkMetrics,
    compute_ggk_metrics,
)

__all__ = [
    "ErlangCMetrics",
    "erlang_c_probability",
    "expected_waiting_time",
    "is_stable",
    "check_stability",
    "traffic_intensity",
    "compute_erlang_c_metrics",
    "variability_factor",
    "expected_waiting_time_ggk",
    "GGkMetrics",
    "compute_ggk_metrics",
]
