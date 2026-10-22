"""Threshold-based queue backlog autoscaler baseline.

Scales VM worker capacity strictly based on static processing queue backlog Q(t)
thresholds, without awareness of arrival rates, variability, or queueing delays.
"""

from __future__ import annotations

from typing import Any, Mapping


class ThresholdAutoscaler:
    """Baseline autoscaler relying solely on processing queue length Q(t)."""

    def __init__(
        self,
        high_threshold: int = 15,
        low_threshold: int = 4,
        k_min: int = 1,
        k_max: int = 25,
        step_up: int = 1,
        step_down: int = 1,
    ):
        if high_threshold <= low_threshold:
            raise ValueError("high_threshold must be strictly greater than low_threshold.")
        if k_min < 1 or k_max < k_min:
            raise ValueError("Invalid server bounds: must satisfy 1 <= k_min <= k_max.")

        self.high_threshold: int = int(high_threshold)
        self.low_threshold: int = int(low_threshold)
        self.k_min: int = int(k_min)
        self.k_max: int = int(k_max)
        self.step_up: int = int(step_up)
        self.step_down: int = int(step_down)

    def _extract_q(self, stats: Any) -> int:
        """Extract queue length Q(t) from input object, dictionary, or integer."""
        if isinstance(stats, (int, float)):
            return int(stats)
        if isinstance(stats, Mapping):
            return int(stats.get("q_len", stats.get("Q", 0)))
        if hasattr(stats, "q_len"):
            return int(stats.q_len)
        if hasattr(stats, "backlog"):
            return int(stats.backlog)
        return 0

    def decide(self, stats: Any, current_k: int) -> int:
        """Decide the next server capacity k based on static Q(t) thresholds.

        Parameters
        ----------
        stats : Any
            System stats object containing queue length Q(t).
        current_k : int
            Current number of active server instances.

        Returns
        -------
        int
            Target server count k in range [k_min, k_max].
        """
        q_len = self._extract_q(stats)
        k = max(self.k_min, min(self.k_max, current_k))

        if q_len > self.high_threshold:
            # Backlog is high: scale up
            target_k = min(self.k_max, k + self.step_up)
        elif q_len < self.low_threshold:
            # Backlog is low: scale down
            target_k = max(self.k_min, k - self.step_down)
        else:
            # Backlog is within hysteresis band: hold current capacity
            target_k = k

        return target_k
