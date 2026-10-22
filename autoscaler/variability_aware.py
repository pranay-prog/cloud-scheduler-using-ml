"""Variability-Aware G/G/k Autoscaler.

Core novelty: Integrates second-order workload variability (C_a^2 and C_s^2)
and queue backlog Q(t) into the provisioning decision using the Allen-Cunneen
G/G/k heavy-traffic approximation.

Enforces a dynamic variability boost:
- Bursty workloads (C_a^2 > 1.5): Enforce strict safety margin (e.g. <= 80% SLA).
- Smooth workloads (C_a^2 <= 0.5): Allow tighter packing (100% SLA).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from queueing.erlang_c import is_stable
from queueing.kingman import expected_waiting_time_ggk


@dataclass
class AutoscalerStats:
    """Input statistics for autoscaling decisions."""

    q_len: int                     # Q(t) processing queue backlog
    arrival_rate: float            # lambda (req/sec)
    service_rate: float            # mu (req/sec per server)
    ca2: float = 1.0               # C_a^2 arrival variability
    cs2: float = 1.0               # C_s^2 service variability
    b_len: int = 0                 # B(t) storage buffer backlog


class VariabilityAwareAutoscaler:
    """Variability-aware autoscaler using the G/G/k Allen-Cunneen approximation."""

    def __init__(
        self,
        sla_target: float = 2.0,
        k_min: int = 1,
        k_max: int = 30,
        scale_down_ratio: float = 0.5,
        max_scale_up_step: Optional[int] = None,
        max_scale_down_step: int = 1,
        q_low_threshold: int = 3,
        target_max_rho: float = 0.85,
    ):
        if sla_target <= 0:
            raise ValueError("sla_target must be strictly positive.")
        if k_min < 1 or k_max < k_min:
            raise ValueError("Invalid server bounds: must satisfy 1 <= k_min <= k_max.")

        self.sla_target: float = float(sla_target)
        self.k_min: int = int(k_min)
        self.k_max: int = int(k_max)
        self.scale_down_ratio: float = float(scale_down_ratio)
        self.max_scale_up_step: Optional[int] = max_scale_up_step
        self.max_scale_down_step: int = int(max_scale_down_step)
        self.q_low_threshold: int = int(q_low_threshold)
        self.target_max_rho: float = float(target_max_rho)

    @staticmethod
    def compute_safety_margin(ca2: float) -> float:
        """Compute variability boost safety margin factor alpha.

        - If C_a^2 > 1.5 (bursty): alpha = 0.80 (strict safety margin: 80% SLA).
        - If C_a^2 <= 0.5 (smooth): alpha = 1.00 (tight packing: 100% SLA).
        - If 0.5 < C_a^2 <= 1.5: linear interpolation between 1.00 and 0.80.
        """
        if ca2 > 1.5:
            return 0.80
        elif ca2 <= 0.5:
            return 1.00
        else:
            # Linear ramp from 1.00 down to 0.80 as ca2 moves from 0.5 to 1.5
            return 1.00 - 0.20 * ((ca2 - 0.5) / 1.00)

    def get_effective_sla(self, ca2: float) -> float:
        """Calculate effective SLA target after applying variability safety margin."""
        alpha = self.compute_safety_margin(ca2)
        return float(self.sla_target * alpha)

    def predict_wait_time(
        self,
        k: int,
        q_len: int,
        arrival_rate: float,
        service_rate: float,
        ca2: float,
        cs2: float,
    ) -> float:
        """Predict expected waiting time under candidate capacity k.

        Composed of:
        1. Steady-state G/G/k waiting time: W_q^(G/G/k)
        2. Transient queue backlog clearing time: Q(t) / (k * mu)
        """
        if not is_stable(arrival_rate, service_rate, k):
            return float("inf")

        w_q_ggk = expected_waiting_time_ggk(
            arrival_rate=arrival_rate,
            service_rate=service_rate,
            k=k,
            ca2=ca2,
            cs2=cs2,
        )

        drain_time = q_len / (k * service_rate) if q_len > 0 else 0.0
        return float(w_q_ggk + drain_time)

    def _parse_stats(self, stats: Any) -> AutoscalerStats:
        """Normalize stats input into AutoscalerStats dataclass."""
        if isinstance(stats, AutoscalerStats):
            return stats
        if isinstance(stats, Mapping):
            return AutoscalerStats(
                q_len=int(stats.get("q_len", stats.get("Q", 0))),
                arrival_rate=float(stats.get("arrival_rate", stats.get("lambda", 1.0))),
                service_rate=float(stats.get("service_rate", stats.get("mu", 1.0))),
                ca2=float(stats.get("ca2", stats.get("C_a^2", 1.0))),
                cs2=float(stats.get("cs2", stats.get("C_s^2", 1.0))),
                b_len=int(stats.get("b_len", stats.get("B", 0))),
            )
        # Attribute-based fallback
        return AutoscalerStats(
            q_len=int(getattr(stats, "q_len", getattr(stats, "backlog", 0))),
            arrival_rate=float(getattr(stats, "arrival_rate", getattr(stats, "lambda_rate", 1.0))),
            service_rate=float(getattr(stats, "service_rate", getattr(stats, "mu_rate", 1.0))),
            ca2=float(getattr(stats, "ca2", 1.0)),
            cs2=float(getattr(stats, "cs2", 1.0)),
            b_len=int(getattr(stats, "b_len", 0)),
        )

    def decide(self, stats: Any, current_k: int) -> int:
        """Decide target worker capacity k based on G/G/k delay and variability.

        Parameters
        ----------
        stats : AutoscalerStats | Mapping | Any
            Workload and system stats: Q, lambda, mu, C_a^2, C_s^2.
        current_k : int
            Current number of active server instances.

        Returns
        -------
        int
            Recommended server capacity k in [k_min, k_max].
        """
        s = self._parse_stats(stats)
        k = max(self.k_min, min(self.k_max, current_k))
        effective_sla = self.get_effective_sla(s.ca2)

        predicted_wait = self.predict_wait_time(
            k=k,
            q_len=s.q_len,
            arrival_rate=s.arrival_rate,
            service_rate=s.service_rate,
            ca2=s.ca2,
            cs2=s.cs2,
        )

        # 1. SCALE UP CONDITION: Predicted delay exceeds effective SLA target
        if predicted_wait > effective_sla:
            # Search for smallest k >= current_k that satisfies effective SLA
            target_k = k
            for cand_k in range(k + 1, self.k_max + 1):
                cand_wait = self.predict_wait_time(
                    k=cand_k,
                    q_len=s.q_len,
                    arrival_rate=s.arrival_rate,
                    service_rate=s.service_rate,
                    ca2=s.ca2,
                    cs2=s.cs2,
                )
                if cand_wait <= effective_sla:
                    target_k = cand_k
                    break
            else:
                target_k = self.k_max

            if self.max_scale_up_step is not None:
                target_k = min(target_k, k + self.max_scale_up_step)

            return target_k

        # 2. SCALE DOWN CONDITION: Predicted wait is well below SLA and queue backlog is low
        scale_down_budget = effective_sla * self.scale_down_ratio
        if predicted_wait < scale_down_budget and s.q_len <= self.q_low_threshold:
            # Check if reducing capacity is safe
            target_k = k
            for step in range(1, self.max_scale_down_step + 1):
                cand_k = k - step
                if cand_k < self.k_min:
                    break

                cand_rho = s.arrival_rate / (cand_k * s.service_rate)
                if cand_rho > self.target_max_rho:
                    # Do not scale down if utilization would exceed target safety cap
                    break

                cand_wait = self.predict_wait_time(
                    k=cand_k,
                    q_len=s.q_len,
                    arrival_rate=s.arrival_rate,
                    service_rate=s.service_rate,
                    ca2=s.ca2,
                    cs2=s.cs2,
                )
                if cand_wait <= effective_sla:
                    target_k = cand_k
                else:
                    break

            return target_k

        # 3. HOLD CAPACITY
        return k
