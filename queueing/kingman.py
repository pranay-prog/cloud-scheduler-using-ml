"""Allen-Cunneen / Kingman heavy-traffic approximation for G/G/k queues.

Estimates waiting time in general multi-server queues (G/G/k) by scaling
the Erlang-C waiting time by the arrival and service variability factor:
    V = (C_a^2 + C_s^2) / 2
    W_q^(G/G/k) = V * P_W / (k * mu - lambda)
                = V * W_q^(M/M/k)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .erlang_c import (
    check_stability,
    erlang_c_probability,
    expected_waiting_time,
)


@dataclass(frozen=True)
class GGkMetrics:
    """Performance metrics for a G/G/k queue."""

    arrival_rate: float      # lambda
    service_rate: float      # mu
    k_servers: int           # k
    ca2: float               # Arrival squared coefficient of variation C_a^2
    cs2: float               # Service squared coefficient of variation C_s^2
    variability_factor: float # V = (C_a^2 + C_s^2) / 2
    traffic_intensity: float # rho = lambda / (k * mu)
    p_wait_mmk: float        # Erlang-C P_W
    w_q_mmk: float           # M/M/k waiting time
    w_q_ggk: float           # G/G/k waiting time
    w_sys_ggk: float         # G/G/k total system time: W_q + 1/mu
    l_q_ggk: float           # G/G/k expected queue length: lambda * W_q
    l_sys_ggk: float         # G/G/k expected system count: lambda * W_sys

    def as_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "arrival_rate": self.arrival_rate,
            "service_rate": self.service_rate,
            "k_servers": float(self.k_servers),
            "ca2": self.ca2,
            "cs2": self.cs2,
            "variability_factor": self.variability_factor,
            "traffic_intensity": self.traffic_intensity,
            "p_wait_mmk": self.p_wait_mmk,
            "w_q_mmk": self.w_q_mmk,
            "w_q_ggk": self.w_q_ggk,
            "w_sys_ggk": self.w_sys_ggk,
            "l_q_ggk": self.l_q_ggk,
            "l_sys_ggk": self.l_sys_ggk,
        }


def variability_factor(ca2: float, cs2: float) -> float:
    """Compute variability factor V = (C_a^2 + C_s^2) / 2.

    Parameters
    ----------
    ca2 : float
        Squared coefficient of variation of inter-arrival times (C_a^2 >= 0).
    cs2 : float
        Squared coefficient of variation of service times (C_s^2 >= 0).

    Returns
    -------
    float
        Variability multiplier V.
    """
    if ca2 < 0:
        raise ValueError(f"ca2 (C_a^2) must be non-negative (got {ca2}).")
    if cs2 < 0:
        raise ValueError(f"cs2 (C_s^2) must be non-negative (got {cs2}).")

    return float((ca2 + cs2) / 2.0)


def expected_waiting_time_ggk(
    arrival_rate: float,
    service_rate: float,
    k: int,
    ca2: float = 1.0,
    cs2: float = 1.0,
) -> float:
    """Compute expected queue waiting time W_q^(G/G/k) via Allen-Cunneen approximation.

    W_q^(G/G/k) = [ (C_a^2 + C_s^2) / 2 ] * [ P_W / (k * mu - lambda) ]
                = V * W_q^(M/M/k)

    Parameters
    ----------
    arrival_rate : float
        Request arrival rate (lambda > 0).
    service_rate : float
        Service rate per server (mu > 0).
    k : int
        Number of parallel servers (k >= 1).
    ca2 : float, default=1.0
        Arrival squared coefficient of variation C_a^2.
    cs2 : float, default=1.0
        Service squared coefficient of variation C_s^2.

    Returns
    -------
    float
        Approximated expected queue waiting time in seconds.
    """
    # Verifies stability (rho < 1) and parameter validity
    check_stability(arrival_rate, service_rate, k)
    v = variability_factor(ca2, cs2)
    w_q_mmk = expected_waiting_time(arrival_rate, service_rate, k)

    return float(v * w_q_mmk)


def compute_ggk_metrics(
    arrival_rate: float,
    service_rate: float,
    k: int,
    ca2: float = 1.0,
    cs2: float = 1.0,
) -> GGkMetrics:
    """Compute comprehensive G/G/k performance metrics."""
    rho = check_stability(arrival_rate, service_rate, k)
    v = variability_factor(ca2, cs2)
    p_w = erlang_c_probability(arrival_rate, service_rate, k)
    w_q_mmk = expected_waiting_time(arrival_rate, service_rate, k)
    w_q_ggk = v * w_q_mmk

    w_sys_ggk = w_q_ggk + (1.0 / service_rate)
    l_q_ggk = arrival_rate * w_q_ggk
    l_sys_ggk = arrival_rate * w_sys_ggk

    return GGkMetrics(
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        k_servers=k,
        ca2=ca2,
        cs2=cs2,
        variability_factor=v,
        traffic_intensity=rho,
        p_wait_mmk=p_w,
        w_q_mmk=w_q_mmk,
        w_q_ggk=w_q_ggk,
        w_sys_ggk=w_sys_ggk,
        l_q_ggk=l_q_ggk,
        l_sys_ggk=l_sys_ggk,
    )
