"""Erlang-C queueing model (M/M/k).

Provides steady-state performance metrics for multi-server Poisson queueing
systems with exponential service times:
- Offered load a = lambda / mu
- Traffic intensity rho = lambda / (k * mu)
- Probability of waiting P_W (Erlang-C delay formula)
- Expected waiting time in queue W_q = P_W / (k * mu - lambda)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ErlangCMetrics:
    """Performance metrics for an M/M/k queue."""

    arrival_rate: float      # lambda
    service_rate: float      # mu
    k_servers: int           # k
    offered_load: float      # a = lambda / mu
    traffic_intensity: float # rho = lambda / (k * mu)
    p_wait: float            # P_W
    w_q: float               # Expected queue waiting time W_q
    w_sys: float             # Expected total system time W = W_q + 1/mu
    l_q: float               # Expected queue length L_q = lambda * W_q
    l_sys: float             # Expected system population L = lambda * W

    def as_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "arrival_rate": self.arrival_rate,
            "service_rate": self.service_rate,
            "k_servers": float(self.k_servers),
            "offered_load": self.offered_load,
            "traffic_intensity": self.traffic_intensity,
            "p_wait": self.p_wait,
            "w_q": self.w_q,
            "w_sys": self.w_sys,
            "l_q": self.l_q,
            "l_sys": self.l_sys,
        }


def traffic_intensity(arrival_rate: float, service_rate: float, k: int) -> float:
    """Compute traffic intensity / server utilization rho = lambda / (k * mu)."""
    if arrival_rate <= 0:
        raise ValueError("arrival_rate (lambda) must be strictly positive.")
    if service_rate <= 0:
        raise ValueError("service_rate (mu) must be strictly positive.")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k (number of servers) must be an integer >= 1.")

    return float(arrival_rate / (k * service_rate))


def is_stable(arrival_rate: float, service_rate: float, k: int) -> bool:
    """Return True if system is stable (rho < 1.0), False otherwise."""
    rho = traffic_intensity(arrival_rate, service_rate, k)
    return rho < 1.0


def check_stability(arrival_rate: float, service_rate: float, k: int) -> float:
    """Check stability condition (rho < 1).

    Raises ValueError if rho >= 1.0, otherwise returns rho.
    """
    rho = traffic_intensity(arrival_rate, service_rate, k)
    if rho >= 1.0:
        raise ValueError(
            f"System is unstable: rho = {rho:.4f} >= 1.0 "
            f"(arrival_rate={arrival_rate}, service_rate={service_rate}, k={k}, capacity={k * service_rate})"
        )
    return rho


def erlang_c_probability(arrival_rate: float, service_rate: float, k: int) -> float:
    """Compute probability that an arriving request must wait (P_W).

    P_W = [ (a^k / k!) * (1 / (1 - rho)) ] /
          [ sum_{n=0}^{k-1} (a^n / n!) + (a^k / k!) * (1 / (1 - rho)) ]

    Parameters
    ----------
    arrival_rate : float
        Request arrival rate (lambda > 0).
    service_rate : float
        Service rate per server (mu > 0).
    k : int
        Number of parallel servers (k >= 1).

    Returns
    -------
    float
        Probability of waiting P_W in range [0, 1).
    """
    rho = check_stability(arrival_rate, service_rate, k)
    a = arrival_rate / service_rate  # offered load

    # Compute terms iteratively to prevent numerical overflow:
    # term_n = a^n / n!
    terms = [1.0]  # n=0
    for n in range(1, k + 1):
        terms.append(terms[-1] * (a / n))

    sum_terms = sum(terms[:k])  # sum from n=0 to k-1 of a^n / n!
    last_term = terms[k]        # a^k / k!

    t_k = last_term / (1.0 - rho)
    p_w = t_k / (sum_terms + t_k)

    return float(min(max(p_w, 0.0), 1.0))


def expected_waiting_time(arrival_rate: float, service_rate: float, k: int) -> float:
    """Compute expected waiting time in queue (W_q) for an M/M/k system.

    W_q = P_W / (k * mu - lambda)

    Parameters
    ----------
    arrival_rate : float
        Request arrival rate (lambda > 0).
    service_rate : float
        Service rate per server (mu > 0).
    k : int
        Number of parallel servers (k >= 1).

    Returns
    -------
    float
        Expected queue waiting time W_q (in seconds).
    """
    check_stability(arrival_rate, service_rate, k)
    p_w = erlang_c_probability(arrival_rate, service_rate, k)
    denominator = (k * service_rate) - arrival_rate

    return float(p_w / denominator)


def compute_erlang_c_metrics(
    arrival_rate: float,
    service_rate: float,
    k: int,
) -> ErlangCMetrics:
    """Compute complete Erlang-C metrics suite for an M/M/k queue."""
    rho = check_stability(arrival_rate, service_rate, k)
    a = arrival_rate / service_rate
    p_w = erlang_c_probability(arrival_rate, service_rate, k)
    w_q = expected_waiting_time(arrival_rate, service_rate, k)

    w_sys = w_q + (1.0 / service_rate)
    l_q = arrival_rate * w_q
    l_sys = arrival_rate * w_sys

    return ErlangCMetrics(
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        k_servers=k,
        offered_load=a,
        traffic_intensity=rho,
        p_wait=p_w,
        w_q=w_q,
        w_sys=w_sys,
        l_q=l_q,
        l_sys=l_sys,
    )
