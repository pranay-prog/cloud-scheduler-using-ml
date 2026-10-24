"""Synthetic workload trace generator with configurable arrival and service variability.

Generates 3 canonical benchmark traces:
- A1: Low variability (Poisson arrivals, Exponential service, C_a^2 ~ 1.0)
- A2: Moderate variability (C_a^2 ~ 2.5, Log-normal / Gamma service)
- A3: Bursty workload (C_a^2 ~ 5.0, Pareto heavy-tailed service)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class RequestItem:
    """A single storage / compute service request in the trace."""

    req_id: int
    inter_arrival: float    # Duration since previous arrival (seconds)
    service_time: float     # Execution time (seconds)
    cpu_req: int            # CPU cores requested
    ram_req: int            # RAM GB requested
    workload_type: int      # 1=Read/Web, 2=Compute, 3=Write/DB Batch
    predicted_service: Optional[float] = None
    deadline: Optional[float] = None  # absolute simulated time; None means no deadline
    deadline_after_arrival: Optional[float] = None
    upper_service: Optional[float] = None
    arrival_timestamp: float = 0.0


class WorkloadTrace:
    """Container for a generated request sequence."""

    def __init__(self, name: str, requests: List[RequestItem]):
        self.name: str = name
        self.requests: List[RequestItem] = requests

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trace to pandas DataFrame."""
        rows = [
            {
                "req_id": r.req_id,
                "inter_arrival": r.inter_arrival,
                "service_time": r.service_time,
                "cpu_req": r.cpu_req,
                "ram_req": r.ram_req,
                "workload_type": r.workload_type,
            }
            for r in self.requests
        ]
        return pd.DataFrame(rows)

    def get_arrival_series(self) -> List[float]:
        return [r.inter_arrival for r in self.requests]

    def get_service_series(self) -> List[float]:
        return [r.service_time for r in self.requests]


def generate_trace(
    name: str,
    num_requests: int = 1000,
    target_lambda: float = 4.0,
    target_mu: float = 1.0,
    target_ca2: float = 1.0,
    service_distribution: str = "exponential",
    seed: int = 42,
    deadline_slack_factor: Optional[float] = None,
    runtime_noise_level: float = 0.0,
) -> WorkloadTrace:
    """Generate a synthetic request trace with exact target arrival rate and variability.

    Uses Gamma renewal process for inter-arrivals:
        Var(A) / E[A]^2 = 1 / shape -> shape = 1 / target_ca2
        E[A] = shape * scale = 1 / lambda -> scale = 1 / (shape * lambda)
    """
    rng = np.random.RandomState(seed)
    if not np.isfinite(runtime_noise_level) or runtime_noise_level < 0:
        raise ValueError("runtime_noise_level must be a finite nonnegative value")

    # 1. Generate inter-arrival times
    if target_ca2 <= 0:
        inter_arrivals = np.full(num_requests, 1.0 / target_lambda)
    else:
        shape_a = 1.0 / target_ca2
        scale_a = 1.0 / (shape_a * target_lambda)
        inter_arrivals = rng.gamma(shape=shape_a, scale=scale_a, size=num_requests)
        # Ensure positive lower bound to prevent instantaneous overlap
        inter_arrivals = np.clip(inter_arrivals, 1e-4, None)

    # 2. Features for ML integration
    cpu_reqs = rng.randint(1, 17, size=num_requests)
    ram_reqs = rng.randint(2, 65, size=num_requests)
    types = rng.randint(1, 4, size=num_requests)

    # 3. Generate service times based on target distribution
    target_mean_s = 1.0 / target_mu

    if service_distribution == "exponential":
        # C_s^2 ~ 1.0
        service_times = rng.exponential(scale=target_mean_s, size=num_requests)
    elif service_distribution == "moderate":
        # Log-normal with moderate variance: C_s^2 ~ 1.5 - 2.0
        # For lognormal with mean m and variance v:
        sigma = 0.8
        mu_ln = np.log(target_mean_s) - 0.5 * (sigma ** 2)
        service_times = rng.lognormal(mean=mu_ln, sigma=sigma, size=num_requests)
    elif service_distribution == "pareto":
        # Heavy-tailed Pareto: C_s^2 ~ 3.0 - 5.0
        # Pareto shape alpha > 2 for finite mean and variance
        alpha_p = 2.4
        # E[X] = alpha * xm / (alpha - 1) -> xm = E[X] * (alpha - 1) / alpha
        scale_xm = target_mean_s * (alpha_p - 1.0) / alpha_p
        # Standard pareto distribution shifted by xm
        raw_pareto = (rng.pareto(a=alpha_p, size=num_requests) + 1.0) * scale_xm
        # Clip extreme outliers to reasonable cloud runtime bounds
        service_times = np.clip(raw_pareto, 0.05, target_mean_s * 25.0)
    else:
        service_times = rng.exponential(scale=target_mean_s, size=num_requests)

    # Correlate slightly with resource footprint so ML can learn patterns
    correlated_shift = (cpu_reqs * 0.02 + ram_reqs * 0.005 + types * 0.05)
    service_times = service_times * (0.8 + 0.4 * (correlated_shift / correlated_shift.mean()))
    # Re-normalize mean so mean service duration matches target_mean_s
    service_times = service_times * (target_mean_s / service_times.mean())
    if runtime_noise_level > 0:
        service_times = service_times + rng.normal(0.0, runtime_noise_level, size=num_requests)
    service_times = np.clip(service_times, 0.02, None)

    requests: List[RequestItem] = []
    for i in range(num_requests):
        requests.append(
            RequestItem(
                req_id=i,
                inter_arrival=float(inter_arrivals[i]),
                service_time=float(service_times[i]),
                cpu_req=int(cpu_reqs[i]),
                ram_req=int(ram_reqs[i]),
                workload_type=int(types[i]),
            )
        )

    if deadline_slack_factor is not None:
        # Deadline is assigned relative to each task's own arrival by the runner.
        # Store a relative allowance until arrival time is known.
        for req in requests:
            req.deadline_after_arrival = float(deadline_slack_factor * req.service_time)
    return WorkloadTrace(name=name, requests=requests)


def get_canonical_benchmark_traces(
    num_requests: int = 1500,
    target_lambda: float = 4.0,
    target_mu: float = 1.0,
    seed: int = 42,
) -> Dict[str, WorkloadTrace]:
    """Generate the 3 canonical benchmark traces: A1 (Low), A2 (Moderate), A3 (Bursty)."""
    t_a1 = generate_trace(
        name="A1 (Low Variability)",
        num_requests=num_requests,
        target_lambda=target_lambda,
        target_mu=target_mu,
        target_ca2=1.0,
        service_distribution="exponential",
        seed=seed,
    )

    t_a2 = generate_trace(
        name="A2 (Moderate Variability)",
        num_requests=num_requests,
        target_lambda=target_lambda,
        target_mu=target_mu,
        target_ca2=2.5,
        service_distribution="moderate",
        seed=seed + 1,
    )

    t_a3 = generate_trace(
        name="A3 (Bursty Workload)",
        num_requests=num_requests,
        target_lambda=target_lambda,
        target_mu=target_mu,
        target_ca2=5.0,
        service_distribution="pareto",
        seed=seed + 2,
    )

    return {"A1": t_a1, "A2": t_a2, "A3": t_a3}
