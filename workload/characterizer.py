"""Workload characterization module.

Computes first- and second-order statistical moments of arrival and service
processes for queueing analysis (lambda, mu, C_a^2, C_s^2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class WorkloadMetrics:
    """Summary metrics of a workload arrival and service process."""

    lambda_rate: float      # Arrival rate: lambda = 1 / E[A]
    mu_rate: float          # Service rate: mu = 1 / E[S]
    mean_arrival: float     # Expected inter-arrival time: E[A]
    var_arrival: float      # Variance of inter-arrival time: Var(A)
    mean_service: float     # Expected service time: E[S]
    var_service: float      # Variance of service time: Var(S)
    ca2: float              # Arrival squared coefficient of variation: Var(A) / E[A]^2
    cs2: float              # Service squared coefficient of variation: Var(S) / E[S]^2

    def as_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "lambda": self.lambda_rate,
            "mu": self.mu_rate,
            "E[A]": self.mean_arrival,
            "Var(A)": self.var_arrival,
            "E[S]": self.mean_service,
            "Var(S)": self.var_service,
            "C_a^2": self.ca2,
            "C_s^2": self.cs2,
        }


def _validate_sample_array(data: Sequence[float], name: str) -> np.ndarray:
    """Validates and cleans input 1D numerical sequence."""
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"'{name}' must be a non-empty 1D sequence of numbers.")
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError(f"'{name}' contains NaN or infinite values.")
    if np.any(arr <= 0):
        raise ValueError(f"All values in '{name}' must be strictly positive (> 0).")
    return arr


def compute_arrival_metrics(
    inter_arrivals: Sequence[float],
    ddof: int = 1,
) -> Tuple[float, float, float, float]:
    """Compute arrival rate and variability parameters from inter-arrival times.

    Parameters
    ----------
    inter_arrivals : Sequence[float]
        Series of inter-arrival durations A.
    ddof : int, default=1
        Delta Degrees of Freedom for variance calculation.

    Returns
    -------
    Tuple[float, float, float, float]
        (lambda_rate, mean_arrival, var_arrival, ca2)
    """
    arr = _validate_sample_array(inter_arrivals, "inter_arrivals")
    effective_ddof = ddof if arr.size > 1 else 0

    mean_a = float(np.mean(arr))
    var_a = float(np.var(arr, ddof=effective_ddof))

    if mean_a <= 0:
        raise ValueError("Mean inter-arrival time E[A] must be strictly positive.")

    lambda_rate = 1.0 / mean_a
    ca2 = var_a / (mean_a ** 2)

    return lambda_rate, mean_a, var_a, float(ca2)


def compute_service_metrics(
    service_times: Sequence[float],
    ddof: int = 1,
) -> Tuple[float, float, float, float]:
    """Compute service rate and variability parameters from service times.

    Parameters
    ----------
    service_times : Sequence[float]
        Series of task/request execution durations S.
    ddof : int, default=1
        Delta Degrees of Freedom for variance calculation.

    Returns
    -------
    Tuple[float, float, float, float]
        (mu_rate, mean_service, var_service, cs2)
    """
    arr = _validate_sample_array(service_times, "service_times")
    effective_ddof = ddof if arr.size > 1 else 0

    mean_s = float(np.mean(arr))
    var_s = float(np.var(arr, ddof=effective_ddof))

    if mean_s <= 0:
        raise ValueError("Mean service duration E[S] must be strictly positive.")

    mu_rate = 1.0 / mean_s
    cs2 = var_s / (mean_s ** 2)

    return mu_rate, mean_s, var_s, float(cs2)


def compute_workload_parameters(
    inter_arrivals: Sequence[float],
    service_times: Sequence[float],
    ddof: int = 1,
) -> WorkloadMetrics:
    """Compute full workload characteristics for queueing approximations.

    Calculates:
        lambda = 1 / E[A]
        mu = 1 / E[S]
        C_a^2 = Var(A) / E[A]^2
        C_s^2 = Var(S) / E[S]^2

    Parameters
    ----------
    inter_arrivals : Sequence[float]
        Observed inter-arrival times between incoming cloud requests.
    service_times : Sequence[float]
        Observed task service durations.
    ddof : int, default=1
        Delta Degrees of Freedom for variance calculation.

    Returns
    -------
    WorkloadMetrics
        Object containing all derived queueing parameters.
    """
    lam, mean_a, var_a, ca2 = compute_arrival_metrics(inter_arrivals, ddof=ddof)
    mu, mean_s, var_s, cs2 = compute_service_metrics(service_times, ddof=ddof)

    return WorkloadMetrics(
        lambda_rate=lam,
        mu_rate=mu,
        mean_arrival=mean_a,
        var_arrival=var_a,
        mean_service=mean_s,
        var_service=var_s,
        ca2=ca2,
        cs2=cs2,
    )
