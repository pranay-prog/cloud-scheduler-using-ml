"""Deterministic task dispatch policies for the simulated processing queue."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite
from typing import Any, Iterable


@dataclass(frozen=True)
class SchedulingPolicy:
    """Select the next task from waiting work using only the named estimate."""

    name: str
    use_deadlines: bool = False
    use_uncertainty: bool = False
    use_slack: bool = False
    oracle_actual_runtime: bool = False
    fifo: bool = False

    def _value(self, task: Any) -> float:
        if self.oracle_actual_runtime:
            return _valid_runtime(getattr(task, "service_time", None), fallback=0.0)
        if self.use_uncertainty:
            return _valid_runtime(getattr(task, "upper_service", None), fallback=getattr(task, "predicted_service", None))
        return _valid_runtime(getattr(task, "predicted_service", None), fallback=0.0)

    def priority(self, task: Any, now: float, queue_wait: float = 0.0) -> tuple:
        arrival = float(getattr(task, "arrival_timestamp", 0.0))
        req_id = int(getattr(task, "req_id", 0))
        if self.fifo:
            return (arrival, req_id)

        runtime = self._value(task)
        deadline = getattr(task, "deadline", None)
        has_deadline = deadline is not None and isfinite(float(deadline))
        slack = float(deadline) - now - max(0.0, queue_wait) - runtime if has_deadline else inf

        # Finite negative slack sorts first (least feasible first: smallest slack).
        # A missed deadline cannot be rescued by ordering; this prioritizes the
        # most urgent/least feasible task, then earliest deadline, then stable ID.
        if self.use_deadlines and has_deadline:
            if self.use_slack:
                # Keep tasks with nonnegative conservative slack in the feasible
                # set and dispatch them by least slack. Already-infeasible work
                # follows feasible work, ordered by deadline then runtime.
                if slack >= 0:
                    return (0, slack, float(deadline), runtime, arrival, req_id)
                return (1, float(deadline), runtime, arrival, req_id)
            return (0, float(deadline), arrival, req_id)
        if self.use_deadlines:
            return (2 if self.use_slack else 1, runtime, arrival, req_id)
        return (runtime, arrival, req_id)

    def choose(self, tasks: Iterable[Any], now: float, queue_wait: float = 0.0) -> int:
        """Return the index of the winning queued task (stable deterministic ties)."""
        items = list(tasks)
        if not items:
            raise ValueError("cannot choose from an empty queue")
        return min(range(len(items)), key=lambda i: self.priority(items[i], now, queue_wait))


def _valid_runtime(value: Any, fallback: Any = None) -> float:
    """Use positive finite values; invalid estimates safely fall back to 0."""
    for candidate in (value, fallback):
        try:
            number = float(candidate)
        except (TypeError, ValueError):
            continue
        if isfinite(number) and number > 0:
            return number
    return 0.0


SCHEDULERS = {
    "FCFS": SchedulingPolicy("FCFS", fifo=True),
    "SJF actual (oracle)": SchedulingPolicy("SJF actual (oracle)", oracle_actual_runtime=True),
    "SJF predicted": SchedulingPolicy("SJF predicted"),
    "EDF": SchedulingPolicy("EDF", use_deadlines=True),
    "Deadline-aware": SchedulingPolicy("Deadline-aware", use_deadlines=True, use_slack=True),
    "Uncertainty-aware": SchedulingPolicy("Uncertainty-aware", use_uncertainty=True),
    "Proposed": SchedulingPolicy("Proposed", use_deadlines=True, use_uncertainty=True, use_slack=True),
}

