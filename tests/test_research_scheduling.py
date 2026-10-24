import numpy as np
import pandas as pd
import pytest

from ml.rf_service_time import train_calibrate_test_predictor
from simulation.metrics import SimulationMetrics
from simulation.scheduling import SCHEDULERS
from system.processing_queue import ProcessingQueue
from workload.generator import RequestItem


def task(req_id, arrival, estimate, deadline=None, upper=None, actual=None):
    return RequestItem(
        req_id=req_id, inter_arrival=0.0, service_time=actual or estimate,
        cpu_req=1, ram_req=2, workload_type=1,
        predicted_service=estimate, upper_service=upper,
        deadline=deadline, arrival_timestamp=arrival,
    )


def test_rf_splits_are_disjoint_and_coverage_is_held_out():
    model, metrics, outputs = train_calibrate_test_predictor(n_samples=120, noise_level=1.0, seed=7)
    assert metrics["n_train"] + metrics["n_calibration"] + metrics["n_validation"] + metrics["n_test"] == 120
    assert len(outputs["test_actual"]) == metrics["n_test"]
    assert metrics["RMSE"] >= 0
    assert 0 <= metrics["upper_coverage"] <= 1
    assert np.all(model.predict_upper([[4, 8, 1]]) >= model.predict([[4, 8, 1]]))


def test_fixed_seed_training_and_trace_generation_are_reproducible():
    from ml.rf_service_time import train_calibrate_test_predictor
    from workload.generator import generate_trace

    p1, m1, a1 = train_calibrate_test_predictor(n_samples=60, noise_level=2.0, seed=33)
    p2, m2, a2 = train_calibrate_test_predictor(n_samples=60, noise_level=2.0, seed=33)
    assert m1 == m2
    np.testing.assert_array_equal(a1["test_prediction"], a2["test_prediction"])
    np.testing.assert_array_equal(p1.predict_upper([[2, 4, 1]]), p2.predict_upper([[2, 4, 1]]))

    t1 = generate_trace("repro", num_requests=20, seed=77)
    t2 = generate_trace("repro", num_requests=20, seed=77)
    assert t1.get_service_series() == t2.get_service_series()
    assert t1.get_arrival_series() == t2.get_arrival_series()


def test_invalid_features_and_small_splits_fail_clearly():
    model, _, _ = train_calibrate_test_predictor(n_samples=50, seed=2)
    with pytest.raises(ValueError):
        model.predict([[np.nan, 2, 1]])
    with pytest.raises(ValueError):
        model.predict(pd.DataFrame({"CPU": [1], "RAM": [2]}))
    with pytest.raises(ValueError):
        train_calibrate_test_predictor(n_samples=4)


def test_dispatch_deadlines_slack_and_deterministic_ties():
    urgent = task(3, 0, 5, deadline=3, upper=7)
    relaxed = task(1, 0, 1, deadline=100, upper=2)
    no_deadline = task(0, 0, 0.5, upper=0.5)
    policy = SCHEDULERS["Proposed"]
    assert policy.choose([urgent, relaxed, no_deadline], now=0, queue_wait=0) == 1
    assert policy.choose([no_deadline, urgent], now=0, queue_wait=0) == 1
    zero_slack = task(4, 0, 2, deadline=5, upper=5)
    assert policy.choose([urgent, zero_slack], now=0, queue_wait=0) == 1
    assert policy.choose([urgent], now=10, queue_wait=0) == 0  # singleton infeasible task still runs

    tied_a, tied_b = task(1, 2, 4), task(2, 2, 4)
    assert SCHEDULERS["SJF predicted"].choose([tied_b, tied_a], 0) == 1
    assert SCHEDULERS["FCFS"].choose([tied_b, tied_a], 0) == 1


def test_queue_empty_and_overflow_and_deadline_metric():
    queue = ProcessingQueue(capacity=1)
    assert queue.dequeue_by(lambda _: 0) is None
    assert queue.enqueue("first")
    assert not queue.enqueue("overflow")
    assert queue.dequeue_by(lambda items: 0) == "first"

    metrics = SimulationMetrics(sla_target=2)
    metrics.record_initial_capacity(1)
    metrics.record_arrival()
    metrics.record_completion(wait_time=0.0, total_latency=1.0)
    metrics.record_deadline_completion(deadline=0.5, completion_time=1.0)
    result = metrics.finalize(1.0, "x", "y")
    assert result.deadline_violation_rate == 1.0

