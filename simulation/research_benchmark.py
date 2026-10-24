"""Reproducible deadline/uncertainty scheduling benchmark.

The default grid spans load, task duration, deadline tightness, arrival shape,
and prediction-noise settings. Repeated seeds are paired across every scheduler.
Results are simulation outputs, not real cloud expenditure or SLA guarantees.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.rf_service_time import train_calibrate_test_predictor
from simulation.runner import SimulationRunner
from simulation.scheduling import SCHEDULERS
from workload.generator import generate_trace


class FixedCapacity:
    def __init__(self, k: int = 4):
        self.k = k

    def decide(self, stats: dict, current_k: int) -> int:
        return self.k


def run_research_benchmark(
    requests: int = 300,
    repeats: int = 2,
    output_dir: str = "research_results",
    quick: bool = False,
) -> pd.DataFrame:
    """Run paired scheduler comparisons over the documented factor grid."""
    os.makedirs(output_dir, exist_ok=True)
    levels = {
        "load": ["low", "medium", "high"],
        "duration": ["short", "medium", "long"],
        "deadline": ["tight", "relaxed"],
        "arrival": ["steady", "bursty"],
        "noise": ["low", "high"],
    }
    if quick:
        levels = {"load": ["medium"], "duration": ["medium"], "deadline": ["tight", "relaxed"], "arrival": ["steady", "bursty"], "noise": ["low"]}

    rows: list[dict[str, Any]] = []
    model_cache: dict[tuple[str, str], tuple[Any, dict[str, float]]] = {}
    factors = list(itertools.product(*(levels[k] for k in levels)))
    for case_index, values in enumerate(factors):
        cfg = dict(zip(levels.keys(), values))
        for repeat in range(repeats):
            seed = 4200 + case_index * 101 + repeat
            mu = {"short": 2.0, "medium": 1.0, "long": 0.5}[cfg["duration"]]
            load = {"low": 0.50, "medium": 0.80, "high": 1.05}[cfg["load"]]
            arrival_rate = max(0.01, 4 * mu * load)
            ca2 = 0.5 if cfg["arrival"] == "steady" else 3.0
            distribution = "moderate"  # fixed so arrival variability is not confounded with service distribution
            deadline_factor = 2.0 if cfg["deadline"] == "tight" else 10.0
            noise_fraction = 0.05 if cfg["noise"] == "low" else 0.50
            noise_level = (1.0 / mu) * noise_fraction
            trace = generate_trace(
                name=f"{cfg['load']}-{cfg['duration']}-{cfg['deadline']}-{cfg['arrival']}-{cfg['noise']}",
                num_requests=requests,
                target_lambda=arrival_rate,
                target_mu=mu,
                target_ca2=ca2,
                service_distribution=distribution,
                seed=seed,
                deadline_slack_factor=deadline_factor,
                runtime_noise_level=noise_level,
            )
            # Four independently generated traces keep model training, residual
            # calibration, validation, and test data disjoint and aligned with
            # the task runtime generator used by the simulation.
            model_key = (cfg["duration"], cfg["noise"])
            if model_key not in model_cache:
                duration_index = {"short": 0, "medium": 1, "long": 2}[cfg["duration"]]
                noise_index = {"low": 0, "high": 1}[cfg["noise"]]
                model_seed = 4200 + duration_index * 100 + noise_index

                def make_model_rows(count: int, split_seed: int) -> pd.DataFrame:
                    generated = generate_trace(
                        name="rf-fit-split", num_requests=count,
                        target_lambda=4 * mu * 0.8, target_mu=mu,
                        target_ca2=0.5, service_distribution=distribution,
                        seed=split_seed, runtime_noise_level=noise_level,
                    ).to_dataframe()
                    return generated.rename(columns={
                        "cpu_req": "CPU", "ram_req": "RAM", "workload_type": "Type",
                        "service_time": "Actual_Time",
                    })

                training = make_model_rows(720, model_seed + 1)
                calibration = make_model_rows(180, model_seed + 2)
                validation = make_model_rows(120, model_seed + 3)
                testing = make_model_rows(180, model_seed + 4)
                model, ml_metrics, _ = train_calibrate_test_predictor(
                    seed=model_seed,
                    train_data=training,
                    calibration_data=calibration,
                    validation_data=validation,
                    test_data=testing,
                )
                model_cache[model_key] = (model, ml_metrics)
            model, ml_metrics = model_cache[model_key]
            for policy_name, policy in SCHEDULERS.items():
                runner = SimulationRunner(
                    trace=trace,
                    autoscaler=FixedCapacity(4),
                    policy_name=policy_name,
                    initial_k=4,
                    boot_delay=0.0,
                    control_interval=4.0,
                    sla_target=2.0,
                    rf_model=model,
                    scheduling_policy=policy,
                    upper_residual_quantile=model.upper_residual_quantile if policy.use_uncertainty else 0.0,
                )
                result = runner.run().as_dict()
                rows.append({
                    **result,
                    **cfg,
                    "repeat": repeat,
                    "seed": seed,
                    "rf_mae": ml_metrics["MAE"],
                    "rf_rmse": ml_metrics["RMSE"],
                    "rf_validation_mae": ml_metrics["validation_MAE"],
                    "rf_validation_rmse": ml_metrics["validation_RMSE"],
                    "validation_upper_runtime_coverage": ml_metrics["validation_upper_coverage"],
                    "upper_runtime_coverage": ml_metrics["upper_coverage"],
                    "calibration_quantile": ml_metrics["calibration_quantile"],
                    "upper_residual_quantile": ml_metrics["upper_residual_quantile"],
                    "train_rows": ml_metrics["n_train"],
                    "calibration_rows": ml_metrics["n_calibration"],
                    "validation_rows": ml_metrics["n_validation"],
                    "test_rows": ml_metrics["n_test"],
                    "oracle_scheduler": policy.oracle_actual_runtime,
                    "vm_hours_is_simulated_cost": True,
                })
        print(f"Completed workload case {case_index + 1}/{len(factors)}: {cfg}", flush=True)
    frame = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    json_path = os.path.join(output_dir, "benchmark_results.json")
    frame.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(frame.to_dict(orient="records"), f, indent=2)
    _plot_results(frame, os.path.join(output_dir, "benchmark_comparison.png"))
    summary = frame.groupby(["policy", "load", "duration", "deadline", "arrival", "noise"], as_index=False).agg(
        deadline_violation_rate=("deadline_violation_rate", "mean"),
        avg_latency=("avg_latency", "mean"),
        p95_latency=("p95_latency", "mean"),
        throughput=("throughput", "mean"),
        avg_waiting_time=("avg_waiting_time", "mean"),
        avg_queue_length=("avg_queue_length", "mean"),
        vm_utilization=("vm_utilization", "mean"),
        simulated_cost_vm_hours=("simulated_cost_vm_hours", "mean"),
    )
    summary.to_csv(os.path.join(output_dir, "benchmark_summary.csv"), index=False)
    overall = frame.groupby("policy", as_index=False).agg(
        runs=("seed", "count"),
        deadline_violation_rate=("deadline_violation_rate", "mean"),
        avg_latency=("avg_latency", "mean"),
        p95_latency=("p95_latency", "mean"),
        p99_latency=("p99_latency", "mean"),
        throughput=("throughput", "mean"),
        avg_waiting_time=("avg_waiting_time", "mean"),
        avg_queue_length=("avg_queue_length", "mean"),
        vm_utilization=("vm_utilization", "mean"),
        simulated_cost_vm_hours=("simulated_cost_vm_hours", "mean"),
        rf_mae=("rf_mae", "mean"), rf_rmse=("rf_rmse", "mean"),
        rf_validation_mae=("rf_validation_mae", "mean"), rf_validation_rmse=("rf_validation_rmse", "mean"),
        validation_upper_runtime_coverage=("validation_upper_runtime_coverage", "mean"),
        upper_runtime_coverage=("upper_runtime_coverage", "mean"),
    )
    overall.to_json(os.path.join(output_dir, "benchmark_overall_summary.json"), orient="records", indent=2)
    return frame


def _plot_results(frame: pd.DataFrame, output_path: str) -> None:
    avg = frame.groupby("policy", as_index=False)[["deadline_violation_rate", "p95_latency", "simulated_cost_vm_hours"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    specs = [("deadline_violation_rate", "Deadline violation rate"), ("p95_latency", "P95 response latency (s)"), ("simulated_cost_vm_hours", "Simulated VM-hours")]
    for ax, (column, title) in zip(axes, specs):
        ax.bar(avg["policy"], avg[column], color="#457b9d")
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=65)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-dir", default="research_results")
    parser.add_argument("--quick", action="store_true", help="Use one load/duration/noise level and all arrival/deadline variants")
    parser.add_argument("--summarize-only", action="store_true", help="Summarize existing benchmark_results.csv without rerunning simulations")
    args = parser.parse_args()
    if args.summarize_only:
        source = os.path.join(args.output_dir, "benchmark_results.csv")
        existing = pd.read_csv(source)
        aggregate = existing.groupby("policy", as_index=False).agg(
            runs=("seed", "count"), deadline_violation_rate=("deadline_violation_rate", "mean"),
            avg_latency=("avg_latency", "mean"), p95_latency=("p95_latency", "mean"),
            p99_latency=("p99_latency", "mean"), throughput=("throughput", "mean"),
            avg_waiting_time=("avg_waiting_time", "mean"), avg_queue_length=("avg_queue_length", "mean"),
            vm_utilization=("vm_utilization", "mean"), simulated_cost_vm_hours=("simulated_cost_vm_hours", "mean"),
            rf_validation_mae=("rf_validation_mae", "mean"), rf_validation_rmse=("rf_validation_rmse", "mean"),
            validation_upper_runtime_coverage=("validation_upper_runtime_coverage", "mean"),
            rf_mae=("rf_mae", "mean"), rf_rmse=("rf_rmse", "mean"), upper_runtime_coverage=("upper_runtime_coverage", "mean"),
        )
        aggregate.to_json(os.path.join(args.output_dir, "benchmark_overall_summary.json"), orient="records", indent=2)
        print(f"Summarized {len(existing)} runs")
        raise SystemExit(0)
    out = run_research_benchmark(args.requests, args.repeats, args.output_dir, args.quick)
    print(f"Wrote {len(out)} measured scheduler runs to {args.output_dir}")
