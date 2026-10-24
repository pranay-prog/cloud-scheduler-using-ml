"""Benchmark suite comparing Fixed, Threshold, and Variability-Aware autoscalers.

Evaluates policies across 3 canonical traces (A1 Low, A2 Moderate, A3 Bursty),
records metrics, writes benchmark_results.csv, renders a Matplotlib comparison chart,
and prints a Markdown results table.
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from workload.generator import get_canonical_benchmark_traces, WorkloadTrace
from autoscaler.threshold import ThresholdAutoscaler
from autoscaler.variability_aware import VariabilityAwareAutoscaler
from simulation.runner import SimulationRunner
from simulation.metrics import SimulationResult


class FixedAutoscaler:
    """Fixed-capacity baseline policy."""

    def __init__(self, k: int = 6):
        self.k: int = int(k)

    def decide(self, stats: dict, current_k: int) -> int:
        return self.k


def train_rf_service_predictor(seed: int = 42) -> RandomForestRegressor:
    """Train Random Forest model to predict task execution time from request features."""
    rng = np.random.RandomState(seed)
    n_samples = 2000

    cpu = rng.randint(1, 17, size=n_samples)
    ram = rng.randint(2, 65, size=n_samples)
    w_type = rng.randint(1, 4, size=n_samples)

    # Underlying service time relationship with noise
    base = (cpu * 0.05) + (ram * 0.015) + (w_type * 0.2) + 0.3
    noise = rng.normal(0, 0.15, size=n_samples)
    target = np.clip(base + noise, 0.05, None)

    X = np.column_stack([cpu, ram, w_type])
    rf = RandomForestRegressor(n_estimators=60, max_depth=10, random_state=seed)
    rf.fit(X, target)
    return rf


def run_benchmark(
    include_rf_policy: bool = True,
    num_requests: int = 800,
    sla_target: float = 2.0,
    output_csv: str = "benchmark_results.csv",
    output_chart: str = "benchmark_comparison.png",
) -> pd.DataFrame:
    """Execute complete benchmark suite across traces and autoscaling policies."""
    print(">>> Initializing Canonical Workload Traces (A1, A2, A3)...")
    traces = get_canonical_benchmark_traces(
        num_requests=num_requests,
        target_lambda=4.0,
        target_mu=1.0,
        seed=42,
    )

    print(">>> Training Random Forest Service Time Predictor...")
    rf_model = train_rf_service_predictor()

    policies: List[Tuple[str, callable, bool]] = [
        ("Fixed (k=6)", lambda: FixedAutoscaler(k=6), False),
        ("Threshold", lambda: ThresholdAutoscaler(high_threshold=12, low_threshold=4, k_min=2, k_max=16), False),
        ("Variability-Aware (Empirical)", lambda: VariabilityAwareAutoscaler(sla_target=sla_target, k_min=2, k_max=16), False),
    ]

    if include_rf_policy:
        policies.append(
            ("Variability-Aware (RF)", lambda: VariabilityAwareAutoscaler(sla_target=sla_target, k_min=2, k_max=16), True)
        )

    results: List[Dict[str, any]] = []

    for trace_key, trace in traces.items():
        print(f"\n=======================================================")
        print(f"  Evaluating Trace: {trace.name}")
        print(f"=======================================================")

        for policy_name, policy_factory, use_rf in policies:
            autoscaler_inst = policy_factory()
            runner = SimulationRunner(
                trace=trace,
                autoscaler=autoscaler_inst,
                policy_name=policy_name,
                initial_k=4,
                boot_delay=10.0,
                buffer_capacity=600,
                queue_capacity=5000,
                control_interval=4.0,
                sla_target=sla_target,
                rf_model=rf_model if use_rf else None,
                use_rf_cs2=use_rf,
            )

            res: SimulationResult = runner.run()
            res_dict = res.as_dict()
            results.append(res_dict)
            print(
                f"  [{policy_name:<30}] Avg Lat: {res.avg_latency:.2f}s | "
                f"P95: {res.p95_latency:.2f}s | "
                f"VM-Hours: {res.vm_hours:.4f} | "
                f"SLA Violations: {res.sla_violation_rate * 100:.1f}% | "
                f"Overflows: {res.rejected_requests}"
            )

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"\n>>> Benchmark results saved to: {output_csv}")

    # Generate comparative Matplotlib chart
    generate_comparison_chart(df_results, output_chart)
    print(f">>> Benchmark visualization saved to: {output_chart}")

    # Render Markdown table
    print("\n" + "=" * 80)
    print("                      BENCHMARK RESULTS TABLE")
    print("=" * 80)
    md_table = format_markdown_table(df_results)
    print(md_table)
    print("=" * 80 + "\n")

    return df_results


def generate_comparison_chart(df: pd.DataFrame, output_path: str) -> None:
    """Generate dual-panel bar chart comparing P95 latency and VM-hours."""
    traces = df["trace"].unique()
    policies = df["policy"].unique()
    n_traces = len(traces)
    n_policies = len(policies)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    width = 0.8 / n_policies
    x = np.arange(n_traces)

    color_map = {
        "Fixed (k=6)": "#95a5a6",
        "Threshold": "#e74c3c",
        "Variability-Aware (Empirical)": "#2ecc71",
        "Variability-Aware (RF)": "#3498db",
    }

    # Chart 1: P95 Latency (Lower is better)
    for i, policy in enumerate(policies):
        subset = df[df["policy"] == policy]
        vals = [
            float(subset[subset["trace"] == t]["p95_latency"].values[0])
            for t in traces
        ]
        offsets = x - 0.4 + (i + 0.5) * width
        bars = ax1.bar(offsets, vals, width, label=policy, color=color_map.get(policy, "#333333"), edgecolor="black", alpha=0.85)
        for b in bars:
            height = b.get_height()
            ax1.annotate(f"{height:.2f}s", xy=(b.get_x() + b.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, weight="bold")

    ax1.set_title("P95 Latency Across Workload Traces (Lower is Better)", weight="bold")
    ax1.set_ylabel("P95 Latency (seconds)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([t.split("(")[0].strip() for t in traces], weight="bold")
    ax1.axhline(y=2.0, color="red", linestyle="--", label="SLA Target (2.0s)", alpha=0.7)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Chart 2: VM-Hours (Resource Cost)
    for i, policy in enumerate(policies):
        subset = df[df["policy"] == policy]
        vals = [
            float(subset[subset["trace"] == t]["vm_hours"].values[0])
            for t in traces
        ]
        offsets = x - 0.4 + (i + 0.5) * width
        bars = ax2.bar(offsets, vals, width, label=policy, color=color_map.get(policy, "#333333"), edgecolor="black", alpha=0.85)
        for b in bars:
            height = b.get_height()
            ax2.annotate(f"{height:.3f}", xy=(b.get_x() + b.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, weight="bold")

    ax2.set_title("Resource Consumption in VM-Hours", weight="bold")
    ax2.set_ylabel("VM-Hours")
    ax2.set_xticks(x)
    ax2.set_xticklabels([t.split("(")[0].strip() for t in traces], weight="bold")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def format_markdown_table(df: pd.DataFrame) -> str:
    """Format benchmark results as a clean Markdown table."""
    lines = [
        "| Trace | Policy | Avg Latency (s) | P95 Latency (s) | P99 Latency (s) | VM-Hours | SLA Viol. (%) | Overflows |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, row in df.iterrows():
        trace_short = row["trace"].split("(")[0].strip()
        lines.append(
            f"| {trace_short} | {row['policy']} | "
            f"{row['avg_latency']:.2f} | {row['p95_latency']:.2f} | {row['p99_latency']:.2f} | "
            f"{row['vm_hours']:.4f} | {row['sla_violation_rate'] * 100:.1f}% | {row['rejected']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run autoscaling benchmarks")
    parser.add_argument("--rf", action="store_true", default=True, help="Include RF variability predictor")
    parser.add_argument("--requests", type=int, default=800, help="Number of requests per trace")
    parser.add_argument("--output-csv", default="benchmark_results.csv", help="CSV results path")
    parser.add_argument("--output-chart", default="benchmark_comparison.png", help="Plot output path")
    args = parser.parse_args()

    run_benchmark(include_rf_policy=args.rf, num_requests=args.requests,
                  output_csv=args.output_csv, output_chart=args.output_chart)
