# Variability-Aware G/G/k Autoscaling Framework for Cloud Storage

[![CI](https://github.com/pranay-prog/cloud-scheduler-using-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/pranay-prog/cloud-scheduler-using-ml/actions/workflows/ci.yml)

A research-driven, variability-aware autoscaling framework designed for cloud storage and compute services. This project bridges classical operations research ($M/M/k$ Erlang-C and $G/G/k$ Allen-Cunneen / Kingman heavy-traffic queueing theory) with predictive machine learning (Random Forest) to provision resources under bursty stochastic workloads.

---

## 🎯 Core Novelty & Highlights

Standard cloud autoscalers only monitor scalar queue backlog $Q(t)$ or CPU utilization, remaining blind to request arrival burstiness ($C_a^2$) and heavy-tailed service times ($C_s^2$). This project introduces:

1. **Separated System State**: Explicitly distinguishes between ingress absorption storage buffer $B(t)$ (with overflow tracking) and execution processing queue $Q(t)$.
2. **$G/G/k$ Closed-Form Delay Prediction**: Uses the Allen-Cunneen heavy-traffic approximation:
   $$V = \frac{C_a^2 + C_s^2}{2}, \quad W_q^{G/G/k} = V \cdot \frac{P_W(k, \lambda, \mu)}{k\mu - \lambda}$$
3. **Critical Variability Boost**:
   - **Bursty Workloads ($C_a^2 > 1.5$)**: Enforces a strict safety margin (e.g., target delay $\le 80\%$ SLA) to absorb burst clusters.
   - **Smooth Workloads ($C_a^2 \le 0.5$)**: Enables tight packing (100% of SLA) to save VM-hours.
4. **Machine Learning Oracle**: Random Forest regressor predicts individual execution times and supplies estimated service variability ($\hat{C_s^2}$).
5. **Realistic Actuation Latency**: Simulates a 10-second VM spin-up delay ($T_{\text{scale}} = 10\text{s}$) during scale-up.

---

## 📁 Repository Structure

```
cloud-scheduler-using-ml/
├── autoscaler/
│   ├── __init__.py
│   ├── threshold.py            # Baseline static Q(t) threshold autoscaler
│   └── variability_aware.py    # G/G/k variability-aware autoscaler (Novelty)
├── docs/
│   ├── ARCHITECTURE.md         # 11-step Mermaid architecture diagram
│   ├── HYPOTHESIS.md           # Formal scientific hypothesis (H1)
│   ├── MIGRATION_PLAN.md       # Full architectural audit and migration roadmap
│   └── NOVELTY.md              # Research novelty statement
├── queueing/
│   ├── __init__.py
│   ├── erlang_c.py             # M/M/k Erlang-C delay & waiting probability P_W
│   └── kingman.py              # G/G/k Allen-Cunneen / Kingman approximation
├── simulation/
│   ├── __init__.py
│   ├── benchmark.py            # Head-to-head benchmark across A1, A2, A3 traces
│   ├── metrics.py              # Average/P95/P99 latency, VM-hours, SLA violations
│   └── runner.py               # SimPy discrete-event simulation engine
├── system/
│   ├── __init__.py
│   ├── processing_queue.py     # Q(t) processing queue model
│   └── storage_buffer.py       # B(t) storage buffer model with overflow rejection
├── tests/
│   ├── test_autoscalers.py     # Unit tests for buffer, queue, and autoscalers
│   ├── test_characterizer.py   # Unit tests for moment & variability estimation
│   ├── test_erlang_c.py        # Unit tests for M/M/k Erlang-C & stability
│   └── test_kingman.py         # Unit tests for G/G/k Kingman consistency
├── workload/
│   ├── __init__.py
│   ├── characterizer.py        # Statistical moments: lambda, mu, C_a^2, C_s^2
│   └── generator.py            # Trace generator: A1 (Low), A2 (Mod), A3 (Bursty)
├── app.py                      # Interactive Streamlit dashboard
├── main.py                     # Original Random Forest simulation
└── requirements.txt            # Dependencies
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install required packages:
```bash
git clone https://github.com/pranay-prog/cloud-scheduler-using-ml.git
cd cloud-scheduler-using-ml
pip install -r requirements.txt
```

### 2. Running Unit Tests

Run the full test suite using `pytest` (with test coverage tracking):
```bash
pytest --cov=. --cov-report=term-missing tests/
```

Or run via Python's built-in `unittest`:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 3. Running the Benchmark

Execute the benchmark suite comparing **Fixed**, **Threshold**, and **Variability-Aware** policies across the canonical traces (`A1` Low, `A2` Moderate, `A3` Bursty):

```bash
python simulation/benchmark.py --requests 800
```

This generates:
- `benchmark_results.csv`: Complete numerical metrics.
- `benchmark_comparison.png`: Dual-panel comparative visualization (P95 Latency vs. VM-Hours).
- A formatted Markdown results table printed directly to stdout.

### 4. Running the Live Animated Frontend Dashboard

Launch the live animated dark-mode Streamlit dashboard with real-time SimPy telemetry, animated pipeline visualization, and burst injection controls:
```bash
streamlit run frontend/dashboard.py
```

### 5. Running the Original Streamlit Demo

Launch the baseline scheduling comparison dashboard:
```bash
streamlit run app.py
```

---

## 🔬 Scientific Foundations & Documentation
- Read [`docs/HYPOTHESIS.md`](file:///c:/Users/anshu/OneDrive/Desktop/RANDOM%20FOREST%20CLOUD%20SCHEDULING/docs/HYPOTHESIS.md) for research hypotheses.
- Read [`docs/NOVELTY.md`](file:///c:/Users/anshu/OneDrive/Desktop/RANDOM%20FOREST%20CLOUD%20SCHEDULING/docs/NOVELTY.md) for novelty articulation.
- Read [`docs/ARCHITECTURE.md`](file:///c:/Users/anshu/OneDrive/Desktop/RANDOM%20FOREST%20CLOUD%20SCHEDULING/docs/ARCHITECTURE.md) for the 11-step Mermaid pipeline diagram.
