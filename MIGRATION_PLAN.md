# Migration Plan: G/G/k Variability-Aware Autoscaling Framework for Cloud Storage

## Executive Summary
This document outlines the architectural migration of `cloud-scheduler-using-ml` from an empirical Random Forest + heuristic queueing simulation to a rigorous **$G/G/k$ variability-aware autoscaling framework** designed for cloud storage systems.

---

## 1. Current State Audit

### 1.1 Existing Components
1. **`main.py` (CLI Experiment)**
   - Generates synthetic cloud task requests (CPU, RAM, UserType) with Gaussian execution noise.
   - Trains `RandomForestRegressor` to predict runtime.
   - Runs a `simpy.PriorityResource` simulation comparing standard FCFS vs. RF-predicted Shortest Job First (RF-SJF).
   - Generates static Matplotlib comparative charts.

2. **`app.py` (Streamlit Dashboard)**
   - Provides a browser GUI for interactive simulation runs.
   - Allows varying VMs, arrival rate $\lambda$, workload size, and noise.
   - Evaluates FCFS, Heuristic SJF, and RF-SJF.

3. **`requirements.txt`**
   - Core libraries: `streamlit`, `simpy`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`.

### 1.2 Current Limitations & Gaps
- **No Analytical Queueing Formulas**: System reliance on SimPy empirical runs without mathematical queueing bounds ($M/M/k$ or $G/G/k$).
- **Neglect of Workload Variability**: Ignores inter-arrival burstiness ($C_a^2$) and service time dispersion ($C_s^2$), critical for cloud storage I/O traffic.
- **Lack of Autoscaling Logic**: No closed-form solver to calculate minimum node/server count $k^*$ required to meet latency Service Level Agreements (SLAs).
- **Storage-Specific Semantics**: Tasks currently represent generic CPU/RAM jobs rather than storage I/O operations (IOPS, read/write ratios, block sizes, queue depths).

---

## 2. Target Architecture

```
cloud-scheduler-using-ml/
├── workload/
│   ├── __init__.py
│   └── characterizer.py       # E[A], E[S], lambda, mu, C_a^2, C_s^2 estimation
├── queueing/
│   ├── __init__.py
│   ├── erlang_c.py           # M/M/k Erlang-C delay & probability of waiting
│   ├── kingman.py            # G/G/k Allen-Cunneen / Kingman approximation
│   └── autoscaler.py         # SLA-driven minimum capacity k* solver
├── storage/
│   ├── __init__.py
│   └── storage_model.py      # Cloud storage workload generator (IOPS, bursts)
├── tests/
│   ├── __init__.py
│   ├── test_characterizer.py
│   ├── test_erlang_c.py
│   └── test_kingman.py
├── app.py                    # Enhanced Streamlit UI with G/G/k autoscaling
├── main.py                   # Preserved original experiment + G/G/k benchmark
└── requirements.txt
```

---

## 3. Mathematical Foundations

### 3.1 Workload Characterization
Given discrete inter-arrival times $A$ and service durations $S$:
- Arrival Rate: $\lambda = \frac{1}{\mathbb{E}[A]}$
- Service Rate: $\mu = \frac{1}{\mathbb{E}[S]}$
- Squared Coefficient of Variation (Arrivals):
  $$C_a^2 = \frac{\mathrm{Var}(A)}{(\mathbb{E}[A])^2}$$
- Squared Coefficient of Variation (Service):
  $$C_s^2 = \frac{\mathrm{Var}(S)}{(\mathbb{E}[S])^2}$$

### 3.2 Erlang-C Formula ($M/M/k$ Baseline)
For offered load $a = \lambda / \mu$ and traffic intensity $\rho = a / k < 1$:
$$P_W = \frac{\frac{a^k}{k!} \frac{1}{1 - \rho}}{\sum_{n=0}^{k-1} \frac{a^n}{n!} + \frac{a^k}{k!} \frac{1}{1 - \rho}}$$
Expected queue wait time:
$$W_q^{M/M/k} = \frac{P_W}{k\mu - \lambda}$$

### 3.3 Allen-Cunneen / Kingman Approximation ($G/G/k$)
Accounts for general inter-arrival and service distributions via variability factor $V$:
$$V = \frac{C_a^2 + C_s^2}{2}$$
$$W_q^{G/G/k} \approx V \cdot W_q^{M/M/k} = \left(\frac{C_a^2 + C_s^2}{2}\right) \frac{P_W}{k\mu - \lambda}$$

*Consistency Condition*: When $C_a^2 = 1$ and $C_s^2 = 1$, $V = 1$ and $W_q^{G/G/k} \equiv W_q^{M/M/k}$.

---

## 4. Phased Implementation Roadmap

- [x] **Phase 0: Audit & Migration Planning** (This document)
- [ ] **Phase 1: Mathematical Foundations**
  - Implement `workload/characterizer.py`
  - Implement `queueing/erlang_c.py`
  - Implement `queueing/kingman.py`
  - Implement unit test suite verifying mathematical consistency
- [ ] **Phase 2: Autoscaling & Capacity Planning Engine**
  - Formulate autoscaler determining $k^* = \min \{ k \in \mathbb{N} \mid \rho < 1 \land W_q(k) \le \mathrm{SLA} \}$
- [ ] **Phase 3: Cloud Storage Workload Modeling**
  - Synthesize realistic trace distributions (e.g., Pareto/log-normal bursts for storage I/O)
- [ ] **Phase 4: Simulation Integration & Benchmark Comparison**
  - Connect analytical formulas with SimPy discrete event validation
- [ ] **Phase 5: Interactive Dashboard Evolution**
  - Upgrade `app.py` with G/G/k autoscaling visualizations, SLA compliance contours, and live parameter tuning.
