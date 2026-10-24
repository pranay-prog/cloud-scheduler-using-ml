# System Architecture: Variability-Aware G/G/k Autoscaler

## 11-Step Pipeline Architecture

The end-to-end framework implements an 11-step feedback control loop connecting ingress cloud storage workloads to closed-form queueing approximations, machine learning inference, and dynamic capacity provisioning.

```mermaid
graph TD
    Step1["1. Workload Ingress Data<br/>(Request Stream with CPU, RAM, IOPS)"] --> Step2["2. Workload Characterization<br/>(Compute lambda, mu, C_a^2, C_s^2)"]
    Step1 --> Step3["3. ML Oracle Inference<br/>(Random Forest predicts S & C_s^2)"]
    
    Step1 --> Step4["4. Storage Buffer State B(t)<br/>(Ingress Absorption & Overflow Check)"]
    Step4 --> Step5["5. Processing Queue State Q(t)<br/>(Active Worker Backlog & Drain Time)"]
    
    Step2 --> Step6["6. Erlang-C M/M/k Baseline<br/>(Offered load a, rho, P_W)"]
    Step3 -.-> Step2
    
    Step6 --> Step7["7. Kingman / Allen-Cunneen G/G/k<br/>(V = (C_a^2 + C_s^2)/2, W_q = V * W_q_MMk)"]
    Step2 --> Step7
    
    Step7 --> Step8["8. SLA Target Evaluation<br/>(Dynamic Variability Boost: 80% SLA if C_a^2 > 1.5)"]
    Step5 --> Step8
    
    Step8 --> Step9["9. Capacity Provisioning Decision<br/>(Compute target k* subject to T_scale = 10s boot lag)"]
    
    Step9 --> Step10["10. System Actuation<br/>(Scale SimPy worker pool, transfer B(t) to Q(t))"]
    
    Step10 --> Step11["11. Telemetry & Metric Feedback<br/>(Track Avg, P95, P99, VM-Hours, Overflow Rate)"]
    Step11 -.->|Continuous Rolling Feedback| Step2
```

---

## Detailed Step Descriptions

1. **Workload Ingress Data**: Cloud requests arrive with resource features and inter-arrival intervals.
2. **Workload Characterization**: Calculates $\lambda = 1/\mathbb{E}[A]$, $\mu = 1/\mathbb{E}[S]$, $C_a^2 = \mathrm{Var}(A)/\mathbb{E}[A]^2$, and $C_s^2 = \mathrm{Var}(S)/\mathbb{E}[S]^2$.
3. **ML Oracle Inference**: A Random Forest regressor predicts runtime based on task metadata to supply estimated service variability $\hat{C_s^2}$.
4. **Storage Buffer State $B(t)$**: Absorbs incoming traffic burstiness; enforces capacity bounds and records rejection overflow events.
5. **Processing Queue State $Q(t)$**: Holds admitted tasks ready for server dispatch; computes backlog drainage delay $\frac{Q(t)}{k\mu}$.
6. **Erlang-C Baseline ($M/M/k$)**: Determines offered load $a = \lambda / \mu$, stability $\rho = a / k < 1$, and probability of delay $P_W$.
7. **Kingman / Allen-Cunneen ($G/G/k$)**: Scales Erlang-C delay by variability factor $V = (C_a^2 + C_s^2)/2$.
8. **SLA Target Evaluation**: Applies variability safety boost: $\alpha = 0.8$ when $C_a^2 > 1.5$ (bursty), $\alpha = 1.0$ when $C_a^2 \le 0.5$ (smooth).
9. **Capacity Provisioning Decision**: Calculates optimal $k^*$ balancing latency SLA vs VM-hour cost, factoring in $T_{\text{scale}} = 10\text{s}$ node spin-up lag.
10. **System Actuation**: Adjusts SimPy active worker instances and serves tasks from $Q(t)$.
11. **Telemetry & Metric Feedback**: Integrates real-time P95/P99 latency, VM-hours, and buffer overflow telemetry back to the controller.

## Research dispatch path

The new task-ordering experiment is separate from capacity autoscaling. Tasks may carry an optional relative deadline allowance, converted to an absolute deadline on arrival. A Random Forest is trained on one split, calibrated from held-out residuals on a second, measured on a validation split, and scored on a disjoint test split. Dispatch may use FIFO, actual runtime (oracle only), predicted runtime, absolute deadline, predicted slack, or slack computed from the calibrated upper runtime. The policy observes waiting tasks when a worker requests its next task; already-running work is not preempted.

```mermaid
flowchart TD
  T[Generated tasks with optional deadlines] --> Q[Ingress buffer and processing queue]
  T --> RF[Random Forest prediction]
  TR[Train split] --> RF
  CA[Calibration split residuals] --> CAL[One-sided upper residual quantile]
  RF --> UP[Upper runtime estimate]
  CAL --> UP
  RF --> DISPATCH[Deterministic queue dispatch]
  UP --> SLACK[Estimated wait and deadline slack]
  Q --> DISPATCH
  SLACK --> DISPATCH
  DISPATCH --> W[SimPy worker pool]
  W --> M[Metrics and raw results]
  TE[Test split] --> EV[MAE, RMSE and empirical coverage]
```

Calibration and held-out test rows are disjoint. A single residual quantile gives the same additive margin to every task. Test coverage is reported empirically and does not imply a guarantee under distribution shift. See `simulation/research_benchmark.py` and the research section of the README for the exact metric definitions and benchmark factors.
