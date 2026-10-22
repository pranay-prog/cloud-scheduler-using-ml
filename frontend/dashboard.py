"""Live Animated Streamlit Frontend for Variability-Aware G/G/k Cloud Storage Autoscaler.

Run with:
    streamlit run frontend/dashboard.py
"""

from __future__ import annotations

import os
import sys
from collections import deque
from typing import Any, Dict, List

# Ensure project root is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import simpy
import streamlit as st

from autoscaler.threshold import ThresholdAutoscaler
from autoscaler.variability_aware import VariabilityAwareAutoscaler
from system.processing_queue import ProcessingQueue
from system.storage_buffer import StorageBuffer
from workload.characterizer import compute_arrival_metrics, compute_service_metrics

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="G/G/k Variability-Aware Autoscaler",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- 1. UI/UX DESIGN & CUSTOM CSS INJECTION ---
def inject_custom_css():
    """Injects modern dark-mode glassmorphism and neon aesthetics."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .stApp {
        background-color: #060911;
        background-image: 
            radial-gradient(circle at 15% 20%, rgba(0, 243, 255, 0.06) 0%, transparent 40%),
            radial-gradient(circle at 85% 80%, rgba(168, 85, 247, 0.08) 0%, transparent 45%),
            linear-gradient(180deg, #060911 0%, #0a0e1a 100%);
        color: #e2e8f0;
    }

    /* Neon Title & Headers */
    .neon-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f3ff 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin-bottom: 2px;
    }
    .neon-sub {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 20px;
    }

    /* Metric Grid & Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 20px;
    }
    .metric-box {
        background: rgba(13, 20, 36, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .metric-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #00f3ff, #a855f7);
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value-cyan {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.9rem;
        font-weight: 700;
        color: #00f3ff;
        text-shadow: 0 0 14px rgba(0, 243, 255, 0.4);
    }
    .metric-value-purple {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.9rem;
        font-weight: 700;
        color: #c084fc;
        text-shadow: 0 0 14px rgba(168, 85, 247, 0.4);
    }

    /* Alert Flashing Red for Latency > SLA */
    @keyframes alert-pulse {
        0%, 100% {
            color: #ff3366;
            text-shadow: 0 0 18px #ff0044;
            transform: scale(1.0);
        }
        50% {
            color: #ffffff;
            text-shadow: 0 0 6px #ff3366;
            transform: scale(1.02);
        }
    }
    .metric-value-alert {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.9rem;
        font-weight: 700;
        animation: alert-pulse 0.9s infinite ease-in-out;
    }

    /* Live Pipeline Visualization */
    .pipeline-wrapper {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(10, 16, 29, 0.85);
        border: 1px solid rgba(0, 243, 255, 0.22);
        border-radius: 16px;
        padding: 18px 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    }
    .stage-node {
        flex: 1;
        background: rgba(18, 27, 49, 0.9);
        border: 1px solid rgba(0, 243, 255, 0.25);
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
    }
    .stage-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .stage-counter {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.45rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .stage-badge {
        display: inline-block;
        font-size: 0.72rem;
        padding: 2px 8px;
        border-radius: 8px;
        background: rgba(0, 243, 255, 0.15);
        color: #38bdf8;
        margin-top: 4px;
        font-weight: 600;
    }
    .stage-badge-purple {
        background: rgba(168, 85, 247, 0.18);
        color: #c084fc;
    }
    .stage-badge-red {
        background: rgba(239, 68, 68, 0.22);
        color: #f87171;
    }

    /* Animated Conduit / Connector */
    .connector-pipe {
        flex: 0 0 65px;
        height: 6px;
        background: rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
        margin: 0 12px;
        border-radius: 3px;
    }
    .pipe-glow {
        position: absolute;
        top: 0;
        left: -30px;
        height: 100%;
        width: 30px;
        background: linear-gradient(90deg, transparent, #00f3ff, transparent);
        border-radius: 3px;
        animation: packet-flow 1.1s infinite linear;
    }
    .pipe-glow-purple {
        position: absolute;
        top: 0;
        left: -30px;
        height: 100%;
        width: 30px;
        background: linear-gradient(90deg, transparent, #a855f7, transparent);
        border-radius: 3px;
        animation: packet-flow 0.85s infinite linear;
    }
    @keyframes packet-flow {
        0% { left: -30px; opacity: 0; }
        30% { opacity: 1; }
        80% { opacity: 1; }
        100% { left: 100%; opacity: 0; }
    }

    section[data-testid="stSidebar"] {
        background-color: #080d19;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    </style>
    """
    st.html(css)


# --- 2. LIVE SIMULATION ENGINE WITH SIMPY ---
class LiveSimulationEngine:
    """SimPy-driven live engine that advances incrementally and reacts to interactive controls."""

    def __init__(self, buffer_cap: int = 500, queue_cap: int = 2500):
        self.buffer_cap = buffer_cap
        self.queue_cap = queue_cap
        self.reset()

    def reset(self):
        """Reinitialize all SimPy primitives and historical telemetry."""
        self.env = simpy.Environment()
        self.buffer = StorageBuffer(capacity=self.buffer_cap)
        self.queue = ProcessingQueue(capacity=self.queue_cap)
        self.task_event = self.env.event()

        # Capacity state
        self.initial_k = 4
        self.active_k = self.initial_k
        self.desired_k = self.initial_k
        self.booting_k = 0
        self.boot_delay = 10.0

        # Autoscalers
        self.va_autoscaler = VariabilityAwareAutoscaler(sla_target=2.0, k_min=2, k_max=20)
        self.threshold_autoscaler = ThresholdAutoscaler(high_threshold=12, low_threshold=4, k_min=2, k_max=20)

        # Workload knobs (dynamically set by sliders)
        self.target_lambda = 4.0
        self.target_ca2 = 1.0
        self.target_mu = 1.0
        self.target_cs2 = 1.0
        self.sla_target = 2.0
        self.policy_name = "Variability-Aware G/G/k"

        # Telemetry rolling history
        self.history_time: List[float] = []
        self.history_latency: List[float] = []
        self.history_p95: List[float] = []
        self.history_sla_target: List[float] = []
        self.history_lambda: List[float] = []
        self.history_queue: List[float] = []
        self.history_k: List[int] = []

        # Rolling observation windows
        self.rolling_arrivals = deque(maxlen=200)
        self.rolling_services = deque(maxlen=200)
        self.rolling_latencies = deque(maxlen=200)

        self.total_arrivals = 0
        self.total_completed = 0
        self.total_overflows = 0

        # Spawn initial workers
        for _ in range(self.initial_k):
            self.env.process(self._worker_loop(is_initial=True))

        # Launch generator and autoscaling controller
        self.env.process(self._traffic_generator())
        self.env.process(self._control_loop())

    def _trigger_task_event(self):
        if not self.task_event.triggered:
            self.task_event.succeed()
            self.task_event = self.env.event()

    def _worker_loop(self, is_initial: bool = False):
        if not is_initial and self.boot_delay > 0:
            yield self.env.timeout(self.boot_delay)

        self.booting_k = max(0, self.booting_k - 1)
        self.active_k += 1

        while True:
            # Scale-down check
            if self.active_k > self.desired_k:
                self.active_k -= 1
                return

            task = self.queue.dequeue()

            # Drain pending request from buffer into queue
            if not self.buffer.is_empty():
                buffered = self.buffer.pop()
                if buffered is not None:
                    self.queue.enqueue(buffered)

            if task is None:
                # Wait for task signal or poll
                yield self.task_event | self.env.timeout(0.5)
                continue

            # Process request
            yield self.env.timeout(task["service_time"])
            latency = self.env.now - task["arrival_time"]
            self.rolling_latencies.append(latency)
            self.total_completed += 1

    def _spawn_worker(self):
        self.booting_k += 1
        self.env.process(self._worker_loop(is_initial=False))

    def _traffic_generator(self):
        """Continuously synthesizes inter-arrival requests reflecting current UI sliders."""
        while True:
            lam = max(0.2, float(self.target_lambda))
            ca2 = max(0.05, float(self.target_ca2))

            # Gamma inter-arrival generation matching target Ca2
            shape_a = 1.0 / ca2
            scale_a = 1.0 / (shape_a * lam)
            inter_a = float(np.random.gamma(shape=shape_a, scale=scale_a))
            inter_a = max(0.001, inter_a)

            yield self.env.timeout(inter_a)
            self.total_arrivals += 1
            self.rolling_arrivals.append(inter_a)

            # Service time generation (mean = 1/mu)
            mu = max(0.2, float(self.target_mu))
            if ca2 > 2.5:
                # Bursty service if bursty workload
                s_time = float(np.random.pareto(2.5) + 1.0) * (0.6 / mu)
            else:
                s_time = float(np.random.exponential(scale=1.0 / mu))
            s_time = max(0.05, min(s_time, 15.0))
            self.rolling_services.append(s_time)

            task = {
                "arrival_time": self.env.now,
                "service_time": s_time,
            }

            # Ingress to B(t)
            admitted = self.buffer.push(task)
            if not admitted:
                self.total_overflows += 1
                continue

            # Transfer to Q(t) if space permits
            if not self.queue.is_full():
                item = self.buffer.pop()
                if item is not None:
                    self.queue.enqueue(item)
                    self._trigger_task_event()

    def _control_loop(self):
        """Periodic decision controller evaluating queueing and autoscaling."""
        while True:
            yield self.env.timeout(2.5)

            # Moment calculation from rolling history
            if len(self.rolling_arrivals) >= 10:
                obs_lam, _, _, obs_ca2 = compute_arrival_metrics(list(self.rolling_arrivals))
            else:
                obs_lam, obs_ca2 = self.target_lambda, self.target_ca2

            if len(self.rolling_services) >= 10:
                obs_mu, _, _, obs_cs2 = compute_service_metrics(list(self.rolling_services))
            else:
                obs_mu, obs_cs2 = self.target_mu, self.target_cs2

            stats = {
                "q_len": self.queue.backlog,
                "b_len": self.buffer.backlog,
                "arrival_rate": obs_lam,
                "service_rate": obs_mu,
                "ca2": obs_ca2,
                "cs2": obs_cs2,
            }

            # Select active policy
            if "Variability-Aware" in self.policy_name:
                self.va_autoscaler.sla_target = self.sla_target
                target_k = self.va_autoscaler.decide(stats, current_k=self.desired_k)
            elif "Threshold" in self.policy_name:
                target_k = self.threshold_autoscaler.decide(stats, current_k=self.desired_k)
            else:
                # Fixed
                target_k = 6

            self.desired_k = target_k
            total_provisioned = self.active_k + self.booting_k
            if target_k > total_provisioned:
                diff = target_k - total_provisioned
                for _ in range(diff):
                    self._spawn_worker()

    def step(self, duration: float = 1.0) -> Dict[str, Any]:
        """Advance the SimPy discrete-event simulation by duration seconds."""
        self.env.run(until=self.env.now + duration)

        now = self.env.now
        b_len = self.buffer.backlog
        q_len = self.queue.backlog

        if self.rolling_latencies:
            lats = list(self.rolling_latencies)
            avg_lat = float(np.mean(lats))
            p95_lat = float(np.percentile(lats, 95))
        else:
            avg_lat, p95_lat = 0.5, 0.5

        if len(self.rolling_arrivals) >= 10:
            obs_lam, _, _, obs_ca2 = compute_arrival_metrics(list(self.rolling_arrivals))
        else:
            obs_lam, obs_ca2 = self.target_lambda, self.target_ca2

        # Record telemetry series
        self.history_time.append(round(now, 1))
        self.history_latency.append(avg_lat)
        self.history_p95.append(p95_lat)
        self.history_sla_target.append(self.sla_target)
        self.history_lambda.append(obs_lam)
        self.history_queue.append(q_len)
        self.history_k.append(self.active_k)

        # Keep plot memory bounded
        if len(self.history_time) > 40:
            self.history_time.pop(0)
            self.history_latency.pop(0)
            self.history_p95.pop(0)
            self.history_sla_target.pop(0)
            self.history_lambda.pop(0)
            self.history_queue.pop(0)
            self.history_k.pop(0)

        return {
            "time": now,
            "b_len": b_len,
            "q_len": q_len,
            "active_k": self.active_k,
            "desired_k": self.desired_k,
            "booting_k": self.booting_k,
            "obs_lambda": obs_lam,
            "obs_ca2": obs_ca2,
            "avg_latency": avg_lat,
            "p95_latency": p95_lat,
            "sla_target": self.sla_target,
            "overflows": self.total_overflows,
            "completed": self.total_completed,
            "is_violating": p95_lat > self.sla_target,
        }


# --- 3. PIPELINE VISUALIZATION HTML BUILDER ---
def render_pipeline_html(data: Dict[str, Any]) -> str:
    """Generates the live animated HTML/CSS pipeline visualization."""
    b_val = data["b_len"]
    q_val = data["q_len"]
    k_val = data["active_k"]
    boot_val = data["booting_k"]
    ca2_val = data["obs_ca2"]

    bursty_tag = (
        '<span class="stage-badge-purple stage-badge">BURSTY (V-BOOST)</span>'
        if ca2_val > 1.5
        else '<span class="stage-badge">SMOOTH</span>'
    )

    booting_tag = (
        f'<span class="stage-badge-red stage-badge">+{boot_val} BOOTING</span>'
        if boot_val > 0
        else '<span class="stage-badge">STABLE POOL</span>'
    )

    html = (
        f'<div class="pipeline-wrapper">'
        f'<div class="stage-node">'
        f'<div class="stage-title">📦 Storage Buffer B(t)</div>'
        f'<div class="stage-counter">{b_val} <span style="font-size:0.85rem;color:#64748b;">reqs</span></div>'
        f'<div class="stage-badge">Ingress Absorption</div>'
        f'</div>'
        f'<div class="connector-pipe"><div class="pipe-glow"></div></div>'
        f'<div class="stage-node">'
        f'<div class="stage-title">⏳ Processing Queue Q(t)</div>'
        f'<div class="stage-counter">{q_val} <span style="font-size:0.85rem;color:#64748b;">tasks</span></div>'
        f'{bursty_tag}'
        f'</div>'
        f'<div class="connector-pipe"><div class="pipe-glow-purple"></div></div>'
        f'<div class="stage-node">'
        f'<div class="stage-title">🖥️ Active VM Pool (k)</div>'
        f'<div class="stage-counter" style="color:#a855f7;">{k_val} <span style="font-size:0.85rem;color:#64748b;">VMs</span></div>'
        f'{booting_tag}'
        f'</div>'
        f'</div>'
    )
    return html


# --- 4. METRIC CARDS HTML BUILDER ---
def render_metric_cards_html(step_data: Dict[str, Any]) -> str:
    """Generates the 4 glassmorphic metric cards in a single responsive flexbox row."""
    k_val = step_data['active_k']
    target_k = step_data['desired_k']
    booting_k = step_data['booting_k']
    ca2_val = step_data['obs_ca2']
    q_val = step_data['q_len']
    b_val = step_data['b_len']
    p95_val = step_data['p95_latency']
    is_violating = step_data['is_violating']

    lat_class = "metric-value-alert" if is_violating else "metric-value-cyan"
    lat_status = (
        "<span style='color:#ff3366; font-weight:700;'>🚨 SLA BREACH!</span>"
        if is_violating
        else "<span style='color:#10b981; font-weight:700;'>✅ WITHIN SLA</span>"
    )

    burst_note = (
        "<span style='color:#c084fc; font-weight:600;'>⚡ Bursty (Safety Boost)</span>"
        if ca2_val > 1.5
        else "<span style='color:#00f3ff; font-weight:600;'>🍃 Smooth</span>"
    )

    html = (
        f'<div class="metric-grid">'
        f'<div class="metric-box">'
        f'<div class="metric-label">Active VMs (k)</div>'
        f'<div class="metric-value-purple">{k_val}</div>'
        f'<span style="font-size:0.75rem; color:#94a3b8;">Target: {target_k} | Booting: +{booting_k}</span>'
        f'</div>'
        f'<div class="metric-box">'
        f'<div class="metric-label">Arrival Variability (C_a²)</div>'
        f'<div class="metric-value-cyan">{ca2_val:.2f}</div>'
        f'<span style="font-size:0.75rem;">{burst_note}</span>'
        f'</div>'
        f'<div class="metric-box">'
        f'<div class="metric-label">Queue Backlog Q(t)</div>'
        f'<div class="metric-value-cyan">{q_val}</div>'
        f'<span style="font-size:0.75rem; color:#94a3b8;">Buffer B(t): {b_val}</span>'
        f'</div>'
        f'<div class="metric-box">'
        f'<div class="metric-label">P95 Latency</div>'
        f'<div class="{lat_class}">{p95_val:.2f}s</div>'
        f'<span style="font-size:0.75rem;">{lat_status}</span>'
        f'</div>'
        f'</div>'
    )
    return html


# --- 5. PLOTLY CHARTS BUILDER ---
def create_plotly_charts(engine: LiveSimulationEngine):
    """Creates dual neon-styled dark-mode live charts."""
    times = engine.history_time
    p95_vals = engine.history_p95
    sla_vals = engine.history_sla_target
    lam_vals = engine.history_lambda
    q_vals = engine.history_queue

    # Chart 1: Latency vs SLA Target
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=times,
            y=p95_vals,
            mode="lines+markers",
            name="P95 Latency",
            line=dict(color="#00f3ff", width=3),
            fill="tozeroy",
            fillcolor="rgba(0, 243, 255, 0.08)",
            marker=dict(size=6, color="#00f3ff"),
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=times,
            y=sla_vals,
            mode="lines",
            name="SLA Target (Threshold)",
            line=dict(color="#ef4444", width=2.5, dash="dash"),
        )
    )
    fig1.update_layout(
        uirevision="constant",
        title=dict(
            text="⏱️ Latency vs SLA Target Over Time",
            font=dict(color="#f8fafc", size=15, family="Outfit"),
        ),
        paper_bgcolor="rgba(15, 23, 42, 0.7)",
        plot_bgcolor="rgba(11, 18, 34, 0.5)",
        margin=dict(l=40, r=20, t=45, b=35),
        height=320,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#94a3b8", size=11),
        ),
        xaxis=dict(
            title="Simulated Time (s)",
            color="#64748b",
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
        ),
        yaxis=dict(
            title="Latency (seconds)",
            color="#64748b",
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
        ),
    )

    # Chart 2: Arrival Rate (lambda) vs Queue Backlog Q(t)
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(
        go.Scatter(
            x=times,
            y=lam_vals,
            mode="lines+markers",
            name="Arrival Rate λ",
            line=dict(color="#c084fc", width=2.5),
            marker=dict(size=5, color="#c084fc"),
        ),
        secondary_y=False,
    )
    fig2.add_trace(
        go.Scatter(
            x=times,
            y=q_vals,
            mode="lines",
            name="Queue Backlog Q(t)",
            line=dict(color="#fbbf24", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(251, 191, 36, 0.08)",
        ),
        secondary_y=True,
    )
    fig2.update_layout(
        uirevision="constant",
        title=dict(
            text="📈 Arrival Rate (λ) vs Processing Queue Q(t)",
            font=dict(color="#f8fafc", size=15, family="Outfit"),
        ),
        paper_bgcolor="rgba(15, 23, 42, 0.7)",
        plot_bgcolor="rgba(11, 18, 34, 0.5)",
        margin=dict(l=40, r=40, t=45, b=35),
        height=320,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#94a3b8", size=11),
        ),
        xaxis=dict(
            title="Simulated Time (s)",
            color="#64748b",
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
        ),
    )
    fig2.update_yaxes(
        title_text="Arrival Rate λ (req/s)",
        color="#c084fc",
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.06)",
        secondary_y=False,
    )
    fig2.update_yaxes(
        title_text="Queue Q(t)",
        color="#fbbf24",
        showgrid=False,
        secondary_y=True,
    )

    return fig1, fig2


# --- 6. AUTO-REFRESHING LIVE FRAGMENT ---
@st.fragment(run_every=0.8)
def live_dashboard_view(engine: LiveSimulationEngine):
    """Auto-refreshing fragment that advances SimPy simulation without blocking user inputs."""
    if st.session_state.get("is_running", True):
        step_data = engine.step(duration=1.2)
    else:
        # Paused snapshot
        step_data = {
            "time": engine.env.now,
            "b_len": engine.buffer.backlog,
            "q_len": engine.queue.backlog,
            "active_k": engine.active_k,
            "desired_k": engine.desired_k,
            "booting_k": engine.booting_k,
            "obs_lambda": engine.target_lambda,
            "obs_ca2": engine.target_ca2,
            "avg_latency": engine.history_latency[-1] if engine.history_latency else 0.5,
            "p95_latency": engine.history_p95[-1] if engine.history_p95 else 0.5,
            "sla_target": engine.sla_target,
            "overflows": engine.total_overflows,
            "completed": engine.total_completed,
            "is_violating": (engine.history_p95[-1] if engine.history_p95 else 0.5) > engine.sla_target,
        }

    # 1. Render Pipeline Visualization
    st.html(render_pipeline_html(step_data))

    # 2. Render Metric Cards
    st.html(render_metric_cards_html(step_data))

    # 3. Render Plotly Charts
    fig1, fig2 = create_plotly_charts(engine)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True, key="chart_latency_p95")
    with col2:
        st.plotly_chart(fig2, use_container_width=True, key="chart_queue_rate")


# --- 7. MAIN CONTROLLER ---
def main():
    inject_custom_css()

    # Session State Initialization
    if "engine" not in st.session_state:
        st.session_state.engine = LiveSimulationEngine()
    if "is_running" not in st.session_state:
        st.session_state.is_running = True

    engine: LiveSimulationEngine = st.session_state.engine

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.markdown("<h2 style='color:#00f3ff; margin-bottom:0;'>🎮 Control Center</h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.85rem; color:#94a3b8;'>Live Simulation & Burst Injection</p>", unsafe_allow_html=True)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            run_toggle = st.toggle("⚡ Running", value=st.session_state.is_running, key="sim_run_state")
            st.session_state.is_running = run_toggle
        with col_s2:
            if st.button("🔄 Reset", use_container_width=True):
                engine.reset()
                st.rerun()

        st.markdown("---")
        st.markdown("<h4 style='color:#f8fafc;'>🛡️ Autoscaling Policy</h4>", unsafe_allow_html=True)
        policy_choice = st.radio(
            "Select Controller",
            options=["Variability-Aware G/G/k", "Threshold Baseline", "Fixed Capacity (k=6)"],
            index=0,
            label_visibility="collapsed",
        )
        engine.policy_name = policy_choice

        st.markdown("---")
        st.markdown("<h4 style='color:#f8fafc;'>🌊 Workload Parameters</h4>", unsafe_allow_html=True)

        # Burst Injection Button
        if st.button("🔥 INJECT SUDDEN BURST SPIKE", type="primary", use_container_width=True):
            engine.target_lambda = 12.0
            engine.target_ca2 = 6.5
            st.toast("🚨 Injected sudden burst spike: λ=12.0 req/s, Ca²=6.5!", icon="🔥")

        slider_lambda = st.slider(
            "Arrival Rate λ (req/sec)",
            min_value=0.5,
            max_value=15.0,
            value=float(engine.target_lambda),
            step=0.5,
        )
        slider_ca2 = st.slider(
            "Arrival Variability C_a²",
            min_value=0.2,
            max_value=8.0,
            value=float(engine.target_ca2),
            step=0.2,
            help="C_a² > 1.5 triggers the novel Variability Boost safety margin.",
        )
        slider_sla = st.slider(
            "SLA Latency Target (seconds)",
            min_value=1.0,
            max_value=5.0,
            value=float(engine.sla_target),
            step=0.25,
        )

        engine.target_lambda = slider_lambda
        engine.target_ca2 = slider_ca2
        engine.sla_target = slider_sla

        st.markdown("---")
        st.info("💡 **G/G/k Novelty**: When $C_a^2 > 1.5$, the autoscaler applies an 80% SLA safety margin to absorb burst clusters.")

    # --- TOP HEADER ---
    st.markdown("<div class='neon-title'>⚡ Variability-Aware G/G/k Cloud Storage Autoscaler</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='neon-sub'>Live Closed-Loop Simulation with SimPy &bull; Active Policy: <span style='color:#00f3ff; font-weight:700;'>{policy_choice}</span></div>",
        unsafe_allow_html=True,
    )

    # --- LIVE ANIMATED VIEW (FRAGMENT) ---
    live_dashboard_view(engine)


if __name__ == "__main__":
    main()
