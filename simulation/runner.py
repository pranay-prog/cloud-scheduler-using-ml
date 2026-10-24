"""SimPy discrete-event simulation runner.

Coordinates:
- Ingress StorageBuffer B(t) with finite capacity and overflow tracking
- Dispatch ProcessingQueue Q(t)
- Dynamic worker pool with 10s boot delay (T_scale) on scale-up
- Periodic autoscaling control loop (Fixed, Threshold, Variability-Aware)
- Optional deterministic task dispatch policy; FIFO remains the default
"""

from __future__ import annotations

from typing import Any, List, Optional
import simpy

from system.storage_buffer import StorageBuffer
from system.processing_queue import ProcessingQueue
from simulation.metrics import SimulationMetrics, SimulationResult
from workload.generator import RequestItem, WorkloadTrace
from workload.characterizer import compute_arrival_metrics, compute_service_metrics
from simulation.scheduling import SchedulingPolicy, SCHEDULERS


class SimulationRunner:
    """SimPy discrete-event simulation engine for cloud storage queueing."""

    def __init__(
        self,
        trace: WorkloadTrace,
        autoscaler: Any,
        policy_name: str,
        initial_k: int = 4,
        boot_delay: float = 10.0,
        buffer_capacity: int = 600,
        queue_capacity: int = 5000,
        control_interval: float = 4.0,
        sla_target: float = 2.0,
        rf_model: Optional[Any] = None,
        use_rf_cs2: bool = False,
        scheduling_policy: Optional[SchedulingPolicy] = None,
        upper_residual_quantile: float = 0.0,
    ):
        self.trace: WorkloadTrace = trace
        self.autoscaler: Any = autoscaler
        self.policy_name: str = policy_name
        self.initial_k: int = int(initial_k)
        self.boot_delay: float = float(boot_delay)
        self.control_interval: float = float(control_interval)
        self.sla_target: float = float(sla_target)
        self.rf_model: Optional[Any] = rf_model
        self.use_rf_cs2: bool = use_rf_cs2
        self.scheduling_policy = scheduling_policy or SCHEDULERS["FCFS"]
        self.upper_residual_quantile = max(0.0, float(upper_residual_quantile))

        # State storage
        self.storage_buffer: StorageBuffer = StorageBuffer(capacity=buffer_capacity)
        self.processing_queue: ProcessingQueue = ProcessingQueue(capacity=queue_capacity)
        self.metrics: SimulationMetrics = SimulationMetrics(sla_target=sla_target)

        # SimPy primitives
        self.env: simpy.Environment = simpy.Environment()
        self.task_available_event: simpy.Event = self.env.event()
        self.simulation_done_event: simpy.Event = self.env.event()
        self._simulation_finished: bool = False

        # Dynamic capacity tracking
        self.active_workers: int = 0
        self.booting_workers: int = 0
        self.desired_k: int = self.initial_k
        self._worker_id_counter: int = 0

        # Recent trace observation buffer for online parameter estimation
        self._recent_arrivals: List[float] = []
        self._recent_services: List[float] = []
        self._recent_rf_preds: List[float] = []

    def _trigger_task_event(self) -> None:
        """Signal waiting workers that work is available in the queue."""
        if not self.task_available_event.triggered:
            self.task_available_event.succeed()
            self.task_available_event = self.env.event()

    def _check_all_finished(self) -> None:
        """Check if all incoming requests have completed or been rejected."""
        finished = self.metrics.buffer_overflows + len(self.metrics.latencies)
        if finished >= len(self.trace.requests):
            self._simulation_finished = True
            if not self.simulation_done_event.triggered:
                self.simulation_done_event.succeed()

    def _worker_process(self, worker_id: int, is_initial: bool = False):
        """Worker lifecycle: boots up, processes tasks from Q(t), shuts down on scale-down."""
        if not is_initial and self.boot_delay > 0:
            yield self.env.timeout(self.boot_delay)

        self.booting_workers = max(0, self.booting_workers - 1)
        self.active_workers += 1
        self.metrics.record_capacity_change(self.active_workers, self.env.now)

        while not self._simulation_finished:
            # Check if this worker should be decommissioned (scale-down)
            if self.active_workers > self.desired_k:
                self.active_workers -= 1
                self.metrics.record_capacity_change(self.active_workers, self.env.now)
                return

            # Pull next item from processing queue
            queued = list(self.processing_queue._items)
            estimated_work = sum(
                max(0.1, float(getattr(t, "upper_service", None) or getattr(t, "predicted_service", None) or 0.1))
                for t in queued
            )
            estimated_wait = estimated_work / max(1, self.active_workers)
            item: Optional[RequestItem] = self.processing_queue.dequeue_by(
                lambda items: self.scheduling_policy.choose(items, self.env.now, estimated_wait)
            )
            self.metrics.record_queue_length(self.processing_queue.backlog, self.env.now)

            # Transfer pending item from storage buffer into processing queue
            if not self.storage_buffer.is_empty():
                buffered_item = self.storage_buffer.pop()
                if buffered_item is not None:
                    self.processing_queue.enqueue(buffered_item)
                    self.metrics.record_queue_length(self.processing_queue.backlog, self.env.now)
                    self._trigger_task_event()

            if item is None:
                # Wait for next task event or 1.0s timeout
                wait_evt = self.task_available_event | self.env.timeout(1.0)
                yield wait_evt
                continue

            # Process the task
            wait_time = self.env.now - item.arrival_timestamp
            yield self.env.timeout(item.service_time)
            total_latency = self.env.now - item.arrival_timestamp
            self.metrics.record_busy_time(item.service_time)
            self.metrics.record_deadline_completion(item.deadline, self.env.now)

            self.metrics.record_completion(wait_time=wait_time, total_latency=total_latency)
            self._check_all_finished()

    def _spawn_worker(self) -> None:
        """Instantiate a new booting worker."""
        self._worker_id_counter += 1
        self.booting_workers += 1
        self.env.process(self._worker_process(self._worker_id_counter, is_initial=False))

    def _arrival_process(self):
        """Simulate request arrivals according to trace inter-arrivals."""
        for req in self.trace.requests:
            if req.inter_arrival > 0:
                yield self.env.timeout(req.inter_arrival)

            self.metrics.record_arrival()
            req.arrival_timestamp = self.env.now
            if req.deadline_after_arrival is not None:
                req.deadline = req.arrival_timestamp + req.deadline_after_arrival

            # Track rolling observation window
            self._recent_arrivals.append(req.inter_arrival)
            self._recent_services.append(req.service_time)
            if self.rf_model is not None:
                feat = [[req.cpu_req, req.ram_req, req.workload_type]]
                try:
                    pred_s = float(self.rf_model.predict(feat)[0])
                    if not (pred_s > 0 and pred_s < float("inf")):
                        pred_s = max(0.1, float(req.service_time))
                except (ValueError, TypeError, FloatingPointError):
                    pred_s = 0.1
                req.predicted_service = pred_s
                req.upper_service = max(0.0, pred_s) + self.upper_residual_quantile
                self._recent_rf_preds.append(pred_s)

            if len(self._recent_arrivals) > 250:
                self._recent_arrivals.pop(0)
                self._recent_services.pop(0)
                if self._recent_rf_preds:
                    self._recent_rf_preds.pop(0)

            # Ingress via StorageBuffer B(t)
            admitted = self.storage_buffer.push(req)
            if not admitted:
                self.metrics.record_buffer_overflow()
                self._check_all_finished()
                continue

            # Transfer to ProcessingQueue Q(t) if space permits
            if not self.processing_queue.is_full():
                buffered_item = self.storage_buffer.pop()
                if buffered_item is not None:
                    self.processing_queue.enqueue(buffered_item)
                    self.metrics.record_queue_length(self.processing_queue.backlog, self.env.now)
                    self._trigger_task_event()

    def _control_loop_process(self):
        """Periodic autoscaling decision loop."""
        while not self._simulation_finished:
            yield self.env.timeout(self.control_interval)

            if len(self._recent_arrivals) >= 10:
                lam, _, _, ca2 = compute_arrival_metrics(self._recent_arrivals)
            else:
                lam, ca2 = 4.0, 1.0

            if self.use_rf_cs2 and len(self._recent_rf_preds) >= 10:
                mu, _, _, cs2 = compute_service_metrics(self._recent_rf_preds)
            elif len(self._recent_services) >= 10:
                mu, _, _, cs2 = compute_service_metrics(self._recent_services)
            else:
                mu, cs2 = 1.0, 1.0

            stats = {
                "q_len": self.processing_queue.backlog,
                "b_len": self.storage_buffer.backlog,
                "arrival_rate": lam,
                "service_rate": mu,
                "ca2": ca2,
                "cs2": cs2,
            }

            target_k = self.autoscaler.decide(stats, current_k=self.desired_k)
            self.desired_k = target_k

            current_total_provisioned = self.active_workers + self.booting_workers
            if target_k > current_total_provisioned:
                diff = target_k - current_total_provisioned
                for _ in range(diff):
                    self._spawn_worker()

    def run(self) -> SimulationResult:
        """Execute discrete-event simulation until all trace requests are completed."""
        self.metrics.record_initial_capacity(self.initial_k, current_time=0.0)

        # Initialize base worker pool without boot delay at t=0
        for _ in range(self.initial_k):
            self._worker_id_counter += 1
            self.env.process(self._worker_process(self._worker_id_counter, is_initial=True))

        # Start arrival generator and autoscaler control loop
        self.env.process(self._arrival_process())
        self.env.process(self._control_loop_process())

        # Run until all requests finish or 10,000s simulation timeout
        max_timeout = self.env.timeout(10000.0)
        self.env.run(until=self.env.any_of([self.simulation_done_event, max_timeout]))

        return self.metrics.finalize(
            end_time=self.env.now,
            policy_name=self.policy_name,
            trace_name=self.trace.name,
        )
