"""Processing Queue Q(t) representation.

Models the dispatch queue Q(t) holding tasks waiting for or being assigned
to active worker nodes (VMs) in the storage service pool.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional


@dataclass
class ProcessingQueueStats:
    """Snapshot statistics of the processing queue Q(t)."""

    backlog: int          # Current queue length Q(t)
    capacity: int         # Maximum queue capacity
    total_enqueued: int   # Cumulative tasks enqueued
    total_dequeued: int   # Cumulative tasks dequeued
    peak_backlog: int     # Max Q(t) observed


class ProcessingQueue:
    """Processing queue Q(t) with FIFO dispatch semantics."""

    def __init__(self, capacity: int = 10000):
        if capacity <= 0:
            raise ValueError("ProcessingQueue capacity must be strictly positive.")
        self.capacity: int = int(capacity)
        self._items: Deque[Any] = deque()
        self.total_enqueued: int = 0
        self.total_dequeued: int = 0
        self.peak_backlog: int = 0

    @property
    def backlog(self) -> int:
        """Current backlog Q(t)."""
        return len(self._items)

    def is_full(self) -> bool:
        """Check if queue has reached maximum capacity."""
        return len(self._items) >= self.capacity

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._items) == 0

    def enqueue(self, item: Any) -> bool:
        """Enqueue task into Q(t). Returns False if queue capacity exceeded."""
        if len(self._items) >= self.capacity:
            return False

        self._items.append(item)
        self.total_enqueued += 1
        if len(self._items) > self.peak_backlog:
            self.peak_backlog = len(self._items)
        return True

    def dequeue(self) -> Optional[Any]:
        """Dequeue next available task from Q(t)."""
        if not self._items:
            return None
        self.total_dequeued += 1
        return self._items.popleft()

    def peek(self) -> Optional[Any]:
        """Inspect the head task in the queue."""
        if not self._items:
            return None
        return self._items[0]

    def get_stats(self) -> ProcessingQueueStats:
        """Retrieve current queue metrics."""
        return ProcessingQueueStats(
            backlog=self.backlog,
            capacity=self.capacity,
            total_enqueued=self.total_enqueued,
            total_dequeued=self.total_dequeued,
            peak_backlog=self.peak_backlog,
        )

    def __len__(self) -> int:
        return self.backlog
