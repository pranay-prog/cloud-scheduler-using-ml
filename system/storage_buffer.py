"""Storage Buffer B(t) representation.

Models an ingress absorption storage buffer before tasks are transferred
to active compute/storage worker processing queues. Enforces capacity limits
and tracks rejection/overflow metrics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional


@dataclass
class StorageBufferStats:
    """Snapshot statistics of the storage buffer B(t)."""

    backlog: int              # B(t) current number of buffered items
    capacity: int             # Maximum storage buffer capacity
    total_arrivals: int       # Total items that attempted entry
    admitted: int             # Total items accepted
    rejected: int             # Total items dropped due to buffer overflow
    rejection_rate: float     # rejected / total_arrivals


class StorageBuffer:
    """Storage buffer B(t) with finite or infinite capacity."""

    def __init__(self, capacity: int = 1000):
        if capacity <= 0:
            raise ValueError("StorageBuffer capacity must be strictly positive.")
        self.capacity: int = int(capacity)
        self._items: Deque[Any] = deque()
        self.total_arrivals: int = 0
        self.admitted_count: int = 0
        self.rejected_count: int = 0

    @property
    def backlog(self) -> int:
        """Current backlog B(t)."""
        return len(self._items)

    def is_full(self) -> bool:
        """Check if buffer has reached capacity."""
        return len(self._items) >= self.capacity

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._items) == 0

    def push(self, item: Any) -> bool:
        """Attempt to add an item to the buffer.

        Returns True if admitted, False if rejected (buffer overflow).
        """
        self.total_arrivals += 1
        if len(self._items) >= self.capacity:
            self.rejected_count += 1
            return False

        self._items.append(item)
        self.admitted_count += 1
        return True

    def pop(self) -> Optional[Any]:
        """Remove and return the oldest item from the buffer, or None if empty."""
        if not self._items:
            return None
        return self._items.popleft()

    def peek(self) -> Optional[Any]:
        """View the oldest item without removing it."""
        if not self._items:
            return None
        return self._items[0]

    def get_stats(self) -> StorageBufferStats:
        """Retrieve current buffer state and historical overflow metrics."""
        rej_rate = (
            float(self.rejected_count / self.total_arrivals)
            if self.total_arrivals > 0
            else 0.0
        )
        return StorageBufferStats(
            backlog=self.backlog,
            capacity=self.capacity,
            total_arrivals=self.total_arrivals,
            admitted=self.admitted_count,
            rejected=self.rejected_count,
            rejection_rate=rej_rate,
        )

    def __len__(self) -> int:
        return self.backlog
