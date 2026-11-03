"""Bounded queues; live producers discard oldest data rather than waiting on consumers."""
from collections import deque
from threading import Condition


class BoundedQueue:
    def __init__(self, capacity=16):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items = deque()
        self._condition = Condition()
        self.closed = False
        self.dropped = 0
        self.dropped_frames = 0
        self.high_water = 0

    def put(self, item, *, wait=False, stop=None):
        with self._condition:
            while wait and len(self._items) >= self.capacity and not self.closed:
                if stop is not None and stop.is_set():
                    return False
                self._condition.wait(0.05)
            if self.closed:
                return False
            if len(self._items) >= self.capacity:
                old = self._items.popleft()
                self.dropped += 1
                self.dropped_frames += getattr(old, "frames", 0)
            self._items.append(item)
            self.high_water = max(self.high_water, len(self._items))
            self._condition.notify_all()
            return True

    def get(self, timeout=0.1):
        with self._condition:
            self._condition.wait_for(lambda: self._items or self.closed, timeout)
            if not self._items:
                return None
            item = self._items.popleft()
            self._condition.notify_all()
            return item

    def close(self):
        with self._condition:
            self.closed = True
            self._condition.notify_all()

    def __len__(self):
        with self._condition:
            return len(self._items)
