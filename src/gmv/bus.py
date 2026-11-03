"""In-process pub/sub. Subscribers receive immutable models in independent bounded queues."""
from threading import Lock
from time import perf_counter

from .buffer import BoundedQueue


class EventBus:
    def __init__(self):
        self._lock = Lock()
        self._subscribers = []
        self._sequence = 0

    def subscribe(self, capacity=256):
        queue = BoundedQueue(capacity)
        with self._lock:
            self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue):
        with self._lock:
            if queue in self._subscribers:
                self._subscribers.remove(queue)
        queue.close()

    def publish(self, message):
        with self._lock:
            self._sequence += 1
            message = message.model_copy(update={
                "published_monotonic": perf_counter(), "sequence": self._sequence})
            for subscriber in self._subscribers:
                subscriber.put(message)
        return message
