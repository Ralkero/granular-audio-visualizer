from threading import Event
import numpy as np
from gmv.advanced import AdvancedWorker, harmonic_percussive_balance
from gmv.bus import EventBus
from gmv.sources.base import AudioBlock


def test_harmonic_proxy_and_silence():
    x = np.sin(2 * np.pi * 440 * np.arange(44100) / 22050).astype(np.float32)
    values = harmonic_percussive_balance(np.column_stack((x, -x)), 22050)
    assert values["harmonic_power_fraction"] > 0.8
    assert values["rms"] > 0.6
    silent = harmonic_percussive_balance(np.zeros((44100, 1)), 22050)
    assert silent["percussive_power_fraction"] == 0


def test_slow_background_has_bounded_queue_and_original_time():
    bus = EventBus()
    subscription = bus.subscribe()
    entered, release = Event(), Event()

    def slow(audio, rate):
        entered.set()
        release.wait(3)
        return {"percussive_power_fraction": 0.2}

    worker = AdvancedWorker(bus, "test", "synthetic", slow)
    worker.start()
    data = np.zeros((4800, 1), dtype=np.float32).tobytes()
    for i in range(20):
        worker.offer(AudioBlock(data, 4800, 48000, 1, 7 + i / 10, 0, 0))
    assert entered.wait(2)
    for i in range(200):
        worker.offer(AudioBlock(data, 4800, 48000, 1, 9 + i / 10, 0, 0))
    assert worker.queue.dropped > 0 and len(worker.queue) <= 64
    release.set()
    worker.close()
    assert subscription.get(timeout=0) is None
    assert worker.discarded_results >= 1


def test_segment_change_discards_inflight_result_and_new_interval_keeps_time():
    bus = EventBus()
    queue = bus.subscribe()
    entered, release = Event(), Event()
    calls = []

    def delayed(audio, rate):
        calls.append(rate)
        if len(calls) == 1:
            entered.set()
            release.wait(3)
        return {"percussive_power_fraction": 0.2}

    worker = AdvancedWorker(bus, "s", "test", delayed)
    worker.start()
    data = np.zeros((96000, 1), dtype=np.float32).tobytes()
    worker.offer(AudioBlock(data, 96000, 48000, 1, 7, 10, 10, 1))
    assert entered.wait(2)
    worker.offer(AudioBlock(data, 96000, 48000, 1, 20, 23, 23, 2, True))
    release.set()
    event = queue.get(timeout=3)
    worker.close()
    assert event.timestamp == 20 and event.interval_end == 22 and event.segment == 2
    assert event.evidence_monotonic == 23 and event.intended_use == "context"
    assert event.provenance == "harmonic_percussive_proxy"
    assert worker.discarded_results == 1 and not worker.thread.is_alive()
