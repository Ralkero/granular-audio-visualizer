from threading import Event
import numpy as np
from gmv.advanced import AdvancedWorker, harmonic_percussive_balance
from gmv.bus import EventBus
from gmv.sources.base import AudioBlock


def test_harmonic_proxy_and_silence():
    x = np.sin(2*np.pi*440*np.arange(44100)/22050).astype(np.float32)
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
        worker.offer(AudioBlock(data, 4800, 48000, 1, 7+i/10, 0, 0))
    assert entered.wait(2)
    for i in range(200):
        worker.offer(AudioBlock(data, 4800, 48000, 1, 9+i/10, 0, 0))
    assert worker.queue.dropped > 0 and len(worker.queue) <= 64
    release.set()
    worker.close()
    event = subscription.get()
    assert event.timestamp == 7 and abs(event.interval_end - 9) < 1e-6
    assert event.analysis_tier == "advanced"
