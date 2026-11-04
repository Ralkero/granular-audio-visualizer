"""Fault injection verifies recovery policy; does not claim USB hotplug validation."""

import sys
from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest
from gmv.buffer import BoundedQueue
from gmv.engine import Engine
from gmv.models import AnalysisConfig
from gmv.sources.base import AudioBlock, AudioSource
from gmv.sources.windows import SystemAudioSource


def test_default_change_closes_and_reopens_capture(monkeypatch):
    import gmv.sources.windows as windows

    calls, statuses = [], []

    class FastStop:
        ended = False

        def is_set(self):
            return self.ended

        def wait(self, seconds):
            return self.ended

    stop = FastStop()

    class Stream:
        def close(self):
            calls.append("close")

        def is_active(self):
            return True

    class Audio:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def get_default_wasapi_loopback(self):
            return {"name": "fake", "index": 0, "defaultSampleRate": 48000, "maxInputChannels": 2}

        def open(self, **kwargs):
            calls.append("open")
            if calls.count("open") == 2:
                stop.ended = True
            return Stream()

    states = iter([("a", ("a",)), ("b", ("b",)), ("b", ("b",))])
    monkeypatch.setattr(windows.sys, "platform", "win32")
    monkeypatch.setattr(windows, "endpoint_snapshot", lambda: next(states))
    monkeypatch.setitem(
        sys.modules,
        "comtypes",
        SimpleNamespace(CoInitialize=lambda: None, CoUninitialize=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", SimpleNamespace(PyAudio=Audio, paFloat32=1))
    queue = BoundedQueue()
    source = SystemAudioSource()
    source.run(queue, stop, lambda *args: statuses.append(args))
    assert calls == ["open", "close", "open", "close"]
    assert source.recoveries == 1 and queue.closed
    assert any(state == "recovering" for state, _ in statuses)


def test_engine_recovers_from_gap_and_format_change():
    class ChangingSource(AudioSource):
        kind, live = "synthetic", False

        def run(self, output, stop, status):
            try:
                for segment, rate, channels, origin in [(1, 48000, 2, 0), (2, 44100, 1, 3)]:
                    for pos in range(0, rate // 2, 441):
                        x = np.zeros((441, channels), dtype=np.float32)
                        output.put(
                            AudioBlock(
                                x.tobytes(), 441, rate, channels, origin + pos / rate, 0, 0, segment
                            ),
                            wait=True,
                            stop=stop,
                        )
            finally:
                output.close()

    engine = Engine(ChangingSource(), logging=False)
    q = engine.bus.subscribe()
    engine.start()
    assert engine.done.wait(5) and not engine.error
    frames = []
    while len(q):
        message = q.get()
        if message.type == "feature_frame":
            frames.append(message)
    assert engine.discontinuities == 1
    assert {f.sample_rate for f in frames} == {48000, 44100}
    assert all(f.timestamp < 0.6 or f.timestamp >= 3 for f in frames)


def test_start_failure_does_not_leave_engine_running(tmp_path):
    class Unused(AudioSource):
        def run(self, output, stop, status):
            raise AssertionError("Must not capture without valid configuration")

    engine = Engine(Unused(), AnalysisConfig(debug_audio_seconds=1), logging=False)
    with pytest.raises(ValueError):
        engine.start()
    assert engine.done.is_set()


def test_file_backpressure_can_be_stopped():
    from threading import Thread

    stop, queue = Event(), BoundedQueue(1)
    queue.put(1)
    results = []
    worker = Thread(target=lambda: results.append(queue.put(2, wait=True, stop=stop)))
    worker.start()
    stop.set()
    worker.join(1)
    assert not worker.is_alive() and results == [False]
