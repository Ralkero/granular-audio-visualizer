from pathlib import Path
from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest
from gmv.engine import Engine
from gmv.models import AnalysisConfig
from gmv.sources.base import AudioSource, AudioBlock


class SmallSource(AudioSource):
    kind, live = "synthetic", False

    def run(self, output, stop, status):
        for i in range(50):
            data = np.zeros((480, 1), dtype=np.float32)
            output.put(
                AudioBlock(data.tobytes(), 480, 48000, 1, i / 100, 0, 0), wait=True, stop=stop
            )
        output.close()


def test_logger_write_failure_preserves_analysis_and_shutdown(tmp_path, monkeypatch):
    real_open = Path.open

    def fail(path, *args, **kwargs):
        if path.name == "messages.jsonl":
            raise OSError("injected write failure; no real disk pressure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail)
    engine = Engine(SmallSource(), session_root=tmp_path)
    engine.start()
    assert engine.done.wait(5)
    assert engine.frame_count > 20 and engine.writer.error
    assert not engine.writer.thread.is_alive()


def test_start_cleanup_failure_still_finishes(tmp_path, monkeypatch):
    import gmv.recording as recording
    import gmv.session as session

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    monkeypatch.setattr(recording, "DebugRecorder", fail)
    original_close = session.SessionWriter.close

    def failing_close(self, *args):
        original_close(self, *args)
        raise OSError("injected finalization failure")

    monkeypatch.setattr(session.SessionWriter, "close", failing_close)
    engine = Engine(SmallSource(), AnalysisConfig(debug_audio_seconds=1), session_root=tmp_path)
    with pytest.raises(OSError):
        engine.start()
    assert engine.done.is_set() and not engine.writer.thread.is_alive()


def test_ambiguous_restart_is_not_selected(monkeypatch):
    from gmv.sources import application

    source = object.__new__(application.ApplicationAudioSource)
    source.exe, source.pid, source.restarts = "test.exe", 99, 0
    processes = [SimpleNamespace(pid=n, info={"exe": "test.exe"}) for n in (10, 11)]
    monkeypatch.setattr(application.psutil, "process_iter", lambda *_: processes)
    monkeypatch.setattr(application, "root_process", lambda p: p)
    assert not source._find_restart() and source.pid == 99 and source.restarts == 0


def test_helper_launch_failure_is_explicit_and_closes_queue(monkeypatch):
    from gmv.sources import application
    from gmv.buffer import BoundedQueue

    source = object.__new__(application.ApplicationAudioSource)
    source.pid = 123
    monkeypatch.setattr(source, "_alive", lambda: True)

    def fail(*args, **kwargs):
        raise OSError("helper unavailable")

    monkeypatch.setattr(application.subprocess, "Popen", fail)
    output = BoundedQueue()
    with pytest.raises(OSError, match="helper unavailable"):
        source.run(output, Event(), lambda *_: None)
    assert output.closed


def test_repeated_lifecycle_has_no_live_workers():
    for _ in range(5):
        engine = Engine(SmallSource(), logging=False)
        engine.start()
        assert engine.done.wait(5) and not engine.error
        assert not any(t.is_alive() for t in engine.threads)
