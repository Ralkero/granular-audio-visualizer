import json
from threading import Event

import numpy as np
import soundfile as sf

from gmv.bus import EventBus
from gmv.engine import Engine
from gmv.session import replay
from gmv.sources.file import FileAudioSource


def test_complete_file_pipeline_and_replay(tmp_path):
    path = tmp_path / "tone.wav"
    sf.write(path, 0.3 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000), 48000)
    engine = Engine(FileAudioSource(path, realtime=False), session_root=tmp_path)
    subscriber = engine.bus.subscribe()
    engine.start()
    assert engine.done.wait(10)
    assert not engine.error and engine.frame_count >= 47 and engine.audio.dropped == 0
    folder = engine.writer.path
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["complete"] and not manifest["raw_audio_saved"]
    assert (folder / "realtime_features.csv").exists()
    bus = EventBus()
    q = bus.subscribe(256)
    replay(folder / "messages.jsonl", bus, Event(), speed=1000)
    originals, copies = [], []
    while len(subscriber):
        originals.append(subscriber.get())
    while len(q):
        copies.append(q.get())
    assert [m.timestamp for m in originals] == [m.timestamp for m in copies]
    assert not list(folder.rglob("*.wav"))


def test_capture_failure_finishes_and_logs(tmp_path):
    engine = Engine(FileAudioSource(tmp_path / "missing.wav"), session_root=tmp_path)
    engine.start()
    assert engine.done.wait(5) and engine.error
    assert (engine.writer.path / "performance.json").exists()
