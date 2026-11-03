from threading import Event
import numpy as np
import soundfile as sf
from gmv.buffer import BoundedQueue
from gmv.sources.file import FileAudioSource
from gmv.sources.windows import CaptureClock


def test_file_timestamps_and_final_partial_block(tmp_path):
    path = tmp_path / "known.wav"
    sf.write(path, np.zeros((1001, 2)), 44100, subtype="FLOAT")
    q = BoundedQueue(8)
    FileAudioSource(path, realtime=False).run(q, Event(), lambda *_: None)
    blocks = [q.get(), q.get(), q.get()]
    assert sum(b.frames for b in blocks) == 1001
    assert blocks[1].timestamp == 441 / 44100
    assert q.closed and q.dropped == 0


def test_capture_clock_gap_does_not_compress_silence():
    clock = CaptureClock(100)
    first = clock.stamp(480, 48000, {}, 100.01)
    second = clock.stamp(480, 48000, {}, 100.02)
    resumed = clock.stamp(480, 48000, {}, 105)
    assert abs(second[0] - first[0] - 0.01) < 1e-9
    assert resumed[0] > 4.9 and resumed[3] and resumed[2] > second[2]


def test_capture_clock_uses_adc_and_flags():
    clock = CaptureClock(100)
    a = clock.stamp(480, 48000, {"input_buffer_adc_time": 1, "current_time": 1.02}, 101)
    b = clock.stamp(480, 48000, {"input_buffer_adc_time": 1.1, "current_time": 1.12}, 101.1)
    assert abs(a[0] - 0.98) < 1e-9 and b[3]
