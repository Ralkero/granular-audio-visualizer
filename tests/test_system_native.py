import io
from threading import Event

import numpy as np
import pytest

from gmv.buffer import BoundedQueue
from gmv.clock_analysis import analyze_timeline
from gmv.sources import application, system_native, windows
from gmv.sources.system_native import NativeSystemAudioSource, QpcSampleClock

RATE, FRAMES, BLOCK = 48000, 480, 0.01


def packet(sequence, qpc_seconds, flags=0, rate=RATE, channels=8, frames=FRAMES):
    data = bytes(frames * channels * 4)
    header = application.HEADER.pack(
        b"GMV1", len(data), rate, channels, sequence, round(qpc_seconds * 1e7), flags
    )
    return header + data


def test_qpc_jitter_keeps_one_contiguous_run():
    clock, rng = QpcSampleClock(100), np.random.default_rng(3)
    stamps = []
    for i in range(6000):
        qpc = 100.5 + i * BLOCK + rng.uniform(-0.001, 0.001)
        stamps.append(clock.stamp(FRAMES, RATE, qpc, qpc + 0.02))
    times = np.array([s[0] for s in stamps])
    assert [s[2] for s in stamps].count(True) == 1 and stamps[0][2]
    assert np.allclose(np.diff(times), BLOCK, atol=1e-9)
    assert clock.reanchors == 0 and clock.max_abs_error_ms < 2.1
    assert {s[3] for s in stamps} == {"wasapi_qpc_sample_clock"}


def test_lost_audio_is_a_discontinuity_and_does_not_drift():
    clock = QpcSampleClock(100)
    clock.stamp(FRAMES, RATE, 101.0, 101.0)
    clock.stamp(FRAMES, RATE, 101.01, 101.01)
    # Three packets (30 ms) vanish without any flag; QPC alone reveals it.
    resumed = clock.stamp(FRAMES, RATE, 101.05, 101.05)
    following = clock.stamp(FRAMES, RATE, 101.06, 101.06)
    assert resumed[2] and resumed[0] == pytest.approx(1.05)
    assert not following[2] and following[0] == pytest.approx(1.06)
    assert clock.reanchors == 1 and abs(clock.error_ms) < 1e-6


def test_slow_clock_skew_stays_signed_then_reanchors():
    clock, gaps = QpcSampleClock(100), []
    for i in range(30000):  # 300 s with the device clock 50 ppm slow against QPC
        qpc = 101 + i * BLOCK * (1 + 50e-6)
        gaps.append(clock.stamp(FRAMES, RATE, qpc, qpc)[2])
        if i == 10000:
            assert clock.error_ms == pytest.approx(5.0, abs=0.01)
    assert gaps.count(True) == 2 and clock.reanchors == 1
    assert clock.max_abs_error_ms == pytest.approx(10, abs=0.1)


def test_timeline_never_moves_backwards_and_invalid_qpc_falls_back():
    clock = QpcSampleClock(100)
    clock.stamp(FRAMES, RATE, 101.0, 101.0)
    earlier = clock.stamp(FRAMES, RATE, 100.9, 101.01, forced=True)
    assert earlier[2] and earlier[0] == pytest.approx(1.01)
    assert clock.error_ms == pytest.approx(-110)
    fallback = clock.stamp(FRAMES, RATE, None, 105.0)
    assert fallback[3] == "pipe_arrival_estimate" and fallback[0] == pytest.approx(4.99)


def test_reader_marks_drops_flags_and_format(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(system_native, "perf_counter", lambda: now[0])
    stream = io.BytesIO(
        packet(1, 1.00, flags=1)
        + packet(2, 1.01)
        + packet(4, 1.03)  # sequence 3 dropped by the helper's bounded channel
        + packet(5, 1.04, flags=1)  # WASAPI data discontinuity
        + packet(6, 1.05, flags=4)  # timestamp error: QPC not trusted
    )
    source, output, errors = NativeSystemAudioSource(), BoundedQueue(16), []
    source.settle_packets = 0
    now[0] = 1.06  # so the arrival fallback for the last packet continues the run
    source._read_audio(stream, 1, QpcSampleClock(0), output, Event(), errors, [0.0])
    blocks = [output.get() for _ in range(5)]
    assert not errors and source.native_drops == 1 and source.device_discontinuities == 1
    assert [b.discontinuity for b in blocks] == [True, False, True, True, False]
    assert [b.channels for b in blocks] == [8] * 5 and source.format == [RATE, 8]
    assert blocks[2].timestamp == pytest.approx(1.03)
    assert blocks[4].timestamp_quality == "pipe_arrival_estimate"


def test_settle_measures_lead_so_evidence_is_never_in_the_future(monkeypatch):
    # QPC runs 21 ms ahead of delivery; the first primed packet arrives 10 ms earlier still.
    arrivals = [0.969] + [0.98 + i * BLOCK for i in range(1, 40)]
    clock_reads = iter(arrivals)
    monkeypatch.setattr(system_native, "perf_counter", lambda: next(clock_reads))
    stream = io.BytesIO(b"".join(packet(i + 1, 1.0 + i * BLOCK) for i in range(40)))
    source, output, errors = NativeSystemAudioSource(), BoundedQueue(64), []
    source._read_audio(stream, 1, QpcSampleClock(0), output, Event(), errors, [0.0])
    blocks = [output.get() for _ in range(20)]
    assert not errors and output.get(timeout=0) is None
    assert source.settle_discarded_frames == 20 * FRAMES
    assert source.qpc_lead == pytest.approx(0.020 + BLOCK + 0.003)
    assert [b.discontinuity for b in blocks] == [True] + [False] * 19
    for block in blocks:
        end = block.capture_monotonic + block.frames / block.sample_rate
        assert end <= block.received_monotonic
        assert block.received_monotonic - end == pytest.approx(0.003)


def test_packet_format_validation():
    assert application.read_packet(io.BytesIO(packet(1, 1.0)), fixed_format=False)[2] == 8
    with pytest.raises(ValueError):
        application.read_packet(io.BytesIO(packet(1, 1.0)))  # process path stays stereo-only
    with pytest.raises(ValueError):
        application.read_packet(io.BytesIO(packet(1, 1.0, rate=1000)), fixed_format=False)


def test_backend_selection_falls_back_without_helper(monkeypatch, tmp_path):
    monkeypatch.setattr(application, "HELPER", tmp_path / "missing.exe")
    assert isinstance(system_native.system_audio_source(), windows.SystemAudioSource)
    with pytest.raises(RuntimeError, match="Build Process Capture"):
        system_native.system_audio_source(backend="native")
    helper = tmp_path / "helper.exe"
    helper.write_bytes(b"")
    monkeypatch.setattr(application, "HELPER", helper)
    assert isinstance(system_native.system_audio_source("X"), NativeSystemAudioSource)
    assert isinstance(
        system_native.system_audio_source(backend="portaudio"), windows.SystemAudioSource
    )


def test_timeline_analysis_reports_signed_offset_and_drift():
    arrival = np.arange(6000) * BLOCK + 1
    steady = analyze_timeline(arrival, arrival + 0.02)
    assert steady["start_ms"] == pytest.approx(-20) and abs(steady["drift_ms"]) < 1e-6
    drifting = analyze_timeline(arrival, arrival - np.arange(6000) * 1e-5)
    assert drifting["drift_ms"] == pytest.approx(50, abs=2)
    assert drifting["slope_ms_per_min"] == pytest.approx(60, rel=0.02)
