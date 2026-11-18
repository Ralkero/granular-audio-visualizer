"""Context recognition: decisions, timing, the worker's failure handling, and the display."""

import json
from time import perf_counter, sleep

import numpy as np
import pytest

from gmv.bus import EventBus
from gmv.contract import parse_message
from gmv.engine import Engine
from gmv.evaluation.detect import estimate_intervals, run_recognition
from gmv.models import AnalysisConfig, FeatureFrame, InstrumentEstimate
from gmv.recognition import Decision, Recognizer, load_backend, register
from gmv.recognition.worker import RecognitionWorker
from gmv.sources.base import AudioBlock
from gmv.sources.file import FileAudioSource
from gmv.visualizer.state import VisualState

RATE = 16000


class LoudIsGuitar:
    """Stand-in model: the score is the level of the newest half second."""

    name, version = "loud_is_guitar", "test"
    labels = ("guitar",)
    sample_rate, context_s, hop_s = 8000, 2.0, 0.5
    thresholds = {"guitar": {"detected": 0.6, "candidate": 0.3}}

    def __init__(self, delay=0.0, fail_after=None):
        self.delay, self.fail_after, self.calls = delay, fail_after, 0
        self.lengths = []

    def infer(self, context):
        self.calls += 1
        self.lengths.append(len(context))
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("model broke")
        sleep(self.delay)
        return np.array([min(1.0, 4 * float(np.sqrt(np.mean(context[-4000:] ** 2))))])


def tone(seconds, level):
    t = np.arange(round(seconds * RATE)) / RATE
    return (level * np.sqrt(2) * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def blocks(audio, rate=RATE, size=160, start=0.0, segment=0):
    for pos in range(0, len(audio), size):
        chunk = audio[pos : pos + size]
        now = perf_counter()
        yield AudioBlock(
            chunk.astype("<f4").tobytes(),
            len(chunk),
            rate,
            1,
            start + pos / rate,
            now,
            now,
            segment,
        )


def collect(bus_queue):
    found = []
    while (message := bus_queue.get(timeout=0)) is not None:
        found.append(message)
    return found


def test_decision_separates_detected_candidate_and_absent_and_holds_through_a_dip():
    decision = Decision(detected=0.8, candidate=0.4, smooth=1)
    assert [decision.step(s) for s in (0.1, 0.5, 0.9, 0.5, 0.3, 0.5)] == [
        "absent",
        "candidate",
        "detected",
        "detected",  # still above the candidate threshold: no flicker
        "absent",
        "candidate",  # coming back needs the higher threshold again
    ]
    smooth = Decision(detected=0.8, candidate=0.4, smooth=2)
    assert [smooth.step(s) for s in (0.0, 1.0, 1.0)] == ["absent", "candidate", "detected"]
    with pytest.raises(ValueError):
        Decision(detected=0.3, candidate=0.5)


def test_estimates_follow_the_audio_timeline_and_see_only_the_past():
    audio = np.concatenate([tone(3, 0.0), tone(3, 0.25), tone(3, 0.0)])[:, None]
    backend = LoudIsGuitar()
    estimates, _ = run_recognition(audio, RATE, backend)
    assert [round(e.end, 3) for e in estimates[:3]] == [0.5, 1.0, 1.5]
    assert all(abs((e.end - e.start) - 0.5) < 1e-6 for e in estimates)
    assert set(backend.lengths) == {16000}  # always the full context at the model's rate
    spans = estimate_intervals(estimates, "guitar")
    assert len(spans) == 1
    start, end = spans[0]
    # Detected within two hops of the entrance (score averaged over two) and never before it.
    assert 3.0 <= start <= 4.0 and 6.0 <= end <= 7.0
    states = [dict(e.states)["guitar"] for e in estimates]
    assert states[0] == "absent" and "candidate" in states


def test_a_format_change_starts_the_context_again():
    recognizer = Recognizer(LoudIsGuitar())
    assert recognizer.push(tone(0.6, 0.25), RATE, 0.0)
    recognizer.estimate()
    recognizer.push(tone(0.2, 0.25), 48000, 5.0)
    assert recognizer.filled == round(0.2 * RATE) and not recognizer.due


def test_worker_publishes_estimates_with_states_and_capture_time():
    bus = EventBus()
    queue = bus.subscribe(1024)
    worker = RecognitionWorker(bus, "s", "test", None, backend=LoudIsGuitar())
    worker.start()
    for block in blocks(np.concatenate([tone(2, 0.0), tone(2, 0.25)])):
        worker.offer(block)
        sleep(0.0005)
    deadline = perf_counter() + 5
    while worker.jobs < 7 and perf_counter() < deadline:
        sleep(0.01)
    worker.close()
    messages = [m for m in collect(queue) if isinstance(m, InstrumentEstimate)]
    assert len(messages) >= 7 and all(m.available for m in messages)
    assert messages[0].backend == "loud_is_guitar" and messages[0].context_s == 2.0
    assert all(m.evidence_monotonic is not None for m in messages)
    assert dict(messages[0].states)["guitar"] == "absent"
    assert dict(messages[-1].states)["guitar"] == "detected"
    assert all(
        a.interval_end <= b.timestamp + 1e-6 for a, b in zip(messages, messages[1:], strict=False)
    )
    assert worker.state == "stopped"
    # The message survives the wire format.
    assert parse_message(json.loads(messages[-1].model_dump_json())) == messages[-1]


def test_missing_or_broken_model_reports_unavailable_and_keeps_nothing(tmp_path):
    for path, text in (
        (tmp_path / "missing.json", None),
        (tmp_path / "broken.json", "not json"),
        (tmp_path / "unknown.json", json.dumps({"kind": "no_such_backend"})),
    ):
        if text is not None:
            path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            load_backend(path)
        bus = EventBus()
        queue = bus.subscribe(64)
        worker = RecognitionWorker(bus, "s", "test", str(path))
        worker.start()
        worker.thread.join(timeout=5)
        messages = collect(queue)
        assert len(messages) == 1 and not messages[0].available and messages[0].detail
        assert worker.state.startswith("unavailable: ")
        for block in blocks(tone(0.2, 0.2)):
            worker.offer(block)
        assert len(worker.queue) == 0
        worker.close()


def test_onnx_description_with_missing_weights_is_a_readable_error(tmp_path):
    pytest.importorskip("onnxruntime")
    description = {
        "kind": "onnx_mel_tagger",
        "name": "x",
        "labels": ["guitar"],
        "sample_rate": 32000,
        "context_s": 10,
        "hop_s": 0.5,
        "thresholds": {"guitar": {"detected": 0.9, "candidate": 0.5}},
        "model": "absent.onnx",
        "frontend": "absent.npz",
    }
    path = tmp_path / "model.json"
    path.write_text(json.dumps(description), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_backend(path)


def test_inference_failure_ends_recognition_only():
    bus = EventBus()
    queue = bus.subscribe(1024)
    worker = RecognitionWorker(bus, "s", "test", None, backend=LoudIsGuitar(fail_after=2))
    worker.start()
    for block in blocks(tone(3, 0.25)):
        worker.offer(block)
        sleep(0.0005)
    worker.thread.join(timeout=5)
    messages = collect(queue)
    assert [m.available for m in messages] == [True, True, False]
    assert "model broke" in messages[-1].detail and worker.state == "unavailable: model broke"
    worker.close()


def test_a_slow_model_skips_hops_instead_of_queueing():
    bus = EventBus()
    queue = bus.subscribe(1024)
    worker = RecognitionWorker(bus, "s", "test", None, backend=LoudIsGuitar(delay=0.4))
    worker.queue.capacity = 100000  # everything is offered at once below
    worker.start()
    for block in blocks(tone(6, 0.25)):
        worker.offer(block)
    deadline = perf_counter() + 10
    while (len(worker.queue) or not worker.jobs) and perf_counter() < deadline:
        sleep(0.01)
    sleep(0.05)  # let the answer that was being computed be published
    began = perf_counter()
    worker.close()
    assert perf_counter() - began < 5
    messages = collect(queue)
    assert 1 <= len(messages) < 12 and worker.skipped_hops >= 1  # 12 hops were offered
    assert messages[-1].interval_end > 5.0  # and the last answer is about the newest audio


def test_a_saturated_queue_invalidates_the_stale_context():
    bus = EventBus()
    worker = RecognitionWorker(bus, "s", "test", None, backend=LoudIsGuitar())
    worker.queue.capacity = 8
    for block in blocks(tone(0.5, 0.25)):  # nothing is consuming: the queue overflows
        worker.offer(block)
    assert worker.queue.dropped > 0 and len(worker.queue) == 8
    assert worker._generation > 1  # audio was lost, so the context starts again


def frame(time, **extra):
    values = dict(
        session_id="s",
        source="test",
        timestamp=time,
        detected_monotonic=0,
        window_end=time + 0.02,
        sample_rate=RATE,
        channels=1,
        rms=0.1,
        peak=0.2,
        loudness_dbfs=-20,
        short_term_loudness_dbfs=-20,
        spectral_centroid_hz=1000,
        spectral_bandwidth_hz=500,
        spectral_rolloff_hz=2000,
        spectral_contrast_db=(1.0,) * 7,
        zero_crossing_rate=0.1,
        band_energy=(0.1,) * 7,
        spectral_flux=0,
        transient_strength=0,
        onset_strength=0,
        dynamic_intensity=0.5,
        silence=False,
    )
    return FeatureFrame(**(values | extra))


def estimate(start, end, state, score=0.7, **extra):
    return InstrumentEstimate(
        session_id="s",
        source="test",
        timestamp=start,
        interval_end=end,
        detected_monotonic=0,
        backend="loud_is_guitar",
        scores=(("guitar", score),),
        states=(("guitar", state),),
        **extra,
    )


def test_lane_and_notes_show_detected_candidate_absent_and_unavailable():
    state = VisualState()
    for index in range(100):  # two seconds of frames with a note candidate each
        time = index * 0.02
        state.consume(
            frame(time, pitch_candidates=((52.0, 0.8),), pitch_time=time, pitch_method="test"),
            arrival=10 + time,
        )
    board = state.board
    assert ("guitar" in board.history, board.recognition) == (False, None)
    notes = state.notes.since()[:, 0]
    assert not board.level_at("guitar", notes).any()  # nothing known: no note is attributed

    for start, end, name in ((0.0, 0.5, "absent"), (0.5, 1.0, "candidate"), (1.0, 1.5, "detected")):
        state.consume(estimate(start, end, name))
    levels = board.level_at("guitar", notes)
    assert set(levels[notes < 0.45]) == {0.0}
    assert set(levels[(notes > 0.55) & (notes < 0.95)]) == {0.5}
    assert set(levels[(notes > 1.05) & (notes < 1.45)]) == {1.0}
    assert set(levels[notes > 1.55]) == {0.0}  # newer than the newest estimate: not yet known
    assert board.revision == 3 and board.recognition["available"]
    assert dict(board.lanes())[board.known["guitar"]] is True

    # A frame detector cannot also draw the lane a recognition model owns.
    state.consume(
        frame(
            2.0,
            instrument_scores=(("guitar", 0.9),),
            instruments_active=("guitar",),
            activity_method="mlp",
        ),
        arrival=12.0,
    )
    assert board.history["guitar"].last[0] == 1.5

    state.consume(estimate(1.5, 1.5, "absent", available=False, detail="model broke"))
    assert board.recognition == {
        "backend": "loud_is_guitar",
        "available": False,
        "detail": "model broke",
    }
    assert board.history["guitar"].last[0] == 1.5  # a failure adds nothing to the lane


def test_a_gap_between_estimates_is_a_hole_in_the_lane():
    state = VisualState()
    state.consume(estimate(0.0, 0.5, "detected"))
    state.consume(estimate(9.5, 10.0, "detected"))
    rows = state.board.history["guitar"].since()
    assert rows[:, 2].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert state.board.level_at("guitar", [5.0])[0] == 0.0


def test_engine_runs_recognition_beside_analysis_and_is_unchanged_without_it(tmp_path):
    import soundfile as sf

    register("loud_test", lambda description, folder: LoudIsGuitar())
    model = tmp_path / "model.json"
    model.write_text(json.dumps({"kind": "loud_test"}), encoding="utf-8")
    path = tmp_path / "tone.wav"
    sf.write(path, np.concatenate([tone(1.5, 0.0), tone(2.5, 0.25)]), RATE, subtype="FLOAT")

    def run(config):
        engine = Engine(FileAudioSource(path, realtime=True), config, logging=False)
        queue = engine.bus.subscribe(8192)
        engine.start()
        assert engine.done.wait(30)
        return engine, collect(queue)

    engine, messages = run(AnalysisConfig(recognition_model_path=str(model)))
    estimates = [m for m in messages if isinstance(m, InstrumentEstimate)]
    metrics = engine.metrics()
    assert metrics["error"] is None and metrics["dropped_blocks"] == 0
    # An answer still being computed when the file ends is thrown away, not published late.
    assert len(estimates) >= 6
    assert metrics["recognition_jobs"] == len(estimates) + metrics["recognition_discarded_results"]
    assert dict(estimates[-1].states)["guitar"] == "detected"
    assert metrics["recognition_drops"] == 0 and metrics["recognition_status"] == "stopped"

    engine, messages = run(AnalysisConfig())
    assert not any(isinstance(m, InstrumentEstimate) for m in messages)
    assert engine.metrics()["recognition_status"] == "disabled" and engine.recognition is None

    # A model that cannot be loaded leaves the session running.
    engine, messages = run(AnalysisConfig(recognition_model_path=str(tmp_path / "nope.json")))
    estimates = [m for m in messages if isinstance(m, InstrumentEstimate)]
    assert engine.metrics()["error"] is None and engine.frame_count > 100
    assert len(estimates) == 1 and not estimates[0].available
