"""Note candidates: the tracker itself, its engine output and how the visualizer holds it."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from test_activity import weights
from test_visualizer import event, feed, frame

from gmv.activity import ActivityBank, ActivityModel
from gmv.evaluation.detect import run_detector
from gmv.models import AnalysisConfig
from gmv.pitch import PitchTracker, midi_to_hz, note_name
from gmv.visualizer.state import VisualState

RATE, HOP = 48000, 960


def pluck(midi, seconds=1.0, harmonics=8, decay=3.0):
    """A plucked-string stand-in: harmonics of one note, decaying."""
    time = np.arange(round(seconds * RATE)) / RATE
    tone = sum(np.sin(2 * np.pi * h * midi_to_hz(midi) * time) / h for h in range(1, harmonics + 1))
    return 0.2 * tone * np.exp(-decay * time)


def run(audio):
    tracker = PitchTracker(RATE, HOP / RATE)
    frames = []
    for start in range(0, len(audio) - HOP + 1, HOP):
        notes = tracker.step(audio[start : start + HOP])
        frames.append([(n.midi, n.attack) for n in notes])
    return frames, tracker


def test_single_notes_across_the_guitar_range():
    for midi in (40, 45, 52, 57, 64, 69, 76):
        frames, _ = run(pluck(midi))
        heard = [set(m for m, _ in f) for f in frames[10:30]]
        assert sum(found == {midi} for found in heard) >= 16, (note_name(midi), heard[5])


def test_two_notes_at_once_are_both_reported():
    frames, _ = run(pluck(52) + pluck(59))  # E3 and B3, a fifth
    heard = [set(m for m, _ in f) for f in frames[10:30]]
    assert sum({52, 59} <= found for found in heard) >= 14


def test_silence_and_noise_produce_no_notes():
    frames, _ = run(np.zeros(RATE))
    assert not any(frames)
    noise = 0.1 * np.random.default_rng(1).standard_normal(2 * RATE)
    frames, _ = run(noise)
    assert sum(bool(f) for f in frames) <= 0.05 * len(frames)


def test_an_attack_is_reported_once_and_a_held_note_is_sustain():
    audio = np.concatenate((pluck(57, 1.2), pluck(57, 1.2)))  # the same string struck twice
    frames, _ = run(audio)
    attacks = [index for index, f in enumerate(frames) if any(a for m, a in f if m == 57)]
    assert len(attacks) == 2 and attacks[1] == pytest.approx(60, abs=4)  # struck at 1.2 s
    sounding = sum(any(m == 57 for m, _ in f) for f in frames)
    assert sounding > 80  # held far longer than the two frames that carry an attack


def test_a_note_must_persist_before_it_is_shown():
    tracker = PitchTracker(RATE, HOP / RATE)
    audio = pluck(64, 0.2)
    shown = [bool(tracker.step(audio[i : i + HOP])) for i in range(0, 5 * HOP, HOP)]
    assert not shown[0] and tracker.lag_s == pytest.approx(0.0427, abs=0.001)


def test_engine_publishes_candidates_only_when_asked():
    audio = np.repeat((pluck(57, 2.0))[:, None], 2, axis=1).astype(np.float32)
    frames, events, *_ = run_detector(audio, RATE, "candidate", AnalysisConfig())
    assert all(f.pitch_method is None and not f.pitch_candidates for f in frames)
    assert not [e for e in events if e.event_type == "note_attack_candidate"]
    frames, events, *_ = run_detector(audio, RATE, "candidate", AnalysisConfig(pitch_tracking=True))
    middle = frames[20]
    assert middle.pitch_method == "harmonic_salience_v1"
    assert [round(m) for m, _ in middle.pitch_candidates] == [57]
    assert 0 < middle.pitch_candidates[0][1] <= 1
    assert middle.pitch_time < middle.window_end  # the long window describes an earlier moment
    attacks = [e for e in events if e.event_type == "note_attack_candidate"]
    assert len(attacks) == 1 and attacks[0].evidence_kind == "candidate"
    assert dict(attacks[0].measurements)["midi"] == 57
    assert attacks[0].timestamp < 0.12  # back-dated to when the note began


def test_two_weight_files_publish_their_labels_side_by_side(tmp_path):
    first = ActivityModel(weights(tmp_path / "a.npz"))
    second = ActivityModel(weights(tmp_path / "b.npz"))
    with pytest.raises(ValueError, match="same instrument label"):
        ActivityBank([first, second], RATE, 2229, 0.02)
    second.labels = ("guitar", "guitar_distorted")
    bank = ActivityBank([first, second], RATE, 2229, 0.02)
    assert bank.labels == ("drums", "bass", "guitar", "guitar_distorted")
    spectrum = np.fft.rfft(np.random.default_rng(0).standard_normal((2229, 2)), axis=0)
    scores, active = bank.step(spectrum, quiet=False)
    assert scores.shape == (4,) and active.tolist() == [True, False, True, False]
    assert bank.step(spectrum, quiet=True)[1].tolist() == [False] * 4


def test_visualizer_keeps_notes_separate_from_the_timeline_and_marks_guitar_time():
    state = VisualState()
    clock = feed(
        state,
        0.2,
        pitch_candidates=((57.0, 0.6), (64.0, 0.4)),
        pitch_time=0.0,
        pitch_method="harmonic_salience_v1",
    )
    assert state.pitch_method and len(state.notes) == 20
    assert state.notes.last.tolist() == [0.0, 64.0, 0.4, 0.0]  # not attributed to guitar
    state.consume(event(0.1, "note_attack_candidate", measurements=(("midi", 57.0),)))
    assert list(state.note_attacks) == [(0.1, 57.0, 0.6)] and not state.markers
    assert state.due(state.now(clock)) == []  # a note attack never flashes the display
    feed(
        state,
        0.1,
        start=0.2,
        clock=100.0,
        pitch_candidates=((57.0, 0.6),),
        pitch_time=0.2,
        pitch_method="harmonic_salience_v1",
        scores=(("guitar", 0.9),),
        active=("guitar",),
        method="guitar_test",
    )
    assert state.notes.last[3] == 1.0  # a guitar detector was active at that moment
    lanes = {lane.key: live for lane, live in state.board.lanes()}
    assert lanes["guitar"]


def test_note_lane_is_drawn_only_when_candidates_are_published():
    from PySide6.QtWidgets import QApplication

    from gmv.bus import EventBus
    from gmv.visualizer.window import VisualizerWindow

    app = QApplication.instance() or QApplication([])
    view = VisualizerWindow()
    view.timer.stop()
    view.resize(1200, 800)
    bus = EventBus()
    view.attach(bus)
    for index in range(100):
        bus.publish(frame(index * 0.02))
    view.tick()
    without = view.grab().toImage()
    for index in range(100, 200):
        time = index * 0.02
        bus.publish(
            frame(
                time,
                pitch_candidates=((45.0 + index % 12, 0.7),),
                pitch_time=time - 0.02,
                pitch_method="harmonic_salience_v1",
            )
        )
        if index % 12 == 0:
            bus.publish(event(time, "note_attack_candidate", measurements=(("midi", 45.0),)))
    view.tick()
    assert view.grab().toImage() != without and len(view.state.notes) == 100
    view.set_shown("notes", False)
    view.tick()
    hidden = view.grab().toImage()
    view.set_shown("notes", True)
    view.tick()
    assert view.grab().toImage() != hidden
    view.close()
    app.processEvents()


def test_trainer_can_learn_guitar_labels_and_remixes_without_each_instrument():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "train_activity", Path(__file__).parents[1] / "scripts" / "train_activity.py"
    )
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    trainer.configure(("guitar", "guitar_distorted", "vocals"))
    assert set(trainer.VARIANTS) == {"no_guitar", "no_vocals", "random"}  # sub-labels follow guitar
    stems = {
        name: np.full((100, 2), value, np.float32)
        for name, value in (("guitar", 1.0), ("vocals", 2.0), ("drums", 4.0), ("piano", 8.0))
    }
    audio, present = trainer.remix(stems, "no_guitar", seed=1)
    assert audio[0, 0] == 14.0 and present == [False, False, True]  # distorted guitar goes too
    audio, present = trainer.remix(stems, "no_vocals", seed=1)
    assert audio[0, 0] == 13.0 and present == [True, True, False]
    with pytest.raises(ValueError, match="Unknown instrument labels"):
        trainer.configure(("guitar", "theremin"))
