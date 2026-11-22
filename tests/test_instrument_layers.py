"""Instrument layers: fast guitar activity from synthetic audio, how it meets recognition,
and how layers and the Guitar lane draw it. No audio hardware, no recordings."""

import os
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import soundfile as sf

from gmv.contract import parse_message
from gmv.engine import Engine
from gmv.evaluation.detect import run_instruments
from gmv.instruments import UNCONFIRMED, analyzers, build, combine
from gmv.instruments.guitar import threshold_scale
from gmv.models import AnalysisConfig, InstrumentActivity, InstrumentEstimate
from gmv.sources.file import FileAudioSource
from gmv.visualizer import layers as layer_registry
from gmv.visualizer.state import COLUMN, VisualState
from gmv.visualizer.strip import Strip

from test_visualizer import frame

RATE = 48000
CHORD_LEFT = (196.0, 247.0, 294.0, 392.0, 494.0, 587.0)  # one take
CHORD_RIGHT = (196.5, 246.2, 295.1, 391.0, 495.5, 589.0)  # a second take, slightly different


def pluck(notes, seconds, decay, rng):
    """One struck chord: every note with a few harmonics, a sharp start and a decay. Phases
    are random, as they are between two performances of the same part."""
    t = np.arange(round(seconds * RATE)) / RATE
    sound = sum(
        np.sin(2 * np.pi * (note * harmonic * t + rng.random())) / harmonic
        for note in notes
        for harmonic in (1, 2, 3, 4, 5)
    )
    return (sound * np.exp(-t / decay) * np.minimum(1.0, t / 0.002)).astype(np.float32)


def guitar(seconds, times, decay=0.12, level=0.05, seed=1):
    """Stereo: a different take of the chord on each side, struck at ``times``."""
    rng = np.random.default_rng(seed)
    out = np.zeros((round(seconds * RATE), 2), np.float32)
    for column, notes in enumerate((CHORD_LEFT, CHORD_RIGHT)):
        for time in times:
            start = round(time * RATE)
            one = pluck(notes, min(seconds - time, 6 * decay + 0.05), decay, rng)
            out[start : start + len(one), column] += level * one[: len(out) - start]
    return out


def drums(seconds, times, level=0.25, seed=0):
    """Centre-panned noise bursts with a low thump: identical in both channels."""
    rng = np.random.default_rng(seed)
    out = np.zeros(round(seconds * RATE), np.float32)
    burst_t = np.arange(round(0.09 * RATE)) / RATE
    for time in times:
        start = round(time * RATE)
        burst = rng.standard_normal(len(burst_t)) * np.exp(-burst_t / 0.02)
        burst += 2 * np.sin(2 * np.pi * 60 * burst_t) * np.exp(-burst_t / 0.05)
        burst *= np.minimum(1.0, (burst_t[-1] - burst_t) / 0.02)  # no click where it ends
        out[start : start + len(burst)] += level * burst[: len(out) - start].astype(np.float32)
    return np.column_stack((out, out))


def attacks(messages):
    return np.array([m.timestamp for m in messages if m.attack > 0])


def nearest(found, reference):
    return np.array([np.abs(reference - time).min() for time in found])


# -- analysis ------------------------------------------------------------------------------


def test_fast_strumming_gives_repeated_attacks_at_its_rate_and_on_the_audio_timeline():
    strokes = np.arange(0.5, 5.0, 0.1)  # ten strokes a second
    messages, times = run_instruments(guitar(5.2, strokes), RATE)
    assert {m.instrument for m in messages} == {"guitar"} and messages[-1].basis == "stereo"
    found = attacks(messages)
    assert len(found) >= 0.9 * len(strokes)
    # Each attack sits on the stroke that caused it, in audio time, whatever the delay of
    # the analysis: an analysis window is 46 ms long and a frame 20 ms.
    assert np.median(nearest(found, strokes)) < 0.025 and nearest(found, strokes).max() < 0.05
    late = [m for m in messages if m.timestamp > 2.5]
    rates = [m.rate_hz for m in late if m.rate_hz is not None]
    assert len(rates) > 0.9 * len(late) and np.median(rates) == pytest.approx(10.0, abs=0.6)
    assert np.median([m.regularity for m in late]) > 0.6
    assert max(times) < 20  # far inside one 20 ms frame, even on a busy machine


def test_a_sustained_chord_is_one_attack_that_keeps_ringing_not_a_repeating_figure():
    held, _ = run_instruments(guitar(4.0, [0.5], decay=1.5), RATE)
    strummed, _ = run_instruments(guitar(4.0, np.arange(0.5, 3.9, 0.1)), RATE)
    assert len(attacks(held)) == 1 and attacks(held)[0] == pytest.approx(0.5, abs=0.03)
    ringing = [m for m in held if 1.0 < m.timestamp < 2.5]
    assert np.mean([m.level for m in ringing]) > 0.5  # still sounding
    # Two takes ringing against each other leave a small residue, far below a stroke.
    assert np.mean([m.flux for m in ringing]) < 0.2
    assert all(m.rate_hz is None for m in ringing)
    busy = [m for m in strummed if 1.0 < m.timestamp < 2.5]
    assert np.mean([m.flux for m in busy]) > 2 * np.mean([m.flux for m in ringing])
    assert np.median([m.rate_hz or 0 for m in busy]) == pytest.approx(10.0, abs=0.6)
    assert len(attacks(strummed)) > 25


def test_drum_hits_alone_do_not_become_guitar_attacks():
    hits = np.arange(0.5, 5.0, 0.125)  # eight hits a second, broadband, in the centre
    messages, _ = run_instruments(drums(5.2, hits), RATE)
    # Nothing is panned, so the analyzer says it is working from the mono signal.
    assert messages[-1].basis == "mono"
    assert len(attacks(messages)) <= 0.15 * len(hits)
    # The same rhythm played on a guitar is found, panned or not.
    strummed = guitar(5.2, hits)
    centred = np.repeat(strummed[:, :1], 2, axis=1)
    for audio, basis in ((strummed, "stereo"), (centred, "mono")):
        found, _ = run_instruments(audio, RATE)
        assert found[-1].basis == basis and len(attacks(found)) >= 0.85 * len(hits)
        assert np.mean([m.flux for m in messages]) < 0.4 * np.mean([m.flux for m in found])


def test_in_a_mix_the_attacks_follow_the_guitar_and_not_the_centre_drums():
    strokes = np.arange(0.5, 6.0, 0.2)  # five strokes a second
    hits = np.arange(0.57, 6.0, 0.3)  # drums on their own grid
    alone = hits[nearest(hits, strokes) > 0.06]  # hits that fall between strokes
    messages, _ = run_instruments(guitar(6.2, strokes) + drums(6.2, hits), RATE)
    found = attacks(messages)
    on_strokes = (nearest(found, strokes) <= 0.045).sum()
    on_lone_hits = (nearest(found, alone) <= 0.03).sum()
    assert on_strokes >= 0.85 * len(strokes)
    assert on_lone_hits <= 0.2 * len(alone)
    assert on_strokes >= 0.85 * len(found)


def test_sensitivity_changes_what_is_reported_and_silence_reports_nothing():
    quiet_strokes = guitar(4.0, np.arange(0.5, 3.9, 0.25), level=0.05)
    quiet_strokes += guitar(4.0, [0.2], decay=3.0, level=0.6)  # a loud chord held over them
    config = AnalysisConfig(instrument_analyzers=("guitar",), spectrum_bands=0)
    counts = [
        len(attacks(run_instruments(quiet_strokes, RATE, config, {"guitar": value})[0]))
        for value in (0.0, 0.5, 1.0)
    ]
    assert counts[0] <= counts[1] <= counts[2] and counts[2] > counts[0]
    assert threshold_scale(0.5) == 1.0 and threshold_scale(1.0) < 1 < threshold_scale(0.0)
    silent, _ = run_instruments(np.zeros((RATE * 2, 2), np.float32), RATE)
    assert all(m.flux == 0 and m.attack == 0 and m.level == 0 for m in silent)
    with pytest.raises(ValueError, match="No instrument analyzer"):
        build(["theremin"], RATE, 2229, 0.02)
    assert "guitar" in analyzers()


def test_the_first_frames_of_a_song_are_analysed_at_once():
    """Start-up, measured: the analyzer needs two frames, then answers one frame late."""
    messages, _ = run_instruments(guitar(1.5, [0.2, 0.4, 0.6]), RATE)
    assert messages[0].timestamp < 0.06
    first = attacks(messages)[0]
    assert first == pytest.approx(0.2, abs=0.03)
    hop = 0.02
    assert build(["guitar"], RATE, 2229, hop)[0].delay_s == pytest.approx(hop)


def test_activity_messages_survive_the_log_format():
    messages, _ = run_instruments(guitar(1.0, [0.3]), RATE)
    again = parse_message(messages[20].model_dump(mode="json"))
    assert isinstance(again, InstrumentActivity) and again == messages[20]
    with pytest.raises(ValueError):
        parse_message({**messages[20].model_dump(mode="json"), "schema_version": "1.0"})


# -- recognition meets fast activity --------------------------------------------------------


def activity(time, attack=0.0, flux=None, instrument="guitar", **extra):
    values = dict(
        session_id="s",
        source="synthetic",
        timestamp=time,
        detected_monotonic=0,
        instrument=instrument,
        method="test",
        level=0.5,
        flux=attack if flux is None else flux,
        attack=attack,
        rate_hz=None,
        regularity=0.0,
        centre=0.5,
        spread=0.5,
        basis="stereo",
    )
    return InstrumentActivity(**(values | extra))


def estimate(end, state, context_s=10.0):
    return InstrumentEstimate(
        session_id="s",
        source="synthetic",
        timestamp=end - 0.5,
        interval_end=end,
        detected_monotonic=0,
        backend="fake",
        scores=(("guitar", {"detected": 0.97, "candidate": 0.7, "absent": 0.2}[state]),),
        states=(("guitar", state),),
        context_s=context_s,
    )


def test_combine_is_the_only_rule_and_absent_overrules_activity():
    assert combine("detected") == (1.0, False)
    weight, tentative = combine("candidate")
    assert 0 < weight < 1 and not tentative
    assert combine("absent") == (0.0, False)
    # No judgement, or a model still listening to the silence before the song: tentative.
    assert combine(None) == (UNCONFIRMED, True)
    assert combine("absent", warming_up=True) == (UNCONFIRMED, True)
    assert combine("detected", warming_up=True) == (1.0, False)


def test_attribution_follows_the_recognition_model_on_the_audio_timeline():
    state = VisualState()
    for index in range(100 * 30):  # 30 s of frames, music from the start
        state.consume(frame(index * 0.02), arrival=10 + index * 0.02)
    times = np.array([1.0, 6.0, 12.0, 14.0, 16.0, 17.2, 19.0])
    # No recognition model at all: everything is tentative.
    weights, tentative = state.attribution("guitar", times)
    assert set(weights) == {UNCONFIRMED} and tentative.all()
    for end in np.arange(0.5, 16.01, 0.5):
        name = "absent" if end <= 8 or 13 < end <= 15 else "detected" if end > 15 else "candidate"
        state.consume(estimate(float(end), name))
    weights, tentative = state.attribution("guitar", times)
    # 1 s and 6 s: judged absent while the model had heard under 10 s of this music.
    assert weights[0] == weights[1] == UNCONFIRMED and tentative[:2].all()
    assert weights[2] == combine("candidate")[0] and not tentative[2]
    assert weights[3] == 0.0 and not tentative[3]  # absent after warm-up: nothing shown
    assert weights[4] == 1.0
    assert weights[5] == 1.0  # 1.2 s after the newest estimate: its judgement is held
    assert weights[6] == UNCONFIRMED and tentative[6]  # 3 s after it: unknown again


def test_activity_history_derives_ring_and_stroke_gaps_once_for_every_reader():
    state = VisualState()
    state.consume(activity(0.00))
    state.consume(activity(0.02, attack=0.8, centre=0.2, spread=0.1))
    state.consume(activity(0.04))
    state.consume(activity(0.12, attack=0.6, centre=0.9, spread=0.9))
    state.consume(activity(0.10, attack=1.0))  # out of order: ignored
    rows = state.activity["guitar"].since()
    assert len(rows) == 4
    assert rows[1, COLUMN["ring"]] == pytest.approx(0.8)
    assert 0 < rows[2, COLUMN["ring"]] < 0.8  # dying away
    assert rows[2, COLUMN["struck_centre"]] == pytest.approx(0.2)  # held from the stroke
    assert rows[3, COLUMN["gap"]] == pytest.approx(0.10) and rows[1, COLUMN["gap"]] == 0
    assert rows[3, COLUMN["struck_time"]] == pytest.approx(0.12)
    state.reset()
    assert state.activity == {}


# -- drawing -----------------------------------------------------------------------------------


def inked(pixmap):
    image = pixmap.toImage()
    return sum(
        image.pixelColor(x, y).alpha() > 0
        for x in range(0, image.width(), 2)
        for y in range(0, image.height(), 2)
    )


def play(view, seconds, strokes=(), estimates=(), rate=None, others=None):
    """Feed frames and activity so that the last of them has only just arrived. ``others``
    maps further instruments to their stroke times."""
    base = perf_counter() - seconds - 0.05
    parts = {"guitar": list(strokes), **(others or {})}
    for index in range(round(seconds / 0.02)):
        time = index * 0.02
        view.state.consume(frame(time), arrival=base + time + 0.05)
        for instrument, times in parts.items():
            hit = any(abs(time - stroke) < 0.01 for stroke in times)
            view.state.consume(
                activity(
                    time,
                    attack=0.8 if hit else 0.0,
                    flux=0.8 if hit else 0.02,
                    instrument=instrument,
                    rate_hz=rate,
                    regularity=0.9 if rate else 0.0,
                )
            )
    for end, name in estimates:
        view.state.consume(estimate(end, name))
    view.canvas.advance(0.016, perf_counter())


def test_guitar_activity_is_drawn_in_the_main_view_and_changes_with_the_playing(window):
    canvas = window.canvas
    canvas.info["instruments"] = ("guitar",)
    strokes = np.arange(0.5, 6.0, 0.1)
    play(window, 6.0, strokes, [(end, "detected") for end in np.arange(0.5, 6.01, 0.5)], rate=10.0)
    assert [layer.key for layer in canvas.active_layers()] == ["guitar"]
    strummed = window.grab().toImage()
    layer = canvas.layers["guitar"]
    busy = inked(layer.strip.pixmap)
    assert busy > 500
    window.set_layer("guitar", enabled=False)
    hidden = window.grab().toImage()
    assert hidden != strummed and canvas.active_layers() == []
    # The same stretch with guitar present but nothing articulated: far less is drawn.
    window.state.reset()
    window.set_layer("guitar", enabled=True)
    play(window, 6.0, (), [(end, "detected") for end in np.arange(0.5, 6.01, 0.5)])
    window.grab()
    assert inked(layer.strip.pixmap) < 0.6 * busy
    # Guitar judged absent: the articulation is still analysed but none of it is shown.
    window.state.reset()
    play(window, 16.0, np.arange(0.5, 16.0, 0.1), [(end, "absent") for end in np.arange(0.5, 16.01, 0.5)])
    window.grab()
    assert len(window.state.activity["guitar"]) > 700
    assert "not detected" in layer.reading(window.state)


def test_the_layer_and_the_guitar_lane_draw_the_same_attacks(window, monkeypatch):
    canvas = window.canvas
    canvas.info["instruments"] = ("guitar",)
    layer = canvas.layers["guitar"]
    seen = {"stage": [], "lane": []}
    stage_paint, lane_paint = layer.paint_history, layer.paint_lane

    def stage(artist, to_x, rows, *rest):
        seen["stage"] += [float(t) for t in rows[1:][rows[1:, COLUMN["attack"]] > 0][:, 0]]
        return stage_paint(artist, to_x, rows, *rest)

    def lane(artist, to_x, rows, *rest):
        seen["lane"] += [float(t) for t in rows[1:][rows[1:, COLUMN["attack"]] > 0][:, 0]]
        return lane_paint(artist, to_x, rows, *rest)

    monkeypatch.setattr(layer, "paint_history", stage)
    monkeypatch.setattr(layer, "paint_lane", lane)
    strokes = np.arange(0.5, 3.4, 0.2)
    play(window, 3.5, strokes, [(end, "detected") for end in np.arange(0.5, 3.51, 0.5)])
    window.grab()
    # Guitar has a live lane from activity alone as well (no recognition needed for that).
    assert dict((lane.key, live) for lane, live in canvas.state.board.lanes())["guitar"]
    assert len(seen["stage"]) == len(strokes) - 1 or len(seen["stage"]) == len(strokes)
    assert sorted(set(seen["lane"])) == pytest.approx(sorted(set(seen["stage"])))


class PulseLayer(layer_registry.Layer):
    """A second instrument for the tests: one filled block per attack."""

    def paint_history(self, artist, to_x, rows, weights, tentative, height, stage):
        self.times = getattr(self, "times", []) + [float(t) for t in rows[1:, 0]]
        for row in rows[1:]:
            if row[COLUMN["attack"]] > 0:
                artist.fillRect(int(to_x(row[0])), 0, 6, int(height), "#f87171")

    def paint_lane(self, artist, to_x, rows, top, bottom):
        pass


def test_two_instruments_are_drawn_at_once_each_from_its_own_activity(window):
    canvas = window.canvas
    profile = layer_registry.Profile("pulse", "Pulse", "#f87171", "test", "test")
    canvas.layers["pulse"] = PulseLayer(profile)
    canvas.info["instruments"] = ("guitar", "pulse")
    base = [(end, "detected") for end in np.arange(0.5, 3.01, 0.5)]
    play(window, 3.0, np.arange(0.5, 2.9, 0.2), base, others={"pulse": np.arange(0.6, 2.9, 0.5)})
    assert [layer.key for layer in canvas.active_layers()] == ["guitar", "pulse"]
    both = window.grab().toImage()
    guitar, pulse = canvas.layers["guitar"], canvas.layers["pulse"]
    assert inked(guitar.strip.pixmap) > 300 and inked(pulse.strip.pixmap) > 50
    # Each layer read its own instrument's rows, and they are not the same picture.
    assert len(window.state.activity["pulse"]) == len(window.state.activity["guitar"])
    assert guitar.strip.pixmap.toImage() != pulse.strip.pixmap.toImage()
    assert guitar.strip.pixmap.height() == pulse.strip.pixmap.height()  # equal shares
    # Hiding one leaves the other drawing, with the room to itself.
    shared = guitar.strip.pixmap.height()
    window.set_layer("pulse", enabled=False)
    alone = window.grab().toImage()
    assert alone != both and [layer.key for layer in canvas.active_layers()] == ["guitar"]
    assert guitar.strip.pixmap.height() > shared and inked(guitar.strip.pixmap) > 300
    window.set_layer("pulse", enabled=True)
    window.set_layer("guitar", enabled=False)
    assert window.grab().toImage() != alone and inked(pulse.strip.pixmap) > 50
    # The whole-mix spectrum can share the area, step behind the layers or leave.
    window.set_layer("guitar", enabled=True)
    pictures = []
    for place in ("beside", "behind", "hidden"):
        window.set_spectrum_place(place)
        pictures.append(window.grab().toImage())
    assert pictures[0] != pictures[1] != pictures[2] and pictures[0] != pictures[2]


def test_intensity_and_style_change_the_picture_and_never_the_activity(window):
    canvas = window.canvas
    canvas.info["instruments"] = ("guitar",)
    play(window, 3.0, np.arange(0.5, 2.9, 0.2), [(end, "detected") for end in (0.5, 1, 1.5, 2, 2.5, 3)])
    window.grab()
    layer = canvas.layers["guitar"]
    before = window.state.activity["guitar"].since().copy()
    normal = layer.strip.pixmap.toImage()
    window.set_layer("guitar", intensity=0.3)
    window.grab()
    faint = layer.strip.pixmap.toImage()
    window.set_layer("guitar", intensity=1.0, style="strokes")
    window.grab()
    strokes = layer.strip.pixmap.toImage()
    assert normal != faint and normal != strokes
    assert np.array_equal(window.state.activity["guitar"].since(), before, equal_nan=True)
    window.set_layer("guitar", style="no such style")
    assert layer.style == "strokes"
    # And the analysis has no way to be told the intensity: its only knob is sensitivity.
    assert "intensity" not in AnalysisConfig.model_fields
    assert set(layer_registry.profiles()) >= {"guitar", "bass", "drums", "vocals"}
    assert [key for key, p in layer_registry.profiles().items() if p.renderer] == ["guitar"]


def test_the_layer_is_drawn_as_activity_arrives_and_redrawn_when_judgement_changes(window):
    """Activity arrives every frame, recognition half a second late. The strip must fill in
    as rows arrive, and what was drawn tentatively must change once the model has judged."""
    from time import sleep

    canvas = window.canvas
    bus_free_start = perf_counter()
    canvas.info["instruments"] = ("guitar",)
    layer = canvas.layers["guitar"]
    due, pictures = 0.5, {}
    while (now := perf_counter() - bus_free_start) < 1.6:
        window.state.consume(frame(now), arrival=perf_counter())
        hit = int(now / 0.1) != int((now - 0.02) / 0.1)
        window.state.consume(activity(now, attack=0.8 if hit else 0.0, flux=0.8 if hit else 0.02))
        if now >= due:
            if due == 0.5:
                canvas.advance(0.016, perf_counter())
                window.grab()
                pictures["tentative"] = inked(layer.strip.pixmap), canvas.state.board.judgement
            window.state.consume(estimate(due, "detected"))
            due += 0.5
        canvas.advance(0.016, perf_counter())
        window.grab()
        sleep(0.02)
    live = inked(layer.strip.pixmap)
    assert pictures["tentative"][0] > 0 and canvas.state.board.judgement > pictures["tentative"][1]
    layer.strip = Strip()  # the same history drawn in one go
    layer.drawn_until = None
    window.grab()
    assert live > 0 and live == pytest.approx(inked(layer.strip.pixmap), rel=0.08)


# -- the whole engine ----------------------------------------------------------------------------


def test_engine_publishes_guitar_activity_without_dropping_audio_and_follows_sensitivity(tmp_path):
    path = tmp_path / "strum.wav"
    sf.write(path, guitar(3.0, np.arange(0.3, 2.9, 0.1)), RATE, subtype="FLOAT")

    def run(config, sensitivity=None):
        engine = Engine(FileAudioSource(path, realtime=True), config, session_root=tmp_path / "s")
        if sensitivity is not None:
            engine.instrument_sensitivity["guitar"] = sensitivity
        queue = engine.bus.subscribe(8192)
        engine.start()
        assert engine.done.wait(30)
        found = []
        while (message := queue.get(timeout=0)) is not None:
            found.append(message)
        return engine, found

    engine, messages = run(AnalysisConfig(instrument_analyzers=("guitar",)))
    metrics = engine.metrics()
    found = [m for m in messages if isinstance(m, InstrumentActivity)]
    assert metrics["error"] is None and metrics["dropped_blocks"] == 0
    assert metrics["queue_high_water"] <= engine.config.queue_blocks
    assert len(found) >= engine.frame_count - 2 and len(attacks(found)) >= 20
    assert metrics["instrument_analysis_ms"]["p95"] < 5.0
    # Published within a fraction of a second of the audio it describes (file source, real
    # time): the analysis delay is one frame.
    ages = [m.age_at_publication_ms for m in found if m.age_at_publication_ms is not None]
    assert ages and np.median(ages) < 150
    log = engine.writer.path / "advanced_analysis" / "instrument_activity.jsonl"
    assert len(log.read_text(encoding="utf-8").splitlines()) == len(found)
    # Sensitivity set on the running engine reaches the analyzer; without analyzers
    # configured the engine publishes no activity at all.
    _, fewer = run(AnalysisConfig(instrument_analyzers=("guitar",)), sensitivity=0.0)
    fewer = [m for m in fewer if isinstance(m, InstrumentActivity)]
    assert np.mean([m.flux for m in fewer]) < np.mean([m.flux for m in found])
    engine, messages = run(AnalysisConfig())
    assert not any(isinstance(m, InstrumentActivity) for m in messages)
    assert engine.metrics()["instrument_analysis_ms"] is None
