"""Drum, bass and vocal layers: analysis on synthetic instruments, the four-layer view, and
proof that the accepted guitar analysis is untouched. No audio hardware, no recordings."""

import hashlib
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import soundfile as sf

import synth_instruments as synth
from gmv.contract import parse_message
from gmv.engine import Engine
from gmv.evaluation.detect import run_instruments, run_layers
from gmv.instruments import analyzers
from gmv.models import AnalysisConfig, InstrumentActivity, InstrumentEstimate
from gmv.sources.file import FileAudioSource
from gmv.visualizer import layers as layer_registry
from gmv.visualizer.state import COLUMN, VisualState

from test_instrument_layers import guitar
from test_visualizer import frame

RATE = synth.RATE
ALL = ("guitar", "drums", "bass", "vocals")


def analyse(audio, names=("drums", "bass", "vocals"), sensitivity=None):
    config = AnalysisConfig(instrument_analyzers=names, spectrum_bands=0)
    _, messages, times = run_layers(audio, RATE, config, sensitivity)
    return {name: [m for m in messages if m.instrument == name] for name in names}, times


def hits(messages, region=None):
    return np.array(
        [
            m.timestamp
            for m in messages
            if (m.attack > 0 if region is None else dict(m.components)[region] > 0)
        ]
    )


def near(found, reference, tolerance=0.05):
    """How many of ``reference`` have something in ``found`` within ``tolerance``."""
    if not len(found):
        return 0
    return int(sum(np.abs(found - time).min() <= tolerance for time in reference))


# -- drums --------------------------------------------------------------------------------

KICKS, SNARES = np.arange(1.0, 5.0, 0.5), np.arange(1.25, 5.0, 0.5)
HATS = np.arange(1.0, 5.0, 0.125)


def kit(seconds=5.4):
    return synth.kick(seconds, KICKS) + synth.snare(seconds, SNARES) + synth.hat(seconds, HATS)


def test_drum_hits_are_separate_impacts_in_the_region_they_were_struck_in():
    drums = analyse(kit())[0]["drums"]
    low, mid, high = (hits(drums, region) for region in ("low", "mid", "high"))
    # Each kind of hit is found where it belongs, on the audio timeline.
    assert near(low, KICKS) >= len(KICKS) - 1 and near(mid, SNARES) >= len(SNARES) - 1
    assert near(high, HATS) >= 0.85 * len(HATS)
    # A snare is not a low hit, and nothing is reported between the hits.
    assert near(low, SNARES, 0.03) == 0
    assert len(low) <= len(KICKS) + 1 and len(high) <= len(HATS) + 2
    # Impacts: the articulation is over within a few frames of each hit, not continuous.
    quiet = [
        m.flux for m in drums if 1.0 < m.timestamp < 5.0 and np.abs(HATS - m.timestamp).min() > 0.05
    ]
    assert np.mean(quiet) < 0.03
    rates = [m.rate_hz for m in drums if m.timestamp > 3 and m.rate_hz is not None]
    # The pattern repeats: named at the bar's pulse or one of its subdivisions.
    assert rates and min(abs(np.median(rates) - pulse) for pulse in (2.0, 4.0, 8.0)) < 0.5


def test_strums_bass_notes_and_held_sounds_are_not_drum_hits():
    strokes = np.arange(1.0, 5.0, 0.25)
    notes = [(1.0 + index, midi, 0.9) for index, midi in enumerate((28, 33, 38, 43))]
    for name, audio, events in (
        ("guitar strums", guitar(5.4, strokes), len(strokes)),
        ("bass notes", synth.bass(5.4, notes), len(notes)),
        ("held bass note", synth.bass(5.4, [(1.0, 28, 4.0)], decay=5.0), 1),
        ("sung phrase", synth.voice(5.4, [(1.0, 3.5, 57, 60)]), 1),
        ("held pad", synth.pad(5.4), 1),
    ):
        drums = analyse(audio)[0]["drums"]
        later = [m for m in drums if m.timestamp > 0.9]
        # Bass notes and strums are not kicks (a strum now and then gets through).
        assert len(hits(later, "low")) <= 0.2 * events, name
        # Played alone, with nothing else to measure against, fewer than half of the
        # strums leave a mark of any kind; a kit's hits nearly all do (the test above).
        assert len(hits(later)) <= max(2, 0.5 * events), name
    # And a kit is found among them, though a dense mix hides some of its hits.
    everything = kit() + guitar(5.4, strokes) + synth.bass(5.4, notes)
    drums = analyse(everything)[0]["drums"]
    assert near(hits(drums), KICKS) + near(hits(drums), SNARES) >= 8
    assert near(hits(drums, "high"), HATS) >= 0.8 * len(HATS)


def test_taking_the_drums_out_of_a_mix_takes_the_impacts_with_them():
    strokes = np.arange(1.0, 5.0, 0.25)
    rest = (
        guitar(5.4, strokes)
        + synth.bass(5.4, [(1.0 + index, 33, 0.9) for index in range(4)])
        + synth.voice(5.4, [(1.2, 1.6, 57, 60), (3.3, 1.6, 60, 57)])
    )

    def strength(audio):
        return sum(m.attack for m in analyse(audio)[0]["drums"])

    with_drums, without = strength(rest + kit()), strength(rest)
    assert with_drums > 10 and without < 0.3 * with_drums


# -- bass -----------------------------------------------------------------------------------


def test_bass_notes_are_read_at_their_pitch_and_a_new_note_is_an_attack():
    notes = [(1.0, 28, 0.9), (2.0, 33, 0.9), (3.0, 38, 0.9), (4.0, 43, 0.9)]  # E1 A1 D2 G2
    bass = analyse(synth.bass(5.4, notes))[0]["bass"]
    for start, midi, _ in notes:
        inside = [m for m in bass if start + 0.25 < m.timestamp < start + 0.8]
        pitches = [m.pitch_midi for m in inside if m.pitch_midi is not None]
        assert len(pitches) >= 0.9 * len(inside)
        assert np.median(pitches) == pytest.approx(midi, abs=0.5)
        assert np.mean([m.level for m in inside]) > 0.5  # heavy while it sounds
    found = hits(bass)
    assert near(found, [start for start, _, _ in notes], 0.12) == 4 and len(found) <= 5
    # Higher note, higher place: what moves the ribbon.
    place = [
        np.median([m.centre for m in bass if s + 0.3 < m.timestamp < s + 0.8]) for s, _, _ in notes
    ]
    assert place == sorted(place) and place[-1] - place[0] > 0.3
    # Between notes there is nothing: the note is not held over silence.
    assert all(m.pitch_midi is None and m.level < 0.2 for m in bass if 5.15 < m.timestamp < 5.3)


def test_a_held_bass_note_stays_and_a_repeated_one_is_struck_again():
    held = analyse(synth.bass(5.0, [(1.0, 28, 3.6)], decay=6.0))[0]["bass"]
    during = [m for m in held if 1.3 < m.timestamp < 4.4]
    assert len(hits(held)) == 1 and min(m.level for m in during) > 0.5
    assert all(m.pitch_midi is not None for m in during)
    assert np.ptp([m.pitch_midi for m in during]) < 0.5
    times = 1.0 + 0.25 * np.arange(12)
    repeated = analyse(synth.bass(5.0, [(t, 33, 0.19) for t in times], decay=0.25))[0]["bass"]
    assert near(hits(repeated), times, 0.1) >= 9
    pitches = [
        m.pitch_midi for m in repeated if m.pitch_midi is not None and 1.2 < m.timestamp < 3.9
    ]
    assert np.median(pitches) == pytest.approx(33, abs=0.5)


def test_kick_drums_do_not_become_bass_notes_and_removing_the_bass_removes_them():
    kicks = analyse(synth.kick(5.4, KICKS))[0]["bass"]
    assert not any(m.pitch_midi is not None for m in kicks) and len(hits(kicks)) == 0
    notes = [(1.0 + index, midi, 0.95) for index, midi in enumerate((28, 33, 38, 43))]
    rest = kit() + guitar(5.4, np.arange(1.0, 5.0, 0.25), level=0.02)
    full = analyse(rest + synth.bass(5.4, notes))[0]["bass"]
    twin = analyse(rest)[0]["bass"]

    def noted(messages):
        return np.mean(
            [m.pitch_midi is not None and m.pitch_midi < 45 for m in messages if m.timestamp > 1.2]
        )

    # With the bass, its notes are read through the kit; without it, hardly any are.
    assert noted(full) > 0.7 and noted(twin) < 0.3 * noted(full)
    for start, midi, _ in notes:
        got = [
            m.pitch_midi for m in full if start + 0.3 < m.timestamp < start + 0.8 and m.pitch_midi
        ]
        assert np.median(got) == pytest.approx(midi, abs=0.6)


# -- vocals -----------------------------------------------------------------------------------


def test_a_sung_phrase_is_continuous_and_phrases_have_a_start_and_an_end():
    phrases = [(1.0, 1.6, 57, 57), (3.2, 1.6, 60, 60)]
    vocals = analyse(synth.voice(5.6, phrases))[0]["vocals"]
    for start, length, midi, _ in phrases:
        inside = [m for m in vocals if start + 0.25 < m.timestamp < start + length - 0.2]
        assert min(m.level for m in inside) > 0.3  # no holes in a held note
        pitches = [m.pitch_midi for m in inside if m.pitch_midi is not None]
        assert len(pitches) > 0.6 * len(inside)
        assert np.median(pitches) == pytest.approx(midi, abs=0.75)
    between = [m.level for m in vocals if 2.9 < m.timestamp < 3.1]
    assert max(between) < 0.15  # it closes between the phrases
    marks = [(round(m.timestamp, 2), m.boundary) for m in vocals if m.boundary]
    assert [kind for _, kind in marks] == ["start", "end", "start", "end"]
    assert marks[0][0] == pytest.approx(1.0, abs=0.2) and marks[2][0] == pytest.approx(3.2, abs=0.2)
    assert marks[1][0] == pytest.approx(2.6 + 0.25, abs=0.2)  # a quarter second after it stops


def test_a_rising_line_moves_the_contour_up():
    vocals = analyse(synth.voice(4.0, [(0.6, 3.0, 57, 64)]))[0]["vocals"]
    early = [m.pitch_midi for m in vocals if 0.9 < m.timestamp < 1.4 and m.pitch_midi]
    late = [m.pitch_midi for m in vocals if 3.0 < m.timestamp < 3.5 and m.pitch_midi]
    assert np.median(late) - np.median(early) == pytest.approx(5.0, abs=1.5)
    centre = [m.centre for m in vocals if 0.9 < m.timestamp < 3.5 and m.pitch_midi]
    assert centre[-1] > centre[0] + 0.1


def test_panned_instruments_and_drums_are_not_phrases_and_removing_the_voice_removes_it():
    strokes = np.arange(1.0, 5.0, 0.25)
    one_side = guitar(5.4, strokes) * np.array([1.0, 0.0], np.float32)  # hard left
    pad_right = synth.pad(5.4) * np.array([0.0, 1.0], np.float32)
    for name, audio in (("guitar on one side", one_side), ("pad on one side", pad_right)):
        vocals = analyse(audio)[0]["vocals"]
        assert not any(m.boundary for m in vocals) and max(m.level for m in vocals) < 0.1, name
    phrases = [(1.0, 1.6, 57, 60), (3.2, 1.6, 60, 57)]
    rest = one_side + pad_right + synth.hat(5.4, HATS)
    full = analyse(rest + synth.voice(5.4, phrases))[0]["vocals"]
    twin = analyse(rest)[0]["vocals"]
    level = np.mean([m.level for m in full if m.timestamp > 1.2])
    # Materially less without the voice; what is left is other sound that reaches the centre.
    assert level > 0.3 and np.mean([m.level for m in twin if m.timestamp > 1.2]) < 0.5 * level
    assert sum(m.boundary == "start" for m in full) == 2
    # What a centred, pitched instrument does is not decided here: the activity model
    # decides, and while it says "no vocals" nothing of this is attributed to a voice.
    state = VisualState()
    for index in range(100):
        state.consume(
            frame(index * 0.02, scores=(("vocals", 0.1),), active=(), method="m"),
            arrival=5 + index * 0.02,
        )
    weights, tentative = state.attribution("vocals", np.array([0.5, 1.5]))
    assert list(weights) == [0.0, 0.0] and not tentative.any()  # absent, and no warm-up excuse


# -- the contract -------------------------------------------------------------------------------


def test_messages_carry_instrument_specific_fields_and_old_logs_still_parse():
    found, _ = analyse(
        kit() + synth.bass(5.4, [(1.0, 33, 3.0)]) + synth.voice(5.4, [(1.0, 3.0, 57, 60)])
    )
    hit = next(m for m in found["drums"] if m.attack > 0)
    assert dict(hit.components).keys() == {"low", "mid", "high"} and hit.pitch_midi is None
    note = next(m for m in found["bass"] if m.pitch_midi is not None)
    assert note.components == () and 0 < note.pitch_confidence <= 1
    assert {m.boundary for m in found["vocals"]} == {"", "start", "end"}
    for message in (hit, note):
        assert parse_message(message.model_dump(mode="json")) == message
    # A message logged before these fields existed (phase 1) is still valid.
    old = {
        k: v
        for k, v in note.model_dump(mode="json").items()
        if k not in ("components", "pitch_midi", "pitch_confidence", "boundary")
    }
    assert parse_message(old).pitch_midi is None
    assert set(analyzers()) == set(ALL)


# -- guitar is as it was accepted ----------------------------------------------------------------


def fingerprint(messages):
    text = ";".join(
        f"{m.timestamp:.4f},{m.level:.4f},{m.flux:.4f},{m.attack:.4f},{m.rate_hz},{m.regularity:.4f},{m.basis}"
        for m in messages
    )
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def test_the_accepted_guitar_analysis_is_unchanged_and_independent_of_the_other_analyzers():
    audio = (
        guitar(5.2, np.arange(0.5, 5.0, 0.1))
        + synth.kick(5.2, KICKS)
        + synth.bass(5.2, [(1.0, 33, 3.0)])
    )
    alone, _ = run_instruments(audio, RATE)
    beside = analyse(audio, ALL)[0]["guitar"]
    # The same numbers, whether or not the other analyzers run beside it.
    assert fingerprint(alone) == fingerprint(beside) and len(alone) == len(beside)
    # The analyzer's source is the file accepted in phase 1 with one added, unused parameter.
    source = Path(__file__).parents[1] / "src" / "gmv" / "instruments" / "guitar.py"
    text = source.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "def step(self, spectrum, quiet=False, samples=None):" in text
    accepted = text.replace(
        "def step(self, spectrum, quiet=False, samples=None):",
        "def step(self, spectrum, quiet=False):",
    )
    assert hashlib.sha256(accepted.encode()).hexdigest()[:16] == GUITAR_SOURCE
    assert fingerprint(alone) == GUITAR_OUTPUT


GUITAR_SOURCE = "ca6dbbadd693d517"
GUITAR_OUTPUT = "1bd726dbeadb6b12"


# -- four layers at once ---------------------------------------------------------------------


def activity(time, instrument, **values):
    base = dict(
        session_id="s",
        source="synthetic",
        timestamp=time,
        detected_monotonic=0,
        instrument=instrument,
        method="test",
        level=0.5,
        flux=0.02,
        attack=0.0,
    )
    return InstrumentActivity(**(base | values))


def play_band(view, seconds=4.0, present=ALL):
    """Feed four instruments' activity, the activity model's judgement in the frames and
    the recognition model's for guitar, so that the last of it has only just arrived."""
    base = perf_counter() - seconds - 0.05
    scores = tuple((name, 0.9 if name in present else 0.05) for name in ("drums", "bass", "vocals"))
    active = tuple(name for name in ("drums", "bass", "vocals") if name in present)
    for index in range(round(seconds / 0.02)):
        time = index * 0.02
        view.state.consume(
            frame(time, scores=scores, active=active, method="activity_test"),
            arrival=base + time + 0.05,
        )
        beat = index % 25
        view.state.consume(
            activity(
                time,
                "guitar",
                attack=0.8 * (index % 10 == 0),
                flux=0.8 * (index % 10 == 0),
                centre=0.4,
                spread=0.6,
            )
        )
        view.state.consume(
            activity(
                time,
                "drums",
                attack=0.9 * (beat in (0, 12)),
                flux=0.9 * (beat in (0, 12)),
                components=(("low", 0.9 * (beat == 0)), ("mid", 0.8 * (beat == 12)), ("high", 0.0)),
            )
        )
        note = 28 + 5 * (index // 50 % 3)
        view.state.consume(
            activity(
                time,
                "bass",
                level=0.7,
                pitch_midi=float(note),
                pitch_confidence=0.9,
                attack=0.7 * (index % 50 == 0),
            )
        )
        phrase = index % 100
        view.state.consume(
            activity(
                time,
                "vocals",
                level=0.6 * (phrase < 70),
                pitch_midi=60.0 + 4 * np.sin(index / 20) if phrase < 70 else None,
                boundary="start" if phrase == 0 else "end" if phrase == 82 else "",
            )
        )
    for end in np.arange(0.5, seconds + 0.01, 0.5):
        view.state.consume(
            InstrumentEstimate(
                session_id="s",
                source="synthetic",
                timestamp=end - 0.5,
                interval_end=float(end),
                detected_monotonic=0,
                backend="fake",
                scores=(("guitar", 0.97),),
                states=(("guitar", "detected" if "guitar" in present else "absent"),),
            )
        )
    view.canvas.advance(0.016, perf_counter())


def inked(pixmap):
    image = pixmap.toImage()
    return sum(
        image.pixelColor(x, y).alpha() > 0
        for x in range(0, image.width(), 2)
        for y in range(0, image.height(), 2)
    )


def test_four_instruments_are_drawn_together_each_with_its_own_picture(window):
    canvas = window.canvas
    canvas.info["instruments"] = ALL
    play_band(window)
    window.grab()
    active = canvas.active_layers()
    assert [layer.key for layer in active] == ["vocals", "guitar", "bass", "drums"]
    assert len({type(layer) for layer in active}) == 4  # four renderers, not one tinted four ways
    pictures = {layer.key: layer.strip.pixmap.toImage() for layer in active}
    assert all(inked(layer.strip.pixmap) > 150 for layer in active)
    keys = list(pictures)
    assert all(pictures[a] != pictures[b] for i, a in enumerate(keys) for b in keys[i + 1 :])
    # Room by need: the six strings get the tallest band, the bass ribbon the shortest.
    tall = {layer.key: layer.strip.pixmap.height() for layer in active}
    assert tall["guitar"] > tall["vocals"] >= tall["drums"] > tall["bass"]
    # How much each kind of picture covers differs in kind: drums are marks, bass is a slab.
    assert inked(canvas.layers["bass"].strip.pixmap) > 2 * inked(
        canvas.layers["drums"].strip.pixmap
    )
    # All of it still works with the spectrum beside, behind or hidden, bars or radial.
    seen = set()
    for mode in ("bars", "radial"):
        window.set_mode(mode)
        for place in ("beside", "behind", "hidden"):
            window.set_spectrum_place(place)
            seen.add(window.grab().toImage().cacheKey())
    assert len(seen) == 6 and max(canvas.paint_ms) < 250


def test_hiding_or_dimming_one_layer_leaves_the_others_and_all_activity_alone(window):
    canvas = window.canvas
    canvas.info["instruments"] = ALL
    play_band(window)
    window.grab()
    before = {key: track.since().copy() for key, track in window.state.activity.items()}
    window.set_layer("drums", enabled=False)
    window.grab()
    assert [layer.key for layer in canvas.active_layers()] == ["vocals", "guitar", "bass"]
    assert all(inked(canvas.layers[key].strip.pixmap) > 150 for key in ("vocals", "guitar", "bass"))
    window.set_layer("drums", enabled=True)
    window.grab()
    normal = canvas.layers["bass"].strip.pixmap.toImage()
    others = {
        key: canvas.layers[key].strip.pixmap.toImage() for key in ("vocals", "guitar", "drums")
    }
    window.set_layer("bass", intensity=0.4)
    window.grab()
    assert canvas.layers["bass"].strip.pixmap.toImage() != normal
    assert all(canvas.layers[key].strip.pixmap.toImage() == image for key, image in others.items())
    for key, rows in before.items():
        assert np.array_equal(window.state.activity[key].since(), rows, equal_nan=True)
    # Styles exist where they mean something, and an absent instrument is not drawn.
    styles = {
        key: [name for name, _ in profile.styles]
        for key, profile in layer_registry.profiles().items()
    }
    assert (
        styles["bass"] == []
        and len(styles["drums"]) == len(styles["vocals"]) == len(styles["guitar"]) == 2
    )
    window.state.reset()
    play_band(window, present=("guitar", "drums"))
    window.grab()
    assert "not detected" in canvas.layers["bass"].reading(window.state)
    assert "not detected" in canvas.layers["vocals"].reading(window.state)
    assert "not detected" not in canvas.layers["drums"].reading(window.state)


def test_each_lane_is_handed_the_same_rows_as_its_layer(window, monkeypatch):
    canvas = window.canvas
    canvas.info["instruments"] = ALL
    seen = {}
    for key in ("drums", "bass", "vocals"):
        layer = canvas.layers[key]
        seen[key] = {"stage": set(), "lane": set()}

        def stage(artist, to_x, rows, *rest, key=key, paint=layer.paint_history):
            seen[key]["stage"].update(float(t) for t in rows[1:, 0])
            return paint(artist, to_x, rows, *rest)

        def lane(artist, to_x, rows, *rest, key=key, paint=layer.paint_lane):
            seen[key]["lane"].update(float(t) for t in rows[1:, 0])
            return paint(artist, to_x, rows, *rest)

        monkeypatch.setattr(layer, "paint_history", stage)
        monkeypatch.setattr(layer, "paint_lane", lane)
    window.set_section("details", shown=True)
    play_band(window, seconds=3.0)
    window.grab()
    for key, where in seen.items():
        assert len(where["stage"]) > 100 and where["stage"] == where["lane"], key
    rows = window.state.activity["drums"].since()
    assert rows[:, COLUMN["low"]].max() > 0 and rows[:, COLUMN["mid"]].max() > 0
    assert np.nanmax(window.state.activity["bass"].since()[:, COLUMN["pitch"]]) == 38
    assert set(window.state.activity["vocals"].since()[:, COLUMN["boundary"]]) == {0.0, 1.0, 2.0}


def test_settings_offer_every_layer_its_own_controls(app, tmp_path):
    from gmv.app.window import MainWindow

    window = MainWindow(session_root=tmp_path, remember=False, discover_sources=False)
    try:
        controls = window.drawer.layer_controls
        assert list(controls) == ["vocals", "guitar", "bass", "drums"]
        settings = window.drawer.settings()
        assert dict(settings.instrument_sensitivity) == {key: 0.5 for key in controls}
        controls["drums"]["sensitivity"].setValue(80)
        controls["drums"]["intensity"].setValue(140)
        controls["bass"]["shown"].setChecked(False)
        assert dict(window.drawer.settings().instrument_sensitivity)["drums"] == 0.8
        layers = window.view.canvas.layers
        assert layers["drums"].intensity == 1.4 and not layers["bass"].enabled
        assert layers["guitar"].intensity == 1.0 and layers["guitar"].enabled  # untouched
        saved = window.drawer.remembered()["layers"]
        assert saved["drums"] == {
            "shown": True,
            "sensitivity": 80,
            "intensity": 140,
            "style": "all",
        }
    finally:
        window.close()


# -- the whole engine ----------------------------------------------------------------------------


def test_engine_runs_all_four_analyzers_inside_one_pass_without_dropping_audio(tmp_path):
    path = tmp_path / "band.wav"
    audio = (
        kit(4.0)[: 4 * RATE]
        + guitar(4.0, np.arange(0.5, 3.9, 0.1))
        + synth.bass(4.0, [(0.5, 33, 1.5), (2.0, 38, 1.5)])
        + synth.voice(4.0, [(0.6, 2.8, 57, 62)])
    )
    sf.write(path, audio, RATE, subtype="FLOAT")
    engine = Engine(
        FileAudioSource(path, realtime=True),
        AnalysisConfig(instrument_analyzers=ALL),
        session_root=tmp_path / "s",
    )
    queue = engine.bus.subscribe(16384)
    engine.start()
    assert engine.done.wait(40)
    found = []
    while (message := queue.get(timeout=0)) is not None:
        if isinstance(message, InstrumentActivity):
            found.append(message)
    metrics = engine.metrics()
    assert metrics["error"] is None and metrics["dropped_blocks"] == 0
    assert metrics["queue_high_water"] <= engine.config.queue_blocks
    by = {name: [m for m in found if m.instrument == name] for name in ALL}
    assert all(len(rows) >= engine.frame_count - 12 for rows in by.values())
    # Four analyzers together stay far inside one 20 ms frame.
    assert metrics["instrument_analysis_ms"]["p95"] < 8.0 and metrics["dsp_ms"]["p95"] < 20.0
    ages = [m.age_at_publication_ms for m in found if m.age_at_publication_ms is not None]
    assert np.percentile(ages, 95) < 400  # the bass looks back 85 ms, the drums 60
    # Every message is on the audio timeline: back-dated by its analyzer's own delay.
    delays = {name: analyzers()[name](RATE, 2229, 0.02).delay_s for name in ALL}
    assert delays["guitar"] == pytest.approx(0.02) and delays["drums"] == pytest.approx(0.06)
    assert 0.06 <= delays["bass"] <= 0.12 and delays["vocals"] == 0
    log = engine.writer.path / "advanced_analysis" / "instrument_activity.jsonl"
    assert len(log.read_text(encoding="utf-8").splitlines()) == len(found)


def test_vocal_phrases_follow_the_activity_models_judgement_when_there_is_one():
    """In a full band the centre is never empty. A phrase is centre sound while the activity
    model reports a voice: the analysis pass hands each frame's judgement to the analyzer."""
    audio = synth.voice(6.6, [(0.5, 5.2, 57, 60)])  # one long sung note, no gap in it
    size, hop = round(RATE * 0.04644), round(RATE * 0.02)
    window = np.hanning(size).astype(np.float32)[:, None]

    def phrases(present):
        analyzer = analyzers()["vocals"](RATE, size, hop / RATE)
        marks = []
        for start in range(0, len(audio) - size, hop):
            time = start / RATE
            analyzer.present = present(time)
            out = analyzer.step(np.fft.rfft(audio[start : start + size] * window, axis=0))
            if out is not None and out.boundary:
                marks.append((round(time, 1), out.boundary))
        return marks

    assert [kind for _, kind in phrases(lambda t: None)] == ["start", "end"]  # no model
    assert phrases(lambda t: False) == []  # the model says nobody sings: no phrase at all
    # The model reports a voice from 1 to 2.5 s and again from 3.5 s: two phrases.
    kinds = phrases(lambda t: 1.0 <= t < 2.5 or t >= 3.5)
    assert [kind for _, kind in kinds] == ["start", "end", "start", "end"]
    assert kinds[0][0] == pytest.approx(1.1, abs=0.15) and kinds[1][0] == pytest.approx(
        2.75, abs=0.2
    )
