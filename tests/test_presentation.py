"""Presentation only: the background glow that follows the most prominent instrument, and
Expressive Mode. Neither may change what was analysed."""

import math
import os
from dataclasses import fields
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtGui import QColor

from gmv.visualizer import layers as layer_registry
from gmv.visualizer import palette
from gmv.visualizer.dominance import BLEND_S, FLOOR, HOLD_S, KEYS, MARGIN, MODELS, Dominance
from gmv.visualizer.response import BASELINE, EXPRESSIVE, Response
from gmv.visualizer.state import VisualState

from test_instrument_layers_phase2 import ALL, activity, play_band
from test_live_rings import coloured
from test_live_segments import Stream, play_segment
from test_visualizer import frame

HOP = 0.02
LOW_HEAVY = (0.3, 0.5, 0.05, 0.05, 0.06, 0.03, 0.01)  # most energy below 250 Hz
MID_HEAVY = (0.0, 0.05, 0.15, 0.4, 0.3, 0.07, 0.03)  # most between 250 Hz and 4 kHz
EVEN = (0.1, 0.2, 0.15, 0.25, 0.15, 0.1, 0.05)
PRESENT = dict.fromkeys(KEYS, 1.0)


def listen(dominance, seconds, playing, bands, weights=PRESENT, silent=False, start=None):
    """``playing``: instrument -> (level or energy, hits a second, hit strength)."""
    time = dominance.time + HOP if start is None and dominance.time is not None else (start or 0.0)
    leaders = []
    for index in range(round(seconds / HOP)):
        now = time + index * HOP
        for key, (drive, per_second, strength) in playing.items():
            every = max(1, round(1 / (per_second * HOP))) if per_second else 0
            dominance.row(key, now, drive, strength if every and index % every == 0 else 0.0)
        leaders.append(dominance.step(now, bands, silent, weights))
    return leaders


# -- prominence and the choice of a leader -------------------------------------------------


@pytest.mark.parametrize(
    "key, playing, bands",
    [
        ("bass", (0.85, 2, 0.7), LOW_HEAVY),
        ("vocals", (0.75, 0, 0.0), MID_HEAVY),
        ("guitar", (0.5, 4, 0.8), MID_HEAVY),
        ("drums", (0.5, 6, 0.9), EVEN),
    ],
)
def test_an_instrument_playing_alone_leads_and_gives_the_glow_its_colour(key, playing, bands):
    dominance = Dominance()
    leaders = listen(dominance, 3.0, {key: playing}, bands)
    assert leaders[-1] == key and dominance.scores[key] > FLOOR
    assert dominance.mix[key] > 0.95 and sum(dominance.mix.values()) == pytest.approx(dominance.mix[key])
    assert all(dominance.scores[other] == 0.0 for other in KEYS if other != key)  # they are silent
    assert key in palette.INSTRUMENTS


def test_scores_are_one_scale_not_the_instruments_own_levels():
    # Every instrument reports a high level of its own; the mix says who is heavy.
    everyone = {"bass": (0.9, 2, 0.6), "vocals": (0.9, 0, 0.0), "guitar": (0.9, 3, 0.6), "drums": (0.9, 4, 0.6)}
    low, mid = Dominance(), Dominance()
    assert listen(low, 4.0, everyone, LOW_HEAVY)[-1] == "bass"
    assert listen(mid, 4.0, everyone, MID_HEAVY)[-1] in ("vocals", "guitar")
    assert low.scores["bass"] > mid.scores["bass"] + 0.3
    for scores in (low.scores, mid.scores):
        assert all(0.0 <= value <= 1.0 and math.isfinite(value) for value in scores.values())
    assert set(MODELS) == set(KEYS) == set(palette.INSTRUMENTS)


def test_presence_decides_whether_an_instrument_counts_but_not_by_itself_who_leads():
    playing = {"bass": (0.85, 2, 0.7), "vocals": (0.75, 0, 0.0)}
    dominance = Dominance()
    assert listen(dominance, 3.0, playing, LOW_HEAVY)[-1] == "bass"
    # The model stops reporting bass: it cannot lead however active its analyzer is.
    absent = PRESENT | {"bass": 0.0}
    listen(dominance, 2.0, playing, LOW_HEAVY, weights=absent)
    assert dominance.scores["bass"] == 0.0 and dominance.leader != "bass"
    # A tentative reading counts for less than a confirmed one, and full presence of an
    # instrument that reports nothing counts for nothing.
    tentative = Dominance()
    listen(tentative, 3.0, playing, LOW_HEAVY, weights=PRESENT | {"bass": 0.4})
    sure = Dominance()
    listen(sure, 3.0, playing, LOW_HEAVY)
    assert tentative.scores["bass"] == pytest.approx(0.4 * sure.scores["bass"], rel=1e-6)
    idle = Dominance()
    listen(idle, 2.0, {"bass": (0.0, 0, 0.0)}, LOW_HEAVY)
    assert idle.leader is None and idle.scores["bass"] == 0.0


def choose(dominance, seconds, **scores):
    leaders = []
    for _ in range(round(seconds / HOP)):
        dominance.scores = dict.fromkeys(KEYS, 0.0) | scores
        leaders.append(dominance.choose(HOP))
    return leaders


def test_near_equals_do_not_trade_the_lead_and_a_brief_overtake_does_not_take_it():
    dominance = Dominance()
    choose(dominance, 1.0, bass=0.6, vocals=0.5)
    assert dominance.leader == "bass"
    # Two nearly equal: the other is ahead by less than the margin, back and forth.
    seen = set()
    for index in range(60):
        ahead = 0.6 + (MARGIN - 0.01) * (index % 2)
        seen.update(choose(dominance, HOP, bass=0.6, vocals=ahead))
    assert seen == {"bass"}
    # Clearly ahead, but for a tenth of a second only (one hit): the lead stays.
    assert set(choose(dominance, 0.1, bass=0.6, vocals=0.9)) == {"bass"}
    assert set(choose(dominance, 1.0, bass=0.6, vocals=0.5)) == {"bass"}
    assert dominance.mix["bass"] > 0.9 and dominance.mix["vocals"] < 0.1


def test_a_real_change_takes_the_lead_soon_and_the_colour_follows_smoothly():
    dominance = Dominance()
    choose(dominance, 2.0, bass=0.7, vocals=0.3)
    leaders = choose(dominance, 2.0, bass=0.3, vocals=0.7)
    switched = leaders.index("vocals") * HOP
    assert HOLD_S - 0.03 <= switched <= HOLD_S + 0.05  # not at once, and not late
    assert leaders[-1] == "vocals" and dominance.mix["vocals"] > 0.95 and dominance.mix["bass"] < 0.05
    # The mix moves in small steps, never jumps, and its parts never add up to more than one.
    dominance = Dominance()
    choose(dominance, 2.0, bass=0.7)
    before, largest = dict(dominance.mix), 0.0
    for _ in range(100):
        choose(dominance, HOP, vocals=0.7)
        largest = max(largest, *(abs(dominance.mix[k] - before[k]) for k in KEYS))
        assert sum(dominance.mix.values()) <= 1.0 + 1e-9
        before = dict(dominance.mix)
    assert largest < 0.07  # 20 ms of an easing over BLEND_S
    assert BLEND_S > 0.2


def test_silence_weak_activity_and_stopped_instruments_go_back_to_neutral():
    dominance = Dominance()
    assert listen(dominance, 3.0, {"bass": (0.85, 2, 0.7)}, LOW_HEAVY)[-1] == "bass"
    # Silence: nobody leads after the hold, and the glow has let go of the colour.
    leaders = listen(dominance, 2.5, {"bass": (0.0, 0, 0.0)}, LOW_HEAVY, silent=True)
    assert leaders[-1] is None and leaders.index(None) * HOP <= HOLD_S + 0.05
    assert sum(dominance.mix.values()) < 0.01
    # Rows that simply stop arriving (the instrument's analyzer went quiet, or was reset).
    dominance = Dominance()
    listen(dominance, 3.0, {"bass": (0.85, 2, 0.7)}, LOW_HEAVY)
    leaders = listen(dominance, 3.0, {}, LOW_HEAVY)
    assert leaders[-1] is None and dominance.sustain["bass"] < 0.02
    # Under the floor nobody is chosen at all.
    dominance = Dominance()
    assert set(choose(dominance, 2.0, bass=FLOOR - 0.02, drums=FLOOR - 0.05)) == {None}
    assert sum(dominance.mix.values()) == 0.0


# -- in the visual state: audio time, segments, replay ---------------------------------------


def bass_heavy(stream, segment, frames):
    for index in range(frames):
        stream.time += HOP
        stream.state.consume(
            frame(stream.time, segment=segment, band_energy=LOW_HEAVY), arrival=stream.clock + stream.time
        )
        stream.bass(segment, energy=0.85, level=0.8, pulse=0.7 * (index % 25 == 0), pitch_midi=33.0)


def test_the_state_keeps_it_from_the_messages_alone_and_a_new_segment_starts_it_afresh():
    stream = Stream()
    bass_heavy(stream, 1, 150)
    state = stream.state
    assert state.dominance.leader == "bass" and state.dominance.mix["bass"] > 0.9
    # The same messages, arriving a second and a half later and never drawn: the same result.
    late = Stream()
    late.clock += 1.5
    bass_heavy(late, 1, 150)
    assert late.state.dominance.mix == state.dominance.mix
    assert late.state.dominance.scores == state.dominance.scores
    # A new segment with continuous timestamps: nothing of the old leader is left.
    stream.frame(2)
    assert state.dominance.leader is None and sum(state.dominance.mix.values()) == 0.0
    assert state.dominance.sustain["bass"] == 0.0
    # Segment 2 is about the voice.
    for index in range(150):
        stream.time += HOP
        state.consume(frame(stream.time, segment=2, band_energy=MID_HEAVY), arrival=stream.clock + stream.time)
        stream.vocals(2, level=0.8, pitch_midi=60.0, boundary="start" if index == 0 else "")
    assert state.dominance.leader == "vocals" and state.dominance.mix["bass"] == 0.0
    # A restarted timeline (the full reset) forgets it as well.
    state.reset()
    assert state.dominance.leader is None and state.dominance.time is None


# -- the glow ------------------------------------------------------------------------------


def glow(canvas):
    return {name: (colour, strength) for name, colour, strength, _ in canvas.glow_parts()}


def bass_led(window):
    """A window in the radial view after three seconds of bass-heavy music."""
    stream = Stream(window.state)
    stream.clock = perf_counter() - 4.1
    stream.time = 1.0
    bass_heavy(stream, 1, 150)
    window.set_mode("radial")
    window.canvas.advance(0.014, perf_counter())
    return window.canvas


def test_the_radial_glow_takes_the_leaders_colour_at_the_strength_it_would_have_had(window):
    canvas = bass_led(window)
    assert canvas.reactive_background and window.state.dominance.leader == "bass"
    parts = glow(canvas)
    colour, strength = parts["glow bass"]
    assert colour == palette.INSTRUMENTS["bass"] and strength > 0.03
    window.set_reactive_background(False)
    neutral = glow(canvas)
    assert set(neutral) == {"glow", "beat glow"} and neutral["glow"][0] == palette.SPECTRUM[1]
    # Only the colour changed hands: the parts add up to the neutral glow's strength.
    total = sum(strength for name, (_, strength) in parts.items() if name.startswith("glow"))
    assert total == pytest.approx(neutral["glow"][1], rel=1e-6)
    assert all(math.isfinite(s) and 0.0 <= s <= 1.0 for _, s in parts.values())
    # In the picture: the background beside the rings is bluer than it is red with it on.
    def beside(image):
        pixel = QColor(image.pixel(int(image.width() * 0.5), int(image.height() * 0.03)))
        return pixel.blue() - pixel.red()

    off = beside(window.grab().toImage())
    window.set_reactive_background(True)
    assert beside(window.grab().toImage()) > off
    # Spectrum Bars has no instrument glow; a stopped stream goes back to neutral.
    window.set_mode("bars")
    assert set(glow(canvas)) == {"glow", "beat glow"}
    window.set_mode("radial")
    window.state.status = "stopped"
    canvas.advance(0.014, perf_counter())
    assert set(glow(canvas)) == {"glow", "beat glow"}


def test_hiding_a_layer_or_not_painting_does_not_stop_the_leader_being_followed(window):
    window.set_layer("bass", enabled=False)  # its ring is not drawn
    canvas = bass_led(window)
    assert window.state.dominance.leader == "bass" and "glow bass" in glow(canvas)
    # Layer colours are not touched by any of this.
    assert palette.INSTRUMENTS["bass"] == "#4cc9f0" and canvas.layers["bass"].profile.colour == "#4cc9f0"


# -- Expressive Mode -------------------------------------------------------------------------


def test_the_baseline_is_all_ones_and_expressive_is_bounded():
    for field in fields(Response):
        if field.name != "name":
            assert getattr(BASELINE, field.name) == 1.0
            value = getattr(EXPRESSIVE, field.name)
            assert math.isfinite(value) and 0.5 <= value <= 2.0
    values = np.linspace(0.0, 1.0, 48)
    assert BASELINE.lifted(values) is values  # untouched, not merely equal
    lifted = EXPRESSIVE.lifted(values)
    assert lifted[0] == 0.0 and lifted[-1] == 1.0 and np.all(lifted >= values) and lifted.max() <= 1.0
    assert np.all(np.diff(lifted) > 0)  # the same bars are the tallest: order is kept
    assert EXPRESSIVE.motion > 1.2 and EXPRESSIVE.glow > 1.2 and EXPRESSIVE.spectrum_curve < 1.0


def stage(response, inner=250.0, outer=330.0):
    return layer_registry.Stage(
        2.0, 6.0, 1.0, 0, None, ring=(400.0, 400.0, inner, outer), live=True, response=response
    )


def test_the_bass_ring_is_larger_in_expressive_mode_with_the_same_phase_and_inside_its_ring():
    state = VisualState()
    stream = Stream(state)
    bass_heavy(stream, 1, 150)
    live = state.live["bass"]
    live.advance(stream.time + 0.005, weight=1.0)
    layer = layer_registry.build()["bass"]
    plain = layer.live_shape(stage(BASELINE), live)
    phase = live.phase(0.5, 3.0)
    bold = layer.live_shape(stage(EXPRESSIVE), live)
    assert live.phase(0.5, 3.0) == phase  # drawing differently moved no clock
    thick = lambda shape: float(np.mean(shape[0] - shape[1]))  # noqa: E731
    assert thick(bold) > 1.25 * thick(plain)
    for shape in (plain, bold):
        assert np.all(np.isfinite(shape[0])) and np.all(np.isfinite(shape[1]))
        assert shape[0].max() <= 330.0 and shape[1].min() >= 250.0 and np.all(shape[0] >= shape[1])
    # Even at the layer's own highest intensity nothing leaves the ring.
    layer.intensity = 2.0
    wild = layer.live_shape(stage(EXPRESSIVE), live)
    assert wild[0].max() <= 330.0 and wild[1].min() >= 250.0


def lit(image):
    return sum(
        QColor(image.pixel(x, y)).value() > 110
        for y in range(0, image.height(), 3)
        for x in range(0, image.width(), 3)
    )


@pytest.mark.parametrize("mode", ["radial", "bars"])
def test_expressive_mode_draws_more_from_the_same_state_and_off_is_the_picture_as_it_was(window, mode):
    canvas = window.canvas
    canvas.info["instruments"] = ALL
    window.set_mode(mode)
    window.set_spectrum_place("beside")
    play_band(window)
    window.state.consume(
        activity(window.state.latest.timestamp + 0.02, "drums", attack=0.9, components=(("low", 0.9), ("mid", 0.9), ("high", 0.8)))
    )
    canvas.advance(0.014, perf_counter())
    snapshot = lambda: (  # noqa: E731  what the analysis produced, as the state holds it
        {key: (list(live.hits), list(live.pulses), live.time, live.origin) for key, live in window.state.live.items()},
        dict(window.state.dominance.scores),
        dict(window.state.dominance.mix),
        {key: track.since().tobytes() for key, track in window.state.activity.items()},
    )
    plain = window.grab().toImage()
    before = snapshot()
    assert canvas.response is BASELINE
    window.set_expressive(True)
    bold = window.grab().toImage()
    assert canvas.response is EXPRESSIVE and bold != plain
    assert lit(bold) > 1.05 * lit(plain)  # measurably more, not just a flag
    for colour in palette.INSTRUMENTS.values():  # every instrument is still there, in its colour
        assert coloured(bold, colour) >= 0.8 * coloured(plain, colour) > 0
    assert snapshot() == before  # nothing analysed was touched
    window.set_expressive(False)
    assert window.grab().toImage() == plain  # exactly the picture it was
    assert max(canvas.paint_ms) < 250


def test_the_two_settings_work_alone_and_together(window):
    canvas = bass_led(window)
    results = {}
    for reactive in (False, True):
        for expressive in (False, True):
            window.set_reactive_background(reactive)
            window.set_expressive(expressive)
            results[reactive, expressive] = (
                glow(canvas),
                dict(window.state.dominance.scores),
                window.state.dominance.leader,
            )
    # Who leads, and by how much, is the same whatever is switched on.
    assert len({str(scores) for _, scores, _ in results.values()}) == 1
    assert {leader for _, _, leader in results.values()} == {"bass"}
    for expressive in (False, True):
        assert set(results[False, expressive][0]) == {"glow", "beat glow"}
        assert results[True, expressive][0]["glow bass"][0] == palette.INSTRUMENTS["bass"]
    # Expressive Mode scales the strength of the glow and leaves its colour alone.
    for reactive, name in ((False, "glow"), (True, "glow bass")):
        plain, bold = results[reactive, False][0][name], results[reactive, True][0][name]
        assert bold[0] == plain[0] and bold[1] == pytest.approx(min(1.0, plain[1] * EXPRESSIVE.glow))


def test_both_settings_are_remembered_and_old_settings_get_the_defaults(app, tmp_path):
    from gmv.app.window import MainWindow

    first = MainWindow(session_root=tmp_path, remember=False, discover_sources=False)
    try:
        saved = first.drawer.remembered()
        assert saved["reactive_background"] is True and saved["expressive"] is False
        assert first.view.canvas.reactive_background and first.view.canvas.response is BASELINE
        # The quick button and the setting are one switch; both act at once.
        first.expressive_button.setChecked(True)
        assert first.drawer.expressive.isChecked() and first.view.canvas.response is EXPRESSIVE
        first.drawer.expressive.setChecked(False)
        assert not first.expressive_button.isChecked() and first.view.canvas.response is BASELINE
        first.drawer.expressive.setChecked(True)
        first.drawer.reactive_background.setChecked(False)
        assert not first.view.canvas.reactive_background
        saved = first.drawer.remembered()
        assert saved["expressive"] is True and saved["reactive_background"] is False
    finally:
        first.close()
    second = MainWindow(session_root=tmp_path, remember=False, discover_sources=False)
    try:
        second.drawer.restore(saved)
        assert second.view.canvas.response is EXPRESSIVE and second.expressive_button.isChecked()
        assert not second.view.canvas.reactive_background
        old = {"source": "system", "mode": "bars", "intensity_colour": False}  # saved before these
        second.drawer.restore(old)
        assert second.view.canvas.response is BASELINE and second.view.canvas.reactive_background
        assert second.view.canvas.mode == "bars" and not second.view.canvas.colour_by_intensity
    finally:
        second.close()


def test_expressive_mode_changes_no_segment_or_event_bookkeeping():
    # The same stream into two states, one of them drawn expressively: identical states.
    def run():
        stream = Stream()
        play_segment(stream, 1, 80)
        play_segment(stream, 2, 80)
        return stream.state

    one, two = run(), run()
    for key in one.live:
        a, b = one.live[key], two.live[key]
        for live in (a, b):
            live.advance(one.latest.timestamp + 0.005)
        assert (list(a.hits), list(a.pulses), a.segment, a.origin) == (list(b.hits), list(b.pulses), b.segment, b.origin)
        assert a.phase(0.5, 3.0) == b.phase(0.5, 3.0)
    assert one.dominance.scores == two.dominance.scores and one.live_epoch == two.live_epoch
