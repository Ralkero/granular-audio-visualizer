"""Live instrument rings: the present state they are drawn from, the bass analyzer's fast
path that feeds it, and the radial picture itself."""

import os
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtGui import QColor

import synth_instruments as synth
from gmv.evaluation import responsiveness as r
from gmv.models import AnalysisConfig
from gmv.visualizer import palette
from gmv.visualizer.live import KEEP_S, Live
from gmv.visualizer.state import VisualState

from test_instrument_layers_phase2 import ALL, activity, play_band
from test_visualizer import frame

RATE = synth.RATE
E1 = 28


# -- the present state -------------------------------------------------------------------


def fed(live, time, **values):
    live.feed(time, time + values.pop("delay", 0.0), **({"level": 0.0, "ring": 0.0} | values))


def test_envelopes_follow_elapsed_time_and_not_the_number_of_frames():
    def run(step):
        live = Live()
        fed(live, 0.0, level=0.0)
        live.advance(0.0)
        for index in range(1, 26):  # the level steps up at 0.02 and back down at 0.3
            time = index * 0.02
            fed(live, time, level=0.8 if time < 0.3 else 0.0, energy=None)
        seen, now = [], 0.0
        while now < 0.5:
            now = round(now + step, 6)
            # Only rows that had arrived by ``now`` are known: replay them in order.
            known = Live()
            for index in range(0, 26):
                time = index * 0.02
                if time <= now:
                    fed(known, time, level=0.8 if 0.02 <= time < 0.3 else 0.0)
            live.time, live._level = known.time, known._level
            live.advance(now)
            seen.append((now, live.level))
        return dict(seen)

    coarse, fine = run(1 / 60), run(1 / 240)
    for now, level in coarse.items():
        nearest = min(fine, key=lambda t: abs(t - now))
        assert level == pytest.approx(fine[nearest], abs=0.08)
    assert max(coarse.values()) > 0.7 and list(coarse.values())[-1] < 0.2  # up fast, down slower
    rising = [level for now, level in coarse.items() if 0.02 < now < 0.12]
    assert rising == sorted(rising)  # no jumps back on the way up


def test_every_attack_is_kept_with_its_age_on_the_instruments_own_clock():
    live = Live()
    fed(live, 0.98, delay=0.06)
    # Two hits 20 ms apart arrive between two display frames: both are there afterwards.
    fed(live, 1.00, delay=0.06, attack=0.9, low=0.9)
    fed(live, 1.02, delay=0.06, attack=0.6, mid=0.6)
    live.advance(1.085)  # the display's present: 5 ms after the newest row arrived
    recent = live.recent(1.0)
    assert [round(hit.strength, 1) for _, hit in recent] == [0.9, 0.6]
    # The analyzer describes a moment 60 ms back; on its own clock the newest hit is new.
    assert live.lag == pytest.approx(0.06) and live.t == pytest.approx(1.025)
    assert recent[1][0] == pytest.approx(0.005) and recent[0][0] == pytest.approx(0.025)
    assert recent[0][1].low == 0.9 and recent[1][1].mid == 0.6
    # They age with audio time and are dropped after a while; the list is bounded.
    live.advance(1.5)
    assert live.recent(1.0)[0][0] == pytest.approx(0.44)
    for index in range(200):
        fed(live, 2.0 + index * 0.02, delay=0.06, attack=0.5)
    assert len(live.hits) == live.hits.maxlen == 64
    live.advance(2.0 + 199 * 0.02 + 0.06 + KEEP_S + 1.0)
    assert not live.hits


def test_it_stops_when_the_rows_stop_and_starts_clean_after_a_hole():
    live = Live(26.0, 56.0)
    for index in range(50):
        fed(live, index * 0.02, level=0.8, ring=0.5, pitch=41.0, attack=0.5 * (index == 40))
    live.advance(1.0)
    assert live.level == pytest.approx(0.8) and live.pitched and live.place == pytest.approx(0.5)
    now = 1.0
    for _ in range(72):  # nothing more arrives for a second
        now += 1 / 72
        live.advance(now)
    assert live.level < 0.02 and not live.pitched and live.ring < 0.05
    fed(live, 5.0, level=0.6)  # the stream resumes four seconds on
    assert not live.hits  # what was before the hole is over
    live.advance(5.0)
    assert live.level == pytest.approx(0.6)  # a jump of the clock starts clean, no long glide
    # The weight is eased, so a presence judgement arriving does not snap the picture.
    live.advance(5.01, weight=1.0)
    live.advance(5.02, weight=0.0)
    assert 0.5 < live.weight < 1.0
    for index in range(60):
        live.advance(5.02 + (index + 1) / 72, weight=0.0)
    assert live.weight < 0.01


def test_the_visual_state_feeds_it_from_the_same_rows_and_replays_the_same():
    def play(delay):
        state, clock = VisualState(), perf_counter()
        for index in range(150):
            time = index * 0.02
            state.consume(frame(time), arrival=clock + time + delay)
            if index >= 3:  # drums describe three frames back, bass four
                state.consume(
                    activity(time - 0.06, "drums", attack=0.8 * (index % 25 == 3), components=(("low", 0.8 * (index % 25 == 3)), ("mid", 0.0), ("high", 0.0)))
                )
            if index >= 4:
                state.consume(
                    activity(time - 0.08, "bass", level=0.6, pitch_midi=33.0, energy=0.7, pulse=0.9 * (index % 25 == 10), lead_s=0.08)
                )
        return state

    state = play(0.03)
    drums, bass = state.live["drums"], state.live["bass"]
    assert drums.lag == pytest.approx(0.06) and bass.lag == pytest.approx(0.08)
    assert len(drums.hits) == 6 and drums.hits[0].low == pytest.approx(0.8)
    assert bass.fast and len(bass.pulses) == 6
    # A pulse is stamped where the analyzer's fast path saw it: not back-dated.
    assert bass.pulses[0][0] == pytest.approx(10 * 0.02)
    now = 149 * 0.02 + 0.01
    for live in (drums, bass):
        live.advance(now)
    assert bass.energy == pytest.approx(0.7) and bass.place == pytest.approx((33 - 26) / 30)
    assert bass.beats(1.0)[-1][0] == pytest.approx(now - 135 * 0.02)
    again = play(1.4)  # the same messages, arriving later and all at once
    for key in ("drums", "bass"):
        again.live[key].advance(now)
        assert again.live[key].hits == state.live[key].hits
        assert again.live[key].pulses == state.live[key].pulses
        assert again.live[key].level == state.live[key].level
    state.reset()
    assert state.live == {}


# -- the bass analyzer's fast path -------------------------------------------------------


def bass_line(times, level=0.4, seconds=None, length=None):
    length = length or 0.9 * (times[1] - times[0])
    return synth.bass(seconds or times[-1] + 1.5, [(t, E1, length) for t in times], level=level)


def analysed(audio):
    rows, _ = r.trace(audio, RATE, AnalysisConfig(instrument_analyzers=("bass",), spectrum_bands=0))
    return rows["bass"]


@pytest.mark.parametrize("per_second", [2, 4, 6])
def test_a_note_played_again_at_the_same_pitch_is_a_pulse_every_time_and_soon(per_second):
    times = [round(float(t), 4) for t in np.arange(1.0, 6.0, 1 / per_second)]
    rows = analysed(bass_line(times))
    pulses = r.match(r.fast(rows), times)
    assert pulses["found"] == len(times) and pulses["extra"] == 0
    assert abs(pulses["placed_s"]["p50"]) <= 0.02 and pulses["placed_s"]["p95"] <= 0.03
    assert pulses["known_s"]["p95"] <= 0.05  # against about 0.1 for the pitch
    # The energy falls back between the notes, so the ring can breathe with them.
    assert r.dips(r.fast(rows), times)["p50"] > 0.3
    # The slower reading is still there and unchanged in kind: the right note, later.
    assert r.pitch_known(rows, [(t, E1) for t in times])["read"] == len(times)


def test_a_held_note_is_one_pulse_and_energy_shows_how_hard_it_is_played():
    rows = analysed(synth.bass(6.0, [(1.0, E1, 4.0)], decay=30.0))
    quick = r.fast(rows)
    assert int((quick[:, r.F["attack"]] > 0).sum()) == 1
    held = quick[(quick[:, 0] > 1.5) & (quick[:, 0] < 4.8), r.F["level"]]
    assert held.min() > 0.8 and held.std() < 0.06
    # Strong notes, then quiet ones: the energy tells them apart, because it is measured
    # against the loudest the low register has been in the last half minute.
    times = [round(float(t), 3) for t in np.arange(1.0, 9.0, 0.5)]
    strong = synth.bass(10.0, [(t, E1, 0.45) for t in times[:8]], level=0.45)
    quiet = synth.bass(10.0, [(t, E1, 0.45) for t in times[8:]], level=0.03)
    quick = r.fast(analysed(strong + quiet))

    def peak(a, b):
        return float(quick[(quick[:, 0] >= a) & (quick[:, 0] < b), r.F["level"]].max())

    assert peak(1.2, 4.8) > 0.9 and peak(1.2, 4.8) - peak(5.2, 8.8) > 0.3
    # And the quiet notes are still pulses.
    assert r.match(quick, times[8:])["found"] >= 7
    # Silence gives neither energy nor pulses.
    silent = r.fast(analysed(np.zeros((RATE * 2, 2), np.float32)))
    assert not len(silent) or (silent[:, r.F["level"]].max() == 0 and silent[:, r.F["attack"]].max() == 0)


# -- the picture -------------------------------------------------------------------------


def coloured(image, colour, step=3):
    """Sampled pixels whose hue is the instrument's and that are clearly lit."""
    hue, found = QColor(colour).hue(), 0
    for y in range(0, image.height(), step):
        for x in range(0, image.width(), step):
            pixel = image.pixelColor(x, y)
            if pixel.value() > 90 and pixel.saturation() > 70:
                found += min(abs(pixel.hue() - hue), 360 - abs(pixel.hue() - hue)) <= 12
    return found


def ready(window, present=ALL):
    canvas = window.canvas
    canvas.info["instruments"] = ALL
    window.set_mode("radial")
    window.set_spectrum_place("hidden")  # only the instruments are in the picture
    window.set_section("timeline", shown=False)  # whose labels have the same colours
    play_band(window, present=present)
    return canvas


def test_live_rings_are_the_default_and_every_instrument_has_its_own_moving_object(window):
    canvas = ready(window)
    assert canvas.radial_live
    # One more hit, just now: the band's last ones are already over.
    window.state.consume(
        activity(window.state.latest.timestamp + 0.02, "drums", attack=0.9, components=(("low", 0.9), ("mid", 0.9), ("high", 0.8)))
    )
    clock = perf_counter()
    canvas.advance(0.014, clock)
    first = window.grab().toImage()
    for key, colour in palette.INSTRUMENTS.items():
        assert coloured(first, colour) > 40, key
    # No history is kept for the rings: nothing was drawn into a sweep.
    assert all(layer.sweep.pixmap is None for layer in canvas.layers.values())
    # Time passing is enough to change the picture: no new row, and still it moves.
    canvas.advance(0.014, clock + 0.014)
    second = window.grab().toImage()
    changed = sum(
        first.pixel(x, y) != second.pixel(x, y)
        for y in range(0, first.height(), 3)
        for x in range(0, first.width(), 3)
    )
    lit = sum(
        QColor(first.pixel(x, y)).value() > 60
        for y in range(0, first.height(), 3)
        for x in range(0, first.width(), 3)
    )
    assert changed > 0.3 * lit  # most of what is drawn moved within one frame
    assert max(canvas.paint_ms) < 250


def test_a_drum_hit_appears_at_once_and_is_gone_again(window):
    canvas = ready(window, present=("drums",))
    state = window.state
    orange = palette.INSTRUMENTS["drums"]
    last = state.latest.timestamp
    base = perf_counter()

    def at(seconds, hit=False):
        """Carry the stream on to ``seconds`` after the band stopped; picture at that time."""
        time = state.latest.timestamp + 0.02
        while time <= last + seconds + 1e-9:
            state.consume(
                frame(time, scores=(("drums", 0.9), ("bass", 0.05), ("vocals", 0.05)), active=("drums",), method="activity_test"),
                arrival=base + (time - last),
            )
            struck = hit and abs(time - 0.06 - (last + 1.0)) < 0.011
            state.consume(
                activity(time - 0.06, "drums", attack=0.9 * struck, flux=0.9 * struck, components=(("low", 0.9 * struck), ("mid", 0.8 * struck), ("high", 0.0)))
            )
            time += 0.02
        canvas.advance(0.014, base + seconds)
        return coloured(window.grab().toImage(), orange)

    quiet = at(0.95)  # the band's own hits have died away
    struck = at(1.09, hit=True)  # the hit is at 1.0 and known 60 ms later
    later = at(1.9)
    assert struck > quiet + 150 and struck > 3 * max(quiet, 1)
    assert later < quiet + 30  # nothing of it is left: it was an impact, not a mark


def test_an_instrument_judged_absent_is_not_drawn_and_the_others_do_not_care(window):
    canvas = ready(window, present=("bass",))
    canvas.advance(0.014, perf_counter() + 0.3)  # let the eased weights settle
    canvas.advance(0.014, perf_counter() + 0.6)
    image = window.grab().toImage()
    assert coloured(image, palette.INSTRUMENTS["bass"]) > 40
    assert coloured(image, palette.INSTRUMENTS["vocals"]) == 0
    assert coloured(image, palette.INSTRUMENTS["guitar"]) == 0
    assert coloured(image, palette.INSTRUMENTS["drums"]) == 0
    assert window.state.live["vocals"].weight < 0.02 and window.state.live["bass"].weight > 0.9


def test_the_history_rings_are_still_there_when_asked_for_and_the_lanes_never_changed(window):
    canvas = ready(window)
    lanes = {key: track.since().copy() for key, track in window.state.activity.items()}
    window.set_radial_rings("history")
    window.grab()
    assert not canvas.radial_live
    assert all(layer.sweep.pixmap is not None for layer in canvas.layers.values())
    window.set_radial_rings("live")
    window.grab()
    assert canvas.radial_live
    window.set_radial_rings("nonsense")
    assert canvas.radial_live
    # Bars keep their strips, and the activity history is what it was.
    window.set_mode("bars")
    window.grab()
    assert all(layer.strip.pixmap is not None for layer in canvas.layers.values())
    assert all(np.array_equal(lanes[key], track.since(), equal_nan=True) for key, track in window.state.activity.items())


def test_the_ring_style_is_remembered(app, tmp_path):
    from gmv.app.window import MainWindow

    first = MainWindow(session_root=tmp_path, remember=False, discover_sources=False)
    try:
        assert first.drawer.remembered()["radial_rings"] == "live"
        first.drawer.radial_rings.setCurrentIndex(first.drawer.radial_rings.findData("history"))
        assert not first.view.canvas.radial_live
        saved = first.drawer.remembered()
    finally:
        first.close()
    second = MainWindow(session_root=tmp_path, remember=False, discover_sources=False)
    try:
        second.drawer.restore(saved)
        assert not second.view.canvas.radial_live
        second.drawer.restore({"source": "system"})  # saved before this existed
        assert second.view.canvas.radial_live
    finally:
        second.close()
