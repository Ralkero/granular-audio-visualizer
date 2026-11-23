"""Radiant's smoke: clouds that drift on their own and that beats push about.

Two kinds of test, as in ``test_radiant_plasma``. Those without a graphics card are about
which events count as beats and what the shader is told. Those on the card (the ``gpu``
fixture: ``plasma_gpu_probe.py``, run once for the session) measure the smoke that is
actually drawn: whether it moves, how, and what a beat does to it.
"""

import math
import os
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from gmv.visualizer.response import BASELINE, EXPRESSIVE
from gmv.visualizer.state import VisualState
from gmv.visualizer.themes import radiant_geometry as geometry
from gmv.visualizer.themes.radiant import Radiant
from gmv.visualizer.themes.radiant_gpu import SHADERS, frame, resources
from gmv.visualizer.themes.radiant_gpu import scene as composition
from gmv.visualizer.themes.radiant_palette import EFFECTS, FACTORY, KEYS, PRESETS, RadiantPalette

from test_live_segments import Stream
from test_radiant import playing, stream
from test_visualizer import event, frame as feature_frame

SIZE = (1280, 720)
SMOKE = ("smoke_primary", "smoke_highlight", "smoke_shadow")
HOP = 0.02


def beat(time, sure=0.7, **more):
    """A beat as the tracker publishes it: a prediction for audio time ``time``."""
    return event(time, "beat_prediction", "predicted", intensity=sure, **more)


def scene(state, now, response=BASELINE, shown=True):
    lay = composition.layout(640.0, 360.0, 300.0, ())
    return Radiant().scene(state, now, lay, values=None, peaks=None, response=response, running=True, beats=shown)


def told(state, now, response=BASELINE, shown=True, colours=None):
    lay = composition.layout(640.0, 360.0, 300.0, ())
    return frame.values(lay, scene(state, now, response, shown), colours or RadiantPalette(), response, size=SIZE, state=state)


def struck(found):
    """The beats a frame's shaders are told of: (age, strength, turn)."""
    return [beat[:3] for beat in found["u_beats"][: found["u_beat_count"]]]


# -- which events are beats (no graphics card) ---------------------------------------------------


def test_only_the_trackers_predicted_beats_count_and_only_once_reached():
    feed = stream()
    state = feed.state
    assert told(state, 3.5)["u_beat_count"] == 0  # the tracker has said nothing: no beat is made up
    # Things that are not beats: onsets, transients, a beat candidate, a delayed context mark.
    state.consume(event(3.0, "onset", "observed", intensity=1.0))
    state.consume(event(3.0, "bass_transient_candidate", "candidate", intensity=1.0))
    state.consume(event(3.0, "beat_candidate", "candidate", intensity=1.0))
    state.consume(event(3.0, "beat_observed", "observed", intensity=1.0))
    state.consume(event(3.0, "beat_prediction", "context", analysis_tier="advanced", intensity=0.7))
    for now in (3.0, 3.2, 3.9):
        assert told(state, now)["u_beat_count"] == 0
    # A prediction is for a time ahead: nothing until audio time gets there.
    state.consume(beat(3.5))
    assert told(state, 3.49)["u_beat_count"] == 0
    found = told(state, 3.5)
    assert struck(found) == [(0.0, 1.0, struck(found)[0][2])] and len(found["u_beats"]) == frame.BEATS == 16
    # It ages by audio time and leaves when its disturbance is over.
    for age in (0.1, 0.9, 2.5):
        assert struck(told(state, 3.5 + age))[0][0] == pytest.approx(age)
    assert told(state, 3.5 + geometry.BEAT_LIFE + 0.01)["u_beat_count"] == 0
    # "Show beats" switches every beat-driven part of the picture, as it always has.
    assert told(state, 3.6, shown=False)["u_beat_count"] == 0 and told(state, 3.6, shown=False)["u_pulse"] == 0.0
    assert told(state, 3.6)["u_pulse"] > 0.5  # the same events drive the pulse and the smoke
    # The beat's turn (where it pushes more, where less) is its own and does not change.
    assert struck(told(state, 3.6))[0][2] == struck(told(state, 4.4))[0][2]
    assert 0.0 <= struck(told(state, 3.6))[0][2] < 1.0


def test_each_beat_is_one_disturbance_with_its_own_time_and_strength():
    feed = stream(seconds=6.0)
    state = feed.state
    times = (4.0, 4.5, 5.0, 5.4, 5.55)  # regular, then closer together
    for index, time in enumerate(times):
        state.consume(beat(time, sure=0.7 if index % 2 == 0 else 0.2))
    state.consume(beat(5.0, sure=0.7))  # the same beat published again
    found = struck(told(state, 5.6))
    assert len(found) == len(times)  # five beats, five disturbances: none doubled, none merged
    assert [age for age, _, _ in found] == pytest.approx(sorted(5.6 - time for time in times))  # newest first
    assert sorted(age for age, _, _ in found) == [age for age, _, _ in found]
    # Strength is the tracker's own confidence (capped by the tracker at 0.7), never nothing.
    strengths = {round(5.6 - age, 2): strength for age, strength, _ in found}
    assert strengths[4.0] == strengths[5.0] == 1.0
    assert strengths[4.5] == pytest.approx(geometry.BEAT_LEAST + (1 - geometry.BEAT_LEAST) * 0.2 / 0.7)
    assert geometry.BEAT_LEAST <= min(strengths.values()) < max(strengths.values()) <= 1.0
    assert len({turn for _, _, turn in found}) == len(times)  # each pushes in a pattern of its own
    # Even beats a round interval apart get turns spread round the whole circle, not a small
    # constant step apart (which a single multiplicative hash gives for any steady tempo).
    for step in (0.5, 0.25, 0.4, 1 / 3, 0.02):
        turns = [frame.own_turn(4.0 + step * n) for n in range(16)]
        assert len(set(turns)) == 16 and all(0.0 <= turn < 1.0 for turn in turns)
        assert len({round(b - a, 3) % 1.0 for a, b in zip(turns, turns[1:], strict=False)}) > 8, step
        assert np.std(turns) > 0.18, step  # about 0.29 for turns spread evenly
    # Presenting the same moment again and again adds nothing.
    for _ in range(5):
        assert struck(told(state, 5.6)) == found
    # More beats than the buffer holds: the newest sixteen, and no more.
    for index in range(40):
        state.consume(beat(5.6 + index * 0.05))
    many = told(state, 7.6)
    assert many["u_beat_count"] == 16 == geometry.MOST_BEATS and len(many["u_beats"]) == 16
    assert max(age for age, _, _ in struck(many)) < 0.8
    assert all(math.isfinite(x) for b in many["u_beats"] for x in b)


def test_expressive_mode_and_colours_change_no_beat():
    feed = stream(seconds=5.0)
    for time in (4.0, 4.4, 4.9):
        feed.state.consume(beat(time, sure=0.5))
    plain, bold = told(feed.state, 5.0, BASELINE), told(feed.state, 5.0, EXPRESSIVE)
    assert struck(bold) == struck(plain) and bold["u_beat_count"] == 3  # same count, times, strengths
    assert bold["u_impact"] > plain["u_impact"] and bold["u_plasma"] > plain["u_plasma"]  # only the drawing
    colours = RadiantPalette()
    for role in SMOKE:
        assert colours.set(role, "#ff1493")
    pink = told(feed.state, 5.0, colours=colours)
    assert struck(pink) == struck(plain) and pink["u_time"] == plain["u_time"]
    assert {k: v for k, v in pink.items() if k.startswith("u_")} == {k: v for k, v in plain.items() if k.startswith("u_")}


def test_a_new_segment_drops_the_old_segments_beats_and_a_pause_changes_nothing():
    feed = Stream()
    for _ in range(100):
        feed.frame(1)
    feed.state.consume(beat(feed.time - 0.3))
    feed.state.consume(beat(feed.time - 0.1))
    assert told(feed.state, feed.time)["u_beat_count"] == 2
    # The audio timeline stands still: the same moment gives the same frame, every time.
    paused = told(feed.state, feed.time)
    for _ in range(4):
        assert told(feed.state, feed.time) == paused
    # It moves on: the beats are older by exactly that much, and nothing else happened.
    assert [age for age, _, _ in struck(told(feed.state, feed.time + 0.5))] == pytest.approx(
        [age + 0.5 for age, _, _ in struck(paused)]
    )
    feed.frame(2)  # a new segment; the timestamps carry on
    after = told(feed.state, feed.time)
    assert after["u_beat_count"] == 0 and after["u_pulse"] == 0.0
    assert all(b == (0.0, 0.0, 0.0, 0.0) for b in after["u_beats"])
    assert after["u_time"] == pytest.approx(feed.time)  # the idle smoke simply goes on from the new time
    feed.state.consume(beat(feed.time + 0.2))  # the new segment's own beats count
    assert told(feed.state, feed.time + 0.25)["u_beat_count"] == 1


def test_the_same_audio_time_gives_the_same_beats_at_any_frame_rate():
    def at_rate(rate, target):
        state = VisualState()
        waiting = []
        for index in range(260):
            time = round(1.0 + index * HOP, 6)
            waiting.append((time, feature_frame(time, segment=1)))
            if index % 25 == 12:  # a prediction, published a quarter of a second ahead
                waiting.append((time, beat(round(time + 0.25, 6), sure=0.3 + 0.1 * (index % 4))))
        tick = 0
        while True:
            now = 1.0 + tick / rate
            while waiting and waiting[0][0] <= now:
                _, message = waiting.pop(0)
                state.consume(message, arrival=1000.0 + message.timestamp if hasattr(message, "window_end") else None)
            if now >= target - 1e-9:
                return told(state, now)
            tick += 1

    for target in (3.0, 4.5, 5.5):  # whole numbers of frames at all five rates
        first = at_rate(60, target)
        assert first["u_beat_count"] >= 4
        for rate in (72, 90, 120, 144):
            other = at_rate(rate, target)
            flat = lambda found: [x for b in found["u_beats"] for x in b]  # noqa: E731
            assert flat(other) == pytest.approx(flat(first), abs=1e-9), (rate, target)
            assert other["u_beat_count"] == first["u_beat_count"] and other["u_time"] == pytest.approx(first["u_time"])


def test_the_smoke_is_told_the_musics_intensity_and_the_beats_lifetime():
    # Musical intensity is relative to the music's own quiet and loud passages: a quiet
    # stretch, then a loud one.
    feed = Stream()
    for index in range(500):
        loud = index >= 300
        feed.time += HOP
        feed.state.consume(
            feature_frame(feed.time, segment=1, loudness_dbfs=-8 if loud else -40, dynamic_intensity=0.9 if loud else 0.15),
            arrival=feed.clock + feed.time,
        )
        if index == 299:
            calm = told(feed.state, feed.time)
    busy = told(feed.state, feed.time)
    assert busy["u_musical"] > calm["u_musical"] + 0.2 and 0.0 <= calm["u_musical"] and busy["u_musical"] <= 1.0
    assert busy["u_musical"] == pytest.approx(feed.state.musical.value)  # the slow whole-mix intensity
    assert calm["u_beat_life"] == busy["u_beat_life"] == geometry.BEAT_LIFE
    quiet = stream(-50, 0.15)
    calm = told(quiet.state, quiet.time)
    assert "u_beat_life" in resources.fragment("smoke") and "u_musical" in resources.fragment("smoke")
    # A quiet passage: nothing struck, and the smoke's clock still runs.
    assert calm["u_beat_count"] == 0 and told(quiet.state, quiet.time + 1.0)["u_time"] == pytest.approx(calm["u_time"] + 1.0)


def test_any_moment_can_be_rebuilt_from_the_messages_and_a_restart_leaves_nothing_behind():
    def messages(until):
        found = []
        for index in range(round((until - 1.0) / HOP) + 1):
            time = round(1.0 + index * HOP, 6)
            found.append((time, feature_frame(time, segment=1)))
            if index % 25 == 12:
                found.append((time, beat(round(time + 0.25, 6), sure=0.3 + 0.1 * (index % 4))))
        return found

    def fed(items):
        state = VisualState()
        for _, message in items:
            state.consume(message, arrival=1000.0 + message.timestamp if hasattr(message, "window_end") else None)
        return state

    everything = messages(6.0)
    played = VisualState()
    for _, message in everything:  # played through, looking at every frame on the way
        played.consume(message, arrival=1000.0 + message.timestamp if hasattr(message, "window_end") else None)
        if hasattr(message, "window_end"):
            told(played, message.timestamp)
    for moment in (3.3, 4.75, 6.0):
        rebuilt = fed([item for item in everything if item[0] <= moment])  # opened straight at that moment
        if moment == 6.0:
            assert told(rebuilt, moment) == told(played, moment)
        assert told(rebuilt, moment)["u_beats"] == told(fed([i for i in everything if i[0] <= moment]), moment)["u_beats"]
        assert told(rebuilt, moment)["u_beat_count"] >= 3
    # The recording is played again from its start: the timeline goes back, the state starts
    # afresh, and none of the first run's beats disturbs the second run's smoke.
    again = played
    for _, message in messages(1.6):
        again.consume(message, arrival=2000.0 + message.timestamp if hasattr(message, "window_end") else None)
    restarted = told(again, 1.6)
    assert restarted["u_beats"] == told(fed(messages(1.6)), 1.6)["u_beats"]
    assert all(age <= 1.6 - 1.0 + 1e-9 for age, _, _ in struck(restarted))


def test_time_stays_usable_through_a_long_session():
    feed = stream()
    view = scene(feed.state, 3.0)
    lay = composition.layout(640.0, 360.0, 300.0, ())
    palette = RadiantPalette()

    def time_at(now):
        view.now = now
        return frame.values(lay, view, palette, BASELINE, size=SIZE)["u_time"]

    assert frame.WRAP_S == 86400.0  # a day: no jump within any session shorter than that
    for hours in (0.5, 1.0, 3.0, 9.0, 23.9):  # across what used to be an hourly wrap, no jump
        now = hours * 3600.0
        assert time_at(now + 0.02) - time_at(now) == pytest.approx(0.02, abs=1e-6)
        assert np.float32(time_at(now + 0.05)) != np.float32(time_at(now))  # still distinct as a 32-bit float
    assert time_at(86400.0 + 5.0) == pytest.approx(5.0)


def test_the_smoke_has_three_colours_of_its_own_saved_and_defaulted(window):
    assert set(SMOKE) < set(KEYS) and set(SMOKE) < set(EFFECTS) and len(KEYS) == 25
    for name, (_, preset) in PRESETS.items():
        assert len({preset[role] for role in SMOKE}) == 3, name  # shadow, body and highlight differ
    found = told(stream().state, 3.0)
    factory = PRESETS[FACTORY][1]
    for role in SMOKE:
        assert found["c_" + role] == frame.linear(factory[role])
        colours = RadiantPalette()
        assert colours.set(role, "#12ab34")
        changed = told(stream().state, 3.0, colours=colours)
        assert {name for name in changed if changed[name] != found[name]} == {"c_" + role}
    # Settings saved before the smoke had colours get the factory ones; damage falls back.
    again = RadiantPalette()
    again.load({"preset": "custom", "colours": {role: "#123456" for role in KEYS if role not in SMOKE}})
    assert all(again[role] == factory[role] for role in SMOKE) and again["plasma_primary"] == "#123456"
    again.load({"colours": {"smoke_primary": "smoke", "smoke_shadow": "#ff00aa"}})
    assert again["smoke_primary"] == factory["smoke_primary"] and again["smoke_shadow"] == "#ff00aa"
    again.restore()
    assert all(again[role] == factory[role] for role in SMOKE)
    # A change is in the very next frame the card is told, with nothing reset.
    canvas = playing(window)
    canvas.gpu = True
    window.grab()
    before = canvas.plasma_values()
    assert window.set_radiant_colour("smoke_primary", "#39ff14")
    after = canvas.plasma_values()
    assert after["c_smoke_primary"] == frame.linear("#39ff14") and after["u_beats"] == before["u_beats"]
    assert after["u_time"] == before["u_time"]


def test_the_smoke_is_a_pass_of_its_own_and_the_field_no_longer_stretches():
    assert "smoke" in SHADERS and SHADERS.index("smoke") == SHADERS.index("plasma_field") + 1
    smoke = resources.fragment("smoke")
    for name in ("c_smoke_primary", "c_smoke_highlight", "c_smoke_shadow", "u_beats", "u_beat_count", "u_time", "u_impact"):
        assert name in smoke, name
    assert "u_beats[16]" in smoke  # a fixed buffer: the newest sixteen
    assert "u_onsets" not in smoke  # onsets do not move it: only beats
    # The old response to an onset, scaling the whole field's coordinates, is gone.
    field = resources.fragment("plasma_field")
    assert "q *= 1.0 - 0.07 * shock" not in field and "u_beats" not in field


def test_the_window_passes_real_beats_through_and_show_beats_gates_them(window):
    canvas = playing(window, onset=False)
    canvas.gpu = True
    state = window.state
    now = canvas.now
    state.consume(beat(now - 0.2, sure=0.6))
    state.consume(beat(now - 0.9, sure=0.3))
    state.consume(beat(now + 0.4, sure=0.7))  # not yet reached
    window.grab()
    found = canvas.plasma_values()
    assert found["u_beat_count"] == 2
    kept = (len(state.markers), state.marked, state.generation)
    window.set_beats(False)
    window.grab()
    assert canvas.plasma_values()["u_beat_count"] == 0  # the smoke only drifts
    assert kept == (len(state.markers), state.marked, state.generation)  # nothing was thrown away
    window.set_beats(True)
    window.grab()
    assert canvas.plasma_values()["u_beat_count"] == 2
    # Without the card there is no smoke at all, and the report does not say there is.
    canvas.gpu = False
    window.grab()
    assert canvas.path == "radiant-cpu" and canvas.plasma_values() is None
    assert "no plasma" in window.renderer_report()["detail"] and window.renderer_report()["backend"] == "cpu"
    began = perf_counter()
    assert window.renderer_report() and perf_counter() - began < 1.0


# -- the smoke that is drawn (on the graphics card) ------------------------------------------------


def test_smoke_is_there_without_any_beat_and_is_not_a_fog_band(gpu):
    assert gpu["smoke light with no beats"] > 15  # visible
    assert gpu["smoke dark share"] > 0.15 and gpu["smoke lit share"] > 0.25  # gaps and clouds, not a ring of fog
    assert gpu["smoke dense share"] > 0.01 and gpu["smoke brightest"] < 250  # dense parts, nothing blown out
    sectors = gpu["smoke light by sector"]
    assert max(sectors) > 1.8 * min(sectors)  # more smoke in some directions than in others
    assert gpu["smoke inside the formations"] == 0.0  # it stays outside
    assert gpu["smoke repeat difference"] == 0.0  # the same moment, the same pixels
    assert gpu["smoke spread"] > 10


def test_intense_music_stirs_the_smoke_more_rather_than_just_brightening_it(gpu):
    # Loud music: more fine structure, still clouds with gaps, and only a little brighter.
    assert gpu["smoke roughness loud"] > 1.25 * gpu["smoke roughness quiet"]
    assert gpu["smoke dark share loud"] > 0.2
    assert 1.0 < gpu["smoke loud"] / gpu["smoke quiet"] < 1.45
    assert gpu["smoke quiet and loud likeness"] > 0.5  # the same clouds, more stirred: not another picture
    # Quiet music: the smoke keeps moving with no beat at all.
    assert gpu["smoke change after 1 s, quiet"] > 4.0


def test_idle_smoke_moves_and_changes_shape_and_has_no_seam(gpu):
    changes = [gpu[f"smoke change after {s} s"] for s in (0.25, 1.0, 4.0, 12.0)]
    assert changes == sorted(changes) and changes[0] > 1.0  # it moves, more over longer
    assert gpu["smoke likeness after 0.25 s"] > 0.95  # smoothly: a quarter of a second on, nearly the same
    assert 0.3 < gpu["smoke likeness after 4.0 s"] < 0.85  # and after four seconds clearly another shape
    # It is not one picture being turned: turned back by its best angle it still does not match.
    assert gpu["smoke likeness of a turned picture turned back"] > 0.99  # the measure finds a turned picture
    assert gpu["smoke best likeness to a turned copy after 4 s"] < 0.85
    # Different parts move by different amounts.
    by_sector = gpu["smoke change by sector after 1 s"]
    assert max(by_sector) > 1.6 * min(by_sector) and min(by_sector) > 0
    # No edge where an angle would wrap: the step across the top is like steps anywhere.
    assert gpu["smoke step across the top seam"] <= 1.2 * gpu["smoke largest step elsewhere"]


def test_one_beat_reaches_the_smoke_displaces_it_unevenly_and_dies_away(gpu):
    # Nothing until the wave gets there; then the smoke is different; then it comes back.
    assert gpu["smoke change 0.0 s after a beat"] == 0.0 and gpu["smoke change before the wave arrives"] == 0.0
    assert gpu["smoke change 0.1 s after a beat"] < 0.5
    peak = gpu["smoke change 0.45 s after a beat"]
    assert gpu["smoke change 0.25 s after a beat"] > 2.0 and peak > 8.0  # there within a quarter of a second
    after = [gpu[f"smoke change {age} s after a beat"] for age in (0.45, 0.9, 1.6, 2.5)]
    assert after == sorted(after, reverse=True) and after[-1] < 0.25 * peak  # it recovers
    assert after[1] > 0.3 * peak  # but not at once: a wake stays behind the wave
    # It is the shape that changes, not the brightness.
    assert abs(gpu["smoke light with a beat passing"] - gpu["smoke light with no beats"]) < 0.25 * gpu["smoke light with no beats"]
    # Unevenly round the circle, and not as one stretch of the whole picture.
    by_sector = gpu["smoke change by sector after a beat"]
    assert max(by_sector) > 1.7 * min(by_sector)
    assert gpu["smoke change a uniform stretch cannot explain"] > 0.7
    # The wave churns it: more fine structure while it passes, back to rest afterwards.
    rest = gpu["smoke roughness at rest"]
    assert gpu["smoke roughness 0.25 s after a beat"] > 1.08 * rest
    assert abs(gpu["smoke roughness 2.5 s after a beat"] - rest) < 0.05 * rest


def test_stronger_beats_and_expressive_mode_move_it_more(gpu):
    assert gpu["smoke change with a strong beat"] > 1.4 * gpu["smoke change with a weak beat"] > 1.0
    assert gpu["smoke change with a beat, expressive"] > 1.2 * gpu["smoke change with a beat, baseline"]


def test_beats_overlap_without_erasing_each_other_and_stay_bounded(gpu):
    both = gpu["smoke change with two beats"]
    assert both > gpu["smoke change with the newer beat alone"] and both > gpu["smoke change with the older beat alone"]
    # Each still shows: the two together differ from either alone.
    assert gpu["smoke two beats differ from the newer alone"] > 2.0 and gpu["smoke two beats differ from the older alone"] > 2.0
    # Sixteen beats at once in loud music: agitated, not torn apart, not white, not escaping.
    assert gpu["smoke brightest with sixteen beats"] <= 255 and gpu["smoke white share with sixteen beats"] < 0.01
    assert gpu["smoke light with sixteen beats"] > 0.4 * gpu["smoke loud"]  # the smoke is still there
    assert gpu["smoke inside the formations with sixteen beats"] == 0.0


def test_regular_beats_react_alike_in_size_but_not_in_shape(gpu):
    sizes = gpu["smoke regular beats sizes"]
    assert max(sizes) < 1.5 * min(sizes) and min(sizes) > 5.0  # consistent reactions
    assert np.mean(gpu["smoke regular beats change-picture likeness"]) < 0.8  # not the same picture
    assert min(gpu["smoke regular beats footprint likeness"]) < 0.75  # nor the same sides of the circle


def test_nothing_jumps_and_the_smoke_never_covers_the_instruments(gpu):
    # A beat leaving the list, or a new one starting while an older wake goes on, changes no
    # pixel by more than a step of rounding.
    assert gpu["smoke step when a beat leaves the list"] <= 2.0
    assert gpu["smoke step when a new beat starts during a wake"] <= 2.0
    assert gpu["smoke over the instruments, no beats"] == 0.0
    assert gpu["smoke over the instruments, sixteen beats, expressive"] == 0.0
    # Sixteen beats in Expressive Mode in loud music: bright, not blown out.
    assert gpu["smoke brightest, sixteen beats, expressive"] < 250
    assert gpu["smoke white share, sixteen beats, expressive"] < 0.005


def test_the_same_audio_time_draws_the_same_pixels_at_any_frame_rate_and_after_rebuilding(gpu):
    # A stream with beats fed as it would arrive at 60, 72, 90, 120 and 144 frames a second:
    # at the same audio times, not one pixel differs, in the smoke or in the whole picture.
    assert gpu["frame rates compared"] == [60, 72, 90, 120, 144]
    assert min(gpu["smoke beats at the compared times"]) >= 2  # there were beats to compare
    assert gpu["smoke largest pixel difference between frame rates"] == 0.0
    assert gpu["whole picture largest pixel difference between frame rates"] == 0.0
    # A state handed the recording up to a moment draws what one that played through draws.
    assert gpu["smoke pixel difference, reconstructed against played through"] == 0.0
    assert gpu["smoke beats, reconstructed and played through"][0] == gpu["smoke beats, reconstructed and played through"][1] > 0


def test_the_smokes_colours_are_its_own_and_change_nothing_else(gpu):
    assert gpu["smoke tint red primary"][0] > 0.95 and gpu["smoke tint green highlight"][1] > 0.95
    assert gpu["smoke tint blue shadow"][2] > 0.95
    for name in ("red primary", "green highlight", "blue shadow"):
        assert gpu[f"smoke light {name}"] > 3, name  # each colours a visible part
    # Shadow and lit smoke are told apart in the same picture.
    assert gpu["smoke lit share of the smoke"] > 0.03 and gpu["smoke shaded share of the smoke"] > 0.03
    # Hot pink stays hot pink where it is bright.
    assert gpu["smoke bright pink share"] > 0.01 and gpu["smoke bright pink saturation"] > 0.6
    assert gpu["smoke shape change when its colour changes"] < 0.05  # the clouds are where they were
    for name in ("plasma_field", "formations", "core"):
        assert gpu[f"{name} change when smoke colour changes"] == 0.0


def test_the_smoke_pass_is_measured_and_cheap_enough(gpu):
    for size in ("1920x1080", "2560x1440"):
        for name in ("no beats", "one beat", "sixteen beats", "sixteen beats, expressive, multicolour"):
            assert gpu[f"smoke pass ms {size}, {name}"] < 6.0, (size, name)
            assert gpu[f"all passes ms with a forced finish {size}, {name}"] > 0
