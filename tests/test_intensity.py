"""The intensity of the music as a whole, and the spectrum colouring that follows it."""

import os
from time import perf_counter
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtGui import QColor

from gmv.visualizer import intensity, palette
from gmv.visualizer.intensity import MusicalIntensity, tint
from gmv.visualizer.state import VisualState
from gmv.visualizer.window import HEAT_STEPS, spectrum_color, spectrum_colors

from test_instrument_layers_phase2 import play_band
from test_visualizer import frame

FULL = (0.3, 0.5, 0.05, 0.05, 0.06, 0.03, 0.01)  # a full band: most energy below 250 Hz
THIN = (0.0, 0.06, 0.13, 0.36, 0.38, 0.05, 0.02)  # a lone guitar or a voice


def passage(tracker, start, seconds, level, bands=THIN, density=3.0, hop=0.02, **changes):
    """Feed a steady passage; returns the time it ends and every value on the way."""
    values = []
    for index in range(round(seconds / hop)):
        values.append(
            tracker.step(
                frame(
                    start + index * hop,
                    loudness_dbfs=level,
                    band_energy=bands,
                    rhythmic_density=density,
                    **changes,
                )
            )
        )
    return start + seconds, values


def song(tracker, offset=0.0, hop=0.02):
    """A quiet opening, the full band, a break, the band again. Values at the end of each."""
    time, ends = 0.0, []
    for seconds, level, bands in ((12, -40, THIN), (20, -24, FULL), (8, -36, THIN), (15, -24, FULL)):
        time, values = passage(tracker, time, seconds, level + offset, bands, hop=hop)
        ends.append(values[-1])
    return ends


def test_a_quiet_passage_is_low_a_full_one_high_and_the_scale_is_zero_to_one():
    opening, band, pause, again = song(MusicalIntensity())
    assert opening < 0.35 and band > 0.85 and again > 0.85
    assert opening < pause < band - 0.2  # a break is told from the band round it
    assert all(0.0 <= value <= 1.0 for value in (opening, band, pause, again))
    # The colouring: none at all in the opening, complete with the full band.
    assert tint(opening) < 0.02 and tint(band) > 0.95 and tint(0.0) == 0.0 and tint(1.0) == 1.0


def test_it_does_not_depend_on_the_playback_volume():
    assert song(MusicalIntensity(), offset=-18.0) == pytest.approx(song(MusicalIntensity()), abs=1e-6)


def test_a_recording_that_is_loud_throughout_does_not_stay_at_the_top():
    # No contrast has been heard: the level alone says little, so it sits in the middle and
    # the spectrum is not left fully red for the whole track.
    tracker = MusicalIntensity()
    _, values = passage(tracker, 0.0, 240.0, -8.0, FULL, density=5.0)
    assert 0.45 < values[-1] < 0.75 and tint(max(values)) < 0.75
    # A thin, quiet recording with nothing to compare it with stays neutral.
    _, values = passage(MusicalIntensity(), 0.0, 60.0, -30.0, THIN, density=2.0)
    assert tint(max(values)) < 0.02


def test_a_fade_in_is_not_taken_for_a_quiet_passage():
    tracker, time = MusicalIntensity(), 0.0
    for level in range(-64, -40, 2):  # two and a half seconds from nothing to the opening
        time, _ = passage(tracker, time, 0.2, level)
    time, values = passage(tracker, time, 10.0, -40.0)
    assert values[-1] < 0.4 and tint(values[-1]) < 0.1
    _, values = passage(tracker, time, 10.0, -24.0, FULL)
    assert values[-1] > 0.85


def test_a_single_hit_barely_moves_it_and_a_sustained_change_does():
    tracker = MusicalIntensity()
    time, values = passage(tracker, 0.0, 20.0, -40.0)
    before = values[-1]
    time, _ = passage(tracker, time, 0.06, -20.0, FULL)  # one drum hit, 20 dB up, 60 ms
    time, values = passage(tracker, time, 3.0, -40.0)
    assert max(values) - before < 0.08
    _, values = passage(tracker, time, 4.0, -20.0, FULL)
    assert values[-1] - before > 0.4


def test_it_rises_promptly_falls_slowly_and_moves_without_jumps():
    tracker = MusicalIntensity()
    time, values = passage(tracker, 0.0, 15.0, -40.0)
    low = values[-1]
    time, rise = passage(tracker, time, 15.0, -22.0, FULL)
    high = rise[-1]
    time, fall = passage(tracker, time, 15.0, -40.0)

    def seconds_to_cross(values, level, rising):
        return next(n for n, v in enumerate(values) if (v > level) == rising) * 0.02

    middle = (low + high) / 2
    up, down = seconds_to_cross(rise, middle, True), seconds_to_cross(fall, middle, False)
    assert 0.3 < up < 2.0 and down > 1.5 * up
    steps = [abs(b - a) for a, b in zip(rise + fall, (rise + fall)[1:], strict=False)]
    assert max(steps) < 0.03  # no visible jump from one 20 ms frame to the next
    # Silence lets go of it completely, within seconds.
    _, quiet = passage(tracker, time, 8.0, -90.0, silence=True)
    assert quiet[-1] < 0.01 and tracker.raw == 0.0


def test_the_smoothing_follows_time_not_the_number_of_frames():
    slow, fast = MusicalIntensity(), MusicalIntensity()
    assert song(fast, hop=0.01) == pytest.approx(song(slow, hop=0.02), abs=0.02)
    assert song(MusicalIntensity(), hop=0.04) == pytest.approx(song(MusicalIntensity()), abs=0.03)


def test_bad_input_a_restart_and_reset_are_harmless():
    tracker = MusicalIntensity()
    time, _ = passage(tracker, 0.0, 5.0, -24.0, FULL)
    for level, bands, density in (
        (float("nan"), FULL, 3.0),
        (float("-inf"), (), 3.0),
        (-24.0, (0.0,) * 7, float("nan")),
        (-24.0, None, None),
    ):
        value = tracker.step(  # the message model would refuse these; the tracker must not mind
            SimpleNamespace(
                timestamp=time,
                loudness_dbfs=level,
                band_energy=bands,
                rhythmic_density=density,
                silence=False,
            )
        )
        time += 0.02
        assert 0.0 <= value <= 1.0
    assert tracker.step(frame(time + 0.02, loudness_dbfs=-24.0, band_energy=FULL)) > 0.3
    tracker.step(frame(0.0, loudness_dbfs=-40.0, band_energy=THIN))  # the timeline restarted
    assert tracker.value == 0.0 or tracker.heard < 0.1
    tracker.reset()
    assert tracker.value == 0.0 and tracker.loud is None and tracker.time is None


def test_the_same_frames_give_the_same_intensity_whenever_they_arrive():
    def play(delay, jitter):
        state = VisualState()
        clock, seen = perf_counter(), []
        for index in range(1500):
            time = index * 0.02
            loud = 20 <= time % 30 < 28
            state.consume(
                frame(
                    time,
                    loudness_dbfs=-24.0 if loud else -40.0,
                    band_energy=FULL if loud else THIN,
                    rhythmic_density=4.0,
                ),
                arrival=clock + time + delay + jitter * (index % 7) / 7,
            )
            seen.append(state.musical.value)
        return seen

    live, replayed = play(0.03, 0.0), play(1.5, 0.2)
    assert live == replayed and max(live) > 0.8 and min(live[200:900]) < 0.35
    # It is part of the state every mode draws from, and goes when the history does.
    state = VisualState()
    state.consume(frame(0.0, loudness_dbfs=-20.0, band_energy=FULL), arrival=perf_counter())
    assert state.musical.value > 0.0
    state.reset()
    assert state.musical.value == 0.0


def test_the_estimate_is_cheap():
    tracker, frames = MusicalIntensity(), [frame(n * 0.02, band_energy=FULL) for n in range(5000)]
    began = perf_counter()
    for one in frames:
        tracker.step(one)
    each = (perf_counter() - began) / len(frames)
    assert each < 100e-6  # 50 frames a second: well under a thousandth of the time


def hue_is_red(colour):
    return colour.red() > 1.8 * colour.green() and colour.red() > 1.5 * colour.blue()


def test_the_colours_run_from_exactly_the_neutral_palette_to_a_dark_red():
    neutral = spectrum_colors(48)
    assert neutral == [spectrum_color(i / 47) for i in range(48)]
    assert neutral[0] == QColor(palette.SPECTRUM[0]) and neutral[-1] == QColor(palette.SPECTRUM[2])
    assert spectrum_colors(48, 0.0) == neutral
    full = spectrum_colors(48, 1.0)
    assert full[0] == QColor(palette.INTENSE[0]) and full[-1] == QColor(palette.INTENSE[2])
    # Dark and rich: red by a wide margin, never bright, never towards orange (green stays
    # under blue), and nowhere near the drums' orange.
    drums = QColor(palette.INSTRUMENTS["drums"])
    for colour in full:
        assert hue_is_red(colour) and colour.red() <= 190 and colour.green() < colour.blue()
        assert colour.lightnessF() < 0.5 and drums.green() - colour.green() > 80
    # In between every bar has gone the same share of the way: one response, not per bar.
    half = spectrum_colors(48, 0.5)
    for a, b, c in zip(neutral, half, full, strict=True):
        assert abs(b.red() - (a.red() + c.red()) / 2) <= 1
        assert abs(b.blue() - (a.blue() + c.blue()) / 2) <= 1
    # The instruments' own colours are what they were.
    assert palette.INSTRUMENTS == {
        "vocals": "#f472b6",
        "guitar": "#f5c451",
        "bass": "#4cc9f0",
        "drums": "#ff8a5b",
    }
    assert palette.PREDICTED == "#a3e635" and palette.SPECTRUM == ("#5b62a8", "#8f8fd6", "#cfd0f2")


def reddish(image):
    """Pixels that are clearly red, of a coarse sample of the picture."""
    found = 0
    for y in range(0, image.height(), 6):
        for x in range(0, image.width(), 6):
            colour = image.pixelColor(x, y)
            found += colour.red() > 90 and hue_is_red(colour)
    return found


def loud_frames(view, level=-20.0, bands=FULL, seconds=6.0):
    start = view.state.latest.timestamp + 0.02 if view.state.latest is not None else 0.0
    base = perf_counter() - seconds
    for index in range(round(seconds / 0.02)):
        time = start + index * 0.02
        view.state.consume(
            frame(time, loudness_dbfs=level, band_energy=bands, rhythmic_density=4.0),
            arrival=base + index * 0.02,
        )
    for _ in range(40):
        view.canvas.advance(0.02, perf_counter())


@pytest.mark.parametrize("mode", ["radial", "bars"])
def test_the_spectrum_reddens_with_the_music_and_can_be_told_not_to(window, mode):
    canvas = window.canvas
    window.set_mode(mode)
    for key in canvas.layers:
        window.set_layer(key, enabled=False)  # it needs no instrument layer to work
    loud_frames(window, level=-44.0, bands=THIN)
    assert canvas.heat < 0.02 and canvas.heat_step == 0
    quiet = window.grab().toImage()
    assert reddish(quiet) == 0  # a quiet passage is drawn in the neutral palette, exactly
    loud_frames(window)
    assert window.state.musical.value > 0.8 and canvas.heat > 0.9
    loud = window.grab().toImage()
    assert reddish(loud) > 200
    # One step for the whole spectrum, and only so many distinct sets of colours.
    assert canvas.heat_step == round(canvas.heat * HEAT_STEPS) <= HEAT_STEPS
    assert len(canvas._colors) <= 2 * HEAT_STEPS + 3
    # Switched off: the intensity is still known, the picture is the neutral one.
    window.set_intensity_colour(False)
    for _ in range(60):
        canvas.advance(0.02, perf_counter())
    assert canvas.heat < 0.01 and window.state.musical.value > 0.8
    assert reddish(window.grab().toImage()) == 0
    window.set_intensity_colour(True)
    for _ in range(60):
        canvas.advance(0.02, perf_counter())
    assert canvas.heat > 0.9
    # Stopping the stream lets the colour go as well.
    window.state.status = "stopped"
    for _ in range(60):
        canvas.advance(0.02, perf_counter())
    assert canvas.heat < 0.01


@pytest.mark.parametrize("mode", ["radial", "bars"])
def test_instrument_layers_are_drawn_the_same_whatever_the_intensity(window, mode):
    canvas = window.canvas
    window.set_mode(mode)
    window.set_spectrum_place("hidden")  # only the instruments are in the picture
    play_band(window)
    window.tick()
    pictures = []
    for heat in (0.0, 1.0):
        canvas.heat = heat
        pictures.append(window.grab().toImage())
    assert pictures[0] == pictures[1]
    # With the spectrum in the picture the same change of intensity does show.
    window.set_spectrum_place("beside")
    window.grab()
    with_spectrum = []
    for heat in (0.0, 1.0):
        canvas.heat = heat
        with_spectrum.append(window.grab().toImage())
    assert with_spectrum[0] != with_spectrum[1]
    assert reddish(with_spectrum[1]) > reddish(with_spectrum[0])


def test_the_setting_is_remembered_and_is_on_unless_saved_otherwise(app, tmp_path):
    from gmv.app.window import MainWindow

    first = MainWindow(session_root=tmp_path, remember=False, discover_sources=False)
    try:
        assert first.drawer.intensity_colour.isChecked() and first.view.canvas.colour_by_intensity
        assert first.drawer.remembered()["intensity_colour"] is True
        first.drawer.intensity_colour.setChecked(False)
        assert not first.view.canvas.colour_by_intensity
        saved = first.drawer.remembered()
        assert saved["intensity_colour"] is False
    finally:
        first.close()
    second = MainWindow(session_root=tmp_path, remember=False, discover_sources=False)
    try:
        second.drawer.restore(saved)
        assert not second.view.canvas.colour_by_intensity
        second.drawer.restore({"source": "system"})  # saved before this existed
        assert second.view.canvas.colour_by_intensity
    finally:
        second.close()


def test_the_module_says_what_it_measures():
    assert sum(intensity.WEIGHTS) == pytest.approx(1.0)
    assert intensity.ATTACK_S < intensity.RELEASE_S and intensity.SPAN_DB[0] < intensity.SPAN_DB[1]
