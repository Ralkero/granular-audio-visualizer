"""GPU Radiant: a renderer of its own. Who draws what, what the shaders are told, what they
draw, and what happens without them.

Two kinds of test. The first needs no graphics card: who owns the central view, the shader
sources, the numbers a frame's shaders are given (``radiant_gpu.frame`` and ``scene``), the
fall-back, the settings. They always run. The second kind runs the shaders:
``plasma_gpu_probe.py`` is started once in a process of its own (these tests use Qt's
``offscreen`` platform, which has no OpenGL on Windows) and measures the pictures. Where
no OpenGL 3.3 context can be made those tests are skipped by name, with the reason;
``test_the_probe_ran_or_says_why`` never skips.
"""

import math
import os
from dataclasses import fields
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage, QRegion
from PySide6.QtWidgets import QWidget

from gmv.visualizer import layers as layer_registry
from gmv.visualizer import palette
from gmv.visualizer.layers.bass import FLOW_BASE, FLOW_PER_ENERGY
from gmv.visualizer.response import BASELINE, EXPRESSIVE, Response
from gmv.visualizer.state import VisualState
from gmv.visualizer.themes import radiant_geometry as geometry
from gmv.visualizer.themes.radiant import Radiant
from gmv.visualizer.themes.radiant_gpu import POST, SHADERS, frame, resources
from gmv.visualizer.themes.radiant_gpu import scene as composition
from gmv.visualizer.themes.radiant_gpu.resources import ShaderError
from gmv.visualizer.themes.radiant_palette import EFFECTS, FACTORY, KEYS, PRESETS, RadiantPalette

from test_live_segments import Stream, play_segment
from test_radiant import VALUES, messages, picture, playing, stream
from test_visualizer import event, frame as feature_frame

SIZE = (1280, 720)
FOUR = ("vocals", "guitar", "bass", "drums")
HOP = 0.02


def plan(keys=FOUR):
    return composition.layout(640.0, 360.0, 300.0, keys)


def scene(state, now, response=BASELINE, keys=(), beats=True, values=VALUES):
    """The scene in GPU Radiant's own composition (the live states must be at ``now``)."""
    return Radiant().scene(
        state, now, plan(keys), values=values, peaks=None, response=response, running=True,
        beats=beats, keys=keys,
    )  # fmt: skip


def told(view, colours=None, response=BASELINE, spectrum=VALUES, keys=FOUR, **more):
    return frame.values(plan(keys), view, colours or RadiantPalette(), response, size=SIZE, spectrum=spectrum, keys=keys, **more)


def numbers(found):
    """Every number in a frame's values, flat."""
    for value in found.values():
        if isinstance(value, list):
            for item in value:
                yield from (item if isinstance(item, tuple) else (item,))
        elif isinstance(value, tuple):
            yield from value
        else:
            yield value


def band_playing(frames=138, segment=1):
    """A stream of bass, drums and a voice, brought to its present."""
    feed = Stream()
    play_segment(feed, segment, frames)
    composition.advance(feed.state, feed.time, FOUR)
    return feed


def canvas_layer(window):
    """What the canvas itself draws, as (alpha, rgb) arrays: laid over whatever is behind."""
    image = QImage(window.width(), window.height(), QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    window.grab()
    window.canvas.render(image, QPoint(), QRegion(), QWidget.RenderFlag.DrawChildren)
    data = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(image.height(), image.bytesPerLine() // 4, 4)
    return data[:, : image.width(), 3].astype(int), data[:, : image.width(), :3].astype(int)


# -- who draws the central view ------------------------------------------------------------------


def test_gpu_radiant_owns_the_central_view_and_the_canvas_draws_none_of_it(window, monkeypatch):
    canvas = playing(window)
    calls = {"stage": 0, "radial": 0, "layers": 0, "paint": 0, "static": 0, "glow": 0, "present": 0}

    def counted(name, original):
        def wrapper(*args, **kwargs):
            calls[name] += 1
            return original(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(layer_registry.Layer, "paint_stage", counted("stage", layer_registry.Layer.paint_stage))
    for name, method in (("radial", "draw_radial"), ("layers", "draw_layers"), ("glow", "draw_glow")):
        monkeypatch.setattr(canvas, method, counted(name, getattr(canvas, method)))
    radiant = canvas.radiant
    for name, method in (("paint", "paint"), ("static", "paint_static"), ("present", "present")):
        monkeypatch.setattr(radiant, method, counted(name, getattr(radiant, method)))

    canvas.gpu = True  # as while the graphics card is drawing Radiant
    painted = radiant.painted
    alpha, _ = canvas_layer(window)
    assert canvas.path == "radiant-gpu"
    # Not one phase 1 routine ran: no QPainter Radiant, no Classic radial, no instrument layer.
    assert calls["stage"] == calls["radial"] == calls["layers"] == calls["paint"] == calls["static"] == calls["glow"] == 0
    assert calls["present"] >= 1 and radiant.painted == painted
    # The whole area of the visualizer is clear: the card's picture is all there is in it.
    _lay, _view, rect, keys, _spectrum = radiant.shown
    area = alpha[round(rect.top()) + 2 : round(rect.bottom()) - 2, round(rect.left()) + 2 : round(rect.right()) - 2]
    assert area.size > 200_000 and area.max() == 0
    assert keys == tuple(layer.key for layer in canvas.active_layers()) and len(keys) == 4
    # The application's shell is still drawn by the canvas: the header, the timeline.
    assert alpha[: round(rect.top()) - 2].max() > 100 and alpha[round(rect.bottom()) + 20 :].max() > 100
    # And the card is told a complete frame: every structure, every instrument.
    found = canvas.plasma_values()
    assert found["u_size"] == (float(canvas.width()), float(canvas.height()))
    assert found["u_area"] == (rect.top() / canvas.height(), rect.bottom() / canvas.height())
    for name in ("u_spectrum", "u_wave", "u_segments", "u_vocals", "u_guitar", "u_bass", "u_drums", "u_core", "u_rim"):
        assert name in found, name
    assert found["u_bass"][0] > 0 and found["u_vocals"][0] > 0 and found["u_guitar"][0] > 0


def test_each_mode_and_theme_has_its_own_drawing_path_and_none_is_left_behind(window, monkeypatch):
    canvas = playing(window, theme="classic")
    assert canvas.path == "classic" and canvas.plasma_values() is None
    classic = picture(window)
    stages = []
    original = layer_registry.Layer.paint_stage
    monkeypatch.setattr(layer_registry.Layer, "paint_stage", lambda self, *a: (stages.append(self.key), original(self, *a))[1])
    window.grab()
    assert set(stages) == set(FOUR)  # Classic draws the instruments with their own layers
    window.set_mode("bars")
    window.grab()
    assert canvas.path == "bars"
    bars = picture(window)
    window.set_mode("radial")
    window.set_radial_theme("radiant")
    window.grab()
    assert canvas.path == "radiant-cpu"  # no OpenGL here: the reduced QPainter Radiant, and it says so
    canvas.gpu = True
    stages.clear()
    window.grab()
    assert canvas.path == "radiant-gpu" and stages == []
    canvas.gpu = False
    # Back and forth: Classic and the bars come back exactly, nothing of Radiant under them.
    for _ in range(3):
        window.set_radial_theme("classic")
        assert np.array_equal(picture(window), classic) and canvas.path == "classic"
        window.set_mode("bars")
        assert np.array_equal(picture(window), bars) and canvas.path == "bars"
        window.set_mode("radial")
        window.set_radial_theme("radiant")
        canvas.gpu = True
        alpha, _ = canvas_layer(window)
        assert canvas.path == "radiant-gpu" and alpha[alpha.shape[0] // 2, alpha.shape[1] // 2] == 0
        canvas.gpu = False
    window.set_radial_theme("classic")
    alpha, _ = canvas_layer(window)
    assert alpha.min() == 255  # Classic is opaque again: no clear ground left from Radiant


def test_the_composition_is_its_own_and_an_instrument_keeps_its_place(window):
    lay = plan()
    order = [0.0, lay.core, lay.core_most, lay.base, lay.tips, lay.contour, lay.segments, lay.frame]
    order += [edge for band in lay.bands for edge in band] + [lay.rim, lay.reach]
    assert order == sorted(order) and len(lay.bands) == 4
    assert [band[0] / lay.reach for band in lay.bands] == pytest.approx([composition.BANDS[key][0] for key in FOUR])
    # It is not phase 1's layout: other proportions throughout.
    phase1 = geometry.layout(640.0, 360.0, 300.0, (1.0, 1.25, 0.9, 1.0), 5.0)
    assert lay.base != phase1.base and lay.contour != phase1.contour and lay.bands != phase1.bands
    # Hiding instruments removes their zones and moves nobody else's.
    fewer = plan(("vocals", "drums"))
    assert fewer.bands == (lay.bands[0], lay.bands[3]) and fewer.rim == lay.rim and fewer.tips == lay.tips
    assert plan(()).bands == () and plan(()).frame == lay.frame
    for size in ((720, 480), (1920, 1080), (2560, 1440), (900, 1400)):
        window.resize(*size)
        canvas = window.canvas
        rect = canvas.rect().adjusted(28, 58, -28, -28)
        cx, cy, reach = canvas.radial_geometry(rect)
        fitted = composition.layout(cx, cy, reach, FOUR)
        assert (fitted.cx, fitted.cy) == (rect.center().x(), rect.center().y())
        assert fitted.rim < min(rect.width(), rect.height()) / 2 and fitted.rim > 0.9 * reach  # it fills the area


# -- shader sources (no graphics card) ---------------------------------------------------------


def test_every_shader_is_a_file_with_its_includes_resolved():
    assert SHADERS == ("plasma_field", "smoke", "formations", "instruments", "filaments", "core")
    assert POST == ("bloom_extract", "blur", "composite")
    assert resources.vertex().startswith("#version 330")
    for name in SHADERS + POST:
        text = resources.fragment(name)
        assert text.startswith("#version 330") and "#include" not in text and "void main()" in text
        assert (resources.FOLDER / f"{name}.frag").is_file()
    for name in SHADERS:  # the scene passes share one set of noise and one idea of place
        text = resources.fragment(name)
        assert "float fbm(vec3 p)" in text and "uniform float u_time" in text
        # Nothing but the uniforms can change a picture: no clock, no counter, no dice.
        assert "gl_FragCoord" not in text and "rand(" not in text and "u_frame_number" not in text
    instruments = resources.fragment("instruments")
    for key in FOUR:  # each instrument has a shader of its own, and its own colour and zone
        assert (resources.FOLDER / f"{key}.glsl").is_file()
        assert f"vec3 {key}(float r, float a, vec2 dir)" in instruments
        assert f"c_{key}" in instruments and f"u_band_{key}" in instruments
    assert "u_spectrum" in resources.fragment("formations") and "spectrum_at(a)" in resources.fragment("formations")
    assert not (resources.FOLDER / "plasma_rings.frag").exists()
    # No shader text is built in Python.
    for module in ("renderer.py", "frame.py", "scene.py", "view.py", "offscreen.py"):
        assert "void main" not in (resources.FOLDER.parent / module).read_text(encoding="utf-8")
    # The renderer does not reach into another renderer: not Classic, not the QPainter layers.
    for module in ("frame.py", "scene.py", "renderer.py"):
        text = (resources.FOLDER.parent / module).read_text(encoding="utf-8")
        imports = [line for line in text.splitlines() if line.lstrip().startswith(("import ", "from "))]
        assert "paint_stage" not in text and not any("layers" in line or "QPainter" in line for line in imports)


def test_a_missing_or_malformed_shader_file_says_which(tmp_path):
    with pytest.raises(ShaderError, match="nothing.frag cannot be read"):
        resources.source("nothing.frag")
    (tmp_path / "loop.frag").write_text('#version 330\n#include "loop.frag"\n', encoding="utf-8")
    (tmp_path / "bad.frag").write_text("#version 330\n#include common.glsl\n", encoding="utf-8")
    (tmp_path / "lost.frag").write_text('#version 330\n#include "gone.glsl"\n', encoding="utf-8")
    for name, message in (("loop.frag", "bad include"), ("bad.frag", "bad include"), ("lost.frag", "gone.glsl cannot be read")):
        with pytest.raises(ShaderError, match=message):
            resources.source(name, tmp_path)


# -- what a frame's shaders are told (no graphics card) ----------------------------------------


def test_the_values_are_finite_bounded_numbers_whatever_goes_in():
    feed = band_playing()
    feed.state.consume(event(feed.time - 0.03, "onset", "observed", intensity=1.0))
    view = scene(feed.state, feed.time, keys=FOUR)
    found = told(view, glow=("#4cc9f0", 0.4), instrument=lambda key: palette.INSTRUMENTS[key], area=(58, 690),
                 state=feed.state)  # fmt: skip
    assert all(isinstance(n, int | float) and math.isfinite(n) for n in numbers(found))
    for name in ("u_level", "u_intensity", "u_flash", "u_pulse", "u_low", "u_mid", "u_high", "u_glow"):
        assert 0.0 <= found[name] <= 1.0, name
    assert 0 < found["u_core"] < found["u_base"] < found["u_tips"] < found["u_contour"] < found["u_seg_low"]
    assert found["u_seg_high"] < found["u_frame"] < found["u_band_vocals"][0] < found["u_band_drums"][1] < found["u_rim"] < 1.0
    assert len(found["u_segments"]) == 40 and len(found["u_spectrum"]) == len(found["u_wave"]) == 48
    assert all(0 <= value <= 1 for name in ("u_segments", "u_spectrum", "u_wave") for value in found[name])
    assert 0 < found["u_onset_count"] <= frame.ONSETS and len(found["u_onsets"]) == frame.ONSETS
    assert len(found["u_bolts"]) == len(found["u_bolt_colours"]) == frame.BOLTS
    assert len(found["u_bass_pulses"]) == composition.PULSES and len(found["u_drum_hits"]) == len(found["u_drum_regions"]) == composition.HITS
    assert 0.0 <= found["u_time"] < frame.WRAP_S and found["u_area"] == (58 / 720, 690 / 720)
    # Damaged input cannot put a NaN, an infinity or a wrong shape into a shader.
    broken = RadiantPalette()
    broken.colours.update(plasma_primary="nonsense", filament_core=None, core_center="#12", core_halo="#gggggg")
    view.level, view.intensity, view.flash = float("nan"), float("inf"), -3.0
    found = told(view, broken, spectrum=np.array([np.nan] * 48), state=feed.state)
    assert all(math.isfinite(n) for n in numbers(found))
    assert found["c_plasma_primary"] == found["c_filament_core"] == found["c_core_halo"] == (0.0, 0.0, 0.0)
    assert found["u_level"] == found["u_low"] == 0.0 and found["u_intensity"] <= 1.0 and found["u_flash"] == 0.0
    assert found["u_spectrum"] == [0.0] * 48
    assert frame.bands(None) == frame.bands(()) == (0.0, 0.0, 0.0)
    assert frame.linear("#ffffff") == (1.0, 1.0, 1.0) and frame.linear("#000000") == (0.0, 0.0, 0.0)
    assert frame.linear("#808080")[0] == pytest.approx(0.2158, abs=1e-3)  # decoded, not divided


def test_the_music_reaches_the_shaders_and_nothing_else_does():
    quiet, loud = stream(-50, 0.15), stream(-8, 0.9)
    low, high = told(scene(quiet.state, quiet.time)), told(scene(loud.state, loud.time))
    assert high["u_level"] > low["u_level"] and high["u_intensity"] > low["u_intensity"]
    assert high["u_core"] > low["u_core"] and high["u_halo"] > low["u_halo"] and high["u_rays"] > low["u_rays"]
    view = scene(loud.state, loud.time)
    # No event, nothing struck: not one onset, not one bolt, at any time.
    for now in np.arange(1.2, 4.0, 0.21):
        found = told(scene(loud.state, float(now)), state=loud.state)
        assert found["u_onset_count"] == found["u_bolt_count"] == 0 and found["u_flash"] == found["u_pulse"] == 0.0
        assert all(bolt == (0.0, 0.0, 0.0, 0.0) for bolt in found["u_bolts"])
        assert found["u_vocals"] == found["u_guitar"] == found["u_bass"] == found["u_drums"] == (0.0,) * 4
    # An onset is there from its time on, ages by audio time, and is gone after a second.
    loud.state.consume(event(3.0, "onset", "observed", intensity=1.0))
    assert told(scene(loud.state, 2.99))["u_onset_count"] == 0
    for age in (0.0, 0.1, 0.4, 0.9):
        found = told(scene(loud.state, 3.0 + age))
        assert found["u_onset_count"] == 1 and found["u_onsets"][0][0] == pytest.approx(age, abs=1e-9)
        assert found["u_flash"] == pytest.approx(math.exp(-age / geometry.FLASH_S))
    assert told(scene(loud.state, 4.05))["u_onset_count"] == 0
    assert told(scene(loud.state, 3.1))["u_bolt_count"] > 0 and told(scene(loud.state, 3.5))["u_bolt_count"] == 0
    # A predicted beat pushes the segments out, only once reached and only if shown.
    loud.state.consume(event(3.6, "beat_prediction", "predicted"))
    assert told(scene(loud.state, 3.6))["u_seg_low"] > told(scene(loud.state, 3.6, beats=False))["u_seg_low"]
    assert told(scene(loud.state, 3.55))["u_pulse"] == 0.0
    # The waveform is the envelope of the frames, newest first, and nothing later than now.
    wave = composition.wave(loud.state.frames, loud.time)
    assert len(wave) == 48 and wave[0] == pytest.approx(wave[1], abs=0.05) and max(wave) <= 1.0 and min(wave[:-2]) > 0.1
    assert composition.wave(loud.state.frames, 0.5) == [0.0] * 48 and composition.wave(None, 3.0) == [0.0] * 48
    assert told(view, state=loud.state)["u_wave"] == composition.wave(loud.state.frames, loud.time)


def test_the_spectrum_reaches_the_shaders_in_its_order_with_nothing_added():
    spectrum = np.zeros(48)
    spectrum[:12] = 1.0
    assert frame.bands(spectrum) == (1.0, 0.0, 0.0)
    view = scene(stream().state, 3.0)
    found = told(view, spectrum=spectrum)
    assert found["u_spectrum"] == spectrum.tolist() and found["u_low"] == 1.0 and found["u_high"] == 0.0
    ramp = np.linspace(0.05, 0.95, 48)
    assert told(view, spectrum=ramp)["u_spectrum"] == pytest.approx(ramp.tolist())  # band for band
    assert frame.spectrum_bands(ramp, 0.78) == pytest.approx((ramp**0.78).tolist())  # the response's lift
    # Another number of bands is laid along the same order; every value lies between two measured ones.
    seven = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4]
    spread = frame.spectrum_bands(seven)
    assert len(spread) == 48 and spread[0] == 0.1 and spread[-1] == pytest.approx(0.4)
    for index, value in enumerate(spread):
        place = index * 6 / 47
        lower, upper = seven[int(place)], seven[min(6, int(place) + 1)]
        assert min(lower, upper) - 1e-9 <= value <= max(lower, upper) + 1e-9
    assert frame.spectrum_bands(None) == frame.spectrum_bands(()) == [0.0] * 48  # a valid quiet scene
    assert frame.spectrum_bands([2.0, -1.0, float("nan")] * 16) == [1.0, 0.0, 0.0] * 16


def test_the_display_spectrum_radiant_uses_is_eased_in_audio_time(window):
    def fed(state, base):
        for index in range(150):
            time = 1.0 + index * HOP
            loud = -20 if index < 100 else -70  # loud for two seconds, then nearly silent
            state.consume(
                feature_frame(time, spectrum_db=tuple(np.linspace(loud, loud - 20, 48).tolist())),
                arrival=base + time,
            )
        return state

    one, other = fed(VisualState(), 100.0), fed(VisualState(), 900.0)
    now = 1.0 + 149 * HOP
    assert one.spectrum_at(now) is not None and np.array_equal(one.spectrum_at(now), other.spectrum_at(now))
    assert VisualState().spectrum_at(1.0) is None and one.spectrum_at(None) is None
    # Between frames it falls by the release, as a function of audio time alone.
    later = one.spectrum_at(now + 0.1)
    assert later == pytest.approx(one.spectrum_at(now) * math.exp(-0.1 / one.audio_spectrum.release_s))
    assert (one.spectrum_at(now) < 0.9 * one.audio_spectrum.peaks + 1e-9).all()  # it is falling after the loud part
    # Asking every frame or once at the end gives the same heights: frames wait, in order, until asked.
    often = VisualState()
    for index in range(150):
        time = 1.0 + index * HOP
        loud = -20 if index < 100 else -70
        often.consume(feature_frame(time, spectrum_db=tuple(np.linspace(loud, loud - 20, 48).tolist())), arrival=5.0 + time)
        often.spectrum_at(time)
    assert np.array_equal(often.spectrum_at(now), one.spectrum_at(now)) and not one._spectrum_waiting
    # It does not depend on how often anything is drawn: the canvas's own display does.
    canvas = window.canvas
    fed(window.state, perf_counter() - 4.0)
    before = window.state.spectrum_at(now).copy()
    for _ in range(40):
        canvas.advance(1 / 144, perf_counter())
    assert np.array_equal(window.state.spectrum_at(now), before)
    # The display's settings reach it; Classic's own display keeps its own state.
    window.set_smoothing(0.4)
    window.set_gain(6.0)
    assert window.state.spectrum_release_s == 0.4 and window.state.spectrum_gain_db == 6.0
    assert canvas.display.release_s == 0.4 and canvas.display is not window.state.audio_spectrum
    window.state.reset()
    assert window.state.spectrum_at(1.0) is None and window.state.spectrum_release_s == 0.4


def test_each_instrument_is_told_from_its_own_live_state():
    feed = band_playing()
    state = feed.state
    found = composition.instruments(state, FOUR)
    vocals, bass, drums = state.live["vocals"], state.live["bass"], state.live["drums"]
    assert found["u_vocals"][:3] == (vocals.weight, pytest.approx(min(1.0, vocals.level)), vocals.place)
    assert found["u_vocals_more"][1] == 1.0  # the voice has a pitch
    assert found["u_bass"][1] == pytest.approx(min(1.0, bass.energy)) and found["u_bass"][2] == bass.place
    # The bass ring's phase is the one the bass layer uses: the rows' integral in audio time.
    assert composition.BASS_FLOW == (FLOW_BASE, FLOW_PER_ENERGY)
    assert found["u_bass_more"][0] == pytest.approx(bass.phase(FLOW_BASE, FLOW_PER_ENERGY) % (2 * math.pi))
    pulses = [pulse for pulse in found["u_bass_pulses"] if pulse[1] > 0]
    assert len(pulses) == len(bass.beats(1.2)) > 0
    assert [age for age, _ in pulses] == pytest.approx([age for age, _ in bass.beats(1.2)])
    hits = [(hit, regions) for hit, regions in zip(found["u_drum_hits"], found["u_drum_regions"], strict=True) if max(regions) > 0]
    recent = drums.recent(0.6)
    assert len(hits) == len(recent) > 0  # one impact per reported hit, no more
    for (hit, regions), (age, source) in zip(hits, recent, strict=True):
        heard = tuple(value * drums.weight for value in (source.low, source.mid, source.high))
        assert hit[0] == pytest.approx(age) and regions == pytest.approx(heard)  # as far as the presence model lets it through
        assert 0.0 <= hit[1] < 1.0
    assert found["u_guitar"] == (0.0,) * 4  # nobody played guitar in this stream
    # An instrument that is hidden is all zeros; the others are told exactly as before.
    without = composition.instruments(state, ("vocals", "drums"))
    assert without["u_bass"] == (0.0,) * 4 and all(pulse[1] == 0 for pulse in without["u_bass_pulses"])
    assert without["u_vocals"] == found["u_vocals"] and without["u_drum_regions"] == found["u_drum_regions"]
    # One the presence model does not hear is silent in the picture, though its data is there.
    kept = len(drums.hits)
    drums.weight = 0.0
    silent = composition.instruments(state, FOUR)
    assert silent["u_drums"] == (0.0,) * 4 and all(max(regions) == 0 for regions in silent["u_drum_regions"])
    assert len(drums.hits) == kept  # telling the shaders changes nothing in the state
    assert composition.instruments(None, (), False)["u_bass"] == (0.0,) * 4


def test_stronger_and_newer_events_give_larger_numbers_and_guitar_strokes_are_read():
    def with_guitar(strength, rate=None):
        feed = Stream()
        from test_instrument_layers_phase2 import activity

        for index in range(100):
            feed.frame(1)
            struck = strength if index % 10 == 0 else 0.0
            feed.state.consume(activity(feed.time - 0.02, "guitar", level=0.5, attack=struck, flux=struck, centre=0.3,
                                        spread=0.5, rate_hz=rate, regularity=0.9 if rate else 0.0, segment=1))  # fmt: skip
        composition.advance(feed.state, feed.time, FOUR)
        return composition.instruments(feed.state, FOUR), feed

    soft, _ = with_guitar(0.3)
    hard, feed = with_guitar(0.9)
    assert hard["u_guitar"][3] > soft["u_guitar"][3] and hard["u_guitar"][2] > soft["u_guitar"][2]
    assert hard["u_guitar_more"][1] == 0.3 and hard["u_guitar_more"][2] == 0.5  # where the stroke sat, how wide
    assert 0.0 <= hard["u_guitar_more"][0] < 0.25 and hard["u_guitar_more"][3] == 0.0
    steady, _ = with_guitar(0.9, rate=5.0)
    assert steady["u_guitar_more"][3] == pytest.approx(0.45)  # steady strumming, by the analyzer's own word
    # The stroke ages with audio time and the ringing dies away.
    composition.advance(feed.state, feed.time + 0.4, FOUR)
    later = composition.instruments(feed.state, FOUR)
    assert later["u_guitar_more"][0] == pytest.approx(hard["u_guitar_more"][0] + 0.4) and later["u_guitar"][2] < hard["u_guitar"][2]
    # A phrase start blooms and dies away; pitch moves the ribbons.
    feed = Stream()
    for index in range(60):
        feed.frame(1)
        began = index == 55
        feed.vocals(1, level=0.7, pitch_midi=50.0 if index < 30 else 76.0, attack=0.8 * began, boundary="start" if began else "")
        composition.advance(feed.state, feed.time, FOUR)
        if index == 25:
            low_place = composition.instruments(feed.state, FOUR)["u_vocals"][2]
    now = composition.instruments(feed.state, FOUR)
    assert now["u_vocals"][2] > low_place and now["u_vocals"][3] > 0.3 and now["u_vocals_more"][3] < 0.2
    composition.advance(feed.state, feed.time + 0.2, FOUR)
    assert composition.instruments(feed.state, FOUR)["u_vocals"][3] < now["u_vocals"][3]


def test_bolts_are_the_reported_events_in_the_instruments_zones_and_colours():
    feed = band_playing()
    view = scene(feed.state, feed.time, keys=FOUR)
    colours = RadiantPalette()
    colours.set("bass", "#39ff14")
    colours.set("drums", "#ff1493")
    found = told(view, colours, instrument=lambda key: colours[key], state=feed.state)
    assert found["u_bolt_count"] == len(view.streaks) > 0
    lay, seen = plan(), set()
    for index in range(found["u_bolt_count"]):
        angle, start, end, strength = found["u_bolts"][index]
        source = next(row[5] for row in view.streaks if row[0] % (2 * math.pi) == pytest.approx(angle))
        low, high = composition.BANDS[source]
        assert start < high and end > low and 0 < strength <= 1 and end <= lay.rim / lay.reach + 1e-9
        assert found["u_bolt_colours"][index][:3] == frame.linear(colours[source])
        seen.add(source)
    assert {"bass", "drums"} <= seen
    # Hidden instruments strike nothing; a new segment leaves nothing of the old one.
    assert told(scene(feed.state, feed.time, keys=()), keys=())["u_bolt_count"] == 0
    feed.state.consume(event(feed.time - 0.02, "onset", "observed", intensity=1.0))
    assert told(scene(feed.state, feed.time, keys=FOUR))["u_onset_count"] == 1
    feed.frame(2)
    composition.advance(feed.state, feed.time, FOUR)
    after = told(scene(feed.state, feed.time, keys=FOUR), state=feed.state)
    assert after["u_onset_count"] == after["u_bolt_count"] == 0 and after["u_flash"] == 0.0
    assert all(pulse[1] == 0 for pulse in after["u_bass_pulses"]) and all(max(r) == 0 for r in after["u_drum_regions"])


def at_rate(rate, target):
    """The frame's values at audio time ``target`` when the display steps at ``rate``."""
    state, waiting = VisualState(), messages()
    base, tick = 1000.0, 0
    while True:
        now = 1.0 + tick / rate
        while waiting and waiting[0].timestamp <= now:
            message = waiting.pop(0)
            state.consume(message, arrival=base + message.timestamp)
        composition.advance(state, now, FOUR)
        if now >= target - 1e-9:
            view = scene(state, now, keys=FOUR, values=state.spectrum_at(now))
            return told(view, spectrum=state.spectrum_at(now), state=state, instrument=lambda key: palette.INSTRUMENTS[key])
        tick += 1


def test_the_same_audio_time_gives_the_same_values_at_any_frame_rate():
    for target in (3.0, 3.5):
        first = at_rate(60, target)
        assert first["u_bolt_count"] > 0 and first["u_time"] == pytest.approx(target) and max(first["u_spectrum"]) > 0
        assert first["u_bass"][0] > 0 and first["u_vocals"][0] > 0
        for rate in (72, 120, 144):
            other = at_rate(rate, target)
            assert other.keys() == first.keys()
            # The events, the spectrum, the waveform, the times: exactly the same.
            for name in ("u_spectrum", "u_wave", "u_segments", "u_onsets", "u_bolts", "u_bass_pulses", "u_drum_hits",
                         "u_drum_regions", "u_time", "u_flash", "u_pulse", "u_level", "u_core", "u_bass_more"):  # fmt: skip
                mine, theirs = first[name], other[name]
                flat = lambda v: [x for item in v for x in (item if isinstance(item, tuple) else (item,))] if isinstance(v, list) else list(v) if isinstance(v, tuple) else [v]  # noqa: E731
                assert flat(theirs) == pytest.approx(flat(mine), abs=1e-6), (name, rate)
            # The envelopes the live states ease between rows agree closely at any rate.
            assert list(numbers(other)) == pytest.approx(list(numbers(first)), abs=0.03), rate
    # Audio time is all the shaders have for a clock; it is wrapped, not restarted.
    view = scene(stream().state, 3.0)
    view.now = frame.WRAP_S * 2 + 12.5
    assert told(view)["u_time"] == pytest.approx(12.5)
    view.now = None
    assert told(view)["u_time"] == 0.0


def test_expressive_mode_scales_the_picture_and_adds_no_event():
    names = {field.name for field in fields(Response)}
    assert {"plasma", "bloom"} <= names
    assert BASELINE.plasma == BASELINE.bloom == 1.0 and 1.0 < EXPRESSIVE.plasma <= 2.0 and 1.0 < EXPRESSIVE.bloom <= 2.0
    feed = band_playing()
    feed.state.consume(event(feed.time - 0.04, "onset", "observed", intensity=0.8))
    plain = told(scene(feed.state, feed.time, BASELINE, FOUR), response=BASELINE, state=feed.state)
    bold = told(scene(feed.state, feed.time, EXPRESSIVE, FOUR), response=EXPRESSIVE, state=feed.state)
    for name in ("u_plasma", "u_bloom", "u_motion", "u_impact", "u_shine", "u_core"):
        assert bold[name] > plain[name], name
    for name in ("u_onset_count", "u_bolt_count", "u_onsets", "u_flash", "u_pulse", "u_level", "u_time", "u_low", "u_high",
                 "u_vocals", "u_bass", "u_bass_pulses", "u_bass_more", "u_drum_hits", "u_drum_regions", "u_wave"):  # fmt: skip
        assert bold[name] == plain[name], name  # the same events, the same music, the same time
    assert [bolt[0] for bolt in bold["u_bolts"]] == [bolt[0] for bolt in plain["u_bolts"]]
    assert all(b >= p for b, p in zip(bold["u_spectrum"], plain["u_spectrum"], strict=True))  # lifted, never past full
    assert max(bold["u_spectrum"]) <= 1.0 and bold["u_core"] <= plan().core_most / plan().reach


def test_every_palette_role_reaches_its_own_uniform_and_no_other():
    view = scene(stream().state, 3.0)
    base = told(view)
    assert frame.COLOURS == KEYS and len(KEYS) == 25 and set(EFFECTS) < set(KEYS)
    for role in KEYS:
        colours = RadiantPalette()
        colours.set(role, "#ff1493" if colours[role] != "#ff1493" else "#00ff00")
        found = told(view, colours)
        changed = {name for name in found if found[name] != base[name]}
        assert changed == {"c_" + role}, role
        assert found["c_" + role] == frame.linear(colours[role])
    # Every role is used by a shader: none is passed and dropped.
    sources = "".join(resources.fragment(name) for name in SHADERS + POST)
    for role in KEYS:
        assert "c_" + role in sources or role in ("energy_accents",), role
    assert "u_bolt_colours" in sources  # energy accents and the instruments' colours colour the bolts
    # Unrelated hues side by side: each uniform is its own role's colour, untouched.
    pairs = dict(plasma_primary="#ff1493", filament_glow="#39ff14", segmented_ring="#facc15", core_center="#a855f7",
                 energy_accents="#22d3ee", drums="#ff7a00", spectrum_body="#00ffff", vocals="#b6ff00")  # fmt: skip
    colours = RadiantPalette()
    for role, colour in pairs.items():
        assert colours.set(role, colour)
    found = told(view, colours)
    for role, colour in pairs.items():
        assert found["c_" + role] == frame.linear(colour)
    # Colours change no geometry and no response to the music; the canonical palette is not theirs.
    assert {k: v for k, v in found.items() if k.startswith("u_")} == {k: v for k, v in base.items() if k.startswith("u_")}
    assert palette.INSTRUMENTS == {"vocals": "#f472b6", "guitar": "#f5c451", "bass": "#4cc9f0", "drums": "#ff8a5b"}
    assert base["c_vocals"] == frame.linear("#f472b6") and base["c_drums"] == frame.linear("#ff8a5b")


def test_the_new_roles_are_saved_restored_and_fall_back_like_the_others():
    colours = RadiantPalette()
    factory = PRESETS[FACTORY][1]
    assert all(role in factory for role in EFFECTS) and len(PRESETS) == 4
    for name, (_, preset) in PRESETS.items():
        assert len({preset[role] for role in EFFECTS}) == len(EFFECTS) == 8, name  # a colour each, not one
    assert colours.set("plasma_primary", "#ff1493") and colours.set("filament_glow", "#39ff14")
    again = RadiantPalette()
    again.load(colours.saved())
    assert again["plasma_primary"] == "#ff1493" and again["filament_glow"] == "#39ff14"
    assert again["plasma_secondary"] == factory["plasma_secondary"]
    old = {"preset": "custom", "colours": {role: "#123456" for role in KEYS if role not in EFFECTS}}
    again.load(old)
    assert all(again[role] == factory[role] for role in EFFECTS) and again["core_center"] == "#123456"
    again.load({"colours": {"plasma_highlights": "blue", "filament_core": 7}})
    assert again["plasma_highlights"] == factory["plasma_highlights"] and again["filament_core"] == factory["filament_core"]
    again.restore()
    assert again.colours == factory


# -- without the graphics card -------------------------------------------------------------------


def test_without_opengl_radiant_is_the_reduced_qpainter_picture_and_says_so(window):
    canvas = playing(window)
    report = window.renderer_report()
    assert report["backend"] == "cpu" and "offscreen platform has no OpenGL" in report["detail"]
    assert "reduced" in report["detail"] and "gpu" not in report["detail"].lower().replace("opengl", "")
    assert window.plasma is None and canvas.gpu is False and canvas.path == "radiant-cpu"
    assert canvas.plasma_values() is None  # nothing is prepared for a card that is not drawing
    fallback = picture(window)
    assert window.performance()["radiant renderer"] == report
    window.set_radiant_effects("off")
    assert "switched off in Settings" in window.renderer_report()["detail"] and np.array_equal(picture(window), fallback)
    window.set_radiant_effects("nonsense")
    assert window.effects == "off"
    window.set_radiant_effects("auto")
    # The card failing later (a shader that does not compile, a lost context) is the same.
    window.effects_problem = ""
    window.effects_failed("core.frag did not compile: ERROR: 0:1")
    assert canvas.gpu is False and "core.frag did not compile" in window.renderer_report()["detail"]
    assert np.array_equal(picture(window), fallback) and canvas.path == "radiant-cpu"
    window.set_radial_theme("classic")
    assert window.renderer_report()["backend"] == "cpu"


def test_the_effects_choice_is_a_setting_and_new_colours_have_controls(app, tmp_path):
    from gmv.app.window import MainWindow

    first = MainWindow(session_root=tmp_path, remember=False, discover_sources=False)
    try:
        drawer = first.drawer
        assert drawer.remembered()["radiant_effects"] == "auto" and first.view.effects == "auto"
        assert set(drawer.radiant_controls) == set(KEYS) and set(EFFECTS) <= set(drawer.radiant_controls)
        drawer.radiant_effects.setCurrentIndex(drawer.radiant_effects.findData("off"))
        assert first.view.effects == "off"
        code = drawer.radiant_controls["plasma_primary"][1]
        code.setText("#ff1493")
        code.editingFinished.emit()
        assert first.view.radiant_palette["plasma_primary"] == "#ff1493"
        saved = drawer.remembered()
        assert saved["radiant_effects"] == "off" and saved["radiant_palette"]["colours"]["plasma_primary"] == "#ff1493"
    finally:
        first.close()
    second = MainWindow(session_root=tmp_path, remember=False, discover_sources=False)
    try:
        second.drawer.restore(saved)
        assert second.view.effects == "off" and second.view.radiant_palette["plasma_primary"] == "#ff1493"
        second.drawer.restore({"mode": "radial", "radial_theme": "radiant"})  # saved by phase 1
        assert second.view.effects == "auto" and second.view.canvas.theme == "radiant"
        assert second.view.radiant_palette["plasma_primary"] == PRESETS[FACTORY][1]["plasma_primary"]
        second.drawer.restore({"radiant_effects": 12})
        assert second.view.effects == "auto"
    finally:
        second.close()


def test_a_colour_change_reaches_the_cards_frame_at_once_and_resets_nothing(window):
    canvas = playing(window)
    canvas.gpu = True
    window.grab()
    state = canvas.state
    kept = (state.generation, state.live_epoch, len(state.frames), state.marked, dict(state.dominance.scores))
    before = canvas.plasma_values()
    assert window.set_radiant_colour("plasma_primary", "#ff1493") and window.set_radiant_colour("bass", "#39ff14")
    after = canvas.plasma_values()  # no paint in between: the very next frame has it
    assert after["c_plasma_primary"] == frame.linear("#ff1493") and after["c_bass"] == frame.linear("#39ff14")
    assert {k: v for k, v in after.items() if k.startswith("u_")} == {k: v for k, v in before.items() if k.startswith("u_")}
    for choice in ("off", "auto", "off", "auto"):
        window.set_radiant_effects(choice)
        window.grab()
    for mode in ("bars", "radial"):
        window.set_mode(mode)
        window.grab()
    assert kept == (state.generation, state.live_epoch, len(state.frames), state.marked, dict(state.dominance.scores))
    # The leading instrument tints the glow and only the glow.
    canvas.gpu = True
    window.grab()
    found = canvas.plasma_values()
    assert found["c_glow"] == frame.linear(canvas.glow_colour()) and found["c_plasma_primary"] == frame.linear("#ff1493")


# -- on the graphics card ----------------------------------------------------------------------------


def leads(tint, channel, by=1.5):
    others = [value for index, value in enumerate(tint) if index != channel]
    return tint[channel] > by * max(others)


def test_the_probe_ran_or_says_why(measured):
    assert isinstance(measured["available"], bool)
    if measured["available"]:
        assert measured["context"]["renderer"] and measured["context"]["glsl"]
    else:
        assert measured["reason"]


def test_every_shader_compiles_links_and_resources_come_and_go_cleanly(gpu):
    assert gpu["programs"] == gpu["programs expected"] == sorted(SHADERS + POST)
    assert "core.frag did not compile" in gpu["broken shader error"] and "ERROR" in gpu["broken shader error"]
    assert "bass.glsl cannot be read" in gpu["missing shader error"]
    assert gpu["buffers before resizing"] == gpu["buffers after resizing"] == 5  # reused, not piled up
    assert gpu["ready after release"] is False and gpu["left after release"] == 0
    assert gpu["set-up and tear-down cycles"] == 6  # six contexts made, drawn in at two sizes and torn down


def test_every_pass_draws_something_and_the_same_thing_every_time(gpu):
    for name in SHADERS:
        assert gpu[f"{name} spread"] > 5.0, name  # not a flat fill
        assert gpu[f"{name} repeat difference"] == 0.0, name  # the same values, the same pixels
    assert gpu["whole picture repeat difference"] == 0.0 and gpu["field other seed repeat difference"] == 0.0


def test_the_plasma_field_has_structure_voids_and_coherent_flow(gpu):
    assert gpu["field dark share in corona"] > 0.15 and gpu["field bright share in corona"] > 0.1
    assert gpu["field over the middle"] < 0.6 * gpu["field in corona"]
    assert 0 < gpu["field change after 0.05 s"] < gpu["field change after 0.5 s"] < gpu["field change after 5.0 s"]
    assert gpu["field likeness after 0.05 s"] > 0.99 and gpu["field likeness after 0.5 s"] > 0.9
    assert 0.2 < gpu["field likeness after 5.0 s"] < 0.95
    assert abs(gpu["field likeness other seed"]) < 0.5  # another seed, another field
    assert gpu["particle light"] > 60 and leads(gpu["particle tint red"], 0)


def test_the_formations_are_surfaces_with_thickness_and_bounded_tendrils(gpu):
    assert gpu["outer formation lit share of a broad ring"] > 0.8  # a body, not an outline
    assert gpu["outer formation spread"] > 15 and gpu["middle formation spread"] > 15  # textured
    assert gpu["outer formation mean"] != gpu["middle formation mean"]
    assert gpu["formations change after 1 s"] > 5  # energy going round
    assert gpu["segment light in segments"] > 1.3 * gpu["segment light in gaps"]
    assert gpu["tendrils with lows"] > gpu["tendrils without"] and gpu["tendril lit share"] > 0.2
    assert gpu["tendrils beyond their reach"] < 0.2 * gpu["tendrils with lows"]  # they end


def test_the_spectrum_shapes_the_inner_formation_in_its_own_order(gpu):
    # Lows light the top and leave the bottom dark; highs the other way round.
    assert gpu["spectrum formation top with lows"] > 20 * gpu["spectrum formation bottom with lows"]
    assert gpu["spectrum formation bottom with highs"] > 20 * gpu["spectrum formation top with highs"]
    # Nothing measured, nothing raised: a valid, subdued floor.
    assert gpu["spectrum formation top with silence"] < 0.05 * gpu["spectrum formation top with full"]
    # A stronger band stands higher and is brighter.
    radii = [gpu[f"spectrum surface radius at {s}"] for s in (0.25, 0.5, 1.0)]
    lights = [gpu[f"spectrum formation light at {s}"] for s in (0.25, 0.5, 1.0)]
    assert radii == sorted(radii) and radii[2] - radii[0] > 0.08 and lights == sorted(lights) and lights[2] > 3 * lights[0]
    quarters = gpu["rising spectrum light by quarter"]
    assert quarters == sorted(quarters) and quarters[3] > 2 * quarters[0]  # frequency order kept
    assert gpu["one band light at its place"] > 20 * gpu["one band light far from it"]  # no light invented
    assert gpu["spectrum formation lit share along a circle"] > 0.95  # one body, not separate spikes


def test_each_instrument_has_its_own_structure_in_its_own_zone_and_colour(gpu):
    assert gpu["nothing playing light"] == 0.0
    for key, channel in (("vocals", 0), ("guitar", 0), ("drums", 0)):
        assert gpu[f"{key} tint"][channel] == max(gpu[f"{key} tint"]), key  # pink, gold and orange are red-led
    assert gpu["bass tint"][2] == max(gpu["bass tint"]) and gpu["bass tint"][0] < 0.25  # cyan
    assert gpu["guitar tint"][1] > gpu["vocals tint"][1] and gpu["vocals tint"][2] > gpu["guitar tint"][2]  # gold is not pink
    for key in ("vocals", "guitar", "bass", "drums"):
        assert gpu[f"{key} light"] > 1.0, key
        assert gpu[f"{key} share inside its zone"] > 0.8, key
        assert leads(gpu[f"{key} tint custom green"], 1), key  # the Radiant palette's colour, whatever it is
        assert gpu[f"{key} light when hidden"] == 0.0, key
        assert gpu[f"{key} adds to the others"] > 5, key  # recognisable beside the other three
        assert gpu[f"{key} hiding changes the others"] <= 3.0, key  # and hiding it moves nobody


def test_each_instrument_answers_its_own_data_the_way_it_should(gpu):
    assert gpu["vocals loud"] > 2 * gpu["vocals soft"] and gpu["vocals with a phrase start"] > 1.2 * gpu["vocals loud"] * 0.8
    assert gpu["vocals radius with a high note"] > gpu["vocals radius with a low note"] and gpu["vocals moves in 0.3 s"] > 0.1
    assert gpu["guitar just struck"] > gpu["guitar struck softly"] > gpu["guitar idle"] > 0
    assert gpu["guitar just struck"] > gpu["guitar ringing later"] > gpu["guitar idle"]
    assert gpu["guitar strumming steadily"] > gpu["guitar single stroke"]
    assert gpu["bass strong"] > 2 * gpu["bass weak"] and gpu["bass thickness strong"] > 2 * gpu["bass thickness weak"]
    assert gpu["bass wave beyond its zone with a note onset"] > 20 * max(0.01, gpu["bass wave beyond its zone without"])
    assert gpu["bass phase moves the ring"] > 0.1
    for region in ("low", "mid", "high"):
        assert gpu[f"drums {region} hit"] > gpu[f"drums weak {region} hit"] > gpu["drums no hit"], region
    shares = {gpu[f"drums {region} hit lit share"] for region in ("low", "mid", "high")}
    assert len(shares) == 3  # three kinds of impact, not one
    assert gpu["drums low hit old"] == pytest.approx(gpu["drums no hit"]) and gpu["drums mid hit old"] == pytest.approx(gpu["drums no hit"])
    assert gpu["drums two hits at once"] > gpu["drums low hit"]  # hits overlap


def test_filaments_are_thin_crooked_and_bolts_come_only_with_events(gpu):
    assert 0.003 < gpu["filaments lit share"] < 0.2  # lines, not fills
    assert gpu["filaments angle spread per radius"] > 0.3  # not straight radial strokes
    assert gpu["filaments light with highs"] > 1.3 * gpu["filaments light calm"]
    assert gpu["bolts without events"] == 0 and gpu["bolts with events"] > 0
    assert gpu["filaments light with bolts"] > 1.5 * gpu["filaments light without bolts"]
    assert gpu["filaments light with old bolts"] < 1.3 * gpu["filaments light without bolts"]  # they decay


def test_the_core_has_depth_keeps_its_detail_and_its_colour(gpu):
    regions = [gpu[f"core {name}"] for name in ("centre", "body", "limb", "corona", "far")]
    assert regions == sorted(regions, reverse=True) and regions[0] > 150 and regions[-1] > 3
    assert regions[0] - regions[1] > 15 and regions[2] - regions[3] > 30  # distinct, not one gradient
    assert gpu["core ray spread"] > 3
    assert gpu["core light loud"] > 2 * gpu["core light soft"] > 0
    assert gpu["core lit radius loud"] > 2 * gpu["core lit radius soft"]
    # At full intensity it is not a white disc: almost none of it is white, and it has texture.
    assert gpu["core white share of its disc when loud"] < 0.15 and gpu["core body spread when loud"] > 6
    # A pink core is pink to its middle.
    assert leads(gpu["pink core tint when loud"], 0, by=1.3) and gpu["pink core tint when loud"][1] < 0.2
    assert gpu["pink core white share when loud"] < 0.05


def test_bloom_lights_what_is_beside_bright_things_and_nothing_far_away(gpu):
    assert gpu["bright share"] > 0.005
    assert gpu["bloom beside bright"] > 20 and gpu["bloom adds beside bright"] > 5
    assert gpu["bloom in the corners"] < 3 and gpu["bloom adds in the corners"] < 3
    assert gpu["picture white share"] < 0.05  # not one white mass


def test_custom_colours_reach_the_effects_they_belong_to_and_stay_saturated(gpu):
    assert leads(gpu["field tint red plasma"], 0) and leads(gpu["field tint green plasma"], 1)
    assert leads(gpu["filament tint green"], 1) and leads(gpu["core tint orange"], 0)
    assert leads(gpu["spectrum tint red"], 0) and leads(gpu["waveform tint green"], 1)
    assert leads(gpu["outer formation tint green"], 1) and gpu["segment tint yellow"][2] < 0.2
    for name in ("plasma_field", "formations", "instruments", "core"):
        assert gpu[f"{name} change when filament colour changes"] == 0.0
    # Red plasma round a blue core, with green filaments: neither pulled towards the other.
    assert leads(gpu["mixed corona tint"], 0) and leads(gpu["mixed core tint"], 2)
    # Hot pink at its brightest is still hot pink, not pastel.
    assert gpu["bright pink share"] > 0.02 and gpu["bright pink saturation"] > 0.6


def test_the_music_changes_the_picture_the_way_it_should(gpu):
    assert gpu["field light loud"] > 1.2 * gpu["field light quiet"]
    assert gpu["field reach loud"] > 1.3 * gpu["field reach quiet"]
    assert gpu["whole picture light loud"] > 1.15 * gpu["whole picture light quiet"]  # quiet is visibly subdued
    assert gpu["field reach with lows"] > 1.2 * gpu["field reach without lows"]
    assert gpu["field veins with highs"] > gpu["field veins without highs"]
    assert gpu["field change with an onset"] > 5 * gpu["field change with an old onset"]
    assert gpu["scene light expressive"] > 1.1 * gpu["scene light baseline"]


def test_the_cost_and_memory_are_measured_and_bounded(gpu):
    assert 0 < gpu["ms per frame with read-back 1920x1080"] < 100
    assert 0 < gpu["buffers MiB 1920x1080"] < 64 and gpu["buffers MiB 2560x1440"] < 96
