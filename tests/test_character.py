"""Sonic Character: measurements, windows, the profile, the tracker and the diagnostics."""

import json
import math
from threading import Event
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from gmv.bus import EventBus
from gmv.character import DIMENSION_NAMES, WINDOWS, CharacterProfile, SonicCharacter
from gmv.character import cli as character_cli
from gmv.character import diagnose
from gmv.character.aggregate import LevelDistribution, Windows
from gmv.character.measures import COUNT, INDEX, NAMES, measure, silent
from gmv.character.profile import DIMENSIONS, unit
from gmv.character.tracker import BREAK_S, LOOSEN
from gmv.cli import main as gmv_main
from gmv.engine import Engine
from gmv.models import FeatureFrame
from gmv.session import replay
from gmv.sources.file import FileAudioSource
from gmv.visualizer.intensity import MusicalIntensity

from test_visualizer import event, frame

RATE = 44100
HOP = 0.02


# -- synthetic frames ---------------------------------------------------------------------


def spectrum(tilt=-2.0, ripple=0.0, level=-30.0, count=48):
    """A display spectrum with the given slope (dB/octave) and peakiness (dB)."""
    octaves = np.log2(np.geomspace(30, 18000, count + 1)[:-1] / 30)
    bumps = ripple * np.where(np.arange(count) % 6 == 0, 1.0, -0.2)
    return tuple((level + tilt * octaves + bumps).tolist())


def sound(time, level=-20.0, **changes):
    """A feature frame of a plain, middling sound; ``level`` moves everything that depends
    on the level together, as turning the volume would."""
    gain = 10 ** ((level + 20.0) / 20.0)
    bands = changes.pop("bands", (0.2, 0.25, 0.15, 0.2, 0.1, 0.07, 0.03))
    shape = changes.pop("spectrum", spectrum())
    values = dict(
        loudness_dbfs=level,
        short_term_loudness_dbfs=level,
        rms=0.1 * gain,
        peak=0.3 * gain,
        band_energy=tuple(b * gain * gain for b in bands),
        spectrum_db=tuple(v + level + 20.0 for v in shape),
        spectral_centroid_hz=500.0,
        spectral_rolloff_hz=3000.0,
        spectral_bandwidth_hz=1500.0,
        spectral_contrast_db=(14.0,) * 7,
        zero_crossing_rate=0.05,
        spectral_flux=0.18,
        attack_energy_ratio=0.04,
        rhythmic_density=3.0,
        tempo_confidence=0.4,
        tempo_bpm=120.0,
    )
    return frame(time, **{**values, **changes})


def play(tracker, start, seconds, hop=HOP, make=sound, **changes):
    """Feed a steady passage; returns the time it ends."""
    for index in range(round(seconds / hop)):
        tracker.frame(make(start + index * hop, **changes))
    return start + seconds


def values(tracker, window="long"):
    return dict(zip(DIMENSION_NAMES, tracker.profile(window).values(), strict=True))


def heard(seconds=30.0, window="long", **changes):
    tracker = SonicCharacter()
    play(tracker, 0.0, seconds, **changes)
    return values(tracker, window)


# -- measurements -------------------------------------------------------------------------


def test_every_measurement_but_the_level_is_the_same_at_any_volume():
    loud, quiet = measure(sound(1.0, level=-12.0)), measure(sound(1.0, level=-42.0))
    assert loud[INDEX["level_db"]] - quiet[INDEX["level_db"]] == pytest.approx(30.0)
    for name in NAMES[1:]:
        a, b = loud[INDEX[name]], quiet[INDEX[name]]
        assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, abs=1e-9), name


def test_the_shape_measurements_read_the_spectrum_as_they_say():
    flat = measure(sound(0.0, spectrum=spectrum(tilt=0.0)))
    dark = measure(sound(0.0, spectrum=spectrum(tilt=-6.0)))
    peaky = measure(sound(0.0, spectrum=spectrum(tilt=-6.0, ripple=12.0)))
    tilt, ripple, occupancy = INDEX["tilt_db_oct"], INDEX["ripple_db"], INDEX["occupancy"]
    assert flat[tilt] == pytest.approx(0.0, abs=1e-6) and dark[tilt] == pytest.approx(-6.0, abs=1e-6)
    assert flat[ripple] < 1e-6 and dark[ripple] < 1e-6  # the slope itself is not ripple
    assert peaky[ripple] > 4.0
    assert flat[occupancy] == 1.0 and dark[occupancy] < 0.7
    assert flat[INDEX["flatness_db"]] == pytest.approx(0.0, abs=1e-6)
    assert dark[INDEX["flatness_db"]] < -3.0


def test_what_a_frame_cannot_give_is_left_out_not_made_up():
    bare = sound(0.0, spectrum_db=(), spectrum_low_hz=None, spectrum_high_hz=None)
    got = measure(bare)
    for name in ("flatness_db", "occupancy", "tilt_db_oct", "ripple_db"):
        assert math.isnan(got[INDEX[name]])
    assert math.isfinite(got[INDEX["centroid_oct"]])
    # Not a FeatureFrame at all, with broken values: nothing raises, nothing is invented.
    broken = SimpleNamespace(
        **{
            **sound(0.0).model_dump(),
            "spectral_centroid_hz": float("nan"),
            "spectral_rolloff_hz": -5.0,
            "band_energy": (0.1, 0.2),
            "rms": 0.0,
            "tempo_bpm": None,
            "spectrum_db": (float("inf"),) * 48,
        }
    )
    got = measure(broken)
    for name in ("centroid_oct", "rolloff_oct", "low_share", "crest_db", "tempo_oct", "tilt_db_oct"):
        assert math.isnan(got[INDEX[name]]), name
    assert len(got) == COUNT
    tracker = SonicCharacter()
    for index in range(200):
        tracker.frame(SimpleNamespace(**{**vars(broken), "timestamp": index * HOP}))
    profile = tracker.profile("long")
    assert profile.dimension("brightness").value is None  # nothing it needs was measured
    assert profile.dimension("brightness").confidence == 0.0
    assert profile.dimension("pace").value is not None
    json.loads(profile.model_dump_json())


def test_a_field_an_old_session_never_recorded_is_unknown_not_zero():
    data = sound(0.0).model_dump()
    for name in ("attack_energy_ratio", "spectrum_db", "spectrum_low_hz", "spectrum_high_hz"):
        del data[name]
    old = FeatureFrame.model_validate(data)
    assert old.attack_energy_ratio == 0  # the model's default...
    assert math.isnan(measure(old)[INDEX["attack"]])  # ...is not taken for a measurement
    tracker = SonicCharacter()
    for index in range(1000):
        tracker.frame(old.model_copy(update={"timestamp": index * HOP}))
    profile = tracker.profile("long")
    assert any(w.startswith("no_display_spectrum") for w in profile.warnings)
    full = SonicCharacter()
    play(full, 0.0, 20.0)
    for name in ("brightness", "density", "roughness"):
        assert profile.dimension(name).value is not None
        assert profile.dimension(name).confidence < full.profile("long").dimension(name).confidence


def test_silence_is_the_flag_or_a_level_under_the_floor():
    assert silent(sound(0.0, silence=True))
    assert silent(sound(0.0, level=-70.0))
    assert not silent(sound(0.0, level=-55.0))  # quiet music is music


# -- normalization ------------------------------------------------------------------------


def test_unit_is_zero_and_one_at_the_anchors_and_held_between_them():
    assert unit(2.0, 2.0, 22.0) == 0.0 and unit(22.0, 2.0, 22.0) == 1.0
    assert unit(12.0, 2.0, 22.0) == pytest.approx(0.5)
    assert unit(-50.0, 2.0, 22.0) == 0.0 and unit(500.0, 2.0, 22.0) == 1.0
    # An anchor pair may run downwards: a low crest factor is a high value.
    assert unit(8.2, 10.8, 8.2) == 1.0 and unit(10.8, 10.8, 8.2) == 0.0 and unit(12.0, 10.8, 8.2) == 0.0


def test_every_dimension_is_defined_by_known_measurements_with_weights_that_add_up():
    from gmv.character.profile import DERIVED

    for name, (_meaning, needed, parts, gate) in DIMENSIONS.items():
        assert needed > 0
        assert sum(part[3] for part in parts) == pytest.approx(1.0), name
        for part in (*parts, *((gate,) if gate else ())):
            assert part[0] in INDEX or part[0] in DERIVED
            assert part[1] != part[2]
    assert "level_db" not in {p[0] for d in DIMENSIONS.values() for p in d[2]}


# -- windows ------------------------------------------------------------------------------


def test_a_window_starts_as_the_plain_average_and_then_forgets():
    windows = Windows(2, (2.0, math.inf))
    for value in (1.0, 3.0):
        windows.add([value, math.nan], 0.02)
    assert windows.mean(0)[0] == pytest.approx(2.0, abs=0.01)  # no pull towards zero
    assert math.isnan(windows.mean(0)[1]) and windows.weight[0][1] == 0.0
    for _ in range(1000):  # 20 s of 5: ten time constants
        windows.add([5.0, 7.0], 0.02)
    assert windows.mean(0)[0] == pytest.approx(5.0, abs=1e-3)
    assert windows.mean(1)[0] == pytest.approx((1 + 3 + 5 * 1000) / 1002)  # everything
    assert windows.weight[0][0] == pytest.approx(2.0, rel=0.02)  # seconds it still weighs
    assert windows.spread(0)[1] == pytest.approx(0.0, abs=1e-6)
    assert windows.spread(1)[0] > 0.05


def test_windows_do_not_depend_on_how_often_they_are_updated():
    def run(steps):
        windows, time = Windows(1, (4.0,)), 0.0
        for step in steps:
            windows.add([math.sin(time)], step)
            time += step
        return windows.mean(0)[0], windows.weight[0][0]

    rng = np.random.default_rng(3)
    even = run([0.02] * 1500)
    fine = run([0.01] * 3000)
    uneven = []
    while sum(uneven) < 30.0:
        uneven.append(float(rng.uniform(0.005, 0.04)))
    uneven[-1] -= sum(uneven) - 30.0
    for other in (fine, run(uneven)):
        assert other[0] == pytest.approx(even[0], abs=0.01)
        assert other[1] == pytest.approx(even[1], rel=0.01)


def test_level_quantiles_move_with_the_volume_and_their_differences_do_not():
    rng = np.random.default_rng(5)
    levels = rng.uniform(-40.0, -10.0, 4000)

    def quantiles(offset):
        distribution = LevelDistribution((math.inf,))
        for level in levels:
            distribution.add(level + offset, 0.02)
        return np.array(distribution.quantiles(0, (0.1, 0.5, 0.95)))

    here, lower = quantiles(0.0), quantiles(-17.3)
    assert here == pytest.approx(np.percentile(levels, (10, 50, 95)), abs=0.6)
    assert here - lower == pytest.approx(17.3, abs=0.05)
    assert np.diff(here) == pytest.approx(np.diff(lower), abs=0.05)
    assert all(math.isnan(v) for v in LevelDistribution((1.0,)).quantiles(0, (0.5,)))


def test_passages_far_under_the_loud_level_are_left_out_of_the_level_range():
    """A fade-out 45 dB down is heard at one volume and lost to the silence threshold at a
    lower one. The range is the same either way, because it never counted it."""

    def ranged(offset):
        tracker, time = SonicCharacter(), 0.0
        for level, seconds in ((-12.0, 20.0), (-24.0, 20.0), (-57.0, 8.0)):
            time = play(tracker, time, seconds, level=level + offset)
        profile = tracker.profile("whole")
        return next(m.mean for m in profile.measurements if m.name == "level_range_db")

    assert ranged(0.0) == pytest.approx(12.0, abs=0.5)  # not 45
    assert ranged(-20.0) == pytest.approx(ranged(0.0), abs=0.3)  # the tail is silence now


# -- the dimensions -----------------------------------------------------------------------


def test_the_scale_is_zero_to_one_and_a_profile_is_plain_serializable_data():
    tracker = SonicCharacter()
    play(tracker, 0.0, 30.0)
    for window in WINDOWS:
        profile = tracker.profile(window)
        assert [d.name for d in profile.dimensions] == list(DIMENSION_NAMES)
        for dimension in profile.dimensions:
            assert 0.0 <= dimension.value <= 1.0 and 0.0 <= dimension.confidence <= 1.0
            assert dimension.components  # the measurements it was made from are kept
        again = CharacterProfile.model_validate_json(profile.model_dump_json())
        assert again == profile
    assert tracker.profile("whole").window_s is None
    assert tracker.profile("long").window_s == WINDOWS["long"]


def test_brightness_follows_the_spectrum_not_the_volume():
    dark = heard(spectral_centroid_hz=300.0, spectrum=spectrum(tilt=-5.0))
    bright = heard(spectral_centroid_hz=1600.0, spectrum=spectrum(tilt=-1.0))
    assert dark["brightness"] < 0.15 and bright["brightness"] > 0.85
    loud_dark = heard(level=-6.0, spectral_centroid_hz=300.0, spectrum=spectrum(tilt=-5.0))
    assert loud_dark["brightness"] == pytest.approx(dark["brightness"], abs=1e-6)


def test_pace_is_onsets_and_regularity_is_the_pulse_and_they_are_separate():
    """Fast but loose is told from slower but tight; the tempo figure decides neither."""

    def rhythm(rate, clarity, jitter, bpm):
        tracker, rng, time, next_onset = SonicCharacter(), np.random.default_rng(7), 0.0, 0.0
        for index in range(round(40.0 / HOP)):
            time = index * HOP
            tracker.frame(
                sound(time, rhythmic_density=rate, tempo_confidence=clarity, tempo_bpm=bpm)
            )
            if time >= next_onset:
                tracker.onset(event(time, "onset", "observed"))
                next_onset += (1.0 / rate) * float(2.0 ** rng.uniform(-jitter, jitter))
        return values(tracker)

    fast_loose = rhythm(rate=5.0, clarity=0.2, jitter=2.2, bpm=180.0)
    slow_tight = rhythm(rate=2.0, clarity=0.65, jitter=0.0, bpm=120.0)
    assert fast_loose["pace"] > 0.9 and slow_tight["pace"] < 0.3
    assert fast_loose["regularity"] < 0.3 and slow_tight["regularity"] > 0.9
    # A high tempo figure with few onsets is not pace.
    assert rhythm(rate=1.0, clarity=0.4, jitter=0.2, bpm=200.0)["pace"] == 0.0


def test_regularity_without_onset_events_rests_on_the_pulse_and_says_so():
    tracker = SonicCharacter()
    play(tracker, 0.0, 40.0, tempo_confidence=0.65)
    profile = tracker.profile("long")
    regularity = profile.dimension("regularity")
    assert regularity.value == pytest.approx(1.0)  # the pulse alone
    assert regularity.confidence <= 0.6 + 1e-9  # ...which is 60% of what it would rest on
    assert any(w.startswith("no_onset_events") for w in profile.warnings)


def test_roughness_is_a_noise_like_percussive_spectrum():
    smooth = heard(spectral_contrast_db=(19.0,) * 7, spectral_flux=0.12, attack_energy_ratio=0.02)
    rough = heard(spectral_contrast_db=(11.0,) * 7, spectral_flux=0.24, attack_energy_ratio=0.07)
    assert smooth["roughness"] == pytest.approx(0.0, abs=1e-6)
    assert rough["roughness"] == pytest.approx(1.0, abs=1e-6)
    # The band above 6 kHz (empty in a lossy file) does not count.
    codec = heard(spectral_contrast_db=(19.0,) * 6 + (60.0,), spectral_flux=0.12, attack_energy_ratio=0.02)
    assert codec["roughness"] == pytest.approx(smooth["roughness"], abs=1e-6)


def test_loud_but_sparse_is_told_from_densely_layered():
    """The same level: one has a few peaks standing out of an empty spectrum and single
    sounds standing out of the rest; the other fills the spectrum evenly."""

    def sparse(time, **changes):
        hit = round(time / HOP) % 25 < 3  # a single sound twice a second
        return sound(time, level=-8.0 if hit else -30.0, spectrum=spectrum(-5.5, 12.0), **changes)

    open_one, crowded = SonicCharacter(), SonicCharacter()
    play(open_one, 0.0, 30.0, make=sparse)
    play(crowded, 0.0, 30.0, level=-10.0, spectrum=spectrum(tilt=-1.5))
    assert values(open_one)["density"] < 0.4 < 0.85 < values(crowded)["density"]


def test_dynamics_is_the_spread_between_quiet_and_powerful_passages():
    even = heard(60.0, "whole")
    assert even["dynamics"] == pytest.approx(0.0, abs=0.02)
    tracker, time = SonicCharacter(), 0.0
    for level in (-45.0, -15.0, -45.0, -15.0):
        time = play(tracker, time, 15.0, level=level)
    assert values(tracker, "whole")["dynamics"] > 0.95
    # One drum hit 20 dB up for 60 ms is not a powerful passage.
    tracker, time = SonicCharacter(), 0.0
    for _ in range(30):
        time = play(tracker, time, 0.94)
        time = play(tracker, time, 0.06, level=0.0)
    assert values(tracker, "whole")["dynamics"] < 0.12


def test_force_needs_a_saturated_sustained_sound_that_keeps_being_struck():
    forceful = heard(peak=0.25, bands=(0.3, 0.3, 0.1, 0.15, 0.08, 0.05, 0.02), spectral_flux=0.22)
    assert forceful["force"] > 0.85
    # The same sustained, saturated, bass-heavy sound, held without being struck: a pad.
    pad = heard(peak=0.25, bands=(0.3, 0.3, 0.1, 0.15, 0.08, 0.05, 0.02), spectral_flux=0.12)
    assert pad["force"] == 0.0
    gentle = heard(peak=0.4, bands=(0.05, 0.1, 0.3, 0.4, 0.1, 0.04, 0.01), spectral_flux=0.15)
    assert gentle["force"] < 0.3


def test_measurements_that_go_together_in_music_are_still_read_apart():
    """Brightness, roughness and density rest on different measurements: each can move
    while the other two stay exactly where they were."""
    base = heard()
    brighter = heard(spectral_centroid_hz=1800.0)
    rougher = heard(spectral_contrast_db=(11.0,) * 7)
    for other in ("roughness", "density", "force", "pace", "dynamics"):
        assert brighter[other] == pytest.approx(base[other], abs=1e-9), other
    assert brighter["brightness"] > base["brightness"] + 0.2
    for other in ("brightness", "density", "force", "pace", "dynamics"):
        assert rougher[other] == pytest.approx(base[other], abs=1e-9), other
    assert rougher["roughness"] > base["roughness"] + 0.15


# -- loudness and MusicalIntensity --------------------------------------------------------


def test_the_volume_alone_moves_no_dimension():
    quiet, loud = heard(level=-48.0), heard(level=-6.0)
    for name in DIMENSION_NAMES:
        assert quiet[name] == pytest.approx(loud[name], abs=1e-6), name


def test_the_same_character_at_different_intensity_and_the_reverse():
    """A passage 18 dB quieter, with the low end thinned out and fewer onsets, is far less
    intense, and the same kind of sound in every other respect."""
    full = dict(bands=(0.3, 0.3, 0.1, 0.15, 0.08, 0.05, 0.02), rhythmic_density=5.0)
    thin = dict(bands=(0.02, 0.1, 0.25, 0.35, 0.18, 0.08, 0.02), rhythmic_density=2.0)
    tracker, intensity, time = SonicCharacter(), MusicalIntensity(), 0.0
    seen = []
    for level, shape in ((-40.0, thin), (-22.0, full), (-40.0, thin), (-22.0, full)):
        for index in range(round(15.0 / HOP)):
            message = sound(time + index * HOP, level=level, **shape)
            tracker.frame(message)
            intensity.step(message)
        time += 15.0
        seen.append((intensity.value, values(tracker, "short")))
    assert seen[1][0] - seen[0][0] > 0.4 and seen[3][0] - seen[2][0] > 0.4  # intensity swings
    for name in ("brightness", "roughness", "regularity"):  # ...these do not move at all
        assert seen[0][1][name] == pytest.approx(seen[1][1][name], abs=1e-6), name
    # Different character at the same intensity: intensity sees the level, the low end
    # and the onsets, and all three are the same in these two.
    intensities, profiles = [], []
    for look in (
        dict(spectral_centroid_hz=350.0, spectrum=spectrum(-5.0, 10.0), spectral_contrast_db=(19.0,) * 7),
        dict(spectral_centroid_hz=1700.0, spectrum=spectrum(-1.0, 0.0), spectral_contrast_db=(11.0,) * 7),
    ):
        tracker, intensity = SonicCharacter(), MusicalIntensity()
        for index in range(round(30.0 / HOP)):
            message = sound(index * HOP, **look)
            tracker.frame(message)
            intensity.step(message)
        intensities.append(intensity.value)
        profiles.append(values(tracker))
    assert intensities[0] == pytest.approx(intensities[1], abs=1e-9)
    for name in ("brightness", "roughness", "density"):
        assert profiles[1][name] - profiles[0][name] > 0.25, name


# -- time ---------------------------------------------------------------------------------

DARK = dict(spectral_centroid_hz=300.0, spectrum=spectrum(tilt=-5.0))
BRIGHT = dict(spectral_centroid_hz=1600.0, spectrum=spectrum(tilt=-1.0))


def test_one_transient_does_not_move_the_identity():
    tracker = SonicCharacter()
    time = play(tracker, 0.0, 60.0, **DARK)
    before = {window: values(tracker, window) for window in WINDOWS}
    play(tracker, time, 0.06, level=0.0, spectral_flux=0.9, attack_energy_ratio=0.9, **BRIGHT)
    for window, most in (("short", 0.1), ("medium", 0.03), ("long", 0.01)):
        after = values(tracker, window)
        for name in ("brightness", "roughness", "density", "force", "pace", "regularity"):
            assert abs(after[name] - before[window][name]) < most, (window, name)


def test_a_real_change_is_followed_first_by_the_short_window_then_the_others():
    tracker = SonicCharacter()
    time = play(tracker, 0.0, 120.0, **DARK)
    time = play(tracker, time, 4.0, **BRIGHT)
    short, medium, long = (values(tracker, w)["brightness"] for w in ("short", "medium", "long"))
    assert short > 0.6 > medium > long > 0.02
    time = play(tracker, time, 16.0, **BRIGHT)  # twenty seconds in
    assert values(tracker, "short")["brightness"] > 0.95
    assert 0.6 < values(tracker, "medium")["brightness"] < 0.85
    assert values(tracker, "long")["brightness"] < 0.4  # the identity has not flipped
    play(tracker, time, 250.0, **BRIGHT)
    assert values(tracker, "long")["brightness"] > 0.9  # ...and comes round in the end
    assert 0.3 < values(tracker, "whole")["brightness"] < 0.85  # the whole recording was both


def test_confidence_grows_with_what_has_been_heard_and_is_nothing_before():
    tracker = SonicCharacter()
    start = tracker.profile("long")
    assert all(v is None for v in start.values()) and all(c == 0.0 for c in start.confidences())
    assert any(w.startswith("little_music") for w in start.warnings)
    time = play(tracker, 0.0, 2.0)
    early = tracker.profile("long")
    time = play(tracker, time, 60.0)
    late = tracker.profile("long")
    for name in DIMENSION_NAMES:
        if name != "regularity":  # no onset events in this stream
            assert early.dimension(name).confidence < late.dimension(name).confidence, name
    # What needs a long look is trusted later than what a second shows.
    assert early.dimension("brightness").confidence > 0.6 > 0.15 > early.dimension("dynamics").confidence
    assert late.dimension("brightness").confidence > 0.99 and late.dimension("dynamics").confidence > 0.85
    # A short window can never be sure of the dynamics: four seconds do not hold them.
    assert tracker.profile("short").dimension("dynamics").confidence < 0.25
    assert all(w.startswith("no_onset_events") for w in late.warnings)


def test_a_capture_near_the_silence_floor_is_trusted_less():
    quiet, normal = SonicCharacter(), SonicCharacter()
    play(quiet, 0.0, 40.0, level=-56.0)
    play(normal, 0.0, 40.0, level=-20.0)
    profile = quiet.profile("long")
    assert any(w.startswith("quiet_capture") for w in profile.warnings)
    assert profile.dimension("density").confidence == pytest.approx(
        0.5 * normal.profile("long").dimension("density").confidence
    )
    assert profile.dimension("density").value is not None  # quiet music is still analysed


def test_silence_and_a_pause_hold_the_profile_as_it_is():
    tracker = SonicCharacter()
    time = play(tracker, 0.0, 30.0, **BRIGHT)
    before = tracker.profile("long")
    for index in range(round(60.0 / HOP)):  # a minute of silence: the player is paused
        tracker.frame(sound(time + index * HOP, level=-90.0, silence=True))
    held = tracker.profile("long")
    assert held.values() == before.values() and held.confidences() == before.confidences()
    assert held.evidence_s == before.evidence_s and held.heard_s == before.heard_s
    assert not held.active and any(w.startswith("held") for w in held.warnings)
    # Only silence since the start: nothing is known and nothing evolves.
    idle = SonicCharacter()
    for index in range(500):
        idle.frame(sound(index * HOP, level=-90.0, silence=True))
    assert idle.heard == 0.0 and all(v is None for v in idle.profile("short").values())


def test_after_a_long_silence_the_old_character_is_kept_but_gives_way_faster():
    def after(silence_s):
        tracker = SonicCharacter()
        time = play(tracker, 0.0, 120.0, **DARK)
        kept = tracker.profile("long")
        time += silence_s
        time = play(tracker, time, HOP, **DARK)
        resumed = tracker.profile("long")
        play(tracker, time, 10.0, **BRIGHT)
        return kept, resumed, values(tracker)["brightness"]

    kept, resumed, short_gap = after(1.0)
    assert resumed.evidence_s == pytest.approx(kept.evidence_s, rel=0.01)
    kept, resumed, long_gap = after(BREAK_S + 1.0)
    assert resumed.values() == pytest.approx(kept.values(), abs=1e-3)  # still the best guess
    assert resumed.evidence_s == pytest.approx(LOOSEN * kept.evidence_s, rel=0.01)
    assert resumed.dimension("dynamics").confidence < kept.dimension("dynamics").confidence
    assert long_gap > short_gap + 0.15  # the new music takes over sooner


# -- segments, seeking, restarts ----------------------------------------------------------


def test_a_new_segment_starts_afresh_and_nothing_of_the_old_one_remains():
    tracker = SonicCharacter()
    time = play(tracker, 0.0, 90.0, segment=0, **DARK)
    generation = tracker.generation
    play(tracker, time, 12.0, segment=1, **BRIGHT)  # timestamps simply continue
    fresh = SonicCharacter()
    play(fresh, time, 12.0, segment=1, **BRIGHT)
    assert tracker.generation == generation + 1 and tracker.segment == 1
    for window in WINDOWS:
        assert tracker.profile(window) == fresh.profile(window)
    assert values(tracker, "whole")["brightness"] > 0.85


def test_a_message_of_a_segment_that_is_over_is_refused():
    tracker = SonicCharacter()
    time = play(tracker, 0.0, 20.0, segment=0, **DARK)
    time = play(tracker, time, 20.0, segment=1, **BRIGHT)
    before = tracker.profile("long")
    assert tracker.frame(sound(time - 0.5, segment=0, **DARK)) is False
    assert tracker.frame(sound(time + 0.5, segment=0, **DARK)) is False
    assert tracker.onset(event(time, "onset", "observed", segment=0)) is False
    assert tracker.profile("long") == before and tracker.segment == 1
    # Far back and an old segment: the timeline started again (a replay from the start).
    assert tracker.frame(sound(0.0, segment=0, **DARK)) is True
    assert tracker.segment == 0 and tracker.heard == pytest.approx(HOP)
    assert values(tracker, "whole")["brightness"] < 0.15


def test_a_seek_or_a_restart_keeps_nothing_stale():
    for target in (10.0, 400.0):  # back to an earlier place, far ahead
        tracker = SonicCharacter()
        play(tracker, 100.0, 60.0, **DARK)
        generation = tracker.generation
        play(tracker, target, 8.0, **BRIGHT)
        fresh = SonicCharacter()
        play(fresh, target, 8.0, **BRIGHT)
        assert tracker.generation == generation + 1
        assert tracker.profile("long") == fresh.profile("long")
    # The same frame twice is taken once.
    tracker = SonicCharacter()
    assert tracker.frame(sound(1.0)) is True and tracker.frame(sound(1.0)) is False
    assert tracker.frames == 1


def test_a_short_hole_in_the_timeline_is_not_music_and_not_a_restart():
    tracker = SonicCharacter()
    time = play(tracker, 0.0, 20.0)
    generation, heard_s = tracker.generation, tracker.heard
    tracker.frame(sound(time + 2.0))  # two seconds missing
    assert tracker.generation == generation
    assert tracker.heard - heard_s < 0.3  # the hole did not count as two seconds of music


# -- determinism --------------------------------------------------------------------------


def song(time, **changes):
    """A stream that varies smoothly with audio time alone."""
    return sound(
        time,
        level=-24.0 + 8.0 * math.sin(time / 7.0),
        spectral_centroid_hz=900.0 * 2.0 ** math.sin(time / 11.0),
        spectral_flux=0.18 + 0.05 * math.sin(time / 5.0),
        spectrum=spectrum(tilt=-3.0 + 1.5 * math.sin(time / 13.0)),
        **changes,
    )


def test_the_same_stream_gives_the_same_profile_every_time():
    def run():
        tracker = SonicCharacter()
        play(tracker, 0.0, 45.0, make=song)
        return {w: tracker.profile(w).model_dump_json() for w in WINDOWS}

    assert run() == run()


def test_reading_the_profile_changes_nothing():
    """A display may ask sixty or a hundred and forty-four times a second, or never."""
    asked, left = SonicCharacter(), SonicCharacter()
    for index in range(round(30.0 / HOP)):
        message = song(index * HOP)
        asked.frame(message)
        left.frame(message)
        for window in WINDOWS:
            asked.profile(window)
    assert asked.profile("long") == left.profile("long")
    assert asked.profile("short") == left.profile("short")


def test_the_profile_does_not_depend_on_how_often_frames_come():
    def run(times):
        tracker = SonicCharacter()
        for time in times:
            tracker.frame(song(time))
        return tracker

    even = run(np.arange(0.0, 60.0, 0.02))
    fine = run(np.arange(0.0, 60.0, 0.01))
    rng = np.random.default_rng(11)
    uneven = run(np.cumsum(rng.uniform(0.008, 0.045, 2300))[:2260])
    for other in (fine, uneven):
        for window in ("short", "medium", "long"):
            a, b = even.profile(window), other.profile(window)
            assert abs(a.time - b.time) < 0.6 or other is uneven
            for name in ("force", "brightness", "roughness", "density", "pace"):
                assert a.dimension(name).value == pytest.approx(b.dimension(name).value, abs=0.03), (
                    window,
                    name,
                )


# -- real analysis, sessions and replay ---------------------------------------------------


def dense_audio(seconds=14.0, level=0.25, seed=1):
    """Broadband noise with a heavy low end, struck eight times a second."""
    rng = np.random.default_rng(seed)
    count = round(seconds * RATE)
    time = np.arange(count) / RATE
    noise = rng.standard_normal(count)
    low = np.sin(2 * np.pi * 70 * time) + 0.6 * np.sin(2 * np.pi * 140 * time)
    strike = 0.55 + 0.45 * (np.modf(time * 8.0)[0] < 0.25)
    audio = np.tanh(2.5 * (0.5 * noise + 1.2 * low) * strike) * level
    return np.column_stack((audio, audio)).astype(np.float32)


def sparse_audio(seconds=14.0, level=0.25):
    """A soft low tone that swells slowly, with long gaps of very little."""
    time = np.arange(round(seconds * RATE)) / RATE
    swell = 0.04 + 0.96 * np.clip(np.sin(2 * np.pi * time / 7.0), 0, 1) ** 2
    audio = level * swell * (np.sin(2 * np.pi * 196 * time) + 0.3 * np.sin(2 * np.pi * 392 * time))
    return np.column_stack((audio, audio)).astype(np.float32)


def analysed(audio, tmp_path, name="audio.wav"):
    path = tmp_path / name
    sf.write(path, audio, RATE, subtype="FLOAT")
    return diagnose.analyse(*diagnose.audio_messages(path)), path


def overall(result):
    return {d["name"]: d["value"] for d in result["overall"]["dimensions"]}


def test_real_analysis_tells_a_dense_wall_from_a_sparse_soft_tone(tmp_path):
    dense, _ = analysed(dense_audio(), tmp_path, "dense.wav")
    sparse, _ = analysed(sparse_audio(), tmp_path, "sparse.wav")
    wall, tone = overall(dense), overall(sparse)
    for name in ("force", "brightness", "roughness", "density"):
        assert wall[name] > tone[name] + 0.3, name
    assert tone["dynamics"] > wall["dynamics"] + 0.3
    assert dense["cost"]["character_us_per_frame"] < 1000  # far under the analysis pass


def test_the_same_audio_eighteen_decibels_quieter_gives_the_same_profile(tmp_path):
    audio = np.concatenate((dense_audio(8.0), dense_audio(8.0, seed=2) * 0.4))
    loud, path = analysed(audio, tmp_path)
    quiet = diagnose.analyse(*diagnose.audio_messages(path, gain_db=-18.0))
    moved = diagnose.level_check(loud, quiet)
    assert max(moved.values()) < 0.02, moved
    assert quiet["source"]["gain_db"] == -18.0


def test_a_recorded_session_gives_back_the_profile_of_the_live_run(tmp_path):
    """The profile is not recorded: it is rebuilt from the recorded frames, exactly."""
    path = tmp_path / "music.wav"
    sf.write(path, np.concatenate((dense_audio(6.0), sparse_audio(6.0))), RATE, subtype="FLOAT")
    engine = Engine(FileAudioSource(path, realtime=False), session_root=tmp_path / "sessions")
    subscriber = engine.bus.subscribe(8192)
    engine.start()
    assert engine.done.wait(30) and not engine.error
    live = SonicCharacter()
    trail = []
    while len(subscriber):
        message = subscriber.get()
        if live.consume(message) and isinstance(message, FeatureFrame):
            trail.append(live.profile("short").values())
    assert subscriber.dropped == 0 and live.heard > 10.0
    # From the session's log, read straight back.
    messages, source, _ = diagnose.session_messages(engine.writer.path)
    logged, logged_trail = SonicCharacter(), []
    for message in messages:
        if logged.consume(message) and isinstance(message, FeatureFrame):
            logged_trail.append(logged.profile("short").values())
    assert source["kind"] == "recorded_session"
    # Through the replay the application uses, at two speeds.
    for speed in (40.0, 400.0):
        bus = EventBus()
        queue = bus.subscribe(8192)
        replay(engine.writer.path / "messages.jsonl", bus, Event(), speed=speed)
        replayed, replayed_trail = SonicCharacter(), []
        while len(queue):
            message = queue.get()
            if replayed.consume(message) and isinstance(message, FeatureFrame):
                replayed_trail.append(replayed.profile("short").values())
        assert queue.dropped == 0
        for window in WINDOWS:
            assert replayed.profile(window) == live.profile(window) == logged.profile(window)
        assert replayed_trail == trail == logged_trail  # the whole evolution, not only the end


def test_the_diagnostics_write_json_a_report_and_charts(tmp_path, capsys):
    paths = []
    for name, audio in (
        ("wall", np.concatenate((dense_audio(16.0), sparse_audio(16.0)))),
        ("tone", sparse_audio(14.0)),
    ):
        paths.append(tmp_path / f"{name}.wav")
        sf.write(paths[-1], audio, RATE, subtype="FLOAT")
    listing = tmp_path / "list.txt"
    listing.write_text(f"# two recordings\n\n{paths[1]}\n", encoding="utf-8")
    out = tmp_path / "out"
    gmv_main(
        ["character", "analyze", str(paths[0]), "--list", str(listing), "--out", str(out), "--gain-db", "-12"]
    )
    assert "Analysed 2 of 2" in capsys.readouterr().out
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["recordings"]) == {"wall", "tone"} and not summary["not_analysed"]
    assert set(summary["dimensions"]) == set(DIMENSION_NAMES)
    assert len(summary["level_checks"]) == 2
    one = json.loads((out / "wall.json").read_text(encoding="utf-8"))
    assert one["overall"]["profile_version"] == "0.1"
    assert set(one["windows_at_end"]) == {"short", "medium", "long"}
    assert len(one["evolution"]["time"]) == len(one["evolution"]["medium"]) > 25
    assert len(one["evolution"]["medium"][0]) == len(DIMENSION_NAMES)
    assert one["distributions"]["flux"]["percentiles"]["p50"] is not None
    # The two halves are told apart as sections, each with its own profile.
    assert len(one["sections"]) == 2 and 12 < one["sections"][1]["start_s"] < 20
    first, second = (s["values"] for s in one["sections"])
    assert first["density"] > second["density"] + 0.3
    report = (out / "report.md").read_text(encoding="utf-8")
    for needle in ("## Overall profiles", "## wall", "## tone", "The same audio at another level", "Sections"):
        assert needle in report
    assert (out / "wall.svg").read_text(encoding="utf-8").startswith("<svg")
    assert (out / "tone.svg").exists()
    # The audio itself was only read.
    assert not list(out.rglob("*.wav")) and sf.info(paths[0]).frames == 32 * RATE


def test_the_diagnostics_say_what_they_could_not_analyse(tmp_path, capsys):
    good, bad = tmp_path / "good.wav", tmp_path / "broken.wav"
    sf.write(good, dense_audio(4.0), RATE, subtype="FLOAT")
    bad.write_bytes(b"not audio")
    with pytest.raises(SystemExit):
        gmv_main(["character", "analyze", str(good), str(bad), "--out", str(tmp_path / "out")])
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert list(summary["recordings"]) == ["good"] and str(bad) in summary["not_analysed"]
    assert any(w.startswith("short_recording") for w in summary["recordings"]["good"]["warnings"])
    with pytest.raises(SystemExit, match="Not found"):
        gmv_main(["character", "analyze", str(tmp_path / "missing.wav")])
    with pytest.raises(SystemExit, match="at least one"):
        character_cli.run(SimpleNamespace(paths=[], listing=None, out=str(tmp_path), gain_db=None, processes=1))


def test_it_is_cheap_beside_the_analysis_pass():
    tracker, frames = SonicCharacter(), [song(index * HOP) for index in range(3000)]
    began = perf_counter()
    for message in frames:
        tracker.frame(message)
    each = (perf_counter() - began) / len(frames)
    assert each < 0.0005  # generous: it measures about 0.00004 s
