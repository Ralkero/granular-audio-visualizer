"""A5.2 Step 1, S1.2: which sample rates B0 supports (L1-L4 on generated audio).

The expected classes come from reading ``activity.FrameFeatures``: construction needs a bin
at or above the top band's lower edge (about 11.86 kHz, so about 23.72 kHz); computation
needs ``searchsorted`` of 16 kHz to stay inside the spectrum (about 32 kHz). These tests
check those derivations by running the real code; a failure here is evidence, not noise.
Exactly 32 kHz is not assumed: its class is checked against the same index arithmetic.
"""

import numpy as np
import pytest

from gmv.activity import BAND_EDGES_HZ
from gmv.character.research.presence_model import b0check, frontend

NOT_CONSTRUCTIBLE = (8000, 16000, 22050, 23600, 23700)
NOT_COMPUTABLE = (23720, 23740, 23800, 24000, 31900, 31990, 31999)
SUPPORTED = (32001, 32050, 44100, 48000, 88200, 96000)


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    return b0check.write_weights(tmp_path_factory.mktemp("b0") / "generated.npz")


@pytest.fixture(scope="module")
def results(weights):
    return {rate: b0check.classify(rate, weights) for rate in b0check.SCAN_RATES}


def test_the_scan_covers_every_listed_rate(results):
    assert set(NOT_CONSTRUCTIBLE + NOT_COMPUTABLE + SUPPORTED) | {32000} == set(b0check.SCAN_RATES) == set(results)


@pytest.mark.parametrize("rate", NOT_CONSTRUCTIBLE)
def test_rates_below_the_construction_boundary_are_not_constructible(results, rate):
    assert results[rate]["classification"] == "not_constructible"
    assert results[rate]["levels"]["L1"]["reason"] == "ValueError"


@pytest.mark.parametrize("rate", NOT_COMPUTABLE)
def test_rates_between_the_boundaries_construct_but_cannot_compute(results, rate):
    levels = results[rate]["levels"]
    assert results[rate]["classification"] == "not_computable"
    assert levels["L1"]["passed"] and levels["L2"]["reason"] == "IndexError"


@pytest.mark.parametrize("rate", SUPPORTED)
def test_supported_rates_pass_every_level(results, rate):
    assert results[rate]["classification"] == "supported", results[rate]["levels"]


def computable_by_index_arithmetic(rate):
    size = frontend.engine_window(rate)
    freq = np.fft.rfftfreq(size, 1 / rate)
    return int(np.searchsorted(freq, BAND_EDGES_HZ)[-1]) < len(freq)


@pytest.mark.parametrize("rate", b0check.SCAN_RATES)
def test_every_class_agrees_with_the_index_arithmetic(results, rate):
    if results[rate]["levels"]["L1"]["passed"]:
        assert computable_by_index_arithmetic(rate) == results[rate]["levels"]["L2"]["passed"]


def test_exactly_32_khz_is_classified_not_assumed(results):
    """The class follows the index arithmetic, whichever way it falls. Observed in Step 1:
    ``rfftfreq`` gives the top bin at 32 kHz as 15999.999999999998 Hz, just below the exact
    16 kHz band edge, so 32 kHz itself is not computable; 32,001 Hz is."""
    assert results[32000]["classification"] in ("not_computable", "supported")
    assert (results[32000]["classification"] == "supported") == computable_by_index_arithmetic(32000)


def test_the_silent_fallback_is_detected_as_a_failure(results):
    assert results[22050]["levels"]["L4"]["reason"] == "silent fallback: no activity score"


def test_the_l4_fixture_contains_quiet_frames_and_matches_its_layout():
    for rate in (44100, 48000):
        audio = b0check.l4_audio(rate, "stereo")
        assert audio.shape == (3 * rate, 2) and audio.dtype == np.float32
        silent = audio[rate : rate + rate // 2]
        assert not silent.any()
        assert np.isclose(np.abs(audio[rate + rate // 2 :, 0]).max(), b0check.TONE_PEAK, rtol=1e-3)
        assert frontend.frame_count(len(audio), rate) == 148


def test_unsupported_paired_rates_stop(weights):
    b0check.require_supported([44100, 48000, 44100], weights)
    with pytest.raises(b0check.UnsupportedRate):
        b0check.require_supported([44100, 24000], weights)


def test_strict_mode_turns_warnings_and_floating_point_errors_into_failures():
    with pytest.raises(FloatingPointError):
        with b0check.strict():
            np.log10(np.zeros(1))
    with pytest.raises(RuntimeWarning):
        with b0check.strict():
            import warnings

            warnings.warn("x", RuntimeWarning, stacklevel=1)
