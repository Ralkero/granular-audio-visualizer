import numpy as np
import pytest

from gmv.clock_analysis import analyze_clock

RATE, BLOCK = 48000, 480


def arrivals(count, *, skew_ppm=0.0, jitter_ms=0.0, seed=1):
    ideal = np.arange(1, count + 1) * (BLOCK / RATE) * (1 + skew_ppm / 1e6)
    jitter = np.random.default_rng(seed).exponential(jitter_ms / 1000, count) if jitter_ms else 0
    return 100 + ideal + jitter


def test_jitter_alone_reports_no_drift():
    result = analyze_clock(arrivals(12000, jitter_ms=2), np.full(12000, BLOCK), RATE)
    assert result["step_count"] == 0
    assert abs(result["envelope_drift_ms"]) < 1
    assert abs(result["smooth_slope_ms_per_min"]) < 0.5


def test_rate_mismatch_is_a_smooth_slope_not_steps():
    result = analyze_clock(arrivals(12000, skew_ppm=2000, jitter_ms=1), np.full(12000, BLOCK), RATE)
    assert result["step_count"] == 0
    assert result["smooth_slope_ppm"] == pytest.approx(2000, rel=0.05)
    assert result["envelope_drift_ms"] == pytest.approx(240, rel=0.05)
    assert result["effective_rate"] == pytest.approx(RATE / 1.002, rel=1e-4)


def test_lost_blocks_are_steps_not_slope():
    times = arrivals(12000, jitter_ms=1)
    for lost_after, blocks in ((3000, 1), (7000, 3), (9000, 2)):
        times[lost_after:] += blocks * BLOCK / RATE
    result = analyze_clock(times, np.full(12000, BLOCK), RATE)
    assert result["step_count"] == 3
    assert result["step_total_ms"] == pytest.approx(60, abs=2)
    assert abs(result["smooth_slope_ms_per_min"]) < 0.5


def test_long_outage_is_counted_apart_from_drift():
    times = arrivals(6000)
    times[3000:] += 5
    result = analyze_clock(times, np.full(6000, BLOCK), RATE)
    assert result["outage_count"] == 1
    assert result["outage_total_s"] == pytest.approx(5, abs=0.01)
    assert result["step_count"] == 0 and abs(result["envelope_drift_ms"]) < 1
