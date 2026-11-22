"""The responsiveness diagnostic: its arithmetic, the analyzers' own latencies it reports,
and the accepted guitar articulation as a regression reference."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

import synth_instruments as synth
from gmv.evaluation import responsiveness as r
from gmv.models import AnalysisConfig

from test_instrument_layers import guitar

RATE = synth.RATE


def rows(*found):
    """Rows of ``r.FIELDS`` from (time, known, level, attack) tuples."""
    out = np.zeros((len(found), len(r.FIELDS)))
    out[:, r.F["pitch"]] = np.nan
    for index, (time, known, level, attack) in enumerate(found):
        out[index, [r.F["time"], r.F["known"], r.F["level"], r.F["attack"]]] = time, known, level, attack
    return out


def test_onsets_are_matched_once_each_and_both_delays_are_kept_apart():
    table = rows((0.98, 1.05, 0.2, 0.0), (1.0, 1.09, 0.8, 0.9), (1.5, 1.6, 0.8, 0.0), (2.03, 2.1, 0.9, 0.7), (2.5, 2.6, 0.3, 0.6))
    found = r.match(table, [1.0, 1.02, 2.0, 3.0])
    assert (found["found"], found["missed"], found["extra"]) == (2, 2, 1)  # one event, one onset
    assert found["placed_s"]["p50"] == pytest.approx(0.015) and found["placed_s"]["n"] == 2
    assert found["known_s"]["p50"] == pytest.approx(0.095)  # (0.09 + 0.10) / 2
    # A continuous signal: moved at 1.0 (0.2 -> 0.8) and not at 1.5 (flat).
    steady = rows((0.96, 1.03, 0.2, 0.0), (1.0, 1.09, 0.8, 0.0), (1.46, 1.5, 0.8, 0.0), (1.5, 1.6, 0.8, 0.0), (1.7, 1.8, 0.8, 0.0))
    moved = r.movement(steady, [1.0, 1.5])
    assert moved["moved"] == 1 and moved["known_s"]["p50"] == pytest.approx(0.09)
    assert r.spread([])["p50"] is None and r.spread([1.0, np.nan])["n"] == 1


def test_dips_say_whether_repeated_notes_can_be_seen_in_a_level():
    time = np.arange(0, 3, 0.02)
    flat = rows(*[(t, t, 0.8, 0.0) for t in time])
    pulsing = rows(*[(t, t, 0.8 * np.exp(-((t % 0.5) / 0.1)), 0.0) for t in time])
    onsets = [0.5, 1.0, 1.5, 2.0]
    assert r.dips(flat, onsets)["p50"] == pytest.approx(0.0)
    assert r.dips(pulsing, onsets)["p50"] > 0.9


def test_every_analyzer_reports_fifty_times_a_second_with_its_own_fixed_latency():
    seconds = 4.0
    beat = np.arange(0.5, 3.5, 0.5)
    audio = (
        synth.kick(seconds, beat)
        + synth.bass(seconds, [(t, 28, 0.45) for t in beat])
        + synth.voice(seconds, [(0.5, 2.5, 60, 62)])
        + guitar(seconds, np.arange(0.5, 3.5, 0.25))
    )
    names = ("guitar", "drums", "bass", "vocals")
    found, times = r.trace(audio, RATE, AnalysisConfig(instrument_analyzers=names, spectrum_bands=0))
    # What each needs after the moment it describes: half the 46 ms analysis window, plus
    # one frame (guitar: a local maximum), three (drums: the decay check) or four (bass:
    # half its 170 ms pitch window). Vocals describe the newest frame.
    expected = {"vocals": 0.023, "guitar": 0.043, "drums": 0.083, "bass": 0.103}
    for name in names:
        cadence = r.cadence(found[name])
        assert cadence["per_second"] == pytest.approx(50.0, abs=0.2)
        assert cadence["interval_s"]["p50"] == pytest.approx(0.02) == cadence["interval_s"]["p95"]
        assert cadence["latency_s"]["p50"] == pytest.approx(expected[name], abs=0.002)
        assert cadence["latency_s"]["p95"] == pytest.approx(expected[name], abs=0.002)
    assert np.isnan(found["bass"][:, r.F["present"]]).all()  # no activity model in this run
    assert len(times) > 150 and float(np.median(times)) < 5.0


def test_the_accepted_guitar_articulation_is_prompt_and_complete():
    # The reference the redesign must not regress: ten strokes a second, every one
    # reported, placed within a frame of where it was played and known 46 ms after it.
    strokes = [round(float(t), 3) for t in np.arange(1.0, 5.0, 0.1)]
    config = AnalysisConfig(instrument_analyzers=("guitar",), spectrum_bands=0)
    found, _ = r.trace(guitar(6.0, strokes), RATE, config)
    result = r.match(found["guitar"], strokes)
    assert result["found"] == len(strokes) and result["extra"] == 0
    assert abs(result["placed_s"]["p50"]) <= 0.01 and abs(result["placed_s"]["p95"]) <= 0.02
    assert result["known_s"]["p95"] <= 0.06
    # Slower strokes too, and the level falls back between them.
    slow = [round(float(t), 3) for t in np.arange(1.0, 5.0, 1 / 3)]
    found, _ = r.trace(guitar(6.0, slow, decay=0.25), RATE, config)
    assert r.match(found["guitar"], slow)["found"] == len(slow)
    assert r.dips(found["guitar"], slow)["p50"] > 0.2


def test_drum_regions_share_one_confirmation_delay():
    hits = [round(float(t), 3) for t in np.arange(1.0, 4.0, 0.5)]
    config = AnalysisConfig(instrument_analyzers=("drums",), spectrum_bands=0)
    hat, _ = r.trace(synth.hat(5.0, hits), RATE, config)
    kick, _ = r.trace(synth.kick(5.0, hits), RATE, config)
    high, low = r.match(hat["drums"], hits, "high"), r.match(kick["drums"], hits, "low")
    assert high["found"] == len(hits) and low["found"] >= len(hits) - 1
    # Only the low region needs the 60 ms decay check; the high region waits for it too.
    assert high["known_s"]["p50"] == pytest.approx(low["known_s"]["p50"], abs=0.011)
    assert 0.07 < high["known_s"]["p50"] < 0.1


def test_the_diagnostic_is_not_part_of_the_application():
    source = Path(__file__).resolve().parents[1] / "src" / "gmv"
    users = [
        path.relative_to(source).as_posix()
        for path in source.rglob("*.py")
        if "responsiveness" in path.read_text(encoding="utf-8") and path.name != "responsiveness.py"
    ]
    assert users == []
