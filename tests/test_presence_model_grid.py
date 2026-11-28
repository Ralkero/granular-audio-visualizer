"""A5.2 Step 1, S1.3: the canonical segment and frame grid (generated values only).

The 44.1 kHz expectations are the T-grid table from the readiness proposal; the 48 kHz ones
come from an independent plain-Python computation of the engine's rule.
"""

import math

import numpy as np
import pytest

from gmv.character.research.presence_model import frontend, grid

RECORDING_S = 20.0


def independent(rate, start, length=10.0, total=None):
    """first, f0, c(f0) and the excerpt frame count, computed without the grid module."""
    total = round(RECORDING_S * rate) if total is None else total
    size, hop = round(rate * 0.04644), round(rate * 0.02)
    first = round(max(0.0, start - 5.0) * rate)
    end = min(total, round((start + length) * rate) + size)
    frames = (end - first - size) // hop + 1
    centres = [(first + i * hop + size / 2) / rate - start for i in range(frames)]
    f0 = sum(1 for c in centres if c < 0)
    return first, f0, centres[f0], sum(1 for c in centres if 0 <= c < length)


@pytest.mark.parametrize(("start", "first", "f0", "c0", "s0"), (
    (3.013, 0, 150, 0.0102200, 75),
    (0.5, 0, 24, 0.0032200, 0),
    (7.0, 88200, 249, 0.0032200, 174),
    (0.0, 0, 0, 0.0232200, 0),
))
def test_the_t_grid_at_44_1_khz(start, first, f0, c0, s0):
    seg = grid.canonical(round(RECORDING_S * 44100), 44100, start, start + 10.0)
    assert (seg.first, seg.f0, seg.context_start()) == (first, f0, s0)
    assert math.isclose(seg.centres[seg.f0], c0, abs_tol=5e-8)
    # 500 excerpt frames, except at the recording start: the first centre is 23.2 ms in
    assert seg.f1 - seg.f0 == (499 if start == 0.0 else 500)
    assert seg.c_rows() == slice(s0, seg.f1)


@pytest.mark.parametrize("start", (3.013, 0.5, 7.0, 0.0, 9.345))
@pytest.mark.parametrize("rate", (44100, 48000))
def test_the_grid_matches_an_independent_computation(rate, start):
    seg = grid.canonical(round(RECORDING_S * rate), rate, start, start + 10.0)
    first, f0, c0, inside = independent(rate, start)
    assert (seg.first, seg.f0, seg.f1 - seg.f0) == (first, f0, inside)
    assert seg.centres[seg.f0] == pytest.approx(c0, abs=1e-12)


def test_a_naive_1_5_s_cut_lands_on_a_different_grid():
    rate, start = 44100, 3.013
    seg = grid.canonical(round(RECORDING_S * rate), rate, start, start + 10.0)
    first, end = frontend.segment(round(RECORDING_S * rate), rate, start, start + 10.0, preroll_s=1.5)
    assert first == 66723 and (first - seg.first) % frontend.engine_hop(rate) == 573
    naive = frontend.frame_centres(frontend.frame_count(end - first, rate), rate, first / rate) - start
    naive_first = naive[np.count_nonzero(naive < 0)]
    assert math.isclose(seg.centres[seg.f0] - naive_first, 0.0070068, abs_tol=1e-6)
    confirmed = [(2.5, 6.0)]
    ours = grid.frame_arrays(seg, confirmed)["training"]
    theirs = grid.frame_arrays(grid.Segment(rate, seg.total_samples, start, start + 10.0, first, end,
                                            len(naive), naive, int(np.count_nonzero(naive < 0)),
                                            int(np.count_nonzero(naive < 10.0))), confirmed)["training"]
    assert ours.tobytes() != theirs.tobytes()


def test_an_excerpt_ending_at_the_recording_end_has_no_padded_frames():
    rate, start = 44100, 10.0
    total = round(20.0 * rate)
    seg = grid.canonical(total, rate, start, 20.0)
    *_rest, inside = independent(rate, start, total=total)
    assert seg.f1 - seg.f0 == inside < 500
    assert (seg.centres < seg.duration).sum() == seg.f1


def test_the_hop_must_be_exactly_20_ms():
    assert all(grid.hop_is_20ms(r) for r in (22050, 32000, 44100, 48000, 88200, 96000))
    assert not grid.hop_is_20ms(11025)
    with pytest.raises(grid.GridError):
        grid.canonical(11025 * 20, 11025, 6.0, 16.0)


def test_training_frames_exclude_pre_roll_collars_and_quiet_frames():
    seg = grid.canonical(round(RECORDING_S * 44100), 44100, 7.0, 17.0)
    quiet = np.zeros(seg.frames, dtype=bool)
    quiet[seg.f0 + 300 : seg.f0 + 310] = True  # 6.0-6.2 s, clear of every collar
    arrays = grid.frame_arrays(seg, [(2.0, 4.0)], quiet=quiet)
    training = arrays["training"]
    assert not training[: seg.f0].any() and not training[seg.f1 :].any()
    assert not training[seg.f0 + 300 : seg.f0 + 310].any()
    collar = np.abs(seg.centres - 2.0) <= 0.3 + 1e-9
    assert collar.any() and not training[collar].any()
    assert arrays["evaluation"][seg.f0 + 300 : seg.f0 + 310].all()  # evaluation keeps quiet frames
    assert not arrays["evaluation"][collar].any()
    with pytest.raises(grid.GridError):
        grid.frame_arrays(seg, [], quiet=quiet[:-1])
