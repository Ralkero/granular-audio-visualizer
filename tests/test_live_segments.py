"""Live states and audio segments, and the bass ring's phase as a function of audio time."""

import math
import os
import pickle
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from gmv.contract import parse_message
from gmv.visualizer import layers as layer_registry
from gmv.visualizer.layers.bass import FLOW_BASE, FLOW_PER_ENERGY
from gmv.visualizer.layers.drums import sparks
from gmv.visualizer.live import STALE_S, Live
from gmv.visualizer.state import COLUMN, VisualState

from test_instrument_layers_phase2 import activity
from test_visualizer import frame

HOP = 0.02


# -- segments ------------------------------------------------------------------------------


class Stream:
    """Feeds a ``VisualState`` the way the engine publishes: each analysis frame, then the
    activity rows of that frame, back-dated by each analyzer's delay."""

    def __init__(self, state=None):
        self.state = state or VisualState()
        self.clock = perf_counter()
        self.time = 1.0 - HOP  # the first frame is at 1 s: back-dated rows stay positive

    def frame(self, segment, time=None):
        self.time = self.time + HOP if time is None else time
        self.state.consume(frame(self.time, segment=segment), arrival=self.clock + self.time)
        return self.time

    def bass(self, segment, **values):
        base = dict(level=0.6, energy=0.6, lead_s=0.08, segment=segment)
        self.state.consume(activity(self.time - 0.08, "bass", **(base | values)))

    def drums(self, segment, strength=0.0):
        self.state.consume(
            activity(
                self.time - 0.06,
                "drums",
                attack=strength,
                flux=strength,
                components=(("low", strength), ("mid", 0.0), ("high", 0.0)),
                segment=segment,
            )
        )

    def vocals(self, segment, **values):
        self.state.consume(activity(self.time, "vocals", **({"segment": segment} | values)))


def play_segment(stream, segment, frames, start=None):
    """Bass with a held note, pulses and attacks; drum hits; a vocal phrase begun."""
    for index in range(frames):
        stream.frame(segment, start if index == 0 and start is not None else None)
        stream.bass(
            segment,
            pitch_midi=33.0,
            pitch_confidence=0.9,
            pulse=0.9 * (index % 25 == 5),
            attack=0.8 * (index % 25 == 5),
            flux=0.8 * (index % 25 == 5),
        )
        stream.drums(segment, 0.9 * (index % 25 == 10))
        stream.vocals(segment, level=0.7, pitch_midi=60.0, boundary="start" if index == 3 else "")


def present(state, key):
    live = state.live[key]
    live.advance(state.latest.timestamp + 0.005)
    return live


def test_a_new_segment_with_continuous_timestamps_starts_the_live_states_afresh():
    stream = Stream()
    play_segment(stream, 1, 100)
    state = stream.state
    bass, drums, vocals = (present(state, key) for key in ("bass", "drums", "vocals"))
    assert bass.energy > 0.5 and bass.hits and bass.pulses and bass.pitched
    assert drums.hits and vocals.sounding and vocals.pitched
    epoch, last = state.live_epoch, stream.time
    # Segment 2 carries on 20 ms later, as if nothing had happened.
    stream.frame(2)
    assert stream.time == pytest.approx(last + HOP) and state.live_epoch == epoch + 1
    assert all(live.time is None and live.segment == 2 for live in state.live.values())
    # Its first row: a quiet low register with no note yet and no onset.
    stream.bass(2, level=0.1, energy=0.1, flux=0.05)
    bass = present(state, "bass")
    assert bass.segment == 2 and bass.origin == pytest.approx(stream.time - 0.08)
    assert not bass.hits and not bass.pulses and bass.beats(5.0) == []
    assert bass.energy == pytest.approx(0.1) and bass.level == pytest.approx(0.1)
    assert not bass.pitched and bass.pitch is None and bass.place == 0.5
    # The ringing envelope the history derives starts again too: no old swell carried over.
    assert 0.04 < bass.ring <= 0.05  # its own flux, 5 ms into dying away
    assert state.activity["bass"].last[COLUMN["ring"]] == pytest.approx(0.05)
    assert state.activity["bass"].last[COLUMN["struck_time"]] < 0
    # Instruments with no row of the new segment yet show nothing of the old one.
    for key in ("drums", "vocals"):
        live = state.live[key]
        live.advance(stream.time)
        assert not live.recent(5.0) and not live.sounding and live.level == 0.0
    stream.vocals(2, level=0.4)
    assert present(state, "vocals").level == pytest.approx(0.4) and not state.live["vocals"].sounding


def test_a_row_of_a_segment_that_is_over_cannot_come_back_into_the_live_state():
    stream = Stream()
    play_segment(stream, 1, 60)
    play_segment(stream, 2, 30)
    state = stream.state
    bass = present(state, "bass")
    before = (list(bass.hits), list(bass.pulses), bass._pitch, bass._energy, bass.time, bass.origin)
    rows = len(state.activity["bass"])
    # Late, newer in time than anything seen, and loud: still segment 1.
    state.consume(
        activity(stream.time + 0.01, "bass", level=1.0, energy=1.0, pulse=1.0, attack=1.0, pitch_midi=50.0, segment=1)
    )
    bass = present(state, "bass")
    after = (list(bass.hits), list(bass.pulses), bass._pitch, bass._energy, bass.time, bass.origin)
    assert after == before and bass.segment == 2
    # It is refused as a late row, as a row out of time order always was: it neither
    # enters the history nor blocks the current segment's next rows from it.
    assert len(state.activity["bass"]) == rows
    state.consume(activity(stream.time - 1.0, "bass", level=1.0, attack=1.0, segment=1))
    assert len(state.activity["bass"]) == rows
    # The current segment's rows carry on normally after it.
    stream.frame(2)
    stream.bass(2, attack=0.7, pulse=0.7)
    assert present(state, "bass").hits[-1].strength == pytest.approx(0.7)


def test_the_history_keeps_both_segments_and_only_the_live_state_is_renewed():
    stream = Stream()
    play_segment(stream, 1, 100)
    boundary, generation = stream.time, stream.state.generation
    play_segment(stream, 2, 100)
    state = stream.state
    rows = state.activity["bass"].since()
    times = rows[:, COLUMN["time"]]
    assert (times < boundary - 0.08).sum() >= 95 and (times > boundary).sum() >= 95
    assert np.all(np.diff(times) > 0)
    struck = rows[rows[:, COLUMN["attack"]] > 0, COLUMN["time"]]
    assert (struck < boundary).sum() == 4 and (struck > boundary).sum() == 4
    # The lanes and strips draw from these same rows: nothing of segment 1 was removed.
    drums = state.activity["drums"].since()
    hits = drums[drums[:, COLUMN["attack"]] > 0, COLUMN["time"]]
    assert len(drums) == 200 and (hits < boundary).sum() == 4 and (hits > boundary).sum() == 4
    assert state.generation == generation  # no full reset happened
    # The live states know only segment 2.
    for live in state.live.values():
        assert live.segment == 2
        assert all(hit.time > boundary - 0.1 for hit in live.hits)
        assert live.origin > boundary - 0.1


def test_a_gap_or_a_restarted_timeline_starts_afresh_exactly_once():
    stream = Stream()
    play_segment(stream, 1, 60)
    state = stream.state
    epoch, generation = state.live_epoch, state.generation
    # A new segment after a two-second gap: one fresh start, the hole is shown.
    play_segment(stream, 2, 40, start=stream.time + 2.0)
    assert state.live_epoch == epoch + 1 and state.generation == generation
    assert state.gaps and present(state, "bass").segment == 2 and present(state, "bass").hits
    # The timeline goes back (a new stream): the existing full reset, once, even though
    # the segment changes in the same frame.
    epoch, generation = state.live_epoch, state.generation
    play_segment(stream, 3, 40, start=0.2)
    assert state.generation == generation + 1 and state.live_epoch == epoch + 1
    assert state.activity["bass"].since()[0, COLUMN["time"]] < 1.0  # history from the restart
    bass = present(state, "bass")
    assert bass.segment == 3 and bass.hits and bass.energy > 0.5


def test_rows_without_a_segment_still_count_as_the_current_one():
    stream = Stream()
    state = stream.state
    for index in range(50):  # frames name their segment; these rows were logged without one
        stream.frame(4)
        state.consume(activity(stream.time, "vocals", level=0.6, attack=0.5 * (index == 20)))
    vocals = present(state, "vocals")
    assert vocals.level == pytest.approx(0.6) and len(vocals.hits) == 1 and vocals.segment is None
    # A logged row without the field is not taken for an arbitrary new segment.
    for _ in range(10):  # a little later, so the change is not only a step ahead
        stream.frame(4)
    row = activity(stream.time, "vocals", level=0.3).model_dump(mode="json")
    del row["segment"]
    legacy = parse_message(row)
    assert "segment" not in legacy.model_fields_set
    state.consume(legacy)
    vocals = present(state, "vocals")
    assert vocals.time == pytest.approx(stream.time) and vocals.level < 0.4  # fed, easing down
    # Messages that crossed from the analysis process keep knowing they had a segment.
    assert "segment" in pickle.loads(pickle.dumps(activity(1.0, "bass", segment=3))).model_fields_set
    # And a session where everything is the default segment 0 never starts afresh.
    old = Stream()
    old.time = 0.0
    epochs = set()
    for _ in range(50):
        old.state.consume(frame(old.time), arrival=old.clock + old.time)
        epochs.add(old.state.live_epoch)
        old.state.consume(activity(old.time, "drums", attack=0.4))
        old.time += HOP
    assert len(epochs) == 1 and len(present(old.state, "drums").hits) == 50


# -- the bass ring's phase -----------------------------------------------------------------


def energy_at(time):
    return 0.5 + 0.45 * math.sin(2 * math.pi * 0.7 * time) * (time < 1.6)


def rows_until(end, start=0.0):
    count = round((end - start) / HOP)
    return [round(start + index * HOP, 6) for index in range(count + 1)]


def run(fps, target=2.0, hidden=(), segment=None, rows=None):
    """A bass live state fed row by row as each row's frame (80 ms later) is reached, and
    advanced once per display frame at ``fps`` except while ``hidden``."""
    live = Live(26.0, 56.0)
    rows = rows_until(target) if rows is None else rows
    fed = 0
    for step in range(round(target * fps) + 1):
        now = step / fps
        while fed < len(rows) and rows[fed] + 0.08 <= now + 1e-9:
            time = rows[fed]
            live.feed(time, time + 0.08, level=0.6, ring=0.0, energy=energy_at(time), lead=0.08, segment=segment)
            fed += 1
        if not any(a <= now < b for a, b in hidden):
            live.advance(now)
    return live


def expected_phase(t, rows):
    """The integral the phase stands for, worked out separately: base rate times time since
    the first row, plus the per-energy rate times each row's energy held until the next."""
    travel = 0.0
    for a, b in zip(rows[:-1], rows[1:], strict=True):
        if b <= t:
            travel += energy_at(a) * min(b - a, STALE_S)
    newest = max(r for r in rows if r <= t + 1e-12)
    travel += energy_at(newest) * min(t - newest, STALE_S)
    return FLOW_BASE * (t - rows[0]) + FLOW_PER_ENERGY * travel


def bass_layer():
    return layer_registry.build()["bass"]


def stage(now):
    return layer_registry.Stage(now, 6.0, 1.0, 0, None, ring=(400.0, 400.0, 250.0, 330.0), live=True)


def test_the_phase_is_the_same_at_60_72_120_and_144_frames_a_second():
    phases = {fps: run(fps).phase(FLOW_BASE, FLOW_PER_ENERGY) for fps in (60, 72, 120, 144)}
    first = phases[60]
    assert all(value == pytest.approx(first, abs=1e-9) for value in phases.values())
    live = run(72)
    assert first == pytest.approx(expected_phase(live.t, rows_until(2.0)), abs=1e-9)


def test_not_drawing_the_ring_for_a_while_does_not_change_its_phase():
    seen = run(72)
    hidden = run(72, hidden=[(0.4, 1.7)])  # hidden while 1.3 s of music played
    assert hidden.phase(FLOW_BASE, FLOW_PER_ENERGY) == pytest.approx(seen.phase(FLOW_BASE, FLOW_PER_ENERGY), abs=1e-9)
    # The renderer itself keeps no clock: its shape comes from the state alone.
    layer = bass_layer()
    for live in (seen, hidden):
        live.energy, live.level, live.weight = 0.6, 0.6, 1.0  # the eased parts, made equal
    a, b = layer.live_shape(stage(2.0), seen), layer.live_shape(stage(2.0), hidden)
    assert np.allclose(a[0], b[0], atol=1e-9) and np.allclose(a[1], b[1], atol=1e-9)
    assert not hasattr(layer, "flow")
    again = layer.live_shape(stage(2.0), seen)
    assert np.array_equal(again[0], a[0])  # drawing twice changes nothing


def test_replaying_the_same_rows_gives_the_same_phase_and_shape():
    one, two = run(72), run(144)
    for live in (one, two):
        live.advance(2.0)  # the same final moment for both
    assert one.phase(FLOW_BASE, FLOW_PER_ENERGY) == two.phase(FLOW_BASE, FLOW_PER_ENERGY)
    layer = bass_layer()
    for live in (one, two):
        live.energy, live.level, live.weight = 0.6, 0.6, 1.0
    shape_one, shape_two = layer.live_shape(stage(2.0), one), layer.live_shape(stage(2.0), two)
    assert np.allclose(shape_one[0], shape_two[0], atol=1e-9)


def test_a_new_segment_counts_its_phase_from_its_own_first_row():
    rows = rows_until(5.0)
    old = run(72, target=5.0, segment=1, rows=rows)
    assert old.phase(FLOW_BASE, FLOW_PER_ENERGY) > 5.0
    # Segment 2 continues at 5.02 s.
    later = rows_until(6.0, start=5.02)
    for time in later:
        old.feed(time, time + 0.08, level=0.6, ring=0.0, energy=energy_at(time), lead=0.08, segment=2)
    old.advance(6.08)
    fresh = Live(26.0, 56.0)
    for time in later:
        fresh.feed(time, time + 0.08, level=0.6, ring=0.0, energy=energy_at(time), lead=0.08, segment=2)
    fresh.advance(6.08)
    assert old.origin == pytest.approx(5.02) and old.elapsed == pytest.approx(0.98)
    assert old.phase(FLOW_BASE, FLOW_PER_ENERGY) == pytest.approx(fresh.phase(FLOW_BASE, FLOW_PER_ENERGY), abs=1e-9)


def test_more_energy_still_means_faster_flow():
    def flowed(energy):
        live = Live(26.0, 56.0)
        for time in rows_until(2.0):
            live.feed(time, time, level=energy, ring=0.0, energy=energy)
        live.advance(2.0)
        return live.phase(FLOW_BASE, FLOW_PER_ENERGY)

    still, half, full = flowed(0.0), flowed(0.5), flowed(1.0)
    assert still == pytest.approx(FLOW_BASE * 2.0)  # 0.5 rad a second at rest
    assert half == pytest.approx((FLOW_BASE + 0.5 * FLOW_PER_ENERGY) * 2.0)
    assert full == pytest.approx((FLOW_BASE + FLOW_PER_ENERGY) * 2.0)  # 3.5 at full energy
    # When the rows stop, the energy part stops after STALE_S; the resting part goes on.
    live = Live(26.0, 56.0)
    for time in rows_until(1.0):
        live.feed(time, time, level=1.0, ring=0.0, energy=1.0)
    live.advance(1.0)
    at_end = live.phase(FLOW_BASE, FLOW_PER_ENERGY)
    live.advance(3.0)
    assert live.phase(FLOW_BASE, FLOW_PER_ENERGY) - at_end == pytest.approx(
        FLOW_BASE * 2.0 + FLOW_PER_ENERGY * STALE_S
    )


def test_drum_sparks_are_the_same_pattern_worked_out_once():
    for seed in (0, 1234, 65535):
        assert np.array_equal(sparks(seed), np.random.default_rng(seed).random((18, 2)))
    assert sparks(77) is sparks(77) and not sparks(77).flags.writeable
    assert sparks.cache_info().maxsize == 128
