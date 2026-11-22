"""What an instrument is doing *now*: the state a live picture is drawn from.

The activity history (``VisualState.activity``) keeps every row an analyzer published, for
the timeline, the lanes and the scrolling strips. A live picture needs something else: not
where activity was, but the present state of the instrument and how long ago things
happened to it. ``Live`` is that, one per instrument. It is fed the same rows as the
history, as they arrive (one owner of the data, no second analysis), and reduces them to

* ``level``, ``energy``: continuous envelopes, eased by elapsed audio time (fast up,
  slower down), so they move smoothly at any frame rate between the 20 ms rows;
* ``ring``: the articulation envelope that jumps at a stroke and dies away;
* ``hits``: the recent attacks, each with its time, strength, place and width. A short
  bounded list, so two attacks between two frames are both there;
* ``pulses``: note onsets of an analyzer's faster path (bass);
* ``pitch`` and ``place``: the note, eased, and where it sits in the instrument's range;
* ``weight``: how much of it the presence judgement lets through, eased so that a
  judgement arriving does not snap the picture.

**Segments.** A live state belongs to one audio segment (the capture stream's
``segment``: it changes at a gap or a format change, even when the timestamps carry on as
if nothing happened). ``begin(segment)`` starts it afresh for a new one: every transient
value goes, the pitch range stays. ``feed`` does the same if a row of a newer segment
reaches it first. Deciding which rows are too old to be fed at all is the visual state's
job, which knows the stream's current segment.

**Phase.** ``phase(base, per_energy)`` is a running angle that advances by
``base + per_energy * energy`` radians per second of audio. It is an integral over the rows
themselves (the energy each row reported, held until the next, for at most ``STALE_S``),
counted from the first row of the segment, and evaluated at the instrument's present. So
it depends only on the rows and the audio time: not on how often, or whether, anything
was drawn in between.

**Clock.** Everything is audio time. An analyzer's rows describe a moment a fixed time
before the newest audio (``lag``: guitar 20 ms, drums 60 ms, bass 80 ms, by how each
works). ``advance(now)`` moves the state to the instrument's own present, ``now - lag``,
so the newest row is "just now" when it arrives and an attack's age starts near zero. The
picture is therefore late by the analyzer's delay and not by more. Ages, decays and
oscillations are functions of that time alone: the same at 60, 72 or 144 frames a second,
and the same again in replay.
"""

import math
from collections import deque
from dataclasses import dataclass

STALE_S = 0.25  # with no row for this long the instrument is taken to have stopped
KEEP_S = 3.0  # how long an attack stays in the list
UP_S, DOWN_S = 0.03, 0.12  # easing of the level
ENERGY_UP_S, ENERGY_DOWN_S = 0.015, 0.10
PITCH_S = 0.04
REST_S = 0.35  # how slowly the place drifts back to the middle without a pitch
WEIGHT_S = 0.15


@dataclass(frozen=True)
class Hit:
    time: float
    strength: float
    centre: float = 0.5
    spread: float = 0.0
    low: float = 0.0
    mid: float = 0.0
    high: float = 0.0
    start: bool = False  # vocals: a phrase began


def eased(value, target, elapsed, seconds):
    return target + (value - target) * math.exp(-elapsed / seconds)


class Live:
    def __init__(self, low=0.0, high=1.0):
        self.range = (low, high)  # pitch range that ``place`` runs over, as MIDI notes
        self.hits = deque(maxlen=64)
        self.pulses = deque(maxlen=64)  # (time, strength)
        self.reset()

    def reset(self):
        self.hits.clear()
        self.pulses.clear()
        self.segment = None  # the audio segment the state belongs to
        self.time = None  # of the newest row
        self.origin = None  # of the first row of the segment
        self._travel, self._drive = 0.0, 0.0  # energy integral up to ``time``; energy there
        self.lag = 0.0
        self.fast = False  # whether the analyzer gives energy and pulses
        self._level = self._energy = self._ring = self._ring_level = 0.0
        self._pitch = None
        self.rate_hz, self.regularity = None, 0.0
        self.sounding = False  # vocals: inside a phrase
        # The present, as of the last ``advance``.
        self.now = self.t = None
        self.dt = 0.0
        self.level = self.energy = self.ring = 0.0
        self.pitch, self.pitched, self.place = None, False, 0.5
        self.weight, self.tentative = 0.0, False
        self.elapsed = self.travel = 0.0

    def begin(self, segment):
        """Belong to ``segment`` from now on; if that is a different one, start afresh."""
        if segment != self.segment:
            self.reset()
            self.segment = segment

    def feed(self, time, frame_time, *, level, ring, attack=0.0, centre=0.5, spread=0.0,
             pitch=None, low=0.0, mid=0.0, high=0.0, boundary="", rate_hz=None,
             regularity=0.0, energy=None, pulse=0.0, lead=0.0, segment=None):  # fmt: skip
        """Take one activity row. ``frame_time`` is the audio time of the newest analysis
        frame when the row arrived: the difference is the analyzer's delay. ``segment``
        is the row's audio segment, None when unknown (taken to be the current one)."""
        if segment is not None:
            if self.segment is None:
                self.segment = segment
            elif segment != self.segment:
                self.begin(segment)  # newer than anything seen: start afresh
        if self.time is not None and (time <= self.time or time - self.time > 0.5):
            if time <= self.time:
                return
            self.hits.clear()  # a hole in the stream: what was before is over
            self.pulses.clear()
        if self.time is None:
            self.origin, self._travel = time, 0.0
        else:
            self._travel += self._drive * min(time - self.time, STALE_S)
        self._drive = energy if energy is not None else level
        self.time = time
        self.lag = min(0.3, max(0.0, frame_time - time))
        self._level, self._ring, self._ring_level = level, ring, level
        self._pitch = pitch
        self.rate_hz, self.regularity = rate_hz, regularity
        if boundary == "start":
            self.sounding = True
        elif boundary == "end":
            self.sounding = False
        if attack > 0:
            self.hits.append(
                Hit(time, attack, centre, spread, low, mid, high, boundary == "start")
            )
        if energy is not None:
            self.fast = True
            self._energy = energy
            if pulse > 0:
                self.pulses.append((time + lead, pulse))

    def advance(self, now, weight=1.0, tentative=False):
        """Move to audio time ``now`` (the display's present)."""
        if self.time is None:
            return
        fresh = self.now is None or now < self.now or now - self.now > 1.0
        self.dt = 0.0 if fresh else now - self.now
        self.now, self.t = now, now - self.lag
        gone = self.t - self.time > STALE_S
        level = 0.0 if gone else self._level
        energy = 0.0 if gone else (self._energy if self.fast else self._level)
        if fresh:
            self.level, self.energy, self.weight = level, energy, weight
        else:
            dt = self.dt
            self.level = eased(self.level, level, dt, UP_S if level > self.level else DOWN_S)
            self.energy = eased(
                self.energy, energy, dt, ENERGY_UP_S if energy > self.energy else ENERGY_DOWN_S
            )
            self.weight = eased(self.weight, weight, dt, WEIGHT_S)
        self.tentative = tentative
        since = max(0.0, self.t - self.time)
        self.ring = self._ring * math.exp(-since / (0.06 + 0.45 * self._ring_level))
        low, high = self.range
        if self._pitch is not None and not gone:
            self.pitch = (
                self._pitch
                if self.pitch is None or fresh or not self.pitched
                else eased(self.pitch, self._pitch, self.dt, PITCH_S)
            )
            self.pitched = True
            place = min(1.0, max(0.0, (self.pitch - low) / (high - low)))
            self.place = place if fresh else eased(self.place, place, self.dt, PITCH_S)
        else:
            self.pitched = False
            if not fresh:
                self.place = eased(self.place, 0.5, self.dt, REST_S)
        ahead = self.t - self.time
        self.elapsed = self.t - self.origin
        self.travel = self._travel + self._drive * (min(ahead, STALE_S) if ahead > 0 else ahead)
        while self.hits and self.t - self.hits[0].time > KEEP_S:
            self.hits.popleft()
        while self.pulses and now - self.pulses[0][0] > KEEP_S:
            self.pulses.popleft()

    # -- reading the present -------------------------------------------------------------

    def recent(self, seconds):
        """(age, hit) of the attacks of the last ``seconds``, newest last."""
        if self.t is None:
            return []
        return [
            (max(0.0, self.t - hit.time), hit)
            for hit in self.hits
            if self.t - hit.time <= seconds
        ]

    def beats(self, seconds):
        """(age, strength) of the recent note onsets: the analyzer's fast pulses when it
        has them (they are not back-dated: their age runs on the display's clock),
        otherwise its attacks."""
        if self.now is None:
            return []
        if self.fast:
            return [
                (max(0.0, self.now - time), strength)
                for time, strength in self.pulses
                if self.now - time <= seconds
            ]
        return [(age, hit.strength) for age, hit in self.recent(seconds)]

    def swell(self, seconds):
        """Sum of the recent onsets, each dying away over ``seconds``: a pulse envelope."""
        return sum(strength * math.exp(-age / seconds) for age, strength in self.beats(6 * seconds))

    def phase(self, base, per_energy):
        """Radians travelled since the segment's first row, at ``base`` per second plus
        ``per_energy`` per second at full energy: a pure function of the rows and the
        audio time of the last ``advance``."""
        return base * self.elapsed + per_energy * self.travel

    @property
    def last(self):
        return self.hits[-1] if self.hits else None
