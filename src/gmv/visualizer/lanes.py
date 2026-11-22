"""Instrument-activity lanes.

A lane shows the score a detector publishes under one label, either with every analysis
frame (``FeatureFrame.instrument_scores``) or, for a context recognition model, a little
later as an ``InstrumentEstimate``. Nothing here decides what is playing: a lane with no
published score is shown as having no detector, never as silent or active.

The third value of a lane row is how far the detector commits: 1 detected, 0.5 candidate
(only a context model reports that), 0 not reported.

Adding a detector needs no renderer change. Publishing a new label creates its lane; the
labels listed in ``KNOWN`` only add a readable title, a colour and a place in the order.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Lane:
    key: str
    title: str
    color: str
    parent: str | None = None


# Order is display order. Sub-lanes (with a parent) appear only once a detector publishes them.
KNOWN = (
    Lane("drums", "Drums", "#ff8a5b"),
    Lane("kick", "Kick", "#ff8a5b", "drums"),
    Lane("snare", "Snare", "#ffa97f", "drums"),
    Lane("cymbals", "Cymbals & hi-hat", "#ffc9a8", "drums"),
    Lane("bass", "Bass", "#4cc9f0"),
    Lane("bass_guitar", "Bass guitar", "#7ad7f5", "bass"),
    Lane("vocals", "Vocals", "#f472b6"),
    Lane("vocals_clean", "Clean vocals", "#f79ac9", "vocals"),
    Lane("vocals_harsh", "Harsh vocals", "#e04f9c", "vocals"),
    Lane("guitar", "Guitar", "#f5c451"),
    Lane("guitar_distorted", "Distorted guitar", "#f59e42", "guitar"),
    Lane("guitar_clean", "Clean guitar", "#f8d98a", "guitar"),
    Lane("guitar_acoustic", "Acoustic guitar", "#e9d8a6", "guitar"),
    Lane("rhythm_guitar", "Rhythm guitar", "#f5c451", "guitar"),
    Lane("lead_guitar", "Lead guitar", "#f8d98a", "guitar"),
    Lane("other_harmonic", "Other harmonic", "#a78bfa"),
    Lane("ambient", "Atmosphere", "#c4b5fd", "other_harmonic"),
)
ALIASES = {"other": "other_harmonic"}
# Shown even without a detector, so that missing capabilities are visible rather than absent.
ALWAYS_SHOWN = ("drums", "bass", "vocals", "guitar", "other_harmonic")
FALLBACK_COLORS = ("#94a3b8", "#5eead4", "#fca5a5", "#bef264")


class Track:
    """Bounded, time-ordered rows (time first) that can be read back as one array."""

    def __init__(self, capacity, columns):
        self.capacity = capacity
        self.rows = np.zeros((2 * capacity, columns))
        self.count = 0

    def append(self, *row):
        if self.count == len(self.rows):
            self.rows[: self.capacity] = self.rows[self.capacity :]
            self.count = self.capacity
        self.rows[self.count] = row
        self.count += 1

    def __len__(self):
        return min(self.count, self.capacity)

    @property
    def last(self):
        return self.rows[self.count - 1] if self.count else None

    def since(self, time=None):
        """Rows at or after ``time`` (all rows when None), oldest first."""
        rows = self.rows[max(0, self.count - self.capacity) : self.count]
        return rows if time is None else rows[np.searchsorted(rows[:, 0], time) :]


class LaneBoard:
    """Bounded history per published label: rows of (time, score, active)."""

    def __init__(self, capacity=4096):
        self.capacity = capacity
        self.known = {lane.key: lane for lane in KNOWN}
        self.order = [lane.key for lane in KNOWN]
        self.clear()

    def clear(self):
        self.history = {}
        self.method = None
        self.owner = {}  # label -> "frames" or "recognition": one source per lane
        # What the context recognition model last said about itself, or None when not in use.
        self.recognition = None
        self.revision = 0  # grows whenever recognition adds to the guitar lane
        # Grows when a lane's committed level changes (or first appears): what was drawn
        # with the level held from the estimate before is then no longer right.
        self.judgement = 0
        self.context_s = 0.0  # how much audio the recognition model listens to

    def register(self, lane):
        """Give a label a title, colour and parent before (or after) it is published."""
        if lane.key not in self.known:
            self.order.append(lane.key)
        self.known[lane.key] = lane

    def _append(self, source, name, time, score, level, gap):
        key = ALIASES.get(name, name)
        if self.owner.setdefault(key, source) != source:
            return None  # another detector already draws this lane
        if key not in self.known:
            color = FALLBACK_COLORS[len(self.known) % len(FALLBACK_COLORS)]
            self.register(Lane(key, key.replace("_", " ").capitalize(), color))
        track = self.history.get(key)
        if track is None:
            track = self.history[key] = Track(self.capacity, 3)
        if len(track) and time <= track.last[0]:
            return None  # rows stay in time order
        if len(track) and time - track.last[0] > gap:
            # A hole in the stream is a hole in the lane, not a held score.
            track.append(track.last[0] + 0.02, 0.0, 0.0)
            track.append(time - 0.02, 0.0, 0.0)
        if not len(track) or track.last[2] != level:
            self.judgement += 1
        track.append(time, min(1.0, max(0.0, float(score))), level)
        return key

    def update(self, time, scores, active, method):
        self.method = method
        active = {ALIASES.get(name, name) for name in active}
        for name, score in scores:
            level = float(ALIASES.get(name, name) in active)
            self._append("frames", name, time, score, level, 0.2)

    def estimate(self, message):
        """Take in an ``InstrumentEstimate``: lane rows for its interval, or its failure."""
        self.recognition = {
            "backend": message.backend,
            "available": message.available,
            "detail": message.detail,
        }
        if not message.available:
            return
        self.context_s = message.context_s
        levels = {"detected": 1.0, "candidate": 0.5}
        states = dict(message.states)
        hop = max(0.05, message.interval_end - message.timestamp)
        for name, score in message.scores:
            level = levels.get(states.get(name), 0.0)
            key = self._append("recognition", name, message.interval_end, score, level, 3 * hop)
            if key == "guitar":
                self.revision += 1

    def level_at(self, key, times):
        """How far the detector of a lane committed at each time: 1, 0.5 or 0. Times the lane
        has nothing about (no detector, before its first or after its newest row) give 0."""
        times = np.asarray(times, dtype=float)
        track = self.history.get(key)
        if track is None or not len(track):
            return np.zeros(len(times))
        rows = track.since()
        index = np.searchsorted(rows[:, 0], times, side="left")
        inside = index < len(rows)
        return np.where(inside, rows[np.minimum(index, len(rows) - 1), 2], 0.0)

    def judged_at(self, key, times, hold=1.5):
        """The detector's level (1, 0.5 or 0) at each time, or NaN where it has said nothing:
        no detector, before its first row, or more than ``hold`` seconds after its newest.
        Up to ``hold`` the newest level is assumed to continue, because a recognition model
        always reports late and the display should not blank while it waits."""
        times = np.asarray(times, dtype=float)
        track = self.history.get(key)
        if track is None or not len(track):
            return np.full(len(times), np.nan)
        rows = track.since()
        index = np.searchsorted(rows[:, 0], times, side="left")
        known = np.where(index < len(rows), rows[np.minimum(index, len(rows) - 1), 2], np.nan)
        held = (index >= len(rows)) & (times - rows[-1, 0] <= hold)
        known = np.where(held, rows[-1, 2], known)
        # Before the first row nothing is known; a hole in the stream is stored as zeros.
        return np.where(times < rows[0, 0] - 0.75, np.nan, known)

    def lanes(self):
        """(lane, has detector output) in display order."""
        return [
            (self.known[key], key in self.history)
            for key in self.order
            if key in self.history or key in ALWAYS_SHOWN
        ]
