"""Instrument-activity lanes.

A lane shows the score a detector publishes under one label in
``FeatureFrame.instrument_scores``. Nothing here decides what is playing: a lane with no
published score is shown as having no detector, never as silent or active.

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

    def register(self, lane):
        """Give a label a title, colour and parent before (or after) it is published."""
        if lane.key not in self.known:
            self.order.append(lane.key)
        self.known[lane.key] = lane

    def update(self, time, scores, active, method):
        self.method = method
        active = {ALIASES.get(name, name) for name in active}
        for name, score in scores:
            key = ALIASES.get(name, name)
            if key not in self.known:
                color = FALLBACK_COLORS[len(self.known) % len(FALLBACK_COLORS)]
                self.register(Lane(key, key.replace("_", " ").capitalize(), color))
            track = self.history.get(key)
            if track is None:
                track = self.history[key] = Track(self.capacity, 3)
            if len(track) and time - track.last[0] > 0.2:
                # A hole in the stream is a hole in the lane, not a held score.
                track.append(track.last[0] + 0.02, 0.0, 0.0)
                track.append(time - 0.02, 0.0, 0.0)
            track.append(time, min(1.0, max(0.0, float(score))), float(key in active))

    def lanes(self):
        """(lane, has detector output) in display order."""
        return [
            (self.known[key], key in self.history)
            for key in self.order
            if key in self.history or key in ALWAYS_SHOWN
        ]
