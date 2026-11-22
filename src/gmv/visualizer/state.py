"""What the visualizer knows about the analysis stream. No Qt here, so it runs headless.

Everything is placed on the audio timeline using the timestamps the engine publishes. The
only use of arrival time is to learn how the audio timeline maps onto the local clock, so
that the display can move smoothly between analysis frames.
"""

from collections import deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..instruments import combine
from ..models import (
    FeatureFrame,
    InstrumentActivity,
    InstrumentEstimate,
    MusicalEvent,
    Status,
)
from .lanes import LaneBoard, Track

# event_type -> (timeline row, plain description). Anything else lands in "changes" under
# its own name, drawn by its evidence kind, so new event types show up without code here.
EVENTS = {
    "onset": ("onsets", "Onset"),
    "beat_prediction": ("beats", "Predicted beat"),
    "beat_observed": ("beats", "Onset on a predicted beat"),
    "beat_candidate": ("beats", "Beat candidate"),
    "bass_transient_candidate": ("regions", "Low-frequency transient"),
    "snare_like_candidate": ("regions", "Mid-frequency transient"),
    "high_frequency_percussion_candidate": ("regions", "High-frequency transient"),
    "energy_rise": ("changes", "Energy rise"),
    "intensity_change": ("changes", "Intensity change"),
    "spectral_change": ("changes", "Spectral change"),
    "silence_started": ("state", "Silence"),
    "note_attack_candidate": ("notes", "Note attack"),
    "audio_resumed": ("state", "Audio resumed"),
}
REGIONS = {
    "bass_transient_candidate": 0,
    "snare_like_candidate": 1,
    "high_frequency_percussion_candidate": 2,
}
ROWS = (
    ("onsets", "Onsets"),
    ("beats", "Beats"),
    ("regions", "Transients by region"),
    ("changes", "Changes"),
    ("context", "Delayed context"),
)
STOPPED = {"stopped", "finished", "error"}
# Columns of an instrument's activity history. The first eight are what the analyzer
# published; the rest are derived once, when a row arrives, so that every renderer and the
# lane read the same thing:
#   ring         envelope that jumps to the articulation and dies away, slower while more is
#                ringing
#   struck_time  when the most recent attack was
#   struck_centre, struck_spread  where and how wide that attack was
#   gap          at an attack: seconds since the attack before it (0 otherwise)
ACTIVITY = (
    "time",
    "level",
    "flux",
    "attack",
    "rate_hz",
    "regularity",
    "centre",
    "spread",
    "ring",
    "struck_time",
    "struck_centre",
    "struck_spread",
    "gap",
    # Instrument-specific, NaN or 0 where an instrument has none:
    "pitch",  # MIDI note, NaN when unpitched
    "pitch_confidence",
    "low",  # drums: hit strength per region at this frame
    "mid",
    "high",
    "boundary",  # vocals: 1 phrase start, 2 phrase end
)
BOUNDARY = {"": 0.0, "start": 1.0, "end": 2.0}
COLUMN = {name: index for index, name in enumerate(ACTIVITY)}
LEVELS = {1.0: "detected", 0.5: "candidate", 0.0: "absent"}


@dataclass(frozen=True)
class Marker:
    time: float
    end: float | None
    row: str
    # observed | candidate | predicted | context: how the mark may be drawn.
    style: str
    kind: str
    label: str
    intensity: float
    score: float
    region: int | None = None


def marker(event):
    row, label = EVENTS.get(
        event.event_type, ("changes", event.event_type.replace("_", " ").capitalize())
    )
    delayed = event.analysis_tier == "advanced" or event.evidence_kind == "context"
    return Marker(
        time=event.timestamp,
        end=event.interval_end,
        row="context" if delayed else row,
        style="context" if delayed else event.evidence_kind,
        kind=event.event_type,
        label=label,
        intensity=event.intensity,
        score=event.confidence,
        region=REGIONS.get(event.event_type),
    )


class VisualState:
    def __init__(self, window_s=15.0, capacity=4096):
        self.window_s = window_s
        self.capacity = capacity
        self.board = LaneBoard(capacity)
        self.status, self.detail = "idle", ""
        self.session_id = None
        self.generation = 0
        self.marked = 0  # markers ever received; lets a drawing find the new ones
        self.reset()

    def reset(self):
        """Forget the history. The stream itself is untouched."""
        self.generation += 1  # lets cached drawings know the history was replaced
        self.frames = Track(self.capacity, 4)  # time, level 0-1, intensity, onset strength
        self.markers = deque(maxlen=2048)
        self.gaps = deque(maxlen=64)  # (start, end) on the audio timeline
        self.board.clear()
        self.activity = {}  # instrument -> Track of ACTIVITY rows
        self.activity_info = {}  # instrument -> {"method": ..., "basis": ...}
        self.music_start = None  # when sound last began after silence (or at all)
        self._silent_from = None
        self.latest = None
        self.spectrum = None
        self.spectrum_range = None
        self.beats = deque(maxlen=64)  # predicted beat times not yet reached
        # Note candidates: (time, MIDI note, score, guitar judged active at that time).
        self.notes = Track(4 * self.capacity, 4)
        self.note_attacks = deque(maxlen=512)  # (time, MIDI note, score)
        self.note_serial = 0  # grows with every note sample or attack received
        self.pitch_method = None
        self.beat_tempo, self.beat_score = None, 0.0
        self.replayed = False
        self.segment = None
        self.last_arrival = None
        self._offsets = deque(maxlen=150)
        self._pending = deque(maxlen=256)  # markers the renderer has not reacted to yet

    # -- clock ---------------------------------------------------------------------------

    def now(self, clock=None):
        """The audio-timeline position that corresponds to this moment, or None."""
        if not self._offsets:
            return None
        return (perf_counter() if clock is None else clock) - min(self._offsets)

    def condition(self, clock=None):
        """idle | live | replay | waiting | stopped."""
        if self.status in STOPPED:
            return "stopped"
        if self.last_arrival is None:
            return "idle" if self.status == "idle" else "waiting"
        clock = perf_counter() if clock is None else clock
        if self.status == "waiting" or clock - self.last_arrival > 1.0:
            return "waiting"
        return "replay" if self.replayed else "live"

    # -- input ---------------------------------------------------------------------------

    def consume(self, message, arrival=None):
        if message.session_id != self.session_id:
            self.session_id = message.session_id
            self.reset()
        if isinstance(message, FeatureFrame):
            self._frame(message, arrival)
        elif isinstance(message, MusicalEvent):
            self.replayed = message.replayed
            mark = marker(message)
            if mark.row == "notes":  # drawn in the note lane, not on the timeline
                midi = dict(message.measurements).get("midi")
                if midi is not None:
                    self.note_attacks.append((mark.time, midi, message.intensity))
                    self.note_serial += 1
                return
            self.markers.append(mark)
            self.marked += 1
            self._pending.append(mark)
            if message.event_type == "beat_prediction":
                self.beats.append(mark.time)
                self.beat_score = message.confidence
                self.beat_tempo = dict(message.measurements).get("tempo_bpm", self.beat_tempo)
        elif isinstance(message, InstrumentEstimate):
            self.board.estimate(message)
        elif isinstance(message, InstrumentActivity):
            self._activity(message)
        elif isinstance(message, Status):
            self.status, self.detail = message.state, message.detail

    def _frame(self, frame, arrival):
        arrival = arrival or frame.published_monotonic or perf_counter()
        previous = self.latest
        if previous is not None and frame.timestamp < previous.timestamp:
            self.reset()  # the timeline restarted
            previous = None
        offset = arrival - frame.timestamp
        broken = previous is not None and (
            frame.segment != previous.segment or frame.timestamp - previous.timestamp > 0.5
        )
        if broken or (self._offsets and offset - min(self._offsets) > 0.5):
            # Capture gap or a stalled stream: learn the clock mapping again, show the hole.
            self._offsets.clear()
            if previous is not None and frame.timestamp - previous.window_end > 0.1:
                self.gaps.append((previous.window_end, frame.timestamp))
        self._offsets.append(offset)
        self.last_arrival = arrival
        self.latest = frame
        self.replayed = frame.replayed
        if self.status in ("idle", "waiting", *STOPPED):
            self.status = "capturing"
        # Where the current music began: a recognition model that has heard only a few
        # seconds of it is still judging mostly silence.
        if frame.silence:
            if self._silent_from is None:
                self._silent_from = frame.timestamp
        else:
            if self.music_start is None or (
                self._silent_from is not None and frame.timestamp - self._silent_from >= 3.0
            ):
                self.music_start = frame.timestamp
            self._silent_from = None
        level = min(1.0, max(0.0, (frame.loudness_dbfs + 60) / 60))
        self.frames.append(frame.timestamp, level, frame.dynamic_intensity, frame.onset_strength)
        if frame.spectrum_db:
            self.spectrum = np.asarray(frame.spectrum_db)
            self.spectrum_range = (frame.spectrum_low_hz, frame.spectrum_high_hz)
        else:  # sessions recorded before the display spectrum existed: seven broad bands
            self.spectrum = 10 * np.log10(np.maximum(frame.band_energy, 1e-12))
            self.spectrum_range = None
        if frame.pitch_method:
            self.pitch_method = frame.pitch_method
            guitar = float("guitar" in frame.instruments_active)
            when = frame.pitch_time if frame.pitch_time is not None else frame.timestamp
            for midi, score in frame.pitch_candidates:
                self.notes.append(when, midi, score, guitar)
            self.note_serial += 1
        if frame.activity_method:
            self.board.update(
                frame.timestamp,
                frame.instrument_scores,
                frame.instruments_active,
                frame.activity_method,
            )

    def _activity(self, message):
        track = self.activity.get(message.instrument)
        if track is None:
            track = self.activity[message.instrument] = Track(self.capacity, len(ACTIVITY))
        last = track.last
        if last is not None and message.timestamp <= last[0]:
            return  # rows stay in time order
        self.activity_info[message.instrument] = {
            "method": message.method,
            "basis": message.basis,
        }
        fresh = last is None or message.timestamp - last[0] > 0.5  # start, or a hole
        ring, struck, centre, spread = (0.0, -1e6, 0.5, 0.0) if fresh else last[8:12]
        # The envelope dies away over 60 ms when little is ringing, half a second when a
        # chord is held: a held chord stays on screen, a muted stroke does not.
        elapsed = 0.02 if fresh else message.timestamp - last[0]
        ring = max(message.flux, ring * float(np.exp(-elapsed / (0.06 + 0.45 * message.level))))
        gap = 0.0
        if message.attack > 0:
            gap = 0.0 if struck < 0 else message.timestamp - struck
            struck, centre, spread = message.timestamp, message.centre, message.spread
        track.append(
            message.timestamp,
            message.level,
            message.flux,
            message.attack,
            np.nan if message.rate_hz is None else message.rate_hz,
            message.regularity,
            message.centre,
            message.spread,
            ring,
            struck,
            centre,
            spread,
            gap,
            np.nan if message.pitch_midi is None else message.pitch_midi,
            message.pitch_confidence,
            *(dict(message.components).get(name, 0.0) for name in ("low", "mid", "high")),
            BOUNDARY[message.boundary],
        )

    def attribution(self, key, times):
        """How much of an instrument's fast activity to show at each time, and where that is
        only a tentative reading: (weights 0..1, tentative flags). See
        ``gmv.instruments.combine``; this applies it to the lane the recognition model fills.
        """
        times = np.asarray(times, dtype=float)
        judged = self.board.judged_at(key, times)
        start = self.music_start if self.music_start is not None else 0.0
        # Only a context model warms up; a frame detector judges each frame as it comes.
        warming = (times < start + (self.board.context_s or 10.0)) & (
            self.board.owner.get(key) == "recognition"
        )
        weights = np.empty(len(times))
        tentative = np.zeros(len(times), dtype=bool)
        cases = [(np.isnan(judged), None, False)]
        for level, name in LEVELS.items():
            for warm in (False, True):
                cases.append(((judged == level) & (warming == warm), name, warm))
        for mask, name, warm in cases:
            if mask.any():
                weights[mask], tentative[mask] = combine(name, warm)
        return weights, tentative

    # -- output --------------------------------------------------------------------------

    def due(self, now):
        """Markers whose audio time has just been reached; each is returned once.

        Marks that are already old when they arrive (delayed analysis, a burst after a stall)
        are dropped here: they still appear on the timeline but must not flash as if fresh.
        """
        ready, waiting = [], deque(maxlen=self._pending.maxlen)
        for mark in self._pending:
            if mark.time > now:
                waiting.append(mark)
            elif now - mark.time <= 0.25 and mark.style != "context":
                ready.append(mark)
        self._pending = waiting
        while self.beats and self.beats[0] <= now:
            self.beats.popleft()
        return ready


class SpectrumDisplay:
    """Turns band levels in dB into smooth 0-1 bar heights.

    Rises follow the analysis at once; falls decay over ``release_s``, so attacks keep their
    timing and only the release is softened. The top of the scale follows the loudest recent
    band slowly, which keeps quiet passages low instead of stretching everything to full.
    """

    RANGE_DB, HEADROOM_DB, FLOOR_DB, TILT_DB_PER_OCTAVE = 46.0, 4.0, -55.0, 2.5

    def __init__(self, release_s=0.22, gain_db=0.0):
        self.release_s, self.gain_db = release_s, gain_db
        self.values = self.peaks = None
        self.ceiling = self.FLOOR_DB

    def step(self, spectrum_db, dt, frequency_range=None):
        count = len(spectrum_db)
        if self.values is None or len(self.values) != count:
            self.values, self.peaks = np.zeros(count), np.zeros(count)
            self.ceiling = self.FLOOR_DB
        octaves = np.log2(frequency_range[1] / frequency_range[0]) if frequency_range else 9.0
        tilted = spectrum_db + np.linspace(0, octaves * self.TILT_DB_PER_OCTAVE, count)
        self.ceiling = max(self.FLOOR_DB, self.ceiling - 0.75 * dt, float(tilted.max()))
        level = tilted - self.ceiling - self.HEADROOM_DB + self.gain_db
        # The curve keeps mid-level bands from all crowding the top in dense passages.
        target = np.clip((level + self.RANGE_DB) / self.RANGE_DB, 0, 1) ** 1.5
        self.values = np.maximum(target, self.values * np.exp(-dt / max(self.release_s, 1e-3)))
        self.peaks = np.maximum(self.values, self.peaks - 0.6 * dt)
        return self.values

    def rest(self, dt):
        """Let the bars fall when no audio is arriving."""
        if self.values is not None:
            self.values = self.values * np.exp(-dt / max(self.release_s, 1e-3))
            self.peaks = np.maximum(self.values, self.peaks - 0.6 * dt)
        return self.values
