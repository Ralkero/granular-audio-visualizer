"""What the visualizer knows about the analysis stream. No Qt here, so it runs headless.

Everything is placed on the audio timeline using the timestamps the engine publishes. The
only use of arrival time is to learn how the audio timeline maps onto the local clock, so
that the display can move smoothly between analysis frames.
"""

from collections import deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..models import FeatureFrame, MusicalEvent, Status
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
