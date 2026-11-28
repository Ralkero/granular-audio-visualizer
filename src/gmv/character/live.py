"""Sonic Character, Phase B: the Phase A tracker run live on the stream the engine publishes.

``LiveCharacter`` is one more subscriber of the ``EventBus``, beside the visualizer and the
session writer. It has no audio path and no analysis of its own: its worker thread takes the
feature frames and onset events from its own bounded queue, in the order they were
published, and hands them to an unchanged ``SonicCharacter``. Given the same messages it
therefore gives exactly the profiles Phase A gives offline, whatever sizes of batches they
arrive in.

* **Nothing runs in the capture or analysis path.** Publishing puts the message on this
  subscriber's queue and returns (``EventBus.publish``); the tracker's work happens on the
  worker thread.
* **Bounded.** The queue holds ``QUEUE`` messages (about a minute of frames). If the worker
  ever fell that far behind, the oldest messages are dropped by the queue, as for every
  subscriber, and the count is reported (``dropped``): the profile would then rest on fewer
  frames than an offline run of the same session.
* **No stale results.** The worker publishes one *latest* snapshot (``snapshot``) and a
  fixed-length rolling history sampled by audio time (``history``). Nothing accumulates: a
  reader that looks rarely, or never, sees the newest state and costs nothing.
* **Sessions.** A message of another session starts everything afresh (the tracker, the
  history), as the visual state does; inside one session the tracker's own rules handle
  segments, gaps, restarts, seeks and silence (``gmv.character.tracker``). A pause is
  silence or no frames at all: the profile is held as it was, never aged.
* **Local only.** Nothing is written or sent. The snapshots hold the seven values, their
  confidences and the profile warnings of three windows, no audio and no measurements per
  frame.

Profiles are built (about 0.2 ms per window) once per ``SAMPLE_S`` of audio time, at the
first frame at or past each sample point, whatever the batches: the snapshots and the history
depend on the stream only. The per-frame cost is the tracker's update.
"""

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter

from ..models import FeatureFrame, MusicalEvent
from .profile import DIMENSION_NAMES
from .tracker import SonicCharacter

QUEUE = 4096  # messages: about a minute of frames at 50 a second, with their events
SAMPLE_S = 0.5  # audio seconds between snapshots and history points
HISTORY_S = 120.0  # audio seconds of rolling history kept
SHOWN = ("short", "medium", "long")  # the windows a snapshot carries
JOIN_S = 2.0  # how long ``detach`` waits for the worker
LOW_CONFIDENCE = 0.3  # under this a value is shown as "not enough signal" (``usable``)


@dataclass(frozen=True)
class Reading:
    """One window's seven values and confidences at one moment."""

    window: str
    values: tuple  # 0..1 or None, in the order of DIMENSION_NAMES
    confidences: tuple  # 0..1: signal support, not a probability of being right
    active: bool  # music at this moment (False: held through silence)
    heard_s: float
    evidence_s: float
    warnings: tuple

    def usable(self, index, threshold=LOW_CONFIDENCE):
        """Whether dimension ``index`` has a value with enough support to be read."""
        return self.values[index] is not None and self.confidences[index] >= threshold


@dataclass(frozen=True)
class Snapshot:
    session_id: str | None
    generation: int  # grows whenever everything started afresh (new session, segment, restart)
    time: float | None  # audio time of the newest frame taken in
    readings: dict  # window -> Reading
    frames: int  # feature frames taken in since the last fresh start
    music_frames: int


@dataclass
class Stats:
    """Counters for diagnostics and the benchmark. Updated by the worker only."""

    messages: int = 0
    frames: int = 0
    onsets: int = 0
    refused: int = 0  # messages the tracker refused (late frames of an old segment, duplicates)
    snapshots: int = 0
    resets: int = 0
    update_s: float = 0.0  # time spent in the tracker's per-message update
    profile_s: float = 0.0  # time spent building snapshots
    backlog_high: int = 0  # most messages waiting in the queue at once
    batches: int = 0
    worst_update_s: float = 0.0
    errors: int = 0
    last_error: str = ""
    extra: dict = field(default_factory=dict)


def reading(profile):
    return Reading(
        window=profile.window,
        values=profile.values(),
        confidences=profile.confidences(),
        active=profile.active,
        heard_s=profile.heard_s,
        evidence_s=profile.evidence_s,
        warnings=profile.warnings,
    )


class LiveCharacter:
    """Attach to a bus with ``attach``; read ``snapshot()`` and ``history()`` from any thread."""

    def __init__(self, capacity=QUEUE, sample_s=SAMPLE_S, history_s=HISTORY_S, windows=SHOWN):
        self.capacity, self.sample_s, self.windows = capacity, sample_s, tuple(windows)
        self._length = max(1, int(math.ceil(history_s / sample_s)))
        self._lock = threading.Lock()
        self._history = deque(maxlen=self._length)  # (time, {window: Reading})
        self._latest = None
        self.tracker = SonicCharacter()
        self.session_id = None
        self.stats = Stats()
        self._sampled_at = None  # audio time of the last snapshot
        self.bus = self.subscription = None
        self._worker = None
        self._dropped_before = 0  # drops of earlier subscriptions

    # -- the stream ------------------------------------------------------------------------

    def attach(self, bus):
        """Follow ``bus`` with a subscription and a worker of this monitor's own; replaces any
        earlier one. The tracker starts afresh."""
        self.detach()
        self._restart(None)
        self.bus, self.subscription = bus, bus.subscribe(self.capacity)
        self._worker = threading.Thread(
            target=self._run, args=(self.subscription,), name="sonic-character", daemon=True
        )
        self._worker.start()

    def detach(self):
        """Stop following the bus. The last snapshot and history stay readable. Returns
        whether the worker stopped in time."""
        stopped = True
        if self.subscription is not None:
            self._dropped_before += self.subscription.dropped
            self.bus.unsubscribe(self.subscription)  # closes the queue: the worker ends
        if self._worker is not None:
            self._worker.join(JOIN_S)
            stopped = not self._worker.is_alive()
        self.bus = self.subscription = self._worker = None
        return stopped

    close = detach

    @property
    def attached(self):
        return self.subscription is not None

    @property
    def dropped(self):
        """Messages the queue dropped because the worker fell behind (all subscriptions)."""
        return self._dropped_before + (self.subscription.dropped if self.subscription else 0)

    @property
    def backlog(self):
        return len(self.subscription) if self.subscription is not None else 0

    def _run(self, queue):
        while True:
            message = queue.get(timeout=0.1)
            if message is None:
                if queue.closed:
                    return
                continue
            batch = [message]
            while len(batch) < 512:  # drain what is there, then build a snapshot at most once
                more = queue.get(timeout=0)
                if more is None:
                    break
                batch.append(more)
            self.stats.backlog_high = max(self.stats.backlog_high, len(batch) + len(queue))
            self.process(batch)

    # -- the work (on the worker thread, or directly in tests) ------------------------------------

    def process(self, messages):
        """Take messages in order (snapshots are made as their audio time comes due)."""
        self.stats.batches += 1
        for message in messages:
            try:
                self._take(message)
            except Exception as exc:  # one bad message never ends the monitor or loses others
                self.stats.errors += 1
                self.stats.last_error = f"{type(exc).__name__}: {exc}"

    def _restart(self, session_id):
        self.session_id = session_id
        self.tracker.reset()
        self.stats.resets += 1
        self._sampled_at = None
        with self._lock:
            self._history.clear()
            self._latest = None

    def _take(self, message):
        self.stats.messages += 1
        is_frame = isinstance(message, FeatureFrame)
        if not is_frame and not (
            isinstance(message, MusicalEvent) and message.event_type == "onset"
        ):
            return  # status messages, other events: nothing the tracker reads
        if message.session_id != self.session_id:
            self._restart(message.session_id)
        generation = self.tracker.generation
        start = perf_counter()
        accepted = self.tracker.consume(message)
        spent = perf_counter() - start
        self.stats.update_s += spent
        self.stats.worst_update_s = max(self.stats.worst_update_s, spent)
        if not accepted:
            self.stats.refused += 1
            return
        if is_frame:
            self.stats.frames += 1
        else:
            self.stats.onsets += 1
        if self.tracker.generation != generation:  # a new segment, restart or seek
            self._sampled_at = None
            with self._lock:
                self._history.clear()
        time = self.tracker.time
        if is_frame and time is not None:
            if self._sampled_at is not None and time < self._sampled_at:
                self._sampled_at = None
            if self._sampled_at is None or time >= self._sampled_at + self.sample_s:
                self._sample()

    def _sample(self):
        self._sampled_at = self.tracker.time
        start = perf_counter()
        tracker = self.tracker
        snap = Snapshot(
            session_id=self.session_id,
            generation=tracker.generation,
            time=tracker.time,
            readings={w: reading(tracker.profile(w)) for w in self.windows},
            frames=tracker.frames,
            music_frames=tracker.music_frames,
        )
        self.stats.profile_s += perf_counter() - start
        self.stats.snapshots += 1
        with self._lock:
            self._latest = snap
            self._history.append(snap)

    # -- reading (any thread) ----------------------------------------------------------------------

    def snapshot(self):
        """The newest snapshot, or None before any frame of this session."""
        with self._lock:
            return self._latest

    def history(self):
        """The rolling history, oldest first: snapshots one ``sample_s`` of audio apart, at most
        ``HISTORY_S`` long. A copy; it never changes after it is returned."""
        with self._lock:
            return tuple(self._history)


def series(history, window, dimension):
    """(times, values, confidences) of one dimension of one window in a history."""
    index = DIMENSION_NAMES.index(dimension)
    times, values, confidences = [], [], []
    for snap in history:
        r = snap.readings[window]
        times.append(snap.time)
        values.append(r.values[index])
        confidences.append(r.confidences[index])
    return times, values, confidences
