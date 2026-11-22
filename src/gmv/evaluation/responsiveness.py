"""How soon, in audio time, the instrument analyzers know about what was played.

Diagnostic only: nothing in the application imports this. ``trace`` runs decoded audio
through the same ``Analyzer`` pass the engine runs and keeps, for every
``InstrumentActivity``, the moment in the audio it describes (``time``) and the end of the
newest audio that had to exist for it to be produced (``known``). ``known - time`` is the
analyzer's own latency: window, confirmation and back-dating, with no computer in it.
The other functions reduce a trace against the true onset times of a synthetic signal.

Clock domains: everything here is audio time (seconds from the start of the audio).
Wall-clock delays (capture, queueing, processing, drawing) are measured elsewhere, by
``scripts/trace_responsiveness.py``, and must not be added to these by hand without saying so.
"""

import numpy as np

from ..dsp import Analyzer, WindowAssembler
from ..sources.base import AudioBlock

FIELDS = (
    "time",  # the moment the message describes
    "known",  # end of the audio that had been analysed when it was produced
    "level",
    "flux",
    "attack",
    "pitch",
    "pitch_confidence",
    "low",
    "mid",
    "high",
    "boundary",  # 1 phrase start, 2 phrase end
    "present",  # the activity model's judgement in that frame: 1, 0, or NaN without a model
    "energy",  # the analyzer's faster reading, NaN when it has none (see ``fast``)
    "pulse",
    "lead",  # seconds by which energy and pulse are newer than ``time``
)
F = {name: index for index, name in enumerate(FIELDS)}
BOUNDARY = {"": 0.0, "start": 1.0, "end": 2.0}


def trace(audio, rate, config, sensitivity=None, frames=None):
    """instrument -> rows of ``FIELDS``, and instrument-analysis milliseconds per frame.
    The feature frames themselves are appended to ``frames`` when a list is given."""
    analyzer = Analyzer(rate, audio.shape[1], config, "responsiveness", "responsiveness", sensitivity)
    assembler = WindowAssembler(config)
    rows, times = {name: [] for name in config.instrument_analyzers}, []
    step = round(rate * config.capture_ms / 1000)
    for pos in range(0, len(audio), step):
        chunk = audio[pos : pos + step]
        block = AudioBlock(
            chunk.astype("<f4").tobytes(), len(chunk), rate, chunk.shape[1], pos / rate, pos / rate, 0
        )
        for window, start, mono, _reset in assembler.push(block):
            frame, _ = analyzer.process(window, start, mono, block, live=False)
            times.append(analyzer.instrument_ms)
            if frames is not None:
                frames.append(frame)
            for message in analyzer.instrument_activity:
                parts = dict(message.components)
                judged = (
                    float(message.instrument in frame.instruments_active)
                    if frame.activity_method
                    and message.instrument in dict(frame.instrument_scores)
                    else np.nan
                )
                rows[message.instrument].append(
                    (
                        message.timestamp,
                        frame.window_end,
                        message.level,
                        message.flux,
                        message.attack,
                        np.nan if message.pitch_midi is None else message.pitch_midi,
                        message.pitch_confidence,
                        parts.get("low", 0.0),
                        parts.get("mid", 0.0),
                        parts.get("high", 0.0),
                        BOUNDARY[message.boundary],
                        judged,
                        np.nan if message.energy is None else message.energy,
                        message.pulse,
                        message.lead_s,
                    )
                )
    return (
        {name: np.asarray(found, dtype=float).reshape(-1, len(FIELDS)) for name, found in rows.items()},
        np.asarray(times),
    )


def fast(rows):
    """The same rows read through the analyzer's faster path: ``time`` moved to the moment
    energy and pulse describe, ``level`` and ``attack`` replaced by them. Rows without a
    faster reading are left out."""
    kept = rows[np.isfinite(rows[:, F["energy"]])].copy()
    kept[:, F["time"]] += kept[:, F["lead"]]
    kept[:, F["level"]] = kept[:, F["energy"]]
    kept[:, F["attack"]] = kept[:, F["pulse"]]
    return kept


def spread(values):
    """Median, 95th percentile and count of ``values`` (None for each when there are none)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"p50": None, "p95": None, "n": 0}
    return {
        "p50": round(float(np.median(values)), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
        "n": int(len(values)),
    }


def match(rows, onsets, column="attack", tolerance=0.07):
    """Each true onset against the nearest reported event in ``column`` within
    ``tolerance`` seconds; an event answers one onset only.

    * found / missed / extra: counts;
    * placed: reported time less true time (how well the event is placed on the timeline);
    * known: end of the analysed audio when it was reported, less the true time (how long
      after the sound began the analyzer could first have told anybody).
    """
    events = rows[rows[:, F[column]] > 0]
    used = np.zeros(len(events), dtype=bool)
    placed, known, strength = [], [], []
    for onset in onsets:
        distance = np.abs(events[:, F["time"]] - onset)
        distance[used] = np.inf
        if not len(distance) or distance.min() > tolerance:
            continue
        index = int(np.argmin(distance))
        used[index] = True
        placed.append(events[index, F["time"]] - onset)
        known.append(events[index, F["known"]] - onset)
        strength.append(events[index, F[column]])
    return {
        "onsets": len(onsets),
        "found": len(placed),
        "missed": len(onsets) - len(placed),
        "extra": int((~used).sum()),
        "placed_s": spread(placed),
        "known_s": spread(known),
        "strength": spread(strength),
    }


def movement(rows, onsets, column="level", rise=0.1, within=0.4):
    """Does a continuous signal visibly move at each onset, and how soon is that known?

    For every onset: the signal's value just before it, and the first row within ``within``
    seconds whose value is at least ``rise`` above that. Returns the share of onsets that
    moved the signal and, for those, how long after the onset that row was known."""
    moved, known = 0, []
    time, value = rows[:, F["time"]], rows[:, F[column]]
    for onset in onsets:
        before = value[(time >= onset - 0.06) & (time < onset - 0.01)]
        base = float(before.min()) if len(before) else 0.0
        after = np.flatnonzero((time >= onset - 0.01) & (time <= onset + within) & (value >= base + rise))
        if len(after):
            moved += 1
            known.append(rows[after[0], F["known"]] - onset)
    return {"onsets": len(onsets), "moved": moved, "known_s": spread(known)}


def dips(rows, onsets, column="level"):
    """How far a continuous signal falls back between consecutive onsets, as a share of its
    peak after the earlier one: 0 means it stays flat and repeated notes cannot be told
    apart from it, 1 means it returns to nothing every time."""
    time, value = rows[:, F["time"]], rows[:, F[column]]
    found = []
    for first, second in zip(onsets[:-1], onsets[1:], strict=True):
        part = value[(time >= first) & (time < second)]
        if len(part) >= 3 and part.max() > 0.02:
            peak = int(np.argmax(part))
            found.append(1.0 - float(part[peak:].min()) / float(part.max()))
    return spread(found)


def pitch_known(rows, notes, semitones=1.0, within=0.6):
    """For [(onset, MIDI note)]: how long after the onset the right pitch was first known."""
    time, pitch = rows[:, F["time"]], rows[:, F["pitch"]]
    known, right = [], 0
    for onset, midi in notes:
        fits = np.flatnonzero(
            (time >= onset - 0.01) & (time <= onset + within) & (np.abs(pitch - midi) <= semitones)
        )
        if len(fits):
            right += 1
            known.append(rows[fits[0], F["known"]] - onset)
    return {"notes": len(notes), "read": right, "known_s": spread(known)}


def cadence(rows):
    """Messages per second and the spacing of their timestamps."""
    if len(rows) < 3:
        return {"per_second": 0.0, "interval_s": spread([]), "latency_s": spread([])}
    gaps = np.diff(rows[:, F["time"]])
    return {
        "per_second": round(float((len(rows) - 1) / (rows[-1, F["time"]] - rows[0, F["time"]])), 2),
        "interval_s": spread(gaps),
        "latency_s": spread(rows[:, F["known"]] - rows[:, F["time"]]),
    }
