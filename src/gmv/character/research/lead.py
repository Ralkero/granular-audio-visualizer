"""The lead line in the centre of the stereo image, and what can be said about a voice.

Nothing here is new signal processing. It summarises what two existing parts of GMV
already put out, to find out what they are worth:

* the vocal analyzer (``gmv.instruments.vocals``): per frame, a level for the centre in the
  voice's range, and a pitch when one harmonic series explains the centre's peaks;
* optionally the activity model (``gmv.activity``, a weights file that is not shipped): per
  frame, whether it judges a voice to be sounding.

**None of it proves a voice.** The centre also holds the snare, the kick, the bass, and any
guitar or synthesizer lead mixed in the middle. Without the activity model every value
here describes "the pitched line in the centre", whatever plays it, and the values that
need to know when a voice sounds (``vocal_*``) are unknown, not zero.

The idea behind ``pitched_share`` and ``vocal_pitched_share`` as a reading of delivery:
sung notes are periodic and have a pitch; screamed, shouted and whispered delivery mostly
does not. Whether that works is an empirical question answered in the report.
"""

import numpy as np

from .frontend import result

LOUD = 0.3  # level of the vocal analyzer from which the centre counts as carrying a line
HELD_ST, HELD_S = 0.5, 0.3  # a note is held when it stays this close for this long
NEEDED_S = 4.0

DEFINITIONS = {
    "centre_line_level": ("0..1", "measured", "mean level of the centre in the voice's range (300 Hz to 3.4 kHz) against its own recent peak", "0.1 .. 0.6"),
    "pitched_share": ("0..1", "measured", "share of the music in which the centre has one clear pitch (A2 to A5)", "0 .. 0.6"),
    "pitch_clarity": ("0..1", "measured", "how much of the centre's peaks the best harmonic series explains, while the centre is loud", "0.2 .. 0.7"),
    "line_range_st": ("semitones", "measured", "spread of that pitch, 10th to 90th percentile", "2 .. 20"),
    "line_motion_st_s": ("semitones / s", "measured", "how fast the pitch moves inside unbroken pitched stretches", "1 .. 15"),
    "held_share": ("0..1", "measured", "share of the pitched time spent on notes held within half a semitone for 0.3 s or more", "0 .. 0.8"),
    "vocal_presence": ("0..1", "inferred", "share of the music in which the activity model judges a voice to sound (needs the model)", "0 .. 0.9"),
    "vocal_pitched_share": ("0..1", "inferred", "share of those frames in which the centre has a clear pitch: high for sung, low for unpitched delivery (needs the model)", "0 .. 0.7"),
    "phrase_s": ("s", "inferred", "median length of a phrase: centre sound while the model reports a voice (needs the model)", "0.5 .. 8"),
}


class Rows:
    """The vocal analyzer's rows of one recording as arrays, with the model's judgement."""

    def __init__(self, rows, frames):
        self.time = np.array([row.timestamp for row in rows])
        self.level = np.array([row.level for row in rows])
        self.pitch = np.array([np.nan if row.pitch_midi is None else row.pitch_midi for row in rows])
        self.clarity = np.array([row.pitch_confidence for row in rows])
        self.start = np.array([row.boundary == "start" for row in rows])
        self.end = np.array([row.boundary == "end" for row in rows])
        self.frame_time = np.array([frame.timestamp for frame in frames])
        self.music = np.array([not frame.silence and frame.loudness_dbfs >= -66.0 for frame in frames])
        self.model = bool(frames) and frames[0].activity_method is not None
        self.vocal = np.array(["vocals" in frame.instruments_active for frame in frames])
        self.hop = float(np.median(np.diff(self.frame_time))) if len(frames) > 1 else 0.02


def held(pitch, hop):
    """Per frame, whether it lies in a stretch of pitch that stays within ``HELD_ST`` of
    its own start for at least ``HELD_S``."""
    out = np.zeros(len(pitch), dtype=bool)
    need, index = max(2, round(HELD_S / hop)), 0
    while index < len(pitch):
        if np.isnan(pitch[index]):
            index += 1
            continue
        stop = index + 1
        while stop < len(pitch) and abs(pitch[stop] - pitch[index]) <= HELD_ST:
            stop += 1
        if stop - index >= need:
            out[index:stop] = True
        index = stop
    return out


def measure(rows, start, end):
    """The lead-line measurements of [start, end) seconds."""
    out = {name: (None, 0.0) for name in DEFINITIONS}
    if not len(rows.time) or len(rows.time) != len(rows.frame_time):
        return out
    part = slice(int(np.searchsorted(rows.time, start)), int(np.searchsorted(rows.time, end)))
    music = rows.music[part]
    seconds = float(music.sum()) * rows.hop
    if seconds < 1.0:
        return out
    enough = 1.0 - np.exp(-seconds / NEEDED_S)
    level, pitch, clarity = rows.level[part][music], rows.pitch[part][music], rows.clarity[part][music]
    pitched = ~np.isnan(pitch)
    out["centre_line_level"] = result(level.mean(), enough)
    out["pitched_share"] = result(pitched.mean(), enough)
    loud = level >= LOUD
    if loud.sum() * rows.hop >= 1.0:
        out["pitch_clarity"] = result(clarity[loud].mean(), enough)
    pitched_s = float(pitched.sum()) * rows.hop
    if pitched_s >= 1.0:  # under a second of pitch says nothing about a line
        sure = enough * (1.0 - np.exp(-pitched_s / 3.0))
        low, high = np.percentile(pitch[pitched], (10, 90))
        out["line_range_st"] = result(high - low, sure)
        steps = np.abs(np.diff(pitch))
        steps = steps[~np.isnan(steps) & (steps < 12)]  # a jump of an octave is a new line
        if len(steps) >= 10:
            out["line_motion_st_s"] = result(steps.mean() / rows.hop, sure)
        out["held_share"] = result(held(pitch, rows.hop)[pitched].mean(), sure)
    if rows.model:
        vocal = rows.vocal[part][music]
        out["vocal_presence"] = result(vocal.mean(), enough)
        vocal_s = float(vocal.sum()) * rows.hop
        if vocal_s >= 2.0:
            out["vocal_pitched_share"] = result(
                pitched[vocal].mean(), enough * (1.0 - np.exp(-vocal_s / 4.0))
            )
        starts = rows.time[part][rows.start[part]]
        ends = rows.time[part][rows.end[part]]
        lengths = [e - s for s in starts for e in ends[ends > s][:1]]
        if len(lengths) >= 2:
            out["phrase_s"] = result(float(np.median(lengths)), enough * (1 - np.exp(-len(lengths) / 3)))
    return out
