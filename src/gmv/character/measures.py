"""What one feature frame says about the sound, as numbers that do not depend on how loud
the recording or the playback is.

Nothing here analyses audio. Every value is read off fields a ``FeatureFrame`` already
carries, so the same frames give the same values live, from a file and from a recorded
session. All of them except ``level_db`` are unchanged when the audio is made louder or
quieter: they are ratios, shares, frequencies or differences in decibels. ``level_db`` is
kept only to take differences of (how far the level moves), never as a level.

A value that a frame cannot give (a field that is missing, not finite or of an unexpected
shape, such as the display spectrum in sessions recorded before it existed) is NaN, and
the aggregation leaves it out for that frame.
"""

import math

import numpy as np

# name -> (unit, what it is). The order is the order of the vector ``measure`` returns.
MEASURES = {
    "level_db": ("dBFS", "level of the frame; used only for differences"),
    "centroid_oct": ("log2 Hz", "spectral centroid: where the energy sits in frequency"),
    "rolloff_oct": ("log2 Hz", "frequency below which 85% of the energy lies"),
    "bandwidth_oct": ("log2 Hz", "spread of the energy round the centroid"),
    "low_share": ("0..1", "share of the energy below 250 Hz"),
    "body_share": ("0..1", "share of the energy from 250 Hz to 2 kHz"),
    "upper_share": ("0..1", "share of the energy from 2 to 6 kHz"),
    "air_share": ("0..1", "share of the energy above 6 kHz"),
    "flatness_db": ("dB", "log-band spectral flatness, 50 Hz to 10 kHz: 0 is noise-like"),
    "occupancy": ("0..1", "share of the log bands within 25 dB of the strongest"),
    "tilt_db_oct": ("dB/octave", "slope of the log-band spectrum, 50 Hz to 10 kHz"),
    "ripple_db": ("dB", "how far the log bands stand off that slope: high is peaky, sparse"),
    "contrast_db": ("dB", "peak-to-valley contrast inside the bands under 6 kHz: high is tonal"),
    "zero_crossing": ("0..1", "share of sample pairs that change sign"),
    "flux": ("0..1", "share of the spectrum that is newly risen since the frame before"),
    "attack": ("0..1", "share of the frame's level that is a rise since the frame before"),
    "crest_db": ("dB", "peak over RMS inside the frame"),
    "onset_rate": ("1/s", "onsets a second over the last four seconds"),
    "pulse_clarity": ("0..1", "strength of the strongest periodicity of the flux, 60-200 BPM"),
    "tempo_oct": ("log2 BPM", "that periodicity as a tempo, when the engine reports one"),
    # Filled in by the tracker, which knows the frames before this one:
    "punch_db": ("dB", "how far the frame stands above the level of the last 0.4 s"),
    "interval_s": ("s", "time between one onset and the next"),
    "interval_oct": ("log2 s", "the same on a ratio scale; its spread is the irregularity"),
}
NAMES = tuple(MEASURES)
INDEX = {name: index for index, name in enumerate(NAMES)}
COUNT = len(NAMES)
NAN = math.nan

SHAPE_HZ = (50.0, 10000.0)  # the part of the display spectrum the shape measures use
OCCUPIED_DB = 25.0  # a band this far or less under the strongest one counts as filled
SILENT_DBFS = -66.0  # as gmv.visualizer.intensity: below this nothing is measured

_shapes = {}  # (bands, low, high) -> (slice, centred octaves, the same / their sum of squares)


def number(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else NAN


def octaves(value):
    value = number(value)
    return math.log2(value) if value > 0 else NAN  # NaN compares false


def _shape(count, low, high):
    key = (count, low, high)
    found = _shapes.get(key)
    if found is None:
        edges = np.geomspace(low, high, count + 1)
        centres = np.sqrt(edges[:-1] * edges[1:])
        keep = np.flatnonzero((centres >= SHAPE_HZ[0]) & (centres <= SHAPE_HZ[1]))
        if len(keep) < 8:  # too little of the range to say anything about its shape
            found = (None, None, None)
        else:
            part = slice(int(keep[0]), int(keep[-1]) + 1)
            position = np.log2(centres[part])
            position -= position.mean()
            found = (part, position, position / float(position @ position))
        _shapes[key] = found
    return found


def given(frame, name):
    """The frame's value of a field that has a default, or NaN when the frame never said:
    a session recorded before the field existed must not read as "zero"."""
    stated = getattr(frame, "model_fields_set", None)
    return number(getattr(frame, name)) if stated is None or name in stated else NAN


def silent(frame):
    """Whether the frame is silence: nothing is measured in it and no time counts."""
    level = number(getattr(frame, "loudness_dbfs", NAN))
    return bool(getattr(frame, "silence", False)) or not level >= SILENT_DBFS


def measure(frame):
    """The measurements of one feature frame, as a vector in the order of ``NAMES``."""
    out = [NAN] * COUNT
    out[0] = number(frame.loudness_dbfs)
    out[1] = octaves(frame.spectral_centroid_hz)
    out[2] = octaves(frame.spectral_rolloff_hz)
    out[3] = octaves(frame.spectral_bandwidth_hz)
    bands = frame.band_energy or ()
    if len(bands) == 7:  # the seven bands of the analysis: 60, 250, 500, 2k, 4k, 6k between
        total = math.fsum(bands)
        if total > 0 and math.isfinite(total):
            out[4] = (bands[0] + bands[1]) / total
            out[5] = (bands[2] + bands[3]) / total
            out[6] = (bands[4] + bands[5]) / total
            out[7] = bands[6] / total
    spectrum = frame.spectrum_db
    low, high = frame.spectrum_low_hz, frame.spectrum_high_hz
    if spectrum and low and high and high > low > 0:
        part, position, slope = _shape(len(spectrum), float(low), float(high))
        if part is not None:
            levels = np.asarray(spectrum[part], dtype=np.float64)
            if np.isfinite(levels).all():
                mean = float(levels.mean())
                power = float(np.power(10.0, levels * 0.1).mean())
                out[8] = mean - 10.0 * math.log10(power) if power > 0 else NAN
                out[9] = float(np.count_nonzero(levels >= levels.max() - OCCUPIED_DB)) / len(levels)
                tilt = float(levels @ slope)
                out[10] = tilt
                rest = levels - mean - tilt * position
                out[11] = math.sqrt(float(rest @ rest) / len(rest))
    # The band above 6 kHz is left out: a lossy file has nothing up there, which reads as
    # a huge contrast that says something about the file and nothing about the music.
    contrast = (frame.spectral_contrast_db or ())[:6]
    if len(contrast) == 6:
        value = math.fsum(contrast) / 6
        out[12] = value if math.isfinite(value) else NAN
    out[13] = number(frame.zero_crossing_rate)
    out[14] = number(frame.spectral_flux)
    out[15] = given(frame, "attack_energy_ratio")
    rms, peak = number(frame.rms), number(frame.peak)
    if rms > 0 and peak > 0:
        out[16] = 20.0 * math.log10(peak / rms)
    out[17] = given(frame, "rhythmic_density")
    out[18] = given(frame, "tempo_confidence")
    out[19] = octaves(frame.tempo_bpm)
    return out
