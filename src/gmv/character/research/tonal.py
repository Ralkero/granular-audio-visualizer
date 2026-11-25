"""Harmony and tonality, from the harmonic part of the spectrogram.

Pitch classes (chroma) are read between 190 Hz and 2.1 kHz, where one FFT bin is narrower
than a semitone, on a grid of a third of a semitone; the recording's own tuning is taken
as the third that gathers most energy, so music tuned off A = 440 Hz is not smeared.

``measured`` values state what is in the signal. ``inferred`` ones interpret it (a key, a
chord change) and are wrong whenever the interpretation does not apply: they are reported
for study and carry that word in ``DEFINITIONS``.

What defeats all of this: distortion (its intermodulation adds real partials that are not
the notes played), unpitched percussion that survives the split, dense mixes where every
pitch class is occupied, and anything whose harmony lives below 190 Hz alone.
"""

import numpy as np

from .frontend import FRAME_S, result

LOW_HZ, HIGH_HZ = 190.0, 2100.0
SMOOTH_S = 0.5  # chroma is averaged over this before anything is read from it
STEP_S = 0.37  # harmonic movement compares the chroma this far before and after
NEEDED_S = 8.0

# name -> (unit, kind, what it is, usual range)
DEFINITIONS = {
    "harmonic_share": ("0..1", "measured", "share of the power that is steady in time (tonal) rather than struck", "0.2 .. 0.9"),
    "pitch_class_focus": ("0..1", "measured", "1 - entropy of the passage's pitch classes: 0 all twelve equally, 1 a single one", "0.02 .. 0.4"),
    "pitch_classes_at_once": ("count 1..12", "measured", "effective number of pitch classes sounding together (exp of the frame entropy)", "4 .. 11"),
    "harmonic_movement": ("tonal distance / s", "measured", "how fast the pitch-class content moves, as distance in Harte's tonal-centroid space per second", "0.1 .. 1.5"),
    "tonal_stability": ("0..1", "measured", "mean similarity of each moment's pitch classes to the passage's own", "0.6 .. 0.98"),
    "key_clarity": ("correlation -1..1", "inferred", "how well the pitch classes fit the best of the 24 major and minor key profiles", "0.3 .. 0.9"),
    "major_lean": ("-1..1", "inferred", "fit of the best major key minus the best minor key", "-0.3 .. 0.3"),
    "chord_change_rate": ("1/s", "inferred", "peaks of harmonic movement a second, read as chord changes", "0.2 .. 2"),
}

MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_ANGLES = np.arange(12)
CENTROID = np.array(
    [
        np.sin(_ANGLES * 7 * np.pi / 6),
        np.cos(_ANGLES * 7 * np.pi / 6),
        np.sin(_ANGLES * 3 * np.pi / 2),
        np.cos(_ANGLES * 3 * np.pi / 2),
        0.5 * np.sin(_ANGLES * 2 * np.pi / 3),
        0.5 * np.cos(_ANGLES * 2 * np.pi / 3),
    ]
)


def chroma(analysis):
    """Pitch-class strength per frame (frames, 12), each row summing to 1 (or all zero),
    and the tuning offset used, in thirds of a semitone (-1, 0 or 1)."""
    freq = analysis.freq
    keep = (freq >= LOW_HZ) & (freq <= HIGH_HZ)
    thirds = np.round(36 * np.log2(freq[keep] / 440.0)).astype(int) % 36
    magnitude = np.sqrt(analysis.harmonic[:, keep])
    fine = np.zeros((len(magnitude), 36))
    for index in range(36):
        fine[:, index] = magnitude[:, thirds == index].sum(axis=1)
    music = analysis.music
    energy = [fine[music][:, offset % 3 :: 3].sum() if music.any() else 0.0 for offset in (0, 1, 2)]
    offset = int(np.argmax(energy))
    offset = offset if offset < 2 else -1
    rolled = np.roll(fine, -offset, axis=1)
    # A semitone is its centre third and half of each neighbour.
    twelve = rolled[:, 0::3] + 0.5 * (np.roll(rolled, 1, axis=1)[:, 0::3] + rolled[:, 1::3])
    twelve = np.roll(twelve, 9, axis=1)  # index 0 is C (the grid starts at A)
    reach = max(1, round(SMOOTH_S / FRAME_S))
    kernel = np.ones(reach) / reach
    smooth = np.apply_along_axis(lambda column: np.convolve(column, kernel, mode="same"), 0, twelve)
    total = smooth.sum(axis=1, keepdims=True)
    return np.where(total > 0, smooth / np.maximum(total, 1e-20), 0.0), offset


def movement(chromagram):
    """Tonal-centroid distance per second between the chroma ``STEP_S`` before and after
    each frame (Harte, Sandler and Gasser's harmonic change function)."""
    centroid = chromagram @ CENTROID.T
    step = max(1, round(STEP_S / FRAME_S))
    change = np.zeros(len(centroid))
    if len(centroid) > 2 * step:
        change[step:-step] = np.linalg.norm(centroid[2 * step :] - centroid[: -2 * step], axis=1)
    return change / (2 * step * FRAME_S)


def key(profile):
    """(best correlation, best major minus best minor) of a pitch-class profile."""
    major = max(np.corrcoef(np.roll(MAJOR, shift), profile)[0, 1] for shift in range(12))
    minor = max(np.corrcoef(np.roll(MINOR, shift), profile)[0, 1] for shift in range(12))
    return max(major, minor), major - minor


def measure(analysis, start, end, cache=None):
    """The tonal measurements of [start, end) seconds."""
    cache = cache if cache is not None else {}
    if "chroma" not in cache:
        cache["chroma"], cache["tuning"] = chroma(analysis)
        cache["movement"] = movement(cache["chroma"])
    part = analysis.frames(start, end)
    music = analysis.music[part]
    seconds = float(music.sum()) * FRAME_S
    out = {name: (None, 0.0) for name in DEFINITIONS}
    if seconds < 1.0:
        return out
    power = analysis.power[part][music]
    share = float(analysis.harmonic[part][music].sum() / max(float(power.sum()), 1e-20))
    enough = 1.0 - np.exp(-seconds / NEEDED_S)
    out["harmonic_share"] = result(share, enough)
    rows = cache["chroma"][part][music]
    rows = rows[rows.sum(axis=1) > 0]
    if len(rows) < 8:
        return out
    # Pitch-class measurements mean little when hardly anything is tonal.
    tonal = enough * min(1.0, share / 0.3)
    profile = rows.mean(axis=0)
    entropy = -(profile * np.log(np.maximum(profile, 1e-12))).sum() / np.log(12)
    out["pitch_class_focus"] = result(1.0 - entropy, tonal)
    each = -(rows * np.log(np.maximum(rows, 1e-12))).sum(axis=1)
    out["pitch_classes_at_once"] = result(float(np.exp(each.mean())), tonal)
    moved = cache["movement"][part][music]
    out["harmonic_movement"] = result(float(moved.mean()), tonal)
    norm = np.linalg.norm(rows, axis=1) * np.linalg.norm(profile)
    out["tonal_stability"] = result(float(((rows @ profile) / np.maximum(norm, 1e-12)).mean()), tonal)
    if profile.std() > 0:
        clarity, lean = key(profile)
        out["key_clarity"] = result(clarity, tonal)
        # A lean is only worth stating when some key fits at all.
        out["major_lean"] = result(lean, tonal * min(1.0, max(0.0, (clarity - 0.4) / 0.4)))
    floor = np.median(moved) + 1.5 * np.median(np.abs(moved - np.median(moved)))
    peaks = (moved[1:-1] > moved[:-2]) & (moved[1:-1] >= moved[2:]) & (moved[1:-1] > floor)
    out["chord_change_rate"] = result(float(peaks.sum()) / seconds, tonal * 0.5)
    return out
