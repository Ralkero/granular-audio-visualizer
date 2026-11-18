"""Where excerpts go and how they are cut.

Three categories, always kept apart:

``core``           five positions per recording chosen without looking at any model or
                   audio feature: start, middle, end and two seeded random positions
``feature``        the passage least like the rest of the track by level and spectral
                   shape (audio measurements only, no model)
``model_flagged``  a passage where the existing model's output suggests a possible miss or
                   false activation: diagnostic material, never part of unbiased results

Excerpts never overlap. Each is written as 32-bit float WAV with the source's own sample
rate and channels: the decoded samples, unchanged.
"""

from dataclasses import dataclass

import numpy as np
import soundfile as sf

from .audio import level_profile

LENGTH_S = 12.0
CATEGORIES = ("core", "feature", "model_flagged")
GRID_S = 0.5


@dataclass(frozen=True)
class Placement:
    excerpt_id: str
    category: str
    position: str  # start / middle / end / random_1 / random_2 / feature / model_flagged
    start_s: float
    end_s: float
    reason: str


def _free(start, length, taken):
    return all(start + length <= a or start >= b for a, b in taken)


def core_placements(recording_id, sha256, duration, length=LENGTH_S):
    """The five unbiased positions, deterministic from the file hash. Short recordings get
    fewer (recorded, not padded)."""
    if duration <= length:
        return [
            Placement(
                f"{recording_id}-c1",
                "core",
                "start",
                0.0,
                round(duration, 3),
                "whole recording (shorter than one excerpt)",
            )
        ]
    taken, out = [], []
    fixed = (
        ("start", 0.0, "fixed position: beginning of the recording"),
        ("end", duration - length, "fixed position: end of the recording"),
        ("middle", duration / 2 - length / 2, "fixed position: middle of the recording"),
    )
    for name, start, reason in fixed:
        start = round(max(0.0, min(start, duration - length)), 3)
        if _free(start, length, taken):
            taken.append((start, start + length))
            out.append((name, start, reason))
    rng = np.random.default_rng(int(sha256[:16], 16))
    for k in (1, 2):
        grid = np.arange(0.0, duration - length + 1e-9, GRID_S)
        free = [round(float(s), 3) for s in grid if _free(s, length, taken)]
        if not free:
            break
        start = free[int(rng.integers(len(free)))]
        taken.append((start, start + length))
        out.append((f"random_{k}", start, f"seeded random position {k} (seed: file SHA-256)"))
    order = {"start": 0, "middle": 1, "end": 2, "random_1": 3, "random_2": 4}
    out.sort(key=lambda item: order[item[0]])
    return [
        Placement(f"{recording_id}-c{i}", "core", name, start, round(start + length, 3), reason)
        for i, (name, start, reason) in enumerate(out, 1)
    ]


def _windows(duration, length, taken):
    grid = np.arange(0.0, duration - length + 1e-9, GRID_S)
    return [float(s) for s in grid if _free(s, length, taken)]


FEATURE_WORDS = {
    ("level", -1): "quietest passage",
    ("level", 1): "loudest passage",
    ("bright", -1): "darkest passage",
    ("bright", 1): "brightest passage",
}


def feature_placement(recording_id, audio, rate, taken, length=LENGTH_S):
    """The free window whose level and brightness differ most from the track's medians
    (robust z-scores). Uses audio measurements only."""
    duration = len(audio) / rate
    windows = _windows(duration, length, taken)
    if not windows:
        return None
    levels = level_profile(audio, rate, GRID_S)
    mono = audio.mean(axis=1)
    size = round(GRID_S * rate)
    count = min(len(levels), len(mono) // size)
    if count < 2:
        return None
    blocks = mono[: count * size].reshape(count, size)
    spectrum = np.abs(np.fft.rfft(blocks * np.hanning(size), axis=1)) ** 2
    freq = np.fft.rfftfreq(size, 1 / rate)
    bright = (spectrum * freq).sum(axis=1) / (spectrum.sum(axis=1) + 1e-12)
    features = {"level": levels[:count], "bright": bright}
    span = round(length / GRID_S)
    best = None
    for start in windows:
        i = round(start / GRID_S)
        scores = {}
        for name, values in features.items():
            median = np.median(values)
            spread = np.median(np.abs(values - median)) + 1e-9
            scores[name] = (np.mean(values[i : i + span]) - median) / spread
        distance = float(np.sqrt(sum(v * v for v in scores.values())))
        if best is None or distance > best[0]:
            best = (distance, start, scores)
    distance, start, scores = best
    name = max(scores, key=lambda k: abs(scores[k]))
    words = FEATURE_WORDS[(name, 1 if scores[name] > 0 else -1)]
    reason = (
        f"{words} relative to the rest of the track (level z {scores['level']:+.1f}, "
        f"brightness z {scores['bright']:+.1f}); audio measurements only"
    )
    return Placement(
        f"{recording_id}-f1",
        "feature",
        "feature",
        round(start, 3),
        round(start + length, 3),
        reason,
    )


def model_placement(recording_id, timeline, thresholds, category, duration, taken, length=LENGTH_S):
    """A possible miss (guitar expected) or possible false activation (guitar not
    expected) according to the model's own timeline. Returned only when the model output
    in that window actually looks that way; marked as diagnostic material."""
    ends, scores = timeline["t_end"], timeline["guitar"]
    windows = _windows(duration, length, taken)
    if not windows or not len(ends):
        return None
    means = []
    for start in windows:
        inside = (ends - 0.5 >= start - 1e-6) & (ends <= start + length + 1e-6)
        means.append(float(scores[inside].mean()) if inside.any() else np.nan)
    means = np.array(means)
    if np.all(np.isnan(means)):
        return None
    expect_guitar = category in ("positive_candidate", "mixed_candidate")
    if expect_guitar:
        i = int(np.nanargmin(means))
        if means[i] >= thresholds["candidate"]:
            return None
        reason = (
            f"possible miss: guitar expected from metadata, mean model score {means[i]:.2f} "
            f"is below the candidate threshold {thresholds['candidate']:.2f}"
        )
    else:
        i = int(np.nanargmax(means))
        if means[i] < thresholds["candidate"]:
            return None
        reason = (
            f"possible false activation: guitar not expected from metadata, mean model "
            f"score {means[i]:.2f} is at or above the candidate threshold {thresholds['candidate']:.2f}"
        )
    start = windows[i]
    return Placement(
        f"{recording_id}-m1",
        "model_flagged",
        "model_flagged",
        round(start, 3),
        round(start + length, 3),
        reason + "; diagnostic only",
    )


def sample_span(placement, rate, frames):
    begin = int(round(placement.start_s * rate))
    end = min(frames, begin + int(round((placement.end_s - placement.start_s) * rate)))
    return begin, end


def write_excerpt(audio, rate, placement, path):
    """Write the decoded samples of one excerpt unchanged (float WAV). Returns the sample
    span written."""
    begin, end = sample_span(placement, rate, len(audio))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio[begin:end], rate, subtype="FLOAT", format="WAV")
    return begin, end


def matches_source(path, audio, begin, end):
    """True when the excerpt file holds exactly the source samples [begin, end)."""
    data, _ = sf.read(path, dtype="float32", always_2d=True)
    source = audio[begin:end]
    return data.shape == source.shape and bool(np.array_equal(data, source))
