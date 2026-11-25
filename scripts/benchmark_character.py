"""What Sonic Character costs: time per frame, time per profile, memory and allocation.

Generated audio goes through the real analysis pass once; the frames it produced are then
fed to ``SonicCharacter`` several times over and each part is timed on its own. Nothing is
captured and no file is read. Usage::

    uv run --no-sync python scripts/benchmark_character.py [--seconds 120] [--json OUT]
"""

import argparse
import json
import math
import platform
import sys
import tracemalloc
from time import perf_counter

import numpy as np

from gmv.character import WINDOWS, SonicCharacter
from gmv.character.aggregate import LevelDistribution, Windows
from gmv.character.measures import COUNT, measure
from gmv.evaluation.detect import run_detector
from gmv.models import FeatureFrame
from gmv.visualizer.intensity import MusicalIntensity

RATE = 48000


def music(seconds, seed=1):
    """Noise bursts over a bass tone with a slow swell: enough to exercise every path."""
    rng = np.random.default_rng(seed)
    time = np.arange(round(seconds * RATE)) / RATE
    burst = (np.modf(time * 4.0)[0] < 0.2) * rng.standard_normal(len(time))
    tone = np.sin(2 * np.pi * 82.4 * time) + 0.4 * np.sin(2 * np.pi * 660 * time)
    swell = 0.3 + 0.7 * np.sin(2 * np.pi * time / 23.0) ** 2
    audio = 0.25 * swell * (0.5 * burst + tone)
    return np.column_stack((audio, audio)).astype(np.float32)


def timed(function, items, repeats=5):
    """Best mean time per item over ``repeats`` passes, in microseconds."""
    best = math.inf
    for _ in range(repeats):
        began = perf_counter()
        for item in items:
            function(item)
        best = min(best, (perf_counter() - began) / len(items))
    return best * 1e6


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--json", help="Also write the figures to this file")
    args = parser.parse_args(argv)

    frames, events, dsp_ms, _ = run_detector(music(args.seconds), RATE)
    onsets = [e for e in events if e.event_type == "onset"]
    stream = sorted(
        [*frames, *onsets], key=lambda m: (m.timestamp, not isinstance(m, FeatureFrame))
    )

    def whole_pass():
        tracker = SonicCharacter()
        began = perf_counter()
        for message in stream:
            tracker.consume(message)
        return (perf_counter() - began) / len(frames) * 1e6, tracker

    per_frame, tracker = min((whole_pass() for _ in range(5)), key=lambda found: found[0])
    vectors = [measure(frame) for frame in frames]
    windows, levels = Windows(COUNT, tuple(WINDOWS.values())), LevelDistribution(tuple(WINDOWS.values()))
    intensity = MusicalIntensity()
    parts = {
        "measure (frame -> measurements)": timed(measure, frames),
        "windows (four lengths)": timed(lambda v: windows.add(v, 0.02), vectors),
        "level distribution": timed(lambda v: levels.add(v[0], 0.02), vectors),
        "MusicalIntensity.step (for comparison)": timed(intensity.step, frames),
    }
    profile_us = timed(lambda name: tracker.profile(name), ["long"] * 2000)
    all_us = timed(lambda _: tracker.profiles(), [None] * 500)

    # Memory: what one tracker holds, and what a frame leaves behind once it is taken in.
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    held = SonicCharacter()
    for message in stream[:2000]:
        held.consume(message)
    middle = tracemalloc.get_traced_memory()[0]
    for message in stream[2000:]:
        held.consume(message)
    after, peak = tracemalloc.get_traced_memory()
    state_bytes = sum(
        stat.size_diff for stat in tracemalloc.take_snapshot().compare_to(before, "filename")
    )
    tracemalloc.stop()
    growth = (after - middle) / max(1, len(stream) - 2000)

    analysis_us = float(np.mean(dsp_ms)) * 1000
    result = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "machine": platform.platform(),
        "frames": len(frames),
        "onset_events": len(onsets),
        "analysis_pass_us_per_frame": round(analysis_us, 1),
        "character_us_per_frame": round(per_frame, 2),
        "share_of_analysis_pass": round(per_frame / analysis_us, 4),
        "share_of_one_core_at_50_frames_a_second": round(per_frame * 50 / 1e6, 5),
        "parts_us_per_frame": {name: round(value, 2) for name, value in parts.items()},
        "profile_us": round(profile_us, 1),
        "all_four_profiles_us": round(all_us, 1),
        "tracker_state_bytes": int(state_bytes),
        "bytes_kept_per_message_after_warm_up": round(growth, 3),
        "peak_traced_bytes": int(peak),
    }
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)


if __name__ == "__main__":
    main()
