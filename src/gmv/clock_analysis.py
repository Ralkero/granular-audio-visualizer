"""Separate capture-clock drift into steps (lost audio) and smooth slope (rate mismatch).

Input is per-callback arrival time and frame count only. Arrival jitter can only delay a
callback, so the per-window minimum of (wall time - delivered sample time) is a jitter-robust
envelope. A step in that envelope means wall time passed without samples being delivered;
a steady slope means samples arrive at a rate different from the nominal one. Losses smaller
than ``jump_ms`` per window are indistinguishable from a slope using timing alone.
"""

import numpy as np


def analyze_clock(arrival, frames, rate, *, window_s=1.0, jump_ms=3.0, gap_s=0.5):
    arrival = np.asarray(arrival, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.float64)
    rate = np.broadcast_to(np.asarray(rate, dtype=np.float64), frames.shape)
    if len(arrival) < 3:
        raise ValueError("Need at least three callbacks")
    block = frames / rate
    dt = np.diff(arrival, prepend=arrival[0])
    is_gap = dt > gap_s
    # Long outages are rebased by the capture clock; remove them so they are counted apart.
    gap_excess = np.where(is_gap, dt - block, 0.0)
    t = arrival - arrival[0]
    offset = t - np.cumsum(block) - np.cumsum(gap_excess)

    index = np.floor(t / window_s).astype(np.int64)
    windows = np.unique(index)
    env_t = np.array([t[index == w].mean() for w in windows])
    env = np.array([offset[index == w].min() for w in windows])
    steps = np.diff(env)
    jumps = np.abs(steps) > jump_ms / 1000
    smooth = env - np.concatenate([[0.0], np.cumsum(np.where(jumps, steps, 0.0))])
    slope = float(np.polyfit(env_t, smooth, 1)[0]) if len(env) > 2 else 0.0

    active = t[-1] - float(gap_excess.sum())
    inter = dt[1:][~is_gap[1:]] * 1000
    sizes, counts = np.unique(frames.astype(np.int64), return_counts=True)
    return {
        "callbacks": int(len(arrival)),
        "elapsed_s": float(t[-1]),
        "active_s": float(active),
        "nominal_rate": float(np.median(rate)),
        "effective_rate": float((frames[1:].sum()) / active) if active > 0 else None,
        "envelope_drift_ms": float((env[-1] - env[0]) * 1000),
        "envelope_start_ms": float(env[0] * 1000),
        "envelope_end_ms": float(env[-1] * 1000),
        "step_count": int(jumps.sum()),
        "step_total_ms": float(steps[jumps].sum() * 1000),
        "step_times_s": [round(float(x), 2) for x in env_t[1:][jumps][:200]],
        "step_sizes_ms": [round(float(x) * 1000, 2) for x in steps[jumps][:200]],
        "smooth_slope_ppm": slope * 1e6,
        "smooth_slope_ms_per_min": slope * 60000,
        "smooth_total_ms": float((smooth[-1] - smooth[0]) * 1000),
        "outage_count": int(is_gap.sum()),
        "outage_total_s": float(gap_excess.sum()),
        "interarrival_ms": {
            "p50": float(np.percentile(inter, 50)),
            "p99": float(np.percentile(inter, 99)),
            "max": float(inter.max()),
        }
        if len(inter)
        else None,
        "frames_per_callback": {int(s): int(c) for s, c in zip(sizes, counts, strict=True)},
        "envelope": {
            "t_s": [round(float(x), 2) for x in env_t],
            "offset_ms": [round(float(x) * 1000, 3) for x in env],
        },
    }


def analyze_timeline(arrival, timestamp, *, window_s=1.0, edge_windows=10):
    """Signed (arrival - session timestamp) envelope: does the published timeline track wall time?

    ``arrival`` is relative to the session origin. A constant value is a fixed pipeline offset
    (it may be negative when the device stamps packets with a presentation time ahead of
    delivery); growth over the run is timestamp drift.
    """
    arrival = np.asarray(arrival, dtype=np.float64)
    age = (arrival - np.asarray(timestamp, dtype=np.float64)) * 1000
    index = np.floor((arrival - arrival[0]) / window_s).astype(np.int64)
    windows = np.unique(index)
    env_t = np.array([arrival[index == w].mean() - arrival[0] for w in windows])
    env = np.array([age[index == w].min() for w in windows])
    edge = max(1, min(edge_windows, len(env) // 2))
    start, end = float(np.median(env[:edge])), float(np.median(env[-edge:]))
    return {
        "start_ms": start,
        "end_ms": end,
        "drift_ms": end - start,
        "envelope_min_ms": float(env.min()),
        "envelope_max_ms": float(env.max()),
        "slope_ms_per_min": float(np.polyfit(env_t, env, 1)[0] * 60) if len(env) > 2 else 0.0,
        "block_age_ms": {
            "p50": float(np.percentile(age, 50)),
            "p99": float(np.percentile(age, 99)),
            "max": float(age.max()),
        },
        "envelope": {
            "t_s": [round(float(x), 2) for x in env_t],
            "age_ms": [round(float(x), 3) for x in env],
        },
    }
