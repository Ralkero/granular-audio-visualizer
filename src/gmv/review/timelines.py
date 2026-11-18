"""The existing model's guitar timeline over whole recordings: model output only.

Each recording is fed through ``gmv.evaluation.detect.run_recognition``, the offline twin
of the application's recognition worker: the same ``Recognizer`` (channel mean, resample to
the model rate, 10 s context ending at each hop, smoothing over two estimates) and the
model description's own thresholds. Nothing here is a label.
"""

import numpy as np

LEVELS = {"absent": 0, "candidate": 1, "detected": 2}
STATE_NAMES = {v: k for k, v in LEVELS.items()}


def compute(audio, rate, backend):
    from ..evaluation.detect import run_recognition

    estimates, times = run_recognition(audio, rate, backend)
    labels = list(backend.labels)
    out = {
        "t_start": np.array([e.start for e in estimates], np.float64),
        "t_end": np.array([e.end for e in estimates], np.float64),
        "inference_ms": np.array(times, np.float32),
    }
    for i, label in enumerate(labels):
        key = "guitar" if label == "guitar" else label.replace("guitar_", "")
        out[key] = np.array([e.scores[i][1] for e in estimates], np.float32)
        out[f"{key}_state"] = np.array([LEVELS[e.states[i][1]] for e in estimates], np.int8)
    return out


def save(path, timeline):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **timeline)


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def runs(mask, hop):
    """[(start_index, length_s)] of runs of True."""
    out, begin = [], None
    for i, value in enumerate(list(mask) + [False]):
        if value and begin is None:
            begin = i
        elif not value and begin is not None:
            out.append((begin, (i - begin) * hop))
            begin = None
    return out


def observations(timeline, levels_db=None, hop=0.5, expect_guitar=None):
    """Descriptive statistics of the model's behaviour on one recording. These are
    observations about the model, not labels: "possible" always means "to be checked by
    listening"."""
    state = timeline["guitar_state"]
    ends = timeline["t_end"]
    detected, active = state == 2, state >= 1
    det_runs = runs(detected, hop)
    idle_runs = runs(~active, hop)
    out = {
        "model_hops": int(len(state)),
        "model_share_detected": round(float(detected.mean()), 4) if len(state) else None,
        "model_share_detected_or_candidate": round(float(active.mean()), 4) if len(state) else None,
        "model_mean_guitar_score": round(float(timeline["guitar"].mean()), 4)
        if len(state)
        else None,
        "model_longest_detected_s": max((r[1] for r in det_runs), default=0.0),
        "model_longest_inactive_s": max((r[1] for r in idle_runs), default=0.0),
        "model_state_changes_per_min": round(
            float(np.count_nonzero(np.diff(state)) / max(1e-9, len(state) * hop / 60)), 2
        )
        if len(state)
        else None,
    }
    if expect_guitar is True:
        spans = [(float(ends[i] - hop), length) for i, length in idle_runs if length >= 10]
        out["possible_missed_passages"] = (
            "; ".join(f"{_clock(s)} for {length:.0f} s" for s, length in spans) or None
        )
    elif expect_guitar is False:
        spans = [(float(ends[i] - hop), length) for i, length in det_runs if length >= 4]
        out["possible_false_activations"] = (
            "; ".join(f"{_clock(s)} for {length:.0f} s" for s, length in spans) or None
        )
    if levels_db is not None and len(levels_db):
        # level of the 0.5 s before each hop end
        index = np.clip(np.round(ends / hop).astype(int) - 1, 0, len(levels_db) - 1)
        quiet = levels_db[index] < -35
        out["quiet_hops"] = int(quiet.sum())
        out["quiet_hops_model_inactive"] = int((quiet & ~active).sum())
    return out


def _clock(seconds):
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def window_readout(timeline, start, end, hop=0.5):
    """Model output over hops whose half-second span lies inside [start, end]."""
    ends = timeline["t_end"]
    inside = (ends - hop >= start - 1e-6) & (ends <= end + 1e-6)
    out = {"model_hops_in_excerpt": int(inside.sum())}
    if not inside.any():
        return out
    for key in ("guitar", "distorted", "clean", "acoustic"):
        if key not in timeline:
            continue
        values, states = timeline[key][inside], timeline[f"{key}_state"][inside]
        out[f"model_{key}_mean"] = round(float(values.mean()), 4)
        out[f"model_{key}_max"] = round(float(values.max()), 4)
        out[f"model_{key}_share_detected"] = round(float((states == 2).mean()), 4)
        out[f"model_{key}_share_candidate_or_detected"] = round(float((states >= 1).mean()), 4)
    counts = np.bincount(timeline["guitar_state"][inside], minlength=3)
    out["model_guitar_majority_state"] = STATE_NAMES[int(np.argmax(counts))]
    out["model_context_complete"] = bool(start >= 10.0 - 1e-6)
    return out
