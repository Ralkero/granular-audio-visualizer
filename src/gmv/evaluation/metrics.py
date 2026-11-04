"""One-to-one MIR scoring, with errors and duplicate counts kept visible."""

import warnings
import numpy as np
import mir_eval


def onset_score(reference, predictions, tolerance=0.05):
    reference, predictions = np.sort(reference), np.sort(predictions)
    matches = mir_eval.util.match_events(reference, predictions, window=tolerance)
    tp, fp, fn = len(matches), len(predictions) - len(matches), len(reference) - len(matches)
    used = {p for _, p in matches}
    duplicates = sum(
        any(abs(t - r) <= tolerance for r in reference)
        for i, t in enumerate(predictions)
        if i not in used
    )
    errors = np.asarray([predictions[p] - reference[r] for r, p in matches]) * 1000
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "duplicates": duplicates,
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        "correct_empty": not len(reference) and not len(predictions),
        "timing_bias_ms": float(np.mean(errors)) if len(errors) else None,
        "median_absolute_error_ms": float(np.median(np.abs(errors))) if len(errors) else None,
        "p95_absolute_error_ms": float(np.percentile(np.abs(errors), 95)) if len(errors) else None,
        "timing_errors_ms": errors.tolist(),
    }


def aggregate(scores):
    tp, fp, fn = (sum(s[key] for s in scores) for key in ("tp", "fp", "fn"))
    errors = [v for score in scores for v in score["timing_errors_ms"]]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "duplicates": sum(s["duplicates"] for s in scores),
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        "timing_bias_ms": float(np.mean(errors)) if errors else None,
        "median_absolute_error_ms": float(np.median(np.abs(errors))) if errors else None,
        "p95_absolute_error_ms": float(np.percentile(np.abs(errors), 95)) if errors else None,
    }


def rhythm_score(fixture, frames, events):
    intervals = []
    for begin, end, bpm in fixture.tempo_intervals:
        estimates = [(f.timestamp, f.tempo_bpm) for f in frames if begin + 4 <= f.timestamp < end]
        strict = [bool(value and abs(value / bpm - 1) <= 0.04) for _, value in estimates]
        octave = [
            bool(value and min(abs(value / (bpm * k) - 1) for k in (0.5, 1, 2)) <= 0.04)
            for _, value in estimates
        ]
        stable, streak = None, 0
        for (time, _), ok in zip(estimates, strict, strict=True):
            streak = streak + 1 if ok else 0
            if streak >= 50:
                stable = time - begin - 0.98
                break
        intervals.append(
            {
                "bpm": bpm,
                "start": begin,
                "end": end,
                "strict_accuracy_after_4s": sum(strict) / len(strict) if strict else None,
                "octave_tolerant_accuracy_after_4s": sum(octave) / len(octave) if octave else None,
                "stable_estimate_seconds": stable,
            }
        )
    observed = [e.timestamp for e in events if e.event_type in ("beat_observed", "beat_candidate")]
    predicted = [e.timestamp for e in events if e.event_type == "beat_prediction"]
    continuity = None
    if len(fixture.beats) >= 2 and len(observed) >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            continuity = dict(
                zip(
                    ("cmlc", "cmlt", "amlc", "amlt"),
                    mir_eval.beat.continuity(np.asarray(fixture.beats), np.asarray(observed)),
                    strict=True,
                )
            )
    return {
        "tempo_intervals": intervals,
        "false_tempo_frames_on_nonrhythmic": sum(f.tempo_bpm is not None for f in frames)
        if fixture.category.endswith("negative")
        else None,
        "observed_beat_50ms": onset_score(fixture.beats, observed),
        "predicted_beat_50ms": onset_score(fixture.beats, predicted),
        "continuity": continuity,
    }
