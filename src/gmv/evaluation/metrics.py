"""One-to-one MIR scoring, with errors and duplicate counts kept visible."""

import warnings
import numpy as np
import mir_eval

from gmv.corpus.intervals import merge, rasterize


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


def _nearest_distance(times, reference):
    times, reference = np.asarray(times, dtype=float), np.sort(np.asarray(reference, dtype=float))
    if not len(reference):
        return np.full(len(times), np.inf)
    index, last = np.searchsorted(reference, times), len(reference) - 1
    left = np.abs(times - reference[np.clip(index - 1, 0, last)])
    return np.minimum(left, np.abs(times - reference[np.clip(index, 0, last)]))


def drop_uncertain(predictions, certain, uncertain, tolerance):
    """Remove predictions explained only by an uncertain reference.

    The annotator abstained there, so the detector earns neither credit nor blame. Returns
    (kept predictions, number ignored).
    """
    predictions = np.asarray(predictions, dtype=float)
    if not len(predictions) or not len(uncertain):
        return predictions.tolist(), 0
    ignore = (_nearest_distance(predictions, uncertain) <= tolerance) & (
        _nearest_distance(predictions, certain) > tolerance
    )
    return predictions[~ignore].tolist(), int(ignore.sum())


def _boundary_errors(reference, predicted, window=0.5):
    errors = []
    for time in reference:
        if len(predicted):
            nearest = predicted[int(np.argmin(np.abs(np.asarray(predicted) - time)))]
            if abs(nearest - time) <= window:
                errors.append((nearest - time) * 1000)
    return {
        "reference": len(reference),
        "matched_within_500ms": len(errors),
        "bias_ms": float(np.mean(errors)) if errors else None,
        "median_absolute_error_ms": float(np.median(np.abs(errors))) if errors else None,
    }


def interval_score(reference, predicted, scored, uncertain=(), hop=0.01):
    """Frame-level activity scoring over the annotated ranges only.

    ``scored`` are the covered time ranges; ``uncertain`` reference spans are excluded and
    reported as abstained. Entrance/exit errors compare interval boundaries.
    """
    reference, predicted = merge(reference), merge(predicted)
    scored, uncertain = merge(scored), merge(uncertain)
    duration = max([e for _, e in reference + predicted + scored + uncertain] or [hop])
    ref, pred = rasterize(reference, duration, hop), rasterize(predicted, duration, hop)
    covered, unsure = rasterize(scored, duration, hop), rasterize(uncertain, duration, hop)
    mask = covered & ~unsure
    tp = float((ref & pred & mask).sum() * hop)
    fp = float((~ref & pred & mask).sum() * hop)
    fn = float((ref & ~pred & mask).sum() * hop)
    inactive = float((~ref & mask).sum() * hop)
    inside = [(s, e) for s, e in reference if any(a <= s < b for a, b in scored)]
    return {
        "scored_seconds": float(mask.sum() * hop),
        "abstained_seconds": float((covered & unsure).sum() * hop),
        "reference_active_seconds": tp + fn,
        "predicted_active_seconds": tp + fp,
        "tp_seconds": tp,
        "fp_seconds": fp,
        "fn_seconds": fn,
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
        "false_positive_rate": fp / inactive if inactive else None,
        "false_negative_rate": fn / (tp + fn) if tp + fn else None,
        "correct_empty": tp + fp + fn == 0,
        "entrances": _boundary_errors([s for s, _ in inside], [s for s, _ in predicted]),
        "exits": _boundary_errors([e for _, e in inside], [e for _, e in predicted]),
    }


def beat_score(reference, estimated, tolerance=0.07):
    """Beat F-measure at +/-70 ms, continuity, and half/double-time diagnosis."""
    score = onset_score(reference, estimated, tolerance)
    reference, estimated = np.sort(reference), np.sort(estimated)
    continuity = None
    if len(reference) >= 2 and len(estimated) >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            continuity = dict(
                zip(
                    ("cmlc", "cmlt", "amlc", "amlt"),
                    (float(v) for v in mir_eval.beat.continuity(reference, estimated)),
                    strict=True,
                )
            )
    ratio, relation = None, None
    if len(reference) >= 3 and len(estimated) >= 3:
        ratio = float(np.median(np.diff(reference)) / np.median(np.diff(estimated)))
        names = {1: "same", 2: "double_time", 0.5: "half_time", 3: "triple_time", 1 / 3: "third"}
        relation = next((n for k, n in names.items() if abs(ratio / k - 1) <= 0.08), "other")
    score.update(
        tolerance_seconds=tolerance,
        continuity=continuity,
        estimated_to_reference_tempo_ratio=ratio,
        metrical_level=relation,
    )
    return score
