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


def tempo_span_score(begin, end, bpm, frames, warmup=4.0, confident=0.5):
    """How well frame tempo estimates follow one annotated tempo span.

    Time in the span with no analysis frames (a capture gap, a stopped stream) is reported
    as a gap and never counted as tracked. A frame without a tempo estimate is untracked.
    Accuracy is given twice: over all analysed frames, as before, and over the whole span,
    where gaps and untracked time count against the tracker.
    """
    inside = [f for f in frames if begin + warmup <= f.timestamp < end]
    span = max(0.0, end - begin - warmup)
    stamps = np.array([f.timestamp for f in frames])
    hop = float(np.median(np.diff(stamps))) if len(stamps) > 1 else 0.0
    values = [f.tempo_bpm for f in inside]
    strict = [bool(v and abs(v / bpm - 1) <= 0.04) for v in values]
    octave = [bool(v and min(abs(v / (bpm * k) - 1) for k in (0.5, 1, 2)) <= 0.04) for v in values]
    tracked = [v for v in values if v]
    sure = [f for f in inside if f.tempo_bpm and f.tempo_confidence >= confident]
    errors = [abs(v / bpm - 1) * 100 for v in tracked]
    analysed = min(span, len(inside) * hop)
    stable, streak = None, 0
    for frame, ok in zip(inside, strict, strict=True):
        streak = streak + 1 if ok else 0
        if streak >= 50:
            stable = frame.timestamp - begin - 0.98
            break

    def share(count):
        return min(1.0, count * hop / span) if span > 0 and hop else None

    return {
        "bpm": bpm,
        "start": begin,
        "end": end,
        "strict_accuracy_after_4s": sum(strict) / len(strict) if strict else None,
        "octave_tolerant_accuracy_after_4s": sum(octave) / len(octave) if octave else None,
        "stable_estimate_seconds": stable,
        "span_seconds_after_warmup": span,
        "analysed_seconds": analysed,
        "gap_seconds": max(0.0, span - analysed),
        "tracked_fraction_of_span": share(len(tracked)),
        "untracked_fraction_of_span": (1 - share(len(tracked))) if share(0) is not None else None,
        "confident_fraction_of_span": share(len(sure)),
        "confidence_threshold": confident,
        "strict_accuracy_of_span": share(sum(strict)),
        "strict_accuracy_on_tracked": sum(strict) / len(tracked) if tracked else None,
        "median_tempo_error_percent_on_tracked": float(np.median(errors)) if errors else None,
    }


def rhythm_score(fixture, frames, events):
    intervals = [
        tempo_span_score(begin, end, bpm, frames) for begin, end, bpm in fixture.tempo_intervals
    ]
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


def attack_recall(reference, onsets, tolerance=0.05, simultaneous=0.02):
    """Recall of full-mix onsets on one instrument's annotated attacks, matched one to one.

    A detected onset can account for at most one attack, so two quick hits answered by a
    single onset count as one found and one missed. Attacks closer together than
    ``simultaneous`` are one acoustic event (a kick and a hi-hat struck together, when the
    label covers the whole kit) and are counted once. Only recall is meaningful: onsets of
    instruments that are not annotated are real, so unmatched detections are not errors.
    """
    annotated = np.sort(np.asarray(reference, dtype=float))
    onsets = np.sort(np.asarray(onsets, dtype=float))
    distinct = (
        annotated[np.concatenate(([True], np.diff(annotated) > simultaneous))]
        if len(annotated)
        else annotated
    )
    matches = mir_eval.util.match_events(distinct, onsets, window=tolerance)
    errors = np.asarray([onsets[p] - distinct[r] for r, p in matches]) * 1000
    loose = int((_nearest_distance(distinct, onsets) <= tolerance).sum()) if len(onsets) else 0
    return {
        "reference_count": len(annotated),
        "distinct_attacks": len(distinct),
        "matched": len(matches),
        "recall": len(matches) / len(distinct) if len(distinct) else None,
        "shared_onsets": loose - len(matches),  # attacks that only a shared onset would cover
        "timing_bias_ms": float(np.mean(errors)) if len(errors) else None,
        "median_absolute_error_ms": float(np.median(np.abs(errors))) if len(errors) else None,
    }


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


def beat_score(reference, estimated, tolerance=0.07, uncertain=()):
    """Beat F-measure at +/-70 ms with off-beat, tempo and phase reported separately.

    ``uncertain`` reference beats earn neither credit nor blame: an estimate explained only
    by one of them is set aside (``ignored_predictions``), never counted as a hit.
    """
    kept, ignored = drop_uncertain(estimated, reference, uncertain, tolerance)
    score = onset_score(reference, kept, tolerance)
    reference, estimated = np.sort(reference), np.sort(kept)
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
    # Off-beat locking: the same estimates scored against the midpoints between beats.
    offbeat, phase, covered = None, None, None
    if len(reference) >= 2:
        middles = (reference[:-1] + reference[1:]) / 2
        offbeat = onset_score(middles, estimated, tolerance)
        if len(estimated):
            # Position of each estimate inside its reference beat: 0 on the beat, 0.5 between.
            index = np.clip(np.searchsorted(reference, estimated) - 1, 0, len(reference) - 2)
            period = reference[index + 1] - reference[index]
            within = (estimated >= reference[0]) & (estimated <= reference[-1])
            position = ((estimated - reference[index]) / period)[within] % 1
            distance = np.minimum(position, 1 - position)
            phase = {
                "estimates_inside_reference": int(within.sum()),
                "median_distance_from_beat_fraction": float(np.median(distance))
                if len(distance)
                else None,
                "share_nearer_the_off_beat": float((distance > 0.25).mean())
                if len(distance)
                else None,
            }
            # Reference beats that fall where the tracker was producing beats at all.
            typical = float(np.median(np.diff(reference)))
            near = _nearest_distance(reference, estimated) <= 1.5 * typical
            covered = float(near.mean())
    lock = None
    if offbeat is not None and (score["tp"] or offbeat["tp"]):
        lock = "off_beat" if offbeat["f1"] > score["f1"] else "on_beat"
    score.update(
        tolerance_seconds=tolerance,
        continuity=continuity,
        estimated_to_reference_tempo_ratio=ratio,
        metrical_level=relation,
        tempo_correct=None if ratio is None else bool(abs(ratio - 1) <= 0.04),
        offbeat_f1=offbeat["f1"] if offbeat else None,
        offbeat_matches=offbeat["tp"] if offbeat else None,
        phase_lock=lock,
        phase=phase,
        reference_beats_with_tracking_fraction=covered,
        abstained_references=len(uncertain),
        ignored_predictions=ignored,
    )
    return score
