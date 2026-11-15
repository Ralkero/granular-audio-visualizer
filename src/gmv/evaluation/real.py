"""Evaluate a detector against annotated recordings from the versioned corpus.

Every class is reported on its own. A layer that was not annotated is reported as
``not_annotated``; an annotated class the engine has no output for is reported as
``no_detector_output`` with its reference counts, never omitted and never scored as zero
by silence. Synthetic and real-corpus results are never aggregated together.
"""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import mir_eval
import numpy as np

from gmv.corpus import registry
from gmv.corpus.compare import consensus
from gmv.corpus.intervals import total
from gmv.corpus.schema import ANNOTATION_SCHEMA_VERSION, CORPUS_SCHEMA_VERSION, event_layer

from .corpus import CLASSES
from .detect import run_detector
from .metrics import (
    aggregate,
    beat_score,
    drop_uncertain,
    interval_score,
    onset_score,
    rhythm_score,
)

# The engine's three band-ratio candidates are not instrument detectors. This mapping only
# says which annotated attacks each candidate is compared with; the confusion table shows
# what the candidates actually coincide with.
CANDIDATE_TARGETS = {
    "bass_transient_candidate": ("kick", "bass"),
    "snare_like_candidate": ("snare",),
    "high_frequency_percussion_candidate": ("cymbals",),
}
OBSERVED_BEATS = ("beat_observed", "beat_candidate")
NOT_ANNOTATED = {"status": "not_annotated"}


def _score_points(certain, uncertain, predictions, tolerance):
    kept, ignored = drop_uncertain(predictions, certain, uncertain, tolerance)
    score = onset_score(certain, kept, tolerance)
    score.update(abstained_references=len(uncertain), ignored_predictions=ignored)
    return score


def _latency(reference, predictions, emitted_at, tolerance=0.05):
    order = np.argsort(predictions)
    reference, sorted_predictions = np.sort(reference), np.asarray(predictions)[order]
    matches = mir_eval.util.match_events(reference, sorted_predictions, window=tolerance)
    delays = [(emitted_at[order[p]] - reference[r]) * 1000 for r, p in matches]
    if not delays:
        return None
    return {
        "matched": len(delays),
        "p50_ms": float(np.percentile(delays, 50)),
        "p95_ms": float(np.percentile(delays, 95)),
        "scope": "end of the analysis window that produced the event minus reference time; "
        "excludes capture, queueing and publication",
    }


def evaluate_recording(
    recording, annotation, audio, rate, detector="candidate", config=None, activity_predictions=None
):
    """Score one recording. ``activity_predictions`` maps label -> [(start, end)] if available."""
    ranges = [(r.start, r.end) for r in annotation.coverage.ranges]

    def inside(time):
        return any(start <= time < end for start, end in ranges)

    def split(items):
        items = [i for i in items if inside(i.time)]
        return (
            [i.time for i in items if i.certainty == "certain"],
            [i.time for i in items if i.certainty == "uncertain"],
        )

    frames, events, times, emitted_at = run_detector(
        audio, rate, detector, config, recording.recording_id
    )
    emitted = {id(e): t for e, t in zip(events, emitted_at, strict=True)}

    def predicted(*kinds):
        return [e for e in events if e.event_type in kinds and inside(e.timestamp)]

    result = {
        "recording": {
            key: getattr(recording, key)
            for key in ("recording_id", "sha256", "split", "group_id", "source_kind", "duration_s")
        }
        | {"tags": list(recording.tags)},
        "annotation": {
            "annotator_id": annotation.annotator_id,
            "status": annotation.status,
            "tool": annotation.tool,
            "uncertainty_seconds": annotation.uncertainty_seconds,
            "layers": list(annotation.coverage.layers),
            "covered_seconds": total(ranges),
        },
    }

    if annotation.covers("onset"):
        certain, uncertain = split([e for e in annotation.events if e.kind == "onset"])
        found = predicted("onset")
        stamps = [e.timestamp for e in found]
        wide = 0.05 + annotation.uncertainty_seconds
        result["onsets"] = {
            "25": _score_points(certain, uncertain, stamps, 0.025),
            "50": _score_points(certain, uncertain, stamps, 0.05),
            "50_plus_annotation_uncertainty": _score_points(certain, uncertain, stamps, wide),
            "latency": _latency(certain, stamps, [emitted[id(e)] for e in found])
            if stamps and certain
            else None,
        }
    else:
        result["onsets"] = NOT_ANNOTATED

    if annotation.covers("beat"):
        certain, _ = split([e for e in annotation.events if e.kind == "beat"])
        result["beats"] = {
            "reference_count": len(certain),
            "observed": beat_score(certain, [e.timestamp for e in predicted(*OBSERVED_BEATS)]),
            "predicted": beat_score(certain, [e.timestamp for e in predicted("beat_prediction")]),
        }
    else:
        result["beats"] = NOT_ANNOTATED
    if annotation.covers("downbeat"):
        certain, uncertain = split([e for e in annotation.events if e.kind == "downbeat"])
        result["downbeats"] = {
            "status": "no_detector_output",
            "reference_count": len(certain),
            "abstained_references": len(uncertain),
        }
    else:
        result["downbeats"] = NOT_ANNOTATED

    if annotation.covers("tempo"):
        marks = sorted(
            (e.time, float(e.value))
            for e in annotation.events
            if e.kind == "tempo_change" and e.certainty == "certain"
        )
        end = max(e for _, e in ranges)
        spans = [
            (time, marks[i + 1][0] if i + 1 < len(marks) else end, bpm)
            for i, (time, bpm) in enumerate(marks)
        ]
        proxy = SimpleNamespace(tempo_intervals=spans, beats=[], category="real_music")
        result["tempo"] = {"spans": rhythm_score(proxy, frames, [])["tempo_intervals"]}
    else:
        result["tempo"] = NOT_ANNOTATED

    hit_layers = [layer[5:] for layer in annotation.coverage.layers if layer.startswith("hits.")]
    hits = {
        label: split([e for e in annotation.instrument_events if e.label == label])
        for label in hit_layers
    }
    result["instrument_events"] = {
        label: {
            "status": "no_detector_output",
            "reference_count": len(certain),
            "abstained_references": len(uncertain),
        }
        for label, (certain, uncertain) in hits.items()
    }
    candidates, confusion = {}, {}
    for kind in CLASSES:
        stamps = [e.timestamp for e in predicted(kind)]
        targets = [label for label in CANDIDATE_TARGETS[kind] if label in hits]
        if targets:
            certain = sorted(t for label in targets for t in hits[label][0])
            uncertain = sorted(t for label in targets for t in hits[label][1])
            candidates[kind] = {
                "compared_with": targets,
                "interpretation": "uncalibrated band-ratio candidate, not an instrument detector",
                "score_50ms": _score_points(certain, uncertain, stamps, 0.05),
            }
        else:
            candidates[kind] = {**NOT_ANNOTATED, "needs_any_of": list(CANDIDATE_TARGETS[kind])}
        if hits:
            counts = {
                label: sum(any(abs(t - r) <= 0.05 for r in certain + uncertain) for t in stamps)
                for label, (certain, uncertain) in hits.items()
            }
            every = [r for certain, uncertain in hits.values() for r in certain + uncertain]
            counts["no_annotated_attack_within_50ms"] = sum(
                not any(abs(t - r) <= 0.05 for r in every) for t in stamps
            )
            confusion[kind] = {"emitted": len(stamps), "coincides_with": counts}
    result["candidates"], result["candidate_coincidence"] = candidates, confusion

    activity = {}
    for layer in annotation.coverage.layers:
        if not layer.startswith("activity."):
            continue
        label = layer[9:]
        spans = [a for a in annotation.activity if a.label == label]
        certain = [(a.start, a.end) for a in spans if a.certainty == "certain"]
        uncertain = [(a.start, a.end) for a in spans if a.certainty == "uncertain"]
        if activity_predictions and label in activity_predictions:
            activity[label] = {
                "status": "scored",
                **interval_score(certain, activity_predictions[label], ranges, uncertain),
            }
        else:
            reference = interval_score(certain, [], ranges, uncertain)
            activity[label] = {
                "status": "no_detector_output",
                "reference_active_seconds": reference["reference_active_seconds"],
                "scored_seconds": reference["scored_seconds"],
                "abstained_seconds": reference["abstained_seconds"],
                "reference_intervals": len(certain),
            }
    result["instrument_activity"] = activity
    result["performance"] = {
        "frames": len(frames),
        "dsp_p95_ms": float(np.percentile(times, 95)) if times else None,
        "dsp_total_ms": float(sum(times)),
    }
    return result


def _choose_annotation(manifest_path, recording, annotator, allow_unreviewed):
    available = registry.annotations_for(manifest_path, recording.recording_id)
    usable = [a for a in available if a.usable_as_reference or allow_unreviewed]
    if annotator != "consensus":
        chosen = [a for a in available if a.annotator_id == annotator]
        if not chosen:
            return None, f"no annotation by {annotator}"
        if chosen[0] not in usable:
            return None, f"annotation by {annotator} is {chosen[0].status}, not reviewed"
        return chosen[0], None
    if not usable:
        return None, "no reviewed annotation" if not available else "annotations not reviewed"
    if len(usable) == 1:
        return usable[0], None
    if len(usable) == 2:
        return consensus(*usable), None
    return None, "more than two reviewed annotators; name one with --annotator"


def _content_hash(manifest_path, used):
    digest = hashlib.sha256(Path(manifest_path).read_bytes())
    for recording_id, annotator_id in sorted(used):
        for part in (
            annotator_id.split(".")[1:] if annotator_id.startswith("consensus.") else [annotator_id]
        ):
            path = registry.annotation_path(manifest_path, recording_id, part)
            if path.is_file():
                digest.update(path.read_bytes())
    return digest.hexdigest()


def evaluate_corpus(
    manifest_path,
    split,
    *,
    annotator="consensus",
    detector="candidate",
    config=None,
    final=False,
    allow_unreviewed=False,
):
    if split == "test" and not final:
        raise ValueError(
            "The test split is held out. Pass --final only for a final evaluation; "
            "do not tune against it."
        )
    manifest = registry.load_manifest(manifest_path)
    paths = registry.load_paths(manifest_path)
    results, skipped, used = [], [], []
    for recording in manifest.recordings:
        if recording.split != split:
            continue
        annotation, reason = _choose_annotation(
            manifest_path, recording, annotator, allow_unreviewed
        )
        if annotation and annotation.recording_sha256 != recording.sha256:
            annotation, reason = None, "annotation describes a different version of the audio"
        path = None
        if annotation:
            path, reason = registry.resolve(recording, paths)
        if not path:
            skipped.append({"recording_id": recording.recording_id, "reason": reason})
            continue
        audio, rate = registry.read_audio(path)
        results.append(evaluate_recording(recording, annotation, audio, rate, detector, config))
        used.append((recording.recording_id, annotation.annotator_id))

    def collect(pick):
        return [s for s in (pick(r) for r in results) if s]

    def onset(key):
        return lambda r: r["onsets"].get(key)

    tags = sorted({tag for r in results for tag in r["recording"]["tags"]})
    summary = {
        "onsets": {
            key: aggregate(collect(onset(key))) if collect(onset(key)) else None
            for key in ("25", "50", "50_plus_annotation_uncertainty")
        },
        "beats_observed": aggregate(collect(lambda r: r["beats"].get("observed")))
        if collect(lambda r: r["beats"].get("observed"))
        else None,
        "beats_predicted": aggregate(collect(lambda r: r["beats"].get("predicted")))
        if collect(lambda r: r["beats"].get("predicted"))
        else None,
        "candidates": {
            kind: aggregate(collect(lambda r, kind=kind: r["candidates"][kind].get("score_50ms")))
            if collect(lambda r, kind=kind: r["candidates"][kind].get("score_50ms"))
            else None
            for kind in CLASSES
        },
        "onsets_50ms_by_tag": {
            tag: aggregate(
                [
                    r["onsets"]["50"]
                    for r in results
                    if tag in r["recording"]["tags"] and "50" in r["onsets"]
                ]
            )
            for tag in tags
            if any(tag in r["recording"]["tags"] and "50" in r["onsets"] for r in results)
        },
        "instrument_activity": _activity_summary(results),
    }
    return {
        "data_kind": "real_corpus",
        "real_music_status": "evaluated" if results else "pending",
        "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "corpus_version": manifest.corpus_version,
        "corpus_content_sha256": _content_hash(manifest_path, used),
        "split": split,
        "final_test_evaluation": bool(final and split == "test"),
        "annotator": annotator,
        "unreviewed_annotations_allowed": allow_unreviewed,
        "matching": "mir_eval.util.match_events one-to-one; uncertain references abstained",
        "evaluated": len(results),
        "skipped": skipped,
        "aggregate": summary,
        "recordings": results,
    }


def _activity_summary(results):
    summary = {}
    for result in results:
        for label, entry in result["instrument_activity"].items():
            row = summary.setdefault(
                label,
                {"recordings": 0, "reference_active_seconds": 0.0, "scored_seconds": 0.0}
                | {"tp_seconds": 0.0, "fp_seconds": 0.0, "fn_seconds": 0.0, "scored_recordings": 0},
            )
            row["recordings"] += 1
            row["reference_active_seconds"] += entry["reference_active_seconds"]
            row["scored_seconds"] += entry["scored_seconds"]
            if entry["status"] == "scored":
                row["scored_recordings"] += 1
                for key in ("tp_seconds", "fp_seconds", "fn_seconds"):
                    row[key] += entry[key]
    for row in summary.values():
        tp, fp, fn = row["tp_seconds"], row["fp_seconds"], row["fn_seconds"]
        if row["scored_recordings"]:
            row.update(
                status="scored",
                precision=tp / (tp + fp) if tp + fp else 0,
                recall=tp / (tp + fn) if tp + fn else 0,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
            )
        else:
            row.update(status="no_detector_output")
    return summary


def render_markdown(report):
    def row(name, score):
        if not score:
            return f"| {name} | — | — | — | — | — | — | — |"
        error = score.get("median_absolute_error_ms")
        return (
            f"| {name} | {score['tp']} | {score['fp']} | {score['fn']} | "
            f"{score['precision']:.3f} | {score['recall']:.3f} | {score['f1']:.3f} | "
            f"{'—' if error is None else f'{error:.1f} ms'} |"
        )

    lines = [
        f"# Corpus evaluation — {report['split']} split",
        "",
        f"- Status: **{report['real_music_status']}**"
        + (" (FINAL held-out test evaluation)" if report["final_test_evaluation"] else ""),
        f"- Corpus version {report['corpus_version']}, content hash "
        f"`{report['corpus_content_sha256'][:16]}`; detector `{report['detector']}` at commit "
        f"`{report['code_commit'][:7]}`",
        f"- Recordings evaluated: {report['evaluated']}; skipped: {len(report['skipped'])}",
        "",
    ]
    if report["skipped"]:
        lines += ["## Skipped", ""]
        lines += [f"- `{s['recording_id']}`: {s['reason']}" for s in report["skipped"]]
        lines.append("")
    if not report["evaluated"]:
        lines.append("No recording could be evaluated. Real-music evaluation is pending.")
        return "\n".join(lines) + "\n"
    summary = report["aggregate"]
    lines += [
        "## Event classes (aggregate over evaluated recordings)",
        "",
        "| Class | TP | FP | FN | Precision | Recall | F1 | Median abs. error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        row("Full-mix onset, ±25 ms", summary["onsets"]["25"]),
        row("Full-mix onset, ±50 ms", summary["onsets"]["50"]),
        row("Beats, observed, ±70 ms", summary["beats_observed"]),
        row("Beats, predicted, ±70 ms", summary["beats_predicted"]),
    ]
    lines += [row(f"`{kind}` vs annotated attacks", s) for kind, s in summary["candidates"].items()]
    lines += ["", "A dash means the layer was not annotated in any evaluated recording.", ""]
    if summary["onsets_50ms_by_tag"]:
        lines += [
            "## Full-mix onsets at ±50 ms by condition tag",
            "",
            "| Tag | TP | FP | FN | Precision | Recall | F1 | Median abs. error |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        lines += [row(tag, score) for tag, score in summary["onsets_50ms_by_tag"].items()]
        lines.append("")
    if summary["instrument_activity"]:
        lines += [
            "## Instrument activity",
            "",
            "| Instrument | Recordings | Reference active | Detector | F1 |",
            "|---|---:|---:|---|---:|",
        ]
        for label, entry in sorted(summary["instrument_activity"].items()):
            score = f"{entry['f1']:.3f}" if entry["status"] == "scored" else "—"
            lines.append(
                f"| {label} | {entry['recordings']} | {entry['reference_active_seconds']:.1f} s | "
                f"{entry['status'].replace('_', ' ')} | {score} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


__all__ = ["evaluate_corpus", "evaluate_recording", "event_layer", "render_markdown"]
