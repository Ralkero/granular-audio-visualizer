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

from gmv.activity import intervals as activity_intervals
from gmv.corpus import registry
from gmv.corpus.compare import combine, consensus
from gmv.corpus.intervals import merge, total
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
    "high_frequency_percussion_candidate": ("cymbals", "hi_hat"),
}
OBSERVED_BEATS = ("beat_observed", "beat_candidate")
# Activity layers that must be sounding for a candidate to be what its name suggests.
# If every available one is silent around the event, the event is refuted. An active
# layer does not confirm the event; this is a lower bound on false positives only.
CANDIDATE_ACTIVITY = {
    "bass_transient_candidate": ("drums", "kick", "bass"),
    "snare_like_candidate": ("drums", "snare"),
    "high_frequency_percussion_candidate": ("drums", "cymbals", "hi_hat"),
}
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

    def kind_of(layer):
        return annotation.source_of(layer).reference_kind

    result = {
        "recording": {
            key: getattr(recording, key)
            for key in ("recording_id", "sha256", "split", "group_id", "source_kind", "duration_s")
            + ("dataset", "dataset_track_id")
        }
        | {"tags": list(recording.tags)},
        "annotation": {
            "annotator_id": annotation.annotator_id,
            "reference_kind": annotation.reference_kind,
            "reference_sources": {
                layer: annotation.source_of(layer).model_dump()
                for layer in annotation.coverage.layers
            },
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
            "reference_kind": kind_of("onset"),
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
            "reference_kind": kind_of("beat"),
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
    onset_times = np.sort([e.timestamp for e in predicted("onset")])

    def found_as_onset(times):
        if not len(onset_times) or not times:
            return 0
        index = np.clip(np.searchsorted(onset_times, times), 0, len(onset_times) - 1)
        nearest = np.minimum(
            np.abs(onset_times[index] - times),
            np.abs(onset_times[np.maximum(index - 1, 0)] - times),
        )
        return int((nearest <= 0.05).sum())

    result["instrument_events"] = {
        label: {
            "status": "no_detector_output",
            "reference_kind": kind_of(f"hits.{label}"),
            "reference_count": len(certain),
            "abstained_references": len(uncertain),
            # Every annotated attack is an acoustic onset in the mix, so recall of the
            # full-mix onset detector on these attacks is valid. Precision is not: other
            # instruments' onsets are real but unannotated here.
            "detected_as_full_mix_onset_50ms": found_as_onset(certain),
            "full_mix_onset_recall_50ms": found_as_onset(certain) / len(certain)
            if certain
            else None,
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

    if activity_predictions is None and any(getattr(f, "activity_method", None) for f in frames):
        stamps = [f.timestamp for f in frames]
        step = stamps[1] - stamps[0] if len(stamps) > 1 else 0.02
        labels = [name for name, _ in next(f for f in frames if f.activity_method).instrument_scores]
        activity_predictions = {
            label: activity_intervals(stamps, [label in f.instruments_active for f in frames], step)
            for label in labels
        }
    activity, sounding = {}, {}
    for layer in annotation.coverage.layers:
        if not layer.startswith("activity."):
            continue
        label = layer[9:]
        spans = [a for a in annotation.activity if a.label == label]
        certain = [(a.start, a.end) for a in spans if a.certainty == "certain"]
        uncertain = [(a.start, a.end) for a in spans if a.certainty == "uncertain"]
        sounding[label] = merge(certain + uncertain)
        if activity_predictions and label in activity_predictions:
            activity[label] = {
                "status": "scored",
                "reference_kind": kind_of(layer),
                **interval_score(certain, activity_predictions[label], ranges, uncertain),
            }
        else:
            reference = interval_score(certain, [], ranges, uncertain)
            activity[label] = {
                "status": "no_detector_output",
                "reference_kind": kind_of(layer),
                "reference_active_seconds": reference["reference_active_seconds"],
                "scored_seconds": reference["scored_seconds"],
                "abstained_seconds": reference["abstained_seconds"],
                "reference_intervals": len(certain),
            }
    result["instrument_activity"] = activity
    refuted = {}
    for kind in CLASSES:
        layers = [label for label in CANDIDATE_ACTIVITY[kind] if label in sounding]
        if not layers:
            refuted[kind] = {**NOT_ANNOTATED, "needs_any_of": list(CANDIDATE_ACTIVITY[kind])}
            continue
        spans = merge([span for label in layers for span in sounding[label]])
        stamps = [e.timestamp for e in predicted(kind)]
        silent = sum(not any(s - 0.05 <= t <= e + 0.05 for s, e in spans) for t in stamps)
        refuted[kind] = {
            "checked_against": layers,
            "reference_kinds": sorted({kind_of(f"activity.{label}") for label in layers}),
            "events": len(stamps),
            "refuted_events": silent,
            "refuted_fraction": silent / len(stamps) if stamps else None,
            "meaning": "events with none of these instruments sounding within 50 ms; "
            "a lower bound on false positives, not an accuracy",
        }
    result["candidates_vs_activity"] = refuted
    result["performance"] = {
        "frames": len(frames),
        "dsp_p95_ms": float(np.percentile(times, 95)) if times else None,
        "dsp_total_ms": float(sum(times)),
    }
    return result


def _choose_annotation(manifest_path, recording, annotator, allow_unreviewed):
    """Pick the reference for a recording.

    ``auto`` (default) takes, per layer, the best usable tier: human-reviewed, then
    dataset-provided, then stem-derived. Pseudo-labels, model predictions and drafts are
    never reference; ``allow_unreviewed`` admits them for exploration and is stamped.
    """
    available = registry.annotations_for(manifest_path, recording.recording_id)
    if annotator not in ("auto", "consensus"):
        chosen = [a for a in available if a.annotator_id == annotator]
        if not chosen:
            return None, f"no annotation by {annotator}"
        if not (chosen[0].usable_as_reference or allow_unreviewed):
            return None, (
                f"annotation by {annotator} is {chosen[0].reference_kind} with status "
                f"{chosen[0].status}, not usable as reference"
            )
        return chosen[0], None
    usable = [a for a in available if a.usable_as_reference]
    humans = [a for a in usable if a.reference_kind == "human_annotation"]
    if len(humans) > 2:
        return None, "more than two reviewed annotators; name one with --annotator"
    if len(humans) == 2:
        usable = [a for a in usable if a not in humans] + [consensus(*humans)]
    if not usable:
        if allow_unreviewed and len(available) == 1:
            return available[0], None
        return None, "no usable reference annotation" if available else "no annotation"
    return (usable[0] if len(usable) == 1 else combine(usable)), None


def _content_hash(manifest_path, used):
    """Manifest plus every annotation file of each evaluated recording."""
    digest = hashlib.sha256(Path(manifest_path).read_bytes())
    for recording_id, _ in sorted(used):
        folder = Path(manifest_path).parent / "annotations" / recording_id
        for path in sorted(folder.glob("*.json")):
            digest.update(path.read_bytes())
    return digest.hexdigest()


def evaluate_corpus(
    manifest_path,
    split,
    *,
    annotator="auto",
    detector="candidate",
    config=None,
    final=False,
    allow_unreviewed=False,
    dataset=None,
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
        if recording.split != split or (dataset and recording.dataset != dataset):
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
        "candidates_vs_activity": _refuted_summary(results),
        "onset_recall_on_instrument_attacks": _attack_recall_summary(results),
        "reference_kinds_by_layer": _reference_summary(results),
        "evaluated_seconds": sum(r["annotation"]["covered_seconds"] for r in results),
    }
    return {
        "data_kind": "real_corpus",
        "real_music_status": "evaluated" if results else "pending",
        "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "corpus_version": manifest.corpus_version,
        "corpus_content_sha256": _content_hash(manifest_path, used),
        "split": split,
        "dataset_filter": dataset,
        "final_test_evaluation": bool(final and split == "test"),
        "annotator": annotator,
        "unreviewed_annotations_allowed": allow_unreviewed,
        "matching": "mir_eval.util.match_events one-to-one; uncertain references abstained",
        "evaluated": len(results),
        "skipped": skipped,
        "aggregate": summary,
        "recordings": results,
    }


def _reference_summary(results):
    summary = {}
    for result in results:
        for layer, source in result["annotation"]["reference_sources"].items():
            kinds = summary.setdefault(layer, {})
            kinds[source["reference_kind"]] = kinds.get(source["reference_kind"], 0) + 1
    return dict(sorted(summary.items()))


def _attack_recall_summary(results):
    summary = {}
    for result in results:
        for label, entry in result["instrument_events"].items():
            row = summary.setdefault(
                label, {"recordings": 0, "reference_count": 0, "detected": 0, "kinds": set()}
            )
            row["recordings"] += 1
            row["reference_count"] += entry["reference_count"]
            row["detected"] += entry["detected_as_full_mix_onset_50ms"]
            row["kinds"].add(entry["reference_kind"])
    return {
        label: {
            "recordings": row["recordings"],
            "reference_count": row["reference_count"],
            "detected_as_full_mix_onset_50ms": row["detected"],
            "recall": row["detected"] / row["reference_count"] if row["reference_count"] else None,
            "reference_kinds": sorted(row["kinds"]),
        }
        for label, row in sorted(summary.items())
    }


def _refuted_summary(results):
    summary = {}
    for kind in CLASSES:
        rows = [r["candidates_vs_activity"][kind] for r in results]
        rows = [row for row in rows if "events" in row]
        if not rows:
            summary[kind] = None
            continue
        events, silent = sum(r["events"] for r in rows), sum(r["refuted_events"] for r in rows)
        summary[kind] = {
            "recordings": len(rows),
            "events": events,
            "refuted_events": silent,
            "refuted_fraction": silent / events if events else None,
            "reference_kinds": sorted({k for r in rows for k in r["reference_kinds"]}),
        }
    return summary


def _activity_summary(results):
    summary = {}
    for result in results:
        for label, entry in result["instrument_activity"].items():
            row = summary.setdefault(
                label,
                {
                    "recordings": 0,
                    "scored_recordings": 0,
                    "reference_kinds": set(),
                    "reference_active_seconds": 0.0,
                    "scored_seconds": 0.0,
                    "tp_seconds": 0.0,
                    "fp_seconds": 0.0,
                    "fn_seconds": 0.0,
                    "inactive_seconds": 0.0,
                },
            )
            row["recordings"] += 1
            row["reference_kinds"].add(entry["reference_kind"])
            row["reference_active_seconds"] += entry["reference_active_seconds"]
            row["scored_seconds"] += entry["scored_seconds"]
            if entry["status"] == "scored":
                row["scored_recordings"] += 1
                for key in ("tp_seconds", "fp_seconds", "fn_seconds"):
                    row[key] += entry[key]
                row["inactive_seconds"] += entry["scored_seconds"] - entry["reference_active_seconds"]
    for row in summary.values():
        row["reference_kinds"] = sorted(row["reference_kinds"])
        tp, fp, fn = row["tp_seconds"], row["fp_seconds"], row["fn_seconds"]
        if row["scored_recordings"]:
            row.update(
                status="scored",
                precision=tp / (tp + fp) if tp + fp else 0,
                recall=tp / (tp + fn) if tp + fn else 0,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
                false_positive_rate=fp / row["inactive_seconds"] if row["inactive_seconds"] else None,
                # What an estimator that always says "active" would score on the same audio.
                always_active_f1=2 * (tp + fn) / (2 * (tp + fn) + row["inactive_seconds"])
                if tp + fn
                else 0,
            )
        else:
            row.update(status="no_detector_output")
    return dict(sorted(summary.items()))


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
        f"# Corpus evaluation — {report['split']} split"
        + (f", {report['dataset_filter']}" if report.get("dataset_filter") else ""),
        "",
        f"- Status: **{report['real_music_status']}**"
        + (" (FINAL held-out test evaluation)" if report["final_test_evaluation"] else ""),
        f"- Corpus version {report['corpus_version']}, content hash "
        f"`{report['corpus_content_sha256'][:16]}`; detector `{report['detector']}` at commit "
        f"`{report['code_commit'][:7]}`",
        f"- Recordings evaluated: {report['evaluated']}; skipped: {len(report['skipped'])}",
        f"- Annotated audio evaluated: {report['aggregate']['evaluated_seconds'] / 60:.1f} min",
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
            "| Instrument | Recordings | Reference | Reference active | Detector | Precision | "
            "Recall | F1 | Always-active F1 | False-positive rate |",
            "|---|---:|---|---:|---|---:|---:|---:|---:|---:|",
        ]
        for label, entry in sorted(summary["instrument_activity"].items()):
            if entry["status"] == "scored":
                rate = entry["false_positive_rate"]
                score = (
                    f"{entry['precision']:.3f} | {entry['recall']:.3f} | {entry['f1']:.3f} | "
                    f"{entry['always_active_f1']:.3f} | {'—' if rate is None else f'{rate:.3f}'}"
                )
            else:
                score = "— | — | — | — | —"
            lines.append(
                f"| {label} | {entry['recordings']} | {', '.join(entry['reference_kinds'])} | "
                f"{entry['reference_active_seconds']:.1f} s | "
                f"{entry['status'].replace('_', ' ')} | {score} |"
            )
        lines.append("")
    if summary["onset_recall_on_instrument_attacks"]:
        lines += [
            "## Full-mix onset detector: recall on annotated instrument attacks (±50 ms)",
            "",
            "Recall only. Precision cannot be measured where other instruments' onsets are "
            "not annotated.",
            "",
            "| Instrument attacks | Recordings | Annotated | Detected | Recall | Reference |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for label, entry in summary["onset_recall_on_instrument_attacks"].items():
            recall = entry["recall"]
            lines.append(
                f"| {label} | {entry['recordings']} | {entry['reference_count']} | "
                f"{entry['detected_as_full_mix_onset_50ms']} | "
                f"{'—' if recall is None else f'{recall:.3f}'} | "
                f"{', '.join(entry['reference_kinds'])} |"
            )
        lines.append("")
    checked = {k: v for k, v in summary["candidates_vs_activity"].items() if v}
    if checked:
        lines += [
            "## Band-ratio candidates refuted by instrument activity",
            "",
            "Events fired while none of the instruments the candidate's name suggests was "
            "sounding. This is a lower bound on false positives, not an accuracy.",
            "",
            "| Candidate | Events | Refuted | Fraction | Reference |",
            "|---|---:|---:|---:|---|",
        ]
        for kind, entry in checked.items():
            fraction = entry["refuted_fraction"]
            lines.append(
                f"| `{kind}` | {entry['events']} | {entry['refuted_events']} | "
                f"{'—' if fraction is None else f'{fraction:.3f}'} | "
                f"{', '.join(entry['reference_kinds'])} |"
            )
        lines.append("")
    lines += ["## Reference used per layer (recordings)", ""]
    lines += [
        f"- `{layer}`: " + ", ".join(f"{kind} ×{count}" for kind, count in sorted(kinds.items()))
        for layer, kinds in summary["reference_kinds_by_layer"].items()
    ]
    lines.append("")
    return "\n".join(lines) + "\n"


__all__ = ["evaluate_corpus", "evaluate_recording", "event_layer", "render_markdown"]
