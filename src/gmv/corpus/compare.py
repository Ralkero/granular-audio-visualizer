"""Agreement between two annotations of one recording, and an explicit consensus.

Disagreements are listed, never silently merged. The consensus keeps only what both
annotators marked: matched point events (at the mean time) and the intersection of activity
intervals, restricted to layers and time ranges both covered. It is ``reviewed`` only when
both inputs are.
"""

from .intervals import intersect, merge, total
from .schema import (
    Activity,
    AnnotationFile,
    Coverage,
    InstrumentEvent,
    MixEvent,
    TimeRange,
    event_layer,
)

POINT_KINDS = {"onset": "onset", "beat": "beat", "downbeat": "downbeat"}


def match_points(a, b, tolerance):
    """Greedy one-to-one matching of two sorted time lists within ``tolerance``."""
    a, b, pairs, i, j = sorted(a), sorted(b), [], 0, 0
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= tolerance:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return a, b, pairs


def _points(annotation, layer):
    if layer.startswith("hits."):
        return [e.time for e in annotation.instrument_events if e.label == layer[5:]]
    return [e.time for e in annotation.events if event_layer(e.kind) == layer]


def _spans(annotation, layer):
    return [(a.start, a.end) for a in annotation.activity if a.label == layer[9:]]


def _within(times, ranges):
    return [t for t in times if any(s <= t < e for s, e in ranges)]


def compare(a, b, tolerance=None):
    if a.recording_id != b.recording_id or a.recording_sha256 != b.recording_sha256:
        raise ValueError("Annotations describe different recordings")
    tolerance = tolerance or max(a.uncertainty_seconds, b.uncertainty_seconds, 0.05)
    ranges = intersect(
        [(r.start, r.end) for r in a.coverage.ranges], [(r.start, r.end) for r in b.coverage.ranges]
    )
    shared = sorted(set(a.coverage.layers) & set(b.coverage.layers))
    layers = {}
    for layer in shared:
        if layer.startswith("activity."):
            first, second = intersect(_spans(a, layer), ranges), intersect(_spans(b, layer), ranges)
            both = intersect(first, second)
            union = total(merge(first + second))
            layers[layer] = {
                "type": "interval",
                "a_seconds": total(first),
                "b_seconds": total(second),
                "agreed_seconds": total(both),
                "iou": total(both) / union if union else None,
                "both_empty": not first and not second,
            }
        elif layer in ("tempo", "sections"):
            layers[layer] = {
                "type": "descriptive",
                "a_count": len(_within(_points(a, layer), ranges)),
                "b_count": len(_within(_points(b, layer), ranges)),
            }
        else:
            first, second, pairs = match_points(
                _within(_points(a, layer), ranges), _within(_points(b, layer), ranges), tolerance
            )
            used_a, used_b = {i for i, _ in pairs}, {j for _, j in pairs}
            agreed = len(pairs)
            layers[layer] = {
                "type": "point",
                "a_count": len(first),
                "b_count": len(second),
                "agreed": agreed,
                "f1": 2 * agreed / (len(first) + len(second)) if first or second else None,
                "only_a": [t for i, t in enumerate(first) if i not in used_a],
                "only_b": [t for j, t in enumerate(second) if j not in used_b],
                "mean_absolute_offset_ms": (
                    1000 * sum(abs(first[i] - second[j]) for i, j in pairs) / agreed
                    if agreed
                    else None
                ),
            }
    return {
        "recording_id": a.recording_id,
        "annotators": [a.annotator_id, b.annotator_id],
        "tolerance_seconds": tolerance,
        "shared_ranges": ranges,
        "layers_only_a": sorted(set(a.coverage.layers) - set(b.coverage.layers)),
        "layers_only_b": sorted(set(b.coverage.layers) - set(a.coverage.layers)),
        "layers": layers,
    }


def consensus(a, b, tolerance=None):
    report = compare(a, b, tolerance)
    ranges, tolerance = report["shared_ranges"], report["tolerance_seconds"]
    if not ranges:
        raise ValueError("Annotations share no covered time range")
    events, hits, activity, layers = [], [], [], []
    for layer in report["layers"]:
        if layer in ("tempo", "sections"):
            continue  # No defensible automatic merge; take these from a single annotator.
        layers.append(layer)
        if layer.startswith("activity."):
            activity += [
                Activity(start=s, end=e, label=layer[9:])
                for s, e in intersect(intersect(_spans(a, layer), _spans(b, layer)), ranges)
            ]
            continue
        first, second, pairs = match_points(
            _within(_points(a, layer), ranges), _within(_points(b, layer), ranges), tolerance
        )
        for i, j in pairs:
            time = (first[i] + second[j]) / 2
            if layer.startswith("hits."):
                hits.append(InstrumentEvent(time=time, label=layer[5:]))
            else:
                events.append(MixEvent(time=time, kind=POINT_KINDS[layer]))
    return AnnotationFile(
        recording_id=a.recording_id,
        recording_sha256=a.recording_sha256,
        annotator_id=f"consensus.{a.annotator_id}.{b.annotator_id}"[:64],
        tool="consensus_v1",
        provenance=(
            f"Matched points within {tolerance:g} s and interval intersection of "
            f"{a.annotator_id} and {b.annotator_id}; disagreements excluded, see compare report"
        ),
        status="reviewed" if a.usable_as_reference and b.usable_as_reference else "in_progress",
        reference_kind="human_annotation",
        uncertainty_seconds=min(0.5, tolerance),
        coverage=Coverage(
            ranges=tuple(TimeRange(start=s, end=e) for s, e in ranges), layers=tuple(layers)
        ),
        events=tuple(sorted(events, key=lambda e: (e.time, e.kind))),
        instrument_events=tuple(sorted(hits, key=lambda e: (e.time, e.label))),
        activity=tuple(sorted(activity, key=lambda x: (x.start, x.label))),
    )


def combine(annotations):
    """One reference per layer from several usable annotations of a recording.

    For each layer the best available tier wins: human-reviewed, then dataset-provided, then
    stem-derived. The result records which annotation supplied each layer, so a report can
    always say what kind of reference a number rests on. Unusable annotations (drafts,
    pseudo-labels, model predictions) are ignored.
    """
    from .schema import REFERENCE_PRIORITY, LayerSource

    usable = [a for a in annotations if a.usable_as_reference and a.reference_kind != "combined"]
    if not usable:
        raise ValueError("No usable reference annotation")
    if len({(a.recording_id, a.recording_sha256) for a in usable}) != 1:
        raise ValueError("Annotations describe different recordings")
    ranked = sorted(usable, key=lambda a: REFERENCE_PRIORITY.index(a.reference_kind))
    chosen = {}
    for annotation in ranked:
        for layer in annotation.coverage.layers:
            chosen.setdefault(layer, annotation)
    contributors = list({id(a): a for a in chosen.values()}.values())
    ranges = [(r.start, r.end) for r in contributors[0].coverage.ranges]
    for other in contributors[1:]:
        ranges = intersect(ranges, [(r.start, r.end) for r in other.coverage.ranges])
    if not ranges:
        raise ValueError("Reference annotations share no covered time range")
    events, hits, activity = [], [], []
    for layer, source in chosen.items():
        events += [e for e in source.events if event_layer(e.kind) == layer]
        hits += [e for e in source.instrument_events if f"hits.{e.label}" == layer]
        activity += [a for a in source.activity if f"activity.{a.label}" == layer]
    first = ranked[0]
    return AnnotationFile(
        recording_id=first.recording_id,
        recording_sha256=first.recording_sha256,
        annotator_id="combined",
        tool="combine_v1",
        provenance="Best available reference tier per layer; see layer_sources",
        status="automatic",
        reference_kind="combined",
        layer_sources={
            layer: LayerSource(
                annotator_id=a.annotator_id, reference_kind=a.reference_kind, tool=a.tool
            )
            for layer, a in chosen.items()
        },
        uncertainty_seconds=max(a.uncertainty_seconds for a in contributors),
        coverage=Coverage(
            ranges=tuple(TimeRange(start=s, end=e) for s, e in ranges),
            layers=tuple(sorted(chosen)),
        ),
        events=tuple(sorted(events, key=lambda e: (e.time, e.kind))),
        instrument_events=tuple(sorted(hits, key=lambda e: (e.time, e.label))),
        activity=tuple(sorted(activity, key=lambda x: (x.start, x.label))),
    )
