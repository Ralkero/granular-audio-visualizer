"""Express a synthetic fixture in the real-corpus annotation schema.

This exists to exercise the whole corpus path (registration, annotation files, evaluation)
without any real audio. The synthetic attack classes are proxies; mapping them onto
instrument labels here does not make them instruments.
"""

from gmv.corpus.schema import (
    Activity,
    AnnotationFile,
    Coverage,
    InstrumentEvent,
    MixEvent,
    TimeRange,
)
from gmv.corpus.stems import activity_intervals

PROXY_LABELS = {
    "bass_transient_candidate": "bass",
    "snare_like_candidate": "snare",
    "high_frequency_percussion_candidate": "cymbals",
}


def fixture_annotation(fixture, recording, annotator_id="synthetic-truth"):
    events = [MixEvent(time=t, kind="onset") for t in fixture.onsets]
    events += [MixEvent(time=t, kind="beat") for t in fixture.beats]
    events += [
        MixEvent(time=start, kind="tempo_change", value=bpm)
        for start, _, bpm in fixture.tempo_intervals
    ]
    layers = ["onset", "beat", "tempo"] + [f"hits.{label}" for label in PROXY_LABELS.values()]
    activity = []
    for label, stem in fixture.stems.items():
        layers.append(f"activity.{label}")
        activity += [
            Activity(start=s, end=e, label=label) for s, e in activity_intervals(stem, fixture.rate)
        ]
    return AnnotationFile(
        recording_id=recording.recording_id,
        recording_sha256=recording.sha256,
        annotator_id=annotator_id,
        tool="synthetic_fixture",
        provenance="Exact synthesis positions of an original procedural CC0 fixture",
        status="automatic",
        reference_kind="dataset_annotation",
        uncertainty_seconds=0.001,
        coverage=Coverage(
            ranges=(TimeRange(start=0, end=recording.duration_s),), layers=tuple(layers)
        ),
        events=tuple(sorted(events, key=lambda e: (e.time, e.kind))),
        instrument_events=tuple(
            sorted(
                (
                    InstrumentEvent(time=round(t, 6), label=PROXY_LABELS[kind])
                    for t, kind in fixture.attacks
                    if kind in PROXY_LABELS
                ),
                key=lambda e: (e.time, e.label),
            )
        ),
        activity=tuple(sorted(activity, key=lambda a: (a.start, a.label))),
    )
