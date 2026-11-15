"""Draft instrument-activity intervals from the separate stems of a multitrack recording.

A stem is called active where its short-term RMS is above both an absolute floor and a level
relative to that stem's own loudest passage. This is a measurement of stem energy, not a
musical judgement: bleed, reverb tails and noise can count as activity. The result is
therefore its own reference tier, ``stem_derived``: usable as reference because it is measured
from the dataset's isolated, time-aligned stems, but always reported separately from
annotations supplied by a dataset or made by a person. Stems flagged as containing bleed
are marked uncertain and are excluded from scoring.

Only stems that ship with a dataset or a multitrack session qualify. The output of a
source-separation model is a model prediction and must never be passed through here as
reference.
"""

import numpy as np

from .intervals import from_mask, merge
from .schema import Activity, AnnotationFile, Coverage, TimeRange

TOOL = "stem_energy_v1"
HOP_S, WINDOW_S = 0.01, 0.05
FLOOR_DBFS, RELATIVE_DB = -60.0, -40.0
JOIN_GAP_S, MIN_LENGTH_S = 0.2, 0.05


def activity_intervals(audio, rate):
    power = np.mean(np.asarray(audio, dtype=np.float64) ** 2, axis=1)
    hop, window = max(1, round(rate * HOP_S)), max(1, round(rate * WINDOW_S))
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    starts = np.arange(0, max(1, len(power) - window + 1), hop)
    rms = np.sqrt(
        (cumulative[np.minimum(starts + window, len(power))] - cumulative[starts]) / window
    )
    level = 20 * np.log10(np.maximum(rms, 1e-12))
    threshold = max(FLOOR_DBFS, float(level.max()) + RELATIVE_DB)
    spans = [
        (start + WINDOW_S / 2 - HOP_S / 2, end + WINDOW_S / 2 - HOP_S / 2)
        for start, end in from_mask(level > threshold, hop / rate)
    ]
    duration = len(power) / rate
    return [
        (max(0.0, s), min(duration, e))
        for s, e in merge(spans, JOIN_GAP_S)
        if e - s >= MIN_LENGTH_S
    ]


def reference(recording, stems, uncertain=()):
    """``stems`` maps label -> (audio, rate). Labels in ``uncertain`` (e.g. stems with bleed)
    are written as uncertain. Returns a ``stem_derived`` AnnotationFile."""
    activity = [
        Activity(
            start=start,
            end=end,
            label=label,
            certainty="uncertain" if label in uncertain else "certain",
        )
        for label, (audio, rate) in sorted(stems.items())
        for start, end in activity_intervals(audio, rate)
    ]
    return AnnotationFile(
        recording_id=recording.recording_id,
        recording_sha256=recording.sha256,
        annotator_id="stem-energy",
        tool=TOOL,
        provenance=(
            f"Stem RMS above max({FLOOR_DBFS:g} dBFS, stem peak {RELATIVE_DB:g} dB); "
            f"{WINDOW_S * 1000:g} ms window; gaps under {JOIN_GAP_S:g} s joined"
        ),
        status="automatic",
        reference_kind="stem_derived",
        uncertainty_seconds=WINDOW_S,
        coverage=Coverage(
            ranges=(TimeRange(start=0, end=recording.duration_s),),
            layers=tuple(f"activity.{label}" for label in sorted(stems)),
        ),
        activity=tuple(sorted(activity, key=lambda a: (a.start, a.label))),
    )


draft = reference  # Former name, kept for callers of the Milestone B API.
