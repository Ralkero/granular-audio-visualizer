"""Corpus manifest and annotation schema. Times are seconds from the start of the recording.

Two kinds of ground truth are kept apart on purpose:

* full-mix ``events`` (an onset, beat, downbeat, tempo change, transition or section change
  heard in the mix), and
* instrument-specific truth: ``activity`` intervals (which may overlap freely) and
  ``instrument_events`` (point attacks such as a kick or snare hit).

A full-mix onset is not an instrument attack; a kick, a cymbal crash and a vocal entrance
at the same instant are three instrument events and one mix onset.
"""

from typing import Literal

from pydantic import Field, model_validator

from ..models import FrozenModel

CORPUS_SCHEMA_VERSION = "1.0"
ANNOTATION_SCHEMA_VERSION = "1.0"

Split = Literal["unassigned", "train", "validation", "test"]
SourceKind = Literal["owned_local", "open_dataset", "original", "public_domain", "licensed"]
Certainty = Literal["certain", "uncertain"]
Status = Literal["machine_draft", "in_progress", "reviewed"]
EventKind = Literal["onset", "beat", "downbeat", "tempo_change", "transition", "section_change"]
Label = Literal[
    "drums",
    "kick",
    "snare",
    "cymbals",
    "bass",
    "guitar",
    "rhythm_guitar",
    "lead_guitar",
    "vocals",
    "vocals_clean",
    "vocals_harsh",
    "other_harmonic",
    "ambient",
    "other",
    "unknown",
]
EVENT_KINDS = EventKind.__args__
LABELS = Label.__args__
# A parent is active whenever any child is; used only for explicit roll-up reporting.
HIERARCHY = {
    "drums": ("kick", "snare", "cymbals"),
    "guitar": ("rhythm_guitar", "lead_guitar"),
    "vocals": ("vocals_clean", "vocals_harsh"),
}
EVENT_LAYERS = ("onset", "beat", "downbeat", "tempo", "sections")
SHA256 = r"^[0-9a-f]{64}$"
IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"


def layer_names():
    """Every annotation layer a file may declare as covered."""
    return (
        EVENT_LAYERS
        + tuple(f"activity.{label}" for label in LABELS)
        + tuple(f"hits.{label}" for label in LABELS)
    )


class Recording(FrozenModel):
    recording_id: str = Field(pattern=r"^r-[0-9a-f]{12}$")
    sha256: str = Field(pattern=SHA256)
    duration_s: float = Field(gt=0)
    sample_rate: int = Field(ge=8000, le=384000)
    channels: int = Field(ge=1, le=32)
    source_kind: SourceKind
    license: str = Field(min_length=1)
    permission_basis: str = Field(min_length=1)
    # Leakage unit: recordings that share material (album, session, dataset song, alternate
    # masters) share a group and therefore a split.
    group_id: str = Field(pattern=IDENTIFIER)
    split: Split = "unassigned"
    tags: tuple[str, ...] = ()
    stems: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self):
        if self.recording_id != "r-" + self.sha256[:12]:
            raise ValueError("recording_id must be 'r-' plus the first 12 hex digits of sha256")
        for label, digest in self.stems.items():
            if label not in LABELS or len(digest) != 64:
                raise ValueError(f"Invalid stem entry: {label}")
        return self


class CorpusManifest(FrozenModel):
    corpus_schema_version: Literal["1.0"] = CORPUS_SCHEMA_VERSION
    corpus_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    recordings: tuple[Recording, ...] = ()

    @model_validator(mode="after")
    def _check(self):
        ids = [r.recording_id for r in self.recordings]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate recording in manifest")
        groups = {}
        for recording in self.recordings:
            groups.setdefault(recording.group_id, set()).add(recording.split)
        leaking = sorted(group for group, splits in groups.items() if len(splits) > 1)
        if leaking:
            raise ValueError(f"Groups assigned to more than one split (leakage): {leaking}")
        return self

    def get(self, recording_id):
        for recording in self.recordings:
            if recording.recording_id == recording_id:
                return recording
        raise KeyError(f"Unknown recording: {recording_id}")


class TimeRange(FrozenModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)

    @model_validator(mode="after")
    def _check(self):
        if self.end <= self.start:
            raise ValueError("Range end must be after its start")
        return self


class MixEvent(FrozenModel):
    time: float = Field(ge=0)
    kind: EventKind
    end: float | None = None
    value: float | str | None = None  # bpm for tempo_change; free label for sections
    certainty: Certainty = "certain"
    intensity: float | None = Field(default=None, ge=0, le=1)


class InstrumentEvent(FrozenModel):
    time: float = Field(ge=0)
    label: Label
    certainty: Certainty = "certain"
    intensity: float | None = Field(default=None, ge=0, le=1)


class Activity(TimeRange):
    label: Label
    certainty: Certainty = "certain"
    intensity: float | None = Field(default=None, ge=0, le=1)


class Coverage(FrozenModel):
    """What was actually annotated. Anything outside is unknown, never 'absent'."""

    ranges: tuple[TimeRange, ...]
    layers: tuple[str, ...]

    @model_validator(mode="after")
    def _check(self):
        unknown = sorted(set(self.layers) - set(layer_names()))
        if unknown:
            raise ValueError(f"Unknown annotation layers: {unknown}")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("Duplicate annotation layer")
        return self


class AnnotationFile(FrozenModel):
    annotation_schema_version: Literal["1.0"] = ANNOTATION_SCHEMA_VERSION
    recording_id: str = Field(pattern=r"^r-[0-9a-f]{12}$")
    recording_sha256: str = Field(pattern=SHA256)
    annotator_id: str = Field(pattern=IDENTIFIER)
    tool: str = Field(min_length=1)
    tool_version: str = ""
    provenance: str = Field(min_length=1)
    status: Status = "in_progress"
    uncertainty_seconds: float = Field(default=0.02, gt=0, le=0.5)
    coverage: Coverage
    events: tuple[MixEvent, ...] = ()
    instrument_events: tuple[InstrumentEvent, ...] = ()
    activity: tuple[Activity, ...] = ()
    notes: str = ""

    @model_validator(mode="after")
    def _check(self):
        layers = set(self.coverage.layers)
        used = {_event_layer(e.kind) for e in self.events}
        used |= {f"hits.{e.label}" for e in self.instrument_events}
        used |= {f"activity.{a.label}" for a in self.activity}
        if used - layers:
            raise ValueError(f"Labels present in undeclared layers: {sorted(used - layers)}")
        return self

    @property
    def usable_as_reference(self):
        return self.status == "reviewed"

    def covers(self, layer):
        return layer in self.coverage.layers


def _event_layer(kind):
    return {
        "tempo_change": "tempo",
        "transition": "sections",
        "section_change": "sections",
    }.get(kind, kind)


event_layer = _event_layer
