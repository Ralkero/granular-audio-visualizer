"""MDB Drums adapter: real full mixes with annotated drum hits and beats.

Expected layout (a clone of the MDB Drums repository)::

    <root>/MDB Drums/audio/full_mix/<Track>_MIX.wav
    <root>/MDB Drums/audio/drum_only/<Track>_Drum.wav
    <root>/MDB Drums/annotations/class/<Track>_class.txt      time <TAB> class
    <root>/MDB Drums/annotations/beats/<Track>_MIX.beats      time [beat number]

``<root>`` may also be the ``MDB Drums`` folder itself. The class annotations are the
dataset's own ground truth for individual drum attacks (``dataset_annotation``). The
drum-only stem additionally gives stem-derived drum activity. Full-mix onsets of the other
instruments are not annotated, so full-mix onset precision cannot be measured here.
"""

from ..schema import AnnotationFile, Coverage, InstrumentEvent, MixEvent, TimeRange
from . import Adapter, Track, artist_key

# Six-class MDB Drums labels -> project labels. Every hit also counts as a "drums" hit.
CLASSES = {"KD": "kick", "SD": "snare", "HH": "hi_hat", "TT": "toms", "CY": "cymbals", "OT": None}
HIT_LAYERS = ("drums", "kick", "snare", "hi_hat", "toms", "cymbals")


def _rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.replace(",", " ").split()
        if not parts:
            continue
        try:
            rows.append((float(parts[0]), parts[1] if len(parts) > 1 else ""))
        except ValueError:
            continue  # header line
    return rows


def _annotation(class_file, beat_file):
    def build(recording):
        hits, unknown = [], set()
        for time, label in _rows(class_file):
            if time > recording.duration_s:
                continue
            if label not in CLASSES:
                unknown.add(label)
            hits.append(InstrumentEvent(time=time, label="drums"))
            if CLASSES.get(label):
                hits.append(InstrumentEvent(time=time, label=CLASSES[label]))
        layers = [f"hits.{label}" for label in HIT_LAYERS]
        events = []
        if beat_file and beat_file.is_file():
            beats = [(t, n) for t, n in _rows(beat_file) if t <= recording.duration_s]
            events = [MixEvent(time=t, kind="beat") for t, _ in beats]
            layers.append("beat")
            numbered = [n for _, n in beats if n]
            if numbered and len(numbered) == len(beats):
                events += [MixEvent(time=t, kind="downbeat") for t, n in beats if float(n) == 1]
                layers.append("downbeat")
        return AnnotationFile(
            recording_id=recording.recording_id,
            recording_sha256=recording.sha256,
            annotator_id="mdbdrums",
            tool="mdbdrums_class_annotations",
            provenance="MDB Drums class and beat annotations as distributed with the dataset"
            + (f"; unrecognised classes kept as drums only: {sorted(unknown)}" if unknown else ""),
            status="automatic",
            reference_kind="dataset_annotation",
            uncertainty_seconds=0.02,
            coverage=Coverage(
                ranges=(TimeRange(start=0, end=recording.duration_s),), layers=tuple(layers)
            ),
            events=tuple(sorted(events, key=lambda e: (e.time, e.kind))),
            instrument_events=tuple(sorted(set(hits), key=lambda e: (e.time, e.label))),
        )

    return build


def discover(root):
    base = root / "MDB Drums" if (root / "MDB Drums").is_dir() else root
    tracks = []
    for mixture in sorted((base / "audio" / "full_mix").glob("*_MIX.wav")):
        name = mixture.name.removesuffix("_MIX.wav")
        class_file = base / "annotations" / "class" / f"{name}_class.txt"
        if not class_file.is_file():
            continue
        drum = base / "audio" / "drum_only" / f"{name}_Drum.wav"
        tracks.append(
            Track(
                track_id=name,
                group_key=artist_key(name.split("_")[0]),
                official_split=None,
                mixture=mixture,
                stems={"drums": [drum]} if drum.is_file() else {},
                annotations=[
                    _annotation(class_file, base / "annotations" / "beats" / f"{name}_MIX.beats")
                ],
            )
        )
    return tracks


ADAPTER = Adapter(
    name="mdbdrums",
    title="MDB Drums",
    audio_license="CC BY-NC-SA 4.0 (non-commercial; derived from MedleyDB)",
    access="Public git repository, about 2 GB: https://github.com/CarlSouthall/MDBDrums",
    homepage="https://github.com/CarlSouthall/MDBDrums",
    reference="23 real full mixes with annotated kick, snare, hi-hat, tom and cymbal attacks, "
    "beats, and a drum-only stem. No annotations for other instruments' onsets.",
    discover=discover,
)
