"""Import research datasets into the corpus without any manual labelling.

An adapter only describes what is on disk: which tracks exist, where the mix and the isolated
stems are, which project label each stem maps to, and any annotations the dataset ships.
``import_dataset`` does the rest: hashing, registration, group and split assignment,
stem-derived activity reference, dataset-provided annotations and validation.

Nothing is downloaded here and no dataset audio is copied into the repository. Each adapter
states the dataset's audio licence and access procedure; accepting those terms and obtaining
the files is the user's step.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import registry, stems
from ..schema import CorpusManifest, Recording

SPLITS = ("train", "validation", "test")


@dataclass
class Track:
    track_id: str  # the dataset's own public track identifier
    group_key: str  # artist or session: everything sharing it stays in one split
    official_split: str | None  # the dataset's own split name, if any
    mixture: Path | None  # None: the mix is the sum of the stems and is rendered locally
    stems: dict[str, list[Path]]  # project label -> one or more aligned stem files
    uncertain: tuple[str, ...] = ()  # labels whose stems contain bleed
    tags: tuple[str, ...] = ()
    annotations: list = field(default_factory=list)  # callables: Recording -> AnnotationFile


@dataclass
class Adapter:
    name: str
    title: str
    audio_license: str
    access: str
    homepage: str
    reference: str  # what reference information the dataset provides
    discover: object  # callable: root Path -> list[Track]
    split_map: dict = field(default_factory=dict)  # official split -> corpus split

    def describe(self):
        return {
            key: getattr(self, key)
            for key in ("name", "title", "audio_license", "access", "homepage", "reference")
        }


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "unknown"


def artist_key(name):
    """Spelling-insensitive artist key ("Music Delta", "MusicDelta" -> "musicdelta")."""
    return re.sub(r"[^a-z0-9]+", "", name.casefold()) or "unknown"


def group_id(dataset, key):
    """Groups are shared across datasets: several of them reuse the same source recordings
    (MUSDB18 and MDB Drums both draw on MedleyDB), so the same artist must land in one
    split everywhere. ``dataset`` is accepted for callers but deliberately not hashed."""
    return f"artist-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:10]}"


def read_sum(paths):
    """Sum aligned stem files sample by sample; returns (audio, rate)."""
    total, rate = None, None
    for path in paths:
        audio, file_rate = registry.read_audio(path)
        if rate not in (None, file_rate):
            raise ValueError(f"Sample rates differ between stems: {path}")
        rate = file_rate
        if total is None:
            total = audio.astype(np.float32, copy=True)
            continue
        if audio.shape[1] != total.shape[1]:
            if 1 not in (audio.shape[1], total.shape[1]):
                raise ValueError(f"Channel layouts differ between stems: {path}")
            audio = np.repeat(audio, total.shape[1], axis=1) if audio.shape[1] == 1 else audio
            total = np.repeat(total, audio.shape[1], axis=1) if total.shape[1] == 1 else total
        if abs(len(audio) - len(total)) > rate // 2:
            raise ValueError(f"Stem lengths differ by more than 0.5 s: {path}")
        length = max(len(audio), len(total))
        padded = np.zeros((length, total.shape[1]), dtype=np.float32)
        padded[: len(total)] += total
        padded[: len(audio)] += audio
        total = padded
    return total, rate


def render_mixture(manifest_path, adapter, track):
    """Write the sum of all stems as 24-bit FLAC under the ignored local corpus directory."""
    target = (
        registry.local_dir(manifest_path)
        / "derived"
        / adapter.name
        / f"{slug(track.track_id)}.flac"
    )
    if target.is_file():
        return target
    audio, rate = read_sum([p for paths in track.stems.values() for p in paths])
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak > 1:
        audio = audio / peak  # A summed mix may exceed full scale; keep it unclipped.
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, audio, rate, format="FLAC", subtype="PCM_24")
    return target


def _assign_splits(adapter, tracks, existing):
    """One split per group. Official test material always stays test; a group that the
    dataset spreads over test and non-test goes to test as a whole. Groups without an
    official mapping are divided deterministically 80/20 into train and validation."""
    groups = {}
    for track in tracks:
        groups.setdefault(group_id(adapter.name, track.group_key), set()).add(
            adapter.split_map.get(track.official_split)
        )
    result = {}
    for group, wanted in groups.items():
        if group in existing:
            result[group] = existing[group]
        elif "test" in wanted:
            result[group] = "test"
        elif wanted == {"validation"}:
            result[group] = "validation"
        elif wanted == {"train"} and "validation" in adapter.split_map.values():
            result[group] = "train"
        else:
            bucket = int(hashlib.sha256(f"gmv-split-1:{group}".encode()).hexdigest(), 16) % 5
            result[group] = "validation" if bucket == 0 else "train"
    return result


def import_dataset(manifest_path, adapter, root, *, limit=None, progress=print):
    """Register every discoverable track of a dataset and write its references."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset folder not found: {root}")
    tracks = sorted(adapter.discover(root), key=lambda t: t.track_id)
    if not tracks:
        raise ValueError(f"No {adapter.title} tracks found under {root}; check the folder layout")
    manifest_path = Path(manifest_path)
    manifest = (
        registry.load_manifest(manifest_path)
        if manifest_path.is_file()
        else CorpusManifest(corpus_version="0.1.0")
    )
    paths = registry.load_paths(manifest_path)
    known = {r.dataset_track_id: r for r in manifest.recordings if r.dataset == adapter.name}
    splits = _assign_splits(
        adapter, tracks, {r.group_id: r.split for r in manifest.recordings if r.split in SPLITS}
    )
    recordings = {r.recording_id: r for r in manifest.recordings}
    summary = {"imported": 0, "already_registered": 0, "failed": [], "layers": {}}
    for track in tracks[:limit] if limit else tracks:
        if track.track_id in known:
            summary["already_registered"] += 1
            continue
        try:
            mixture = track.mixture or render_mixture(manifest_path, adapter, track)
            duration, rate, channels = registry.probe(mixture)
            digest = registry.file_sha256(mixture)
            single = {label: files[0] for label, files in track.stems.items() if len(files) == 1}
            group = group_id(adapter.name, track.group_key)
            recording = Recording(
                recording_id="r-" + digest[:12],
                sha256=digest,
                duration_s=duration,
                sample_rate=rate,
                channels=channels,
                source_kind="open_dataset",
                license=adapter.audio_license,
                permission_basis=f"Obtained by the user under the {adapter.title} terms; "
                "used locally for evaluation and not redistributed",
                group_id=group,
                split=splits[group],
                tags=tuple(sorted(set(track.tags) | {adapter.name})),
                stems={label: registry.file_sha256(path) for label, path in single.items()},
                dataset=adapter.name,
                dataset_track_id=track.track_id,
                dataset_split=track.official_split,
            )
            written = []
            if track.stems:
                audio = {label: read_sum(files) for label, files in track.stems.items()}
                written.append(stems.reference(recording, audio, uncertain=track.uncertain))
            written += [build(recording) for build in track.annotations]
        except Exception as exc:
            summary["failed"].append({"track": track.track_id, "reason": str(exc)})
            progress(f"  failed {track.track_id}: {exc}")
            continue
        recordings[recording.recording_id] = recording
        paths["recordings"][recording.recording_id] = str(mixture)
        if single:
            paths["stems"][recording.recording_id] = {k: str(v) for k, v in single.items()}
        for annotation in written:
            registry.save_annotation(manifest_path, annotation)
            for layer in annotation.coverage.layers:
                kinds = summary["layers"].setdefault(layer, {})
                kinds[annotation.reference_kind] = kinds.get(annotation.reference_kind, 0) + 1
        # Save after every track so an interrupted import can simply be run again.
        registry.save_manifest(
            manifest_path,
            CorpusManifest(
                corpus_version=manifest.corpus_version, recordings=tuple(recordings.values())
            ),
        )
        registry.save_paths(manifest_path, paths)
        summary["imported"] += 1
        progress(f"  imported {recording.recording_id} ({recording.split}, {duration:.0f} s)")
    counts = {}
    for recording in recordings.values():
        if recording.dataset == adapter.name:
            counts[recording.split] = counts.get(recording.split, 0) + 1
    summary["recordings_per_split"] = counts
    summary["layers"] = dict(sorted(summary["layers"].items()))
    return summary


def adapters():
    from . import mdbdrums, moisesdb, musdb

    return {m.ADAPTER.name: m.ADAPTER for m in (mdbdrums, musdb, moisesdb)}
