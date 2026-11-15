"""Manifest and annotation I/O plus the ignored local map from recording id to audio path.

Paths, file names, titles and artists never enter the committed manifest; they live only in
``corpus-local/paths.json`` next to the corpus directory.
"""

import hashlib
import json
from pathlib import Path

import soundfile as sf

from .schema import AnnotationFile, CorpusManifest

MAX_BYTES = 2 * 1024**3
MAX_DECODED_BYTES = 1024**3


def file_sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def load_manifest(path):
    return CorpusManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_manifest(path, manifest):
    ordered = manifest.model_copy(
        update={"recordings": tuple(sorted(manifest.recordings, key=lambda r: r.recording_id))}
    )
    write_json(path, CorpusManifest.model_validate(ordered.model_dump()).model_dump(mode="json"))


def annotation_path(manifest_path, recording_id, annotator_id):
    return Path(manifest_path).parent / "annotations" / recording_id / f"{annotator_id}.json"


def load_annotation(path):
    return AnnotationFile.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_annotation(manifest_path, annotation):
    path = annotation_path(manifest_path, annotation.recording_id, annotation.annotator_id)
    write_json(path, annotation.model_dump(mode="json"))
    return path


def annotations_for(manifest_path, recording_id):
    folder = Path(manifest_path).parent / "annotations" / recording_id
    return [load_annotation(p) for p in sorted(folder.glob("*.json"))] if folder.is_dir() else []


def local_dir(manifest_path):
    return Path(manifest_path).resolve().parent.with_name("corpus-local")


def load_paths(manifest_path):
    path = local_dir(manifest_path) / "paths.json"
    if not path.is_file():
        return {"recordings": {}, "stems": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_paths(manifest_path, paths):
    write_json(local_dir(manifest_path) / "paths.json", paths)


def resolve(recording, paths, *, verify_hash=True):
    """Return (path, None) or (None, reason). Never guesses another file."""
    raw = paths["recordings"].get(recording.recording_id)
    if not raw:
        return None, "not registered on this machine"
    path = Path(raw)
    if not path.is_file():
        return None, "audio file is missing"
    if verify_hash and file_sha256(path) != recording.sha256:
        return None, "audio file changed (hash mismatch)"
    return path, None


def resolve_stems(recording, paths, *, verify_hash=True):
    found, problems = {}, {}
    for label, digest in recording.stems.items():
        raw = paths["stems"].get(recording.recording_id, {}).get(label)
        if not raw or not Path(raw).is_file():
            problems[label] = "stem file is missing"
        elif verify_hash and file_sha256(raw) != digest:
            problems[label] = "stem file changed (hash mismatch)"
        else:
            found[label] = Path(raw)
    return found, problems


def probe(path):
    """Size-checked header read: (duration_s, sample_rate, channels)."""
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Audio file exceeds 2 GiB")
    with sf.SoundFile(path) as file:
        if file.frames <= 0:
            raise ValueError("Audio file has no samples")
        return file.frames / file.samplerate, file.samplerate, file.channels


def read_audio(path):
    with sf.SoundFile(path) as file:
        if file.frames * file.channels * 4 > MAX_DECODED_BYTES:
            raise ValueError("Decoded audio would exceed 1 GiB; trim the recording")
        return file.read(dtype="float32", always_2d=True), file.samplerate
