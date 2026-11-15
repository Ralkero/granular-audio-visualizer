"""Round-trip annotations through Audacity label-track files (File > Export > Export Labels).

One text file per layer, each line ``start<TAB>end<TAB>text`` in seconds; a point label has
start == end. Files are named ``<layer>.txt``, optionally prefixed ``<recording_id>.``:

==================  =========================================================================
``onset.txt``       points: full-mix acoustic onsets
``beat.txt``        points: beats
``downbeat.txt``    points: downbeats
``tempo.txt``       points whose text is the tempo in BPM from that time on
``sections.txt``    points or regions; text is a free section name, or ``transition``
``activity.X.txt``  regions where instrument X is sounding (overlap between files is normal)
``hits.X.txt``      points: individual attacks of instrument X (kick, snare, cymbals ...)
``coverage.txt``    optional regions that were actually annotated (default: whole recording)
==================  =========================================================================

A label text ending in ``?`` marks that label uncertain. A file that exists but is empty means
"this layer was annotated and nothing is present"; a missing file means "not annotated".
"""

from pathlib import Path

from .schema import (
    EVENT_LAYERS,
    LABELS,
    Activity,
    AnnotationFile,
    Coverage,
    InstrumentEvent,
    MixEvent,
    TimeRange,
    event_layer,
    layer_names,
)

TOOL = "audacity_labels"


def read_labels(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.startswith("\\"):  # spectral-selection continuation lines
            continue
        parts = line.split("\t")
        try:
            start, end = float(parts[0].replace(",", ".")), float(parts[1].replace(",", "."))
        except (IndexError, ValueError):
            raise ValueError(
                f"{Path(path).name}:{number}: expected start<TAB>end<TAB>text"
            ) from None
        if start < 0 or end < start:
            raise ValueError(f"{Path(path).name}:{number}: invalid times")
        rows.append((start, end, parts[2].strip() if len(parts) > 2 else ""))
    return rows


def _split_text(text):
    uncertain = text.endswith("?")
    return text.rstrip("?").strip(), ("uncertain" if uncertain else "certain")


def find_layers(folder, recording_id):
    found = {}
    for path in sorted(Path(folder).glob("*.txt")):
        layer = path.stem.removeprefix(recording_id + ".")
        if layer == "coverage" or layer in layer_names():
            if layer in found:
                raise ValueError(f"Two files for layer {layer}")
            found[layer] = path
        else:
            raise ValueError(f"{path.name}: unknown layer name '{layer}'")
    return found


def import_folder(folder, recording, annotator_id, *, uncertainty_seconds=0.02, tool_version=""):
    files = find_layers(folder, recording.recording_id)
    if not set(files) - {"coverage"}:
        raise ValueError("No label files found")
    events, hits, activity = [], [], []
    for layer, path in files.items():
        for start, end, raw in read_labels(path):
            text, certainty = _split_text(raw)
            if end > recording.duration_s + 0.05:
                raise ValueError(f"{path.name}: label at {end:.3f}s is past the recording end")
            if layer in ("onset", "beat", "downbeat"):
                events.append(MixEvent(time=start, kind=layer, certainty=certainty))
            elif layer == "tempo":
                try:
                    bpm = float(text.lower().removesuffix("bpm").strip().replace(",", "."))
                except ValueError:
                    raise ValueError(f"{path.name}: tempo label must be a BPM number") from None
                events.append(
                    MixEvent(time=start, kind="tempo_change", value=bpm, certainty=certainty)
                )
            elif layer == "sections":
                kind = "transition" if text.lower() == "transition" else "section_change"
                events.append(
                    MixEvent(
                        time=start,
                        kind=kind,
                        end=end if end > start else None,
                        value=text or None,
                        certainty=certainty,
                    )
                )
            elif layer.startswith("activity."):
                if end <= start:
                    raise ValueError(f"{path.name}: activity labels must be regions, not points")
                activity.append(
                    Activity(start=start, end=end, label=layer[9:], certainty=certainty)
                )
            elif layer.startswith("hits."):
                hits.append(InstrumentEvent(time=start, label=layer[5:], certainty=certainty))
    ranges = (
        tuple(TimeRange(start=s, end=e) for s, e, _ in read_labels(files["coverage"]) if e > s)
        if "coverage" in files
        else ()
    ) or (TimeRange(start=0, end=recording.duration_s),)
    return AnnotationFile(
        recording_id=recording.recording_id,
        recording_sha256=recording.sha256,
        annotator_id=annotator_id,
        tool=TOOL,
        tool_version=tool_version,
        provenance="Manual annotation in Audacity label tracks",
        uncertainty_seconds=uncertainty_seconds,
        coverage=Coverage(ranges=ranges, layers=tuple(sorted(set(files) - {"coverage"}))),
        events=tuple(sorted(events, key=lambda e: (e.time, e.kind))),
        instrument_events=tuple(sorted(hits, key=lambda e: (e.time, e.label))),
        activity=tuple(sorted(activity, key=lambda a: (a.start, a.label))),
    )


def export_folder(annotation, folder):
    """Write one label file per covered layer so Audacity can import and edit them."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = {layer: [] for layer in annotation.coverage.layers}

    def mark(text, certainty):
        return text + ("?" if certainty == "uncertain" else "")

    for event in annotation.events:
        layer = event_layer(event.kind)
        if layer == "tempo":
            text = f"{event.value:g}"
        elif layer == "sections":
            text = str(event.value or event.kind)
        else:
            text = ""
        rows[layer].append((event.time, event.end or event.time, mark(text, event.certainty)))
    for hit in annotation.instrument_events:
        rows[f"hits.{hit.label}"].append((hit.time, hit.time, mark("", hit.certainty)))
    for span in annotation.activity:
        rows[f"activity.{span.label}"].append((span.start, span.end, mark("", span.certainty)))
    rows["coverage"] = [(r.start, r.end, "") for r in annotation.coverage.ranges]
    written = []
    for layer, lines in rows.items():
        path = folder / f"{annotation.recording_id}.{layer}.txt"
        text = "".join(f"{s:.6f}\t{e:.6f}\t{t}\n" for s, e, t in sorted(lines))
        path.write_text(text, encoding="utf-8", newline="\n")
        written.append(path)
    return written


__all__ = ["EVENT_LAYERS", "LABELS", "TOOL", "export_folder", "import_folder", "read_labels"]
