"""The external datasets this project knows about, as one table: what each is, its licences
by layer (audio, annotations, code are often licensed differently), what its labels can and
cannot say about guitar, and what is on this machine.

Counts that depend on local files are measured when the table is written; the rest is taken
from each dataset's own documentation and is dated in ``checked``.
"""

import csv
import json
from pathlib import Path

from . import openmic, write_csv

CHECKED = "2026-11-21"
FIELDS = (
    "dataset",
    "version",
    "source",
    "role",
    "audio_licence",
    "annotation_licence",
    "code_licence",
    "derived_weights",
    "redistribution",
    "splits",
    "label_types",
    "temporal_annotation",
    "recordings",
    "guitar_examples",
    "guitar_negative_examples",
    "known_limitations",
    "local_path",
    "local_availability",
    "processing_status",
    "checked",
)


def _openmic(root):
    try:
        counts = openmic.summary(openmic.read(root))
    except (FileNotFoundError, OSError):
        return None
    guitar = counts["by_instrument"]["guitar"]
    return counts, guitar


def _jamendo(path):
    path = Path(path)
    if not path.is_file():
        return None
    guitar_tags = {f"instrument---{t}" for t in ("guitar", "electricguitar", "acousticguitar", "classicalguitar")}
    total = with_guitar = 0
    with open(path, encoding="utf-8") as handle:
        for row in list(csv.reader(handle, delimiter="\t"))[1:]:
            total += 1
            with_guitar += bool(guitar_tags & set(row[5:]))
    return total, with_guitar


def _moisesdb(root, jobs):
    root, jobs = Path(root), Path(jobs)
    songs = sum(1 for p in root.iterdir() if p.is_dir()) if root.is_dir() else 0
    used = guitar = 0
    if jobs.is_file():
        listed = json.loads(jobs.read_text(encoding="utf-8"))
        used = len(listed)
        guitar = sum(1 for j in listed if "guitar" in j.get("stems", []))
    return songs, used, guitar


def manifest(workdir, openmic_root, moisesdb_root, moisesdb_jobs, medleydb_root=None):
    workdir = Path(workdir)
    rows = []
    found = _openmic(openmic_root) if openmic_root else None
    if found:
        counts, guitar = found
        splits = ", ".join(f"{k} {v}" for k, v in counts["splits"].items())
        licences = "; ".join(f"{k}: {v}" for k, v in list(counts["audio_licences"].items())[:6])
    rows.append(
        {
            "dataset": "OpenMIC-2018",
            "version": "1.0.0 (Zenodo 1432913)",
            "source": "https://zenodo.org/records/1432913",
            "role": "reference evidence: human judgments of instrument presence in real music; "
            "independent test of the audio models",
            "audio_licence": "per clip, the Creative Commons licence of its Free Music Archive track"
            + (f" ({licences})" if found else ""),
            "annotation_licence": "CC BY 4.0",
            "code_licence": "MIT (github.com/cosmir/openmic-2018)",
            "derived_weights": "audio is mostly NonCommercial and partly NoDerivatives: treat any "
            "weights trained on it as non-commercial, and exclude NoDerivatives clips from training "
            "until that question is settled",
            "redistribution": "annotations yes with attribution; audio only under each clip's licence",
            "splits": f"original partition kept ({splits})" if found else "original train / test",
            "label_types": "20 instruments; aggregated relevance 0-1 and number of responses; "
            "individual responses; only some instruments asked per clip",
            "temporal_annotation": "none finer than the 10 s clip",
            "recordings": counts["clips"] if found else "20000",
            "guitar_examples": guitar.get("present", 0) if found else "",
            "guitar_negative_examples": guitar.get("absent", 0) if found else "",
            "known_limitations": "guitar was asked about for "
            + (str(guitar.get("present", 0) + guitar.get("absent", 0) + guitar.get("uncertain", 0)) if found else "a minority")
            + " clips only; the rest are unknown, not guitar-free. Guitar type is not labelled. "
            "Crowd answers, trust-weighted",
            "local_path": str(openmic_root or ""),
            "local_availability": "complete" if found else "not present",
            "processing_status": "labels imported; guitar-labelled clips scored by both audio models"
            if found
            else "not processed",
            "checked": CHECKED,
        }
    )
    songs, used, guitar_songs = _moisesdb(moisesdb_root, moisesdb_jobs)
    rows.append(
        {
            "dataset": "MoisesDB",
            "version": "0.1",
            "source": "https://github.com/moises-ai/moises-db",
            "role": "training material for guitar_mn10_v1 (mixes made from stems); its held-out "
            "artist folds are the only MoisesDB material that may be scored",
            "audio_licence": "CC BY-NC-SA 4.0",
            "annotation_licence": "CC BY-NC-SA 4.0 (stem taxonomy shipped with the audio)",
            "code_licence": "MIT (loader)",
            "derived_weights": "non-commercial, share-alike, with attribution",
            "redistribution": "not redistributed by this project",
            "splits": "by artist; folds fixed in experiments/recognition/head.py",
            "label_types": "isolated stems with a two-level taxonomy (guitar: clean, distorted, acoustic, ...)",
            "temporal_annotation": "derived from stem energy (not annotated by hand)",
            "recordings": songs,
            "guitar_examples": f"{guitar_songs} of the {used} train and validation songs have a guitar stem",
            "guitar_negative_examples": "guitar-removed remixes of the same songs (same artists)",
            "known_limitations": "82-85% of the time has guitar; negatives are remixes, not other "
            "music; mixes are cleaner than released records; already used for training, so no "
            "new evidence is available from it",
            "local_path": str(moisesdb_root),
            "local_availability": "complete" if songs else "not present",
            "processing_status": "in use since the guitar recognition milestone",
            "checked": CHECKED,
        }
    )
    medley = bool(medleydb_root and Path(medleydb_root).is_dir())
    rows.append(
        {
            "dataset": "MedleyDB",
            "version": "1.0 (122 songs) and 2.0 (74 songs)",
            "source": "https://medleydb.weebly.com/downloads.html",
            "role": "intended final test: real multitracks with typed guitar stems, never trained on",
            "audio_licence": "CC BY-NC-SA 4.0; the maintainers ask that it is not republished",
            "annotation_licence": "CC BY-NC-SA 4.0",
            "code_licence": "MIT (github.com/marl/medleydb)",
            "derived_weights": "non-commercial, share-alike (none planned: test use only)",
            "redistribution": "no",
            "splits": "none official; artists shared with MUSDB18 and MDB Drums map to one group",
            "label_types": "stems and raw tracks with instrument names; instrument activations; "
            "has_bleed flag per song",
            "temporal_annotation": "activation envelopes computed from stems (not hand labelled); "
            "unreliable where has_bleed is set",
            "recordings": 196,
            "guitar_examples": "not counted: not available locally",
            "guitar_negative_examples": "not counted: not available locally",
            "known_limitations": "access is granted on request through Zenodo and needs the "
            "owner's account; about 100 GB",
            "local_path": str(medleydb_root or ""),
            "local_availability": "complete" if medley else "not present: access request pending with the owner",
            "processing_status": "adapter not written (nothing to read)",
            "checked": CHECKED,
        }
    )
    tags = _jamendo(workdir / "datasets" / "jamendo_autotagging_instrument.tsv")
    rows.append(
        {
            "dataset": "MTG-Jamendo",
            "version": "repository master, autotagging_instrument subset",
            "source": "https://mtg.github.io/mtg-jamendo-dataset/",
            "role": "weak positive evidence only (uploader tags); a source of varied licensed "
            "audio if a targeted download is decided later",
            "audio_licence": "per track, Creative Commons (several variants, many NonCommercial)",
            "annotation_licence": "CC BY-NC-SA 4.0 (metadata)",
            "code_licence": "Apache-2.0",
            "derived_weights": "per-track licences apply; treat as non-commercial",
            "redistribution": "metadata under its licence; audio only under each track's licence",
            "splits": "five official artist-disjoint splits (not used yet)",
            "label_types": "instrument tags chosen by uploaders (41 tags)",
            "temporal_annotation": "none (whole track)",
            "recordings": tags[0] if tags else "25135 in the instrument subset",
            "guitar_examples": f"{tags[1]} tracks carry a guitar tag (weak evidence)" if tags else "",
            "guitar_negative_examples": "none: a missing guitar tag is unknown, not guitar-free",
            "known_limitations": "tags are not verified and incomplete; no negatives; full audio "
            "is hundreds of GB and was not downloaded",
            "local_path": str(workdir / "datasets" / "jamendo_autotagging_instrument.tsv"),
            "local_availability": "tag file only" if tags else "not present",
            "processing_status": "tags counted; no audio fetched",
            "checked": CHECKED,
        }
    )
    for name, source, licence, note in (
        ("MUSDB18-HQ", "https://zenodo.org/records/3338373", "mixed; academic, non-commercial use",
         "no guitar stem (guitar is inside 'other'): no guitar labels"),
        ("MusicNet", "https://zenodo.org/records/5120004", "CC BY 4.0",
         "classical chamber music without guitar: guitar-free negatives only"),
        ("GuitarSet", "https://zenodo.org/records/3371780", "CC BY 4.0",
         "solo acoustic guitar: no mixes, used for pitch tracking only"),
    ):
        rows.append(
            {
                "dataset": name,
                "version": "as documented in docs/DATASETS.md",
                "source": source,
                "role": "already in the project; listed for completeness",
                "audio_licence": licence,
                "annotation_licence": licence,
                "code_licence": "",
                "derived_weights": "follow the audio licence",
                "redistribution": "not redistributed by this project",
                "splits": "see docs/DATASETS.md",
                "label_types": note,
                "temporal_annotation": "see docs/DATASETS.md",
                "recordings": "",
                "guitar_examples": "",
                "guitar_negative_examples": "",
                "known_limitations": note,
                "local_path": "",
                "local_availability": "present",
                "processing_status": "unchanged in this milestone",
                "checked": CHECKED,
            }
        )
    write_csv(workdir / "External_Dataset_Manifest.csv", rows, FIELDS)
    return rows
