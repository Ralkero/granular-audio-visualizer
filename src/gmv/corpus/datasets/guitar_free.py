"""Datasets of real recordings that contain no guitar: MusicNet and The Spheres.

They add what the multitrack datasets lack: complete recordings (piano, strings, winds,
orchestra) on which any reported guitar is a false alarm. Neither has a guitar anywhere in
its instrumentation, so guitar is declared absent for the whole recording; nothing else is
labelled.

MusicNet layout (the extracted archive plus the metadata file from the same record)::

    <root>/[musicnet/]{train_data,test_data}/<id>.wav
    <root>/musicnet_metadata.csv      id, composer, composition, movement, ensemble, ...

The Spheres layout (the "StereoMix" archive)::

    <root>/[TheSpheresDataset-StereoMix/]TheSpheres_data/<piece>/StereoMix/<instrument>.flac

The Spheres ships one file per instrument section; the mix is their sum, rendered locally.
"""

import csv

from . import GUITAR_LABELS, Adapter, Track, absent, slug


def musicnet(root):
    meta = next(iter(sorted(root.glob("**/musicnet_metadata.csv"))), None)
    if meta is None:
        return []
    with meta.open(encoding="utf-8", newline="") as handle:
        rows = {row["id"]: row for row in csv.DictReader(handle)}
    tracks = []
    for split in ("train", "test"):
        for path in sorted(root.glob(f"**/{split}_data/*.wav")):
            row = rows.get(path.stem)
            if row is None:
                continue
            ensemble = slug(row["ensemble"])
            tracks.append(
                Track(
                    track_id=path.stem,
                    # Movements of one work stay together.
                    group_key=f"musicnet{slug(row['composer'])}{slug(row['composition'])}",
                    official_split=split,
                    mixture=path,
                    stems={},
                    tags=("guitar_free", "classical", f"ensemble_{ensemble}"),
                    annotations=[absent(GUITAR_LABELS, "MusicNet has no guitar recordings")],
                )
            )
    return tracks


def spheres(root):
    tracks = []
    for folder in sorted(root.glob("**/TheSpheres_data/*/StereoMix")):
        files = sorted(folder.glob("*.flac"))
        if not files:
            continue
        piece = folder.parent.name
        tracks.append(
            Track(
                track_id=slug(piece),
                group_key=f"spheres{slug(piece)}",
                official_split=None,
                mixture=None,
                # One label for the whole orchestra: the sections are not project labels.
                stems={"other_harmonic": files},
                tags=("guitar_free", "classical", "ensemble_orchestra"),
                annotations=[absent(GUITAR_LABELS, "Orchestra without guitar")],
            )
        )
    return tracks


MUSICNET = Adapter(
    name="musicnet",
    title="MusicNet",
    audio_license="CC BY 4.0",
    access="Direct download from https://zenodo.org/records/5120004",
    homepage="https://zenodo.org/records/5120004",
    reference="330 classical recordings (piano, strings, winds). No guitar: guitar is "
    "declared absent. Note labels are not imported.",
    discover=musicnet,
    split_map={"train": None, "test": "test"},
)
SPHERES = Adapter(
    name="spheres",
    title="The Spheres Dataset (stereo mix)",
    audio_license="CC BY-SA 4.0",
    access="Direct download from https://zenodo.org/records/17347681",
    homepage="https://zenodo.org/records/17347681",
    reference="Two orchestral pieces, one file per instrument section. No guitar.",
    discover=spheres,
    # Two pieces only: both are kept for testing.
    split_map={None: "test"},
)
