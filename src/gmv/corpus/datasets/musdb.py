"""MUSDB18-HQ adapter (uncompressed WAV edition).

Expected layout, as extracted from the MUSDB18-HQ archive::

    <root>/train/<Artist - Title>/{mixture,drums,bass,vocals,other}.wav
    <root>/test/<Artist - Title>/{mixture,drums,bass,vocals,other}.wav

The dataset provides four stems and a mixture per track. It has no onset, beat, downbeat,
tempo or instrument-attack annotations, and no activity annotations; instrument activity is
derived here from the stems and labelled ``stem_derived``.
"""

from . import Adapter, Track, artist_key

STEMS = ("drums", "bass", "vocals", "other")


def discover(root):
    tracks = []
    for split in ("train", "test"):
        folder = root / split
        if not folder.is_dir():
            continue
        for track in sorted(p for p in folder.iterdir() if p.is_dir()):
            mixture = track / "mixture.wav"
            stems = {s: [track / f"{s}.wav"] for s in STEMS if (track / f"{s}.wav").is_file()}
            if not mixture.is_file() or not stems:
                continue
            tracks.append(
                Track(
                    track_id=track.name,
                    group_key=artist_key(track.name.split(" - ")[0]),
                    official_split=split,
                    mixture=mixture,
                    stems=stems,
                )
            )
    return tracks


ADAPTER = Adapter(
    name="musdb18hq",
    title="MUSDB18-HQ",
    audio_license="MUSDB18 terms: academic, non-commercial use only; per-track licences vary",
    access="musdb18hq.zip, 22.7 GB, https://zenodo.org/records/3338373",
    homepage="https://sigsep.github.io/datasets/musdb.html",
    reference="Mixture plus drums, bass, vocals and other stems. No event or activity annotations.",
    discover=discover,
    split_map={"train": None, "test": "test"},
)
