"""MoisesDB songs summed without their guitar files: guitar-free "twins".

Same folder as the ``moisesdb`` adapter. Every song that has guitar files and something else
becomes a second recording, the sum of all other stem files. Its artist group, and with it
its split, is that of the original song, so a twin never crosses from training into testing.

Guitar is absent from a twin by construction. Faint guitar can remain where another file
contains bleed (the dataset flags such files, mostly vocals); that is not corrected.
"""

from dataclasses import replace

from . import GUITAR_LABELS, Adapter, absent, moisesdb


def discover(root):
    tracks = []
    for track in moisesdb.discover(root):
        guitar = set(track.stems.get("guitar", ()))
        stems = {
            label: [path for path in files if path not in guitar]
            for label, files in track.stems.items()
        }
        stems = {label: files for label, files in stems.items() if files}
        if not guitar or not stems:
            continue
        tracks.append(
            replace(
                track,
                track_id=f"{track.track_id}-noguitar",
                stems=stems,
                uncertain=tuple(label for label in track.uncertain if label in stems),
                tags=(*track.tags, "guitar_removed"),
                annotations=[absent(GUITAR_LABELS, "Summed without the dataset's guitar files")],
            )
        )
    return tracks


ADAPTER = Adapter(
    name="moisesdb_noguitar",
    title="MoisesDB without guitar files",
    audio_license=moisesdb.ADAPTER.audio_license,
    access=moisesdb.ADAPTER.access,
    homepage=moisesdb.ADAPTER.homepage,
    reference="The MoisesDB multitracks summed without their guitar files; guitar absent "
    "by construction, other instruments from their stems.",
    discover=discover,
    holdout=True,
)
