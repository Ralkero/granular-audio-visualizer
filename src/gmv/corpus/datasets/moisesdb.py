"""MoisesDB adapter: multitracks with separate kick, snare, tom, cymbal, guitar, bass, vocal stems.

Expected layout (the extracted archive)::

    <root>/[moisesdb_v0.1/]<track uuid>/data.json
    <root>/[moisesdb_v0.1/]<track uuid>/<stemName>/<track id>.<extension>

``data.json`` lists ``artist``, ``song``, ``genre`` and ``stems[] -> {stemName, tracks[] ->
{id, extension, trackType, has_bleed}}``. There is no mixture file; the mix is the sum of all
stems and is rendered locally on import. The dataset ships no activity, onset or beat
annotations: instrument activity is derived from the stems (``stem_derived``), and stems
flagged ``has_bleed`` are marked uncertain.
"""

import json

from . import Adapter, Track, artist_key, slug

STEM_LABEL = {
    "vocals": "vocals",
    "bass": "bass",
    "drums": "drums",
    "guitar": "guitar",
    "piano": "other_harmonic",
    "other_keys": "other_harmonic",
    "bowed_strings": "other_harmonic",
    "wind": "other_harmonic",
    "other_plucked": "other_harmonic",
    "percussion": "other",
    "other": "other",
}
# Individual drum microphones. Overheads, full kits and drum machines only count as "drums".
DRUM_TRACK_LABEL = {
    "kick_drum": "kick",
    "snare_drum": "snare",
    "toms": "toms",
    "cymbals": "cymbals",
    "hi_hat": "hi_hat",
    "hihat": "hi_hat",
}

# Guitar tracks by how they sound. Anything else in the guitar stem only counts as "guitar".
GUITAR_TRACK_LABEL = {
    "distorted_electric_guitar": "guitar_distorted",
    "clean_electric_guitar": "guitar_clean",
    "acoustic_guitar": "guitar_acoustic",
}


def discover(root):
    tracks = []
    files = sorted(set(root.glob("*/data.json")) | set(root.glob("*/*/data.json")))
    for meta in files:
        data = json.loads(meta.read_text(encoding="utf-8"))
        stems, uncertain = {}, set()
        for stem in data.get("stems", []):
            name = slug(stem.get("stemName", ""))
            label = STEM_LABEL.get(name)
            if label is None:
                continue
            for item in stem.get("tracks", []):
                extension = item.get("extension", "wav")
                candidates = [
                    meta.parent / folder / f"{item['id']}.{extension}"
                    for folder in (stem["stemName"], name, stem["stemName"].replace(" ", "_"))
                ]
                path = next((p for p in candidates if p.is_file()), None)
                if path is None:
                    continue
                labels = [label]
                if name == "drums" and slug(item.get("trackType", "")) in DRUM_TRACK_LABEL:
                    labels.append(DRUM_TRACK_LABEL[slug(item["trackType"])])
                if name == "guitar" and slug(item.get("trackType", "")) in GUITAR_TRACK_LABEL:
                    labels.append(GUITAR_TRACK_LABEL[slug(item["trackType"])])
                for target in labels:
                    stems.setdefault(target, []).append(path)
                    if item.get("has_bleed"):
                        uncertain.add(target)
        if not stems:
            continue
        genre = slug(data.get("genre", ""))
        tracks.append(
            Track(
                track_id=meta.parent.name,
                group_key=artist_key(data.get("artist", meta.parent.name)),
                official_split=None,
                mixture=None,
                stems=stems,
                uncertain=tuple(sorted(uncertain)),
                tags=(f"genre_{genre}",) if genre != "unknown" else (),
            )
        )
    return tracks


ADAPTER = Adapter(
    name="moisesdb",
    title="MoisesDB",
    audio_license="CC BY-NC-SA 4.0; non-commercial research use only",
    access="Download button and form at https://music.ai/research/ (a person must submit it)",
    homepage="https://github.com/moises-ai/moises-db",
    reference="240 multitracks. Stems incl. kick, snare, toms, cymbals, bass, vocals, guitar. "
    "No mixture file and no activity, onset or beat annotations.",
    discover=discover,
    holdout=True,
)
