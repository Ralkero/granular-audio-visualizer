"""Run the MusicBrainz enrichment over an inventory. Resumable: every answer is cached, so
a second run only asks what is still missing and then rewrites the two output files."""

from collections import Counter

from . import musicbrainz as mb
from . import read_inventory, recording_id, write_jsonl


def enrich(inventory, workdir, limit=None, offline=False, log=print, client=None):
    rows = read_inventory(inventory)[: limit or None]
    client = client or mb.Client(workdir / "cache" / "musicbrainz", offline=offline)
    identities, evidence = [], []
    release_cache, artist_cache = {}, {}
    for index, row in enumerate(rows, 1):
        rid = recording_id(row["sha256"])
        found = mb.identify(client, row)
        chosen = found["chosen"]
        entry = {
            "recording_id": rid,
            "status": found["status"],
            "method": found["method"],
            "confidence": found["confidence"],
            "recording_mbid": chosen["recording_mbid"] if chosen else "",
            "mb_title": chosen["title"] if chosen else "",
            "mb_artist": chosen["artist"] if chosen else "",
            "mb_disambiguation": chosen["disambiguation"] if chosen else "",
            "mb_length_s": chosen["length_s"] if chosen else None,
            "artist_mbids": chosen["artist_mbids"] if chosen else [],
            "isrcs": chosen["isrcs"] if chosen else [],
            "fit": chosen["fit"] if chosen else None,
            "release_mbid": "",
            "release_group_mbid": "",
            "release_title": "",
            "release_choice": "",
            "tags": [],
            "alternatives": found["alternatives"],
            "sources": found["sources"],
        }
        credits = []
        if chosen:
            release, how = mb.pick_release(row, chosen)
            own = None
            if release:
                key = release["release_mbid"]
                if key not in release_cache:
                    release_cache[key] = mb.release_credits(client, key)
                whole, per_recording = release_cache[key]
                entry.update(
                    release_mbid=key,
                    release_group_mbid=release.get("release_group_mbid", ""),
                    release_title=release.get("title", ""),
                    release_choice=how,
                )
                credits += whole
                own = per_recording.get(chosen["recording_mbid"])
            if own is None:  # no release chosen, or it does not list this recording
                own, entry["tags"] = mb.recording_credits(client, chosen["recording_mbid"])
            credits += own
            for artist in chosen["artist_mbids"][:2]:
                if artist not in artist_cache:
                    artist_cache[artist] = mb.artist_credits(client, artist)
                members, tags = artist_cache[artist]
                credits += members
                entry["tags"] = sorted(set(entry["tags"]) | set(tags))
        identities.append(entry)
        for credit in credits:
            evidence.append(
                {"recording_id": rid, "match_confidence": found["confidence"], **credit}
            )
        if index % 25 == 0 or index == len(rows):
            log(f"{index}/{len(rows)} recordings, {client.requests} requests this run", flush=True)
    write_jsonl(workdir / "musicbrainz" / "identity.jsonl", identities)
    write_jsonl(workdir / "Metadata_Evidence.jsonl", evidence)
    counts = Counter((e["status"], e["method"], e["confidence"]) for e in identities)
    return {
        "recordings": len(rows),
        "complete": not offline and limit is None,
        "requests": client.requests,
        "matches": {" / ".join(k): v for k, v in sorted(counts.items())},
        "credits": len(evidence),
    }
