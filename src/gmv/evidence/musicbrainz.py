"""Identify recordings in MusicBrainz and collect instrument credits, through its documented
web service only (https://musicbrainz.org/doc/MusicBrainz_API).

Rules kept here:

* one request a second at most, a descriptive User-Agent, every response cached on disk,
  back-off when the service asks for it;
* a recording is matched by, in order: the MusicBrainz recording id in the file's tags, its
  ISRC, then a scored search on artist, title, album and duration. The method, a confidence
  class and the alternatives that were not chosen are kept;
* a match is never silently made across versions: when the best candidates differ in kind
  (live, remix, acoustic, ...) or cannot be told apart, the identity stays ``ambiguous``;
* credits keep their scope: ``recording`` (this recording), ``release`` (the whole release,
  which says nothing about one track) and ``artist`` (band membership). A missing credit is
  not evidence of anything.
"""

import hashlib
import json
import re
import subprocess
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path

API = "https://musicbrainz.org/ws/2/"
USER_AGENT = (
    "GranularMusicVisualizer/0.1.0 ( https://github.com/Ralkero/granular-audio-visualizer )"
)
VERSION_WORDS = (
    "live",
    "remix",
    "acoustic",
    "demo",
    "instrumental",
    "remaster",
    "edit",
    "version",
    "mix",
    "session",
    "karaoke",
    "cover",
)
INSTRUMENT_FAMILIES = {
    "guitar": ("guitar",),
    "bass": ("bass",),
    "keyboard": ("keyboard", "piano", "organ", "synth", "rhodes", "mellotron", "clavinet"),
    "strings": ("violin", "viola", "cello", "strings", "double bass", "contrabass", "fiddle"),
    "drums": ("drum", "percussion"),
    "vocals": ("vocal", "voice", "choir"),
}


def family_of(instrument):
    """Coarse family of an instrument name; bass guitar is bass, not guitar."""
    name = instrument.lower()
    if "bass" in name and "double" not in name and "contra" not in name:
        return "bass"
    for family, words in INSTRUMENT_FAMILIES.items():
        if any(word in name for word in words):
            return family
    return "other"


def guitar_kind(instrument):
    name = instrument.lower()
    if family_of(name) != "guitar":
        return ""
    if "acoustic" in name or "classical" in name or "nylon" in name or "12 string" in name:
        return "acoustic"
    if "electric" in name:
        return "electric"
    if "steel" in name or "slide" in name or "resonator" in name or "dobro" in name:
        return "steel_or_slide"
    return "unspecified"


def norm(text):
    """Lower case, no accents, no punctuation, no bracketed remarks."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text)
    text = re.sub(r"\b(feat|featuring|ft)\b.*", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def version_words(text):
    """Version markers in a title or disambiguation: in brackets or after a dash."""
    low = (text or "").lower()
    remarks = " ".join(a or b for a, b in re.findall(r"\(([^)]*)\)|\[([^\]]*)\]", low))
    if " - " in low:
        remarks += " " + low.split(" - ", 1)[1]
    return {word for word in VERSION_WORDS if re.search(rf"\b{word}", remarks)}


def similar(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    return 1.0 if a == b else SequenceMatcher(None, a, b).ratio()


class Client:
    """Cached, rate-limited GET requests. ``fetch`` can be replaced in tests."""

    def __init__(self, cache, interval=1.1, fetch=None, offline=False):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.interval, self.offline = interval, offline
        self.fetch = fetch or self._http
        self.last = 0.0
        self.requests = 0

    def _http(self, url):
        try:
            import ssl  # noqa: F401  (a damaged Python install can lack it)
        except ImportError:
            return self._curl(url)
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def _curl(self, url):
        """The same request through the system's curl, for a Python without TLS."""
        done = subprocess.run(
            ["curl", "-sS", "--max-time", "30", "-A", USER_AGENT]
            + ["-H", "Accept: application/json", "-w", "\\n%{http_code}", url],
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if done.returncode:
            raise urllib.error.URLError(done.stderr.decode("utf-8", "replace").strip())
        body, _, code = done.stdout.decode("utf-8").rpartition("\n")
        if code != "200":
            raise urllib.error.HTTPError(url, int(code or 0), body[:200], None, None)
        return json.loads(body)

    def get(self, path, **query):
        query["fmt"] = "json"
        url = API + path + "?" + urllib.parse.urlencode(sorted(query.items()))
        file = self.cache / (hashlib.sha1(url.encode()).hexdigest() + ".json")
        if file.is_file():
            return json.loads(file.read_text(encoding="utf-8"))
        if self.offline:
            return {"url": url, "retrieved": None, "status": "not_cached", "body": None}
        delay = 2.0
        for _attempt in range(6):
            wait = self.interval - (time.monotonic() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            self.requests += 1
            try:
                body, status = self.fetch(url), "ok"
                break
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    body, status = None, "not_found"
                    break
                if error.code in (429, 503):
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
            except (urllib.error.URLError, TimeoutError):
                time.sleep(delay)
                delay *= 2
        else:
            raise RuntimeError(f"MusicBrainz did not answer: {url}")
        record = {
            "url": url,
            "retrieved": datetime.now(UTC).isoformat(timespec="seconds"),
            "status": status,
            "body": body,
        }
        file.write_text(json.dumps(record), encoding="utf-8")
        return record


# ------------------------------------------------------------------------- matching
def candidate(recording):
    """The parts of a MusicBrainz recording used for matching and reporting."""
    releases = recording.get("releases") or []
    return {
        "recording_mbid": recording.get("id", ""),
        "title": recording.get("title", ""),
        "disambiguation": recording.get("disambiguation", ""),
        "artist": "".join(
            c.get("name", "") + c.get("joinphrase", "") for c in recording.get("artist-credit", [])
        ),
        "artist_mbids": [
            c["artist"]["id"] for c in recording.get("artist-credit", []) if c.get("artist")
        ],
        "length_s": (recording.get("length") or 0) / 1000 or None,
        "isrcs": list(recording.get("isrcs") or []),
        "releases": [
            {
                "release_mbid": r.get("id", ""),
                "title": r.get("title", ""),
                "release_group_mbid": (r.get("release-group") or {}).get("id", ""),
                "secondary_types": (r.get("release-group") or {}).get("secondary-types", []),
                "date": r.get("date", ""),
            }
            for r in releases
        ],
        "video": bool(recording.get("video")),
    }


def score(row, found):
    """How well a candidate fits the file's own tags: 0 to 1, with the parts kept."""
    title = similar(row.get("title", ""), found["title"])
    artist = max(
        similar(row.get("artist", ""), found["artist"]),
        similar(row.get("album_artist", ""), found["artist"]),
    )
    album = max((similar(row.get("album", ""), r["title"]) for r in found["releases"]), default=0.0)
    try:
        wanted = float(row.get("duration_s") or 0)
    except ValueError:
        wanted = 0.0
    if wanted and found["length_s"]:
        gap = abs(wanted - found["length_s"])
        duration = 1.0 if gap <= 2 else 0.7 if gap <= 5 else 0.3 if gap <= 12 else 0.0
    else:
        gap, duration = None, 0.5
    mine = version_words(row.get("title", "")) | version_words(row.get("album", ""))
    theirs = version_words(found["title"]) | version_words("(" + found["disambiguation"] + ")")
    same_kind = mine == theirs
    total = 0.4 * title + 0.25 * artist + 0.15 * album + 0.2 * duration
    return {
        "total": round(total - (0.0 if same_kind else 0.15), 3),
        "before_version_penalty": round(total, 3),
        "title": round(title, 3),
        "artist": round(artist, 3),
        "album": round(album, 3),
        "duration_gap_s": None if gap is None else round(gap, 1),
        "same_kind": same_kind,
    }


def choose(row, candidates, method):
    """Pick a candidate or decline. Returns (chosen or None, confidence, status, ranked)."""
    ranked = sorted(
        ({**c, "fit": score(row, c)} for c in candidates),
        key=lambda c: c["fit"]["total"],
        reverse=True,
    )
    if not ranked:
        return None, "none", "unmatched", []
    best = ranked[0]
    fit = best["fit"]
    rivals = [
        c
        for c in ranked[1:]
        if c["recording_mbid"] != best["recording_mbid"]
        and best["fit"]["total"] - c["fit"]["total"] < 0.05
    ]
    # Rivals that differ only in which release they sit on are the same performance as far
    # as can be told here; rivals of another kind, or of clearly different length, are not.
    conflicting = [
        c
        for c in rivals
        if c["fit"]["same_kind"] != fit["same_kind"]
        or (
            c["length_s"]
            and best["length_s"]
            and abs(c["length_s"] - best["length_s"]) > 5
        )
    ]
    # Another kind of recording (live, remix, ...) that fits the tags just as well apart
    # from its version marker: the tags cannot tell the two apart.
    conflicting += [
        c
        for c in ranked[1:]
        if c not in conflicting
        and c["recording_mbid"] != best["recording_mbid"]
        and c["fit"]["same_kind"] != fit["same_kind"]
        and fit["before_version_penalty"] - c["fit"]["before_version_penalty"] < 0.05
        and c["fit"]["duration_gap_s"] is not None
        and c["fit"]["duration_gap_s"] <= 5
    ]
    if method in ("tag_mbid", "tag_track_mbid"):
        return best, "high", "matched", ranked
    gap = fit["duration_gap_s"]
    if method == "isrc":
        if len(ranked) == 1 or not conflicting:
            close = gap is None or gap <= 5
            return best, "high" if close else "medium", "matched", ranked
        return None, "low", "ambiguous", ranked
    strong = fit["title"] >= 0.92 and fit["artist"] >= 0.85 and fit["same_kind"]
    if strong and gap is not None and gap <= 3 and not conflicting:
        return best, "high" if fit["album"] >= 0.85 else "medium", "matched", ranked
    if strong and gap is not None and gap <= 8 and not conflicting:
        return best, "low", "matched", ranked
    if strong or fit["total"] >= 0.75:
        return None, "low", "ambiguous", ranked
    return None, "none", "unmatched", ranked


def lucene(text):
    return re.sub(r'([+\-&|!(){}\[\]^"~*?:\\/])', r"\\\1", text)


def identify(client, row):
    """Identity of one inventory row. Never raises for "not found"."""
    out = {
        "method": "",
        "confidence": "none",
        "status": "unmatched",
        "chosen": None,
        "alternatives": [],
        "sources": [],
    }

    def settle(candidates, method, source):
        chosen, confidence, status, ranked = choose(row, candidates, method)
        out["sources"].append(source)
        if status == "unmatched" and out["status"] != "unmatched":
            return False
        out.update(method=method, confidence=confidence, status=status, chosen=chosen)
        out["alternatives"] = [
            {k: c[k] for k in ("recording_mbid", "title", "disambiguation", "artist", "length_s", "fit")}
            for c in ranked
            if chosen is None or c["recording_mbid"] != chosen["recording_mbid"]
        ][:5]
        return status == "matched"

    mbid = (row.get("musicbrainz_recording_id") or "").strip()
    inc = "artist-credits+isrcs+releases+release-groups"
    if re.fullmatch(r"[0-9a-f-]{36}", mbid):
        record = client.get(f"recording/{mbid}", inc=inc)
        if record["body"]:
            if settle([candidate(record["body"])], "tag_mbid", record["url"]):
                return out
        else:  # some taggers store the id of the track on a release under this name
            record = client.get("recording", query=f"tid:{mbid}", limit="5")
            found = [candidate(r) for r in (record["body"] or {}).get("recordings", [])]
            if found and settle(found, "tag_track_mbid", record["url"]):
                return out
    for isrc in re.findall(r"[A-Z]{2}[A-Z0-9]{3}\d{7}", (row.get("isrc") or "").upper()):
        record = client.get(f"isrc/{isrc}", inc="artist-credits+releases+release-groups+isrcs")
        found = [candidate(r) for r in (record["body"] or {}).get("recordings", [])]
        if found and settle(found, "isrc", record["url"]):
            return out
    title, artist = row.get("title", "").strip(), row.get("artist", "").strip()
    if title and artist:
        query = f'recording:"{lucene(norm(title))}" AND artist:"{lucene(norm(artist))}"'
        record = client.get("recording", query=query, limit="15")
        found = [candidate(r) for r in (record["body"] or {}).get("recordings", [])]
        settle(found, "search", record["url"])
    return out


# -------------------------------------------------------------------------- credits
def credits_from(relations, scope, scope_id, source, retrieved):
    """Performer credits among ``relations`` of a recording, a release or an artist."""
    out = []
    for relation in relations or []:
        kind = relation.get("type", "")
        if relation.get("target-type") != "artist" or kind not in (
            "instrument",
            "performer",
            "vocal",
            "member of band",
            "performing orchestra",
        ):
            continue
        if kind == "member of band" and scope != "artist":
            continue
        performer = relation.get("artist") or {}
        attributes = [a for a in relation.get("attributes", []) if a]
        qualifiers = [a for a in attributes if a in ("guest", "solo", "additional", "original")]
        instruments = [a for a in attributes if a not in qualifiers]
        if kind == "vocal":
            instruments = ["vocals"]
        if kind in ("performer", "performing orchestra") and not instruments:
            instruments = [""]  # a performer credit that names no instrument
        if kind == "member of band" and not instruments:
            continue
        for instrument in instruments:
            out.append(
                {
                    "scope": scope,
                    "scope_mbid": scope_id,
                    "relation": kind,
                    "performer": performer.get("name", ""),
                    "performer_mbid": performer.get("id", ""),
                    "instrument": instrument,
                    "family": family_of(instrument) if instrument else "unspecified",
                    "guitar_kind": guitar_kind(instrument) if instrument else "",
                    "qualifiers": qualifiers,
                    "source": "musicbrainz",
                    "source_url": source,
                    "retrieved": retrieved,
                }
            )
    return out


RELEASE_INC = "artist-rels+recordings+recording-level-rels+release-groups+artist-credits"


def release_credits(client, release_mbid):
    """(release-level credits, {recording mbid: recording-level credits}) of one release."""
    record = client.get(f"release/{release_mbid}", inc=RELEASE_INC)
    body = record["body"]
    if not body:
        return [], {}
    url = f"https://musicbrainz.org/release/{release_mbid}"
    whole = credits_from(body.get("relations"), "release", release_mbid, url, record["retrieved"])
    per_recording = {}
    for medium in body.get("media", []):
        for track in medium.get("tracks", []):
            recording = track.get("recording") or {}
            if recording.get("id"):
                per_recording[recording["id"]] = credits_from(
                    recording.get("relations"),
                    "recording",
                    recording["id"],
                    f"https://musicbrainz.org/recording/{recording['id']}",
                    record["retrieved"],
                )
    return whole, per_recording


def recording_credits(client, recording_mbid):
    record = client.get(f"recording/{recording_mbid}", inc="artist-rels+tags+genres")
    body = record["body"] or {}
    url = f"https://musicbrainz.org/recording/{recording_mbid}"
    tags = sorted(
        {t["name"] for t in body.get("tags", []) + body.get("genres", []) if t.get("name")}
    )
    return (
        credits_from(body.get("relations"), "recording", recording_mbid, url, record["retrieved"]),
        tags,
    )


def artist_credits(client, artist_mbid):
    record = client.get(f"artist/{artist_mbid}", inc="artist-rels+tags+genres")
    body = record["body"] or {}
    url = f"https://musicbrainz.org/artist/{artist_mbid}"
    tags = sorted(
        {t["name"] for t in body.get("tags", []) + body.get("genres", []) if t.get("name")}
    )
    # For a group, "member of band" relations point at its members and carry instruments.
    members = [r for r in body.get("relations", []) if r.get("direction") == "backward"]
    return credits_from(members, "artist", artist_mbid, url, record["retrieved"]), tags


def pick_release(row, chosen):
    """The release of a matched recording that fits the file's album tag, if one does. The
    tagged release id wins; a release is otherwise taken only on a close title match."""
    tagged = (row.get("musicbrainz_album_id") or "").strip()
    releases = chosen["releases"]
    for release in releases:
        if tagged and release["release_mbid"] == tagged:
            return release, "tag"
    if re.fullmatch(r"[0-9a-f-]{36}", tagged):
        return {"release_mbid": tagged, "title": row.get("album", ""), "release_group_mbid": ""}, "tag"
    ranked = sorted(releases, key=lambda r: similar(row.get("album", ""), r["title"]), reverse=True)
    if ranked and similar(row.get("album", ""), ranked[0]["title"]) >= 0.85:
        return ranked[0], "album_title"
    return None, ""
