"""The excerpt dataset for labelling vocal delivery: which recordings, which ten seconds of
each, and which artists are held out.

Everything here is decided before anybody listens, is written once to ``dataset.json`` and
then only ever grows:

* **Artists are split first.** An artist (a corpus group: one artist, or artists the corpus
  merged because they share recordings) is *evaluation* when it owns a reference track or
  when a hash of its name falls in the lowest quarter; everything else is *development*.
  The rule needs no other artist, so adding recordings later never moves anybody.
* **Excerpts are placed without knowing any label.** Each artist contributes in turn,
  recording by recording, so none dominates. Placements rotate through four methods:
  evenly spaced positions, seeded random positions, passages the existing vocal analysis
  thinks likely to hold a voice, and passages it thinks likely to have none. The analysis
  only proposes where to listen; the label is whatever the owner hears.
* **Nothing is copied.** An excerpt is a recording and two times. Playback reads those ten
  seconds from the original file, which is never written.

Excerpt identifiers (``VOCAL-0001`` ...) are given in a seeded random order, so the number
says nothing about artist, recording, method or split, and the presentation order is that
same saved shuffle.
"""

import hashlib
import json
import math
import os
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

FORMAT = "gmv-vocal-delivery-labels"
VERSION = 1
SEED = "gmv-vocal-delivery-a3"
EXCERPT_S = 10.0
EXCERPT_RANGE_S = (8.0, 12.0)  # what counts as a valid excerpt length
HELD_OUT_SHARE = 0.25  # the share of artists the hash rule holds out
TARGET = 96
PER_ARTIST = 4
PER_RECORDING = 2
METHODS = ("even", "likely_vocal", "random", "likely_instrumental")
EVEN = (0.2, 0.5, 0.8, 0.35, 0.65)  # fractions of a recording the even method visits
QUIET_DBFS = -45.0  # an excerpt quieter than this on average is not offered
EDGE_S = 1.0  # excerpts keep this far from the ends of a recording
GAP_S = 2.0  # two excerpts of one recording are at least this far apart
AUDIO_SUFFIXES = (".mp3", ".flac", ".wav", ".ogg", ".opus", ".aiff", ".aif", ".m4a")


class DatasetError(Exception):
    """A problem the owner or the developer has to resolve; the message says what."""


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def held_out(artist, seed=SEED):
    """Whether the hash rule puts ``artist`` in evaluation (reference artists aside)."""
    digest = hashlib.sha256(f"{seed}:{artist}".encode()).hexdigest()
    return int(digest[:8], 16) / 2**32 < HELD_OUT_SHARE


def side_of(artist, references=(), seed=SEED):
    return "evaluation" if artist in references or held_out(artist, seed) else "development"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(path):
    """What a recording is, from its header and bytes: raises ``DatasetError`` for a file
    that is missing, unreadable or not audio."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError("not found")
    try:
        info = sf.info(str(path))
    except Exception as exc:  # not audio, or a format libsndfile cannot read
        raise DatasetError(f"not a readable audio file ({type(exc).__name__})") from exc
    if info.samplerate <= 0 or info.frames <= 0 or info.channels <= 0:
        raise DatasetError("the file holds no audio")
    return {
        "size": path.stat().st_size,
        "duration_s": info.frames / info.samplerate,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
    }


def read_excerpt(path, start_s, end_s):
    """The samples of one excerpt (frames, channels) as float32, and the sample rate. Reads
    only those seconds; the file is never written."""
    with sf.SoundFile(str(path)) as file:
        rate = file.samplerate
        first = int(round(start_s * rate))
        count = int(round((end_s - start_s) * rate))
        if first < 0 or first + count > file.frames:
            raise DatasetError("the excerpt lies outside the recording")
        file.seek(first)
        audio = file.read(count, dtype="float32", always_2d=True)
    if len(audio) != count:
        raise DatasetError("the recording ended early")
    return audio, rate


def level_dbfs(audio):
    power = float(np.mean(np.square(audio, dtype=np.float64))) if audio.size else 0.0
    return 10.0 * math.log10(max(power, 1e-12))


def check_excerpt(excerpt, recording=None):
    """Problems with one excerpt's definition (a list of sentences; empty when fine)."""
    problems = []
    length = excerpt["end_s"] - excerpt["start_s"]
    if not EXCERPT_RANGE_S[0] <= length <= EXCERPT_RANGE_S[1]:
        problems.append(f"{excerpt['id']}: {length:.2f} s long, outside {EXCERPT_RANGE_S}")
    if excerpt["start_s"] < 0:
        problems.append(f"{excerpt['id']}: starts before the recording")
    if recording is not None and excerpt["end_s"] > recording["duration_s"] + 1e-6:
        problems.append(f"{excerpt['id']}: ends after the recording")
    return problems


# -- the recordings -----------------------------------------------------------------------


def corpus_recordings(manifest, paths_map, replace=()):
    """Recordings of a registered corpus: (path, artist, registered sha256, corpus id).

    ``manifest`` is a corpus manifest (``recording_id``, ``group_id``, ``sha256``) and
    ``paths_map`` its local path map; ``replace`` turns the paths written on another
    system into this one's, as (prefix, replacement) pairs (``E:\\`` to a mount point)."""
    manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    located = json.loads(Path(paths_map).read_text(encoding="utf-8"))["recordings"]
    found = []
    for record in manifest["recordings"]:
        path = located.get(record["recording_id"])
        if not path:
            continue
        for old, new in replace:
            if path.startswith(old):
                path = new + path[len(old) :]
                break
        path = path.replace("\\", "/") if replace else path
        found.append((path, record["group_id"], record.get("sha256"), record["recording_id"]))
    return found


def gather(candidates, references=None, checksums=True, progress=None):
    """Check every candidate recording. ``candidates`` are (path, artist, expected sha256 or
    None, source); ``references`` maps path -> a short reference name. Returns (recordings
    by id, problems by path). Ids are ``REC-0001`` ... in the order given."""
    recordings, problems = {}, {}
    references = references or {}
    for number, (path, artist, expected, source) in enumerate(candidates, 1):
        if progress:
            progress(f"checking recording {number}/{len(candidates)}")
        try:
            facts = describe(path)
            if facts["duration_s"] < EXCERPT_S + 2 * EDGE_S:
                raise DatasetError(f"only {facts['duration_s']:.1f} s long")
            digest = sha256(path) if checksums or expected else None
            if expected and digest != expected:
                raise DatasetError("its contents differ from the registered recording")
        except DatasetError as exc:
            problems[str(path)] = str(exc)
            continue
        identity = f"REC-{len(recordings) + 1:04d}"
        recordings[identity] = {
            "path": str(path),
            "artist": artist,
            "sha256": digest,
            "source": source,
            "reference": references.get(str(path)),
            **facts,
        }
    return recordings, problems


# -- placing excerpts ---------------------------------------------------------------------


def free(start, taken, length=EXCERPT_S):
    return all(start + length + GAP_S <= a or start >= b + GAP_S for a, b in taken)


def vocal_scores(path, activity_model=None):
    """Per second of the recording, how likely the existing analysis thinks a voice is:
    the activity model's vocal judgement when its weights are given and can be used at the
    file's rate, otherwise the vocal analyzer's centre level counted in full where it has a
    pitch. Returns (seconds, score) arrays, or None when the analysis cannot run."""
    from ..gmvpass import run
    from ..lead import Rows

    try:
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        _, rows, frames = run(audio, rate, activity_model)
    except Exception:
        return None
    lines = Rows(rows, frames)
    if not len(lines.time):
        return None
    if lines.model:
        score = lines.vocal.astype(float)
    else:
        score = lines.level * np.where(np.isnan(lines.pitch), 0.35, 1.0)
    score = np.where(lines.music, score, np.nan)
    seconds = np.floor(lines.time).astype(int)
    total = np.bincount(seconds, weights=np.nan_to_num(score))
    count = np.bincount(seconds, weights=np.isfinite(score).astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.arange(len(total)), np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _scores_job(job):
    path, model = job
    return path, vocal_scores(path, model)


def precompute_scores(paths, model=None, processes=1, progress=None):
    """Vocal scores of many recordings, in parallel processes: path -> scores (or None)."""
    paths = sorted(set(paths))
    if processes <= 1:
        return {path: vocal_scores(path, model) for path in paths}
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    limits = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    kept = {name: os.environ.get(name) for name in limits}
    os.environ.update(dict.fromkeys(limits, "1"))
    try:
        found = {}
        with ProcessPoolExecutor(processes, mp_context=get_context("spawn")) as pool:
            for count, (path, scores) in enumerate(pool.map(_scores_job, [(p, model) for p in paths]), 1):
                found[path] = scores
                if progress and (count % 10 == 0 or count == len(paths)):
                    progress(f"analysed {count}/{len(paths)} recordings for likely passages")
        return found
    finally:
        for name, value in kept.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def candidates(duration, length=EXCERPT_S, step=1.0):
    last = duration - length - EDGE_S
    if last < EDGE_S:
        return []
    return [round(EDGE_S + step * k, 3) for k in range(int((last - EDGE_S) / step) + 1)]


def place(method, recording, taken, rng, scores=None, visits=0):
    """Ordered start times to try for one excerpt by ``method``."""
    duration = recording["duration_s"]
    starts = [s for s in candidates(duration) if free(s, taken)]
    if not starts:
        return []
    if method == "even":
        # The next of a fixed set of fractions of the recording, nearest free start first.
        span = duration - EXCERPT_S - 2 * EDGE_S
        wanted = EDGE_S + EVEN[visits % len(EVEN)] * span
        return sorted(starts, key=lambda s: (abs(s - wanted), s))
    if method == "random" or scores is None:
        rng.shuffle(starts)
        return starts
    seconds, score = scores
    def window(start):
        part = score[int(start) : int(start + EXCERPT_S)]
        part = part[np.isfinite(part)]
        return float(part.mean()) if len(part) else np.nan
    rated = [(window(s), s) for s in starts]
    rated = [(v, s) for v, s in rated if np.isfinite(v)]
    if not rated:
        rng.shuffle(starts)
        return starts
    rated.sort(reverse=(method == "likely_vocal"))
    quarter = rated[: max(1, len(rated) // 4)]
    rng.shuffle(quarter)  # a random choice among the most (or least) likely, not the extreme
    return [s for _, s in quarter] + [s for _, s in rated[len(quarter) :]]


def select(recordings, target=TARGET, scores=None, seed=SEED, progress=None):
    """Place up to ``target`` excerpts. ``scores`` is a callable path -> vocal scores (or
    None), asked only for recordings that get a ``likely_*`` placement. Returns the
    excerpts, without identifiers, in the order they were placed."""
    rng = random.Random(seed)
    by_artist = {}
    for identity, recording in sorted(recordings.items()):
        by_artist.setdefault(recording["artist"], []).append(identity)
    references = {}
    for artist, owned in by_artist.items():
        # Reference tracks are offered first, until they have their excerpts; the artist's
        # other recordings then take turns in a seeded order.
        random.Random(f"{seed}:{artist}").shuffle(owned)
        references[artist] = [r for r in owned if recordings[r]["reference"]]
        by_artist[artist] = [r for r in owned if not recordings[r]["reference"]]
    order = sorted(by_artist, key=lambda a: hashlib.sha256(f"{seed}:order:{a}".encode()).hexdigest())
    picked, taken, used, visits, cache = [], {}, {}, {}, {}
    rounds = 0
    while len(picked) < target and rounds < PER_ARTIST:
        progressed = False
        for artist in order:
            if len(picked) >= target:
                break
            rest = by_artist[artist]
            turn = rounds % len(rest) if rest else 0
            for identity in references[artist] + rest[turn:] + rest[:turn]:
                if used.get(identity, 0) >= PER_RECORDING:
                    continue
                recording = recordings[identity]
                method = METHODS[len(picked) % len(METHODS)]
                found = None
                if method.startswith("likely") and scores is not None:
                    if identity not in cache:
                        if progress:
                            progress(f"analysing recording {identity} for likely passages")
                        cache[identity] = scores(recording["path"])
                for start in place(method, recording, taken.get(identity, []), rng, cache.get(identity), visits.get(identity, 0)):
                    try:
                        audio, _ = read_excerpt(recording["path"], start, start + EXCERPT_S)
                    except DatasetError:
                        continue
                    level = level_dbfs(audio)
                    if level >= QUIET_DBFS:
                        found = (start, level)
                        break
                if found is None:
                    continue
                start, level = found
                actual = method if not method.startswith("likely") or cache.get(identity) else "random"
                picked.append(
                    {
                        "recording": identity,
                        "start_s": start,
                        "end_s": round(start + EXCERPT_S, 3),
                        "selection": actual,
                        "level_dbfs": round(level, 1),
                    }
                )
                taken.setdefault(identity, []).append((start, start + EXCERPT_S))
                used[identity] = used.get(identity, 0) + 1
                visits[identity] = visits.get(identity, 0) + 1
                progressed = True
                break
        rounds += 1
        if not progressed:
            break
    return picked


def artist_key(name):
    """The key of an artist as the corpus writes it: ``artist-`` and a lower-case slug."""
    slug = "".join(c if c.isalnum() else "-" for c in str(name).lower()).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug if slug.startswith(("artist-", "merged-")) else f"artist-{slug}"


def artist_name(key):
    """A readable name for an artist key (for the owner's own choices only)."""
    name = key.split("-", 1)[1] if key.startswith(("artist-", "merged-")) else key
    return name.replace("-", " ").replace(".", " / ").title()


def number(excerpts, first=1, seed=SEED, salt="initial"):
    """Give excerpts identifiers in a seeded random order; returns them in that order."""
    shuffled = list(excerpts)
    random.Random(f"{seed}:ids:{salt}").shuffle(shuffled)
    for offset, excerpt in enumerate(shuffled):
        excerpt["id"] = f"VOCAL-{first + offset:04d}"
    return shuffled


# -- the dataset file ---------------------------------------------------------------------


def build(recordings, excerpts, references, seed=SEED, notes=()):
    artists = {}
    for recording in recordings.values():
        artist = recording["artist"]
        side = side_of(artist, references, seed)
        artists[artist] = {
            "split": side,
            "reason": "owns a reference track" if artist in references else "hash rule",
        }
    numbered = number(excerpts, seed=seed)
    used = {e["recording"] for e in numbered}
    return {
        "format": FORMAT,
        "version": VERSION,
        "created_utc": now(),
        "seed": seed,
        "excerpt_s": EXCERPT_S,
        "split_rule": (
            f"An artist is held out for evaluation if it owns a reference track, or if the "
            f"first 8 hex digits of sha256('{seed}:' + artist) divided by 2^32 are below "
            f"{HELD_OUT_SHARE}. Fixed before any label; never changed afterwards."
        ),
        "reference_artists": sorted(references),
        "artists": dict(sorted(artists.items())),
        "recordings": {k: v for k, v in recordings.items() if k in used},
        "excerpts": sorted(numbered, key=lambda e: e["id"]),
        "order": [e["id"] for e in numbered],
        "notes": list(notes),
    }


def split_of(dataset, excerpt):
    return dataset["artists"][dataset["recordings"][excerpt["recording"]]["artist"]]["split"]


def validate(dataset):
    """Every integrity rule the dataset must keep; returns problems (empty when sound)."""
    problems = []
    if dataset.get("format") != FORMAT or dataset.get("version") != VERSION:
        problems.append("not a vocal-delivery dataset of a known version")
        return problems
    ids = [e["id"] for e in dataset["excerpts"]]
    if len(set(ids)) != len(ids):
        problems.append("two excerpts share an identifier")
    if sorted(dataset["order"]) != sorted(ids):
        problems.append("the presentation order does not list every excerpt exactly once")
    for excerpt in dataset["excerpts"]:
        recording = dataset["recordings"].get(excerpt["recording"])
        if recording is None:
            problems.append(f"{excerpt['id']}: its recording is not in the dataset")
            continue
        problems += check_excerpt(excerpt, recording)
        if recording["artist"] not in dataset["artists"]:
            problems.append(f"{excerpt['id']}: its artist has no split")
    spans = {}
    for excerpt in dataset["excerpts"]:
        spans.setdefault(excerpt["recording"], []).append((excerpt["start_s"], excerpt["end_s"], excerpt["id"]))
    for found in spans.values():
        found.sort()
        for (_a0, a1, a), (b0, _b1, b) in zip(found, found[1:], strict=False):
            if b0 < a1:
                problems.append(f"{a} and {b} overlap")
    for artist in dataset.get("reference_artists", []):
        if dataset["artists"].get(artist, {}).get("split") != "evaluation":
            problems.append(f"reference artist {artist} is not held out")
    for recording in dataset["recordings"].values():
        if recording.get("reference") and dataset["artists"][recording["artist"]]["split"] != "evaluation":
            problems.append("a reference track is not held out")
    return problems


def write(folder, dataset):
    """Write ``dataset.json`` atomically (a crash leaves the old file or the new one)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    problems = validate(dataset)
    if problems:
        raise DatasetError("; ".join(problems))
    target = folder / "dataset.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(dataset, indent=1, ensure_ascii=False), encoding="utf-8")
    with temporary.open("rb") as file:
        os.fsync(file.fileno())
    temporary.replace(target)
    return target


def load(folder):
    path = Path(folder) / "dataset.json"
    if not path.is_file():
        raise DatasetError("no dataset has been prepared in this folder yet")
    try:
        dataset = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise DatasetError("the dataset file is damaged") from exc
    problems = validate(dataset)
    if problems:
        raise DatasetError("the dataset failed its integrity checks: " + "; ".join(problems[:5]))
    return dataset


def prepare(folder, recordings, references, target=TARGET, scores=None, seed=SEED, notes=(), progress=None):
    """Build and write a new dataset. Refused once anything has been labelled there: the
    split and the excerpts are then fixed, and only ``add_recordings`` may extend them."""
    from .store import Labels

    if Labels(folder).events():
        raise DatasetError(
            "labels already exist in this folder; the dataset is fixed. Add recordings "
            "from the labelling window instead of preparing it again"
        )
    excerpts = select(recordings, target, scores, seed, progress)
    dataset = build(recordings, excerpts, set(references), seed, notes)
    write(folder, dataset)
    return dataset


def add_recordings(folder, dataset, files, artist, per_recording=PER_RECORDING):
    """Add recordings by one artist, chosen by the owner, and excerpts from them.

    Existing artists keep their side; a new artist gets the side the hash rule gives it, so
    nothing labelled so far can move between development and evaluation. Excerpts are placed
    by the even and random methods only (no analysis) and numbered after the existing ones,
    in a seeded random order, and appended to the presentation order. Returns (excerpts
    added, problems by file)."""
    artist = " ".join(str(artist).split())
    if not artist.strip("-"):
        raise DatasetError("say who the artist is")
    key = artist if artist in dataset["artists"] else artist_key(artist)
    known = {r["path"] for r in dataset["recordings"].values()}
    hashes = {r["sha256"] for r in dataset["recordings"].values() if r.get("sha256")}
    candidates_ = []
    problems = {}
    for file in files:
        file = str(Path(file).expanduser().resolve())
        if file in known:
            problems[file] = "already in the dataset"
            continue
        candidates_.append((file, key, None, "added by the owner"))
    found, more = gather(candidates_, checksums=True)
    problems.update(more)
    for identity, recording in list(found.items()):
        if recording["sha256"] in hashes:
            problems[recording["path"]] = "the same audio is already in the dataset"
            del found[identity]
    if not found:
        return [], problems
    renumbered = {}
    start = 1 + max((int(k.split("-")[1]) for k in dataset["recordings"]), default=0)
    for offset, (_identity, recording) in enumerate(sorted(found.items())):
        renumbered[f"REC-{start + offset:04d}"] = recording
    rng = random.Random(f"{dataset['seed']}:add:{len(dataset['excerpts'])}")
    placed = []
    for identity, recording in renumbered.items():
        taken = []
        for pick in range(per_recording):
            method = ("even", "random")[pick % 2]
            for begin in place(method, recording, taken, rng, None, pick):
                try:
                    audio, _ = read_excerpt(recording["path"], begin, begin + EXCERPT_S)
                except DatasetError:
                    continue
                level = level_dbfs(audio)
                if level >= QUIET_DBFS:
                    placed.append({"recording": identity, "start_s": begin, "end_s": round(begin + EXCERPT_S, 3),
                                   "selection": method, "level_dbfs": round(level, 1)})
                    taken.append((begin, begin + EXCERPT_S))
                    break
    if not placed:
        return [], {**problems, **{r["path"]: "no passage loud enough to label" for r in renumbered.values()}}
    first = 1 + max(int(e["id"].split("-")[1]) for e in dataset["excerpts"])
    numbered = number(placed, first, dataset["seed"], salt=f"add:{first}")
    updated = json.loads(json.dumps(dataset))
    if key not in updated["artists"]:
        updated["artists"][key] = {
            "split": side_of(key, (), dataset["seed"]),
            "reason": "hash rule (added after the start)",
        }
    used = {e["recording"] for e in numbered}
    updated["recordings"].update({k: v for k, v in renumbered.items() if k in used})
    updated["excerpts"] = sorted(updated["excerpts"] + numbered, key=lambda e: e["id"])
    updated["order"] = updated["order"] + [e["id"] for e in numbered]
    updated.setdefault("notes", []).append(f"{now()}: {len(numbered)} excerpts added from {len(used)} recordings")
    write(folder, updated)
    dataset.clear()
    dataset.update(updated)
    return numbered, problems


def summary(dataset):
    """Counts for reports: recordings, artists and excerpts on each side."""
    out = {"recordings": len(dataset["recordings"]), "excerpts": len(dataset["excerpts"])}
    for side in ("development", "evaluation"):
        artists = [a for a, v in dataset["artists"].items() if v["split"] == side]
        excerpts = [e for e in dataset["excerpts"] if split_of(dataset, e) == side]
        out[side] = {
            "artists": len({dataset["recordings"][e["recording"]]["artist"] for e in excerpts}),
            "artists_assigned": len(artists),
            "excerpts": len(excerpts),
            "recordings": len({e["recording"] for e in excerpts}),
        }
    out["methods"] = {m: sum(e["selection"] == m for e in dataset["excerpts"]) for m in METHODS}
    return out
