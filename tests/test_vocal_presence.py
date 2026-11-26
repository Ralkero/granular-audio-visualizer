"""Phase A5.1: presence annotation, the protected dataset, the annotation window.
Generated audio and generated projects only; no real answer file is ever used."""

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from gmv.character.research.delivery import data as delivery_data
from gmv.character.research.labeling import dataset as labelling
from gmv.character.research.labeling import store as a3_store
from gmv.character.research.presence import identity, loaders, project, schema, selection, split
from gmv.character.research.presence import prepare as preparing
from gmv.character.research.presence.store import Store

RATE = 22050
POISON = "POISON-HELD-OUT"
CATEGORIES = ("cleans", "high_screams", "low_screams", "gutturals", "sing_screams")


# -- a generated world: an A3 project, recordings and an inventory -----------------------------


def sound(seconds, seed, voice=True):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    mono = 0.05 * rng.standard_normal(len(t))
    if voice:
        mono += 0.2 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 0.25 * t) > 0)
    return np.stack([mono, mono], axis=1).astype(np.float32)


def pick_artists(development=6, evaluation=2):
    dev, held = [], []
    for number in range(300):
        key = f"artist-world-band-{number:02d}"
        side = labelling.side_of(key)
        if side == "development" and len(dev) < development:
            dev.append(key)
        elif side == "evaluation" and len(held) < evaluation:
            held.append(key)
    return dev, held


def write_wav(path, seconds, seed, voice=True):
    sf.write(path, sound(seconds, seed, voice), RATE)
    return {"sha256": labelling.sha256(path), "size": path.stat().st_size}


def build_world(root):
    root = Path(root)
    music = root / "music"
    music.mkdir(parents=True)
    a3 = root / "local-evidence" / "sonic-character" / "vocal-labeling"
    a3.mkdir(parents=True)
    dev, held = pick_artists()
    recordings, excerpts, inventory = {}, [], []
    for a, artist in enumerate(dev + held):
        for r in range(2):
            identity_ = f"rec-{a:02d}-{r}"
            path = music / f"{identity_}.wav"
            facts = write_wav(path, 24.0, a * 10 + r)
            recordings[identity_] = {"path": str(path), "artist": artist, "source": "test", "reference": False,
                                     "duration_s": 24.0, "sample_rate": RATE, "channels": 2, "frames": int(24 * RATE), **facts}
            for start in (1.0, 12.0):
                excerpts.append({"recording": identity_, "start_s": start, "end_s": start + 10.0, "selection": "even", "level_dbfs": -20.0})
            # Credits spelled differently from the key: case, punctuation, a leading "The".
            spoken = labelling.artist_name(artist)
            credit = ("The " + spoken.upper()) if r == 1 else spoken
            inventory.append({"sha256": facts["sha256"], "artist": credit, "album_artist": "Various Artists", "mb_artist": "",
                              "mb_artist_ids": "", "title": f"Track title {a}.{r}", "leakage_group": "", "same_audio_group": "",
                              "inventory_version_group": "", "instrumental_or_vocal": "Vocal", "content_category": "band",
                              "genre": "Metalcore", "duration_s": "24.0", "file_path": str(path)})
    dataset = labelling.build(recordings, excerpts, set())
    labelling.write(a3, dataset)
    (a3 / "titles.json").write_text(json.dumps({k: {"title": f"Melody number {n}"} for n, k in enumerate(recordings)}))
    side = delivery_data.sides(dataset)
    by_recording = {}
    for e in dataset["excerpts"]:
        by_recording.setdefault(e["recording"], []).append(e["id"])
    labels = a3_store.Labels(a3)
    lines = []
    for n, e in enumerate(dataset["excerpts"]):
        if side[e["id"]] == "evaluation":
            lines.append(json.dumps({"time": "t", "excerpt": e["id"], "event": "label", "value": POISON, "scheme": 2}))
        elif n % 7 == 6:
            labels.skip(e["id"], "no_vocals")
        else:
            labels.label(e["id"], CATEGORIES[n % len(CATEGORIES)])
    with (a3 / a3_store.LABELS).open("a") as file:
        file.write("\n".join(lines) + "\n")
    leak = sorted(k for k in recordings if recordings[k]["artist"] == dev[3])[0]
    note = a3.parent / "expansion-audit"
    note.mkdir()
    ids = by_recording[leak]
    (note / "leakage-watch.md").write_text(f"- **{ids[0]} and {ids[1]}** come from \"Melody number {sorted(recordings).index(leak)}\".\n")

    def extra(name, artist, instrumental, category, genre, title, seconds=40.0, **fields):
        path = music / f"{name}.wav"
        facts = write_wav(path, seconds, hash(name) % 1000, voice=not instrumental)
        row = {"sha256": facts["sha256"], "artist": artist, "album_artist": "", "mb_artist": "", "mb_artist_ids": "",
               "title": title, "leakage_group": "", "same_audio_group": "", "inventory_version_group": "",
               "instrumental_or_vocal": "Instrumental" if instrumental else "Vocal", "content_category": category,
               "genre": genre, "duration_s": str(seconds), "file_path": str(path)}
        row.update(fields)
        inventory.append(row)
        return row

    spoken = {k: labelling.artist_name(k) for k in dev + held}
    extra("soundtrack-1", spoken[dev[0]], True, "soundtrack", "Ambient", "Calm Theme")
    extra("soundtrack-2", spoken[dev[1]], True, "soundtrack", "", "Battle Theme")
    extra("instrumental-band", spoken[dev[2]], True, "band", "Progressive Metal", "Shred Instrumental")
    extra("vocal-band", spoken[dev[4]], False, "band", "Metalcore", "Opening Song")
    extra("fresh-1", "Totally New Band", False, "band", "Rock", "Brand New Song")
    extra("guest", f"{spoken[dev[5]]} feat. Guest Singer", False, "band", "Metalcore", "Guest Track")
    extra("guest-own", "Guest Singer", False, "band", "Metalcore", "Guest's Own Song")
    inventory_path = root / "inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    return {"root": root, "a3": a3, "dataset": dataset, "dev": dev, "held": held, "inventory": inventory_path,
            "rows": inventory, "leak": ids, "spoken": spoken}


def a3_prints(a3):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(a3).iterdir()) if p.is_file()}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("world"))


@pytest.fixture
def prepared(world, tmp_path):
    root = tmp_path / "presence"
    summary = preparing.prepare(root, world["a3"], world["inventory"], say=lambda text: None)
    return root, summary


# -- the schema ---------------------------------------------------------------------------------


def test_intervals_must_lie_inside_the_excerpt_and_run_forwards():
    assert schema.check_interval(1.23456, 2.0, 10.0) == (1.235, 2.0)
    for start, end in ((5, 4), (-0.1, 1), (9.5, 10.2), (3.0, 3.01), ("a", 2), (float("nan"), 2)):
        with pytest.raises(schema.SchemaError):
            schema.check_interval(start, end, 10.0)


def test_overlapping_voices_become_one_union_and_targets_follow_it():
    store = Store(Path(pytest.importorskip("tempfile").mkdtemp()), {"X": 10.0})
    store.add_interval("X", "confirmed", 1.0, 3.0)
    store.add_interval("X", "confirmed", 2.5, 4.0)
    store.add_interval("X", "confirmed", 4.0, 4.5)
    store.add_interval("X", "uncertain", 3.5, 6.0)
    store.set_review("X", "complete")
    state = store.of("X")
    assert schema.spans(state, "confirmed") == [(1.0, 4.5)]
    target, known = schema.targets(state, np.array([0.5, 2.0, 4.2, 5.0, 8.0]))
    assert target.tolist() == [0, 1, 1, 0, 0] and known.tolist() == [True, True, True, False, True]


def test_statuses_negative_needs_a_completed_review_with_no_marks():
    durations = {"A": 10.0, "B": 10.0, "C": 10.0, "D": 10.0, "E": 10.0}
    store = Store(Path(pytest.importorskip("tempfile").mkdtemp()), durations)
    store.set_review("A", "complete")  # explicit No vocals
    store.add_interval("B", "confirmed", 1, 2)  # marked but not completed
    store.add_interval("C", "uncertain", 1, 2)
    store.set_review("C", "complete")
    store.set_review("D", "unusable", "clipped audio")
    store.set_tags("E", ["guitar_lead_solo"])  # a guitar-solo tag alone is not "no vocals"
    states = store.state()
    assert [schema.status(states.get(k)) for k in "ABCDE"] == ["negative", "incomplete", "uncertain", "unusable", "incomplete"]
    assert not schema.targets(states["B"], [1.5])[1].any() and not schema.targets(states["D"], [1.5])[1].any()
    with pytest.raises(schema.SchemaError):
        store.set_review("A", "unusable", "")


def test_several_confounder_tags_and_unknown_tags_refused():
    store = Store(Path(pytest.importorskip("tempfile").mkdtemp()), {"X": 10.0})
    store.set_tags("X", ["synth_keyboard", "layered_vocals", "synth_keyboard", "dense_mix"])
    assert store.of("X")["tags"] == ["dense_mix", "layered_vocals", "synth_keyboard"]
    with pytest.raises(schema.SchemaError):
        store.set_tags("X", ["vocals_present"])


def test_edits_and_removals_are_new_revisions_and_the_chain_detects_tampering(tmp_path):
    store = Store(tmp_path, {"X": 10.0})
    first = store.add_interval("X", "confirmed", 1.0, 2.0)["id"]
    store.edit_interval("X", first, end=2.5, kind="uncertain")
    second = store.add_interval("X", "confirmed", 5.0, 6.0)["id"]
    store.remove_interval("X", second)
    again = Store(tmp_path, {"X": 10.0})
    assert again.intact and [e["event"] for e in again.history("X")] == ["interval_add", "interval_edit", "interval_add", "interval_remove"]
    assert again.of("X")["intervals"] == {first: {"kind": "uncertain", "start": 1.0, "end": 2.5, "note": "", "revision": 2}}
    lines = (tmp_path / "annotations.jsonl").read_text().splitlines()
    (tmp_path / "annotations.jsonl").write_text("\n".join([lines[1], lines[0], *lines[2:]]) + "\n")  # reordered
    assert not Store(tmp_path, {"X": 10.0}).intact


def test_a_line_cut_off_by_a_crash_is_skipped_and_the_chain_continues(tmp_path):
    store = Store(tmp_path, {"X": 10.0})
    store.add_interval("X", "confirmed", 1.0, 2.0)
    with (tmp_path / "annotations.jsonl").open("a") as file:
        file.write('{"excerpt": "X", "event": "interval_ad')  # the crash
    store = Store(tmp_path, {"X": 10.0})
    assert store.damaged == 1 and store.intact
    store.add_interval("X", "confirmed", 3.0, 4.0)
    again = Store(tmp_path, {"X": 10.0})
    assert again.intact and again.damaged == 1 and len(again.of("X")["intervals"]) == 2


def test_events_for_unknown_excerpts_and_bad_kinds_are_never_written(tmp_path):
    store = Store(tmp_path, {"X": 10.0})
    for call in (lambda: store.add_interval("Y", "confirmed", 1, 2), lambda: store.add_interval("X", "maybe", 1, 2),
                 lambda: store.edit_interval("X", "X#9", start=1), lambda: store.remove_interval("X", "X#9")):
        with pytest.raises(schema.SchemaError):
            call()
    assert not (tmp_path / "annotations.jsonl").exists()


# -- identity, registry and selection -------------------------------------------------------------


def test_spellings_aliases_featured_credits_and_duplicates_join_groups():
    rows = [
        {"sha256": "a", "artist": "Sample Band Of Examples", "title": "x"},
        {"sha256": "b", "artist": "sample band of examples", "title": "y"},
        {"sha256": "c", "artist": "The Example Band", "title": "z"},
        {"sha256": "d", "artist": "Example Band, Featured Person", "title": "w"},
        {"sha256": "e", "artist": "Solo Project", "title": "Song (feat. Featured Person)"},
        {"sha256": "f", "artist": "Other Name", "title": "v"},
        {"sha256": "g", "artist": "Unrelated", "title": "u", "same_audio_group": "dup-1"},
        {"sha256": "h", "artist": "Unrelated Too", "title": "t", "same_audio_group": "dup-1"},
        {"sha256": "i", "artist": "Various Artists", "album_artist": "Various Artists", "title": "s"},
        {"sha256": "j", "artist": "Various Artists", "title": "r"},
        {"sha256": "k", "artist": "Mononym", "mb_artist_ids": "mbid-1", "title": "q"},
        {"sha256": "l", "artist": "Mononym (Live)", "mb_artist_ids": "mbid-1", "title": "p"},
    ]
    aliases = {identity.normalise("Other Name"): identity.normalise("Solo Project")}
    found = identity.build(rows, aliases=aliases)["recording_group"]
    assert found["a"] == found["b"]  # capitalisation never makes a new group
    assert found["c"] == found["d"] == found["e"] == found["f"]  # featured person, then alias
    assert found["g"] == found["h"]  # duplicate audio
    assert found["i"] != found["j"]  # "Various Artists" links nothing
    assert found["k"] == found["l"]  # MusicBrainz identifier


def test_groups_never_move_and_conflicts_stop_everything():
    groups = {"g1": {"nodes": ["a3:dev-a", "rec:1"]}, "g2": {"nodes": ["a3:dev-b", "rec:2"]}, "g3": {"nodes": ["rec:3", "name:new"]}}
    first = split.assign(groups, {}, consumed=set(), development_keys={"dev-a", "dev-b"})
    assert first["entries"]["g3"]["partition"] == "reserved"
    assert {first["entries"][k]["partition"] for k in ("g1", "g2")} <= {"development", "validation"}
    grown = {"g1": {"nodes": ["a3:dev-a", "rec:1", "rec:9"]}, "g2": {"nodes": ["a3:dev-b", "rec:2"]}, "g3": {"nodes": ["rec:3", "name:new"]}}
    again = split.assign(grown, first, consumed=set(), development_keys={"dev-a", "dev-b"})
    for key in ("g1", "g2", "g3"):
        assert again["entries"][key]["partition"] == first["entries"][key]["partition"]
    assert "rec:9" in again["entries"]["g1"]["nodes"]
    joined = {"g1": {"nodes": ["a3:dev-a", "rec:1", "rec:3", "name:new"]}}  # an alias joins dev and reserved
    with pytest.raises(split.ConflictError):
        split.assign(joined, first, consumed=set(), development_keys={"dev-a"})
    with pytest.raises(split.ConflictError):  # a used group newly linked to consumed material
        split.assign({"g1": {"nodes": ["a3:dev-a", "rec:1", "a3:held"]}}, first, consumed={"a3:held"}, development_keys={"dev-a"})


def test_the_final_test_cannot_be_frozen_without_enough_never_used_groups():
    groups = {f"new-{i}": {"nodes": [f"rec:{i}"]} for i in range(5)}
    registry = split.assign(groups, {}, consumed=set(), development_keys=set())
    with pytest.raises(split.FreezeRefused):
        split.freeze_final_test(registry, set(groups), minimum=20)
    frozen = split.freeze_final_test(registry, set(groups), minimum=5)
    assert {e["partition"] for e in frozen["entries"].values()} == {"final_test"}
    assert {e["partition"] for e in registry["entries"].values()} == {"reserved"}  # the input is untouched


def test_selection_is_deterministic_and_spreads_over_groups_and_recordings():
    pool = [{"id": f"E{i}", "recording": f"r{i // 2}", "group": f"g{(i // 5) % 6}", "category": CATEGORIES[i % 5]} for i in range(60)]
    first = selection.development_batch(pool, target=20)
    assert first == selection.development_batch(list(reversed(pool)), target=20)
    assert len({i["recording"] for i in first}) == len(first)  # one excerpt per recording
    assert len(first) <= 20 and len({i["group"] for i in first}) >= 4


# -- preparing the project from a generated A3 project ---------------------------------------------


def test_prepare_reads_a3_without_changing_it_and_never_decodes_held_out(world, tmp_path, monkeypatch):
    before = a3_prints(world["a3"])
    decoded = []
    original = delivery_data._decode

    def spy(line):
        decoded.append(line)
        return original(line)

    monkeypatch.setattr(delivery_data, "_decode", spy)
    summary = preparing.prepare(tmp_path / "p", world["a3"], world["inventory"], say=lambda text: None)
    side = delivery_data.sides(world["dataset"])
    held = {e for e, s in side.items() if s == "evaluation"}
    assert decoded and not any(h in line for line in decoded for h in held)
    assert a3_prints(world["a3"]) == before and summary["a3_unchanged"]
    for path in (tmp_path / "p").rglob("*"):
        if path.is_file():
            assert POISON not in path.read_text(errors="ignore")


def test_prepare_places_groups_conservatively_and_reports_the_final_test_unfrozen(world, prepared):
    root, summary = prepared
    registry = project.read_registry(root)
    by_node = {n: e["partition"] for e in registry["entries"].values() for n in e["nodes"]}
    for key in world["held"]:
        assert by_node[f"a3:{key}"] == "excluded"
    assert by_node["name:totally new band"] == "reserved"
    assert by_node["name:guest singer"] == by_node[f"a3:{world['dev'][5]}"]  # the featured singer joins the band
    assert summary["final_test"]["frozen"] is False and project.read_manifest(root, "final_test") is None
    for partition in ("development", "validation"):
        for excerpt in (project.read_manifest(root, partition) or {"excerpts": []})["excerpts"]:
            assert excerpt["id"] not in world["leak"] and not excerpt["source"].endswith(tuple(world["leak"]))


def test_prepare_twice_gives_the_same_manifests_and_keeps_existing_excerpts(world, prepared, tmp_path):
    root, _summary = prepared
    other = tmp_path / "other"
    preparing.prepare(other, world["a3"], world["inventory"], say=lambda text: None)
    for partition in ("development", "validation"):
        a, b = project.read_manifest(root, partition), project.read_manifest(other, partition)
        assert (a or {}).get("excerpts") == (b or {}).get("excerpts")
    preparing.prepare(root, world["a3"], world["inventory"], say=lambda text: None, target=5)  # a smaller batch later
    assert len(project.read_manifest(root, "development")["excerpts"]) == len(project.read_manifest(other, "development")["excerpts"])


def test_manifests_are_checked_against_fingerprints_and_the_registry(world, prepared):
    root, _summary = prepared
    loaders.check_project(root)
    path = root / "manifests" / "development.json"
    original = path.read_text()
    path.write_text(original.replace('"start_s": 1.0', '"start_s": 1.5', 1))
    with pytest.raises(project.IntegrityError):
        loaders.check_project(root)
    path.write_text(original)
    manifest = project.read_manifest(root, "development")
    registry = project.read_registry(root)
    wrong = dict(manifest["excerpts"][0], group=next(k for k, e in registry["entries"].items() if e["partition"] == "excluded"))
    project.write_manifest(root, "development", [wrong, *manifest["excerpts"][1:]])
    with pytest.raises(project.IntegrityError, match="belongs to excluded"):
        loaders.check_project(root)
    unknown = dict(manifest["excerpts"][0], group="artist-nobody")
    project.write_manifest(root, "development", [unknown, *manifest["excerpts"][1:]])
    with pytest.raises(project.IntegrityError, match="not in the registry"):
        loaders.check_project(root)
    project.write_manifest(root, "development", manifest["excerpts"])


def frozen_final_test(root, world):
    """A final-test cohort for protection tests: the never-used group's recording, frozen."""
    registry = project.read_registry(root)
    key = next(k for k, e in registry["entries"].items() if e["partition"] == "reserved")
    registry = split.freeze_final_test(registry, {key}, minimum=1)
    project.write_registry(root, registry)
    sha = next(n[4:] for n in registry["entries"][key]["nodes"] if n.startswith("rec:"))
    row = next(r for r in world["rows"] if r["sha256"] == sha)
    excerpt = {"id": project.excerpt_id(sha, 1.0), "source": "new", "recording": sha, "path": row["file_path"],
               "size": Path(row["file_path"]).stat().st_size, "duration_s": 40.0, "sample_rate": RATE, "channels": 2,
               "start_s": 1.0, "end_s": 11.0, "group": key, "selection": "final-test cohort"}
    project.write_manifest(root, "final_test", [excerpt])
    store = Store(project.partition_folder(root, "final_test"), {excerpt["id"]: 10.0})
    store.add_interval(excerpt["id"], "confirmed", 2.0, 3.0, note=POISON)
    store.set_review(excerpt["id"], "complete")
    return excerpt


def test_development_loaders_cannot_read_final_test_labels(world, prepared, monkeypatch):
    root, _summary = prepared
    excerpt = frozen_final_test(root, world)
    for partitions in (("final_test",), ("development", "final_test"), ()):
        with pytest.raises(loaders.ProtectedSplitError):
            loaders.development_examples(root, partitions)
    final_folder = project.partition_folder(root, "final_test").resolve()
    original_read, original_open = Path.read_text, Path.open

    def guarded_read(self, *args, **kwargs):
        assert final_folder not in self.resolve().parents, "a final-test file was read"
        return original_read(self, *args, **kwargs)

    def guarded_open(self, *args, **kwargs):
        assert final_folder not in self.resolve().parents, "a final-test file was opened"
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    monkeypatch.setattr(Path, "open", guarded_open)
    for partitions in (("development",), ("validation",), ("development", "validation")):
        found = loaders.development_examples(root, partitions)
        assert all(e["partition"] in partitions for e in found)
        assert excerpt["id"] not in {e["id"] for e in found} and POISON not in json.dumps(found)


def test_changing_the_development_selection_cannot_change_the_final_test(world, prepared):
    root, _summary = prepared
    frozen_final_test(root, world)
    final_bytes = (root / "manifests" / "final-test.json").read_bytes()
    final_print = json.loads((root / "fingerprints" / "manifests.json").read_text())["final-test.json"]
    manifest = project.read_manifest(root, "development")
    project.write_manifest(root, "development", manifest["excerpts"][:-1])  # a development change
    preparing.prepare(root, world["a3"], world["inventory"], say=lambda text: None)  # and a full re-preparation
    assert (root / "manifests" / "final-test.json").read_bytes() == final_bytes
    assert json.loads((root / "fingerprints" / "manifests.json").read_text())["final-test.json"] == final_print
    assert project.read_registry(root)["entries"]  # registry kept, frozen group still final_test
    assert "final_test" in {e["partition"] for e in project.read_registry(root)["entries"].values()}


def test_a_recording_in_two_partitions_is_rejected(world, prepared):
    root, _summary = prepared
    excerpt = frozen_final_test(root, world)
    manifest = project.read_manifest(root, "development")
    clash = dict(manifest["excerpts"][0], recording=excerpt["recording"],
                 id=project.excerpt_id(excerpt["recording"], manifest["excerpts"][0]["start_s"]))
    project.write_manifest(root, "development", [clash, *manifest["excerpts"][1:]])
    with pytest.raises(project.IntegrityError):
        loaders.check_project(root)


def test_a_broken_history_is_refused_by_the_loader(world, prepared):
    root, _summary = prepared
    manifest = project.read_manifest(root, "development")
    store = Store(project.partition_folder(root, "development"), project.durations(manifest))
    first = manifest["excerpts"][0]["id"]
    store.add_interval(first, "confirmed", 1.0, 2.0)
    store.set_review(first, "complete")
    path = project.partition_folder(root, "development") / "annotations.jsonl"
    path.write_text(path.read_text().replace('"end": 2.0', '"end": 3.0'))
    with pytest.raises(project.IntegrityError, match="broken"):
        loaders.development_examples(root)


# -- the annotation window ---------------------------------------------------------------------------


class FakePlayer:
    def __init__(self):
        self.audio, self.rate, self.playing, self.pos, self.calls = None, 0, False, 0.0, []

    def load(self, audio, rate):
        self.audio, self.rate, self.pos, self.playing = np.asarray(audio), rate, 0.0, False

    def unload(self):
        self.audio, self.playing = None, False

    def play(self):
        self.playing = True
        self.calls.append(("play", self.pos))

    def pause(self):
        self.playing = False

    def toggle(self):
        self.pause() if self.playing else self.play()

    def replay(self):
        self.pos, self.playing = 0.0, True

    def seek(self, seconds):
        self.pos = seconds
        self.calls.append(("seek", seconds))

    def close(self):
        self.playing = False

    @property
    def finished(self):
        return False

    @property
    def duration_s(self):
        return len(self.audio) / self.rate if self.audio is not None else 0.0

    @property
    def position_s(self):
        return self.pos


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(prepared, app):
    from gmv.character.research.presence.window import PresenceWindow

    root, _summary = prepared
    windows = []

    def open_window(partition="development"):
        w = PresenceWindow(root, partition, player=FakePlayer())
        windows.append(w)
        return w

    yield root, open_window
    for w in windows:
        w.close()


def test_marks_land_at_the_playback_position_and_are_saved_at_once(window):
    root, open_window = window
    w = open_window()
    first = w.current
    w.player.pos = 1.23456
    w.mark_start()
    w.player.pos = 3.5
    w.mark_end("confirmed")
    w.player.pos = 6.0
    w.mark_start()
    w.player.pos = 5.0  # an end before the start is reordered, not lost
    w.mark_end("uncertain")
    manifest = project.read_manifest(root, "development")
    saved = Store(project.partition_folder(root, "development"), project.durations(manifest)).of(first)["intervals"]
    assert sorted((i["kind"], i["start"], i["end"]) for i in saved.values()) == [("confirmed", 1.235, 3.5), ("uncertain", 5.0, 6.0)]


def test_seeking_clicking_and_listening_around_a_boundary(window):
    _root, open_window = window
    w = open_window()
    w.seek(4.0)
    assert w.player.pos == 4.0
    w.move(-0.1)
    assert w.player.pos == pytest.approx(3.9)
    w.timeline.resize(1016, 120)
    w.timeline.clicked.emit(2.5)
    assert w.player.pos == 2.5
    w.listen_around(5.0)
    assert w.player.playing and w.player.pos == pytest.approx(4.0) and w.stop_at == pytest.approx(5.5)
    w.player.pos = 5.6
    w._tick()
    assert not w.player.playing and w.stop_at is None


def test_editing_removing_and_resuming(window):
    root, open_window = window
    w = open_window()
    first = w.current
    w.player.pos = 2.0
    w.mark_start()
    w.player.pos = 4.0
    w.mark_end("confirmed")
    w.adjust("start+")
    w.adjust("end-")
    w.player.pos = 4.2
    w.adjust("end=")
    w.adjust("kind")
    state = w.state()["intervals"][w.selected]
    assert (state["kind"], state["start"], state["end"], state["revision"]) == ("uncertain", 2.05, 4.2, 5)  # added, then 4 edits
    w.adjust("remove")
    assert not w.state()["intervals"]
    w.step(1)
    second = w.current
    w.close()
    again = open_window()
    assert again.current == second  # resumes where the owner stopped
    again.open(first)
    events = [e["event"] for e in again.store.history(first)]
    assert events == ["interval_add", "interval_edit", "interval_edit", "interval_edit", "interval_edit", "interval_remove"]


def test_completing_tags_unusable_and_the_open_mark_guard(window):
    _root, open_window = window
    w = open_window()
    w.player.pos = 1.0
    w.mark_start()
    w.complete()
    assert schema.status(w.state()) == "incomplete" and "open mark" in w.saved.text()
    w.cancel_mark()
    w.complete()
    assert schema.status(w.state()) == "negative"  # completed with no marks: no voice at all
    w.tag_boxes["guitar_lead_solo"].setChecked(True)
    w.tag_boxes["layered_vocals"].setChecked(True)
    assert w.state()["tags"] == ["guitar_lead_solo", "layered_vocals"]
    w.step(1)
    w.unusable("")
    assert schema.status(w.state()) == "incomplete"
    w.unusable("distorted recording")
    assert schema.status(w.state()) == "unusable"


def test_titles_come_from_the_private_metadata_never_from_file_names(window, world):
    root, open_window = window
    w = open_window()
    assert w.song.text().startswith("♪  Melody number ") or w.song.text().startswith("♪  ")
    entry = w.excerpts[w.current]
    assert Path(entry["path"]).stem not in w.song.text()
    titles = json.loads((root / "identity" / "titles.json").read_text())
    titles.pop(entry["recording"], None)
    (root / "identity" / "titles.json").write_text(json.dumps(titles))
    w.close()
    again = open_window()
    again.open(entry["id"])
    assert again.song.text() == "♪  Song title not known"


def test_backup_holds_the_history_and_manifests_but_no_audio(window, tmp_path):
    _root, open_window = window
    w = open_window()
    w.player.pos = 1.0
    w.mark_start()
    w.player.pos = 2.0
    w.mark_end("confirmed")
    target = w.save_backup(tmp_path / "backups")
    names = set(zipfile.ZipFile(target).namelist())
    assert {"annotations.jsonl", "session.json", "development.json", "manifests.json", "titles.json"} <= names
    assert not any(n.endswith((".wav", ".mp3", ".flac", ".lock")) for n in names)


def test_an_unplayable_excerpt_cannot_be_marked_or_completed(window):
    root, open_window = window
    w = open_window()
    entry = w.excerpts[w.current]
    path = Path(entry["path"])
    path.rename(path.with_suffix(".away"))
    try:
        w.open(entry["id"])
        assert not w.available and "cannot be played" in w.audio_problem.text()
        assert not w.start_button.isEnabled() and not w.complete_button.isEnabled()
        w.mark_start()
        w.complete()
        assert w.pending is None and schema.status(w.state()) == "incomplete"
    finally:
        path.with_suffix(".away").rename(path)


def test_a_second_window_and_an_unfrozen_final_test_are_refused(window):
    _root, open_window = window
    first = open_window()
    second = open_window()
    assert "already open" in second.problem
    first.close()
    final = open_window("final_test")
    assert "No final-test cohort" in final.problem and final.current is None


def test_annotating_never_touches_the_a3_project(window, world):
    before = a3_prints(world["a3"])
    _root, open_window = window
    w = open_window()
    w.player.pos = 1.0
    w.mark_start()
    w.player.pos = 2.0
    w.mark_end("confirmed")
    w.complete()
    w.close()
    assert a3_prints(world["a3"]) == before


def test_the_status_command_counts_without_reading_final_test_answers(world, prepared, capsys):
    from gmv.cli import main

    root, _summary = prepared
    main(["character", "presence", "status", "--folder", str(root)])
    out = json.loads(capsys.readouterr().out)
    assert out["partitions"]["final_test"] == {"frozen": False, "excerpts": 0, "annotations": "not read by this command"}
    assert out["partitions"]["development"]["incomplete"] == out["partitions"]["development"]["excerpts"]


def test_private_folders_are_ignored_by_git():
    root = Path(__file__).resolve().parents[1]
    assert "local-evidence/" in (root / ".gitignore").read_text().splitlines()
    assert str(project.FOLDER).startswith("local-evidence")


def test_a_backup_restores_the_same_annotations(window, tmp_path):
    root, open_window = window
    w = open_window()
    first = w.current
    w.player.pos = 1.0
    w.mark_start()
    w.player.pos = 2.5
    w.mark_end("confirmed")
    w.complete()
    target = w.save_backup(tmp_path / "backups")
    restored = tmp_path / "restored"
    zipfile.ZipFile(target).extractall(restored)
    durations = project.durations(json.loads((restored / "development.json").read_text()))
    assert Store(restored, durations).state() == w.store.state()
    assert Store(restored, durations).intact and schema.status(Store(restored, durations).of(first)) == "positive"


def test_duplicate_identifiers_and_recordings_outside_their_group_are_rejected(world, prepared):
    root, _summary = prepared
    manifest = project.read_manifest(root, "development")
    with pytest.raises(project.IntegrityError, match="share an identifier"):
        project.write_manifest(root, "development", [manifest["excerpts"][0], manifest["excerpts"][0]])
    registry = project.read_registry(root)
    other = next(k for k, e in registry["entries"].items()
                 if e["partition"] == "development" and k != manifest["excerpts"][0]["group"])
    moved = dict(manifest["excerpts"][0], group=other)
    project.write_manifest(root, "development", [moved, *manifest["excerpts"][1:]])
    with pytest.raises(project.IntegrityError, match="not a member of its artist group"):
        loaders.check_project(root)
    project.write_manifest(root, "development", manifest["excerpts"])
