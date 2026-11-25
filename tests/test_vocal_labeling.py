"""Phase A3: the vocal-delivery labelling dataset, answer store, playback and window."""

import json
import os
import sys
import textwrap

import numpy as np
import pytest
import soundfile as sf

from gmv.character.research.labeling import CLASSES
from gmv.character.research.labeling import dataset as data
from gmv.character.research.labeling import store
from gmv.character.research.labeling.playback import PacatPlayer, prepared

RATE = 8000
ARTISTS = ("artist-alpha", "artist-bravo", "artist-charlie", "artist-delta", "artist-echo", "artist-foxtrot")


def tone(seconds, rate=RATE, channels=2, seed=0, level=0.2):
    rng = np.random.default_rng(seed)
    time = np.arange(round(seconds * rate)) / rate
    left = level * np.sin(2 * np.pi * (110 + 40 * seed) * time) + 0.02 * rng.standard_normal(len(time))
    if channels == 1:
        return left.astype(np.float32)
    right = level * np.sin(2 * np.pi * (165 + 40 * seed) * time) + 0.02 * rng.standard_normal(len(time))
    return np.column_stack([left, right][:channels]).astype(np.float32)


@pytest.fixture
def corpus(tmp_path):
    """A small corpus: three recordings for each of six artists, plus a reference track by
    a seventh, with a manifest and a path map written as on another system."""
    music = tmp_path / "music"
    music.mkdir()
    manifest, located, seed = [], {}, 0
    for artist in ARTISTS:
        for number in range(3):
            seed += 1
            path = music / f"{artist}-song{number}.wav"
            sf.write(path, tone(40 + 5 * number, seed=seed, channels=1 if seed % 4 == 0 else 2), RATE)
            identity = f"r-{seed:012d}"
            manifest.append({"recording_id": identity, "group_id": artist, "sha256": data.sha256(path)})
            located[identity] = "X:\\" + path.name
    (tmp_path / "manifest.json").write_text(json.dumps({"recordings": manifest}))
    (tmp_path / "paths.json").write_text(json.dumps({"recordings": located}))
    reference = music / "the reference.wav"
    sf.write(reference, tone(60, seed=99), RATE)
    return tmp_path, music, reference


def gathered(corpus, reference_artist="artist-zulu"):
    root, music, reference = corpus
    candidates = data.corpus_recordings(root / "manifest.json", root / "paths.json", [("X:\\", str(music) + "/")])
    candidates.append((str(reference), reference_artist, None, "reference track"))
    recordings, problems = data.gather(candidates, {str(reference): "the reference"})
    assert not problems
    return recordings


def prepared_dataset(corpus, folder, target=14, reference_artist="artist-zulu"):
    recordings = gathered(corpus, reference_artist)
    return data.prepare(folder, recordings, {reference_artist}, target, scores=None)


# -- the dataset --------------------------------------------------------------------------


def test_the_manifest_lists_every_excerpt_with_its_recording_times_and_split(corpus, tmp_path):
    folder = tmp_path / "labels"
    found = prepared_dataset(corpus, folder)
    on_disk = json.loads((folder / "dataset.json").read_text())
    assert on_disk["format"] == data.FORMAT and on_disk["version"] == data.VERSION
    assert data.validate(on_disk) == []
    assert len(on_disk["excerpts"]) == 14 and on_disk == json.loads(json.dumps(found))
    ids = [e["id"] for e in on_disk["excerpts"]]
    assert ids == [f"VOCAL-{n:04d}" for n in range(1, 15)]
    assert sorted(on_disk["order"]) == ids
    for excerpt in on_disk["excerpts"]:
        recording = on_disk["recordings"][excerpt["recording"]]
        assert excerpt["end_s"] - excerpt["start_s"] == pytest.approx(data.EXCERPT_S)
        assert data.EDGE_S <= excerpt["start_s"] and excerpt["end_s"] <= recording["duration_s"] - data.EDGE_S + 1e-9
        assert recording["artist"] in on_disk["artists"]
        assert excerpt["selection"] in data.METHODS
        assert os.path.isfile(recording["path"]) and recording["sha256"] == data.sha256(recording["path"])
    # Nobody dominates: at most four excerpts an artist, two a recording.
    per_artist, per_recording = {}, {}
    for excerpt in on_disk["excerpts"]:
        artist = on_disk["recordings"][excerpt["recording"]]["artist"]
        per_artist[artist] = per_artist.get(artist, 0) + 1
        per_recording[excerpt["recording"]] = per_recording.get(excerpt["recording"], 0) + 1
    assert max(per_artist.values()) <= data.PER_ARTIST and max(per_recording.values()) <= data.PER_RECORDING
    assert len(per_artist) == 7  # every artist contributes before anyone gets a second turn


def test_the_same_inputs_give_the_same_excerpts_and_identifiers(corpus, tmp_path):
    first = prepared_dataset(corpus, tmp_path / "one")
    again = prepared_dataset(corpus, tmp_path / "two")
    for found in (first, again):
        found.pop("created_utc")
    assert first == again
    other = data.prepare(tmp_path / "three", gathered(corpus), {"artist-zulu"}, 14, None, seed="another seed")
    assert [e["start_s"] for e in other["excerpts"]] != [e["start_s"] for e in first["excerpts"]] or other["order"] != first["order"]


def test_each_excerpt_is_exactly_those_seconds_of_its_own_recording(corpus, tmp_path):
    found = prepared_dataset(corpus, tmp_path / "labels")
    for excerpt in found["excerpts"]:
        recording = found["recordings"][excerpt["recording"]]
        whole, rate = sf.read(recording["path"], dtype="float32", always_2d=True)
        part, part_rate = data.read_excerpt(recording["path"], excerpt["start_s"], excerpt["end_s"])
        first = round(excerpt["start_s"] * rate)
        assert part_rate == rate == recording["sample_rate"]
        assert part.shape == (round(data.EXCERPT_S * rate), recording["channels"])
        assert np.array_equal(part, whole[first : first + len(part)])
    # Excerpts of one recording never overlap, and keep a gap.
    spans = {}
    for excerpt in found["excerpts"]:
        spans.setdefault(excerpt["recording"], []).append((excerpt["start_s"], excerpt["end_s"]))
    for found_spans in spans.values():
        found_spans.sort()
        for (_a, end), (start, _b) in zip(found_spans, found_spans[1:], strict=False):
            assert start >= end + data.GAP_S


def test_excerpt_lengths_and_dataset_integrity_are_checked(corpus, tmp_path):
    good = {"id": "VOCAL-0001", "start_s": 5.0, "end_s": 15.0}
    assert data.check_excerpt(good, {"duration_s": 30.0}) == []
    assert data.check_excerpt({**good, "end_s": 12.0}, None)  # 7 s
    assert data.check_excerpt({**good, "end_s": 18.0}, None)  # 13 s
    assert data.check_excerpt({**good, "start_s": -1.0, "end_s": 9.0}, None)
    assert data.check_excerpt(good, {"duration_s": 12.0})  # beyond the recording
    found = prepared_dataset(corpus, tmp_path / "labels")
    broken = json.loads(json.dumps(found))
    broken["excerpts"][1]["start_s"] = broken["excerpts"][0]["start_s"]
    broken["excerpts"][1]["end_s"] = broken["excerpts"][0]["end_s"]
    broken["excerpts"][1]["recording"] = broken["excerpts"][0]["recording"]
    assert any("overlap" in p for p in data.validate(broken))
    broken = json.loads(json.dumps(found))
    broken["order"].pop()
    assert any("presentation order" in p for p in data.validate(broken))
    with pytest.raises(data.DatasetError):
        data.write(tmp_path / "bad", broken)
    assert not (tmp_path / "bad" / "dataset.json").exists()


# -- the split ----------------------------------------------------------------------------


def test_no_artist_is_on_both_sides_and_the_rule_is_fixed(corpus, tmp_path):
    found = prepared_dataset(corpus, tmp_path / "labels")
    for artist, entry in found["artists"].items():
        assert entry["split"] == data.side_of(artist, found["reference_artists"], found["seed"])
    sides = {}
    for excerpt in found["excerpts"]:
        artist = found["recordings"][excerpt["recording"]]["artist"]
        sides.setdefault(artist, set()).add(data.split_of(found, excerpt))
    assert all(len(found_sides) == 1 for found_sides in sides.values())
    # The rule depends on the artist alone: about a quarter of many names are held out.
    names = [f"artist-{n}" for n in range(4000)]
    share = sum(data.held_out(name) for name in names) / len(names)
    assert 0.22 < share < 0.28
    assert [data.held_out(n) for n in names[:50]] == [data.held_out(n) for n in names[:50]]


def test_reference_tracks_and_their_artist_are_always_held_out(corpus, tmp_path):
    # An artist the hash rule would put in development is held out once it owns a reference.
    developing = next(f"artist-ref-{n}" for n in range(1000) if not data.held_out(f"artist-ref-{n}"))
    found = prepared_dataset(corpus, tmp_path / "labels", reference_artist=developing)
    assert found["artists"][developing] == {"split": "evaluation", "reason": "owns a reference track"}
    reference = [k for k, r in found["recordings"].items() if r["reference"]]
    assert len(reference) == 1
    picked = [e for e in found["excerpts"] if e["recording"] == reference[0]]
    assert len(picked) == data.PER_RECORDING  # the reference track gets its excerpts first
    assert all(data.split_of(found, e) == "evaluation" for e in picked)
    moved = json.loads(json.dumps(found))
    moved["artists"][developing]["split"] = "development"
    assert any("reference" in p for p in data.validate(moved))


def test_adding_recordings_never_moves_an_artist_or_an_existing_excerpt(corpus, tmp_path):
    folder = tmp_path / "labels"
    found = prepared_dataset(corpus, folder)
    store.Labels(folder).label(found["order"][0], "clean")
    before = json.loads(json.dumps(found))
    root, music, _reference = corpus
    newcomer = music / "new song.wav"
    sf.write(newcomer, tone(50, seed=7), RATE)
    another = music / "by alpha.wav"
    sf.write(another, tone(45, seed=8), RATE)
    added, problems = data.add_recordings(folder, found, [newcomer], "Brand New Band")
    assert len(added) == 2 and not problems
    added_alpha, _ = data.add_recordings(folder, found, [another], "artist-alpha")
    assert found["artists"]["artist-alpha"] == before["artists"]["artist-alpha"]
    assert found["artists"]["artist-brand-new-band"]["split"] == data.side_of("artist-brand-new-band")
    for artist, entry in before["artists"].items():
        assert found["artists"][artist] == entry
    assert found["excerpts"][: len(before["excerpts"])] == before["excerpts"]
    assert found["order"][: len(before["order"])] == before["order"]
    assert {e["id"] for e in added + added_alpha} == {f"VOCAL-{n:04d}" for n in range(15, 19)}
    assert data.load(folder) == found
    # The same file, or the same audio under another name, is not added twice.
    copy = music / "copy of new song.wav"
    copy.write_bytes(newcomer.read_bytes())
    again, problems = data.add_recordings(folder, found, [newcomer, copy], "Brand New Band")
    assert again == [] and set(problems.values()) == {"already in the dataset", "the same audio is already in the dataset"}
    with pytest.raises(data.DatasetError):
        data.add_recordings(folder, found, [newcomer], "   ")


def test_preparing_again_is_refused_once_anything_is_labelled(corpus, tmp_path):
    folder = tmp_path / "labels"
    prepared_dataset(corpus, folder)
    prepared_dataset(corpus, folder)  # nothing labelled yet: allowed
    store.Labels(folder).skip("VOCAL-0001")
    with pytest.raises(data.DatasetError, match="already exist"):
        prepared_dataset(corpus, folder)


# -- recordings that cannot be used --------------------------------------------------------


def test_missing_unreadable_and_too_short_recordings_are_reported_not_used(corpus, tmp_path):
    root, music, _reference = corpus
    garbage = music / "not audio.mp3"
    garbage.write_bytes(b"ID3 this is not an mp3 at all" * 50)
    short = music / "short.wav"
    sf.write(short, tone(5), RATE)
    empty = music / "empty.wav"
    empty.write_bytes(b"")
    changed = music / f"{ARTISTS[0]}-song0.wav"
    candidates = [
        (str(music / "missing.wav"), "artist-x", None, "test"),
        (str(garbage), "artist-x", None, "test"),
        (str(short), "artist-x", None, "test"),
        (str(empty), "artist-x", None, "test"),
        (str(changed), ARTISTS[0], "0" * 64, "test"),
    ]
    recordings, problems = data.gather(candidates)
    assert recordings == {}
    assert problems[str(music / "missing.wav")] == "not found"
    assert problems[str(garbage)].startswith("not a readable audio file")
    assert problems[str(short)].startswith("only 5.0 s long")
    assert problems[str(empty)].startswith("not a readable audio file")
    assert problems[str(changed)] == "its contents differ from the registered recording"
    with pytest.raises(data.DatasetError):
        data.load(tmp_path / "nowhere")


# -- the answers --------------------------------------------------------------------------


def test_answers_are_saved_at_once_changed_safely_and_kept_in_history(tmp_path):
    labels = store.Labels(tmp_path)
    labels.label("VOCAL-0001", "clean")
    assert (tmp_path / store.LABELS).read_text().count("\n") == 1  # on disk already
    labels.label("VOCAL-0001", "scream")
    labels.note("VOCAL-0001", "growl at the end")
    labels.skip("VOCAL-0002")
    reopened = store.Labels(tmp_path)
    state = reopened.state()
    assert state["VOCAL-0001"] == {"label": "scream", "skipped": False, "note": "growl at the end", "changes": 1}
    assert state["VOCAL-0002"]["skipped"] and state["VOCAL-0002"]["label"] is None
    assert [e["value"] for e in reopened.history("VOCAL-0001") if e["event"] == "label"] == ["clean", "scream"]
    with pytest.raises(ValueError):
        reopened.label("VOCAL-0003", "singing")  # only the five answers
    # A skipped excerpt that gets an answer is no longer skipped; skipping a labelled one
    # does not erase its answer.
    reopened.label("VOCAL-0002", "none")
    reopened.skip("VOCAL-0001")
    state = reopened.state()
    assert state["VOCAL-0002"] == {"label": "none", "skipped": False, "note": "", "changes": 0}
    assert state["VOCAL-0001"]["label"] == "scream" and not state["VOCAL-0001"]["skipped"]
    counts = store.progress(["VOCAL-0001", "VOCAL-0002", "VOCAL-0003", "VOCAL-0004"], state)
    assert counts == {"total": 4, "labelled": 2, "skipped": 0, "remaining": 2}


def test_a_damaged_line_is_left_out_and_every_other_answer_kept(tmp_path):
    labels = store.Labels(tmp_path)
    labels.label("VOCAL-0001", "shout")
    labels.label("VOCAL-0002", "mixed")
    with (tmp_path / store.LABELS).open("a") as file:
        file.write('{"time": "2026-11-24T00:00:00", "excerpt": "VOCAL-0003", "eve')  # cut off
    reopened = store.Labels(tmp_path)
    assert reopened.damaged == 1
    assert {k: v["label"] for k, v in reopened.state().items()} == {"VOCAL-0001": "shout", "VOCAL-0002": "mixed"}
    reopened.label("VOCAL-0003", "clean")  # a new answer still lands on a line of its own
    final = store.Labels(tmp_path)
    assert final.state()["VOCAL-0003"]["label"] == "clean" and final.damaged == 1


def test_resuming_opens_the_excerpt_left_open_else_the_first_without_an_answer():
    order = ["VOCAL-0003", "VOCAL-0001", "VOCAL-0002"]
    state = {"VOCAL-0003": {"label": "clean"}, "VOCAL-0001": {"skipped": True}}
    assert store.resume_point(order, state, {"current": "VOCAL-0001"}) == "VOCAL-0001"
    assert store.resume_point(order, state, {}) == "VOCAL-0002"
    assert store.resume_point(order, state, {"current": "VOCAL-9999"}) == "VOCAL-0002"
    state["VOCAL-0002"] = {"label": "none"}
    assert store.resume_point(order, state, {}) == "VOCAL-0001"  # only a skipped one is left
    assert store.resume_point([], {}, {}) is None


def test_the_presentation_order_is_shuffled_reproducibly_and_saved(corpus, tmp_path):
    found = prepared_dataset(corpus, tmp_path / "labels")
    assert found["order"] == [e["id"] for e in data.number([dict(e) for e in found["excerpts"]], seed=found["seed"])]
    by_placement = data.number([{"i": n} for n in range(30)])
    assert [e["i"] for e in by_placement] != list(range(30))  # shuffled
    assert [e["i"] for e in data.number([{"i": n} for n in range(30)])] == [e["i"] for e in by_placement]
    assert data.load(tmp_path / "labels")["order"] == found["order"]


def test_a_backup_holds_the_dataset_and_the_answers(corpus, tmp_path):
    import zipfile

    folder = tmp_path / "labels"
    found = prepared_dataset(corpus, folder)
    store.Labels(folder).label(found["order"][0], "clean")
    archive = store.backup(folder, tmp_path)
    with zipfile.ZipFile(archive) as zipped:
        assert set(zipped.namelist()) == {"dataset.json", store.LABELS}
    rows = store.export(found, store.Labels(folder).state())
    assert rows[0]["split"] in ("development", "evaluation") and len(rows) == 14


# -- playback -----------------------------------------------------------------------------

STAND_IN = textwrap.dedent(
    """
    import sys, time
    out = open(sys.argv[1], "wb")
    args = sys.argv[2:]
    open(sys.argv[1] + ".args", "w").write(" ".join(args))
    while True:
        block = sys.stdin.buffer.read(4096)
        if not block:
            break
        out.write(block)
        out.flush()
    """
)


def stand_in(tmp_path):
    script = tmp_path / "pacat_stand_in.py"
    script.write_text(STAND_IN)
    received = tmp_path / "received.raw"
    return [sys.executable, str(script), str(received)], received


def wait(condition, seconds=10):
    import time

    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.02)
    return False


def test_playback_sends_exactly_the_excerpt_at_its_own_rate_and_channels(tmp_path):
    program, received = stand_in(tmp_path)
    player = PacatPlayer(program)
    audio = tone(1.5, rate=22050, channels=2, seed=3)
    player.load(audio, 22050)
    player.play()
    assert wait(lambda: player.finished)
    sent = np.frombuffer(received.read_bytes(), dtype="<f4").reshape(-1, 2)
    assert np.array_equal(sent, prepared(audio))
    arguments = (received.parent / (received.name + ".args")).read_text()
    assert "--rate=22050" in arguments and "--channels=2" in arguments and "--format=float32le" in arguments
    assert "--device" not in arguments and "volume" not in arguments  # nothing about the system is changed
    assert player.position_s == pytest.approx(1.5)
    # Played again from a position: only what follows it is sent.
    received.unlink()
    player.seek(1.0)
    player.play()
    assert wait(lambda: player.finished)
    sent = np.frombuffer(received.read_bytes(), dtype="<f4").reshape(-1, 2)
    assert np.array_equal(sent, prepared(audio)[22050:])


def test_playback_of_mono_and_of_more_than_two_channels(tmp_path):
    program, received = stand_in(tmp_path)
    player = PacatPlayer(program)
    mono = tone(0.5, channels=1)
    player.load(mono, RATE)
    player.play()
    assert wait(lambda: player.finished)
    assert np.array_equal(np.frombuffer(received.read_bytes(), dtype="<f4"), mono)
    player.load(np.column_stack([mono, mono, mono]), RATE)
    assert player.note == "playing the first two channels" and player.audio.shape[1] == 2


def test_pause_stops_at_once_and_play_continues_from_there(tmp_path):
    program, received = stand_in(tmp_path)
    slow = tmp_path / "slow.py"
    slow.write_text(STAND_IN.replace("out.flush()", "out.flush(); time.sleep(0.05)"))
    player = PacatPlayer([sys.executable, str(slow), str(received)])
    player.load(tone(10, rate=RATE), RATE)
    player.play()
    assert wait(lambda: player.playing and player.position_s > 0.2)
    player.pause()
    assert not player.playing
    stopped = player.position_s
    assert 0.2 <= stopped < 10
    player.play()
    assert player.playing and player.position_s == pytest.approx(stopped, abs=0.3)
    player.close()
    assert not player.playing


def test_without_a_sound_program_playback_says_so(tmp_path):
    player = PacatPlayer()
    player.program = None
    player.load(tone(1), RATE)
    player.play()
    assert player.error and "pacat" in player.error and not player.playing
    failing = PacatPlayer([sys.executable, "-c", "import sys; sys.stderr.write('Connection refused'); sys.exit(1)"])
    failing.load(tone(1), RATE)
    failing.play()
    assert wait(lambda: not failing.playing)
    assert wait(lambda: failing.error is not None)
    assert "Connection refused" in failing.error


# -- the window ---------------------------------------------------------------------------


class FakePlayer:
    """Remembers what the window asked of it; plays nothing."""

    def __init__(self):
        self.audio, self.rate, self.calls, self.playing, self.pos, self.error, self.note = None, 0, [], False, 0.0, None, ""

    def load(self, audio, rate):
        self.audio, self.rate, self.pos, self.playing = np.asarray(audio), rate, 0.0, False
        self.calls.append(("load", len(audio), rate))

    def unload(self):
        self.audio, self.playing = None, False

    def play(self):
        self.playing = True
        self.calls.append(("play",))

    def pause(self):
        self.playing = False
        self.calls.append(("pause",))

    def toggle(self):
        self.pause() if self.playing else self.play()

    def replay(self):
        self.pos, self.playing = 0.0, True
        self.calls.append(("replay",))

    def seek(self, seconds):
        self.pos = seconds

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
def labelling(corpus, tmp_path, app):
    folder = tmp_path / "labels"
    found = prepared_dataset(corpus, folder)
    windows = []

    def open_window(autoplay=False):
        from gmv.character.research.labeling.window import LabelWindow

        window = LabelWindow(folder, player=FakePlayer(), autoplay=autoplay)
        windows.append(window)
        return window

    yield folder, found, open_window
    for window in windows:
        window.close()


def test_the_window_opens_on_the_first_excerpt_and_loads_only_its_seconds(labelling):
    folder, found, open_window = labelling
    window = open_window()
    first = found["order"][0]
    assert window.current == first and window.identifier.text() == first
    excerpt = next(e for e in found["excerpts"] if e["id"] == first)
    rate = found["recordings"][excerpt["recording"]]["sample_rate"]
    assert window.player.calls[0] == ("load", round(data.EXCERPT_S * rate), rate)
    assert "Not labelled yet" in window.status.text()
    assert "Still to label 14" in window.counts.text()
    window.close()


def test_clicking_an_answer_saves_it_moves_on_and_can_be_changed(labelling, app):
    folder, found, open_window = labelling
    window = open_window()
    first, second = found["order"][:2]
    window.answer_buttons["scream"].click()
    assert store.Labels(folder).state()[first]["label"] == "scream"
    assert "Saved" in window.saved.text()
    window.next_timer.timeout.emit()  # the short pause before moving on
    assert window.current == second
    window.previous_button.click()
    assert window.current == first and window.answer_buttons["scream"].isChecked()
    assert "Screaming" in window.status.text()
    window.advance.setChecked(False)
    window.answer_buttons["mixed"].click()
    assert window.current == first  # stays when "go to the next" is off
    state = store.Labels(folder).state()[first]
    assert state["label"] == "mixed" and state["changes"] == 1
    assert "Labelled 1" in window.counts.text()


def test_skipping_and_notes(labelling):
    folder, found, open_window = labelling
    window = open_window()
    first, second = found["order"][:2]
    window.note.setText("too quiet to tell")
    window.skip_button.click()
    state = store.Labels(folder).state()
    assert state[first]["skipped"] and state[first]["note"] == "too quiet to tell"
    assert window.current == second and window.note.text() == ""
    assert "Skipped 1" in window.counts.text() and "Still to label 14" in window.counts.text()
    window.open(first)
    assert window.note.text() == "too quiet to tell" and "Skipped" in window.status.text()


def test_closing_and_opening_again_resumes_where_the_owner_left_off(labelling):
    folder, found, open_window = labelling
    window = open_window()
    for answer in ("clean", "none", "shout"):
        window.choose(answer)
        window.next_unlabelled()
    third = found["order"][3]
    window.open(found["order"][1])
    window.close()
    again = open_window()
    assert again.current == found["order"][1]
    state = store.Labels(folder).state()
    assert [state[e]["label"] for e in found["order"][:3]] == ["clean", "none", "shout"]
    assert "Labelled 3" in again.counts.text()
    again.next_unlabelled()
    assert again.current == third
    assert data.load(folder)["order"] == found["order"]  # the order never changes


def test_a_second_window_on_the_same_answers_is_refused(labelling):
    folder, _found, open_window = labelling
    first = open_window()
    second = open_window()
    assert second.dataset is None and "already open" in second.problem
    assert not any(b.isEnabled() for b in second.answer_buttons.values())
    first.close()


def test_an_excerpt_whose_recording_is_missing_cannot_be_labelled(labelling):
    folder, found, open_window = labelling
    first = found["order"][0]
    excerpt = next(e for e in found["excerpts"] if e["id"] == first)
    os.rename(found["recordings"][excerpt["recording"]]["path"], str(folder / "moved away.wav"))
    window = open_window()
    assert window.current == first and not window.available
    assert "cannot be played right now" in window.audio_problem.text()
    assert not window.answer_buttons["clean"].isEnabled()
    window.choose("clean")
    assert first not in store.Labels(folder).state()
    window.skip()
    assert store.Labels(folder).state()[first]["skipped"]


def test_keys_and_listening_controls(labelling, app):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    folder, found, open_window = labelling
    window = open_window()
    window.show()
    window.activateWindow()
    assert QTest.qWaitForWindowActive(window)
    window.advance.setChecked(False)
    QTest.keyClick(window, Qt.Key.Key_2)
    assert store.Labels(folder).state()[window.current]["label"] == "scream"
    window.play_button.click()
    assert window.player.playing
    window.restart_button.click()
    assert window.player.calls[-1] == ("replay",)
    QTest.keyClick(window, Qt.Key.Key_Right)
    assert window.current == found["order"][1]


def visible_text(window):
    from PySide6.QtWidgets import QAbstractButton, QLabel, QLineEdit, QListWidget, QWidget

    texts = [window.windowTitle()]
    for widget in window.findChildren(QWidget):
        texts += [widget.toolTip(), widget.statusTip(), widget.whatsThis(), widget.accessibleName(), widget.windowTitle()]
        if isinstance(widget, (QLabel, QAbstractButton)):
            texts.append(widget.text())
        if isinstance(widget, QLineEdit):
            texts += [widget.text(), widget.placeholderText()]
        if isinstance(widget, QListWidget):
            for row in range(widget.count()):
                texts += [widget.item(row).text(), widget.item(row).toolTip()]
    return "\n".join(t for t in texts if t)


def test_nothing_about_the_recording_or_the_split_is_shown(labelling):
    folder, found, open_window = labelling
    window = open_window()
    secrets = {"development", "evaluation", "held out", "reference", "REC-", "hash", *data.METHODS}
    for recording in found["recordings"].values():
        secrets |= {recording["artist"], data.artist_name(recording["artist"]), os.path.basename(recording["path"])}
        secrets |= {os.path.splitext(os.path.basename(recording["path"]))[0], os.path.dirname(recording["path"])}
    for excerpt_id in found["order"]:
        window.open(excerpt_id)
        shown = visible_text(window).lower()
        leaked = [s for s in secrets if s.lower() in shown]
        assert not leaked, leaked
    # Nor when a recording is missing.
    excerpt = found["excerpts"][0]
    os.rename(found["recordings"][excerpt["recording"]]["path"], str(folder / "elsewhere.wav"))
    window.open(excerpt["id"])
    shown = visible_text(window).lower()
    assert not [s for s in secrets if s.lower() in shown]


def test_no_prepared_excerpts_gives_a_plain_message(tmp_path, app):
    from gmv.character.research.labeling.window import LabelWindow

    window = LabelWindow(tmp_path / "empty", player=FakePlayer())
    assert "No excerpts are ready" in window.problem and not window.play_button.isEnabled()
    window.close()


def test_the_answers_offered_are_the_five_classes():
    assert list(CLASSES) == ["clean", "scream", "shout", "mixed", "none"]
    assert [name for name, _ in CLASSES.values()] == [
        "Clean singing", "Screaming", "Shouting", "Mixed / ambiguous", "No vocals",
    ]


def test_the_label_command_finds_the_folder_in_the_repository(monkeypatch, tmp_path):
    from gmv import paths
    from gmv.character.research.labeling import cli

    monkeypatch.chdir(tmp_path)  # the working directory does not matter
    assert cli.default_folder() == paths.REPOSITORY / "local-evidence" / "sonic-character" / "vocal-labeling"
    assert "local-evidence/" in (paths.REPOSITORY / ".gitignore").read_text()
