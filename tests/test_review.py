"""Excerpt review pipeline: placement, exact cutting, source safety, annotation storage,
export, category-separated scoring, the review window and playback. Procedural audio only."""

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import soundfile as sf

from gmv.corpus import registry
from gmv.review import annotations as ann
from gmv.review import audio as audio_tools
from gmv.review import batch as batch_tools
from gmv.review import excerpts as ex
from gmv.review import report, timelines
from gmv.review.pipeline import build, load_tables, verify

RATE = 8000


class ToneIsGuitar:
    """Stand-in model: the score is the share of the newest half second's energy that is
    the 220 Hz tone, whatever the level (the recogniser evens out level before the model)."""

    name, version = "tone_is_guitar", "test"
    labels = ("guitar", "guitar_distorted", "guitar_clean", "guitar_acoustic")
    sample_rate, context_s, hop_s = 8000, 2.0, 0.5
    thresholds = {label: {"detected": 0.6, "candidate": 0.3} for label in labels}

    def infer(self, context):
        newest = context[-4000:].astype(np.float64)
        phase = 2 * np.pi * 220 * np.arange(4000) / 8000
        tone = 2 * (np.mean(newest * np.sin(phase)) ** 2 + np.mean(newest * np.cos(phase)) ** 2)
        level = min(1.0, tone / max(float(np.mean(newest**2)), 1e-12))
        return np.array([level, level / 2, 0.1, 0.05])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def song(seconds, loud_from, loud_to, seed=0):
    """Stereo mix: a loud tone (the stand-in model's "guitar") between two times, quiet
    noise with a trace of the tone elsewhere."""
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * RATE)) / RATE
    loud = (t >= loud_from) & (t < loud_to)
    tone, noise = np.where(loud, 0.4, 0.002), np.where(loud, 0.005, 0.02)
    mono = tone * np.sin(2 * np.pi * 220 * t) + noise * rng.standard_normal(len(t))
    return np.stack([mono, 0.9 * mono], axis=1).astype(np.float32)


# ------------------------------------------------------------------ placement
def test_core_placements_are_five_deterministic_and_disjoint():
    digest = "ab" * 32
    first = ex.core_placements("r-abababababab", digest, 240.0)
    again = ex.core_placements("r-abababababab", digest, 240.0)
    assert first == again
    assert [p.position for p in first] == ["start", "middle", "end", "random_1", "random_2"]
    assert all(p.category == "core" for p in first)
    spans = sorted((p.start_s, p.end_s) for p in first)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False))
    assert spans[0][0] == 0.0 and abs(spans[-1][1] - 240.0) < 1e-6
    other = ex.core_placements("r-cdcdcdcdcdcd", "cd" * 32, 240.0)
    assert [p.start_s for p in other][3:] != [p.start_s for p in first][3:]


def test_short_recordings_get_fewer_core_excerpts():
    assert len(ex.core_placements("r-000000000000", "00" * 32, 8.0)) == 1
    few = ex.core_placements("r-000000000000", "00" * 32, 40.0)
    assert 1 < len(few) < 5
    spans = sorted((p.start_s, p.end_s) for p in few)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False))


def test_feature_and_model_excerpts_stay_off_core_excerpts():
    audio = song(300, 100, 200)
    core = ex.core_placements("r-111111111111", "11" * 32, 300.0)
    taken = [(p.start_s, p.end_s) for p in core]
    feature = ex.feature_placement("r-111111111111", audio, RATE, taken)
    assert feature.category == "feature"
    assert all(feature.end_s <= a or feature.start_s >= b for a, b in taken)
    timeline = timelines.compute(audio, RATE, ToneIsGuitar())
    th = ToneIsGuitar.thresholds["guitar"]
    taken.append((feature.start_s, feature.end_s))
    flagged = ex.model_placement("r-111111111111", timeline, th, "negative_candidate", 300.0, taken)
    assert flagged is not None and flagged.category == "model_flagged"
    assert "diagnostic only" in flagged.reason and 100 <= flagged.start_s and flagged.end_s <= 200.5
    assert all(flagged.end_s <= a or flagged.start_s >= b for a, b in taken)
    quiet = song(300, 400, 500)  # never loud: nothing to flag as a false activation
    timeline = timelines.compute(quiet, RATE, ToneIsGuitar())
    assert (
        ex.model_placement("r-111111111111", timeline, th, "negative_candidate", 300.0, taken)
        is None
    )


# ------------------------------------------------------------- exact cutting
def test_excerpts_hold_the_exact_source_samples_and_source_is_untouched(tmp_path):
    source = tmp_path / "source.flac"
    sf.write(source, song(60, 10, 20), RATE, subtype="PCM_24")
    before = sha(source)
    audio, rate, info = audio_tools.decode(source)
    assert info["sha256"] == before and info["frames_decoded"] == len(audio)
    placement = ex.Placement("r-000000000000-c3", "core", "middle", 24.0, 36.0, "test")
    begin, end = ex.write_excerpt(audio, rate, placement, tmp_path / "x.wav")
    assert (begin, end) == (24 * RATE, 36 * RATE)
    data, written_rate = sf.read(tmp_path / "x.wav", dtype="float32", always_2d=True)
    assert written_rate == rate and data.shape == (12 * RATE, 2)
    assert np.array_equal(data, audio[begin:end])
    assert ex.matches_source(tmp_path / "x.wav", audio, begin, end)
    assert sha(source) == before


def test_integrity_flags_clipping_identical_channels_and_quiet_sections():
    t = np.arange(10 * RATE) / RATE
    mono = 1.2 * np.sin(2 * np.pi * 100 * t)
    mono[3 * RATE : 6 * RATE] = 0.0001  # an interior quiet stretch
    stereo = np.stack([mono, mono], axis=1).astype(np.float32)
    q = audio_tools.integrity(stereo, RATE)
    assert q["clipped_runs"] > 0 and q["samples_beyond_full_scale"] > 0
    assert q["identical_channels"] is True
    assert q["interior_low_volume_sections"] == 1
    info = {
        **q,
        "rate": RATE,
        "frames_declared": len(stereo),
        "frames_decoded": len(stereo),
        "decoder_messages": ["error: bad frame"],
    }
    issues = audio_tools.source_issues(
        info, {"codec": "mp3", "bitrate_kbps": "64", "extension": "mp3"}
    )
    for issue in (
        "clipping_runs",
        "identical_channels",
        "interior_low_volume_sections",
        "decoder_messages",
        "low_bitrate",
        "lossy_source",
        "sample_rate_below_model_rate",
        "decoded_beyond_full_scale",
    ):
        assert issue in issues


# -------------------------------------------------------------- annotations
def test_store_accepts_only_annotation_fields_and_keeps_every_revision(tmp_path):
    store = ann.AnnotationStore(tmp_path)
    with pytest.raises(ValueError):
        store.save("e1", "r1", {"model_guitar_mean": 0.9})
    with pytest.raises(ValueError):
        store.save("e1", "r1", {"guitar_presence": "detected"})
    with pytest.raises(ValueError):
        store.save("e1", "r1", {"other_instruments": ["kazoo"]})
    assert store.save("e1", "r1", {}) is None  # nothing chosen, nothing written
    first = store.save(
        "e1", "r1", {"guitar_presence": "uncertain", "difficulty_note": "synth or guitar?"}
    )
    assert first["revision"] == 1 and first["annotation_source"] == ann.SOURCE
    assert store.save("e1", "r1", {"guitar_presence": "uncertain"}) is None  # unchanged
    store.save(
        "e1",
        "r1",
        {
            "guitar_presence": "present_intermittent",
            "intermittent_spans_note": "0:00–0:04 and 0:09-0:12",
        },
    )
    reopened = ann.AnnotationStore(tmp_path)
    assert reopened.values("e1")["guitar_presence"] == "present_intermittent"
    assert [e["guitar_presence"] for e in reopened.history("e1")] == [
        "uncertain",
        "present_intermittent",
    ]
    assert reopened.values("e1")["difficulty_note"] == "synth or guitar?"


def test_parse_spans_is_approximate_and_bounded():
    assert ann.parse_spans("0:00–0:04 and 0:09-0:12", 12) == [(0.0, 4.0), (9.0, 12.0)]
    assert ann.parse_spans("2-5 s, 7 to 9", 12) == [(2.0, 5.0), (7.0, 9.0)]
    assert ann.parse_spans("guitar near the end", 12) is None
    assert ann.parse_spans("0:05-0:40", 12) is None  # beyond the excerpt


def excerpt_rows(rid="r-0123456789ab"):
    rows = []
    for i, (category, start) in enumerate(
        (
            ("core", 0),
            ("core", 20),
            ("core", 40),
            ("core", 60),
            ("feature", 80),
            ("model_flagged", 100),
        )
    ):
        rows.append(
            {
                "excerpt_id": f"{rid}-{category[0]}{i}",
                "recording_id": rid,
                "excerpt_category": category,
                "excerpt_duration_s": "12",
                "source_start_s": str(start),
                "source_end_s": str(start + 12),
            }
        )
    return rows


def test_human_annotations_csv_has_no_model_fields_and_starts_blank(tmp_path):
    store = ann.AnnotationStore(tmp_path)
    rows = store.write_csv(excerpt_rows(), tmp_path / "Human_Annotations.csv")
    assert all(r["guitar_presence"] == "" and r["annotation_status"] == "unreviewed" for r in rows)
    header = next(csv.reader(open(tmp_path / "Human_Annotations.csv", encoding="utf-8-sig")))
    assert not [c for c in header if c.startswith("model") or "score" in c]


def test_corpus_export_keeps_intermittent_and_uncertain_out_of_certain_time(tmp_path):
    from gmv.corpus.schema import CorpusManifest, Recording

    digest = "0123456789ab" + "0" * 52
    manifest = tmp_path / "corpus" / "manifest.json"
    registry.save_manifest(
        manifest,
        CorpusManifest(
            corpus_version="0.1.0",
            recordings=(
                Recording(
                    recording_id="r-0123456789ab",
                    sha256=digest,
                    duration_s=200,
                    sample_rate=RATE,
                    channels=2,
                    source_kind="owned_local",
                    license="test",
                    permission_basis="test",
                    group_id="artist-x",
                ),
            ),
        ),
    )
    rows = excerpt_rows()
    store = ann.AnnotationStore(tmp_path)
    answers = (
        "present_throughout",
        "absent",
        "present_intermittent",
        "uncertain",
        "present_throughout",
        "absent",
    )
    for row, answer in zip(rows, answers, strict=True):
        store.save(
            row["excerpt_id"],
            row["recording_id"],
            {"guitar_presence": answer, "intermittent_spans_note": "0:02-0:05"},
        )
    [path] = ann.export_corpus(store, rows, manifest)
    exported = registry.load_annotation(path)
    assert exported.reference_kind == "human_annotation" and exported.status == "reviewed"
    assert [(r.start, r.end) for r in exported.coverage.ranges] == [
        (0, 12),
        (20, 32),
        (40, 52),
    ]  # core only
    assert [(a.start, a.certainty) for a in exported.activity] == [
        (0, "certain"),
        (40, "uncertain"),
    ]
    spec = importlib.util.spec_from_file_location(
        "train_activity", Path("scripts/train_activity.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure(("guitar",))
    times = np.array([5.0, 25.0, 45.0, 65.0, 85.0, 150.0])
    active, known = module.frame_labels(
        manifest, registry.load_manifest(manifest).get("r-0123456789ab"), times
    )
    assert active[:, 0].tolist() == [True, False, False, False, False, False]
    # throughout and absent are known; intermittent, uncertain, other categories and the rest are not
    assert known[:, 0].tolist() == [True, True, False, False, False, False]


# ------------------------------------------------------------ batch + build
def write_collection(tmp_path):
    root = tmp_path / "music"
    specs = [
        ("Band A", "One", "Album 1", 90, 20, 70),
        ("Band A", "Two", "Album 1", 90, 0, 90),
        ("Band A", "Three", "Album 2", 90, 10, 50),
        ("Ambient", "Calm", "Score", 80, 200, 300),
        ("Ambient", "Calm (Remix)", "Score", 85, 200, 300),
    ]
    rows = []
    for i, (artist, title, album, seconds, a, b) in enumerate(specs):
        path = root / artist / f"{title}.flac"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, song(seconds, a, b, seed=i), RATE, subtype="PCM_16")
        rows.append(
            {
                "file_path": str(path),
                "relative_path": str(Path(artist) / f"{title}.flac"),
                "top_folder": artist,
                "artist": artist,
                "title": title,
                "album": album,
                "sha256": sha(path),
                "duration_s": str(seconds),
                "codec": "flac",
                "extension": "flac",
                "bitrate_kbps": "",
                "bit_depth": "16",
                "exact_duplicate_is_first": "",
                "alternate_version_group": "V001" if artist == "Ambient" else "",
                "guitar_presence_estimate": "High" if artist == "Band A" else "Low",
                "model_guitar_share_detected": "0.5",
            }
        )
    inventory = tmp_path / "inventory.csv"
    with inventory.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "name": "test-batch",
                "seed": "s",
                "max_per_album": 1,
                "min_duration_s": 60,
                "picks": [
                    {
                        "path": str(Path("Band A") / "Two.flac"),
                        "category": "positive_candidate",
                        "reason": "named",
                    }
                ],
                "quotas": [
                    {
                        "folder": "Band A",
                        "count": 3,
                        "category": "positive_candidate",
                        "reason": "draw",
                    },
                    {
                        "folder": "Ambient",
                        "count": 2,
                        "category": "negative_candidate",
                        "reason": "draw",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return root, inventory, selection, rows


def test_selection_is_reproducible_spreads_albums_and_takes_one_per_version_group(tmp_path):
    _, inventory, selection, _ = write_collection(tmp_path)
    inv = batch_tools.read_inventory(inventory)
    spec = batch_tools.load_spec(selection)
    first, problems = batch_tools.select(inv, spec)
    second, _ = batch_tools.select(inv, spec)
    assert [r["sha256"] for r in first] == [r["sha256"] for r in second]
    band = [r for r in first if r["top_folder"] == "Band A"]
    assert len(band) == 2  # the named pick plus one draw from the other album (album cap 1)
    assert sorted(r["album"] for r in band) == ["Album 1", "Album 2"]
    ambient = [r for r in first if r["top_folder"] == "Ambient"]
    assert len(ambient) == 1  # the remix shares a version group
    assert any("short" in p for p in problems)
    assert {batch_tools.group_of(r) for r in band} == {"artist-band-a"}


def test_build_end_to_end_keeps_labels_empty_and_sources_untouched(tmp_path):
    root, inventory, selection, rows = write_collection(tmp_path)
    hashes = {r["file_path"]: r["sha256"] for r in rows}
    workdir = tmp_path / "work"
    manifest = tmp_path / "corpus" / "manifest.json"
    build(
        inventory, selection, workdir, None, manifest, backend=ToneIsGuitar(), log=lambda *_: None
    )
    batch, excerpts = load_tables(workdir)
    assert len(batch) == 3 and all(r["processing_status"] == "ok" for r in batch)
    core = [e for e in excerpts if e["excerpt_category"] == "core"]
    assert len(core) == 15
    assert all(e["excerpt_matches_source_samples"] == "true" for e in excerpts)
    assert all(e["excerpt_resampled"] == "false" for e in excerpts)
    for e in excerpts:
        start = int(e["source_start_sample"])
        assert abs(start / int(e["source_sample_rate_hz"]) - float(e["source_start_s"])) < 1e-3
    human = list(csv.DictReader(open(workdir / "Human_Annotations.csv", encoding="utf-8-sig")))
    assert len(human) == len(excerpts)
    assert all(h["guitar_presence"] == "" and h["annotation_status"] == "unreviewed" for h in human)
    assert not (workdir / "annotations.jsonl").exists()  # the build never writes a label
    timeline = list(
        csv.DictReader(open(workdir / "Guitar_Activity_Timelines.csv", encoding="utf-8-sig"))
    )
    assert timeline and {t["output_kind"] for t in timeline} == {"model_prediction"}
    corpus = registry.load_manifest(manifest)
    assert {r.source_kind for r in corpus.recordings} == {"owned_local"}
    assert {r.split for r in corpus.recordings} == {"unassigned"}
    assert {r.group_id for r in corpus.recordings} == {"artist-band-a", "artist-ambient"}
    assert all(sha(path) == digest for path, digest in hashes.items())
    result = verify(workdir)
    assert result["passed"] and result["excerpts_checked"] == len(excerpts)
    # rerun is resumable and gives the same excerpts
    build(
        inventory, selection, workdir, None, manifest, backend=ToneIsGuitar(), log=lambda *_: None
    )
    assert [e["excerpt_sha256"] for e in load_tables(workdir)[1]] == [
        e["excerpt_sha256"] for e in excerpts
    ]
    report.write_report(workdir)
    text = (workdir / "Validation_Batch_Report.md").read_text(encoding="utf-8")
    assert "No excerpts have been labelled yet" in text


def test_scoring_keeps_categories_apart(tmp_path):
    root, inventory, selection, _ = write_collection(tmp_path)
    workdir = tmp_path / "work"
    build(
        inventory,
        selection,
        workdir,
        None,
        tmp_path / "c" / "manifest.json",
        backend=ToneIsGuitar(),
        log=lambda *_: None,
    )
    _, excerpts = load_tables(workdir)
    store = ann.AnnotationStore(workdir)
    for e in excerpts:
        # label every core excerpt absent, every other category present throughout
        answer = "absent" if e["excerpt_category"] == "core" else "present_throughout"
        store.save(e["excerpt_id"], e["recording_id"], {"guitar_presence": answer})
    pools, _ = report.evaluate(workdir, store)
    assert pools["core"]["levels"]["detected"]["positive_hops"] == 0
    assert pools["core"]["levels"]["detected"]["negative_hops"] > 0
    for other in ("feature", "model_flagged"):
        assert pools[other]["levels"]["detected"]["negative_hops"] == 0
    report.write_report(workdir)
    text = (workdir / "Validation_Batch_Report.md").read_text(encoding="utf-8")
    assert "Primary result: core excerpts" in text and "diagnostic only" in text


# --------------------------------------------------------------- playback
class FakeStream:
    def __init__(self, callback):
        self.callback, self.active = callback, False

    def start_stream(self):
        self.active = True

    def stop_stream(self):
        self.active = False

    def is_active(self):
        return self.active

    def close(self):
        self.active = False


class FakeOutput:
    def __init__(self):
        self.streams = []

    def open(self, rate, channels, callback):
        stream = FakeStream(callback)
        self.streams.append((rate, channels, stream))
        return stream

    def complete(self):
        return 1

    def proceed(self):
        return 0

    def close(self):
        pass


def test_player_plays_pauses_replays_and_seeks(tmp_path):
    path = tmp_path / "x.wav"
    sf.write(path, song(2, 0, 2), RATE, subtype="FLOAT")
    output = FakeOutput()
    from gmv.review.player import Player

    player = Player(output)
    player.load(path)
    player.play()
    assert player.playing
    data, flag = output.streams[-1][2].callback(None, 4000, None, 0)
    assert len(data) == 4000 * 2 * 4 and flag == 0 and abs(player.position_s - 0.5) < 1e-6
    player.pause()
    assert not player.playing
    player.seek(1.9)
    _, flag = output.streams[-1][2].callback(None, 4000, None, 0)
    assert flag == 1  # reached the end
    player.replay()
    assert player.position_s == 0 and player.playing


# ------------------------------------------------------------------ window
@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_window_starts_empty_hides_model_and_resumes_after_restart(app, tmp_path):
    from gmv.review.window import ReviewWindow

    _, inventory, selection, _ = write_collection(tmp_path)
    workdir = tmp_path / "work"
    build(
        inventory,
        selection,
        workdir,
        None,
        tmp_path / "c" / "manifest.json",
        backend=ToneIsGuitar(),
        log=lambda *_: None,
    )
    window = ReviewWindow(
        workdir, player=__import__("gmv.review.player", fromlist=["Player"]).Player(FakeOutput())
    )
    first = window.current["excerpt_id"]
    assert window.current["excerpt_category"] == "core"
    assert window._value(window.presence) is None and window._value(window.gtype) is None
    assert not window.spans.isEnabled()
    assert window.model_text.isHidden() and not window.model_hidden.isHidden()
    window._press(window.presence, "present_intermittent")
    assert window.spans.isEnabled()
    window.spans.setText("0:01-0:06")
    window._press(window.presence, "uncertain")
    window.difficulty.setText("guitar or synth pad?")
    window.instruments["synthesizer"].setChecked(True)
    entry = window.save_current()
    assert entry["guitar_presence"] == "uncertain" and entry["other_instruments"] == ["synthesizer"]
    assert window.model_hidden.isHidden() and not window.model_text.isHidden()
    window.close()

    again = ReviewWindow(
        workdir, player=__import__("gmv.review.player", fromlist=["Player"]).Player(FakeOutput())
    )
    assert again.current["excerpt_id"] == first  # resumes where the reviewer was
    assert again._value(again.presence) == "uncertain"
    assert again.difficulty.text() == "guitar or synth pad?"
    again.filter.setCurrentIndex(
        [k for k, _ in __import__("gmv.review.window", fromlist=["FILTERS"]).FILTERS].index(
            "uncertain"
        )
    )
    assert again.list.count() == 1
    again._press(again.presence, "absent")
    again.save_current()
    assert [e["guitar_presence"] for e in again.store.history(first)] == ["uncertain", "absent"]
    human = list(csv.DictReader(open(workdir / "Human_Annotations.csv", encoding="utf-8-sig")))
    assert next(h for h in human if h["excerpt_id"] == first)["revisions_total"] == "2"
    again.close()
