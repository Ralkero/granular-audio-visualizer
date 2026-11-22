"""Evidence enrichment: identity, credits, dataset references, fusion and the review queue.
No network and no real recordings: MusicBrainz answers are canned."""

import json
import urllib.error

import numpy as np
import pytest

from gmv.evidence import fingerprint, fusion, openmic, queue
from gmv.evidence import musicbrainz as mb


def recording(mbid, title, artist="The Band", length=200000, release="The Album", **more):
    return {
        "id": mbid,
        "title": title,
        "length": length,
        "artist-credit": [{"name": artist, "artist": {"id": "a" * 36}}],
        "releases": [{"id": "e" * 36, "title": release, "release-group": {"id": "g" * 36}}],
        **more,
    }


class Canned:
    """Answers by a fragment of the URL; counts calls."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def __call__(self, url):
        self.calls.append(url)
        for fragment, body in self.answers.items():
            if fragment in url:
                if body is None:
                    raise urllib.error.HTTPError(url, 404, "not found", None, None)
                return body
        raise urllib.error.HTTPError(url, 404, "not found", None, None)


def client(tmp_path, answers):
    fetch = Canned(answers)
    return mb.Client(tmp_path / "cache", interval=0, fetch=fetch), fetch


ROW = {
    "title": "Song",
    "artist": "The Band",
    "album": "The Album",
    "album_artist": "",
    "duration_s": "200.4",
    "isrc": "",
    "musicbrainz_recording_id": "",
    "musicbrainz_album_id": "",
}
ID1, ID2 = "1" * 8 + "-1111-1111-1111-" + "1" * 12, "2" * 8 + "-2222-2222-2222-" + "2" * 12


# ---------------------------------------------------------------------- identity
def test_a_tagged_recording_id_is_looked_up_once_and_cached(tmp_path):
    api, fetch = client(tmp_path, {f"recording/{ID1}": recording(ID1, "Song")})
    row = dict(ROW, musicbrainz_recording_id=ID1)
    found = mb.identify(api, row)
    assert (found["status"], found["method"], found["confidence"]) == ("matched", "tag_mbid", "high")
    assert found["chosen"]["recording_mbid"] == ID1
    mb.identify(api, row)
    assert len(fetch.calls) == 1  # the second time came from the cache
    assert "User-Agent" not in fetch.calls[0] and mb.USER_AGENT.startswith("GranularMusicVisualizer/")


def test_a_search_match_needs_title_artist_and_length_to_agree(tmp_path):
    hits = {"recordings": [recording(ID1, "Song"), recording(ID2, "Another Song", length=90000)]}
    api, _ = client(tmp_path, {"query=": hits})
    found = mb.identify(api, ROW)
    assert found["status"] == "matched" and found["method"] == "search"
    assert found["chosen"]["recording_mbid"] == ID1 and found["confidence"] == "high"
    assert [a["recording_mbid"] for a in found["alternatives"]] == [ID2]


def test_versions_are_not_silently_conflated(tmp_path):
    # Same title and artist; one studio take and one live take of very different length.
    hits = {
        "recordings": [
            recording(ID1, "Song", length=200000),
            recording(ID2, "Song", length=200500, disambiguation="live"),
        ]
    }
    api, _ = client(tmp_path, {"query=": hits})
    found = mb.identify(api, ROW)
    assert found["status"] == "ambiguous" and found["chosen"] is None
    assert len(found["alternatives"]) == 2
    # A file that says it is live is not matched to the studio take either.
    live_row = dict(ROW, title="Song (Live)")
    only_studio = {"recordings": [recording(ID1, "Song")]}
    api, _ = client(tmp_path / "b", {"query=": only_studio})
    assert mb.identify(api, live_row)["status"] != "matched"


def test_nothing_found_stays_unmatched(tmp_path):
    api, _ = client(tmp_path, {"query=": {"recordings": []}})
    found = mb.identify(api, ROW)
    assert found["status"] == "unmatched" and found["chosen"] is None and found["confidence"] == "none"


def test_requests_are_spaced_and_back_off_when_told_to(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(mb.time, "sleep", waits.append)
    state = {"n": 0}

    def flaky(url):
        state["n"] += 1
        if state["n"] == 1:
            raise urllib.error.HTTPError(url, 503, "slow down", None, None)
        return {"recordings": []}

    api = mb.Client(tmp_path / "cache", interval=1.1, fetch=flaky)
    api.get("recording", query="x")
    api.get("recording", query="y")
    assert state["n"] == 3 and 2.0 in waits  # one retry after a 503, with a back-off
    assert any(0 < w <= 1.1 for w in waits)  # and never two requests within the interval


# ----------------------------------------------------------------------- credits
def relation(kind, name, attributes=(), direction="backward"):
    return {
        "type": kind,
        "target-type": "artist",
        "direction": direction,
        "attributes": list(attributes),
        "artist": {"name": name, "id": "p" * 36},
    }


def test_credit_scopes_stay_distinct(tmp_path):
    release = {
        "relations": [relation("instrument", "Album Guitarist", ["guitar"])],
        "media": [
            {
                "tracks": [
                    {"recording": {"id": ID1, "relations": [relation("instrument", "Track Player", ["electric guitar", "guest"])]}},
                    {"recording": {"id": ID2, "relations": [relation("instrument", "Pianist", ["piano"])]}},
                ]
            }
        ],
    }
    api, _ = client(tmp_path, {"release/": release})
    whole, per_recording = mb.release_credits(api, "e" * 36)
    assert [c["scope"] for c in whole] == ["release"]
    first = per_recording[ID1]
    assert first[0]["scope"] == "recording" and first[0]["guitar_kind"] == "electric"
    assert first[0]["qualifiers"] == ["guest"] and first[0]["source_url"].endswith(ID1)
    # The album-level guitarist does not become a credit of the piano track.
    piano_track = fusion.credit_summary(whole + per_recording[ID2])
    assert piano_track["scope"] == "release_level_guitar_credit"
    assert fusion.credit_summary(whole + first)["scope"] == "recording_level_guitar_credit"


def test_bass_guitar_is_not_guitar_and_a_bare_performer_names_no_instrument():
    found = mb.credits_from(
        [relation("instrument", "B", ["bass guitar"]), relation("performer", "P")],
        "recording",
        ID1,
        "url",
        "now",
    )
    assert [c["family"] for c in found] == ["bass", "unspecified"]
    summary = fusion.credit_summary(found)
    assert summary["scope"] == "credits_without_guitar" and summary["unspecified_performers"] == 1


# ----------------------------------------------------------------------- OpenMIC
def test_openmic_unasked_instruments_are_unknown_not_absent(tmp_path):
    root = tmp_path / "openmic-2018"
    (root / "partitions").mkdir(parents=True)
    (root / "openmic-2018-metadata.csv").write_text(
        "sample_key,artist_id,track_id,license_title,license_url\n"
        "000001_0,7,1,Attribution,http://cc/by\n000002_0,8,2,Attribution-NonCommercial,http://cc/by-nc\n",
        encoding="utf-8",
    )
    (root / "partitions" / "split01_train.csv").write_text("000001_0\n", encoding="utf-8")
    (root / "partitions" / "split01_test.csv").write_text("000002_0\n", encoding="utf-8")
    (root / "openmic-2018-aggregated-labels.csv").write_text(
        "sample_key,instrument,relevance,num_responses\n"
        "000001_0,guitar,1.0,3\n000001_0,piano,0.0,3\n000002_0,piano,0.6,3\n000002_0,violin,1.0,2\n",
        encoding="utf-8",
    )
    clips = openmic.read(tmp_path)
    assert clips["000001_0"]["labels"]["guitar"]["verdict"] == "present"
    assert clips["000001_0"]["labels"]["piano"]["verdict"] == "absent"
    assert "guitar" not in clips["000002_0"]["labels"]  # never asked: unknown
    assert clips["000002_0"]["labels"]["piano"]["verdict"] == "uncertain"  # low agreement
    assert clips["000002_0"]["labels"]["violin"]["verdict"] == "uncertain"  # too few answers
    rows = openmic.label_rows(clips)
    assert not any(r["sample_key"] == "000002_0" and r["instrument"] == "guitar" for r in rows)
    assert {r["split"] for r in rows} == {"train", "test"}
    assert {r["audio_licence"] for r in rows} == {"Attribution", "Attribution-NonCommercial"}
    counts = openmic.summary(clips)["by_instrument"]["guitar"]
    assert counts == {"present": 1, "unknown": 1}


# ------------------------------------------------------------------------ fusion
def models(seconds, own, panns, level=-20.0):
    count = round(seconds / fusion.HOP_S)
    return (
        np.full(count, own, np.float32),
        np.full(count, panns, np.float32),
        np.full(count, level, np.float32),
    )


def fuse(own, panns, credits=(), confidence="high", human=(), separation=None, seconds=60):
    gmv, second, level = models(seconds, own, panns)
    return fusion.candidates_for(
        {
            "id": "r-test",
            "gmv": gmv,
            "panns": second,
            "level_db": level,
            "credits": fusion.credit_summary(list(credits)),
            "match_confidence": confidence,
            "human": list(human),
            "separation": separation,
        }
    )


GUITAR_CREDIT = {"scope": "recording", "family": "guitar", "performer": "G", "instrument": "guitar", "guitar_kind": "unspecified"}


def test_agreement_of_both_models_is_a_pseudo_label_never_a_reference():
    rows, classes = fuse(0.99, 0.8)
    assert [r["label"] for r in rows] == ["present"]
    assert rows[0]["confidence_class"] == "moderate_agreement" and rows[0]["evidence_category"] == "B"
    assert rows[0]["start_s"] == 10.0  # the recogniser's first ten seconds are left undecided
    assert (classes[:20] == 0).all()
    assert not any(r["evidence_category"] == "A" for r in rows)


def test_a_recording_level_credit_strengthens_presence_only():
    rows, _ = fuse(0.99, 0.8, credits=[GUITAR_CREDIT])
    assert rows[0]["confidence_class"] == "strong_agreement"
    assert "recording-level guitar credit" in rows[0]["supporting_sources"]
    # A poorly matched identity lends its credits no weight.
    rows, _ = fuse(0.99, 0.8, credits=[GUITAR_CREDIT], confidence="low")
    assert rows[0]["confidence_class"] == "moderate_agreement"
    # Models hearing no guitar are not contradicted or supported by any credit.
    rows, _ = fuse(0.01, 0.001, credits=[GUITAR_CREDIT])
    assert rows[0]["label"] == "absent" and rows[0]["confidence_class"] == "weak_evidence"
    assert not rows[0]["training_suitable"]  # two models hearing nothing is not a negative label


def test_missing_credits_and_single_models_never_make_a_label():
    assert fuse(0.99, 0.05)[0] == []  # one model alone
    assert fuse(0.5, 0.5)[0] == []  # neither sure
    rows, classes = fuse(0.99, 0.001)  # contradiction
    assert [r["confidence_class"] for r in rows] == ["conflict"]
    assert rows[0]["label"] == "unknown" and not rows[0]["training_suitable"]
    # No credits at all: "absent" still only ever comes from the audio models agreeing.
    summary = fusion.credit_summary([])
    assert summary["scope"] == "no_instrument_credits"
    rows, _ = fuse(0.5, 0.5, credits=[])
    assert rows == []


def test_short_agreements_and_silence_are_not_labelled():
    gmv, second, level = models(60, 0.5, 0.5)
    gmv[40:48], second[40:48] = 0.99, 0.9  # four seconds of agreement
    level[60:] = -90.0
    gmv[60:], second[60:] = 0.0, 0.0  # silence is not "guitar absent" evidence
    rows, classes = fusion.candidates_for(
        {"id": "r", "gmv": gmv, "panns": second, "level_db": level, "credits": fusion.credit_summary([]), "match_confidence": "none", "human": []}
    )
    assert rows == [] and (classes[60:] == -9).all()


def test_human_labels_win_and_intermittent_is_not_given_times():
    human = [(20.0, 32.0, "absent", ""), (40.0, 52.0, "present_intermittent", "clean_electric")]
    rows, _ = fuse(0.99, 0.9, human=human)
    reference = [r for r in rows if r["evidence_category"] == "A"]
    assert [(r["label"], r["training_suitable"]) for r in reference] == [("absent", True), ("intermittent", False)]
    assert "not timed" in reference[1]["temporal_precision"]
    # The models' "present" run that covers the reviewed stretches is not offered against them.
    automatic = [r for r in rows if r["evidence_category"] == "B"]
    assert all(r["end_s"] <= 20.0 or r["start_s"] >= 52.0 for r in automatic) or automatic == []


def test_separation_can_support_or_contradict():
    count = round(60 / fusion.HOP_S)
    rows, _ = fuse(0.01, 0.001, separation=np.full(count, -50.0))
    assert rows[0]["confidence_class"] == "moderate_agreement" and rows[0]["label"] == "absent"
    assert fuse(0.99, 0.8, separation=np.full(count, -5.0))[0][0]["confidence_class"] == "strong_agreement"
    rows, _ = fuse(0.01, 0.001, separation=np.full(count, -5.0))
    assert rows[0]["confidence_class"] == "conflict" and rows[0]["review_required"]


# ------------------------------------------------------------------------- queue
def for_queue(own, panns, others=0.0, credits=(), rid="r-1", artist="A", seconds=120, human=()):
    gmv, second, level = models(seconds, own, panns)
    summary = fusion.credit_summary(list(credits))
    recording = {"id": rid, "gmv": gmv, "panns": second, "level_db": level, "credits": summary, "match_confidence": "high", "human": list(human)}
    found, classes = fusion.candidates_for(recording)
    count = len(gmv)
    by_class = {k: np.zeros(count, np.float32) for k in ("piano", "synthesizer", "string_section", "violin", "organ", "acoustic_guitar", "electric_guitar")}
    by_class["guitar"] = second
    by_class["piano"][:] = others
    return {
        "id": rid,
        "row": {"top_folder": artist, "artist": artist, "title": "T", "relative_path": f"{artist}\\{rid}.mp3", "bitrate_kbps": "320"},
        "classes": classes,
        "gmv": {"guitar": gmv, "distorted": gmv * 0, "clean": gmv * 0, "acoustic": gmv * 0},
        "panns": by_class,
        "credits": summary,
        "view": fusion.recording_view(classes, summary, found),
        "candidates": found,
        "human": list(human),
        "duration": float(seconds),
        "trained_artist": False,
    }


def test_queue_holds_disagreements_and_an_audit_sample_and_skips_reviewed_time():
    fooled = for_queue(0.99, 0.001, others=0.8, rid="r-piano")
    agreed = for_queue(0.99, 0.9, rid="r-agree", artist="B")
    reviewed = for_queue(0.99, 0.001, others=0.8, rid="r-done", artist="C", human=[(0.0, 120.0, "absent", "")])
    rows = queue.build([fooled, agreed, reviewed])
    by_recording = {r["recording_id"]: r for r in rows}
    assert by_recording["r-piano"]["queue"] == "possible_false_positive"
    assert by_recording["r-agree"]["queue"] == "random_audit"
    assert "r-done" not in by_recording
    assert all(r["end_s"] - r["start_s"] == 12.0 for r in rows)
    assert queue.build([fooled, agreed, reviewed]) == rows  # seeded: the same every time
    spec = queue.selection(rows)
    assert {p["path"] for p in spec["picks"]} == {"A\\r-piano.mp3", "B\\r-agree.mp3"}
    assert spec["picks"][0]["excerpts"][0]["queue"] in queue.LIMITS


def test_queue_limits_one_artist(tmp_path):
    many = [for_queue(0.99, 0.001, others=0.8, rid=f"r-{i}", artist="Same") for i in range(20)]
    rows = queue.build(many)
    assert len(rows) == queue.PER_ARTIST


# ------------------------------------------------------------------ fingerprints
def test_fingerprint_similarity_finds_the_same_audio_with_an_offset():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2**32, 400, dtype=np.uint64).astype(np.uint32)
    shifted = np.concatenate([rng.integers(0, 2**32, 7, dtype=np.uint64).astype(np.uint32), a])
    other = rng.integers(0, 2**32, 400, dtype=np.uint64).astype(np.uint32)
    assert fingerprint.similarity(a, shifted) > 0.99
    assert 0.4 < fingerprint.similarity(a, other) < 0.6


def test_acoustid_lookup_refuses_without_approval(tmp_path):
    with pytest.raises(PermissionError, match="acoustid.org"):
        fingerprint.lookup("inventory.csv", tmp_path, "fpcalc", "key")
    with pytest.raises(ValueError, match="application key"):
        fingerprint.lookup("inventory.csv", tmp_path, "fpcalc", "", approved=True)


def test_acoustid_answers_are_only_assigned_when_one_recording_is_clearly_meant():
    def answer(*results):
        return {"status": "ok", "results": list(results)}

    def result(score, *recordings):
        found = [{"id": r, "duration": d} for r, d in recordings]
        return {"id": "t", "score": score, "recordings": found}

    assert fingerprint.read_lookup(answer(), 200)[0] == "unmatched"
    assert fingerprint.read_lookup(answer(result(0.97, ("one", 201))), 200)[0] == "matched"
    # Two different recordings share the fingerprint: both are kept, neither is assigned.
    status, candidates = fingerprint.read_lookup(
        answer(result(0.97, ("one", 201), ("two", 199))), 200
    )
    assert status == "ambiguous" and {c["recording_mbid"] for c in candidates} == {"one", "two"}
    assert fingerprint.read_lookup(answer(result(0.6, ("one", 201))), 200)[0] == "ambiguous"
    assert fingerprint.read_lookup(answer(result(0.97, ("one", 320))), 200)[0] == "ambiguous"


def test_acoustid_sends_only_fingerprint_duration_and_key(tmp_path, monkeypatch):
    from gmv.evidence import write_jsonl

    inventory = tmp_path / "inventory.csv"
    lines = ["sha256,file_path,title", "a" * 64 + ",private name.mp3,Secret", "b" * 64 + ",known.mp3,Known"]
    inventory.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_jsonl(
        tmp_path / "musicbrainz" / "identity.jsonl",
        [
            {"recording_id": "r-" + "a" * 12, "status": "unmatched"},
            {"recording_id": "r-" + "b" * 12, "status": "matched"},
        ],
    )

    class Done:
        returncode = 0
        stdout = json.dumps({"duration": 200.4, "fingerprint": "AQAA"}).encode()

    monkeypatch.setattr(fingerprint.subprocess, "run", lambda *a, **k: Done())
    sent = []

    def post(url, fields):
        sent.append((url, fields))
        return {"status": "ok", "results": []}

    first = fingerprint.lookup(
        inventory, tmp_path, "fpcalc", "KEY", approved=True, post=post, interval=0
    )
    assert first == {"eligible": 1, "sent_this_run": 1, "results": {"unmatched": 1}}
    url, fields = sent[0]
    assert url == fingerprint.ACOUSTID_URL and "submit" not in url
    assert fields == {"client": "KEY", "duration": "200", "fingerprint": "AQAA", "meta": "recordings"}
    again = fingerprint.lookup(
        inventory, tmp_path, "fpcalc", "KEY", approved=True, post=post, interval=0
    )
    assert again["sent_this_run"] == 0 and len(sent) == 1  # answered from the cache


# ------------------------------------------------------------- nothing personal
def test_the_work_folder_is_ignored_by_git():
    from pathlib import Path

    ignored = (Path(__file__).parents[1] / ".gitignore").read_text(encoding="utf-8").split()
    assert "research-local/" in ignored and "review-local/" in ignored


def test_cached_answers_are_plain_json_with_their_source(tmp_path):
    api, _ = client(tmp_path, {"query=": {"recordings": []}})
    record = api.get("recording", query="x")
    stored = json.loads(next((tmp_path / "cache").glob("*.json")).read_text(encoding="utf-8"))
    assert stored["url"] == record["url"] and stored["retrieved"] and stored["status"] == "ok"


# ----------------------------------------------------------------------- leakage
def test_same_audio_in_two_artist_folders_joins_them_for_every_recording():
    from gmv.evidence import leakage

    def row(name, folder, **more):
        return {"sha256": name * 64, "top_folder": folder, "artist": folder, **more}

    rows = [
        row("a", "Band One"),
        row("b", "Band One"),
        row("c", "Soundtrack"),
        row("d", "Soundtrack", alternate_version_group="v1"),
        row("e", "Band Three", alternate_version_group="v1"),
        row("f", "Unrelated"),
    ]
    rid = {name: "r-" + name * 12 for name in "abcdef"}
    same_audio = [{"group": "same-audio-001", "recordings": [rid["a"], rid["c"]]}]
    found = leakage.groups(rows, same_audio)
    joined = {found[rid[name]][0] for name in "abcde"}
    assert len(joined) == 1 and next(iter(joined)).startswith("merged-")
    # "b" never matched anything itself, but shares a folder with "a": it moves too.
    assert found[rid["b"]][0] == found[rid["c"]][0]
    assert found[rid["f"]][0] == "artist-unrelated" and found[rid["f"]][1] == []
    assert any("same audio" in reason for reason in found[rid["a"]][1])
    # A reliably matched identical MusicBrainz recording joins folders as well.
    mbid = "9" * 36
    identities = [
        {"recording_id": rid["f"], "recording_mbid": mbid, "confidence": "high"},
        {"recording_id": rid["b"], "recording_mbid": mbid, "confidence": "high"},
    ]
    assert leakage.groups(rows, same_audio, identities)[rid["f"]][0].startswith("merged-")
    low = [dict(i, confidence="low") for i in identities]
    assert leakage.groups(rows, same_audio, low)[rid["f"]][0] == "artist-unrelated"
