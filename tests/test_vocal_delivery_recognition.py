"""Phase A4: the vocal-delivery recognition experiment. Generated audio and a generated
dataset only; the owner's answer files are never used."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from gmv.character.research.delivery import SUPPORTED, data, evaluate, features, models
from gmv.character.research.delivery import run as experiment
from gmv.character.research.labeling import dataset as labelling
from gmv.character.research.labeling import store

RATE = 22050
REFERENCE = "artist-reference-band"
POISON = "POISON-HELD-OUT-VALUE"


def pick_artists(development=6, evaluation=2):
    """Artist keys the fixed split rule puts on each side."""
    dev, held = [], []
    for number in range(200):
        name = f"artist-test-band-{number:02d}"
        side = labelling.side_of(name, (REFERENCE,))
        if side == "development" and len(dev) < development:
            dev.append(name)
        elif side == "evaluation" and len(held) < evaluation:
            held.append(name)
    return dev, held


def sound(kind, seconds, seed):
    """Stereo test audio: a clean harmonic voice-like tone, high noise, or low rough noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    if kind == "cleans":
        f0 = 220.0 * (1 + 0.01 * np.sin(2 * np.pi * 5 * t))
        phase = 2 * np.pi * np.cumsum(f0) / RATE
        mono = sum(np.sin(k * phase) / k for k in range(1, 10)) * 0.1
    elif kind == "high_screams":
        noise = rng.standard_normal(len(t))
        spectrum = np.fft.rfft(noise)
        freq = np.fft.rfftfreq(len(t), 1 / RATE)
        spectrum[(freq < 1500) | (freq > 6000)] = 0
        mono = np.fft.irfft(spectrum, len(t))
        mono = 0.2 * mono / np.abs(mono).max()
    else:
        noise = rng.standard_normal(len(t))
        spectrum = np.fft.rfft(noise)
        freq = np.fft.rfftfreq(len(t), 1 / RATE)
        spectrum[(freq < 80) | (freq > 900)] = 0
        mono = np.fft.irfft(spectrum, len(t)) * (1 + 0.8 * np.sin(2 * np.pi * 60 * t))
        mono = 0.2 * mono / np.abs(mono).max()
    mono = mono + 0.002 * rng.standard_normal(len(t))
    return np.stack([mono, mono], axis=1).astype(np.float32)


@pytest.fixture
def labelled(tmp_path):
    """A labelling folder: 6 development artists and 2 held-out ones, two recordings each,
    two excerpts per recording; answers given, held-out answers carry a poison value; a
    leakage note names both excerpts of one development recording."""
    folder = tmp_path / "local-evidence" / "sonic-character" / "vocal-labeling"
    folder.mkdir(parents=True)
    music = tmp_path / "music"
    music.mkdir()
    dev, held = pick_artists()
    kinds = ("cleans", "high_screams", "low_screams")
    recordings, excerpts, kind_of = {}, [], {}
    for a, artist in enumerate(dev + held):
        for r in range(2):
            identity = f"rec-{a:02d}-{r}"
            path = music / f"{identity}.wav"
            kind = kinds[(a + r) % 3]
            sf.write(path, sound(kind, 24.0, a * 10 + r), RATE)
            recordings[identity] = {
                "path": str(path), "artist": artist, "sha256": labelling.sha256(path), "source": "test",
                "reference": False, "size": path.stat().st_size, "duration_s": 24.0, "sample_rate": RATE,
                "channels": 2, "frames": int(24.0 * RATE),
            }
            kind_of[identity] = kind
            for start in (1.0, 12.0):
                excerpts.append({"recording": identity, "start_s": start, "end_s": start + 10.0,
                                 "selection": "even", "level_dbfs": -20.0})
    dataset = labelling.build(recordings, excerpts, set())
    labelling.write(folder, dataset)
    (folder / "titles.json").write_text(json.dumps({k: {"title": f"Tune number {k}"} for k in recordings}))
    side = data.sides(dataset)
    by_recording = {}
    for excerpt in dataset["excerpts"]:
        by_recording.setdefault(excerpt["recording"], []).append(excerpt["id"])
    labels = store.Labels(folder)
    lines = []
    for excerpt in dataset["excerpts"]:
        if side[excerpt["id"]] == "evaluation":
            lines.append(json.dumps({"time": "t", "excerpt": excerpt["id"], "event": "label", "value": POISON, "scheme": 2}))
        else:
            labels.label(excerpt["id"], kind_of[excerpt["recording"]])
    dev_recordings = [k for k in recordings if recordings[k]["artist"] in dev]
    special = {
        "no_vocals": by_recording[dev_recordings[1]][0],
        "gutturals": by_recording[dev_recordings[1]][1],
        "later": by_recording[dev_recordings[2]][0],
    }
    labels.skip(special["no_vocals"], "no_vocals")  # ignored: it already has an answer
    held_ids = [e for e, s in side.items() if s == "evaluation"]
    lines.append(json.dumps({"time": "t", "excerpt": held_ids[0], "event": "skip", "reason": "no_vocals"}))
    with (folder / store.LABELS).open("a", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    leak = dev_recordings[3]
    note = tmp_path / "local-evidence" / "sonic-character" / "expansion-audit"
    note.mkdir(parents=True)
    ids = by_recording[leak]
    (note / "leakage-watch.md").write_text(
        f"# Leakage watch\n\n- **{ids[0]} and {ids[1]}** come from \"Tune number {leak}\" (guest vocalist).\n"
    )
    return {
        "folder": folder, "dataset": dataset, "dev": dev, "held": held, "leak": ids, "kind_of": kind_of,
        "by_recording": by_recording, "special": special, "note": note / "leakage-watch.md",
    }


def reload(labelled):
    return data.load(labelled["folder"])


# -- reading the answers ----------------------------------------------------------------


def test_only_development_answers_are_decoded(labelled, monkeypatch):
    side = data.sides(labelled["dataset"])
    held = {e for e, s in side.items() if s == "evaluation"}
    decoded = []
    original = data._decode

    def spy(line):
        decoded.append(line)
        return original(line)

    monkeypatch.setattr(data, "_decode", spy)
    found = reload(labelled)
    assert decoded and not any(h in line for line in decoded for h in held)
    assert not any(POISON in line for line in decoded)
    public = {k: v for k, v in found.items() if k != "dataset"}
    assert POISON not in json.dumps(public, default=str)
    assert found["audit"]["held_out_values_decoded"] == 0
    assert found["audit"]["held_out_lines_not_decoded"] == len(held) + 1
    assert {e["excerpt"] for e in found["examples"]}.isdisjoint(held)


def test_the_held_out_side_is_counted_but_never_by_answer(labelled):
    found = reload(labelled)
    held = found["held_out"]
    assert held == {"excerpts": 8, "answered": 8, "no_vocals": 0, "review_later": 0, "neither": 0}
    assert set(held) == {"excerpts", "answered", "no_vocals", "review_later", "neither"}


def test_exclusions_by_reason_and_the_leakage_answers_as_recorded(labelled):
    folder = labelled["folder"]
    labels = store.Labels(folder)
    special = labelled["special"]
    # An excerpt answered "gutturals", one put aside, one never answered: none is used.
    labels.label(special["gutturals"], "gutturals")
    fresh_ids = [e for e in labelled["by_recording"][next(iter(labelled["by_recording"]))]]
    found = reload(labelled)
    used = {e["excerpt"] for e in found["examples"]}
    assert special["gutturals"] not in used and found["excluded"]["gutturals"] == 1
    assert found["excluded"]["vocalist_overlap"] == 2 and not used & set(labelled["leak"])
    recorded = {p["excerpt"]: p["recorded"] for p in found["private"]["excluded"]}
    leak_kind = labelled["kind_of"][labelled["dataset"]["excerpts"][[e["id"] for e in labelled["dataset"]["excerpts"]].index(labelled["leak"][0])]["recording"]]
    assert recorded[labelled["leak"][0]] == leak_kind  # the answer on record, not assumed
    assert all(e["label"] in SUPPORTED for e in found["examples"])
    assert fresh_ids  # the fixture has recordings


def test_no_vocals_review_later_and_unanswered_are_left_out(tmp_path, labelled):
    folder = labelled["folder"]
    dataset = labelled["dataset"]
    side = data.sides(dataset)
    dev = [e["id"] for e in dataset["excerpts"] if side[e["id"]] == "development" and e["id"] not in labelled["leak"]]
    events = (folder / store.LABELS).read_text().splitlines()
    kept = [line for line in events if not any(f'"{d}"' in line for d in dev[:3])]
    kept += [
        json.dumps({"time": "t", "excerpt": dev[0], "event": "skip", "reason": "no_vocals"}),
        json.dumps({"time": "t", "excerpt": dev[1], "event": "skip", "reason": "later"}),
        json.dumps({"time": "t", "excerpt": dev[2], "event": "label", "value": "clean", "scheme": 1}),
    ]
    (folder / store.LABELS).write_text("\n".join(kept) + "\n")
    found = reload(labelled)
    assert found["excluded"]["no_vocals"] == 1
    assert found["excluded"]["review_later"] == 1
    assert found["excluded"]["earlier_answer_only"] == 1
    assert {e["excerpt"] for e in found["examples"]}.isdisjoint(dev[:3])


@pytest.mark.parametrize("problem", ["missing", "wrong title", "one of two", "held out", "no excerpt"])
def test_a_missing_or_unverifiable_leakage_note_stops_the_experiment(labelled, problem):
    note = labelled["note"]
    a, b = labelled["leak"]
    side = data.sides(labelled["dataset"])
    held = next(e for e, s in side.items() if s == "evaluation")
    text = {
        "wrong title": f'- **{a} and {b}** come from "Another song"\n',
        "one of two": f'- **{a}** comes from "{note.read_text().split(chr(34))[1]}"\n',
        "held out": f'- **{held}** comes from "Tune"\n',
        "no excerpt": "Nothing named here.\n",
    }
    if problem == "missing":
        note.unlink()
    else:
        note.write_text(text[problem])
    with pytest.raises(data.DataError):
        reload(labelled)


def test_a_tampered_split_stops_the_experiment(labelled):
    path = labelled["folder"] / "dataset.json"
    dataset = json.loads(path.read_text())
    artist = labelled["held"][0]
    dataset["artists"][artist]["split"] = "development"
    path.write_text(json.dumps(dataset))
    with pytest.raises(data.DataError):
        reload(labelled)


def test_development_answers_follow_the_labelling_tools_own_rules(tmp_path):
    rng = np.random.default_rng(3)
    events = []
    for _ in range(400):
        excerpt = f"VOCAL-{int(rng.integers(1, 9)):04d}"
        kind = rng.choice(["label", "label_old", "skip_nv", "skip_later", "note"])
        if kind == "label":
            events.append({"time": "t", "excerpt": excerpt, "event": "label", "value": str(rng.choice(list(SUPPORTED) + ["gutturals"])), "scheme": 2})
        elif kind == "label_old":
            events.append({"time": "t", "excerpt": excerpt, "event": "label", "value": str(rng.choice(["clean", "scream", "none"]))})
        elif kind == "skip_nv":
            events.append({"time": "t", "excerpt": excerpt, "event": "skip", "reason": "no_vocals"})
        elif kind == "skip_later":
            events.append({"time": "t", "excerpt": excerpt, "event": "skip", "reason": "later"})
        else:
            events.append({"time": "t", "excerpt": excerpt, "event": "note", "value": "x"})
    (tmp_path / store.LABELS).write_text("\n".join(json.dumps(e) for e in events) + "\n")
    expected = store.Labels(tmp_path).state()
    found = data.fold_development(events)
    for excerpt, state in expected.items():
        assert found[excerpt]["label"] == state["label"]
        assert found[excerpt]["skip"] == (state["skip_reason"] if state["skipped"] else None)
        assert (found[excerpt]["earlier"] is not None) == (state["earlier"] is not None)


# -- groups and folds -------------------------------------------------------------------


def test_folds_never_share_a_group_artist_or_recording(labelled):
    found = reload(labelled)
    examples = found["examples"]
    plan = evaluate.folds([e["group"] for e in examples])
    assert len(plan) == len({e["group"] for e in examples})
    for _group, train, test in plan:
        evaluate.check_fold(examples, train, test)
        for key in ("group", "artist", "recording"):
            assert {examples[i][key] for i in train}.isdisjoint({examples[i][key] for i in test})
    other = next(i for i, e in enumerate(examples) if e["group"] != examples[0]["group"])
    broken = [dict(e) for e in examples]
    broken[other]["recording"] = broken[0]["recording"]  # one recording in two groups
    with pytest.raises(evaluate.LeakageError):
        evaluate.check_fold(broken, np.array([other]), np.array([0]))
    broken[other]["artist"] = broken[0]["artist"]
    with pytest.raises(evaluate.LeakageError):
        evaluate.check_fold(broken, np.array([other]), np.array([0]))


def test_a_person_credited_on_two_artists_recordings_joins_their_groups(labelled, tmp_path):
    dataset = labelled["dataset"]
    dev, held = labelled["dev"], labelled["held"]
    rows = []
    for identity, recording in dataset["recordings"].items():
        artist = recording["artist"]
        credit = labelling.artist_name(artist)
        if artist == dev[0] and identity.endswith("-0"):
            credit += " feat. Guest Singer"
        if artist == dev[1] and identity.endswith("-1"):
            credit += ", Guest Singer"
        if artist == held[0] and identity.endswith("-0"):
            credit += " & " + labelling.artist_name(dev[2])
        rows.append({"sha256": recording["sha256"], "artist": credit, "mb_artist": "", "title": "x"})
    inventory = tmp_path / "inventory.csv"
    with inventory.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    found = data.load(labelled["folder"], inventory=inventory)
    links = found["private"]["credit_links_development"]
    assert (min(dev[0], dev[1]), max(dev[0], dev[1]), "artist-guest-singer") in links
    groups = {e["artist"]: e["group"] for e in found["examples"]}
    assert groups[dev[0]] == groups[dev[1]]
    assert any(dev[2] in pair and held[0] in pair for pair in found["private"]["credit_links_across_sides"])
    for _g, train, test in evaluate.folds([e["group"] for e in found["examples"]]):
        evaluate.check_fold(found["examples"], train, test)


# -- features ---------------------------------------------------------------------------


def test_features_are_deterministic_and_unknown_is_never_zero():
    tone = sound("cleans", 10.0, 1)
    first, second = features.measure(tone, RATE), features.measure(tone, RATE)
    assert first.shape == (len(features.NAMES),) and np.array_equal(first, second, equal_nan=True)
    silent = features.measure(np.zeros((10 * RATE, 2), np.float32), RATE)
    assert np.isnan(silent).all()
    noise = features.measure(sound("high_screams", 10.0, 2), RATE)
    index = {n: i for i, n in enumerate(features.NAMES)}
    assert noise[index["voice_flatness"]] > first[index["voice_flatness"]]
    assert first[index["voice_harmonic_share"]] > noise[index["voice_harmonic_share"]]
    assert abs(first[index["pitch_median_midi"]] - 57.0) < 1.0  # 220 Hz is A3


def test_feature_names_carry_no_metadata_and_renaming_changes_nothing(labelled, tmp_path):
    banned = {"artist", "title", "album", "genre", "path", "file", "split", "id", "recording", "excerpt"}
    assert not any(banned & set(name.split("_")) for name in features.NAMES)
    found = reload(labelled)
    X, kept, dropped, _n = experiment.extract(found, tmp_path / "a", use_cache=False)
    renamed = json.loads(json.dumps(found["dataset"]))
    mapping = {k: f"other-{n}" for n, k in enumerate(renamed["recordings"])}
    renamed["recordings"] = {mapping[k]: v for k, v in renamed["recordings"].items()}
    for excerpt in renamed["excerpts"]:
        excerpt["recording"] = mapping[excerpt["recording"]]
        excerpt["id"] = excerpt["id"].replace("VOCAL", "OTHER")
    examples = [dict(e, excerpt=e["excerpt"].replace("VOCAL", "OTHER"), recording=mapping[e["recording"]],
                     artist="artist-renamed", group="x") for e in found["examples"]]
    Y, _kept, _dropped, _m = experiment.extract({"dataset": renamed, "examples": examples}, tmp_path / "b", use_cache=False)
    assert np.array_equal(X, Y, equal_nan=True) and not dropped


def test_unavailable_or_changed_audio_is_dropped_and_counted(labelled, tmp_path):
    found = reload(labelled)
    first = found["examples"][0]
    path, *_rest = data.source(found["dataset"], first["excerpt"])
    data_bytes = path.read_bytes()
    path.write_bytes(data_bytes + b"\0")  # size no longer matches: changed
    X, kept, dropped, _n = experiment.extract(found, tmp_path / "out", use_cache=False)
    gone = sum(1 for e in found["examples"] if e["recording"] == first["recording"])
    assert dropped == {"audio_unavailable": gone} and len(kept) == len(found["examples"]) - gone
    path.unlink()
    assert experiment.measure_one((str(path), 1.0, 11.0, len(data_bytes))) == "audio_unavailable"
    path.write_bytes(data_bytes)


def test_the_feature_cache_is_used_only_for_the_same_audio_and_version(labelled, tmp_path):
    found = reload(labelled)
    out = tmp_path / "out"
    X, _k, _d, measured = experiment.extract(found, out, use_cache=True)
    assert measured == len(found["examples"])
    Y, _k, _d, again = experiment.extract(found, out, use_cache=True)
    assert again == 0 and np.array_equal(X, Y, equal_nan=True)
    Z, _k, _d, fresh = experiment.extract(found, out, use_cache=False)
    assert fresh == len(found["examples"]) and np.array_equal(X, Z, equal_nan=True)


# -- models and validation --------------------------------------------------------------


def synthetic(n=120, groups=8, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(3, n, p=[0.6, 0.2, 0.2])
    X = rng.standard_normal((n, 6))
    X[:, 0] += 3.0 * y
    X[:, 1] += 3.0 * (y == 2)
    X[rng.random(X.shape) < 0.05] = np.nan
    examples = [{"group": f"g{i % groups}", "artist": f"g{i % groups}", "recording": f"r{i % groups}-{i // 16}"} for i in range(n)]
    return X, y, examples


def test_preprocessing_and_class_weights_come_from_the_training_fold_only(monkeypatch):
    X, y, examples = synthetic()
    seen = []
    original = models.Preprocess.fit

    def spy(self, data_rows):
        seen.append(np.array(data_rows))
        return original(self, data_rows)

    monkeypatch.setattr(models.Preprocess, "fit", spy)
    fitted = []
    evaluate.cross_validate(X, y, examples, "logistic", 1, fitted=fitted)
    plan = evaluate.folds([e["group"] for e in examples])
    for (group, train, test), rows, (fitted_group, prepare, _model, weights) in zip(plan, seen, fitted, strict=True):
        assert fitted_group == group
        assert np.array_equal(rows, X[train], equal_nan=True)
        assert np.allclose(prepare.mean[:6], np.where(np.isnan(X[train]), prepare.median, X[train]).mean(axis=0))
        assert np.allclose(weights, models.class_weights(y[train], 3))
        assert len(test)


def test_class_weights_balance_the_answers():
    y = np.array([0] * 6 + [1] * 2 + [2] * 4)
    weights = models.class_weights(y, 3)
    assert np.allclose(weights, [12 / 18, 12 / 6, 12 / 12])
    assert np.allclose(models.class_weights(np.array([0, 0, 2]), 3), [3 / 4, 0, 3 / 2])


def test_unknown_values_get_training_medians_and_an_indicator():
    train = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 8.0]])
    prepare = models.Preprocess().fit(train)
    assert np.allclose(prepare.median, [3.0, 6.0]) and list(prepare.indicators) == [1]
    out = prepare.transform(np.array([[np.nan, np.nan], [100.0, 0.0]]))
    assert out.shape == (2, 3)
    assert np.isclose(out[0, 0], (3.0 - 3.0) / train[:, 0].std())  # filled with the training median
    assert out[0, 2] > out[1, 2]  # the indicator marks the unknown value


def test_models_are_deterministic_and_beat_the_majority_on_a_clear_signal(monkeypatch):
    monkeypatch.setattr(models, "TREES", 60)
    X, y, examples = synthetic()
    scores = {}
    for name in models.MODELS:
        first, _ = evaluate.cross_validate(X, y, examples, name, 7)
        second, _ = evaluate.cross_validate(X, y, examples, name, 7)
        assert np.array_equal(first, second)
        assert np.allclose(first.sum(axis=1), 1.0)
        scores[name] = evaluate.metrics(y, evaluate.predict(first))["balanced_accuracy"]
    assert scores["majority"] == pytest.approx(1 / 3)
    assert scores["logistic"] > 0.7 and scores["forest"] > 0.6


def test_metrics_match_hand_counts():
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2])
    p = np.array([0, 0, 1, 2, 1, 0, 2, 2, 0])
    found = evaluate.metrics(y, p)
    assert found["confusion"] == [[2, 1, 1], [1, 1, 0], [1, 0, 2]]
    assert found["recall"] == pytest.approx([0.5, 0.5, 2 / 3])
    assert found["precision"] == pytest.approx([0.5, 0.5, 2 / 3])
    assert found["f1"] == pytest.approx([0.5, 0.5, 2 / 3])
    assert found["balanced_accuracy"] == pytest.approx((0.5 + 0.5 + 2 / 3) / 3)
    assert found["macro_f1"] == pytest.approx((0.5 + 0.5 + 2 / 3) / 3)
    assert found["accuracy"] == pytest.approx(5 / 9)
    collapsed = evaluate.collapsed(y, p)
    assert collapsed["clean_vs_harsh"]["confusion"] == [[2, 2], [2, 3]]


def test_per_group_results_and_consistency():
    examples = [{"group": g} for g in "aaabbbccc"]
    y = np.array([0, 1, 2, 0, 0, 0, 1, 1, 2])
    p = np.array([0, 1, 0, 0, 0, 1, 0, 1, 2])
    per = evaluate.per_group(examples, y, p)
    assert per["a"]["accuracy"] == pytest.approx(2 / 3) and per["a"]["recall"] == {0: 1.0, 1: 1.0, 2: 0.0}
    found = evaluate.consistency(per)
    assert found["groups"] == 3 and found["accuracy_min"] == pytest.approx(2 / 3)
    assert found["by_answer"][2]["groups_with_no_example_right"] == 1


def test_clean_against_harsh_is_also_counted_inside_artists_and_recordings_with_both():
    examples = [{"group": g, "recording": r} for g, r in
                [("a", "a1"), ("a", "a1"), ("a", "a2"), ("b", "b1"), ("b", "b1"), ("c", "c1")]]
    y = np.array([0, 1, 2, 0, 0, 2])
    p = np.array([0, 2, 0, 0, 1, 2])
    found = evaluate.within(examples, y, p)
    assert found["clean_vs_harsh_within_groups"]["units"] == 1  # only group a has both
    assert found["clean_vs_harsh_within_groups"]["confusion"] == [[1, 0], [1, 1]]
    assert found["clean_vs_harsh_within_recordings"]["units"] == 1  # only recording a1
    assert found["clean_vs_harsh_within_recordings"]["confusion"] == [[1, 0], [0, 1]]
    assert found["high_vs_low_within_groups"]["units"] == 1
    assert found["high_vs_low_within_groups"]["confusion"] == [[0, 1], [0, 0]]
    assert found["groups_by_content"] == {"clean_only": 1, "harsh_only": 1, "both": 1}


def test_abstention_withholds_the_least_confident_and_says_so():
    proba = np.array([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25], [0.2, 0.7, 0.1], [0.34, 0.33, 0.33], [0.1, 0.1, 0.8]])
    y = np.array([0, 1, 1, 2, 2])
    found = evaluate.abstention(y, proba, coverages=(1.0, 0.6), target=0.6)
    assert [c["kept"] for c in found["curve"]] == [5, 3]
    assert found["curve"][1]["accuracy"] == 1.0
    assert found["rule"]["coverage"] == pytest.approx(0.6)
    outcomes = evaluate.outcome(proba, found["rule"]["threshold"])
    assert outcomes[0] == ("predicted", 0) and outcomes[3] == ("abstained", None)
    assert all(kind in ("predicted", "abstained") for kind, _ in outcomes)


def test_unsupported_answers_are_never_trained_or_mapped(labelled):
    folder = labelled["folder"]
    labels = store.Labels(folder)
    side = data.sides(labelled["dataset"])
    dev = [e for e, s in sorted(side.items()) if s == "development" and e not in labelled["leak"]]
    labels.label(dev[0], "gutturals")
    labels.label(dev[1], "sing_screams")
    found = reload(labelled)
    used = {e["excerpt"]: e["label"] for e in found["examples"]}
    assert dev[0] not in used and dev[1] not in used
    assert set(used.values()) <= set(SUPPORTED)
    assert found["excluded"]["gutturals"] >= 1 and found["excluded"]["sing_screams"] == 1


# -- the whole experiment ---------------------------------------------------------------


def test_the_experiment_runs_twice_alike_and_its_summary_names_nothing(labelled, tmp_path, monkeypatch):
    monkeypatch.setattr(models, "TREES", 40)
    monkeypatch.setattr(models, "make", lambda name, seed: models.Forest(seed, trees=40) if name == "forest" else
                        models.Majority() if name == "majority" else models.Logistic())
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    first = experiment.run(labelled["folder"], out_a, say=lambda text: None)
    second = experiment.run(labelled["folder"], out_b, use_cache=False, say=lambda text: None)
    for summary in (first, second):
        summary.pop("timing")
        for model in summary["models"].values():
            model.pop("seconds")
    assert first == second
    assert (out_a / "predictions.csv").read_text() == (out_b / "predictions.csv").read_text()
    public = (out_a / "summary.json").read_text() + (out_a / "summary.md").read_text()
    dataset = labelled["dataset"]
    for excerpt in dataset["excerpts"]:
        assert excerpt["id"] not in public and excerpt["recording"] not in public
    for artist in dataset["artists"]:
        assert artist not in public
    assert "Tune number" not in public and str(labelled["folder"].parent.parent) not in public
    assert POISON not in public and POISON not in (out_a / "report.json").read_text()
    assert set(json.loads((out_a / "summary.json").read_text())["models"]) == set(models.MODELS)
    private = (out_a / "report.json").read_text()
    assert labelled["leak"][0] in private  # identities stay in the private report only


def test_a_public_summary_carrying_a_private_value_is_refused(labelled):
    found = reload(labelled)
    found["folder"] = str(labelled["folder"])
    artist = labelled["dev"][0]
    with pytest.raises(experiment.PrivacyError):
        experiment.check_public(f"best on {artist}", found)
    with pytest.raises(experiment.PrivacyError):
        experiment.check_public(f"Tune number {next(iter(found['dataset']['recordings']))}", found)
    experiment.check_public("balanced accuracy 0.5, macro F1 0.4", found)


def test_outputs_go_to_an_ignored_folder():
    from gmv.character.research import cli

    root = Path(__file__).resolve().parents[1]
    assert cli.DELIVERY_OUT.startswith("local-evidence/")
    assert "local-evidence/" in (root / ".gitignore").read_text().splitlines()


def test_the_command_stops_on_unusable_data(tmp_path, capsys):
    from gmv.cli import main

    with pytest.raises(SystemExit) as stopped:
        main(["character", "research", "delivery", "--folder", str(tmp_path / "nothing"), "--out", str(tmp_path / "out")])
    assert "Stopped" in str(stopped.value)
    assert not (tmp_path / "out" / "summary.json").exists()
