"""Sonic Character research (phase A2): the experimental measurements and their evaluation.

Generated signals check that each measurement does the arithmetic it claims. They say
nothing about whether it matches what a listener hears.
"""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from gmv.character import diagnose
from gmv.character.research import evaluate, lead, report, rhythm, stems, tonal
from gmv.character.research.frontend import RATE as FRONT_RATE
from gmv.character.research.frontend import Analysis, result
from gmv.character.research.profile import MEDIANS, NAMES, blocks, measure_audio, summarise
from gmv.cli import main as gmv_main

RATE = 44100


# -- generated audio ----------------------------------------------------------------------


def tone(midi, seconds, level=0.2, partials=4, cents=0.0):
    time = np.arange(round(seconds * RATE)) / RATE
    f0 = 440.0 * 2.0 ** ((midi - 69 + cents / 100.0) / 12.0)
    return level * sum(np.sin(2 * np.pi * f0 * k * time) / k for k in range(1, partials + 1))


def chords(progression, each_s=2.0, cents=0.0, repeats=1):
    parts = [sum(tone(note, each_s, 0.08, cents=cents) for note in chord) for chord in progression]
    return np.tile(np.concatenate(parts), repeats)


def clicks(times, seconds, level=0.5, seed=0):
    out = np.zeros(round(seconds * RATE))
    rng = np.random.default_rng(seed)
    burst = rng.standard_normal(700) * np.exp(-np.arange(700) / 120.0)
    for time in times:
        start = round(time * RATE)
        if 0 <= start < len(out) - len(burst):
            out[start : start + len(burst)] += level * burst
    return out


def stereo(signal):
    return np.column_stack((signal, signal)).astype(np.float32)


C_MAJOR = ((60, 64, 67), (65, 69, 72), (67, 71, 74), (60, 64, 67))  # C F G C
WANDERING = ((60, 64, 67), (61, 65, 68), (66, 70, 73), (63, 67, 70), (68, 71, 75), (59, 62, 66))


def values(found):
    return {name: value for name, (value, _confidence) in found.items()}


# -- the front end ------------------------------------------------------------------------


def test_the_split_calls_a_held_tone_harmonic_and_clicks_percussive():
    held = Analysis(stereo(tone(57, 6.0)), RATE)
    struck = Analysis(stereo(clicks(np.arange(0.2, 6.0, 0.25), 6.0)), RATE)
    assert held.harmonic[held.music].sum() / held.power[held.music].sum() > 0.9
    assert struck.harmonic[struck.music].sum() / struck.power[struck.music].sum() < 0.35
    assert held.percussive.min() >= -1e-6  # the two parts are a split of the power
    assert not held.stereo  # both channels the same: no stereo image
    wide = np.column_stack((tone(57, 3.0), tone(64, 3.0))).astype(np.float32)
    assert Analysis(wide, RATE).stereo


def test_silence_is_left_out_and_the_level_matches_the_analysis_pass():
    audio = np.concatenate((tone(57, 3.0, level=0.2), np.zeros(3 * RATE)))
    analysis = Analysis(stereo(audio), RATE)
    assert analysis.music[: len(analysis.music) // 2 - 4].all()
    assert not analysis.music[len(analysis.music) // 2 + 4 :].any()
    # A full-scale sine is -3 dBFS RMS; this one is 0.2 with three weaker partials.
    expected = 10 * np.log10(0.5 * 0.2**2 * sum(1 / k**2 for k in range(1, 5)))
    assert analysis.level_db[20] == pytest.approx(expected, abs=0.3)
    assert analysis.times[0] == pytest.approx(1024 / FRONT_RATE)


def test_a_result_is_unknown_or_a_value_with_a_confidence_inside_zero_and_one():
    assert result(None, 0.9) == (None, 0.0)
    assert result(float("nan"), 0.9) == (None, 0.0)
    assert result(0.0, 0.5) == (0.0, 0.5)  # zero is a value; unknown is not zero
    assert result(2.0, 7.0) == (2.0, 1.0) and result(2.0, -1.0) == (2.0, 0.0)


# -- harmony ------------------------------------------------------------------------------


def tonal_of(signal, start=0.0, end=None):
    analysis = Analysis(stereo(signal), RATE)
    return tonal.measure(analysis, start, end or analysis.duration_s)


def test_a_progression_in_one_key_fits_a_key_and_a_wandering_one_does_not():
    in_key = values(tonal_of(chords(C_MAJOR, repeats=3)))
    wandering = values(tonal_of(chords(WANDERING, repeats=2)))
    assert in_key["key_clarity"] > 0.8 > 0.7 > wandering["key_clarity"]
    assert in_key["major_lean"] > 0.05
    assert in_key["harmonic_share"] > 0.85 and wandering["harmonic_share"] > 0.85
    assert in_key["pitch_class_focus"] > wandering["pitch_class_focus"]
    assert all(v is not None for v in in_key.values())


def test_harmonic_movement_is_low_on_a_held_chord_and_high_when_chords_change_fast():
    held = values(tonal_of(chords((C_MAJOR[0],), each_s=16.0)))
    slow = values(tonal_of(chords(C_MAJOR, each_s=4.0)))
    fast = values(tonal_of(chords(WANDERING, each_s=0.75, repeats=4)))
    assert held["harmonic_movement"] < 0.3 * slow["harmonic_movement"]
    assert fast["harmonic_movement"] > 2.0 * slow["harmonic_movement"]
    assert held["tonal_stability"] > 0.98 > fast["tonal_stability"]
    assert fast["chord_change_rate"] > slow["chord_change_rate"] >= held["chord_change_rate"]


def test_music_tuned_a_third_of_a_semitone_off_reads_the_same():
    in_tune = Analysis(stereo(chords(C_MAJOR, repeats=2)), RATE)
    sharp = Analysis(stereo(chords(C_MAJOR, repeats=2, cents=33.0)), RATE)
    assert tonal.chroma(in_tune)[1] == 0 and tonal.chroma(sharp)[1] == 1
    a = values(tonal.measure(in_tune, 0.0, in_tune.duration_s))
    b = values(tonal.measure(sharp, 0.0, sharp.duration_s))
    for name in ("key_clarity", "pitch_class_focus", "tonal_stability"):
        assert a[name] == pytest.approx(b[name], abs=0.06), name


def test_harmony_is_unknown_in_silence_and_doubted_when_nothing_is_tonal():
    quiet = tonal_of(np.zeros(8 * RATE))
    assert all(found == (None, 0.0) for found in quiet.values())
    noise = tonal_of(clicks(np.arange(0.05, 10.0, 0.11), 10.0))
    music = tonal_of(chords(C_MAJOR, repeats=2))
    assert noise["harmonic_share"][0] < 0.35
    # The share itself is measured confidently; what is read from the pitch classes is not.
    assert noise["key_clarity"][1] < 0.75 * music["key_clarity"][1]
    assert noise["major_lean"][1] <= noise["key_clarity"][1]
    assert music["chord_change_rate"][1] == pytest.approx(0.5 * music["key_clarity"][1])


# -- the lead line ------------------------------------------------------------------------


def lead_rows(pitches, level=0.5, vocal=None, hop=0.02, boundaries=()):
    """Rows as the vocal analyzer and the analysis pass would give them."""
    rows = [
        SimpleNamespace(
            timestamp=index * hop,
            level=level,
            pitch_midi=None if pitch is None else float(pitch),
            pitch_confidence=0.0 if pitch is None else 0.7,
            boundary=dict(boundaries).get(index, ""),
        )
        for index, pitch in enumerate(pitches)
    ]
    frames = [
        SimpleNamespace(
            timestamp=index * hop,
            silence=False,
            loudness_dbfs=-20.0,
            activity_method=None if vocal is None else "model",
            instruments_active=("vocals",) if vocal is not None and vocal[index] else (),
        )
        for index in range(len(pitches))
    ]
    return lead.Rows(rows, frames)


def test_the_lead_line_is_summarised_as_what_it_is():
    sung = [60] * 100 + [62] * 100 + [None] * 50 + [67] * 100 + [None] * 50  # 8 s
    found = values(lead.measure(lead_rows(sung), 0.0, 8.0))
    assert found["pitched_share"] == pytest.approx(0.75)
    assert found["held_share"] == pytest.approx(1.0)
    assert found["line_range_st"] == pytest.approx(7.0)
    assert found["centre_line_level"] == pytest.approx(0.5)
    assert found["pitch_clarity"] == pytest.approx(0.7 * 0.75)
    sliding = list(np.linspace(55, 75, 400))  # never still for 0.3 s
    moving = values(lead.measure(lead_rows(sliding), 0.0, 8.0))
    assert moving["held_share"] < 0.2 and moving["line_motion_st_s"] == pytest.approx(2.5, rel=0.05)
    assert moving["line_motion_st_s"] > 5 * found["line_motion_st_s"]


def test_no_pitch_is_unknown_range_not_a_range_of_zero():
    unpitched = lead.measure(lead_rows([None] * 400), 0.0, 8.0)
    assert unpitched["pitched_share"][0] == 0.0  # measured: nothing had a pitch
    for name in ("line_range_st", "line_motion_st_s", "held_share"):
        assert unpitched[name] == (None, 0.0), name  # ...so nothing is known of a line
    assert all(found == (None, 0.0) for found in lead.measure(lead_rows([]), 0.0, 8.0).values())


def test_what_needs_to_know_when_a_voice_sounds_is_unknown_without_the_model():
    pitches = [60] * 200 + [None] * 200
    without = lead.measure(lead_rows(pitches), 0.0, 8.0)
    for name in ("vocal_presence", "vocal_pitched_share", "phrase_s"):
        assert without[name] == (None, 0.0), name
    flags = [True] * 300 + [False] * 100  # the model hears a voice for six seconds
    with_model = values(
        lead.measure(lead_rows(pitches, vocal=flags, boundaries=((0, "start"), (100, "end"), (150, "start"), (300, "end"))), 0.0, 8.0)
    )
    assert with_model["vocal_presence"] == pytest.approx(0.75)
    assert with_model["vocal_pitched_share"] == pytest.approx(200 / 300)
    assert with_model["phrase_s"] == pytest.approx(2.5)  # median of 2 s and 3 s
    # A voice the model never reports: presence is zero, delivery is unknown.
    absent = lead.measure(lead_rows(pitches, vocal=[False] * 400), 0.0, 8.0)
    assert absent["vocal_presence"][0] == 0.0 and absent["vocal_pitched_share"] == (None, 0.0)


# -- rhythm -------------------------------------------------------------------------------


def rhythm_of(signal, floor=True):
    # A faint hum under everything, as a recording has: bare clicks leave digital silence
    # between them, which is not music and is left out.
    if floor:
        signal = signal + tone(40, len(signal) / RATE, 0.004, partials=1)
    analysis = Analysis(stereo(signal), RATE)
    return rhythm.measure(analysis, 0.0, analysis.duration_s)


def test_a_steady_pulse_is_found_and_its_level_is_stated_as_ambiguous():
    found = values(rhythm_of(clicks(np.arange(0.25, 16.0, 0.5), 16.0)))  # 120 a minute
    assert found["pulse_clarity"] > 0.7
    assert min(abs(found["tempo_bpm"] / target - 1) for target in (60, 120, 240)) < 0.03
    assert found["tempo_ambiguity"] > 0.8  # half or double fits as well: said, not hidden
    assert found["on_beat_share"] > 0.6 and found["bar_repetition"] > 0.6


def test_no_clear_pulse_leaves_the_beat_relative_values_unknown():
    rng = np.random.default_rng(4)
    scattered = rhythm_of(clicks(np.sort(rng.uniform(0.2, 15.8, 60)), 16.0))
    assert scattered["pulse_clarity"][0] < rhythm.CLEAR
    for name in ("on_beat_share", "events_per_beat", "offbeat_weight", "triplet_share", "bar_repetition", "accent_alternation"):
        assert scattered[name] == (None, 0.0), name  # "no pulse found", not "irregular"
    held = rhythm_of(tone(57, 16.0))  # nothing is struck at all
    assert all(found == (None, 0.0) for found in held.values())
    assert all(found == (None, 0.0) for found in rhythm_of(np.zeros(16 * RATE), floor=False).values())


def test_busy_against_spacious_at_the_same_pulse_and_triplets_against_straight():
    beat = np.arange(0.25, 16.0, 0.5)
    spacious = values(rhythm_of(clicks(beat, 16.0)))
    busy = values(rhythm_of(clicks(beat, 16.0) + clicks(np.arange(0.25, 16.0, 0.125), 16.0, 0.2, seed=1)))
    per_second = lambda found: found["events_per_beat"] * found["tempo_bpm"] / 60.0  # noqa: E731
    assert per_second(busy) > 2.5 * per_second(spacious)
    sixteenths = clicks(beat, 16.0) + sum(clicks(beat + s, 16.0, 0.3, seed=2) for s in (0.125, 0.25, 0.375))
    triplets = clicks(beat, 16.0) + sum(clicks(beat + s, 16.0, 0.3, seed=3) for s in (1 / 6, 2 / 6))
    assert values(rhythm_of(triplets))["triplet_share"] > 0.6 > 0.35 > values(rhythm_of(sixteenths))["triplet_share"]
    # With nothing between the beats there is no subdivision to call straight or triple.
    assert rhythm_of(clicks(beat, 16.0))["triplet_share"] == (None, 0.0)


def test_a_backbeat_alternates_and_an_even_pulse_does_not():
    beat = np.arange(0.25, 16.0, 0.5)
    backbeat = clicks(beat[0::2], 16.0, 0.2) + clicks(beat[1::2], 16.0, 0.6, seed=5)
    found = values(rhythm_of(backbeat))
    assert found["pulse_clarity"] > 0.5
    # Read at the faster level the strong and weak pulses alternate; at the slower level
    # every pulse is the strong one. Which level is reported is the stated ambiguity.
    if abs(found["tempo_bpm"] / 120 - 1) < 0.05:
        assert found["accent_alternation"] > 0.3
    else:
        assert abs(found["tempo_bpm"] / 60 - 1) < 0.05 and found["tempo_ambiguity"] > 0.5
    even = values(rhythm_of(clicks(beat, 16.0)))
    assert abs(even["accent_alternation"]) < 0.25


# -- level and determinism ----------------------------------------------------------------


def piece(seconds=14.0):
    beat = np.arange(0.25, seconds, 0.5)
    return chords(C_MAJOR, each_s=seconds / 4)[: round(seconds * RATE)] + clicks(beat, seconds, 0.3)


def test_twenty_decibels_quieter_gives_the_same_measurements():
    loud = measure_audio(stereo(piece()), RATE, baseline=False)
    quiet = measure_audio(stereo(piece()) * np.float32(0.1), RATE, baseline=False)
    for family, names in loud["families"].items():
        for name, found in names.items():
            other = quiet["families"][family][name]
            assert (found["value"] is None) == (other["value"] is None), name
            if found["value"] is not None:
                assert other["value"] == pytest.approx(found["value"], rel=0.03, abs=0.03), name
                assert other["confidence"] == pytest.approx(found["confidence"], abs=0.03), name


def test_the_same_audio_gives_the_same_result_and_the_phase_a_profile_is_untouched():
    audio = stereo(piece())
    first, again = measure_audio(audio, RATE), measure_audio(audio, RATE)
    strip = lambda found: json.dumps({k: v for k, v in found.items() if k != "phase_a"}, sort_keys=True)  # noqa: E731
    assert strip(first) == strip(again)
    # The research pass switches the vocal analyzer on. The Phase A profile made from its
    # frames is the one the Phase A diagnostics make from the plain analysis pass.
    path = None
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        path = f"{folder}/piece.wav"
        sf.write(path, audio, RATE, subtype="FLOAT")
        plain = diagnose.analyse(*diagnose.audio_messages(path))
    assert first["phase_a"]["overall"]["dimensions"] == plain["overall"]["dimensions"]
    assert first["phase_a"]["evolution"]["medium"] == plain["evolution"]["medium"]
    assert any(w.startswith("mono") for w in first["warnings"])
    assert any(w.startswith("no_activity_model") for w in first["warnings"])
    assert first["families"]["lead"]["vocal_presence"]["value"] is None


def test_blocks_overlap_by_half_and_families_use_their_own_lengths():
    assert blocks(30.0, 12.0) == [(0.0, 12.0), (6.0, 18.0), (12.0, 24.0), (18.0, 30.0)]
    assert blocks(5.0, 12.0) == [(0.0, 5.0)]
    found = measure_audio(stereo(piece(26.0)), RATE, baseline=False)
    assert len(found["evolution"]["lead"]["start_s"]) > len(found["evolution"]["tonal"]["start_s"])
    assert len(found["evolution"]["tonal"]["harmonic_share"]) == len(found["evolution"]["tonal"]["start_s"])


def test_a_summary_weighs_by_confidence_and_keeps_unknown_apart_from_zero():
    assert summarise([(None, 0.0), (None, 0.0)]) == {"value": None, "confidence": 0.0, "spread": None, "available": 0.0}
    found = summarise([(1.0, 0.9), (3.0, 0.3), (None, 0.0), (0.0, 0.0)])
    assert found["value"] == pytest.approx((0.9 + 0.9) / 1.2)
    assert found["confidence"] == pytest.approx(1.2 / 4) and found["available"] == 0.5
    # A tempo read at two metrical levels has no meaningful mean: the middle is taken.
    assert "tempo_bpm" in MEDIANS
    assert summarise([(60.0, 0.5), (120.0, 0.5), (120.0, 0.5)], middle=True)["value"] == 120.0


# -- evaluation ---------------------------------------------------------------------------


def test_the_split_is_by_group_fixed_and_can_be_forced():
    rows = [{"id": f"r{i}", "group": f"g{i % 12}", "split": "", "labels": {}} for i in range(60)]
    sides = evaluate.assign(rows)
    assert sides == evaluate.assign(list(reversed(rows)))  # order does not matter
    assert set(sides.values()) == {"calibration", "evaluation"}
    more = evaluate.assign(rows + [{"id": "new", "group": "g99", "split": "", "labels": {}}])
    assert all(more[group] == where for group, where in sides.items())  # nobody moves
    rows[0]["split"] = "evaluation" if sides["g0"] == "calibration" else "calibration"
    assert evaluate.assign(rows)["g0"] == rows[0]["split"]  # the whole group moves
    assert {k: v for k, v in evaluate.assign(rows).items() if k != "g0"} == {k: v for k, v in sides.items() if k != "g0"}


def test_between_share_and_retrieval_and_the_auc():
    apart = [[0.0, 0.1, -0.1], [5.0, 5.1, 4.9], [10.0, 9.9, 10.1]]
    mixed = [[0.0, 5.0, 10.0], [0.1, 5.1, 9.9], [-0.1, 4.9, 10.1]]
    assert evaluate.between_share(apart) > 0.99 and evaluate.between_share(mixed) < 0.05
    assert evaluate.between_share([[1.0, 2.0], [3.0]]) is None  # too few groups to say
    scores = np.array([[0.0], [0.1], [5.0], [5.1], [9.0]])
    rate, chance, counted = evaluate.retrieval(scores, ["a", "a", "b", "b", "c"])
    assert rate == 1.0 and counted == 4 and chance == pytest.approx(0.25)
    assert stems.auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert stems.auc([0.5, 0.5, 0.5, 0.5], [False, True, False, True]) == 0.5
    assert stems.auc([1, 2, 3], [True, True, True]) is None
    assert stems.spearman(list(range(10)), [v * v for v in range(10)])[0] == pytest.approx(1.0)
    assert stems.spearman([1, 2, None], [1, 2, 3]) == (None, 2)


def fake(dimensions, tonal_value, blocks_of=4):
    """A research result with given Phase A values and one tonal value; the rest unknown."""
    families = {
        family: {name: {"value": None, "confidence": 0.0, "spread": None, "available": 0.0} for name in names}
        for family, names in (("tonal", tonal.DEFINITIONS), ("lead", lead.DEFINITIONS), ("rhythm", rhythm.DEFINITIONS))
    }
    families["tonal"]["harmonic_share"] = {"value": tonal_value, "confidence": 0.9, "spread": 0.0, "available": 1.0}
    evolution = {
        family: {
            "start_s": [6.0 * k for k in range(blocks_of)],
            "end_s": [6.0 * k + 12.0 for k in range(blocks_of)],
            **{name: [None] * blocks_of for name in names},
        }
        for family, names in families.items()
    }
    evolution["tonal"]["harmonic_share"] = [tonal_value + 0.01 * k for k in range(blocks_of)]
    seconds = 90
    return {
        "families": families,
        "evolution": evolution,
        "phase_a": {
            "overall": {"dimensions": [{"value": v} for v in dimensions]},
            "evolution": {
                "time": [float(t) for t in range(seconds)],
                "intensity": [0.5 + 0.004 * t for t in range(seconds)],
                "active": [True] * seconds,
                "short": [list(dimensions)] * seconds,  # a recording that never changes
            },
        },
    }


def test_the_corpus_evaluation_reports_on_the_evaluation_groups_only():
    rng = np.random.default_rng(8)
    rows, results = [], {}
    for g in range(16):
        centre = 0.2 + 0.04 * g  # each group has its own harmonic share
        for k in range(4):
            identity = f"g{g:02d}-{k}"
            rows.append({"id": identity, "path": "x", "group": f"g{g:02d}", "split": "", "labels": {"bright": float(g >= 8)}})
            results[identity] = fake(list(rng.uniform(0, 1, 7)), centre + rng.normal(0, 0.004))
    quieter = {i: fake([d["value"] for d in r["phase_a"]["overall"]["dimensions"]], r["families"]["tonal"]["harmonic_share"]["value"] + 0.001) for i, r in list(results.items())[::3]}
    found = evaluate.evaluate(rows, results, {-10.0: quieter}, [("g00-0", "g01-0")])
    split = found["split"]
    assert not set(split["calibration_groups"]) & set(split["evaluation_groups"])
    assert split["calibration_recordings"] + split["evaluation_recordings"] == 64
    share = found["measurements"]["tonal.harmonic_share"]
    assert share["available_share"] == 1.0 and share["between_artists_share"] > 0.9
    assert share["label_auc"]["bright"] == 1.0
    assert abs(share["predicted_by_phase_a_r2"]) < 0.6  # random Phase A values predict nothing
    unknown = found["measurements"]["rhythm.pulse_clarity"]
    assert unknown["available_share"] == 0.0 and unknown["between_artists_share"] is None
    hit = found["retrieval_same_artist"]["tonal"]
    assert hit["hit_rate"] > 0.9 > hit["chance"] and found["retrieval_same_artist"]["Phase A"]["hit_rate"] < 0.5
    level = found["level"]["-10"]["measurements"]
    assert level["tonal.harmonic_share"]["largest"] < 0.05 and level["rhythm.pulse_clarity"]["median"] is None
    pair = found["pairs"][0]
    assert pair["measurements"]["rhythm.pulse_clarity"] is None
    assert set(pair["distance"]) >= {"Phase A", "Phase A + tonal", "Phase A + all"}
    # Inside these made-up recordings nothing changes, so every boundary between two of
    # them is caught without one false reset.
    assert found["transitions"]["thresholds"][0]["boundaries_caught"] == 1.0
    assert found["timescales_between_recordings_share"]["tonal.harmonic_share"]["1"] > 0.9
    assert found["blocks_with_intensity"]["rhythm.pulse_clarity"]["recordings"] == 0
    json.dumps(found, allow_nan=False)


# -- the command --------------------------------------------------------------------------


def test_the_research_command_writes_reports_and_keeps_to_ids(tmp_path, capsys):
    paths = {}
    for name, signal in (("alpha", piece(10.0)), ("beta", chords(WANDERING, each_s=1.5, repeats=1) * 1.0), ("gamma", piece(10.0) * 0.5)):
        paths[name] = tmp_path / f"private title {name}.wav"
        sf.write(paths[name], stereo(signal), RATE, subtype="FLOAT")
    out = tmp_path / "out"
    gmv_main(["character", "research", "analyze", str(paths["alpha"]), str(paths["beta"]), "--out", str(out)])
    assert "Analysed 2 of 2" in capsys.readouterr().out
    report = (out / "research.md").read_text(encoding="utf-8")
    assert "unknown" in report and "inferred" in report and "## private title alpha" in report
    assert "heuristic" in report and "not a calibrated probability" in report
    one = json.loads((out / "private_title_alpha.research.json").read_text(encoding="utf-8"))
    assert one["research_version"] and set(one["families"]) == {"tonal", "lead", "rhythm"}
    assert (out / "private_title_alpha.research.svg").read_text(encoding="utf-8").startswith("<svg")
    table = tmp_path / "corpus.csv"
    with table.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "path", "group", "split", "label:vocal"])
        writer.writerow(["a-1", paths["alpha"], "a", "calibration", "1"])
        writer.writerow(["b-1", paths["beta"], "b", "evaluation", "0"])
        writer.writerow(["c-1", paths["gamma"], "c", "evaluation", ""])
    corpus = tmp_path / "corpus"
    gmv_main(["character", "research", "corpus", "--table", str(table), "--out", str(corpus), "--gain-db", "-10", "--gain-every", "2", "--pair", "b-1", "c-1"])
    found = json.loads((corpus / "corpus.json").read_text(encoding="utf-8"))
    assert found["split"]["recording_split"] == {"a-1": "calibration", "b-1": "evaluation", "c-1": "evaluation"}
    assert found["level"]["-10"]["recordings"] == 2 and found["pairs"][0]["a"] == "b-1"
    assert "not a calibrated probability" in (corpus / "corpus.md").read_text(encoding="utf-8")
    text = (corpus / "corpus.md").read_text(encoding="utf-8") + (corpus / "corpus.json").read_text(encoding="utf-8")
    assert "private title" not in text  # reports name recordings by id only
    assert "private title" not in "".join(p.read_text(encoding="utf-8") for p in (corpus / "recordings").iterdir())
    assert len(NAMES) == len(tonal.DEFINITIONS) + len(lead.DEFINITIONS) + len(rhythm.DEFINITIONS)
    with pytest.raises(SystemExit, match="Not found"):
        gmv_main(["character", "research", "analyze", str(tmp_path / "missing.wav"), "--out", str(out)])
    with pytest.raises(SystemExit, match="Name a dataset"):
        gmv_main(["character", "research", "stems", "--out", str(out)])


def test_the_phase_a_command_still_works_beside_it(tmp_path, capsys):
    path = tmp_path / "piece.wav"
    sf.write(path, stereo(piece(8.0)), RATE, subtype="FLOAT")
    gmv_main(["character", "analyze", str(path), "--out", str(tmp_path / "a")])
    assert "Analysed 1 of 1" in capsys.readouterr().out
    assert (tmp_path / "a" / "report.md").is_file()


# -- drum removal: paired differences and pairing ---------------------------------------


def versions(pairs, key="tonal.key_clarity"):
    """Track results holding two versions of one measurement: (track, with, without)."""
    out = []
    for track, first, second in pairs:
        found = {"track": track}
        if first is not ...:
            found[f"{key}.mix"] = first
        if second is not ...:
            found[f"{key}.no_drums"] = second
        out.append(found)
    return out


def test_paired_differences_are_taken_per_recording_with_their_sign_and_scale():
    tracks = versions([("b", 1.0, 1.5), ("a", 2.0, 1.0), ("c", 0.0, 0.0), ("d", 1.0, 3.0)])
    found = stems.paired(tracks, "tonal.key_clarity", "mix", "no_drums", scale=0.5)
    assert found["direction"] == "no_drums minus mix"
    assert found["pairs"] == 4 and found["missing"] == 0 and found["invalid"] == 0
    assert [row["track"] for row in found["recordings"]] == ["a", "b", "c", "d"]
    assert [row["difference"] for row in found["recordings"]] == [-1.0, 0.5, 0.0, 2.0]
    assert found["median_abs"] == pytest.approx(0.75)
    assert found["p90_abs"] == pytest.approx(1.7)
    assert found["max_abs"] == 2.0
    assert found["mean_signed"] == pytest.approx(0.375)
    assert found["positive_share"] == 0.5 and found["negative_share"] == 0.25  # one unchanged
    assert found["calibration_sd"] == 0.5
    assert found["median_abs_sd"] == pytest.approx(1.5) and found["max_abs_sd"] == pytest.approx(4.0)
    assert found["mean_signed_sd"] == pytest.approx(0.75) and found["p90_abs_sd"] == pytest.approx(3.4)
    assert found["across_recordings_sd"] == pytest.approx(np.std([2.0, 1.0, 0.0, 1.0]))
    assert found["rank_correlation"] is None  # fewer than eight pairs: not stated


def test_missing_and_invalid_values_are_counted_and_left_out():
    tracks = versions(
        [
            ("ok-1", 0.5, 0.6),
            ("ok-2", 0.4, 0.4),
            ("none", None, 0.5),
            ("absent", 0.5, ...),
            ("nan", float("nan"), 0.5),
            ("inf", 0.5, float("inf")),
            ("text", "high", 0.5),
        ]
    )
    found = stems.paired(tracks, "tonal.key_clarity", "mix", "no_drums")
    assert found["pairs"] == 2 and found["missing"] == 2 and found["invalid"] == 3
    assert found["missing_recordings"] == ["absent", "none"]
    assert found["invalid_recordings"] == ["inf", "nan", "text"]
    assert found["max_abs"] == pytest.approx(0.1) and found["mean_signed"] == pytest.approx(0.05)
    # No usable scale: the differences in calibration standard deviations are not stated.
    for scale in (None, 0.0, float("nan"), -1.0):
        got = stems.paired(tracks, "tonal.key_clarity", "mix", "no_drums", scale=scale)
        assert got["calibration_sd"] is None and got["median_abs_sd"] is None
    nothing = stems.paired(versions([("x", None, None), ("y", float("nan"), 1.0)]), "tonal.key_clarity", "mix", "no_drums", 0.1)
    assert nothing["pairs"] == 0 and nothing["median_abs"] is None and nothing["mean_signed_sd"] is None
    assert nothing["missing"] == 1 and nothing["invalid"] == 1
    json.dumps(nothing, allow_nan=False)


def test_a_rank_correlation_of_one_does_not_mean_the_values_agree():
    """The reason the paired figures were added: a constant shift keeps the order."""
    tracks = versions([(f"t{i}", 0.1 * i, 0.1 * i + 0.3) for i in range(12)])
    found = stems.paired(tracks, "tonal.key_clarity", "mix", "no_drums", scale=0.1)
    assert found["rank_correlation"] == pytest.approx(1.0)
    assert found["median_abs"] == pytest.approx(0.3) and found["mean_signed"] == pytest.approx(0.3)
    assert found["median_abs_sd"] == pytest.approx(3.0)


def test_paired_differences_do_not_depend_on_the_order_of_the_recordings():
    rng = np.random.default_rng(12)
    tracks = versions([(f"t{i:02d}", float(rng.normal()), float(rng.normal())) for i in range(30)])
    first = stems.paired(tracks, "tonal.key_clarity", "mix", "no_drums", scale=0.2)
    shuffled = list(tracks)
    rng.shuffle(shuffled)
    again = stems.paired(shuffled, "tonal.key_clarity", "mix", "no_drums", scale=0.2)
    assert json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_drum_removal_reports_harmonic_share_apart_and_uses_each_measurements_scale():
    tracks = []
    for i in range(10):
        found = {"track": f"t{i}"}
        for name in tonal.DEFINITIONS:
            found[f"tonal.{name}.mix"] = 0.5 + 0.01 * i
            found[f"tonal.{name}.no_drums"] = 0.5 + 0.01 * i + (0.2 if name == "harmonic_share" else 0.01)
        tracks.append(found)
    scales = {f"tonal.{name}": {"sd": 0.1} for name in tonal.DEFINITIONS}
    scales["tonal.key_clarity"] = {"sd": 0.05}
    removal = stems.drum_removal(tracks, scales)
    assert "harmonic_share" not in removal["pitch_class"]
    assert set(removal["pitch_class"]) == set(tonal.DEFINITIONS) - {"harmonic_share"}
    assert removal["harmonic_share"]["mean_signed_sd"] == pytest.approx(2.0)
    assert removal["pitch_class"]["key_clarity"]["median_abs_sd"] == pytest.approx(0.2)
    assert removal["pitch_class"]["tonal_stability"]["median_abs_sd"] == pytest.approx(0.1)
    split = removal["by_tonal_content"]
    assert split["split_at_harmonic_share"] == pytest.approx(0.545)  # median of 0.50 .. 0.59
    assert split["less_tonal"]["key_clarity"]["pairs"] == 5 and split["more_tonal"]["key_clarity"]["pairs"] == 5
    assert split["less_tonal"]["key_clarity"]["median_abs_sd"] == pytest.approx(0.2)
    assert "harmonic_share" not in split["less_tonal"]
    assert stems.drum_removal(tracks[:3])["by_tonal_content"] is None  # too few to split
    assert stems.drum_removal(tracks)["scale"] is None
    assert stems.drum_removal(tracks)["pitch_class"]["key_clarity"]["median_abs_sd"] is None
    summary = stems.summarise_musdb(
        [{**t, "split": "test", "pairing": {"stem_sum_db": -70.0 - i}} for i, t in enumerate(tracks)], scales
    )
    assert summary["pairing"]["checked"] == 10 and summary["pairing"]["worst_stem_sum_db"] == -70.0
    text = report.stems({"musdb": {"summary": summary, "excerpt_s": [30, 120]}})
    assert "Harmonic share, apart" in text and "not whether they give the same values" in text
    assert "| key_clarity | 10 | 0 / 0 |" in text and "By tonal content" in text


def song_folder(folder, rate=8000, seconds=60.0, frames=None, sums=True, leave_out=()):
    """A MUSDB-like folder: four stems and their mixture, small and quick to read."""
    folder.mkdir(parents=True)
    rng = np.random.default_rng(3)
    count = round(seconds * rate)
    parts = {name: 0.1 * rng.standard_normal((count, 2)).astype(np.float32) for name in stems.STEMS}
    mix = sum(parts.values()) if sums else 0.1 * rng.standard_normal((count, 2)).astype(np.float32)
    for name, audio in {"mixture": mix, **parts}.items():
        if name in leave_out:
            continue
        length = (frames or {}).get(name, count)
        sf.write(folder / f"{name}.wav", audio[:length], rate, subtype="FLOAT")
    return folder


def test_a_mixture_is_only_compared_with_its_own_stems(tmp_path):
    good = song_folder(tmp_path / "good")
    assert stems.check_files(good) is None
    assert stems.check_files(song_folder(tmp_path / "gap", leave_out=("bass",))) == "no bass.wav"
    short = song_folder(tmp_path / "short", frames={"drums": 8000 * 59})
    assert "differ in rate, length or channels" in stems.check_files(short)
    found = stems.musdb_track((short, None))
    assert found["skipped"].startswith("mixture and stems differ")
    other = song_folder(tmp_path / "other", sums=False)
    found = stems.musdb_track((other, None))
    assert found["skipped"].startswith("the stems do not add up to the mixture")
    # The check itself: exact sums are far under the tolerance, anything else is not.
    parts = [np.full((100, 2), v, dtype=np.float32) for v in (0.1, 0.2, 0.3, 0.4)]
    assert stems.stem_sum_db(sum(parts), parts) < stems.SUM_WITHIN_DB
    assert stems.stem_sum_db(sum(parts[:3]), parts) > stems.SUM_WITHIN_DB
    assert stems.stem_sum_db(np.zeros((100, 2)), parts) is None
    assert stems.stem_sum_db(sum(parts), [parts[0][:50], *parts[1:]]) is None

