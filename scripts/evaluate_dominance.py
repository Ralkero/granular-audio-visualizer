"""Does the dominant-instrument choice name the instrument that really is the loudest?

    python scripts/evaluate_dominance.py --moisesdb C:/Datasets/moisesdb_v0.1 --first 101 \
        --songs 40 --activity-model .model-cache/activity_v2.npz \
        --guitar-model .model-cache/guitar_mn10_v1.json [--json out.json] [--log progress.log]

**What is run.** Each song's full mix goes through the analysis pass and the guitar
recognizer, block by block as the engine does it, and every message is handed to a real
``VisualState`` in the engine's order: a feature frame, then that frame's activity rows,
and a recognition estimate after the block that completed its hop. The leader, the scores
and the presence weights are read from that state. So the attribution (detected, candidate,
tentative while the recognition model warms up, held between reports), the 100 ms refresh
of the weights, the segment handling, the stale-row rule and the choice of a leader are the
application's own code, not a copy. Two things are simulated: a recognition estimate
arrives as soon as its audio has (the application adds the model's 25 to 50 ms), and
messages arrive at their audio time (the display clock plays no part in the dominance).

**The reference.** From the song's own stems: the loudest of vocals, guitar, bass and
drums over 0.4 s. A frame is a *clear case* where that stem leads the next by 3 dB or more
and is above -45 dB. "Loudest stem" is a measurable stand-in. It is not what a listener
would necessarily call the lead: a quiet lead vocal over a loud rhythm section counts as
bass or drums here. Perceived prominence cannot be had from stems.

**Which songs.** Folders of the collection in sorted order, numbered from 1. The shipped
coefficients were fitted on numbers 1 to 60 (``scripts/fit_dominance.py``). Numbers 61 to
100 were looked at once while the hold and margin of the leader choice were set, so they
are a development set, not an independent test. Numbers from 101 on were not used for
anything before the run reported in docs/VISUAL_PRESENTATION.md. A song is skipped when
fewer than three of its stems cover the 45 s excerpt (35 s to 80 s); the first 10 s of
each excerpt are not scored (the recognition model is still warming up, as it would be).

**Baselines.** Chance (25%); always the commonest truly-loudest instrument of the test
songs; always the instrument this method picks most often; and what the application could
have done before this model: the highest own level times presence weight.

Audio is only read. Nothing is written but the optional JSON and progress log.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_responsiveness_stems import read_stem
from gmv.dsp import Analyzer, WindowAssembler
from gmv.models import AnalysisConfig, InstrumentEstimate
from gmv.recognition import Recognizer, downmix, load_backend
from gmv.sources.base import AudioBlock
from gmv.visualizer.dominance import KEYS
from gmv.visualizer.state import COLUMN, VisualState

START_S, LENGTH_S, WARM_S = 35.0, 45.0, 10.0


def stem_db(stem, rate, times, window=0.4):
    mono = stem.mean(axis=1)
    power = np.concatenate(([0.0], np.cumsum(mono.astype(np.float64) ** 2)))
    half = round(window * rate / 2)
    centre = np.clip((times * rate).astype(int), 0, len(mono) - 1)
    a, b = np.clip(centre - half, 0, len(mono)), np.clip(centre + half, 0, len(mono))
    return 10 * np.log10((power[b] - power[a]) / np.maximum(1, b - a) + 1e-12)


def run_song(mix, rate, config, backend):
    """Per analysis frame: time, leader, naive pick, presence weights. Through a VisualState."""
    analyzer = Analyzer(rate, mix.shape[1], config, "evaluation", "evaluation")
    assembler = WindowAssembler(config)
    recognizer = Recognizer(backend)
    state = VisualState()
    state.set_band_edges(config.band_edges)
    step = round(rate * config.capture_ms / 1000)
    rows = []
    for pos in range(0, len(mix), step):
        chunk = mix[pos : pos + step]
        block = AudioBlock(
            chunk.astype("<f4").tobytes(), len(chunk), rate, chunk.shape[1], pos / rate, pos / rate, 0
        )
        for window, start, mono, _reset in assembler.push(block):
            frame, events = analyzer.process(window, start, mono, block, live=False)
            state.consume(frame, arrival=frame.timestamp)
            for event in events:
                state.consume(event)
            for activity in analyzer.instrument_activity:
                state.consume(activity)
            # What the application had before: the highest own level times presence weight.
            weights = {key: state._weights.get(key, 0.0) for key in KEYS}
            naive, best = None, 0.05
            for key in KEYS:
                track = state.activity.get(key)
                value = float(track.last[COLUMN["level"]]) * weights[key] if track is not None and len(track) else 0.0
                if value > best:
                    naive, best = key, value
            rows.append((frame.timestamp, state.dominance.leader, naive, tuple(weights[k] for k in KEYS)))
        if recognizer.push(downmix(chunk), rate, pos / rate):
            estimate = recognizer.estimate()
            state.consume(
                InstrumentEstimate(
                    session_id="evaluation",
                    source="evaluation",
                    segment=0,
                    detected_monotonic=0,
                    backend=backend.name,
                    backend_version=backend.version,
                    context_s=backend.context_s,
                    timestamp=estimate.start,
                    interval_end=estimate.end,
                    scores=estimate.scores,
                    states=estimate.states,
                )
            )
    return rows


def rate_of(right, total):
    return round(right / total, 3) if total else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moisesdb", required=True)
    parser.add_argument("--first", type=int, default=101)
    parser.add_argument("--songs", type=int, default=40)
    parser.add_argument("--activity-model", required=True)
    parser.add_argument("--guitar-model", required=True)
    parser.add_argument("--json")
    parser.add_argument("--log")
    args = parser.parse_args()
    log = open(args.log, "a", encoding="utf-8", buffering=1) if args.log else None
    backend = load_backend(args.guitar_model)
    config = AnalysisConfig(instrument_analyzers=KEYS, activity_model_path=args.activity_model)
    folders = [f for f in sorted(Path(args.moisesdb).iterdir()) if f.is_dir()]
    chosen = list(enumerate(folders, 1))[args.first - 1 : args.first - 1 + args.songs]
    if log:
        log.write(f"task Dominant instrument against stems: songs {args.first} to {args.first + len(chosen) - 1}\n")
    index = {key: number for number, key in enumerate(KEYS)}
    confusion = np.zeros((4, 5), dtype=int)  # actual x predicted (the fifth: nobody)
    naive_confusion = np.zeros((4, 5), dtype=int)
    weight_sum, weight_zero, weight_part = np.zeros(4), np.zeros(4, int), np.zeros(4, int)
    frames = clear_total = changes = 0
    led = np.zeros(5, dtype=int)
    songs, skipped = [], []
    for place, (number, folder) in enumerate(chosen, 1):
        if log:
            log.write(f"step {place}/{len(chosen)} song {number}\n")
        parts = {}
        for stem in sorted(p.name for p in folder.iterdir() if p.is_dir()):
            audio, rate = read_stem(folder / stem, START_S, LENGTH_S)
            if audio is not None and len(audio) >= rate * (LENGTH_S - 5):
                parts[stem] = audio
        if len(parts) < 3:
            skipped.append(number)
            continue
        size = min(len(a) for a in parts.values())
        mix = sum(v[:size] for v in parts.values())
        rows = run_song(mix, rate, config, backend)
        times = np.array([row[0] for row in rows])
        keep = (times >= WARM_S) & (times <= size / rate - 1.0)
        true = np.stack(
            [stem_db(parts[k][:size], rate, times) if k in parts else np.full(len(times), -120.0) for k in KEYS],
            axis=1,
        )
        order = np.sort(true, axis=1)
        clear = keep & (order[:, -1] - order[:, -2] >= 3.0) & (order[:, -1] > -45.0)
        truth = true.argmax(axis=1)
        own = np.zeros((4, 5), dtype=int)
        previous = None
        for at in np.flatnonzero(keep):
            _, leader, naive, weights = rows[at]
            frames += 1
            led[index[leader] if leader else 4] += 1
            if leader is not None:
                changes += previous is not None and leader != previous
                previous = leader
            if clear[at]:
                actual = truth[at]
                own[actual, index[leader] if leader else 4] += 1
                naive_confusion[actual, index[naive] if naive else 4] += 1
                weight_sum[actual] += weights[actual]
                weight_zero[actual] += weights[actual] == 0.0
                weight_part[actual] += 0.0 < weights[actual] < 1.0
        confusion += own
        clear_total += int(own.sum())
        right, cases = int(np.trace(own[:, :4])), int(own.sum())
        songs.append(
            {
                "song": number,
                "seconds_scored": round(float(keep.sum() * 0.02), 1),
                "clear_cases": cases,
                "accuracy": rate_of(right, cases),
                "clear_by_instrument": dict(zip(KEYS, own.sum(axis=1).tolist(), strict=True)),
            }
        )
    support = confusion.sum(axis=1)
    selected = confusion[:, :4].sum(axis=0)
    right = np.diag(confusion[:, :4])
    with_leader = int(confusion[:, :4].sum())
    per = {}
    for key, number in index.items():
        precision = right[number] / selected[number] if selected[number] else None
        recall = right[number] / support[number] if support[number] else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
        per[key] = {
            "clear_cases_where_it_is_loudest": int(support[number]),
            "share_of_clear_cases": rate_of(int(support[number]), clear_total),
            "times_selected_in_clear_cases": int(selected[number]),
            "precision": None if precision is None else round(float(precision), 3),
            "recall": None if recall is None else round(float(recall), 3),
            "f1": None if f1 is None else round(float(f1), 3),
            "when_it_is_loudest": {
                "mean_presence_weight": rate_of(float(weight_sum[number]), int(support[number])),
                "share_with_weight_zero": rate_of(int(weight_zero[number]), int(support[number])),
                "share_with_partial_weight": rate_of(int(weight_part[number]), int(support[number])),
            },
        }
    scored = [song for song in songs if song["clear_cases"] >= 250]  # at least 5 s of clear cases
    most_picked = int(np.argmax(selected))
    result = {
        "songs_requested": [chosen[0][0], chosen[-1][0]] if chosen else [],
        "songs_processed": len(songs),
        "songs_skipped_for_short_stems": skipped,
        "minutes_scored": round(frames * 0.02 / 60, 1),
        "frames_scored": frames,
        "clear_cases": clear_total,
        "not_clear": frames - clear_total,
        "clear_cases_without_a_leader": int(confusion[:, 4].sum()),
        "coverage": rate_of(with_leader, clear_total),
        "accuracy_when_a_leader_was_chosen": rate_of(int(right.sum()), with_leader),
        "accuracy_over_all_clear_cases": rate_of(int(right.sum()), clear_total),
        "accuracy_per_song_mean": round(float(np.mean([s["accuracy"] for s in scored])), 3) if scored else None,
        "accuracy_per_song_median": round(float(np.median([s["accuracy"] for s in scored])), 3) if scored else None,
        "accuracy_per_song_quartiles": [round(float(v), 3) for v in np.percentile([s["accuracy"] for s in scored], [25, 75])] if scored else None,
        "songs_in_per_song_figures": len(scored),
        "baselines_over_all_clear_cases": {
            "chance": 0.25,
            "always_the_commonest_loudest": {KEYS[int(np.argmax(support))]: rate_of(int(support.max()), clear_total)},
            "always_the_most_picked": {KEYS[most_picked]: rate_of(int(support[most_picked]), clear_total)},
            "highest_own_level_times_presence": rate_of(int(np.trace(naive_confusion[:, :4])), clear_total),
        },
        "confusion_actual_by_predicted": {
            KEYS[a]: dict(zip((*KEYS, "nobody"), confusion[a].tolist(), strict=True)) for a in range(4)
        },
        "per_instrument": per,
        "share_of_all_scored_time_leading": dict(zip((*KEYS, "nobody"), np.round(led / max(1, frames), 3).tolist(), strict=True)),
        "changes_of_leader_per_minute": round(changes / max(1e-9, frames * 0.02 / 60), 1),
        "per_song": songs,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_song"}, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1), encoding="utf-8")
    if log:
        log.write(
            f"line songs {args.first}+: accuracy {result['accuracy_over_all_clear_cases']} over "
            f"{clear_total} clear cases, coverage {result['coverage']}\n"
        )


if __name__ == "__main__":
    main()
