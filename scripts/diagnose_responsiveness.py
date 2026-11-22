"""How soon the instrument analyzers know what was played: controlled signals.

    python scripts/diagnose_responsiveness.py [--out responsiveness.json] [--only bass]

Synthetic signals with known onset times go through the same analysis pass the engine
runs (one analyzer at a time, no presence model, so this is the analyzer alone). Every
figure is audio time: "known" is how much audio after an onset had to exist before the
analyzer reported it. No computer speed is in these numbers; wall-clock delays are measured
by ``scripts/trace_responsiveness.py``.

Reported per signal:

    attacks   onsets found / missed / extra, where they were placed, when they were known
    level     share of onsets that visibly move the continuous level (by 0.1), and when
    dip       how far the level falls back between consecutive onsets (0 flat .. 1 fully)
    pitch     bass and vocals: when the right note was first known
    ms        instrument-analysis time per frame

Nothing is written unless ``--out`` is given; audio is generated, never read.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import numpy as np
import synth_instruments as synth
from scipy.signal import butter, sosfilt
from test_instrument_layers import guitar

from gmv.evaluation import responsiveness as r
from gmv.models import AnalysisConfig

RATE = synth.RATE
E1, A1, D2, G2 = 28, 33, 38, 43


def every(first, last, gap):
    return [round(float(t), 4) for t in np.arange(first, last, gap)]


def tom(seconds, times, level=0.4, hz=110.0, seed=5):
    """A falling tone with a short stick noise, in the centre."""
    rng = np.random.default_rng(seed)
    t = np.arange(round(0.3 * RATE)) / RATE
    out = np.zeros(round(seconds * RATE), np.float32)
    band = butter(4, [300, 3000], btype="band", fs=RATE, output="sos")
    for time in times:
        one = np.sin(2 * np.pi * (hz * t + 0.6 * hz * 0.05 * (1 - np.exp(-t / 0.05)))) * np.exp(-t / 0.12)
        one += 0.3 * sosfilt(band, rng.standard_normal(len(t))) * np.exp(-t / 0.01)
        begin = round(time * RATE)
        one = one[: len(out) - begin]
        out[begin : begin + len(one)] += level * one.astype(np.float32)
    return np.column_stack((out, out))


def sung(seconds, phrases, level=0.3, vibrato=0.15, vibrato_hz=5.5, swell=None):
    """``synth.voice`` with a chosen vibrato depth (semitones) and an optional swell
    (level factor from, to) across each phrase."""
    out = np.zeros(round(seconds * RATE), np.float32)
    for start, length, first, last in phrases:
        t = np.arange(round(length * RATE)) / RATE
        midi = first + (last - first) * t / length + vibrato * np.sin(2 * np.pi * vibrato_hz * t)
        hz = 440.0 * 2 ** ((midi - 69) / 12)
        phase = 2 * np.pi * np.cumsum(hz) / RATE
        tone = np.zeros(len(t))
        for h in range(1, 11):
            f = hz * h
            shape = np.exp(-(((f - 700) / 500) ** 2)) + 0.6 * np.exp(-(((f - 1200) / 500) ** 2)) + 0.15
            tone += shape * np.sin(h * phase)
        tone *= np.minimum(1.0, t / 0.06) * np.minimum(1.0, (length - t) / 0.08)
        if swell:
            tone *= swell[0] + (swell[1] - swell[0]) * t / length
        begin = round(start * RATE)
        tone = tone[: len(out) - begin]
        out[begin : begin + len(tone)] += level * tone.astype(np.float32)
    return np.column_stack((out, out))


def add(*parts):
    size = min(len(p) for p in parts)
    return sum(p[:size] for p in parts)


def band_under(seconds):
    """Everything but the instrument under test, for the "in a mix" signals."""
    beat = every(0.5, seconds - 0.3, 0.5)
    return add(
        synth.kick(seconds, beat[::2]),
        synth.snare(seconds, beat[1::2]),
        synth.hat(seconds, every(0.5, seconds - 0.3, 0.25)),
        synth.pad(seconds),
    )


def bass_notes(times, midi, length, level=0.4, decay=1.2, seconds=None):
    notes = [(t, m, length) for t, m in zip(times, midi if hasattr(midi, "__len__") else [midi] * len(times), strict=True)]
    return synth.bass(seconds or times[-1] + length + 0.5, notes, level=level, decay=decay)


def scenarios():
    """instrument -> [(name, audio, onsets, {extras})]."""
    found = {"bass": [], "guitar": [], "vocals": [], "drums": []}
    b = found["bass"]
    for per_second in (2, 4, 6, 8):
        times = every(1.0, 7.0, 1 / per_second)
        b.append((f"same note, {per_second} a second", bass_notes(times, E1, 0.9 / per_second), times, {"notes": [(t, E1) for t in times]}))
    times = every(1.0, 7.0, 0.25)
    walk = [(E1, A1, D2, G2)[i % 4] for i in range(len(times))]
    b.append(("changing notes, 4 a second", bass_notes(times, walk, 0.23), times, {"notes": list(zip(times, walk, strict=True))}))
    b.append(("one held tone, 4 s", synth.bass(6.0, [(1.0, E1, 4.0)], decay=30.0), [1.0], {"notes": [(1.0, E1)], "hold": (1.5, 4.8)}))
    times = every(1.0, 7.0, 1.0)
    b.append(("plucked, dying away, 1 a second", bass_notes(times, A1, 0.9, decay=0.25), times, {"notes": [(t, A1) for t in times]}))
    times = every(1.0, 9.0, 0.5)
    swell = add(*[bass_notes([t], E1, 0.45, level=0.04 + 0.5 * i / len(times), seconds=10.0) for i, t in enumerate(times)])
    b.append(("same note getting louder, 2 a second", swell, times, {"notes": [(t, E1) for t in times], "growth": True}))
    times = every(1.0, 9.0, 0.5)
    two = add(bass_notes(times[:8], E1, 0.45, level=0.06, seconds=10.0), bass_notes(times[8:], E1, 0.45, level=0.45, seconds=10.0))
    b.append(("quiet line, then a strong one", two, times, {"notes": [(t, E1) for t in times], "halves": 8}))
    times = every(1.0, 9.0, 0.5)
    b.append(("same note, 2 a second, in a band", add(bass_notes(times, E1, 0.45, seconds=10.0), band_under(10.0), guitar(10.0, every(1.0, 9.0, 0.25))), times, {"notes": [(t, E1) for t in times]}))
    times = every(1.0, 9.0, 0.25)
    b.append(("same note, 4 a second, in a band", add(bass_notes(times, E1, 0.22, seconds=10.0), band_under(10.0), guitar(10.0, every(1.0, 9.0, 0.25))), times, {"notes": [(t, E1) for t in times]}))

    times = every(1.0, 13.0, 0.5)
    swell = add(*[bass_notes([t], E1, 0.45, level=0.03 + 0.5 * i / len(times), seconds=14.0) for i, t in enumerate(times)])
    b.append(("same note getting louder over 12 s, in a band", add(swell, band_under(14.0), guitar(14.0, every(1.0, 13.0, 0.25))), times, {"notes": [(t, E1) for t in times], "growth": True}))

    g = found["guitar"]
    for name, gap, decay in (("strums, 10 a second", 0.1, 0.12), ("strums, 3 a second", 1 / 3, 0.25), ("single strokes dying away, 1 a second", 1.0, 0.35)):
        times = every(1.0, 7.0, gap)
        g.append((name, guitar(8.0, times, decay=decay), times, {}))
    g.append(("one held chord, 4 s", guitar(6.0, [1.0], decay=1.5), [1.0], {"hold": (1.5, 4.5)}))
    times = every(1.0, 7.0, 0.25)
    strong = add(guitar(8.0, times[::2], level=0.08), guitar(8.0, times[1::2], level=0.03, seed=2))
    g.append(("alternating strong and weak, 4 a second", strong, times, {"alternate": True}))
    times = every(1.0, 9.0, 0.125)
    intro = add(guitar(10.0, times[:32], level=0.02), guitar(10.0, times[32:], level=0.06, seed=2), 0 * band_under(10.0))
    late = band_under(10.0)
    late[: round(5.0 * RATE)] = 0
    g.append(("quiet alone, then with a band, 8 a second", add(intro, late), times, {"halves": 32}))
    times = every(1.0, 7.0, 0.1)
    mono = guitar(8.0, times).mean(axis=1, keepdims=True).repeat(2, axis=1)
    g.append(("strums, 10 a second, mono recording", mono, times, {}))

    v = found["vocals"]
    v.append(("one held vowel, 3 s", sung(5.0, [(1.0, 3.0, 62, 62)], vibrato=0.0), [1.0], {"notes": [(1.0, 62)], "hold": (1.4, 3.8), "ends": [4.0]}))
    times = every(1.0, 8.0, 0.8)
    v.append(("short phrases, 0.45 s on, 0.35 s off", sung(9.5, [(t, 0.45, 60, 60) for t in times]), times, {"notes": [(t, 60) for t in times], "ends": [t + 0.45 for t in times]}))
    v.append(("one vowel getting louder over 4 s", sung(6.0, [(1.0, 4.0, 62, 62)], swell=(0.1, 1.0)), [1.0], {"notes": [(1.0, 62)], "growth": (1.3, 4.8)}))
    v.append(("pitch sweep, an octave in 3 s", sung(5.0, [(1.0, 3.0, 55, 67)], vibrato=0.0), [1.0], {"sweep": (1.0, 3.0, 55, 67)}))
    v.append(("vibrato, 1 semitone, 5.5 a second", sung(5.0, [(1.0, 3.0, 62, 62)], vibrato=1.0), [1.0], {"vibrato": (1.3, 3.8, 5.5)}))
    times = every(1.0, 8.0, 1.6)
    quiet = add(sung(9.5, [(t, 1.2, 60, 60) for t in times[:2]], level=0.05), sung(9.5, [(t, 1.2, 60, 60) for t in times[2:]], level=0.4))
    v.append(("quiet singing, then strong", quiet, times, {"notes": [(t, 60) for t in times], "halves": 2}))
    times = every(1.0, 8.0, 1.6)
    v.append(("phrases in a band", add(sung(9.5, [(t, 1.2, 62, 62) for t in times]), band_under(9.5), guitar(9.5, every(1.0, 9.0, 0.25))), times, {"notes": [(t, 62) for t in times], "ends": [t + 1.2 for t in times]}))

    v.append(("one vowel getting louder over 8 s, in a band", add(sung(10.0, [(1.0, 8.0, 62, 62)], swell=(0.05, 1.0)), band_under(10.0), guitar(10.0, every(1.0, 9.0, 0.25))), [1.0], {"growth": (1.5, 8.5)}))

    d = found["drums"]
    times = every(1.0, 7.0, 0.5)
    d.append(("kick, 2 a second", synth.kick(8.0, times), times, {"region": "low"}))
    d.append(("snare, 2 a second", synth.snare(8.0, times), times, {"region": "mid"}))
    times = every(1.0, 7.0, 0.125)
    d.append(("hi-hat, 8 a second", synth.hat(8.0, times), times, {"region": "high"}))
    times = every(1.0, 7.0, 0.5)
    d.append(("tom, 2 a second", tom(8.0, times), times, {"region": "mid"}))
    d.append(("kick and snare together, 2 a second", add(synth.kick(8.0, times), synth.snare(8.0, times)), times, {"both": True}))
    times = every(1.0, 5.0, 0.125)
    d.append(("kick, 8 a second", synth.kick(6.0, times), times, {"region": "low"}))
    times = every(1.0, 5.0, 1 / 12)
    d.append(("snare, 12 a second", synth.snare(6.0, times), times, {"region": "mid"}))
    times = every(1.0, 9.0, 0.5)
    soft = add(synth.snare(10.0, times[:8], level=0.04), synth.snare(10.0, times[8:], level=0.4))
    d.append(("quiet snare hits, then strong", soft, times, {"region": "mid", "halves": 8}))
    times = every(1.0, 9.0, 0.5)
    d.append(("kick, 2 a second, with a bass line", add(synth.kick(10.0, times), bass_notes(every(1.0, 9.0, 0.5), E1, 0.45, seconds=10.0)), times, {"region": "low"}))
    times = every(1.0, 9.0, 0.5)
    soft = add(synth.snare(10.0, times[:8], level=0.04), synth.snare(10.0, times[8:], level=0.4), synth.pad(10.0), bass_notes(every(1.0, 9.0, 0.5), E1, 0.45, seconds=10.0))
    d.append(("quiet snare hits, then strong, over a pad and bass", soft, times, {"region": "mid", "halves": 8}))
    return found


def measure(instrument, name, audio, onsets, extra):
    config = AnalysisConfig(instrument_analyzers=(instrument,), spectrum_bands=0)
    rows, times = r.trace(audio, RATE, config)
    rows = rows[instrument]
    T, L = r.F["time"], r.F["level"]
    result = {
        "signal": name,
        "cadence": r.cadence(rows),
        "attacks": r.match(rows, onsets),
        "level": r.movement(rows, onsets, "level"),
        "flux": r.movement(rows, onsets, "flux", rise=0.15),
        "dip": r.dips(rows, onsets, "level"),
        "ms": r.spread(times),
    }
    if "notes" in extra:
        result["pitch"] = r.pitch_known(rows, extra["notes"])
    quick = r.fast(rows)
    if len(quick):
        result["fast"] = {
            "attacks": r.match(quick, onsets),
            "level": r.movement(quick, onsets, "level"),
            "dip": r.dips(quick, onsets, "level"),
        }
    if "region" in extra:
        result["region"] = {
            name: r.match(rows, onsets, name) for name in ("low", "mid", "high")
        }
        result["region"]["expected"] = extra["region"]
    if "both" in extra:
        result["region"] = {name: r.match(rows, onsets, name) for name in ("low", "mid", "high")}
    if "ends" in extra:
        starts = rows.copy()
        starts[:, r.F["boundary"]] = starts[:, r.F["boundary"]] == 1.0
        result["phrase_start"] = r.match(starts, onsets, "boundary", tolerance=0.3)
        ends = rows[rows[:, r.F["boundary"]] == 2.0]
        result["phrase_end_known_s"] = r.spread(
            [ends[np.argmin(np.abs(ends[:, T] - end)), r.F["known"]] - end for end in extra["ends"] if len(ends)]
        )
    if "hold" in extra:
        part = rows[(rows[:, T] >= extra["hold"][0]) & (rows[:, T] <= extra["hold"][1])]
        result["held"] = {
            "level_mean": round(float(part[:, L].mean()), 3),
            "level_wobble": round(float(part[:, L].std()), 4),
            "flux_mean": round(float(part[:, r.F["flux"]].mean()), 4),
            "attacks_while_held": int((part[:, r.F["attack"]] > 0).sum()),
        }
    if "halves" in extra or extra.get("growth") is True or extra.get("alternate"):
        peaks = []
        for first, second in zip(onsets, onsets[1:] + [onsets[-1] + (onsets[-1] - onsets[-2])], strict=True):
            part = rows[(rows[:, T] >= first - 0.02) & (rows[:, T] < second - 0.02)]
            peaks.append((float(part[:, L].max()), float(part[:, r.F["attack"]].max())) if len(part) else (0.0, 0.0))
        peaks = np.asarray(peaks)
        if "halves" in extra:
            cut = extra["halves"]
            result["quiet_then_strong"] = {
                "level_quiet": round(float(peaks[:cut, 0].mean()), 3),
                "level_strong": round(float(peaks[cut:, 0].mean()), 3),
                "attack_quiet": round(float(peaks[:cut, 1].mean()), 3),
                "attack_strong": round(float(peaks[cut:, 1].mean()), 3),
                "attacks_found_quiet": int((peaks[:cut, 1] > 0).sum()),
                "attacks_found_strong": int((peaks[cut:, 1] > 0).sum()),
            }
        elif extra.get("alternate"):
            result["alternating"] = {
                "attack_strong": round(float(peaks[::2, 1].mean()), 3),
                "attack_weak": round(float(peaks[1::2, 1].mean()), 3),
            }
        else:
            third = len(peaks) // 3
            result["growth"] = {
                "level_first_third": round(float(peaks[:third, 0].mean()), 3),
                "level_last_third": round(float(peaks[-third:, 0].mean()), 3),
                "attack_first_third": round(float(peaks[:third, 1].mean()), 3),
                "attack_last_third": round(float(peaks[-third:, 1].mean()), 3),
            }
    if isinstance(extra.get("growth"), tuple):
        a, b = extra["growth"]
        part = rows[(rows[:, T] >= a) & (rows[:, T] <= b)]
        third = len(part) // 3
        result["growth"] = {
            "level_first_third": round(float(part[:third, L].mean()), 3),
            "level_last_third": round(float(part[-third:, L].mean()), 3),
        }
    if "sweep" in extra:
        a, length, first, last = extra["sweep"]
        part = rows[(rows[:, T] >= a + 0.2) & (rows[:, T] <= a + length - 0.2)]
        truth = first + (last - first) * (part[:, T] - a) / length
        read = np.isfinite(part[:, r.F["pitch"]])
        result["sweep"] = {
            "frames_with_a_pitch": round(float(read.mean()), 3),
            "error_semitones": r.spread(np.abs(part[read, r.F["pitch"]] - truth[read])),
            # A lagging contour reads low on a rising sweep: lag = error / slope.
            "lag_s": r.spread((truth[read] - part[read, r.F["pitch"]]) / ((last - first) / length)),
        }
    if "vibrato" in extra:
        a, b, hz = extra["vibrato"]
        part = rows[(rows[:, T] >= a) & (rows[:, T] <= b)]
        read = np.isfinite(part[:, r.F["pitch"]])
        pitch = part[read, r.F["pitch"]]
        result["vibrato"] = {
            "frames_with_a_pitch": round(float(read.mean()), 3),
            "true_swing_semitones": 2.0,
            "read_swing_semitones": round(float(np.percentile(pitch, 95) - np.percentile(pitch, 5)), 2) if len(pitch) else None,
        }
    return result


def brief(value, scale=1000, unit="ms"):
    if not value or value.get("p50") is None:
        return "-"
    return f"{value['p50'] * scale:.0f} / {value['p95'] * scale:.0f} {unit}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--only", default="")
    parser.add_argument("--log", help="progress lines for a watcher window")
    args = parser.parse_args()
    log = open(args.log, "w", encoding="utf-8", buffering=1) if args.log else None
    chosen = {k: v for k, v in scenarios().items() if not args.only or k in args.only.split(",")}
    total = sum(len(v) for v in chosen.values())
    if log:
        log.write(f"task Analyzer responsiveness on {total} controlled signals\n")
    results, done = {}, 0
    for instrument, cases in chosen.items():
        results[instrument] = []
        print(f"\n{instrument.upper()}  (attacks: found/onsets +extra; times are median / p95)")
        for name, audio, onsets, extra in cases:
            done += 1
            if log:
                log.write(f"step {done}/{total} {instrument}: {name}\n")
            found = measure(instrument, name, audio, onsets, extra)
            results[instrument].append(found)
            a = found["attacks"]
            print(
                f"  {name:44s} attacks {a['found']:3d}/{a['onsets']:<3d} +{a['extra']:<2d} "
                f"placed {brief(a['placed_s']):>12s}  known {brief(a['known_s']):>13s}  "
                f"level moved {found['level']['moved']:3d}/{found['level']['onsets']:<3d} known {brief(found['level']['known_s']):>13s}  "
                f"dip {brief(found['dip'], 100, '%'):>11s}"
                + (f"  pitch {found['pitch']['read']}/{found['pitch']['notes']} known {brief(found['pitch']['known_s'])}" if "pitch" in found else "")
            )
            if "fast" in found:
                q = found["fast"]
                print(
                    f"      fast path: pulses {q['attacks']['found']}/{q['attacks']['onsets']} +{q['attacks']['extra']} "
                    f"placed {brief(q['attacks']['placed_s'])} known {brief(q['attacks']['known_s'])}  "
                    f"energy moved {q['level']['moved']}/{q['level']['onsets']} known {brief(q['level']['known_s'])}  dip {brief(q['dip'], 100, '%')}"
                )
            for key in ("region", "held", "quiet_then_strong", "alternating", "growth", "sweep", "vibrato", "phrase_start", "phrase_end_known_s"):
                if key in found:
                    value = found[key]
                    if key == "region":
                        value = {n: f"{v['found']}/{v['onsets']} known {brief(v['known_s'])}" if isinstance(v, dict) else v for n, v in value.items()}
                    elif key == "phrase_start":
                        value = f"{value['found']}/{value['onsets']} known {brief(value['known_s'])}"
                    elif key == "phrase_end_known_s":
                        value = brief(value)
                    print(f"      {key}: {value}")
        every_ms = [c["ms"]["p50"] for c in results[instrument] if c["ms"]["p50"] is not None]
        print(f"  cadence {results[instrument][0]['cadence']}  analysis ms p50 {np.median(every_ms):.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    if log:
        log.write("done\n")


if __name__ == "__main__":
    main()
