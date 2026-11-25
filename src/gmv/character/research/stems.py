"""Checks of the experimental measurements against things that are known.

A measurement made on a full mix claims to describe one part of it. Where a dataset has
that part on its own, the claim can be tested:

* **MUSDB18-HQ** has the vocals, drums, bass and the rest of each song as separate,
  time-aligned stems. The vocal stem says when a voice really sounds, what its pitch is and
  how much of it is pitched at all; the mix without drums says what the harmony is without
  percussion in the way; the drum stem says what the rhythm is without everything else.
* **MDB Drums** has annotated beats.

The stems are references for what is in the signal. They are not labels of how anything
is perceived: "share of the vocal stem that has a pitch" is a measurable property of the
voice, used here as a stand-in for sung against unpitched delivery, and that is all.

Datasets are read where they are, from paths given explicitly. Nothing is copied.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from . import gmvpass, lead, rhythm, tonal
from .frontend import Analysis
from .profile import MEDIANS, blocks, summarise

EXCERPT = (30.0, 120.0)  # seconds of each song that are analysed
ACTIVE_DB = -45.0  # a stem frame this far or less under the stem's loudest counts as sounding
STEMS = ("vocals", "drums", "bass", "other")
# The four stems of a MUSDB18-HQ song add up to its mixture: measured 67 to 71 dB under the
# mixture's level on the songs checked, which is rounding. A pair that does not come within
# this is not the mixture and its own stems, and is not compared.
SUM_WITHIN_DB = -40.0


def read(path, start, end):
    with sf.SoundFile(str(path)) as file:
        rate = file.samplerate
        file.seek(min(int(start * rate), max(0, file.frames - 1)))
        return file.read(int((end - start) * rate), dtype="float32", always_2d=True), rate


def auc(scores, truth):
    """Area under the ROC curve: the chance that a true frame scores above a false one."""
    scores, truth = np.asarray(scores, dtype=float), np.asarray(truth, dtype=bool)
    if truth.all() or not truth.any():
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # Ties share their mean rank.
    values, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    ranks = (sums / counts)[inverse]
    positive = truth.sum()
    return float((ranks[truth].sum() - positive * (positive + 1) / 2) / (positive * (~truth).sum()))


def check_files(folder):
    """None when the mixture and its four stems can belong together as files (all present,
    one sample rate, one length, one channel count); otherwise the reason they cannot."""
    found = {}
    for name in ("mixture", *STEMS):
        path = Path(folder) / f"{name}.wav"
        if not path.is_file():
            return f"no {name}.wav"
        info = sf.info(str(path))
        found[name] = (info.samplerate, info.frames, info.channels)
    if len(set(found.values())) != 1:
        return "mixture and stems differ in rate, length or channels: " + ", ".join(
            f"{name} {rate} Hz {frames} frames {channels} ch"
            for name, (rate, frames, channels) in found.items()
        )
    return None


def stem_sum_db(mix, stems):
    """How far the sum of ``stems`` is from ``mix``, in dB under the level of ``mix`` (more
    negative is closer); None for a silent mix or arrays of different shapes."""
    if any(stem.shape != mix.shape for stem in stems):
        return None
    level = float(np.mean(np.square(mix, dtype=np.float64)))
    if level <= 0:
        return None
    error = float(np.mean(np.square(mix - sum(stems), dtype=np.float64)))
    return 10.0 * float(np.log10(max(error, 1e-30) / level))


def family(module, analysis, length):
    cache = {}
    found = [module.measure(analysis, *span, cache=cache) for span in blocks(analysis.duration_s, length)]
    return {
        name: summarise([block[name] for block in found], name in MEDIANS)["value"]
        for name in module.DEFINITIONS
    }


def musdb_track(job):
    """One song: the mix against its own stems. Returns plain numbers."""
    folder, activity_model = job
    folder = Path(folder)
    try:
        mismatch = check_files(folder)
        if mismatch:
            return {"track": folder.name, "skipped": mismatch}
        mix, rate = read(folder / "mixture.wav", *EXCERPT)
        if len(mix) < 20 * rate:
            return {"track": folder.name, "skipped": "shorter than the excerpt"}
        stems = {name: read(folder / f"{name}.wav", *EXCERPT)[0] for name in STEMS}
    except Exception as exc:
        return {"track": folder.name, "skipped": f"{type(exc).__name__}: {exc}"}
    # Every comparison below is between versions of this one excerpt. Check that the stems
    # really are this mixture's: same excerpt, and they add up to it.
    residual = stem_sum_db(mix, [stems[name] for name in STEMS])
    if residual is None or residual > SUM_WITHIN_DB:
        return {
            "track": folder.name,
            "skipped": "the stems do not add up to the mixture"
            + ("" if residual is None else f" ({residual:.1f} dB)"),
        }
    vocals, drums = stems["vocals"], stems["drums"]
    out = {
        "track": folder.name,
        "split": folder.parent.name,
        "pairing": {"sample_rate": int(rate), "excerpt_frames": int(len(mix)), "stem_sum_db": round(residual, 1)},
    }
    # -- the voice ---------------------------------------------------------------------
    _, mix_rows, mix_frames = gmvpass.run(mix, rate, activity_model)
    _, stem_rows, stem_frames = gmvpass.run(vocals, rate)
    in_mix, in_stem = lead.Rows(mix_rows, mix_frames), lead.Rows(stem_rows, stem_frames)
    count = min(len(in_mix.time), len(in_stem.time))
    stem_level = np.array([frame.loudness_dbfs for frame in stem_frames[:count]])
    sounding = stem_level >= max(stem_level.max() + ACTIVE_DB, -60.0)
    music = in_mix.music[:count]
    truth = sounding[music]
    out["vocal_time_share"] = float(truth.mean())
    out["presence_auc_centre_level"] = auc(in_mix.level[:count][music], truth)
    pitched_mix = ~np.isnan(in_mix.pitch[:count])
    pitched_stem = ~np.isnan(in_stem.pitch[:count])
    out["presence_auc_pitch_clarity"] = auc(in_mix.clarity[:count][music], truth)
    if in_mix.model:
        flagged = in_mix.vocal[:count][music]
        out["model_recall"] = float(flagged[truth].mean()) if truth.any() else None
        out["model_false_alarm"] = float(flagged[~truth].mean()) if (~truth).any() else None
    # Where the mix has a pitch: is it while the voice sounds, and is it the voice's pitch?
    if pitched_mix[music].sum() >= 50:
        out["mix_pitch_while_voice"] = float(sounding[music][pitched_mix[music]].mean())
    both = pitched_mix & pitched_stem & music
    if both.sum() >= 50:
        apart = np.abs(in_mix.pitch[:count][both] - in_stem.pitch[:count][both])
        out["pitch_within_semitone"] = float((apart <= 1.0).mean())
        out["pitch_octave_error"] = float((np.abs(apart - 12.0) <= 1.0).mean())
    # Delivery: how much of the voice has a pitch, from the stem (reference) and the mix.
    if sounding.sum() * in_stem.hop >= 10.0:
        out["stem_pitched_share"] = float(pitched_stem[sounding].mean())
        out["stem_pitch_clarity"] = float(in_stem.clarity[:count][sounding].mean())
        out["mix_pitched_share"] = float(pitched_mix[music].mean())
        out["mix_pitched_share_while_voice"] = float(pitched_mix[sounding & music].mean())
        out["mix_pitch_clarity_while_voice"] = float(in_mix.clarity[:count][sounding & music].mean())
        if in_mix.model and (in_mix.vocal[:count] & music).sum() * in_mix.hop >= 5.0:
            out["mix_vocal_pitched_share_model"] = float(pitched_mix[in_mix.vocal[:count] & music].mean())
    # -- harmony: the mix against itself without the drums ------------------------------
    # "Without the drums" is the same excerpt of the same mixture less its own drum stem.
    whole, without, only = Analysis(mix, rate), Analysis(mix - drums, rate), Analysis(drums, rate)
    for name, value in family(tonal, whole, 12.0).items():
        out[f"tonal.{name}.mix"] = value
    for name, value in family(tonal, without, 12.0).items():
        out[f"tonal.{name}.no_drums"] = value
    # -- rhythm: the mix against the drums alone ----------------------------------------
    for name, value in family(rhythm, whole, 12.0).items():
        out[f"rhythm.{name}.mix"] = value
    for name, value in family(rhythm, only, 12.0).items():
        out[f"rhythm.{name}.drums"] = value
    return out


def mdb_track(job):
    """One MDB Drums mix: the pulse found against the annotated beats."""
    audio_path, beats_path = job
    audio, rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    beats = np.array([float(line.split()[0]) for line in Path(beats_path).read_text().splitlines() if line.strip()])
    if len(beats) < 4:
        return {"track": Path(audio_path).stem, "skipped": "no beats"}
    analysis = Analysis(audio, rate)
    found = family(rhythm, analysis, 12.0)
    annotated = 60.0 / float(np.median(np.diff(beats)))
    out = {"track": Path(audio_path).stem, "annotated_bpm": annotated, **found}
    if found["tempo_bpm"]:
        ratio = found["tempo_bpm"] / annotated
        out["tempo_ratio"] = ratio
        out["tempo_relation"] = next(
            (name for name, target in (("same", 1.0), ("half", 0.5), ("double", 2.0), ("two thirds", 2 / 3), ("three halves", 1.5), ("third", 1 / 3), ("triple", 3.0))
             if abs(ratio / target - 1.0) <= 0.06),
            "other",
        )
    return out


def spearman(a, b):
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None and np.isfinite([x, y]).all()]
    if len(pairs) < 8:
        return None, len(pairs)
    x, y = (np.argsort(np.argsort(column)).astype(float) for column in zip(*pairs, strict=True))
    return float(np.corrcoef(x, y)[0, 1]), len(pairs)


def paired(tracks, key, first, second, scale=None):
    """Paired differences of one measurement between two versions of each recording.

    ``tracks`` are per-recording results holding ``f"{key}.{first}"`` and
    ``f"{key}.{second}"``; the difference is *second minus first*, recording by recording,
    never across recordings. A recording where either value is absent or None is counted as
    missing, one where either is not a finite number as invalid; neither is used. ``scale``
    is a standard deviation to express the differences in (the calibration recordings'
    spread of the measurement); without a usable one those figures are None.

    The rank correlation is kept as supplementary information: it says whether the two
    versions put the recordings in the same order, not whether they give the same values.
    A measurement shifted by a constant, or scaled, keeps a rank correlation of 1."""
    rows, missing, invalid = [], [], []
    for track in sorted(tracks, key=lambda t: str(t.get("track"))):
        a, b = track.get(f"{key}.{first}"), track.get(f"{key}.{second}")
        if a is None or b is None:
            missing.append(track.get("track"))
            continue
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            invalid.append(track.get("track"))
            continue
        if not (np.isfinite(a) and np.isfinite(b)):
            invalid.append(track.get("track"))
            continue
        rows.append({"track": track.get("track"), first: a, second: b, "difference": b - a})
    out = {
        "direction": f"{second} minus {first}",
        "pairs": len(rows),
        "missing": len(missing),
        "invalid": len(invalid),
        "median_abs": None,
        "p90_abs": None,
        "max_abs": None,
        "mean_signed": None,
        "positive_share": None,
        "negative_share": None,
        "across_recordings_sd": None,
        "calibration_sd": None,
        "median_abs_sd": None,
        "p90_abs_sd": None,
        "max_abs_sd": None,
        "mean_signed_sd": None,
        "rank_correlation": None,
        "recordings": rows,
        "missing_recordings": missing,
        "invalid_recordings": invalid,
    }
    if not rows:
        return out
    difference = np.array([row["difference"] for row in rows])
    size = np.abs(difference)
    out.update(
        median_abs=float(np.median(size)),
        p90_abs=float(np.percentile(size, 90)),
        max_abs=float(size.max()),
        mean_signed=float(difference.mean()),
        positive_share=float((difference > 0).mean()),
        negative_share=float((difference < 0).mean()),
        across_recordings_sd=float(np.std([row[first] for row in rows])) if len(rows) > 1 else None,
    )
    if scale is not None and np.isfinite(scale) and scale > 0:
        out.update(
            calibration_sd=float(scale),
            median_abs_sd=out["median_abs"] / scale,
            p90_abs_sd=out["p90_abs"] / scale,
            max_abs_sd=out["max_abs"] / scale,
            mean_signed_sd=out["mean_signed"] / scale,
        )
    out["rank_correlation"] = spearman([row[first] for row in rows], [row[second] for row in rows])[0]
    return out


def drum_removal(tracks, scales=None):
    """The tonal measurements of each mixture against the same mixture without its drums.

    ``harmonic_share`` is reported apart: removing the drums removes percussive energy, so
    it is expected to change and its change is not a failure of robustness. The others
    describe pitch classes, which drums should not alter."""
    scales = scales or {}

    def one(name):
        scale = (scales.get(f"tonal.{name}") or {}).get("sd")
        return paired(tracks, f"tonal.{name}", "mix", "no_drums", scale)

    out = {
        "direction": "value without drums minus value with drums, per recording",
        "scale": "calibration standard deviations" if scales else None,
        "pitch_class": {name: one(name) for name in tonal.DEFINITIONS if name != "harmonic_share"},
        "harmonic_share": one("harmonic_share"),
        "by_tonal_content": None,
    }
    # Is the change larger where there is less tonal content? The recordings are split at
    # the median harmonic share of the mixtures; one without a usable share is in neither.
    shares = {}
    for track in tracks:
        value = track.get("tonal.harmonic_share.mix")
        if isinstance(value, (int, float)) and np.isfinite(value):
            shares[track.get("track")] = float(value)
    if len(shares) >= 4:
        middle = float(np.median(list(shares.values())))
        halves = {
            "less_tonal": [t for t in tracks if t.get("track") in shares and shares[t.get("track")] < middle],
            "more_tonal": [t for t in tracks if t.get("track") in shares and shares[t.get("track")] >= middle],
        }
        keep = ("pairs", "median_abs", "p90_abs", "median_abs_sd", "p90_abs_sd")
        out["by_tonal_content"] = {
            "split_at_harmonic_share": middle,
            **{
                half: {
                    name: {key: one_of[key] for key in keep}
                    for name in out["pitch_class"]
                    for one_of in (
                        paired(rows, f"tonal.{name}", "mix", "no_drums", (scales.get(f"tonal.{name}") or {}).get("sd")),
                    )
                }
                for half, rows in halves.items()
            },
        }
    return out


def summarise_musdb(tracks, scales=None):
    """Aggregate figures over the songs that were analysed. ``scales`` (measurement ->
    {"sd": ...}, from a corpus run) lets differences be stated in calibration standard
    deviations."""
    tracks = [t for t in tracks if "skipped" not in t]
    column = lambda key, rows=tracks: [row.get(key) for row in rows]  # noqa: E731

    def middle(key, rows=tracks):
        values = [v for v in column(key, rows) if v is not None]
        return {"median": float(np.median(values)), "low_quartile": float(np.percentile(values, 25)),
                "high_quartile": float(np.percentile(values, 75)), "songs": len(values)} if values else None

    out = {"songs": len(tracks), "voice": {}, "delivery": {}, "rhythm": {}}
    residuals = [t["pairing"]["stem_sum_db"] for t in tracks if t.get("pairing")]
    out["pairing"] = {
        "checked": len(residuals),
        "worst_stem_sum_db": max(residuals) if residuals else None,
        "rule": f"same files' rate, length and channels; stems sum to the mixture within {SUM_WITHIN_DB:g} dB",
    }
    for key in ("vocal_time_share", "presence_auc_centre_level", "presence_auc_pitch_clarity",
                "mix_pitch_while_voice", "pitch_within_semitone", "pitch_octave_error"):
        out["voice"][key] = middle(key)
    held_out = [t for t in tracks if t["split"] == "test"]
    for key in ("model_recall", "model_false_alarm"):
        out["voice"][key + "_test_split"] = middle(key, held_out)
    reference = column("stem_pitched_share")
    for key in ("mix_pitched_share", "mix_pitched_share_while_voice", "mix_pitch_clarity_while_voice"):
        rho, count = spearman(reference, column(key))
        out["delivery"][key] = {"rank_correlation_with_stem": rho, "songs": count}
    rho, count = spearman(column("stem_pitched_share", held_out), column("mix_vocal_pitched_share_model", held_out))
    out["delivery"]["mix_vocal_pitched_share_model_test_split"] = {"rank_correlation_with_stem": rho, "songs": count}
    out["delivery"]["stem_pitched_share"] = middle("stem_pitched_share")
    out["drum_removal"] = drum_removal(tracks, scales)
    for name in rhythm.DEFINITIONS:
        rho, count = spearman(column(f"rhythm.{name}.mix"), column(f"rhythm.{name}.drums"))
        out["rhythm"][name] = {"rank_correlation_mix_vs_drums": rho, "songs": count}
    ratios = [t["rhythm.tempo_bpm.mix"] / t["rhythm.tempo_bpm.drums"] for t in tracks
              if t.get("rhythm.tempo_bpm.mix") and t.get("rhythm.tempo_bpm.drums")]
    if ratios:
        ratios = np.array(ratios)
        out["rhythm"]["tempo_same_within_6_percent"] = float((np.abs(ratios - 1) <= 0.06).mean())
        out["rhythm"]["tempo_half_or_double"] = float(((np.abs(ratios / 2 - 1) <= 0.06) | (np.abs(ratios * 2 - 1) <= 0.06)).mean())
    return out


def summarise_mdb(tracks):
    tracks = [t for t in tracks if "skipped" not in t]
    relations = [t.get("tempo_relation", "no pulse") for t in tracks]
    return {
        "mixes": len(tracks),
        "tempo_relation_counts": {name: relations.count(name) for name in sorted(set(relations))},
        "median_pulse_clarity": float(np.median([t["pulse_clarity"] for t in tracks if t["pulse_clarity"] is not None])),
        "median_tempo_ambiguity": float(np.median([t["tempo_ambiguity"] for t in tracks if t["tempo_ambiguity"] is not None])),
    }


def run(musdb=None, mdb=None, activity_model=None, processes=1, limit=None, progress=print, scales=None):
    """Run the checks on the datasets given. Returns per-track figures and summaries."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    import os

    from ..diagnose import THREAD_LIMITS

    out = {}
    kept = {name: os.environ.get(name) for name in THREAD_LIMITS}
    os.environ.update(dict.fromkeys(THREAD_LIMITS, "1"))  # for the processes started here
    pool = ProcessPoolExecutor(max(1, processes), mp_context=get_context("spawn"))
    list(pool.map(int, range(max(1, processes))))
    for name, value in kept.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    with pool:
        if musdb:
            folders = sorted(p for split in ("test", "train") for p in (Path(musdb) / split).iterdir() if p.is_dir())
            folders = folders[:: max(1, len(folders) // limit)][:limit] if limit else folders
            tracks = []
            for number, found in enumerate(pool.map(musdb_track, [(f, activity_model) for f in folders]), 1):
                progress(f"[MUSDB {number}/{len(folders)}] " + ("skipped: " + found["skipped"] if "skipped" in found else "done"))
                tracks.append(found)
            out["musdb"] = {"tracks": tracks, "summary": summarise_musdb(tracks, scales),
                            "excerpt_s": list(EXCERPT), "activity_model": bool(activity_model)}
        if mdb:
            audio = sorted((Path(mdb) / "audio" / "full_mix").glob("*.wav"))
            jobs = [(a, Path(mdb) / "annotations" / "beats" / (a.stem + ".beats")) for a in audio]
            jobs = [job for job in jobs if job[1].is_file()]
            tracks = []
            for number, found in enumerate(pool.map(mdb_track, jobs), 1):
                progress(f"[MDB Drums {number}/{len(jobs)}] {found.get('tempo_relation', found.get('skipped', 'no pulse'))}")
                tracks.append(found)
            out["mdb_drums"] = {"tracks": tracks, "summary": summarise_mdb(tracks)}
    return out
