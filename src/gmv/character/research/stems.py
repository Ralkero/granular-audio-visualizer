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
        mix, rate = read(folder / "mixture.wav", *EXCERPT)
        if len(mix) < 20 * rate:
            return {"track": folder.name, "skipped": "shorter than the excerpt"}
        vocals, _ = read(folder / "vocals.wav", *EXCERPT)
        drums, _ = read(folder / "drums.wav", *EXCERPT)
    except Exception as exc:
        return {"track": folder.name, "skipped": f"{type(exc).__name__}: {exc}"}
    out = {"track": folder.name, "split": folder.parent.name}
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


def summarise_musdb(tracks):
    """Aggregate figures over the songs that were analysed."""
    tracks = [t for t in tracks if "skipped" not in t]
    column = lambda key, rows=tracks: [row.get(key) for row in rows]  # noqa: E731

    def middle(key, rows=tracks):
        values = [v for v in column(key, rows) if v is not None]
        return {"median": float(np.median(values)), "low_quartile": float(np.percentile(values, 25)),
                "high_quartile": float(np.percentile(values, 75)), "songs": len(values)} if values else None

    out = {"songs": len(tracks), "voice": {}, "delivery": {}, "tonal": {}, "rhythm": {}}
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
    for name in tonal.DEFINITIONS:
        rho, count = spearman(column(f"tonal.{name}.mix"), column(f"tonal.{name}.no_drums"))
        out["tonal"][name] = {"rank_correlation_mix_vs_no_drums": rho, "songs": count}
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


def run(musdb=None, mdb=None, activity_model=None, processes=1, limit=None, progress=print):
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
            out["musdb"] = {"tracks": tracks, "summary": summarise_musdb(tracks),
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
