"""Build a review batch: select, check sources, run the model timeline, cut excerpts.

Resumable: each recording's results are cached under ``<workdir>/cache``; a rerun only
processes what is missing. Source files are opened read-only and their SHA-256 is checked
against the inventory on the same read that decodes them.
"""

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import audio as audio_tools
from . import batch as batch_tools
from . import excerpts as ex
from . import timelines as tl
from .annotations import AnnotationStore, read_rows, write_rows

TOOL_VERSION = "gmv-review 1.0"
EXCERPT_TRANSFORM = "decoded only: no resampling, gain, normalisation, filtering or dither"
MODEL_PREPROCESSING = (
    "inside the model path only: channel mean (downmix), resample to the model rate, "
    "log-mel front end of the model description; excerpt files are not affected"
)

BATCH_COLUMNS = [
    "recording_id",
    "artist",
    "title",
    "album",
    "album_artist",
    "year",
    "genre",
    "top_folder",
    "file_path",
    "sha256",
    "duration_s",
    "selection_category",
    "selection_method",
    "selection_reason",
    "selection_priority",
    "group_id",
    "exact_duplicate_group",
    "potential_duplicate_group",
    "alternate_version_group",
    "version_type",
    "instrumental_or_vocal",
    "preliminary_guitar_estimate",
    "preliminary_guitar_type",
    "preliminary_confidence",
    "inventory_priority_tier",
    "inventory_model_share_detected",
    "source_container",
    "source_subtype",
    "source_codec",
    "source_bitrate_kbps",
    "source_bit_depth",
    "source_sample_rate_hz",
    "source_channels",
    "source_frames_declared",
    "source_frames_decoded",
    "source_decoded_duration_s",
    "source_inventory_duration_s",
    "source_peak_dbfs",
    "source_samples_beyond_full_scale",
    "source_samples_at_full_scale",
    "source_clipped_runs",
    "source_rms_dbfs",
    "source_dc_offset",
    "source_low_volume_share",
    "source_longest_low_volume_s",
    "source_interior_low_volume_sections",
    "source_identical_channels",
    "source_channel_correlation",
    "source_effective_bandwidth_hz",
    "source_decoder_messages",
    "source_integrity_issues",
    "source_hash_matches_inventory",
    "model_name",
    "model_hops",
    "model_mean_guitar_score",
    "model_share_detected",
    "model_share_detected_or_candidate",
    "model_longest_detected_s",
    "model_longest_inactive_s",
    "model_state_changes_per_min",
    "possible_missed_passages",
    "possible_false_activations",
    "quiet_hops",
    "quiet_hops_model_inactive",
    "model_mean_inference_ms",
    "excerpts_core",
    "excerpts_feature",
    "excerpts_model_flagged",
    "processing_status",
    "processing_error",
]
MEASURE_KEYS = (
    "spectral_centroid_hz",
    "spectral_rolloff85_hz",
    "spectral_flatness",
    "onset_density_per_s",
    "harmonic_energy_ratio",
)
READOUT_KEYS = [
    "model_hops_in_excerpt",
    "model_guitar_majority_state",
    "model_context_complete",
] + [
    f"model_{k}_{s}"
    for k in ("guitar", "distorted", "clean", "acoustic")
    for s in ("mean", "max", "share_detected", "share_candidate_or_detected")
]
EXCERPT_COLUMNS = [
    "excerpt_id",
    "recording_id",
    "excerpt_category",
    "excerpt_position",
    "excerpt_selection_reason",
    "recording_selection_category",
    "group_id",
    "artist",
    "title",
    "album",
    "source_start_s",
    "source_end_s",
    "source_start_clock",
    "source_end_clock",
    "source_start_sample",
    "source_end_sample",
    "source_sample_rate_hz",
    "source_channels",
    "excerpt_duration_s",
    "excerpt_path",
    "excerpt_sha256",
    "excerpt_format",
    "excerpt_sample_rate_hz",
    "excerpt_channels",
    "excerpt_resampled",
    "excerpt_transformations",
    "excerpt_matches_source_samples",
    "excerpt_peak_dbfs",
    "excerpt_rms_dbfs",
    "excerpt_samples_at_full_scale",
    "excerpt_clipped_runs",
    "excerpt_low_volume_share",
    *[f"measure_{k}" for k in MEASURE_KEYS],
    *READOUT_KEYS,
    "model_thresholds",
    "model_preprocessing",
]
TIMELINE_COLUMNS = [
    "recording_id",
    "artist",
    "title",
    "output_kind",
    "model",
    "hop_start_s",
    "hop_end_s",
    "guitar_score",
    "guitar_state",
    "distorted_score",
    "distorted_state",
    "clean_score",
    "clean_state",
    "acoustic_score",
    "acoustic_state",
]

_BACKEND = {}


def _init(model_path):
    from ..recognition import load_backend

    _BACKEND["backend"] = load_backend(model_path)
    _BACKEND["description"] = json.loads(Path(model_path).read_text(encoding="utf-8"))


def clock(seconds):
    return f"{int(seconds // 60)}:{seconds % 60:06.3f}"


def process(job):
    """Everything for one recording. Runs in a worker process."""
    row, workdir = job["row"], Path(job["workdir"])
    rid = "r-" + row["sha256"][:12]
    cache = workdir / "cache" / f"{rid}.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    backend = _BACKEND["backend"]
    result = {"recording_id": rid, "excerpts": [], "error": None}
    try:
        audio, rate, info = audio_tools.decode(row["file_path"])
        quality = audio_tools.integrity(audio, rate)
        quality.update(info)
        issues = audio_tools.source_issues(quality, row)
        duration = len(audio) / rate
        timeline = tl.compute(audio, rate, backend)
        tl.save(workdir / "timelines" / f"{rid}.npz", timeline)
        levels = audio_tools.level_profile(audio, rate, 0.5)
        expect = {
            "positive_candidate": True,
            "negative_candidate": False,
            "hard_negative_candidate": False,
        }.get(row["selection_category"])
        observed = tl.observations(timeline, levels, backend.hop_s, expect)
        if job.get("queue_excerpts"):
            # A batch made from an evidence review queue: only the stretches it names.
            placements = [
                ex.Placement(
                    f"{rid}-e{index}",
                    "evidence",
                    item["queue"],
                    max(0.0, min(float(item["start_s"]), max(0.0, duration - ex.LENGTH_S))),
                    min(duration, max(0.0, min(float(item["start_s"]), max(0.0, duration - ex.LENGTH_S))) + ex.LENGTH_S),
                    item["reason"],
                )
                for index, item in enumerate(job["queue_excerpts"], 1)
            ]
        else:
            placements = ex.core_placements(rid, row["sha256"], duration)
            taken = [(p.start_s, p.end_s) for p in placements]
            feature = ex.feature_placement(rid, audio, rate, taken)
            if feature:
                placements.append(feature)
                taken.append((feature.start_s, feature.end_s))
            flagged = ex.model_placement(
                rid, timeline, backend.thresholds["guitar"], row["selection_category"], duration, taken
            )
            if flagged:
                placements.append(flagged)
        for p in placements:
            path = workdir / "excerpts" / rid / f"{p.excerpt_id}.wav"
            begin, end = ex.write_excerpt(audio, rate, p, path)
            piece = audio[begin:end]
            excerpt_quality = audio_tools.integrity(piece, rate)
            measures = audio_tools.excerpt_measures(piece, rate)
            readout = tl.window_readout(timeline, p.start_s, p.end_s, backend.hop_s)
            result["excerpts"].append(
                {
                    "excerpt_id": p.excerpt_id,
                    "recording_id": rid,
                    "excerpt_category": p.category,
                    "excerpt_position": p.position,
                    "excerpt_selection_reason": p.reason,
                    "source_start_s": round(begin / rate, 4),
                    "source_end_s": round(end / rate, 4),
                    "source_start_clock": clock(begin / rate),
                    "source_end_clock": clock(end / rate),
                    "source_start_sample": begin,
                    "source_end_sample": end,
                    "source_sample_rate_hz": rate,
                    "source_channels": audio.shape[1],
                    "excerpt_duration_s": round((end - begin) / rate, 4),
                    "excerpt_path": str(path.resolve()),
                    "excerpt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "excerpt_format": "WAV, 32-bit float",
                    "excerpt_sample_rate_hz": rate,
                    "excerpt_channels": audio.shape[1],
                    "excerpt_resampled": False,
                    "excerpt_transformations": EXCERPT_TRANSFORM,
                    "excerpt_matches_source_samples": ex.matches_source(path, audio, begin, end),
                    "excerpt_peak_dbfs": excerpt_quality["peak_dbfs"],
                    "excerpt_rms_dbfs": excerpt_quality["rms_dbfs"],
                    "excerpt_samples_at_full_scale": excerpt_quality["samples_at_full_scale"],
                    "excerpt_clipped_runs": excerpt_quality["clipped_runs"],
                    "excerpt_low_volume_share": excerpt_quality["low_volume_share"],
                    **{f"measure_{k}": v for k, v in measures.items()},
                    **readout,
                    "model_thresholds": json.dumps(backend.thresholds["guitar"]),
                    "model_preprocessing": MODEL_PREPROCESSING,
                }
            )
        counts = {c: sum(1 for p in placements if p.category == c) for c in ex.CATEGORIES}
        result.update(
            {
                "duration_s": round(duration, 4),
                "rate": rate,
                "channels": audio.shape[1],
                "sha256_read": info["sha256"],
                "source": {
                    "source_container": info["container"],
                    "source_subtype": info["subtype"],
                    "source_codec": row.get("codec"),
                    "source_bitrate_kbps": row.get("bitrate_kbps"),
                    "source_bit_depth": row.get("bit_depth") or None,
                    "source_sample_rate_hz": rate,
                    "source_channels": audio.shape[1],
                    "source_frames_declared": info["frames_declared"],
                    "source_frames_decoded": info["frames_decoded"],
                    "source_decoded_duration_s": round(duration, 4),
                    "source_inventory_duration_s": row.get("duration_s"),
                    **{
                        f"source_{k}": v
                        for k, v in quality.items()
                        if k
                        in (
                            "peak_dbfs",
                            "samples_beyond_full_scale",
                            "samples_at_full_scale",
                            "clipped_runs",
                            "rms_dbfs",
                            "dc_offset",
                            "low_volume_share",
                            "longest_low_volume_s",
                            "interior_low_volume_sections",
                            "identical_channels",
                            "channel_correlation",
                            "effective_bandwidth_hz",
                        )
                    },
                    "source_decoder_messages": " | ".join(info["decoder_messages"]) or None,
                    "source_integrity_issues": ";".join(issues) or None,
                    "source_hash_matches_inventory": info["sha256"] == row["sha256"],
                },
                "model": {
                    "model_name": f"{backend.name} {backend.version}",
                    **observed,
                    "model_mean_inference_ms": round(float(np.mean(timeline["inference_ms"])), 2)
                    if len(timeline["inference_ms"])
                    else None,
                },
                "excerpts_core": counts["core"],
                "excerpts_feature": counts["feature"],
                "excerpts_model_flagged": counts["model_flagged"],
            }
        )
    except Exception as exc:  # one bad file must not stop the batch
        result["error"] = f"{type(exc).__name__}: {exc}"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not result["error"]:
        cache.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def register(results, rows, manifest_path, tag):
    """Add the batch recordings to a corpus manifest as ``owned_local``: hashes and
    durations only, paths in the ignored local map, splits left unassigned."""
    from ..corpus import registry
    from ..corpus.schema import CorpusManifest, Recording

    manifest_path = Path(manifest_path)
    manifest = (
        registry.load_manifest(manifest_path)
        if manifest_path.is_file()
        else CorpusManifest(corpus_version="0.1.0")
    )
    paths = registry.load_paths(manifest_path)
    existing = {r.recording_id: r for r in manifest.recordings}
    for row in rows:
        result = results.get("r-" + row["sha256"][:12])
        if not result or result.get("error"):
            continue
        rid = result["recording_id"]
        if rid not in existing:
            existing[rid] = Recording(
                recording_id=rid,
                sha256=row["sha256"],
                duration_s=result["duration_s"],
                sample_rate=result["rate"],
                channels=result["channels"],
                source_kind="owned_local",
                license="commercial recording in the owner's collection; not redistributed",
                permission_basis="owner's personal copy, used locally for research only",
                group_id=row["group_id"],
                tags=(tag, row["selection_category"]),
            )
        paths["recordings"][rid] = row["file_path"]
    registry.save_manifest(
        manifest_path,
        CorpusManifest(corpus_version=manifest.corpus_version, recordings=tuple(existing.values())),
    )
    registry.save_paths(manifest_path, paths)


def build(
    inventory_path,
    selection_path,
    workdir,
    model_path,
    manifest_path,
    processes=6,
    log=print,
    backend=None,
):
    """``backend``: an already loaded backend to run in this process (tests); otherwise
    ``processes`` worker processes each load ``model_path``."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    spec = batch_tools.load_spec(selection_path)
    inventory = batch_tools.read_inventory(inventory_path)
    leakage = batch_tools.load_leakage(spec["leakage_groups"]) if spec.get("leakage_groups") else {}
    for row in inventory:
        row["same_audio_group"] = leakage.get(row["sha256"], ("", ""))[1]
    rows, problems = batch_tools.select(inventory, spec)
    for row in rows:
        row["group_id"] = leakage.get(row["sha256"], (batch_tools.group_of(row),))[0]
        row["recording_id"] = "r-" + row["sha256"][:12]
    log(f"{len(rows)} recordings selected; problems: {problems or 'none'}")
    results = {}
    queued = {
        os.path.normcase(pick["path"]): pick["excerpts"]
        for pick in spec.get("picks", [])
        if pick.get("excerpts")
    }
    jobs = [
        {
            "row": row,
            "workdir": str(workdir),
            "queue_excerpts": queued.get(os.path.normcase(row["relative_path"])),
        }
        for row in rows
    ]

    def done(n, result):
        results[result["recording_id"]] = result
        log(
            f"{n}/{len(jobs)} {result['recording_id']} "
            f"{'error: ' + result['error'] if result['error'] else str(len(result['excerpts'])) + ' excerpts'}"
        )

    if backend is not None:
        _BACKEND["backend"] = backend
        for n, job in enumerate(jobs, 1):
            done(n, process(job))
    else:
        with ProcessPoolExecutor(processes, initializer=_init, initargs=(str(model_path),)) as pool:
            futures = [pool.submit(process, job) for job in jobs]
            for n, future in enumerate(as_completed(futures), 1):
                done(n, future.result())
    register(results, rows, manifest_path, spec.get("name", "review"))
    write_tables(workdir, rows, results, spec, problems)
    return rows, results, problems


def write_tables(workdir, rows, results, spec, problems):
    workdir = Path(workdir)
    batch_rows, excerpt_rows, timeline_rows = [], [], []
    for row in rows:
        result = results.get(row["recording_id"], {})
        out = {k: row.get(k) for k in BATCH_COLUMNS if k in row}
        out.update(
            {
                "preliminary_guitar_estimate": row.get("guitar_presence_estimate"),
                "preliminary_guitar_type": row.get("likely_guitar_type"),
                "preliminary_confidence": row.get("assessment_confidence"),
                "inventory_priority_tier": row.get("priority_tier"),
                "inventory_model_share_detected": row.get("model_guitar_share_detected"),
                "duration_s": result.get("duration_s"),
                **result.get("source", {}),
                **result.get("model", {}),
                "excerpts_core": result.get("excerpts_core"),
                "excerpts_feature": result.get("excerpts_feature"),
                "excerpts_model_flagged": result.get("excerpts_model_flagged"),
                "processing_status": "error" if result.get("error") or not result else "ok",
                "processing_error": result.get("error"),
            }
        )
        batch_rows.append(out)
        for item in result.get("excerpts", []):
            excerpt_rows.append(
                {
                    **item,
                    "recording_selection_category": row["selection_category"],
                    "group_id": row["group_id"],
                    "artist": row.get("artist"),
                    "title": row.get("title"),
                    "album": row.get("album"),
                }
            )
        timeline_path = workdir / "timelines" / f"{row['recording_id']}.npz"
        if timeline_path.is_file():
            timeline = tl.load(timeline_path)
            for i in range(len(timeline["t_end"])):
                timeline_rows.append(
                    {
                        "recording_id": row["recording_id"],
                        "artist": row.get("artist"),
                        "title": row.get("title"),
                        "output_kind": "model_prediction",
                        "model": result.get("model", {}).get("model_name"),
                        "hop_start_s": round(float(timeline["t_start"][i]), 3),
                        "hop_end_s": round(float(timeline["t_end"][i]), 3),
                        **{
                            f"{k}_score": round(float(timeline[k][i]), 4)
                            for k in ("guitar", "distorted", "clean", "acoustic")
                            if k in timeline
                        },
                        **{
                            f"{k}_state": tl.STATE_NAMES[int(timeline[f"{k}_state"][i])]
                            for k in ("guitar", "distorted", "clean", "acoustic")
                            if k in timeline
                        },
                    }
                )
    batch_rows.sort(key=lambda r: (r["selection_category"], r["top_folder"], r.get("title") or ""))
    write_rows(workdir / "Validation_Batch.csv", BATCH_COLUMNS, batch_rows)
    write_rows(workdir / "Excerpt_Manifest.csv", EXCERPT_COLUMNS, excerpt_rows)
    write_rows(workdir / "Guitar_Activity_Timelines.csv", TIMELINE_COLUMNS, timeline_rows)
    AnnotationStore(workdir).write_csv(excerpt_rows, workdir / "Human_Annotations.csv")
    (workdir / "build.json").write_text(
        json.dumps(
            {
                "tool": TOOL_VERSION,
                "selection": spec.get("name"),
                "selection_problems": problems,
                "recordings": len(batch_rows),
                "excerpts": len(excerpt_rows),
                "excerpts_by_category": {
                    c: sum(1 for e in excerpt_rows if e["excerpt_category"] == c)
                    for c in ex.CATEGORIES
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_tables(workdir):
    workdir = Path(workdir)
    return read_rows(workdir / "Validation_Batch.csv"), read_rows(workdir / "Excerpt_Manifest.csv")


def verify(workdir):
    """Re-hash every source, re-read every excerpt and compare it with a fresh decode."""
    batch, excerpts = load_tables(workdir)
    report = {
        "sources_checked": 0,
        "source_hash_mismatch": [],
        "sources_missing": [],
        "excerpts_checked": 0,
        "excerpt_mismatch": [],
        "excerpt_missing": [],
    }
    by_recording = {}
    for e in excerpts:
        by_recording.setdefault(e["recording_id"], []).append(e)
    for row in batch:
        path = Path(row["file_path"])
        if not path.is_file():
            report["sources_missing"].append(row["file_path"])
            continue
        audio, rate, info = audio_tools.decode(path)
        report["sources_checked"] += 1
        if info["sha256"] != row["sha256"]:
            report["source_hash_mismatch"].append(row["file_path"])
        for e in by_recording.get(row["recording_id"], []):
            excerpt_path = Path(e["excerpt_path"])
            if not excerpt_path.is_file():
                report["excerpt_missing"].append(e["excerpt_id"])
                continue
            report["excerpts_checked"] += 1
            ok = ex.matches_source(
                excerpt_path, audio, int(e["source_start_sample"]), int(e["source_end_sample"])
            )
            if (
                not ok
                or hashlib.sha256(excerpt_path.read_bytes()).hexdigest() != e["excerpt_sha256"]
            ):
                report["excerpt_mismatch"].append(e["excerpt_id"])
    report["passed"] = not any(
        report[k]
        for k in ("source_hash_mismatch", "sources_missing", "excerpt_mismatch", "excerpt_missing")
    )
    (Path(workdir) / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
