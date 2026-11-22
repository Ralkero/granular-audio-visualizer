"""Bring everything gathered so far together and write the local tables.

Reads what the other steps left in the work folder (identities, credits, fingerprints,
model timelines, separation levels) and the review batch's human labels; writes the
enriched inventory, the model timelines as a table, the label candidates, the review queue
and a summary. Running it again rewrites the tables from whatever is present now.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path


from . import fusion, queue, read_inventory, read_jsonl, recording_id, write_csv
from .fusion import HOP_S

INVENTORY_KEEP = (
    "artist",
    "title",
    "album",
    "album_artist",
    "year",
    "genre",
    "duration_s",
    "file_path",
    "relative_path",
    "top_folder",
    "codec",
    "bitrate_kbps",
    "sample_rate_hz",
    "isrc",
    "musicbrainz_recording_id",
    "musicbrainz_album_id",
    "instrumental_or_vocal",
    "version_type",
    "content_category",
    "sha256",
)
PANNS_TABLE = ("guitar", "electric_guitar", "acoustic_guitar", "bass_guitar", "piano", "synthesizer", "string_section", "violin")
BLOCK_S = 10.0


def artist_group(row):
    from ..review.batch import group_of

    return group_of(row)


def timeline_rows(rid, gmv, panns, thresholds):
    """Both models as ten-second blocks: mean and maximum score, each model on its own."""
    rows = []
    per = round(BLOCK_S / HOP_S)
    if gmv is not None:
        for key in ("guitar", "distorted", "clean", "acoustic"):
            scores = gmv[key]
            name = "guitar" if key == "guitar" else f"guitar_{key}"
            cut = thresholds[name]
            for first in range(0, len(scores), per):
                piece = scores[first : first + per]
                mean = float(piece.mean())
                rows.append(
                    {
                        "recording_id": rid,
                        "start_s": first * HOP_S,
                        "end_s": min(len(scores), first + per) * HOP_S,
                        "model": str(gmv["model"]),
                        "model_version": str(gmv["model_version"]),
                        "instrument_class": name,
                        "score_mean": round(mean, 4),
                        "score_max": round(float(piece.max()), 4),
                        "decision_threshold": f"detected {cut['detected']}, candidate {cut['candidate']}",
                        "predicted_state": "detected"
                        if mean >= cut["detected"]
                        else "candidate"
                        if mean >= cut["candidate"]
                        else "absent",
                        "context_complete": first * HOP_S >= float(gmv["context_s"]),
                        "preprocessing": str(gmv["preprocessing"]),
                    }
                )
    if panns is not None:
        classes = [str(c) for c in panns["classes"]]
        for key in PANNS_TABLE:
            scores = panns["mean"][:, classes.index(key)]
            for first in range(0, len(scores), per):
                piece = scores[first : first + per]
                rows.append(
                    {
                        "recording_id": rid,
                        "start_s": first * HOP_S,
                        "end_s": min(len(scores), first + per) * HOP_S,
                        "model": str(panns["model"]),
                        "model_version": str(panns["weights_sha256"])[:12],
                        "instrument_class": key,
                        "score_mean": round(float(piece.mean()), 4),
                        "score_max": round(float(piece.max()), 4),
                        "decision_threshold": "none of its own; see fusion cut-offs",
                        "predicted_state": "",
                        "context_complete": True,
                        "preprocessing": "ffmpeg decode, channel mean, 32 kHz; frame-wise output averaged per 0.5 s",
                    }
                )
    return rows


def fuse(inventory, workdir, review_workdir=None, log=print):
    workdir = Path(workdir)
    rows = read_inventory(inventory)
    identities = {e["recording_id"]: e for e in read_jsonl(workdir / "musicbrainz" / "identity.jsonl")}
    credits = defaultdict(list)
    for credit in read_jsonl(workdir / "Metadata_Evidence.jsonl"):
        credits[credit["recording_id"]].append(credit)
    same_audio = {}
    for group in read_jsonl(workdir / "same_audio_groups.jsonl"):
        for rid in group["recordings"]:
            same_audio[rid] = group["group"]
    human = fusion.human_intervals(review_workdir) if review_workdir else {}
    by_id = {recording_id(r["sha256"]): r for r in rows}
    trained_groups = {artist_group(by_id[rid]) for rid in human if rid in by_id}
    model_dirs = sorted(p for p in (workdir / "timelines").glob("*") if p.is_dir())
    gmv_dir = model_dirs[0] if model_dirs else None

    inventory_out, candidates, timeline_table, for_queue = [], [], [], []
    counts = Counter()
    for index, row in enumerate(rows, 1):
        rid = recording_id(row["sha256"])
        identity = identities.get(rid, {})
        summary = fusion.credit_summary(credits.get(rid, []))
        gmv = fusion.load_npz(gmv_dir / f"{rid}.npz") if gmv_dir else None
        panns = fusion.load_npz(workdir / "panns" / "timelines" / f"{rid}.npz")
        separation = fusion.load_npz(workdir / "separation" / f"{rid}.npz")
        recording = {
            "id": rid,
            "credits": summary,
            "match_confidence": identity.get("confidence", "none"),
            "human": sorted(human.get(rid, [])),
            "gmv": gmv["guitar"] if gmv else None,
            "panns": None,
            "level_db": gmv["level_dbfs"] if gmv else None,
            "separation": separation["guitar_db"] if separation else None,
        }
        panns_by_class = None
        if panns is not None:
            names = [str(c) for c in panns["classes"]]
            panns_by_class = {name: panns["mean"][:, i] for i, name in enumerate(names)}
            recording["panns"] = panns_by_class["guitar"]
        found, classes = fusion.candidates_for(recording)
        view = fusion.recording_view(classes, summary, found)
        candidates += found
        if summary["scope"] == "recording_level_guitar_credit":
            candidates.append(
                {
                    "recording_id": rid,
                    "start_s": "",
                    "end_s": "",
                    "label": "present_somewhere",
                    "guitar_type": "/".join(sorted(summary["guitar_kinds"])),
                    "evidence_category": "C",
                    "confidence_class": "weak_evidence",
                    "supporting_sources": "recording-level guitar credit (MusicBrainz): "
                    + ", ".join(sorted(set(summary["recording_guitar"]))[:4]),
                    "conflicting_sources": "",
                    "temporal_precision": "none: the credit covers the recording, not a time",
                    "training_suitable": False,
                    "suggested_training_weight": 0.0,
                    "review_required": False,
                }
            )
        thresholds = json.loads(str(gmv["thresholds"])) if gmv else {}
        timeline_table += timeline_rows(rid, gmv, panns, thresholds)
        group = artist_group(row)
        trained = group in trained_groups
        for_queue.append(
            {
                "id": rid,
                "row": row,
                "classes": classes,
                "gmv": gmv,
                "panns": panns_by_class,
                "credits": summary,
                "view": view,
                "candidates": found,
                "human": recording["human"],
                "duration": float(row["duration_s"] or 0),
                "trained_artist": trained,
            }
        )
        usable = [c for c in found if c["training_suitable"] and c["evidence_category"] == "B"]
        status = identity.get("status", "not_looked_up")
        fit = identity.get("fit") or {}
        conflicts = []
        if fit and fit.get("duration_gap_s") is not None and fit["duration_gap_s"] > 5:
            conflicts.append(f"length differs from MusicBrainz by {fit['duration_gap_s']} s")
        if fit and fit.get("title", 1) < 0.85:
            conflicts.append("title differs from MusicBrainz")
        if fit and not fit.get("same_kind", True):
            conflicts.append("version markers differ (live / remix / acoustic ...)")
        if status == "ambiguous":
            action = "identity ambiguous: resolve by fingerprint lookup (needs approval) or by hand"
        elif status in ("unmatched", "not_looked_up"):
            action = "no MusicBrainz match: fingerprint lookup (needs approval) or leave unmatched"
        elif classes is None:
            action = "audio models not run yet"
        elif usable:
            action = "automatic label candidates available; covered by the audit sample"
        else:
            action = "no automatic label: evidence too thin or contradictory"
        counts[(status, summary["scope"])] += 1
        inventory_out.append(
            {
                "recording_id": rid,
                **{k: row.get(k, "") for k in INVENTORY_KEEP},
                "leakage_group": group,
                "same_audio_group": same_audio.get(rid, ""),
                "inventory_version_group": row.get("alternate_version_group", "")
                or row.get("potential_duplicate_group", ""),
                "artist_used_in_rv1_training": trained,
                "human_labelled_excerpts": len(recording["human"]),
                "mb_match_status": status,
                "mb_match_method": identity.get("method", ""),
                "mb_match_confidence": identity.get("confidence", ""),
                "mb_recording_id": identity.get("recording_mbid", ""),
                "mb_release_id": identity.get("release_mbid", ""),
                "mb_release_group_id": identity.get("release_group_mbid", ""),
                "mb_artist_ids": ";".join(identity.get("artist_mbids", [])),
                "mb_title": identity.get("mb_title", ""),
                "mb_artist": identity.get("mb_artist", ""),
                "mb_disambiguation": identity.get("mb_disambiguation", ""),
                "mb_length_s": identity.get("mb_length_s") or "",
                "mb_isrcs": ";".join(identity.get("isrcs", [])),
                "mb_alternative_matches": len(identity.get("alternatives", [])),
                "mb_tags": ";".join(identity.get("tags", [])[:12]),
                "mb_source_url": f"https://musicbrainz.org/recording/{identity['recording_mbid']}"
                if identity.get("recording_mbid")
                else "",
                "acoustid": "not looked up (needs approval)",
                "discogs_release_id": "not looked up",
                "credit_scope": summary["scope"],
                "recording_level_guitarists": "; ".join(sorted(set(summary["recording_guitar"]))),
                "recording_level_guitar_kinds": "; ".join(sorted(summary["guitar_kinds"])),
                "release_level_guitarists": "; ".join(sorted(set(summary["release_guitar"]))),
                "artist_level_guitarists": "; ".join(sorted(set(summary["artist_guitar"]))),
                "recording_level_instruments": "; ".join(sorted(summary["recording_instruments"])),
                "release_level_instruments": "; ".join(sorted(summary["release_instruments"])),
                "performers_without_instrument": summary["unspecified_performers"],
                "metadata_conflicts": "; ".join(conflicts),
                "audio_evidence": "; ".join(
                    name
                    for name, present in (
                        ("gmv recogniser", gmv is not None),
                        ("PANNs", panns is not None),
                        ("separation", separation is not None),
                        ("human excerpts", bool(recording["human"])),
                    )
                    if present
                ),
                **view,
                "automatic_candidate_intervals": len(usable),
                "recommended_next_action": action,
            }
        )
        if index % 200 == 0:
            log(f"{index}/{len(rows)} recordings fused", flush=True)
    queue_rows = queue.build(for_queue)
    write_csv(workdir / "Enriched_Music_Inventory.csv", inventory_out)
    write_csv(workdir / "Audio_Evidence_Timelines.csv", timeline_table)
    write_csv(workdir / "Automated_Label_Candidates.csv", candidates)
    write_csv(workdir / "Evidence_Review_Queue.csv", queue_rows)
    (workdir / "review_queue_selection.json").write_text(
        json.dumps(queue.selection(queue_rows), indent=1), encoding="utf-8"
    )
    automatic = defaultdict(float)
    for c in candidates:
        if c["evidence_category"] == "B" and c["training_suitable"]:
            automatic[c["recording_id"]] += c["end_s"] - c["start_s"]
    total_s = sum(float(r["duration_s"] or 0) for r in rows)
    views = Counter(r["preliminary_guitar_classification"] for r in inventory_out)
    summary = {
        "recordings": len(rows),
        "hours": round(total_s / 3600, 1),
        "identity": dict(Counter(r["mb_match_status"] for r in inventory_out)),
        "identity_by_method": dict(
            Counter(f"{r['mb_match_method'] or '-'} / {r['mb_match_confidence'] or '-'}" for r in inventory_out)
        ),
        "credit_scope": dict(Counter(r["credit_scope"] for r in inventory_out)),
        "analysed_by_gmv": sum(1 for r in for_queue if r["gmv"] is not None),
        "analysed_by_panns": sum(1 for r in for_queue if r["panns"] is not None),
        "analysed_by_separation": len(list((workdir / "separation").glob("*.npz")))
        if (workdir / "separation").is_dir()
        else 0,
        "preliminary_classification": dict(views),
        "candidates": fusion.summarise(candidates),
        "recordings_with_automatic_candidates": len(automatic),
        "share_of_collection_time_with_automatic_candidates": round(sum(automatic.values()) / total_s, 3),
        "review_queue": dict(Counter(r["queue"] for r in queue_rows)),
        "review_queue_excerpts": len(queue_rows),
        "review_queue_minutes": round(sum(r["estimated_review_s"] for r in queue_rows) / 60),
        "trained_artist_groups": len(trained_groups),
        "recordings_of_trained_artists": sum(1 for r in inventory_out if r["artist_used_in_rv1_training"]),
        "same_audio_groups": len(set(same_audio.values())),
        "cut_offs": fusion.CUTS,
    }
    fusion.dump(workdir / "fusion_summary.json", summary)
    return summary


def hop_level_check(workdir, review_workdir):
    """How often the automatic classes are right where a person has answered. This is the
    measurement behind the classes; it uses the models on stretches people labelled, with
    the human label left out of the fusion for that stretch."""
    human = fusion.human_intervals(review_workdir)
    workdir = Path(workdir)
    gmv_dir = sorted(p for p in (workdir / "timelines").glob("*") if p.is_dir())[0]
    tally = Counter()
    for rid, spans in human.items():
        gmv = fusion.load_npz(gmv_dir / f"{rid}.npz")
        panns = fusion.load_npz(workdir / "panns" / "timelines" / f"{rid}.npz")
        if gmv is None or panns is None:
            continue
        names = [str(c) for c in panns["classes"]]
        classes = fusion.hop_classes(gmv["guitar"], panns["mean"][:, names.index("guitar")], gmv["level_dbfs"])
        for start, end, presence, _kind in spans:
            if presence not in ("present_throughout", "absent"):
                continue
            piece = classes[round(start / HOP_S) : round(end / HOP_S)]
            truth = "guitar" if presence == "present_throughout" else "no_guitar"
            for value, name in ((2, "both_present"), (-2, "both_absent"), (9, "conflict"), (0, "undecided"), (-9, "silence")):
                tally[(truth, name)] += int((piece == value).sum())
    out = {}
    for name in ("both_present", "both_absent", "conflict", "undecided", "silence"):
        yes, no = tally[("guitar", name)], tally[("no_guitar", name)]
        out[name] = {
            "hops_in_guitar_excerpts": yes,
            "hops_in_guitar_free_excerpts": no,
            "share_guitar": round(yes / (yes + no), 3) if yes + no else None,
        }
    total_yes = sum(v for (t, _), v in tally.items() if t == "guitar")
    total_no = sum(v for (t, _), v in tally.items() if t == "no_guitar")
    out["coverage"] = {
        "guitar_time_called_present": round(tally[("guitar", "both_present")] / max(1, total_yes), 3),
        "guitar_free_time_called_absent": round(tally[("no_guitar", "both_absent")] / max(1, total_no), 3),
    }
    return out
