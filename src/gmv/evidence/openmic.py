"""OpenMIC-2018 as reference evidence (Humphrey, Durand, McFee, ISMIR 2018; Zenodo 1432913).

20,000 ten-second clips of Free Music Archive tracks. For each clip only some of the twenty
instruments were put to annotators; ``relevance`` is their trust-weighted agreement that the
instrument is present (0 to 1) and ``num_responses`` how many answered.

What a label means here:

``present``    relevance >= 0.8 with at least 3 responses
``absent``     relevance <= 0.2 with at least 3 responses
``uncertain``  the instrument was asked about but agreement is lower or responses are fewer
``unknown``    the instrument was never asked about for this clip. This is not "absent".

The dataset authors use 0.5 as the dividing line; the stricter bands are this project's
choice and are recorded with every label. A label covers the whole ten-second clip and says
nothing finer. The original train / test partition is kept; the artist id is kept so that
no artist needs to appear on both sides of a later split.

Licences differ by layer: annotations and metadata CC BY 4.0; each clip's audio carries the
licence of its Free Music Archive track (several Creative Commons variants, some
NonCommercial or NoDerivatives), which is copied per clip.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

PRESENT, ABSENT, MIN_RESPONSES = 0.8, 0.2, 3
FOCUS = (
    "guitar",
    "bass",
    "piano",
    "synthesizer",
    "organ",
    "violin",
    "cello",
    "banjo",
    "mandolin",
    "ukulele",
    "drums",
    "voice",
)


def verdict(relevance, responses):
    if responses >= MIN_RESPONSES and relevance >= PRESENT:
        return "present"
    if responses >= MIN_RESPONSES and relevance <= ABSENT:
        return "absent"
    return "uncertain"


def find_root(root):
    root = Path(root)
    for candidate in (root, *root.iterdir()):
        if (candidate / "openmic-2018-aggregated-labels.csv").is_file():
            return candidate
    raise FileNotFoundError("openmic-2018-aggregated-labels.csv not found under " + str(root))


def read(root):
    """{sample_key: {"split", "artist_id", "licence", "licence_url", "labels": {instrument:
    {"relevance", "responses", "verdict"}}}}. Instruments not asked about are absent from
    ``labels``: callers must treat them as unknown."""
    root = find_root(root)
    clips = {}
    with open(root / "openmic-2018-metadata.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            clips[row["sample_key"]] = {
                "split": "",
                "artist_id": row["artist_id"],
                "track_id": row["track_id"],
                "licence": row["license_title"],
                "licence_url": row["license_url"],
                "labels": {},
            }
    for split in ("train", "test"):
        with open(root / "partitions" / f"split01_{split}.csv", encoding="utf-8") as handle:
            for line in handle:
                if line.strip() in clips:
                    clips[line.strip()]["split"] = split
    with open(root / "openmic-2018-aggregated-labels.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            relevance, responses = float(row["relevance"]), int(row["num_responses"])
            clips[row["sample_key"]]["labels"][row["instrument"]] = {
                "relevance": relevance,
                "responses": responses,
                "verdict": verdict(relevance, responses),
            }
    return clips


def audio_path(root, sample_key):
    return find_root(root) / "audio" / sample_key[:3] / f"{sample_key}.ogg"


def label_rows(clips):
    """One row per clip and focus instrument that was asked about."""
    rows = []
    for key, clip in sorted(clips.items()):
        for instrument in FOCUS:
            label = clip["labels"].get(instrument)
            if label:
                rows.append(
                    {
                        "dataset": "openmic-2018",
                        "sample_key": key,
                        "split": clip["split"],
                        "artist_id": clip["artist_id"],
                        "instrument": instrument,
                        "relevance": round(label["relevance"], 4),
                        "num_responses": label["responses"],
                        "label": label["verdict"],
                        "label_rule": f">={PRESENT} present, <={ABSENT} absent, >={MIN_RESPONSES} responses",
                        "temporal_resolution": "whole 10 s clip",
                        "evidence_class": "reference_human_aggregated",
                        "audio_licence": clip["licence"],
                        "audio_licence_url": clip["licence_url"],
                    }
                )
    return rows


def summary(clips):
    counts = defaultdict(Counter)
    for clip in clips.values():
        for instrument in FOCUS:
            label = clip["labels"].get(instrument)
            counts[instrument][label["verdict"] if label else "unknown"] += 1
    licences = Counter(clip["licence"] for clip in clips.values())
    return {
        "clips": len(clips),
        "splits": dict(Counter(clip["split"] for clip in clips.values())),
        "artists": len({clip["artist_id"] for clip in clips.values()}),
        "by_instrument": {k: dict(v) for k, v in counts.items()},
        "audio_licences": dict(licences.most_common()),
    }
