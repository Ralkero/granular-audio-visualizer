"""Cache the segments of a training table with their real model context.

Runs in the application environment. Reads the segment table written by
``gmv evidence trainset`` (human answers and automatic "present" candidates, each with its
weight, leakage group and fold) and, for every usable segment, decodes the source recording
(read-only), mixes it to mono, resamples it to the model rate and cuts it from 10 s before
the segment to its end (zeros before the recording starts). That is what the live recogniser
holds while it answers about the segment. One file per recording goes to a local cache
folder, with a manifest naming the table it was made from; nothing goes into the repository.

    python experiments/recognition/cache_review.py \
        --segments research-local/evidence-v1/trainset-v1/Training_Segments.csv \
        --inventory research-local/evidence-v1/Enriched_Music_Inventory.csv \
        --output E:/datasets/gmv-cache/recognition/trainset_v1
"""

import argparse
import csv
import sys
from collections import Counter
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
import review_data as rd  # noqa: E402

from gmv.recognition import downmix  # noqa: E402
from gmv.review.audio import decode  # noqa: E402

RATE, CONTEXT_S = rd.RATE, 10.0


def rows(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def usable(table):
    """Usable rows by recording, with the checks that must hold for any training table."""
    by_recording = {}
    for row in table:
        if row["usable"] != "True":
            continue
        if row["label"] not in ("present", "absent"):
            raise SystemExit(f"{row['segment_id']}: usable but labelled {row['label']}")
        if row["source"] != "human" and (row["label"] != "present" or row["evaluation_pool"]):
            raise SystemExit(f"{row['segment_id']}: an automatic segment must be 'present' and unscored")
        if float(row["weight"]) != rd.WEIGHTS.get(row["source"]):
            raise SystemExit(f"{row['segment_id']}: weight {row['weight']} does not match {row['source']}")
        by_recording.setdefault(row["recording_id"], []).append(row)
    for rid, items in by_recording.items():
        if len({(r["leakage_group"], r["fold"]) for r in items}) != 1:
            raise SystemExit(f"{rid}: segments of one recording are in different groups or folds")
    return by_recording


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--segments", required=True, help="Training_Segments.csv")
    parser.add_argument("--inventory", required=True, help="Enriched_Music_Inventory.csv (paths, hashes)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    table = rows(args.segments)
    by_recording = usable(table)
    inventory = {r["recording_id"]: r for r in rows(args.inventory)}
    lead = round(CONTEXT_S * RATE)
    wanted = {f"{rid}.npz" for rid in by_recording}
    for stale in out.glob("r-*.npz"):
        if stale.name not in wanted:
            stale.unlink()  # a recording that left the table must not linger in the cache
    for rid, items in sorted(by_recording.items()):
        source = inventory[rid]
        audio, rate, info = decode(source["file_path"])
        if info["sha256"] != source["sha256"]:
            raise SystemExit(f"{rid}: the source file changed since the inventory was made")
        mono = downmix(audio)
        divisor = gcd(rate, RATE)
        mono = resample_poly(mono, RATE // divisor, rate // divisor).astype(np.float32)
        padded = np.concatenate([np.zeros(lead, np.float32), mono])
        arrays, meta = {}, []
        for index, row in enumerate(items):
            start, end = float(row["start_s"]), float(row["end_s"])
            first = round(start * RATE)
            size = lead + round((end - start) * RATE)
            clip = padded[first : first + size]
            arrays[f"audio_{index}"] = np.pad(clip, (0, size - len(clip))).astype(np.float16)
            meta.append(
                (
                    row["segment_id"],
                    row["label"],
                    row["guitar_type"],
                    row["source"],
                    row["weight"],
                    f"{start:.3f}",
                    f"{end - start:.3f}",
                    row["evaluation_pool"],
                    row["selection"],
                )
            )
        np.savez(
            out / f"{rid}.npz",
            meta=np.array(meta),
            group=np.array(items[0]["leakage_group"]),
            fold=np.array(int(items[0]["fold"])),
            same_audio=np.array(source.get("same_audio_group", "")),
            **arrays,
        )
        print(f"{rid}: {len(items)} segments", flush=True)
    counts = Counter(f"{r['source']} / {r['label']}" for items in by_recording.values() for r in items)
    rd.write_json(
        out / "manifest.json",
        {
            "segments_table": str(Path(args.segments).resolve()),
            "segments_table_sha256": rd.sha256_of(args.segments),
            "fold_assignment_sha256": rd.sha256_of(Path(args.segments).with_name("fold_assignment.csv")),
            "recordings": len(by_recording),
            "segments": dict(sorted(counts.items())),
            "segment_ids_sha256": rd.digest_of(
                r["segment_id"] for items in by_recording.values() for r in items
            ),
            "sample_rate": RATE,
            "context_s": CONTEXT_S,
        },
    )
    print(dict(counts))


if __name__ == "__main__":
    main()
