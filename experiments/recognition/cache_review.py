"""Cache the reviewer-labelled excerpts of a review batch with their real model context.

Runs in the application environment. For every labelled excerpt of a ``gmv review`` work
folder the source recording is decoded (read-only), mixed to mono, resampled to the model
rate and cut from 10 s before the excerpt to its end (zeros before the recording starts),
which is what the live recogniser holds while it answers about the excerpt. One file per
recording is written to a local cache folder; nothing goes into the repository.

    python experiments/recognition/cache_review.py --workdir review-local/pilot-v1 \
        --output E:/datasets/gmv-cache/recognition/review_pilot_v1
"""

import argparse
import csv
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from gmv.recognition import downmix
from gmv.review.audio import decode

RATE, CONTEXT_S = 32000, 10.0


def rows(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--leakage",
        help="leakage_groups.csv from gmv evidence fuse: split groups that also join folders "
        "holding the same audio or versions of one recording",
    )
    args = parser.parse_args()
    merged = {}
    if args.leakage:
        merged = {r["recording_id"]: r["leakage_group"] for r in rows(args.leakage)}
    work, out = Path(args.workdir), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    batch = {r["recording_id"]: r for r in rows(work / "Validation_Batch.csv")}
    excerpts = {r["excerpt_id"]: r for r in rows(work / "Excerpt_Manifest.csv")}
    labelled = {}
    for row in rows(work / "Human_Annotations.csv"):
        if row["guitar_presence"]:
            labelled.setdefault(row["recording_id"], []).append(row)
    for rid, items in sorted(labelled.items()):
        audio, rate, info = decode(batch[rid]["file_path"])
        if info["sha256"] != batch[rid]["sha256"]:
            raise SystemExit(f"{rid}: the source file changed since the batch was built")
        mono = downmix(audio)
        divisor = gcd(rate, RATE)
        mono = resample_poly(mono, RATE // divisor, rate // divisor).astype(np.float32)
        lead = round(CONTEXT_S * RATE)
        padded = np.concatenate([np.zeros(lead, np.float32), mono])
        clips, meta = [], []
        for item in items:
            excerpt = excerpts[item["excerpt_id"]]
            start, end = float(excerpt["source_start_s"]), float(excerpt["source_end_s"])
            first = round(start * RATE)
            size = lead + round((end - start) * RATE)
            clip = padded[first : first + size]
            clips.append(np.pad(clip, (0, size - len(clip))).astype(np.float16))
            meta.append(
                (
                    item["excerpt_id"],
                    item["excerpt_category"],
                    item["guitar_presence"],
                    item["guitar_type"],
                    f"{start:.3f}",
                    f"{end - start:.3f}",
                )
            )
        length = max(len(c) for c in clips)
        np.savez(
            out / f"{rid}.npz",
            audio=np.stack([np.pad(c, (0, length - len(c))) for c in clips]),
            meta=np.array(meta),
            group=np.array(merged.get(rid, excerpts[items[0]["excerpt_id"]]["group_id"])),
        )
        print(f"{rid}: {len(items)} excerpts", flush=True)


if __name__ == "__main__":
    main()
