"""Cache MoisesDB songs as a few mono 32 kHz parts, for training on freshly made mixes.

Runs in the project environment. Per song: distorted, clean, acoustic and other guitar
files, keyboards (piano and other keys), and everything else, each summed to one mono
track. The parts add up to the song. Stored as 16-bit floats outside the repository.

    python experiments/recognition/cache_stems.py --jobs JOBS.json \
        --dataset-root C:/Datasets/moisesdb_v0.1 --output E:/datasets/gmv-cache/recognition/parts
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from gmv.corpus import registry
from gmv.corpus.datasets import slug

RATE = 32000
GUITAR = {
    "distorted_electric_guitar": "distorted",
    "clean_electric_guitar": "clean",
    "acoustic_guitar": "acoustic",
}
PARTS = ("distorted", "clean", "acoustic", "guitar_other", "keys", "rest")


def part_of(stem, kind):
    if stem == "guitar":
        return GUITAR.get(kind, "guitar_other")
    return "keys" if stem in ("piano", "other_keys") else "rest"


def song_parts(folder):
    data = json.loads((folder / "data.json").read_text(encoding="utf-8"))
    parts, bleed = {}, False
    for stem in data.get("stems", []):
        name = slug(stem.get("stemName", ""))
        for item in stem.get("tracks", []):
            path = folder / stem["stemName"] / f"{item['id']}.{item.get('extension', 'wav')}"
            if not path.is_file():
                continue
            audio, rate = sf.read(path, dtype="float32", always_2d=True)
            mono = audio.mean(1)
            if rate != RATE:
                mono = resample_poly(mono, RATE, rate).astype(np.float32)
            key = part_of(name, slug(item.get("trackType", "")))
            bleed |= key != "rest" and key != "keys" and bool(item.get("has_bleed"))
            if key in parts:
                length = max(len(parts[key]), len(mono))
                parts[key] = np.pad(parts[key], (0, length - len(parts[key]))) + np.pad(
                    mono, (0, length - len(mono))
                )
            else:
                parts[key] = mono
    return parts, bleed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", default="corpus-data/corpus/manifest.json")
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard", default="0/1")
    args = parser.parse_args()
    index, count = (int(v) for v in args.shard.split("/"))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    track_of = {
        r.recording_id: r.dataset_track_id for r in registry.load_manifest(args.corpus).recordings
    }
    root = Path(args.dataset_root)
    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))[index::count]
    for job in jobs:
        target = out / f"{job['id']}.npz"
        if target.exists():
            continue
        folder = next(iter(root.glob(f"**/{track_of[job['id']]}/data.json"))).parent
        parts, bleed = song_parts(folder)
        length = max(len(audio) for audio in parts.values())
        arrays = {
            key: np.pad(audio, (0, length - len(audio))).astype(np.float16)
            for key, audio in parts.items()
        }
        np.savez(target, guitar_bleed=bleed, **arrays)
        print(f"{job['id']} {length / RATE:.0f}s {sorted(parts)}", flush=True)


if __name__ == "__main__":
    main()
