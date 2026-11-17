"""Cache timbre band power separately for each guitar type and for everything else.

Band power adds (approximately) when signals are mixed, so with these parts the trainer can
build new mixes on the fly: the guitar of one artist over the backing of another, at any
level, or a backing with no guitar at all. That breaks the link between "this artist's
production" and "has guitar", which a small dataset otherwise teaches. Train and validation
recordings only; the held-out test split is never read.

    python experiments/guitar/cache_timbre_parts.py --corpus corpus-data/corpus/manifest.json \
        --dataset-root C:/Datasets/moisesdb_v0.1 --cache E:/datasets/gmv-cache/timbre_parts_v1
"""

import argparse
from pathlib import Path

import numpy as np

from cache_timbre import trainer_module
from gmv.corpus import registry
from gmv.corpus.datasets import adapters, read_sum, unique_files
from gmv.models import AnalysisConfig
from timbre import LogBands

TYPES = ("guitar_distorted", "guitar_clean", "guitar_acoustic")
LABELS = ("guitar",) + TYPES


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--dataset", default="moisesdb")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--shard", default="0/1")
    args = parser.parse_args()
    trainer = trainer_module()
    trainer.configure(LABELS)
    index, count = (int(part) for part in args.shard.split("/"))
    manifest = registry.load_manifest(args.corpus)
    tracks = {t.track_id: t for t in adapters()[args.dataset].discover(Path(args.dataset_root))}
    args.cache.mkdir(parents=True, exist_ok=True)
    config = AnalysisConfig()
    chosen = [
        r for r in manifest.recordings if r.dataset == args.dataset and r.split in trainer.SPLITS
    ]
    for recording in chosen[index::count]:
        target = args.cache / f"{recording.recording_id}.npz"
        if target.exists():
            continue
        track = tracks[recording.dataset_track_id]
        loaded = {}

        def read(path, loaded=loaded):
            if path not in loaded:
                loaded[path] = registry.read_audio(path)
            return loaded[path]

        guitar_files = set(track.stems.get("guitar", ()))
        typed = {name: track.stems.get(name, []) for name in TYPES}
        if guitar_files - {p for files in typed.values() for p in files}:
            print(f"skipped {recording.recording_id}: a guitar file has no known type")
            continue
        groups = {"rest": [p for p in unique_files(track) if p not in guitar_files], **typed}
        arrays, frames = {}, None
        for name, files in groups.items():
            if not files:
                continue
            audio, rate = read_sum(files, read)
            bands = LogBands(rate)
            hop = round(rate * config.hop_ms / 1000)
            arrays[name] = bands.frames(audio.mean(axis=1).astype(np.float32), hop)
            frames = len(arrays[name]) if frames is None else min(frames, len(arrays[name]))
        if "rest" not in arrays:
            print(f"skipped {recording.recording_id}: nothing but guitar")
            continue
        times = (np.arange(frames) * hop + bands.size) / rate - 0.02
        active, known = trainer.frame_labels(args.corpus, recording, times)
        np.savez(
            target,
            y=active,
            k=known,
            hop=hop / rate,
            **{name: values[:frames].astype(np.float16) for name, values in arrays.items()},
        )
        print(f"cached {recording.recording_id}: {frames} frames, {sorted(arrays)}", flush=True)


if __name__ == "__main__":
    main()
