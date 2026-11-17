"""Cache timbre features (``timbre.py``) and labels for training an instrument detector.

Reuses the activity trainer's corpus walk, remixes and label rules; only the per-frame
features differ. Also writes ``groups.json`` (recording -> artist group and split) so the
trainer can form folds by artist. The held-out test split is never read.

    python experiments/guitar/cache_timbre.py --corpus corpus-data/corpus/manifest.json --dataset moisesdb \
        --dataset-root C:/Datasets/moisesdb_v0.1 --cache E:/datasets/gmv-cache/timbre_v1 \
        --labels guitar,guitar_distorted,guitar_clean,guitar_acoustic
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from gmv.corpus import registry
from timbre import LogBands, normalise


def trainer_module():
    spec = importlib.util.spec_from_file_location(
        "train_activity", Path(__file__).parents[2] / "scripts" / "train_activity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frame_rows(audio, rate, config):
    """Same contract as the activity trainer's ``frame_rows``, with timbre features."""
    hop = round(rate * config.hop_ms / 1000)
    bands = LogBands(rate)
    mono = audio.mean(axis=1).astype(np.float32)
    rows = normalise(bands.frames(mono, hop), hop / rate)
    count = len(rows)
    # Level of the newest hop of each window, as the engine's silence gate sees it.
    ends = np.arange(count) * hop + bands.size
    power = np.concatenate(([0.0], np.cumsum(mono.astype(np.float64) ** 2)))
    rms = np.sqrt((power[ends] - power[ends - hop]) / hop)
    level = np.maximum(-120, 20 * np.log10(np.maximum(rms, 1e-6))).astype(np.float32)
    # The label belongs to what is sounding "now": the end of the window, less the attack
    # of the analysis window.
    times = ends / rate - 0.02
    return rows.astype(np.float16), level, times, hop / rate


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--shard", default="0/1")
    args = parser.parse_args()
    trainer = trainer_module()
    trainer.configure(tuple(label.strip() for label in args.labels.split(",")))
    trainer.frame_rows = frame_rows
    shard = tuple(int(part) for part in args.shard.split("/"))
    trainer.cache_features(
        Path(args.corpus), args.dataset, args.cache, root=args.dataset_root, shard=shard
    )
    manifest = registry.load_manifest(args.corpus)
    groups = {
        r.recording_id: {"group": r.group_id, "split": r.split}
        for r in manifest.recordings
        if r.dataset == args.dataset and r.split in trainer.SPLITS
    }
    (args.cache / "groups.json").write_text(json.dumps(groups, indent=1), encoding="utf-8")
    (args.cache / "labels.json").write_text(json.dumps(list(trainer.LABELS)), encoding="utf-8")


if __name__ == "__main__":
    main()
