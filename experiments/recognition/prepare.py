"""List the recordings a recognition experiment reads, and write their reference labels.

Runs in the project environment. For each recording of the chosen splits it writes

* a job (mix path, split, artist group, and for MoisesDB the stem files that make up the
  same song without its guitar: the "twin"), and
* ``<labels>/<id>.npz``: guitar / distorted / clean / acoustic activity every 20 ms from the
  dataset's own stems (``y``), which of those frames are certain (``k``), and the level of
  the summed guitar files and of the mix (``guitar_db``, ``mix_db``) for the "audible"
  reference.

The embedding step (``embed.py``) runs in the experiments environment and only needs the
job list.

    python experiments/recognition/prepare.py --corpus corpus-data/corpus/manifest.json \
        --dataset moisesdb --dataset-root C:/Datasets/moisesdb_v0.1 --splits train,validation \
        --labels E:/datasets/gmv-cache/recognition/labels --jobs local-evidence/guitar/jobs.json
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from gmv.corpus import registry
from gmv.corpus.datasets import adapters, read_sum, unique_files
from gmv.corpus.stems import levels

LABELS = ("guitar", "guitar_distorted", "guitar_clean", "guitar_acoustic")
HOP_S = 0.02


def trainer_module():
    path = Path(__file__).parents[2] / "scripts" / "train_activity.py"
    spec = importlib.util.spec_from_file_location("train_activity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure(LABELS)
    return module


def stem_names(track_file):
    """Names of the dataset's own stems for a MoisesDB song (for condition tags)."""
    meta = Path(track_file).parents[1] / "data.json"
    if not meta.is_file():
        return []
    data = json.loads(meta.read_text(encoding="utf-8"))
    return sorted({stem.get("stemName", "") for stem in data.get("stems", [])})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--jobs", required=True)
    args = parser.parse_args()
    trainer = trainer_module()
    manifest = registry.load_manifest(args.corpus)
    paths = registry.load_paths(args.corpus)
    out = Path(args.labels)
    out.mkdir(parents=True, exist_ok=True)
    tracks = {}
    if args.dataset_root:
        found = adapters()[args.dataset].discover(Path(args.dataset_root))
        tracks = {t.track_id: t for t in found}
    jobs = []
    for recording in manifest.recordings:
        if recording.dataset != args.dataset or recording.split not in args.splits.split(","):
            continue
        path, problem = registry.resolve(recording, paths, verify_hash=False)
        if not path:
            print(f"skipped {recording.recording_id}: {problem}")
            continue
        job = {
            "id": recording.recording_id,
            "dataset": recording.dataset,
            "split": recording.split,
            "group": recording.group_id,
            "duration": recording.duration_s,
            "path": str(path),
        }
        target = out / f"{recording.recording_id}.npz"
        track = tracks.get(recording.dataset_track_id)
        guitar = sorted(track.stems.get("guitar", [])) if track else []
        if track:
            rest = [p for p in unique_files(track) if p not in guitar]
            job["stems"] = stem_names(unique_files(track)[0])
            # The same song without its guitar files. Not possible when guitar is all there is.
            if guitar and rest:
                job["twin_files"] = [str(p) for p in rest]
        if not target.exists():
            times = np.arange(0, recording.duration_s, HOP_S) + HOP_S / 2
            y, k = trainer.frame_labels(args.corpus, recording, times)
            extra = {}
            if guitar:
                audio, rate = registry.read_audio(path)
                stem, _ = read_sum(guitar)
                extra = {
                    "guitar_db": levels(stem, rate)[0].astype(np.float32),
                    "mix_db": levels(audio, rate)[0].astype(np.float32),
                }
            np.savez(target, y=y, k=k, hop=HOP_S, labels=np.array(LABELS), **extra)
            print(f"labels {recording.split} {recording.recording_id}", flush=True)
        jobs.append(job)
    Path(args.jobs).parent.mkdir(parents=True, exist_ok=True)
    Path(args.jobs).write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    twins = sum("twin_files" in job for job in jobs)
    print(f"{len(jobs)} recordings, {twins} with a guitar-free twin")


if __name__ == "__main__":
    main()
