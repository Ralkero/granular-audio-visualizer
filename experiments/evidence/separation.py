"""Separation as supporting evidence: the level of HTDemucs six-source's guitar output
relative to the mix, over chosen stretches of chosen recordings.

Not a labeller. The guitar output contains bleed from other instruments (measured earlier
in this project: recall 0.92 with a 0.25 false-positive rate as a detector). What is kept is
one number per half second, the guitar stem's level relative to the mix in dB, so the fusion
step can ask "does a third, differently trained method point the same way?". No separated
audio is written.

Runs in the experiments environment, CPU only.

    # pilot: stretches around human-labelled excerpts, to see what the number is worth
    python experiments/evidence/separation.py pilot --inventory INVENTORY.csv \
        --review review-local/pilot-v1 --output research-local/evidence-v1/separation
    # chosen stretches: a JSON list of {"recording_id", "file_path", "spans": [[a, b], ...]}
    python experiments/evidence/separation.py spans --jobs JOBS.json --output OUT_DIR
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "separation"))
HOP_S = 0.5
RATE = 44100
PAD_S = 8.0
WEIGHTS = Path(".model-cache/5c90dfd2-34c22ccb.th")
MODEL = "htdemucs_6s/5c90dfd2, shifts=0, overlap=0.25"


def load():
    from demucs.states import load_model

    return load_model(WEIGHTS).cpu().eval()


def decode(path, start, length):
    done = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(path)]
        + ["-map", "0:a:0", "-ac", "2", "-ar", str(RATE), "-f", "f32le", "-"],
        capture_output=True,
    )
    if done.returncode:
        raise RuntimeError(done.stderr.decode("utf-8", "replace").strip()[:300])
    return torch.from_numpy(np.frombuffer(done.stdout, dtype="<f4").reshape(-1, 2).T.copy())


def block_db(audio):
    """RMS level per half second of (channels, samples) audio, in dBFS."""
    mono_power = (audio.double() ** 2).mean(0).numpy()
    size = round(HOP_S * RATE)
    count = len(mono_power) // size
    blocks = mono_power[: count * size].reshape(count, size).mean(axis=1)
    return 10 * np.log10(np.maximum(blocks, 1e-12))


def separate(model, mix):
    from demucs.apply import apply_model

    mono = mix.mean(0, keepdim=True)
    mean, std = mono.mean(), mono.std().clamp(min=1e-8)
    with torch.inference_mode():
        out = apply_model(
            model, ((mix - mean) / std)[None], shifts=0, split=True, overlap=0.25, device="cpu", progress=False
        )[0]
    return out * std + mean


def run(jobs, output, threads):
    torch.set_num_threads(threads)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = load()
    names = list(model.sources)
    weights_sha = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    audio_s, started = 0.0, perf_counter()
    for job in jobs:
        target = output / f"{job['recording_id']}.npz"
        if target.is_file():
            continue
        hops = int(job["duration_s"] / HOP_S) + 1
        levels = {name: np.full(hops, np.nan, np.float32) for name in (*names, "mix")}
        try:
            for start, end in job["spans"]:
                start = max(0.0, np.floor((start - PAD_S) / HOP_S) * HOP_S)
                length = min(job["duration_s"], end + PAD_S) - start
                mix = decode(job["file_path"], start, length)
                if mix.shape[1] < RATE:
                    continue
                stems = separate(model, mix)
                first = round(start / HOP_S)
                for name, level in (*zip(names, map(block_db, stems), strict=True), ("mix", block_db(mix))):
                    count = min(len(level), hops - first)
                    levels[name][first : first + count] = level[:count]
                audio_s += length
        except Exception as error:
            print(f"FAILED {job['recording_id']}: {error}", flush=True)
            continue
        temporary = target.with_suffix(".part.npz")
        np.savez_compressed(
            temporary,
            guitar_db=levels["guitar"] - levels["mix"],
            piano_db=levels["piano"] - levels["mix"],
            other_db=levels["other"] - levels["mix"],
            mix_dbfs=levels["mix"],
            hop_s=HOP_S,
            model=MODEL,
            weights_sha256=weights_sha,
        )
        temporary.replace(target)
        spent = perf_counter() - started
        print(f"{job['recording_id']}: {audio_s / 60:.1f} min of audio so far, {audio_s / spent:.2f}x real time", flush=True)
    return audio_s, perf_counter() - started


def pilot_jobs(inventory, review, limit):
    """Recordings with human labels, spread over what was answered: guitar-free first (the
    rare kind), then each guitar type; at most two per artist folder."""
    with open(inventory, newline="", encoding="utf-8-sig") as handle:
        rows = {"r-" + r["sha256"][:12]: r for r in csv.DictReader(handle)}
    review = Path(review)
    with open(review / "Excerpt_Manifest.csv", newline="", encoding="utf-8-sig") as handle:
        spans = {r["excerpt_id"]: (float(r["source_start_s"]), float(r["source_end_s"])) for r in csv.DictReader(handle)}
    labelled = {}
    with open(review / "Human_Annotations.csv", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["annotation_status"] == "labelled" and row["excerpt_category"] == "core":
                kind = "absent" if row["guitar_presence"] == "absent" else row["guitar_type"] or "guitar"
                labelled.setdefault(row["recording_id"], []).append((spans[row["excerpt_id"]], kind))
    order = sorted(labelled, key=lambda rid: hashlib.sha256(f"gmv-separation-pilot:{rid}".encode()).hexdigest())
    chosen, per_artist = [], {}
    for wanted, quota in (("absent", 8), ("clean_electric", 4), ("distorted_electric", 4), ("acoustic", 3), ("", limit)):
        for rid in order:
            if len(chosen) >= limit or quota <= 0:
                break
            if rid in chosen or rid not in rows:
                continue
            kinds = {kind for _, kind in labelled[rid]}
            artist = rows[rid]["top_folder"]
            if (wanted and wanted not in kinds) or per_artist.get(artist, 0) >= 2:
                continue
            chosen.append(rid)
            per_artist[artist] = per_artist.get(artist, 0) + 1
            quota -= 1
    return [
        {
            "recording_id": rid,
            "file_path": rows[rid]["file_path"],
            "duration_s": float(rows[rid]["duration_s"]),
            "spans": [list(span) for span, _ in labelled[rid]],
        }
        for rid in chosen
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--threads", type=int, default=12)
    sub = parser.add_subparsers(dest="command", required=True)
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--inventory", required=True)
    pilot.add_argument("--review", required=True)
    pilot.add_argument("--output", required=True)
    pilot.add_argument("--recordings", type=int, default=20)
    spans = sub.add_parser("spans")
    spans.add_argument("--jobs", required=True)
    spans.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "pilot":
        jobs = pilot_jobs(args.inventory, args.review, args.recordings)
    else:
        jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    audio_s, spent = run(jobs, args.output, args.threads)
    print(f"finished: {len(jobs)} recordings, {audio_s / 60:.1f} min of audio in {spent / 60:.1f} min")


if __name__ == "__main__":
    main()
