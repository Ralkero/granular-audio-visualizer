"""A second audio model, independent of the project's own recogniser in architecture and
training recipe: PANNs CNN14 with frame-wise output (Kong et al. 2020, ``DecisionLevelMax``).

Runs in the experiments environment, CPU only. It is evidence, not a labeller: both this
model and the project's EfficientAT model were pretrained on AudioSet, so their agreement is
not statistically independent confirmation.

The model gives 527 AudioSet class probabilities 100 times a second from 32 kHz mono audio.
Kept here: the classes in ``CLASSES``, as the mean and the maximum of each half second
(``hop``), so timelines line up with the recogniser's hops.

    python experiments/evidence/panns.py --weights Cnn14_DecisionLevelMax_mAP=0.385.pth \
        scan --inventory INVENTORY.csv --output OUT_DIR [--jobs 2] [--limit N]
    python experiments/evidence/panns.py --weights W.pth review --cache REVIEW_CACHE --output OUT.npz
"""

import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

RATE = 32000
HOP_S = 0.5
CLASSES = {
    "guitar": "Guitar",
    "electric_guitar": "Electric guitar",
    "acoustic_guitar": "Acoustic guitar",
    "steel_guitar": "Steel guitar, slide guitar",
    "strum": "Strum",
    "plucked_string": "Plucked string instrument",
    "bass_guitar": "Bass guitar",
    "banjo": "Banjo",
    "mandolin": "Mandolin",
    "keyboard": "Keyboard (musical)",
    "piano": "Piano",
    "electric_piano": "Electric piano",
    "organ": "Organ",
    "synthesizer": "Synthesizer",
    "orchestra": "Orchestra",
    "bowed_string": "Bowed string instrument",
    "string_section": "String section",
    "violin": "Violin, fiddle",
    "cello": "Cello",
    "brass": "Brass instrument",
    "drum_kit": "Drum kit",
    "singing": "Singing",
    "music": "Music",
    "speech": "Speech",
}
MODEL = "PANNs Cnn14_DecisionLevelMax (mAP 0.385)"


def load(weights, threads):
    torch.set_num_threads(threads)
    from panns_inference import SoundEventDetection, labels

    index = [labels.index(name) for name in CLASSES.values()]
    model = SoundEventDetection(checkpoint_path=str(weights), device="cpu", interpolate_mode="nearest")
    return model, index


def framewise(model, index, mono, segment_s=30.0):
    """(hops, classes) mean and maximum per half second over the whole of ``mono``."""
    per_hop = round(HOP_S * 100)
    size = round(segment_s * RATE)
    means, maxima = [], []
    for start in range(0, len(mono), size):
        piece = mono[start : start + size]
        hops = len(piece) // round(HOP_S * RATE)
        if hops == 0:
            break
        if len(piece) < RATE:  # the network needs about a second
            piece = np.pad(piece, (0, RATE - len(piece)))
        with torch.inference_mode():
            out = model.inference(piece[None, :].astype(np.float32))[0][:, index]
        out = out[: hops * per_hop]
        if len(out) < hops * per_hop:
            out = np.pad(out, ((0, hops * per_hop - len(out)), (0, 0)), mode="edge")
        out = out.reshape(hops, per_hop, len(index))
        means.append(out.mean(axis=1))
        maxima.append(out.max(axis=1))
    if not means:
        empty = np.zeros((0, len(index)), np.float32)
        return empty, empty
    return np.concatenate(means).astype(np.float32), np.concatenate(maxima).astype(np.float32)


def decode(path):
    """32 kHz mono float32 through ffmpeg (every format in the collection); read only."""
    done = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0", "-ac", "1"]
        + ["-ar", str(RATE), "-f", "f32le", "-"],
        capture_output=True,
    )
    if done.returncode:
        raise RuntimeError(done.stderr.decode("utf-8", "replace").strip()[:300])
    return np.frombuffer(done.stdout, dtype="<f4")


def scan(args):
    with open(args.inventory, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    rows = [r for i, r in enumerate(rows) if i % args.jobs == args.job][: args.limit or None]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model, index = load(args.weights, args.threads)
    weights_sha = hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()
    done = audio_s = 0.0
    started = perf_counter()
    for row in rows:
        target = out / f"r-{row['sha256'][:12]}.npz"
        if target.is_file():
            continue
        try:
            mono = decode(row["file_path"])
            mean, peak = framewise(model, index, mono)
        except Exception as error:  # one unreadable file must not stop the scan
            print(f"FAILED {target.stem}: {error}", flush=True)
            continue
        temporary = target.with_suffix(".part.npz")
        np.savez_compressed(
            temporary,
            mean=mean,
            max=peak,
            classes=np.array(list(CLASSES)),
            hop_s=HOP_S,
            model=MODEL,
            weights_sha256=weights_sha,
            source_sha256=row["sha256"],
        )
        temporary.replace(target)
        done += 1
        audio_s += len(mono) / RATE
        if done % 20 == 0:
            spent = perf_counter() - started
            print(f"{int(done)} recordings, {audio_s / spent:.1f}x real time", flush=True)
    spent = perf_counter() - started
    print(f"finished: {int(done)} new, {audio_s / 3600:.2f} h of audio in {spent / 60:.1f} min")


def review(args):
    """Scores of the human-labelled excerpts cached by ``cache_review.py``: per excerpt, the
    hops inside the excerpt (the cached audio starts 10 s before it)."""
    model, index = load(args.weights, args.threads)
    out = {}
    started, audio_s = perf_counter(), 0.0
    for path in sorted(Path(args.cache).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            audio, meta = data["audio"], data["meta"]
        for clip, row in zip(audio, meta, strict=True):
            mean, peak = framewise(model, index, clip.astype(np.float32))
            lead = round(10 / HOP_S)
            out[str(row[0])] = np.stack([mean[lead:], peak[lead:]])
            audio_s += len(clip) / RATE
    np.savez_compressed(args.output, classes=np.array(list(CLASSES)), **out)
    print(f"{len(out)} excerpts, {audio_s / (perf_counter() - started):.1f}x real time")


def clips(args):
    """Whole-clip scores of a list of short files (``key<TAB>path`` per line): mean and
    maximum over the clip of every class."""
    model, index = load(args.weights, args.threads)
    keys, means, peaks = [], [], []
    for line in Path(args.list).read_text(encoding="utf-8").splitlines():
        key, path = line.split("	")
        mean, peak = framewise(model, index, decode(path))
        keys.append(key)
        means.append(mean.mean(axis=0))
        peaks.append(peak.max(axis=0))
    np.savez_compressed(
        args.output,
        keys=np.array(keys),
        classes=np.array(list(CLASSES)),
        mean=np.array(means, np.float32),
        max=np.array(peaks, np.float32),
    )
    print(f"{len(keys)} clips")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--threads", type=int, default=6)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--inventory", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--jobs", type=int, default=1, help="split the inventory over this many runs")
    s.add_argument("--job", type=int, default=0)
    s.add_argument("--limit", type=int)
    r = sub.add_parser("review")
    r.add_argument("--cache", required=True)
    r.add_argument("--output", required=True)
    c = sub.add_parser("clips")
    c.add_argument("--list", required=True)
    c.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.exit({"scan": scan, "review": review, "clips": clips}[args.command](args))


if __name__ == "__main__":
    main()
