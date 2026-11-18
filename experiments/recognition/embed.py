"""Cache frozen EfficientAT outputs for the recordings listed by ``prepare.py``.

Runs in the experiments environment (torch). Every ``--hop`` seconds the last 10 seconds of
audio are given to the model, exactly what a live worker would have (see ``windows``). No
audio is written.

For MoisesDB songs the same is done for the "twin": the song summed from its stem files
without the guitar files, saved as ``<id>.noguitar.npz``.

    python experiments/recognition/embed.py --jobs JOBS.json --source EfficientAT-src \
        --model mn10_as --output E:/datasets/gmv-cache/recognition/mn10_as
"""

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
import soxr
import torch

RATE = 32000
CONTEXT = 10.0
RECENT = (2, 3, 6)  # columns of 0.32 s: about 0.6, 1 and 2 seconds


def load_model(source, name):
    # The EfficientAT code reads its metadata and stores downloaded weights relative to the
    # working directory.
    sys.path.insert(0, str(source))
    here = os.getcwd()
    os.chdir(source)
    try:
        from models.preprocess import AugmentMelSTFT

        if name.startswith("dymn"):
            from models.dymn.model import get_model
        else:
            from models.mn.model import get_model
        from helpers.utils import NAME_TO_WIDTH

        model = get_model(width_mult=NAME_TO_WIDTH(name), pretrained_name=name).eval()
        mel = AugmentMelSTFT(n_mels=128, sr=RATE, win_length=800, hopsize=320).eval()
    finally:
        os.chdir(here)
    return model, mel


def mono(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    return audio.mean(1), rate


def twin(files):
    total, rate = None, None
    for path in files:
        audio, rate = mono(path)
        if total is None:
            total = audio
        else:
            length = max(len(total), len(audio))
            total = np.pad(total, (0, length - len(total))) + np.pad(
                audio, (0, length - len(audio))
            )
    return total, rate


def windows(model, mel, audio, hop, batch=16):
    """Model outputs for the context ending at each hop.

    The model is given the last ``CONTEXT`` seconds (silence before the start of the
    recording): it was trained on 10 s clips and its outputs on clips shorter than about
    5 s are not usable. Its last feature map has one column per 0.32 s. Kept per hop: the
    map averaged over the newest ``k`` columns for each ``k`` in ``RECENT`` (what was
    played just now), averaged over the whole context, and the 527 AudioSet outputs of
    the whole context (before the sigmoid).
    """
    ends = np.arange(round(hop * RATE), len(audio) + 1, round(hop * RATE))
    size = round(CONTEXT * RATE)
    padded = torch.cat([torch.zeros(size), torch.from_numpy(audio)])
    out = {f"e{k}": [] for k in RECENT} | {"eall": [], "lall": []}
    for begin in range(0, len(ends), batch):
        chunk = torch.stack([padded[e : e + size] for e in ends[begin : begin + batch]])
        with torch.inference_mode():
            fmap = model.features(mel(chunk).unsqueeze(1))
            pooled = fmap.mean((2, 3))
            out["lall"].append(model.classifier[2:](pooled).half().numpy())
            out["eall"].append(pooled.half().numpy())
            for k in RECENT:
                out[f"e{k}"].append(fmap[..., -k:].mean((2, 3)).half().numpy())
    return {name: np.concatenate(rows) for name, rows in out.items()}, ends / RATE


def save(target, model, mel, audio, rate, hop):
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak > 1:
        audio = audio / peak  # as the corpus does for rendered mixes
    audio = soxr.resample(audio, rate, RATE) if rate != RATE else audio
    result, times = windows(model, mel, audio.astype(np.float32), hop)
    np.savez(target, t=times.astype(np.float32), **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--source", required=True, help="checkout of the EfficientAT code")
    parser.add_argument("--model", default="mn10_as")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hop", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--shard", default="0/1")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    index, count = (int(v) for v in args.shard.split("/"))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model, mel = load_model(Path(args.source).resolve(), args.model)
    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))[index::count]
    for job in jobs:
        started = perf_counter()
        target = out / f"{job['id']}.npz"
        if not target.exists():
            save(target, model, mel, *mono(job["path"]), args.hop)
        target = out / f"{job['id']}.noguitar.npz"
        if job.get("twin_files") and not target.exists():
            save(target, model, mel, *twin(job["twin_files"]), args.hop)
        print(f"{job['id']} {job['duration']:.0f}s in {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
