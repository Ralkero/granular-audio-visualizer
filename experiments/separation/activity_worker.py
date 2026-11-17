"""Separate dataset mixtures with HTDemucs and keep only each estimated stem's level curve.

Runs in the isolated separation environment (torch, Demucs). Separated audio stays in
memory and is never written; the output per track is the short-term level of each
estimated stem, measured exactly as ``gmv.corpus.stems`` measures the dataset's own stems.

Two passes per track:

``full``    the whole track at once (Demucs' usual overlapping segments). Every sample is
            separated with several seconds of audio on both sides. Not possible live.
``causal``  what a live worker could do: every ``--hop`` seconds the latest model segment
            (7.8 s) ending *now* is separated and only its last ``--hop`` seconds are kept,
            so no sample is separated with more than ``--hop`` seconds of later audio.
"""

import argparse
import io
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.hf import load_safetensors_model

from setup_model import verify

HOP_S, WINDOW_S = 0.01, 0.05  # gmv.corpus.stems
WEIGHTS = Path(".model-cache/955717e8.safetensors")


def levels(audio, rate):
    """Level in dBFS of (channels, samples) audio; same windows as gmv.corpus.stems."""
    power = (audio.double() ** 2).mean(0).numpy()
    hop, window = max(1, round(rate * HOP_S)), max(1, round(rate * WINDOW_S))
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    starts = np.arange(0, max(1, len(power) - window + 1), hop)
    rms = np.sqrt(
        (cumulative[np.minimum(starts + window, len(power))] - cumulative[starts]) / window
    )
    return (20 * np.log10(np.maximum(rms, 1e-12))).astype(np.float32)


def normalised(mix):
    mono = mix.mean(-2, keepdim=True)
    mean = mono.mean(-1, keepdim=True)
    std = mono.std(-1, keepdim=True).clamp(min=1e-8)
    return (mix - mean) / std, mean, std


def separate_full(model, mix):
    scaled, mean, std = normalised(mix)
    with torch.inference_mode():
        out = apply_model(
            model, scaled[None], shifts=0, split=True, overlap=0.25, device="cpu", progress=False
        )[0]
    return out * std + mean


def separate_causal(model, mix, hop, batch):
    """Yield (start, stems) for consecutive ``hop``-sample pieces, each separated from the
    model segment that ends where the piece ends."""
    segment = int(model.segment * model.samplerate)
    total = mix.shape[-1]
    ends = list(range(hop, total, hop)) + [total]
    padded = torch.nn.functional.pad(mix, (segment, 0))
    for first in range(0, len(ends), batch):
        group = ends[first : first + batch]
        windows = torch.stack([padded[:, end : end + segment] for end in group])
        scaled, mean, std = normalised(windows)
        with torch.inference_mode():
            out = apply_model(model, scaled, shifts=0, split=False, device="cpu", progress=False)
        out = out * std[:, None] + mean[:, None]
        for row, end in zip(out, group, strict=True):
            start = end - hop if end % hop == 0 else end - end % hop
            yield start, row[..., segment - (end - start) :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True, help="JSON list of {id, path}")
    parser.add_argument("--output", required=True, help="Directory for per-track .npz")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--hop", type=float, default=2.0, help="Seconds of new audio per pass")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--limit-seconds", type=float, help="Only the start of each track")
    parser.add_argument("--passes", default="full,causal")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    verify(WEIGHTS)
    model = load_safetensors_model(WEIGHTS).cpu().eval()
    rate = model.samplerate
    index, count = map(int, args.shard.split("/"))
    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))[index::count]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        target = output / f"{job['id']}.npz"
        if target.exists():
            continue
        audio, file_rate = sf.read(
            io.BytesIO(Path(job["path"]).read_bytes()), dtype="float32", always_2d=True
        )
        if file_rate != rate or audio.shape[1] != 2:
            raise ValueError(f"{job['id']}: expected {rate} Hz stereo")
        if args.limit_seconds:
            audio = audio[: int(args.limit_seconds * rate)]
        mix = torch.from_numpy(audio.T.copy())
        result = {"sources": np.array(model.sources), "seconds": mix.shape[-1] / rate}
        if "full" in args.passes:
            began = perf_counter()
            stems = separate_full(model, mix)
            result["full_seconds"] = perf_counter() - began
            result["full"] = np.stack([levels(stem, rate) for stem in stems])
            del stems
        if "causal" in args.passes:
            began = perf_counter()
            stems = torch.zeros(len(model.sources), *mix.shape)
            passes = 0
            for start, piece in separate_causal(model, mix, int(args.hop * rate), args.batch):
                stems[..., start : start + piece.shape[-1]] = piece
                passes += 1
            result["causal_seconds"] = perf_counter() - began
            result["causal_passes"] = passes
            result["causal"] = np.stack([levels(stem, rate) for stem in stems])
            del stems
        result.update(threads=args.threads, hop=args.hop, batch=args.batch)
        temporary = target.with_suffix(".partial.npz")
        np.savez(temporary, **result)
        temporary.replace(target)
        print(
            job["id"],
            f"{result['seconds']:.0f}s",
            *(f"{k} {result[k]:.1f}s" for k in ("full_seconds", "causal_seconds") if k in result),
            flush=True,
        )


if __name__ == "__main__":
    main()
