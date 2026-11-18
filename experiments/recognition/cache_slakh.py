"""Cache a subset of Slakh2100 as the same mono 32 kHz parts ``cache_stems.py`` writes.

Slakh2100 is synthesized (MIDI rendered with sampled instruments), so it is training help
only and is never pooled with real recordings in a result. Runs in the project environment.

Two steps, because the archive stores the files of all tracks interleaved:

``extract``  reads the archive once and writes only the metadata and stems of every
             ``--every``-th training track below ``--output``. Member names are checked
             against the expected pattern; nothing else is written.
``parts``    turns each extracted track into parts: guitar by General MIDI program name
             (distorted / overdriven, clean / jazz / muted, nylon / steel, other), keyboards
             (piano, organ, synth lead and pad classes) and everything else. Only the middle
             ``--seconds`` of a track are kept.

    python experiments/recognition/cache_slakh.py extract --archive slakh2100_flac_redux.tar.gz \
        --output E:/datasets/slakh2100_subset --every 4
    python experiments/recognition/cache_slakh.py parts --root E:/datasets/slakh2100_subset \
        --output E:/datasets/gmv-cache/recognition/parts_slakh
"""

import argparse
import re
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

RATE = 32000
MEMBER = re.compile(
    r"^slakh2100_flac_redux/train/(Track(\d{5}))/(metadata\.yaml|stems/S\d\d\.flac)$"
)
GUITAR = {
    "Distortion Guitar": "distorted",
    "Overdriven Guitar": "distorted",
    "Electric Guitar (clean)": "clean",
    "Electric Guitar (jazz)": "clean",
    "Electric Guitar (muted)": "clean",
    "Acoustic Guitar (nylon)": "acoustic",
    "Acoustic Guitar (steel)": "acoustic",
}
KEYS = ("Piano", "Organ", "Synth Lead", "Synth Pad")


def extract(args):
    out = Path(args.output)
    kept = set()
    with tarfile.open(args.archive, "r|gz") as archive:
        for member in archive:
            match = MEMBER.match(member.name)
            if not match or not member.isfile() or int(match.group(2)) % args.every:
                continue
            target = out / match.group(1) / match.group(3)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, open(target, "wb") as handle:
                while chunk := source.read(1 << 20):
                    handle.write(chunk)
            if match.group(1) not in kept:
                kept.add(match.group(1))
                if len(kept) % 50 == 0:
                    print(f"{len(kept)} tracks started", flush=True)
    print(f"{len(kept)} tracks extracted to {out}")


def stems_of(metadata):
    """{stem id: {field: value}} from Slakh's metadata.yaml (a plain two-level mapping)."""
    stems, current, inside = {}, None, False
    for line in metadata.read_text(encoding="utf-8").splitlines():
        if line.startswith("stems:"):
            inside = True
        elif inside and re.match(r"^  S\d\d:", line):
            current = stems.setdefault(line.strip()[:-1], {})
        elif inside and current is not None and line.startswith("    ") and ":" in line:
            key, value = line.strip().split(":", 1)
            current[key] = value.strip()
        elif inside and line and not line.startswith(" "):
            inside = False
    return stems


def part_of(stem):
    if stem.get("is_drum") == "true":
        return "rest"
    if stem.get("inst_class") == "Guitar":
        return GUITAR.get(stem.get("midi_program_name", ""), "guitar_other")
    return "keys" if stem.get("inst_class") in KEYS else "rest"


def parts(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    index, count = (int(v) for v in args.shard.split("/"))
    tracks = sorted(p for p in Path(args.root).iterdir() if (p / "metadata.yaml").is_file())
    for track in tracks[index::count]:
        target = out / f"slakh-{track.name}.npz"
        if target.exists():
            continue
        found = {}
        for name, stem in stems_of(track / "metadata.yaml").items():
            path = track / "stems" / f"{name}.flac"
            if stem.get("audio_rendered") != "true" or not path.is_file():
                continue
            audio, rate = sf.read(path, dtype="float32", always_2d=True)
            mono = audio.mean(1)
            middle = len(mono) // 2
            half = round(args.seconds * rate / 2)
            mono = mono[max(0, middle - half) : middle + half]
            if rate != RATE:
                mono = resample_poly(mono, RATE, rate).astype(np.float32)
            key = part_of(stem)
            if key in found:
                length = max(len(found[key]), len(mono))
                found[key] = np.pad(found[key], (0, length - len(found[key]))) + np.pad(
                    mono, (0, length - len(mono))
                )
            else:
                found[key] = mono
        if not found:
            continue
        length = max(len(audio) for audio in found.values())
        # Slakh mixes are quiet; bring the sum to the level range of the real songs.
        peak = float(np.abs(sum(np.pad(a, (0, length - len(a))) for a in found.values())).max())
        scale = 0.7 / peak if peak > 0 else 1.0
        arrays = {
            key: (np.pad(audio, (0, length - len(audio))) * scale).astype(np.float16)
            for key, audio in found.items()
        }
        np.savez(target, guitar_bleed=False, **arrays)
        print(f"{track.name} {length / RATE:.0f}s {sorted(found)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("extract")
    one.add_argument("--archive", required=True)
    one.add_argument("--output", required=True)
    one.add_argument("--every", type=int, default=4)
    two = sub.add_parser("parts")
    two.add_argument("--root", required=True)
    two.add_argument("--output", required=True)
    two.add_argument("--seconds", type=float, default=90.0)
    two.add_argument("--shard", default="0/1")
    args = parser.parse_args()
    (extract if args.command == "extract" else parts)(args)


if __name__ == "__main__":
    main()
