"""Fine-tune EfficientAT on guitar presence with mixes made on the fly from MoisesDB parts.

Runs in the experiments environment (torch, CPU). The pretrained network keeps its body;
its last feature map (one column per 0.3125 s over a 10 s clip) gets a new four-output
layer: guitar, distorted, clean, acoustic, per column.

Every training clip is put together from the cached parts (``cache_stems.py``):

``own``    a song with its own guitar parts at changed levels, some parts left out
``none``   the song without any guitar
``cross``  the song's backing with the guitar parts of another song at a random level

and, in any of these, sometimes the keyboards of a third song on top (the hard negative).

Target per column and label, from the parts actually mixed in: present when the guitar
part is within ``PRESENT_DB`` of the mix and above the floor, absent when more than
``ABSENT_DB`` below (or missing), not used in between.

Scoring is done the way a live worker would run: every ``--hop`` seconds the last 10 s
ending *now* are given to the network and only its newest column is kept. Held-out songs
are scored as they are and as their guitar-free twin.

    python experiments/recognition/finetune.py --jobs JOBS.json --parts PARTS_DIR \
        --labels LABELS_DIR --source EfficientAT-src --fold 0/4 --scores OUT_DIR \
        --weights .model-cache/guitar_mn10_ft_fold0.pt
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import embed  # noqa: E402
from head import LABELS, fold_of, window_labels  # noqa: E402

RATE, CLIP = 32000, 320000
COLUMNS = 32
GUITAR_PARTS = ("distorted", "clean", "acoustic", "guitar_other")
PRESENT_DB, ABSENT_DB, FLOOR_DBFS = -20.0, -30.0, -55.0


class Net(torch.nn.Module):
    def __init__(self, body):
        super().__init__()
        self.body = body
        self.drop = torch.nn.Dropout(0.2)
        self.out = torch.nn.Conv1d(960, len(LABELS), 1)

    def forward(self, mel):
        return self.out(self.drop(self.body(mel).mean(2)))  # (batch, labels, columns)


def load_parts(folder, jobs):
    songs = []
    for job in jobs:
        path = Path(folder) / f"{job['id']}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            parts = {k: data[k] for k in data.files if k != "guitar_bleed"}
        songs.append({"id": job["id"], "group": job["group"], "parts": parts})
    return songs


def cut(song, rng, names):
    """Random 10 s of the named parts of a song (zeros where a part does not exist)."""
    length = len(next(iter(song["parts"].values())))
    start = int(rng.integers(0, max(1, length - CLIP)))
    out = {}
    for name in names:
        if name in song["parts"]:
            audio = song["parts"][name][start : start + CLIP].astype(np.float32)
            out[name] = np.pad(audio, (0, CLIP - len(audio)))
    return out


def rms_db(audio):
    return 10 * np.log10(np.mean(audio.reshape(COLUMNS, -1) ** 2, axis=1) + 1e-12)


def gain(rng, low, high):
    return 10 ** (rng.uniform(low, high) / 20)


def make_clip(songs, rng):
    """(audio, target, weight): one training clip and its per-column labels."""
    song = songs[rng.integers(len(songs))]
    mode = rng.choice(["own", "none", "cross"], p=[0.45, 0.25, 0.30])
    backing = cut(song, rng, ("rest", "keys", *GUITAR_PARTS))
    rest = backing.get("rest", np.zeros(CLIP, np.float32)) * gain(rng, -3, 3)
    if "keys" in backing and rng.random() > 0.1:
        rest = rest + backing["keys"] * gain(rng, -6, 6)
    if rng.random() < 0.2:  # somebody else's keyboards on top
        other = cut(songs[rng.integers(len(songs))], rng, ("keys",))
        if "keys" in other:
            rest = rest + other["keys"] * gain(rng, -6, 6)
    guitars = {}
    if mode == "own":
        guitars = {
            name: backing[name] * gain(rng, -9, 6)
            for name in GUITAR_PARTS
            if name in backing and rng.random() > 0.15
        }
    elif mode == "cross":
        donor = cut(songs[rng.integers(len(songs))], rng, GUITAR_PARTS)
        level = np.sqrt(np.mean(rest**2)) + 1e-6
        for name, audio in donor.items():
            own = np.sqrt(np.mean(audio**2))
            if own > 1e-4:
                guitars[name] = audio * (level / own) * gain(rng, -15, 3)
    mix = rest + sum(guitars.values()) if guitars else rest
    scale = gain(rng, -12, 0) / max(1.0, float(np.abs(mix).max()))
    mix_db = rms_db(mix * scale)
    target = np.zeros((len(LABELS), COLUMNS), np.float32)
    weight = np.ones_like(target)
    groups = {
        "guitar": GUITAR_PARTS,
        "guitar_distorted": ("distorted",),
        "guitar_clean": ("clean",),
        "guitar_acoustic": ("acoustic",),
    }
    for index, label in enumerate(LABELS):
        chosen = [guitars[name] for name in groups[label] if name in guitars]
        if not chosen:
            continue
        level = rms_db(sum(chosen) * scale)
        relative = level - mix_db
        present = (relative > PRESENT_DB) & (level > FLOOR_DBFS)
        absent = (relative < ABSENT_DB) | (level < FLOOR_DBFS - 10)
        target[index] = present
        weight[index] = present | absent
    return (mix * scale).astype(np.float32), target, weight


def causal_scores(net, mel, audio, hop, batch=24):
    """Per hop: the network's newest column, given only the 10 s ending at that hop."""
    ends = np.arange(round(hop * RATE), len(audio) + 1, round(hop * RATE))
    padded = torch.cat([torch.zeros(CLIP), torch.from_numpy(audio)])
    rows = []
    for begin in range(0, len(ends), batch):
        chunk = torch.stack([padded[e : e + CLIP] for e in ends[begin : begin + batch]])
        with torch.inference_mode():
            rows.append(torch.sigmoid(net(mel(chunk).unsqueeze(1))[:, :, -1]).numpy())
    return ends / RATE, np.concatenate(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--parts", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", default="mn10_as")
    parser.add_argument("--fold", default="0/4", help="held-out artist fold; -1/4 holds out none")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--rate", type=float, default=2e-4)
    parser.add_argument("--hop", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--scores")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    fold, folds = (int(v) for v in args.fold.split("/"))
    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    held = [j for j in jobs if fold_of(j["group"], folds) == fold]
    songs = load_parts(args.parts, [j for j in jobs if fold_of(j["group"], folds) != fold])
    print(f"{len(songs)} training songs, {len(held)} held out", flush=True)

    body, mel = embed.load_model(Path(args.source).resolve(), args.model)
    mel.freqm, mel.timem = torch.nn.Identity(), torch.nn.Identity()
    net = Net(body.features)
    net.eval()  # the pretrained normalisation statistics are kept
    net.drop.train()
    optimiser = torch.optim.AdamW(
        [
            {"params": net.body.parameters(), "lr": args.rate},
            {"params": net.out.parameters(), "lr": args.rate * 10},
        ],
        weight_decay=1e-2,
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, [args.rate, args.rate * 10], total_steps=args.steps, pct_start=0.1
    )
    started, running = perf_counter(), 0.0
    for step in range(1, args.steps + 1):
        clips = [make_clip(songs, rng) for _ in range(args.batch)]
        audio = torch.from_numpy(np.stack([c[0] for c in clips]))
        target = torch.from_numpy(np.stack([c[1] for c in clips]))
        weight = torch.from_numpy(np.stack([c[2] for c in clips]))
        with torch.no_grad():
            spec = mel(audio).unsqueeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(spec), target, weight)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        schedule.step()
        running += float(loss)
        if step % 100 == 0:
            minutes = (perf_counter() - started) / 60
            print(f"step {step} loss {running / 100:.4f} ({minutes:.1f} min)", flush=True)
            running = 0.0
    net.drop.eval()
    Path(args.weights).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "labels": LABELS, "model": args.model}, args.weights)

    if not args.scores or not held:
        return
    out = Path(args.scores)
    out.mkdir(parents=True, exist_ok=True)
    for song in load_parts(args.parts, held):
        parts = song["parts"]
        guitar = [parts[name].astype(np.float32) for name in GUITAR_PARTS if name in parts]
        rest = sum(parts[name].astype(np.float32) for name in ("rest", "keys") if name in parts)
        versions = {"": rest + sum(guitar) if guitar else rest}
        if guitar and not isinstance(rest, int):
            versions[".noguitar"] = rest
        for suffix, audio in versions.items():
            audio = audio / max(1.0, float(np.abs(audio).max()))
            times, scores = causal_scores(net, mel, audio.astype(np.float32), args.hop)
            y, certain, _ = window_labels(Path(args.labels) / f"{song['id']}.npz", times)
            n = min(len(y), len(scores))
            if suffix:
                y, certain = np.zeros_like(y), np.ones_like(certain)
            np.savez(
                out / f"{song['id']}{suffix}.npz",
                t=times[:n],
                score=scores[:n].astype(np.float32),
                y=y[:n],
                certain=certain[:n],
            )
        print(f"scored {song['id']}", flush=True)


if __name__ == "__main__":
    main()
