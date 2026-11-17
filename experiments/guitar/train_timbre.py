"""Train the experimental timbre detector from a cache written by ``cache_timbre.py``.

Needs PyTorch, which the application itself does not: run it with an interpreter that has
torch and numpy (on the development machine, ``.separation-venv``). ``--output`` writes the
weights as a plain ``.npz``; nothing in the application reads that file.

Model choices are compared on folds split by artist (``--fold i/n``): the recordings of the
held-out artists are scored, nothing is written. Without ``--fold`` every cached recording is
used and the weights are written.

    python experiments/guitar/train_timbre.py --cache E:/datasets/gmv-cache/timbre_v1 --fold 0/4
    python experiments/guitar/train_timbre.py --cache E:/datasets/gmv-cache/timbre_v1 --output guitar.npz
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

FEATURE_VERSION = "timbre_features_v1"
SILENCE_DB = -65.0
FLOOR = -4.0


def load(cache):
    groups = json.loads((cache / "groups.json").read_text(encoding="utf-8"))
    labels = json.loads((cache / "labels.json").read_text(encoding="utf-8"))
    files = []
    for path in sorted(cache.glob("*.npz")):
        recording = path.name.split("-", 1)[1].split(".")[0]
        data = np.load(path)
        files.append(
            {
                "recording": recording,
                "group": groups[recording]["group"],
                "variant": path.name.count(".") > 1,
                "x": data["x"].astype(np.float32),
                "y": data["y"].astype(np.float32),
                "w": (data["k"] & (data["db"] >= SILENCE_DB)[:, None]).astype(np.float32),
            }
        )
    return files, labels, groups


TYPES = ("guitar_distorted", "guitar_clean", "guitar_acoustic")
LEVEL_S, CEILING, EMPTY = 3.0, 2.0, -10.0


def load_parts(folder, groups):
    """Per recording: log band power of the backing and of each guitar type, with labels."""
    parts = []
    for path in sorted(Path(folder).glob("*.npz")):
        data = np.load(path)
        item = {name: data[name] for name in ("rest", *TYPES) if name in data}
        item.update(y=data["y"], k=data["k"], hop=float(data["hop"]), recording=path.stem)
        item["group"] = groups[path.stem]["group"]
        power = (10.0 ** item["rest"][::5].astype(np.float32)).sum(1)
        item["loudness"] = float(np.log10(power.mean() + 1e-10))
        parts.append(item)
    return parts


AUDIBLE_BELS, INAUDIBLE_BELS = -1.5, -3.0


def audible(part_power, mix_power, active, known):
    """Labels for "can be heard in the mix": the part is within 15 dB of the whole mix
    (over about 0.1 s). Fainter than 30 dB below, or not playing at all, counts as absent;
    in between, and where the stem reference itself is uncertain, the frame is not used."""
    kernel = np.ones(5) / 5
    part = np.convolve(part_power.sum(1), kernel, mode="same")
    mix = np.convolve(mix_power.sum(1), kernel, mode="same") + 1e-12
    ratio = np.log10(part / mix + 1e-12)
    present = (ratio > AUDIBLE_BELS) & (active > 0)
    absent = (ratio < INAUDIBLE_BELS) | ((active == 0) & (known > 0))
    return present.astype(np.float32), ((present | absent) & (known > 0)).astype(np.float32)


def relabel(files, parts, labels):
    """Give the real mixes of ``files`` audibility labels from their cached parts."""
    by_recording = {item["recording"]: item for item in parts}
    kept = []
    for item in files:
        source = by_recording.get(item["recording"])
        if item["variant"] or source is None:
            continue
        count = min(len(item["x"]), len(source["y"]))
        powers = {n: 10.0 ** source[n][:count].astype(np.float32) for n in TYPES if n in source}
        rest = 10.0 ** source["rest"][:count].astype(np.float32)
        mix = rest + sum(powers.values()) if powers else rest
        y, w = np.zeros((count, len(labels)), np.float32), np.ones((count, len(labels)), np.float32)
        for name in ("guitar", *TYPES):
            column = labels.index(name)
            part = sum(powers.values()) if name == "guitar" else powers.get(name)
            if part is None or isinstance(part, int):
                continue
            y[:, column], w[:, column] = audible(
                part, mix, source["y"][:count, column], source["k"][:count, column]
            )
        silent = mix.sum(1) <= 1e-6
        w[silent] = 0
        kept.append({**item, "x": item["x"][:count], "y": y, "w": w})
    return kept


def slide(x, shift):
    """Transpose by sliding the semitone band axis."""
    if not shift:
        return x
    moved = np.full_like(x, EMPTY)
    if shift > 0:
        moved[:, shift:] = x[:, :-shift]
    else:
        moved[:, :shift] = x[:, -shift:]
    return moved


def normalise(frames, hop):
    """Same as ``timbre.normalise``; the level starts from the first half second."""
    a = 1 - np.exp(-hop / LEVEL_S)
    loud = frames.max(axis=1)
    level = np.empty_like(loud)
    current = float(loud[:25].mean())
    for index, value in enumerate(loud):
        current += a * (float(value) - current)
        level[index] = current
    return np.clip(frames - level[:, None], FLOOR, CEILING)


def cut(array, start, crop):
    piece = array[start : start + crop].astype(np.float32)
    return np.pad(piece, ((0, crop - len(piece)), (0, 0)), constant_values=EMPTY)


def synthetic(parts, labels, rng, crop, shift, by_ear=False):
    """A new mix: one recording's backing, optionally another recording's guitar on top."""
    backing = parts[rng.integers(len(parts))]
    start = int(rng.integers(0, max(1, len(backing["rest"]) - crop)))
    power = 10.0 ** slide(cut(backing["rest"], start, crop), int(rng.integers(-shift, shift + 1)))
    active = np.zeros((crop, len(labels)), np.float32)
    known = np.ones((crop, len(labels)), np.float32)
    if rng.random() < 0.65:
        while True:
            source = parts[rng.integers(len(parts))]
            present = [name for name in TYPES if name in source]
            if present:
                break
        chosen = [name for name in present if rng.random() < 0.8] or [
            present[rng.integers(len(present))]
        ]
        begin = int(rng.integers(0, max(1, len(source["y"]) - crop)))
        move = int(rng.integers(-shift, shift + 1))
        # The guitar keeps roughly the level it had against its own backing, give or take.
        gain = 10.0 ** (backing["loudness"] - source["loudness"] + rng.uniform(-0.8, 0.4))
        all_known = np.ones(crop, np.float32)
        added = {}
        for name in chosen:
            added[name] = gain * 10.0 ** slide(cut(source[name], begin, crop), move)
            power = power + added[name]
            column = labels.index(name)
            y = source["y"][begin : begin + crop, column].astype(np.float32)
            k = source["k"][begin : begin + crop, column].astype(np.float32)
            active[: len(y), column], known[: len(k), column] = y, k
            all_known[: len(k)] *= k
        any_guitar = active[:, [labels.index(name) for name in chosen]].max(1)
        column = labels.index("guitar")
        active[:, column] = any_guitar
        known[:, column] = np.maximum(any_guitar, all_known)
        if by_ear:
            for name, part in (("guitar", sum(added.values())), *added.items()):
                column = labels.index(name)
                active[:, column], known[:, column] = audible(
                    part, power, active[:, column], known[:, column]
                )
    known *= (power.sum(1) > 1e-6)[:, None]
    return normalise(np.log10(power + 1e-10), backing["hop"]), active, known


def bucket(group, folds):
    return int(hashlib.sha256(f"gmv-fold-1:{group}".encode()).hexdigest(), 16) % folds


class Net(nn.Module):
    def __init__(self, bands, outputs, width, dilations, taps, dropout):
        super().__init__()
        self.dilations, self.taps = dilations, taps
        sizes = [bands] + [width] * len(dilations)
        self.convs = nn.ModuleList(
            nn.Conv1d(sizes[i], sizes[i + 1], taps, dilation=d) for i, d in enumerate(dilations)
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(width, outputs)

    def forward(self, x):  # (batch, time, bands) -> logits (batch, time, outputs)
        x = x.transpose(1, 2)
        for conv, dilation in zip(self.convs, self.dilations, strict=True):
            x = nn.functional.pad(x, ((self.taps - 1) * dilation, 0))  # causal: past only
            x = self.drop(torch.relu(conv(x)))
        return self.head(x.transpose(1, 2))


def auc(score, truth):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    positive = truth.sum()
    negative = len(truth) - positive
    return float((ranks[truth > 0].sum() - positive * (positive + 1) / 2) / (positive * negative))


def evaluate(net, files, mean, scale, labels):
    net.eval()
    scores, truth, weight, absent = [], [], [], [[] for _ in labels]
    with torch.no_grad():
        for item in files:
            x = torch.from_numpy((item["x"] - mean) / scale)[None]
            s = torch.sigmoid(net(x))[0].numpy()
            scores.append(s)
            truth.append(item["y"])
            weight.append(item["w"])
            for k in range(len(labels)):
                use = item["w"][:, k] > 0
                if use.any() and not item["y"][use, k].any():
                    absent[k].append(float((s[use, k] >= 0.5).mean()))
    scores, truth, weight = (np.concatenate(v) for v in (scores, truth, weight))
    result = {}
    for k, label in enumerate(labels):
        use = weight[:, k] > 0
        actual, predicted = truth[use, k] > 0, scores[use, k] >= 0.5
        tp, fp = (predicted & actual).sum(), (predicted & ~actual).sum()
        fn, tn = (~predicted & actual).sum(), (~predicted & ~actual).sum()
        result[label] = {
            "auc": round(auc(scores[use, k], actual), 3),
            "recall": round(float(tp / max(1, tp + fn)), 3),
            "false_positive_rate": round(float(fp / max(1, fp + tn)), 3),
            "precision": round(float(tp / max(1, tp + fp)), 3),
            "lit_in_recordings_without_it": round(float(np.mean(absent[k])), 3)
            if absent[k]
            else None,
            "recordings_without_it": len(absent[k]),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--fold", help="i/n: hold out the artists of fold i, score them")
    parser.add_argument("--output")
    parser.add_argument("--name", default="guitar_timbre_v1")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--dilations", default="1,2,4,8")
    parser.add_argument("--taps", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--crop", type=int, default=500)
    parser.add_argument("--shift", type=int, default=2, help="Pitch-shift augmentation, semitones")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parts", help="Folder written by cache_timbre_parts.py")
    parser.add_argument(
        "--audible", action="store_true", help="Learn 'audible in the mix', not 'stem is on'"
    )
    parser.add_argument(
        "--synthetic", type=float, default=0.7, help="Share of crops mixed across recordings"
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    files, labels, groups = load(args.cache)
    parts = load_parts(args.parts, groups) if args.parts else []
    held, held_parts = [], []
    if args.fold:
        index, folds = (int(part) for part in args.fold.split("/"))
        held = [f for f in files if bucket(f["group"], folds) == index and not f["variant"]]
        files = [f for f in files if bucket(f["group"], folds) != index]
        held_parts = [f for f in parts if bucket(f["group"], folds) == index]
        parts = [f for f in parts if bucket(f["group"], folds) != index]
    held_by_ear = relabel(held, held_parts, labels) if held_parts else []
    if args.audible:
        files = relabel(files, parts, labels)
    dilations = tuple(int(d) for d in args.dilations.split(","))
    stacked = np.concatenate([f["x"][::7] for f in files if not f["variant"]])
    mean, scale = stacked.mean(0), stacked.std(0) + 1e-3
    weights = np.concatenate([f["w"] for f in files])
    truth = np.concatenate([f["y"] for f in files])
    share = (truth * weights).sum(0) / weights.sum(0)
    on, off = torch.tensor(0.5 / share), torch.tensor(0.5 / (1 - share))
    net = Net(stacked.shape[1], len(labels), args.width, dilations, args.taps, args.dropout)
    optimiser = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=args.decay)
    lengths = np.array([len(f["x"]) for f in files], float)
    steps = int(lengths.sum() / args.crop / 32)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=2e-3, total_steps=steps * args.epochs
    )
    print(
        f"{len(files)} training files, {len(held)} held-out recordings, "
        f"{sum(p.numel() for p in net.parameters())} parameters, {steps} steps per epoch",
        flush=True,
    )
    for epoch in range(args.epochs):
        net.train()
        began, total = time.time(), 0.0
        for _ in range(steps):
            xs, ys, ws = [], [], []
            made = int(rng.binomial(32, args.synthetic)) if parts else 0
            for _ in range(made):
                x, y, w = synthetic(parts, labels, rng, args.crop, args.shift, args.audible)
                xs.append((x - mean) / scale)
                ys.append(y)
                ws.append(w)
            for pick in rng.choice(len(files), 32 - made, p=lengths / lengths.sum()):
                item = files[pick]
                start = rng.integers(0, max(1, len(item["x"]) - args.crop))
                x = item["x"][start : start + args.crop]
                shift = int(rng.integers(-args.shift, args.shift + 1))
                if shift:  # the band axis is in semitones: sliding it transposes the music
                    moved = np.full_like(x, FLOOR)
                    if shift > 0:
                        moved[:, shift:] = x[:, :-shift]
                    else:
                        moved[:, :shift] = x[:, -shift:]
                    x = moved
                pad = args.crop - len(x)
                xs.append(np.pad((x - mean) / scale, ((0, pad), (0, 0))))
                ys.append(np.pad(item["y"][start : start + args.crop], ((0, pad), (0, 0))))
                ws.append(np.pad(item["w"][start : start + args.crop], ((0, pad), (0, 0))))
            x, y, w = (torch.from_numpy(np.stack(v).astype(np.float32)) for v in (xs, ys, ws))
            logits = net(x)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
            weight = (w * (y * on + (1 - y) * off)).float()
            weight[:, : (args.taps - 1) * sum(dilations)] = 0  # no history yet in a crop
            loss = (loss * weight).sum() / weight.sum()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            total += float(loss)
        line = f"epoch {epoch} loss {total / steps:.4f} ({time.time() - began:.0f} s)"
        if held and (epoch == args.epochs - 1 or epoch % 2 == 1):
            scored = evaluate(net, held, mean, scale, labels)
            line += " | " + " ".join(f"{k} {v['auc']:.3f}" for k, v in scored.items())
        print(line, flush=True)
    if held:
        print("stem reference", json.dumps(evaluate(net, held, mean, scale, labels)))
    if held_by_ear:
        print("audible", json.dumps(evaluate(net, held_by_ear, mean, scale, labels)))
    if args.output:
        state = {k: v.detach().numpy() for k, v in net.state_dict().items()}
        arrays = {
            "feature_version": FEATURE_VERSION,
            "name": args.name,
            "labels": np.array(labels),
            "mean": mean,
            "scale": scale,
            "dilations": np.array(dilations),
            "head": state["head.weight"],
            "head_bias": state["head.bias"],
            "on": np.full(len(labels), 0.6),
            "off": np.full(len(labels), 0.4),
        }
        for index in range(len(dilations)):
            arrays[f"w{index}"] = state[f"convs.{index}.weight"]
            arrays[f"b{index}"] = state[f"convs.{index}.bias"]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, **arrays)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
