"""Continue the fine-tuning with excerpts a person labelled in ``gmv review``.

Runs in the experiments environment (torch, CPU). Starts from weights made by
``finetune.py`` and keeps training on the same MoisesDB mixes, with a share of the clips
taken from reviewed excerpts of ordinary recordings (``cache_review.py``):

* a clip ends at a random moment inside an excerpt and holds the 10 s before it, as the
  live recogniser would; columns inside the excerpt carry the reviewer's answer
  ("present throughout" -> guitar, "absent" -> no guitar), columns before it are not used;
* the guitar type is used only where the reviewer named one type; the other two types are
  then left unknown;
* some clips of either kind have everything before a random point silenced, which is what
  the recogniser sees in the first seconds of a recording.

``--hold k/n`` keeps the reviewed artists of fold k out of training and scores them, so the
gain on recordings is measured on artists the model was not shown. The MoisesDB fold given
by ``--fold`` must be the one the starting weights were trained without; it is scored too.

    python experiments/recognition/finetune_review.py --jobs JOBS.json --parts PARTS \
        --labels LABELS --source EfficientAT-src --init .model-cache/guitar_mn10_ft_fold0.pt \
        --fold 0/4 --review REVIEW_CACHE --hold 0/3 --weights W.pt --scores OUT
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import embed  # noqa: E402
import finetune as ft  # noqa: E402
from head import LABELS, fold_of, window_labels  # noqa: E402

RATE, CLIP, COLUMNS = ft.RATE, ft.CLIP, ft.COLUMNS
TYPES = {"distorted_electric": 1, "clean_electric": 2, "acoustic": 3}


def review_fold(group, folds):
    return int(hashlib.sha256(f"gmv-review-fold-1:{group}".encode()).hexdigest(), 16) % folds


def load_review(folder):
    """Reviewed excerpts: [{id, recording, group, category, presence, type, audio, ...}]."""
    items = []
    for path in sorted(Path(folder).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            audio, meta, group = data["audio"], data["meta"], str(data["group"])
        for clip, (eid, category, presence, kind, start, length) in zip(audio, meta, strict=True):
            items.append(
                {
                    "id": str(eid),
                    "recording": path.stem,
                    "group": group,
                    "category": str(category),
                    "presence": str(presence),
                    "type": str(kind),
                    "start": float(start),
                    "length": float(length),
                    "audio": clip,
                }
            )
    return items


def silence_before(audio, target, weight, rng):
    """Silence everything before a random point; those columns are guitar-free."""
    keep = int(rng.integers(1, 9)) * RATE  # 1 to 8 s of audio stay
    audio = audio.copy()
    audio[: CLIP - keep] = 0
    empty = (CLIP - keep) * COLUMNS // CLIP
    target[:, :empty], weight[:, :empty] = 0, 1
    return audio, target, weight


def review_clip(pools, rng):
    """One clip from a reviewed excerpt: positives and negatives equally often, recordings
    equally often within each."""
    pool = pools[int(rng.random() < 0.5)]
    recording = pool[rng.integers(len(pool))]
    item = recording[rng.integers(len(recording))]
    lead = CLIP  # the excerpt begins 10 s into the cached audio
    end = int(rng.integers(lead + RATE, len(item["audio"]) + 1))
    audio = item["audio"][end - CLIP : end].astype(np.float32)
    inside = (np.arange(COLUMNS) * (CLIP // COLUMNS) + (end - CLIP)) >= lead
    target = np.zeros((len(LABELS), COLUMNS), np.float32)
    weight = np.zeros_like(target)
    if item["presence"] == "absent":
        weight[:, inside] = 1
    else:
        target[0, inside], weight[0, inside] = 1, 1
        if item["type"] in TYPES:
            target[TYPES[item["type"]], inside], weight[TYPES[item["type"]], inside] = 1, 1
    scale = ft.gain(rng, -12, 0) / max(1.0, float(np.abs(audio).max()))
    return audio * scale, target, weight


def score_review(net, mel, items, hop, batch=24):
    """Per excerpt: scores at every hop inside it, each from the 10 s ending at that hop."""
    out = {}
    for item in items:
        audio = torch.from_numpy(item["audio"].astype(np.float32))
        step = round(hop * RATE)
        ends = CLIP + np.arange(1, (len(audio) - CLIP) // step + 1) * step
        rows = []
        for begin in range(0, len(ends), batch):
            chunk = torch.stack([audio[e - CLIP : e] for e in ends[begin : begin + batch]])
            with torch.inference_mode():
                rows.append(torch.sigmoid(net(mel(chunk).unsqueeze(1))[:, :, -1]).numpy())
        out[item["id"]] = np.concatenate(rows).astype(np.float32)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--parts", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--init", required=True, help="weights from finetune.py to start from")
    parser.add_argument("--fold", default="0/4", help="MoisesDB fold the init was trained without")
    parser.add_argument("--review", required=True, help="folder written by cache_review.py")
    parser.add_argument("--hold", default="-1/3", help="reviewed-artist fold kept out; -1: none")
    parser.add_argument("--review-share", type=float, default=0.4)
    parser.add_argument("--silence", type=float, default=0.15)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--rate", type=float, default=5e-5)
    parser.add_argument("--hop", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
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
    songs = ft.load_parts(args.parts, [j for j in jobs if fold_of(j["group"], folds) != fold])
    hold, holds = (int(v) for v in args.hold.split("/"))
    review = load_review(args.review)
    kept_out = [r for r in review if review_fold(r["group"], holds) == hold]
    usable = [
        r
        for r in review
        if review_fold(r["group"], holds) != hold
        and r["category"] == "core"
        and r["presence"] in ("present_throughout", "absent")
    ]
    pools = []
    for presence in ("absent", "present_throughout"):
        by_recording = {}
        for item in usable:
            if item["presence"] == presence:
                by_recording.setdefault(item["recording"], []).append(item)
        pools.append(list(by_recording.values()))
    print(
        f"{len(songs)} MoisesDB songs; reviewed: {len(usable)} excerpts for training "
        f"({len(pools[1])} recordings with guitar, {len(pools[0])} with guitar-free excerpts), "
        f"{len(kept_out)} excerpts kept out",
        flush=True,
    )

    body, mel = embed.load_model(Path(args.source).resolve(), "mn10_as")
    mel.freqm, mel.timem = torch.nn.Identity(), torch.nn.Identity()
    net = ft.Net(body.features)
    saved = torch.load(args.init, map_location="cpu")
    net.load_state_dict(saved["state"])
    net.eval()
    out = Path(args.scores) if args.scores else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        if kept_out:
            np.savez(out / "review_before.npz", **score_review(net, mel, kept_out, args.hop))

    if args.steps:
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
            clips = []
            for _ in range(args.batch):
                if rng.random() < args.review_share:
                    clip = review_clip(pools, rng)
                else:
                    clip = ft.make_clip(songs, rng)
                if rng.random() < args.silence:
                    clip = silence_before(*clip, rng)
                clips.append(clip)
            audio = torch.from_numpy(np.stack([c[0] for c in clips]).astype(np.float32))
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
            if step % 50 == 0:
                minutes = (perf_counter() - started) / 60
                print(f"step {step} loss {running / 50:.4f} ({minutes:.1f} min)", flush=True)
                running = 0.0
        net.drop.eval()
    Path(args.weights).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "labels": LABELS, "model": "mn10_as"}, args.weights)
    if not out:
        return
    if kept_out:
        np.savez(out / "review_after.npz", **score_review(net, mel, kept_out, args.hop))
        print("scored the reviewed excerpts kept out", flush=True)
    for song in ft.load_parts(args.parts, held):
        parts = song["parts"]
        guitar = [parts[n].astype(np.float32) for n in ft.GUITAR_PARTS if n in parts]
        rest = sum(parts[n].astype(np.float32) for n in ("rest", "keys") if n in parts)
        versions = {"": rest + sum(guitar) if guitar else rest}
        if guitar and not isinstance(rest, int):
            versions[".noguitar"] = rest
        for suffix, audio in versions.items():
            audio = audio / max(1.0, float(np.abs(audio).max()))
            times, scores = ft.causal_scores(net, mel, audio.astype(np.float32), args.hop)
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
    print("scored the MoisesDB fold", flush=True)


if __name__ == "__main__":
    main()
