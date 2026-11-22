"""One run of the experiment on the owner's recordings: train an arm without one fold, or
score the starting model, and record everything needed to repeat and to check the run.

Runs in the experiments environment (torch, CPU). Reads the segment cache written by
``cache_review.py`` and keeps training on the MoisesDB mixes of ``finetune.py`` with a share
of the clips taken from the owner's recordings (``review_data.py`` has the rules).

``--arm baseline``  no training: the starting weights score every human-labelled excerpt
                    and the MoisesDB control fold. This is "before" for both arms.
``--arm H``         human-confirmed segments only.
``--arm H+A``       the same, plus automatic "guitar present" candidates at their weights.

``--hold k`` keeps fold k of the owner's recordings out of training; only its human-labelled
excerpts are scored afterwards. ``--fold`` is the MoisesDB artist fold kept out of the
training mix and scored as the control.

Seeds come from the fold alone (``review_data.seeds``), never from the arm, and each kind of
random decision has its own stream, so the two arms of a fold differ only in which guitar
clip of the owner's recordings is drawn.

Scoring follows the application: each answer is about the 10 s ending at a hop, and quiet
audio is raised to ordinary level first (``gmv.recognition.levelled``).

A training run needs ``--owner-approved``. ``--dry-run`` loads everything, draws a few
batches, checks the loss against a reference and scores a few excerpts, without an
optimiser, a gradient or a weights file.

    python experiments/recognition/finetune_review.py --jobs JOBS.json --parts PARTS \
        --labels LABELS --source EfficientAT-src --init .model-cache/guitar_mn10_ft_all.pt \
        --review CACHE --runs RUNS --arm H+A --hold 0 --owner-approved
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
import embed  # noqa: E402
import finetune as ft  # noqa: E402
import review_data as rd  # noqa: E402
from head import LABELS, fold_of, window_labels  # noqa: E402

from gmv.recognition import levelled  # noqa: E402

RATE, CLIP, COLUMNS = ft.RATE, ft.CLIP, ft.COLUMNS
assert (RATE, CLIP, COLUMNS, tuple(LABELS)) == (rd.RATE, rd.CLIP, rd.COLUMNS, rd.LABELS)
FOLDS = 5


def loss_of(logits, target, weight):
    """Weighted binary cross-entropy, mean over labels and columns. ``weight`` carries the
    segment's training weight; columns with weight 0 contribute nothing."""
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target, weight)


def score_ends(net, mel, padded, ends, batch=24):
    """Scores for the 10 s ending at each of ``ends`` (samples into ``padded``), levelled
    as the application levels them."""
    rows = []
    for begin in range(0, len(ends), batch):
        chunk = np.stack(
            [levelled(padded[e - CLIP : e].astype(np.float32)) for e in ends[begin : begin + batch]]
        )
        with torch.inference_mode():
            rows.append(torch.sigmoid(net(mel(torch.from_numpy(chunk)).unsqueeze(1))[:, :, -1]).numpy())
    return np.concatenate(rows).astype(np.float32)


def score_review(net, mel, items, hop):
    """Per excerpt: scores at every hop inside it."""
    out = {}
    step = round(hop * RATE)
    for item in items:
        audio = item["audio"]
        ends = CLIP + np.arange(1, (len(audio) - CLIP) // step + 1) * step
        out[item["id"]] = score_ends(net, mel, audio, ends)
    return out


def score_moisesdb(net, mel, songs, labels, hop, out):
    """The control fold: every song and its guitar-free twin."""
    step = round(hop * RATE)
    for song in songs:
        parts = song["parts"]
        guitar = [parts[n].astype(np.float32) for n in ft.GUITAR_PARTS if n in parts]
        rest = sum(parts[n].astype(np.float32) for n in ("rest", "keys") if n in parts)
        versions = {"": rest + sum(guitar) if guitar else rest}
        if guitar and not isinstance(rest, int):
            versions[".noguitar"] = rest
        for suffix, audio in versions.items():
            audio = (audio / max(1.0, float(np.abs(audio).max()))).astype(np.float32)
            ends = np.arange(step, len(audio) + 1, step)
            scores = score_ends(net, mel, np.concatenate([np.zeros(CLIP, np.float32), audio]), ends + CLIP)
            times = ends / RATE
            y, certain, _ = window_labels(Path(labels) / f"{song['id']}.npz", times)
            n = min(len(y), len(scores))
            if suffix:
                y, certain = np.zeros_like(y), np.ones_like(certain)
            np.savez(out / f"{song['id']}{suffix}.npz", t=times[:n], score=scores[:n], y=y[:n], certain=certain[:n])


def git_state():
    def run(*command):
        return subprocess.run(command, capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()

    return {"commit": run("git", "rev-parse", "HEAD"), "uncommitted_changes": bool(run("git", "status", "--porcelain"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--parts", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--init", required=True, help="starting weights; the same file for every run")
    parser.add_argument("--fold", default="0/4", help="MoisesDB artist fold kept out and scored")
    parser.add_argument("--review", required=True, help="folder written by cache_review.py")
    parser.add_argument("--runs", required=True, help="root folder of the experiment's runs")
    parser.add_argument("--arm", required=True, choices=["baseline", *rd.ARMS])
    parser.add_argument("--hold", type=int, default=-1, help="fold of the owner's recordings kept out")
    parser.add_argument("--review-share", type=float, default=0.4)
    parser.add_argument("--silence", type=float, default=0.15)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--rate", type=float, default=5e-5)
    parser.add_argument("--hop", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--weights-dir", default=".model-cache/v2-experiment")
    parser.add_argument("--owner-approved", action="store_true", help="the owner approved training")
    parser.add_argument("--dry-run", action="store_true", help="check everything; train nothing")
    parser.add_argument("--dry-batches", type=int, default=2)
    args = parser.parse_args()
    training = args.arm != "baseline"
    if training and args.hold not in range(FOLDS):
        raise SystemExit(f"--hold must be 0 to {FOLDS - 1} for a training arm")
    if training and not args.dry_run and not args.owner_approved:
        raise SystemExit("Training needs the owner's approval: pass --owner-approved once it is given.")
    name = "baseline" if not training else f"{args.arm}/hold{args.hold}"
    out = Path(args.runs) / ("dry-run" if args.dry_run else "") / name
    weights = Path(args.weights_dir) / f"{args.arm}_hold{args.hold}.pt"
    if not args.dry_run:
        earlier = out / "run.json"
        if earlier.is_file() and json.loads(earlier.read_text(encoding="utf-8"))["status"] == "complete":
            raise SystemExit(f"{out} already holds a complete run; move it away to repeat it")
        if training and weights.exists():
            raise SystemExit(f"{weights} exists; a run never overwrites weights")
    out.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    fold, folds = (int(v) for v in args.fold.split("/"))
    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    control = [j for j in jobs if fold_of(j["group"], folds) == fold]
    mixes = [j for j in jobs if fold_of(j["group"], folds) != fold]
    items = rd.load_segments(args.review)
    cache = json.loads((Path(args.review) / "manifest.json").read_text(encoding="utf-8"))
    if training:
        train = rd.training_items(items, args.arm, args.hold)
        held = rd.evaluation_items(items, args.hold)
        problems = rd.isolation_problems(train, held)
        if problems:
            raise SystemExit("Training and evaluation are not separate: " + "; ".join(problems))
        seeds = rd.seeds(args.hold)
    else:
        train, held, seeds = [], rd.evaluation_items(items), {}
    config = {
        "steps": args.steps if training else 0,
        "batch": args.batch,
        "learning_rate": args.rate,
        "head_learning_rate": args.rate * 10,
        "weight_decay": 1e-2,
        "schedule": "OneCycleLR, 10% warm-up",
        "review_share": args.review_share,
        "silenced_lead_in_share": args.silence,
        "sampling": "label 50/50, then leakage group uniformly, then segment uniformly",
        "loss": "binary cross-entropy x segment weight, mean over labels and columns",
        "hop_s": args.hop,
        "scoring": "10 s context ending at each hop, levelled as in the application",
        "threads": args.threads,
        "moisesdb_control_fold": args.fold,
    }
    manifest = {
        "arm": args.arm,
        "held_out_fold": args.hold if training else None,
        "status": "dry run" if args.dry_run else "started",
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
        "config": config,
        "seeds": seeds,
        "initial_weights": {"file": Path(args.init).name, "sha256": rd.sha256_of(args.init)},
        "dataset": {
            "cache_manifest": cache,
            "training_segments": rd.composition(train),
            "training_ids": sorted(i["id"] for i in train),
            "training_ids_sha256": rd.digest_of(i["id"] for i in train),
            "training_groups": sorted({i["group"] for i in train}),
            "scored_ids": sorted(i["id"] for i in held),
            "scored_ids_sha256": rd.digest_of(i["id"] for i in held),
            "scored_groups": sorted({i["group"] for i in held}),
            "moisesdb_training_songs": len(mixes),
            "moisesdb_control_songs": len(control),
            "moisesdb_jobs_sha256": rd.sha256_of(args.jobs),
        },
        "code": git_state(),
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0]},
    }
    rd.write_json(out / "run.json", manifest)
    print(f"{name}: {len(train)} training segments, {len(held)} scored excerpts", flush=True)
    print(json.dumps(manifest["dataset"]["training_segments"], indent=1), flush=True)

    body, mel = embed.load_model(Path(args.source).resolve(), "mn10_as")
    mel.freqm, mel.timem = torch.nn.Identity(), torch.nn.Identity()
    net = ft.Net(body.features)
    net.load_state_dict(torch.load(args.init, map_location="cpu")["state"])
    net.eval()

    if args.dry_run:
        report = {"scored_sample": {}}
        for item in held[:3]:
            report["scored_sample"][item["id"]] = score_review(net, mel, [item], args.hop)[item["id"]][:, 0].round(4).tolist()
        if training:
            songs = ft.load_parts(args.parts, mixes[:6])  # enough to draw from; not the full set
            rngs = rd.generators(args.hold)
            planned = rd.plan(args.steps, args.batch, args.review_share, args.silence, rngs["plan"])
            pool = rd.pools(train)
            kinds, checks = [], []
            for step in range(args.dry_batches):
                clips, what = rd.draw_batch(step, planned, pool, rngs, lambda rng: ft.make_clip(songs, rng))
                kinds += what
                audio = torch.from_numpy(np.stack([c[0] for c in clips]).astype(np.float32))
                target = np.stack([c[1] for c in clips])
                weight = np.stack([c[2] for c in clips])
                with torch.no_grad():
                    logits = net(mel(audio).unsqueeze(1))
                    loss = float(loss_of(logits, torch.from_numpy(target), torch.from_numpy(weight)))
                checks.append({"torch": loss, "reference": rd.weighted_bce(logits.numpy(), target, weight)})
            report["batch_kinds"] = {k: kinds.count(k) for k in sorted(set(kinds))}
            report["planned_review_share"] = float(planned["review"].mean())
            report["planned_silenced_share"] = float(planned["silenced"].mean())
            report["loss_matches_reference"] = all(abs(c["torch"] - c["reference"]) < 1e-4 for c in checks)
            report["loss_checks"] = checks
            report["pool_groups"] = {k: len(v) for k, v in pool.items()}
        rd.write_json(out / "dry_run.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "scored_sample"}, indent=1))
        print("dry run: nothing was trained and no weights were written")
        return

    started = perf_counter()
    if training:
        songs = ft.load_parts(args.parts, mixes)
        rngs = rd.generators(args.hold)
        torch.manual_seed(seeds["torch"])
        planned = rd.plan(args.steps, args.batch, args.review_share, args.silence, rngs["plan"])
        pool = rd.pools(train)
        drawn = {}
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
        running, losses = 0.0, []
        for step in range(args.steps):
            clips, kinds = rd.draw_batch(step, planned, pool, rngs, lambda rng: ft.make_clip(songs, rng))
            for kind in kinds:
                drawn[kind] = drawn.get(kind, 0) + 1
            audio = torch.from_numpy(np.stack([c[0] for c in clips]).astype(np.float32))
            target = torch.from_numpy(np.stack([c[1] for c in clips]))
            weight = torch.from_numpy(np.stack([c[2] for c in clips]))
            with torch.no_grad():
                spec = mel(audio).unsqueeze(1)
            loss = loss_of(net(spec), target, weight)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            running += float(loss)
            if (step + 1) % 50 == 0:
                minutes = (perf_counter() - started) / 60
                losses.append(round(running / 50, 4))
                print(f"step {step + 1} loss {running / 50:.4f} ({minutes:.1f} min)", flush=True)
                running = 0.0
        net.drop.eval()
        weights.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": net.state_dict(), "labels": LABELS, "model": "mn10_as"}, weights)
        manifest["weights"] = {"file": str(weights), "sha256": rd.sha256_of(weights)}
        manifest["clips_drawn"] = dict(sorted(drawn.items()))
        manifest["loss_per_50_steps"] = losses
    np.savez(out / "review.npz", **score_review(net, mel, held, args.hop))
    print("scored the human-labelled excerpts", flush=True)
    (out / "moisesdb").mkdir(exist_ok=True)
    score_moisesdb(net, mel, ft.load_parts(args.parts, control), args.labels, args.hop, out / "moisesdb")
    print("scored the MoisesDB control fold", flush=True)
    manifest["status"] = "complete"
    manifest["minutes"] = round((perf_counter() - started) / 60, 1)
    manifest["outputs"] = {"review_scores": "review.npz", "moisesdb_scores": "moisesdb/*.npz"}
    rd.write_json(out / "run.json", manifest)


if __name__ == "__main__":
    main()
