"""Train the instrument-activity network from a local corpus with stem-derived references.

Two steps, both local: cache per-frame features and labels for the train and validation
splits, then fit a one-hidden-layer network on train and report on validation. The held-out
test split is never read. The weights file is derived from the audio of the dataset: keep it
out of the repository unless the licence of that dataset allows redistribution.

    python scripts/train_activity.py --corpus corpus-data/corpus/manifest.json \
        --dataset musdb18hq --cache E:/gmv-cache/activity --output .model-cache/activity_v1.npz
"""

import argparse
from pathlib import Path

import numpy as np

from gmv.activity import FEATURE_VERSION, FrameFeatures, add_context
from gmv.corpus import registry
from gmv.corpus.datasets import adapters, read_sum
from gmv.models import AnalysisConfig

from gmv.corpus.schema import HIERARCHY  # noqa: E402
from gmv.corpus.schema import LABELS as ALL_LABELS  # noqa: E402

LABELS = ("drums", "bass", "vocals")
CHILDREN = {child for children in HIERARCHY.values() for child in children}
SPLITS = ("train", "validation")


STEMS = ("drums", "bass", "vocals", "other")
# Remixes of the train stems. Removing a stem gives the same music without that instrument,
# which is the most direct counter-example for its detector. A bass-less remix is left out:
# on validation it lowered bass recall on real mixes (AUC 0.94 against 0.96 without it).
VARIANTS = {"no_vocals": ("vocals",), "no_drums": ("drums",), "random": ()}


def configure(labels):
    """Learn another set of labels. Remixes then use every stem a recording has, and drop
    one learned instrument at a time so the model sees the same music without it."""
    global LABELS, STEMS, VARIANTS
    unknown = set(labels) - set(ALL_LABELS)
    if unknown:
        raise ValueError(f"Unknown instrument labels: {sorted(unknown)}")
    LABELS, STEMS = tuple(labels), ()
    VARIANTS = {f"no_{label}": (label,) for label in labels if label not in CHILDREN}
    VARIANTS["random"] = ()


def frame_rows(audio, rate, config):
    size, hop = round(rate * config.window_ms / 1000), round(rate * config.hop_ms / 1000)
    window = np.hanning(size).astype(np.float32)
    features = FrameFeatures(rate, size, hop / rate)
    count = (len(audio) - size) // hop + 1
    rows = np.empty((count, features.count), np.float32)
    level = np.empty(count, np.float32)
    for begin in range(0, count, 512):
        index = np.arange(begin, min(count, begin + 512))
        frames = audio[index[:, None] * hop + np.arange(size)[None, :]]
        spectra = np.fft.rfft(frames * window[None, :, None], axis=1)
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=(1, 2)))
        level[index] = np.maximum(-120, 20 * np.log10(np.maximum(rms, 1e-6)))
        for k, i in enumerate(index):
            rows[i] = features.step(spectra[k])
    times = np.arange(count) * hop / rate + size / rate / 2
    return rows, level, times, hop / rate


def frame_labels(manifest_path, recording, times):
    """(active, known) per frame and label. Activity measured on a stem that contains bleed
    is uncertain: those frames are neither active nor inactive and are left out of training
    and scoring for that label."""
    labels = np.zeros((len(times), len(LABELS)), bool)
    unsure = np.zeros_like(labels)
    for annotation in registry.annotations_for(manifest_path, recording.recording_id):
        if not annotation.usable_as_reference:
            continue
        for span in annotation.activity:
            # A certain sub-label (a clean guitar track without bleed) also makes its
            # parent certain, even if another track of that instrument has bleed.
            names = [span.label]
            if span.certainty == "certain":
                names += [p for p, children in HIERARCHY.items() if span.label in children]
            for name in names:
                if name in LABELS:
                    inside = (times >= span.start) & (times < span.end)
                    target = labels if span.certainty == "certain" else unsure
                    target[inside, LABELS.index(name)] = True
    return labels, ~unsure | labels


def remix(stems, variant, seed):
    """Return (audio, labels still present) for one remix of the stems."""
    rng = np.random.default_rng(seed)
    names = sorted(stems)
    removed = set(VARIANTS[variant])
    gains = dict.fromkeys(names, 1.0)
    if variant == "random":
        gains = {name: 10 ** (rng.uniform(-9, 6) / 20) for name in names}
        removed = {name for name in names if rng.random() < 0.15}
        if len(removed) == len(names):
            removed.discard(names[-1])
    audio = sum(gains[name] * stems[name] for name in names if name not in removed)
    # A label is gone when its own stem is, or (for a sub-label) when its parent's stem is.
    gone = removed | {c for p in removed for c in HIERARCHY.get(p, ())}
    return audio.astype(np.float32), [name not in gone for name in LABELS]


def dataset_stems(track):
    """Disjoint instrument groups of a dataset track, each summed from all of its files.
    Needed where one instrument is spread over several files (MoisesDB)."""
    top = {label: files for label, files in track.stems.items() if label not in CHILDREN}
    return {label: read_sum(files) for label, files in top.items()}


def cache_features(
    manifest_path, dataset, cache, augment=True, progress=print, root=None, shard=(0, 1)
):
    manifest = registry.load_manifest(manifest_path)
    paths = registry.load_paths(manifest_path)
    config = AnalysisConfig()
    cache.mkdir(parents=True, exist_ok=True)
    tracks = {t.track_id: t for t in adapters()[dataset].discover(Path(root))} if root else {}
    chosen = [r for r in manifest.recordings if r.dataset == dataset and r.split in SPLITS]
    for recording in chosen[shard[0] :: shard[1]]:
        target = cache / f"{recording.split}-{recording.recording_id}.npz"
        if not target.exists():
            path, reason = registry.resolve(recording, paths, verify_hash=False)
            if not path:
                progress(f"skipped {recording.recording_id}: {reason}")
                continue
            audio, rate = registry.read_audio(path)
            rows, level, times, hop = frame_rows(audio, rate, config)
            labels, known = frame_labels(manifest_path, recording, times)
            np.savez(target, x=rows, y=labels, k=known, db=level, hop=hop)
            progress(f"cached {recording.split} {recording.recording_id}: {len(rows)} frames")
        todo = [
            v for v in VARIANTS if not (cache / f"train-{recording.recording_id}.{v}.npz").exists()
        ]
        # Only train tracks are remixed; validation stays the real mixes.
        if not augment or recording.split != "train" or not todo:
            continue
        if recording.dataset_track_id in tracks:
            stems = dataset_stems(tracks[recording.dataset_track_id])
            width = max(audio.shape[1] for audio, _ in stems.values())
            stems = {
                name: (np.repeat(audio, width, axis=1) if audio.shape[1] < width else audio, rate)
                for name, (audio, rate) in stems.items()
            }
        else:
            found, problems = registry.resolve_stems(recording, paths, verify_hash=False)
            wanted = STEMS if STEMS else tuple(found)
            if problems or not set(wanted) <= set(found):
                progress(f"no remixes for {recording.recording_id}: stems incomplete")
                continue
            stems = {name: registry.read_audio(found[name]) for name in wanted}
        rate = next(iter(stems.values()))[1]
        length = min(len(audio) for audio, _ in stems.values())
        stems = {name: audio[:length] for name, (audio, _) in stems.items()}
        for variant in todo:
            seed = int(recording.sha256[:8], 16)
            audio, present = remix(stems, variant, seed)
            rows, level, times, hop = frame_rows(audio, rate, config)
            labels, known = frame_labels(manifest_path, recording, times)
            # A removed instrument is known to be absent, whatever its stem contained.
            labels, known = labels & np.array(present), known | ~np.array(present)
            name = f"train-{recording.recording_id}.{variant}.npz"
            np.savez(cache / name, x=rows, y=labels, k=known, db=level, hop=hop)
        progress(f"remixed {recording.recording_id}: {len(todo)} variants")


def load(cache, split, silence_dbfs, fraction=1.0, seed=0):
    """Frames above the silence level. ``fraction`` keeps a random share of each file:
    neighbouring frames are nearly identical, and the remixes multiply the data."""
    rng = np.random.default_rng(seed)
    xs, ys, ks = [], [], []
    for file in sorted(cache.glob(f"{split}-*.npz")):
        data = np.load(file)
        keep = data["db"] >= silence_dbfs
        if fraction < 1:
            keep &= rng.random(len(keep)) < fraction
        xs.append(add_context(data["x"], float(data["hop"]))[keep].astype(np.float32))
        ys.append(data["y"][keep][:, : len(LABELS)])
        ks.append(data["k"][keep] if "k" in data else np.ones_like(ys[-1]))
    known = np.concatenate(ks).astype(np.float32)
    return np.concatenate(xs), np.concatenate(ys).astype(np.float32), known


def auc(score, truth):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    positive = truth.sum()
    negative = len(truth) - positive
    return float((ranks[truth > 0].sum() - positive * (positive + 1) / 2) / (positive * negative))


def report(name, scores, truth, known, progress=print):
    for k, label in enumerate(LABELS):
        use = known[:, k] > 0
        predicted, actual = scores[use, k] >= 0.5, truth[use, k] > 0
        if not actual.any() or actual.all():
            progress(f"{name} {label:7s} not scorable: reference is all one class")
            continue
        tp, fp = (predicted & actual).sum(), (predicted & ~actual).sum()
        fn, tn = (~predicted & actual).sum(), (~predicted & ~actual).sum()
        share = actual.mean()
        progress(
            f"{name} {label:7s} active {share:.2f} | precision {tp / (tp + fp):.3f} "
            f"recall {tp / (tp + fn):.3f} F1 {2 * tp / (2 * tp + fp + fn):.3f} "
            f"(always-active F1 {2 * share / (1 + share):.3f}) | "
            f"false-positive rate {fp / (fp + tn):.3f} AUC {auc(scores[use, k], actual):.3f}"
        )


def train(
    cache,
    output,
    hidden=128,
    epochs=4,
    decay=1e-2,
    seed=0,
    fraction=0.3,
    name="activity_mlp_v2",
    progress=print,
):
    silence = AnalysisConfig().silence_dbfs
    x, y, known = load(cache, "train", silence, fraction, seed)
    vx, vy, vknown = load(cache, "validation", silence)
    progress(f"train frames {len(x)}, validation frames {len(vx)}")
    mean, scale = x.mean(0), x.std(0) + 1e-6
    x, vx = (x - mean) / scale, (vx - mean) / scale
    rng = np.random.default_rng(seed)
    params = [
        (rng.standard_normal((x.shape[1], hidden)) * np.sqrt(2 / x.shape[1])).astype(np.float32),
        np.zeros(hidden, np.float32),
        (rng.standard_normal((hidden, len(LABELS))) * np.sqrt(1 / hidden)).astype(np.float32),
        np.zeros(len(LABELS), np.float32),
    ]
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    share = (y * known).sum(0) / known.sum(0)
    # Active and inactive frames carry equal total weight, so "always active" is not a solution.
    weight_on, weight_off = 0.5 / share, 0.5 / (1 - share)

    def forward(batch):
        hidden_out = np.maximum(batch @ params[0] + params[1], 0)
        return hidden_out, 1 / (1 + np.exp(-(hidden_out @ params[2] + params[3])))

    step, rate = 0, 1e-3
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        for begin in range(0, len(order), 1024):
            index = order[begin : begin + 1024]
            batch, target = x[index], y[index]
            hidden_out, scores = forward(batch)
            weight = (target * weight_on + (1 - target) * weight_off) * known[index]
            error = (scores - target) * weight / len(index)
            back = (error @ params[2].T) * (hidden_out > 0)
            gradients = [
                batch.T @ back + decay * params[0],
                back.sum(0),
                hidden_out.T @ error + decay * params[2],
                error.sum(0),
            ]
            step += 1
            for k, gradient in enumerate(gradients):
                first[k] = 0.9 * first[k] + 0.1 * gradient
                second[k] = 0.999 * second[k] + 0.001 * gradient**2
                momentum = first[k] / (1 - 0.9**step)
                update = momentum / (np.sqrt(second[k] / (1 - 0.999**step)) + 1e-8)
                params[k] -= (rate * update).astype(np.float32)
        if epoch == epochs - 3:
            rate = 3e-4
        report(f"epoch {epoch} validation", forward(vx)[1], vy, vknown, progress)
    report("train", forward(x)[1], y, known, progress)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        feature_version=FEATURE_VERSION,
        name=name,
        labels=np.array(LABELS),
        mean=mean,
        scale=scale,
        w1=params[0],
        b1=params[1],
        w2=params[2],
        b2=params[3],
        on=np.full(len(LABELS), 0.6),
        off=np.full(len(LABELS), 0.4),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--labels",
        help="Comma-separated instrument labels to learn (default drums,bass,vocals). "
        "For guitar on MoisesDB: guitar,guitar_distorted,guitar_clean,guitar_acoustic",
    )
    parser.add_argument("--name", default="activity_mlp_v2", help="Name stored in the weights")
    parser.add_argument(
        "--dataset-root",
        help="Dataset folder: remixes are built from the dataset files themselves. Needed "
        "where an instrument is spread over several files (MoisesDB)",
    )
    parser.add_argument("--shard", default="0/1", help="i/n: cache only every n-th recording")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--decay", type=float, default=1e-2)
    args = parser.parse_args()
    if args.labels:
        configure(tuple(label.strip() for label in args.labels.split(",")))
    shard = tuple(int(part) for part in args.shard.split("/"))
    cache_features(Path(args.corpus), args.dataset, args.cache, root=args.dataset_root, shard=shard)
    if not args.cache_only:
        train(
            args.cache,
            args.output,
            hidden=args.hidden,
            epochs=args.epochs,
            decay=args.decay,
            name=args.name,
        )


if __name__ == "__main__":
    main()
