"""What a training run on the owner's recordings may read, and how a batch is drawn.

No torch here, so the rules can be tested without the experiments environment. The cached
segments come from ``cache_review.py``, which reads the table written by
``gmv evidence trainset``.

Arms. ``H`` uses segments a person confirmed. ``H+A`` uses the same segments plus the
automatic "guitar present" candidates of the two agreement classes. Nothing else differs
between them: see ``draw_batch`` for how the random streams are kept apart so that both arms
see the same MoisesDB clips, the same guitar-free clips, the same silenced lead-ins and the
same gains, and differ only in which guitar clip of the owner's recordings is drawn.

Evaluation. Only human answers are scored against, and only by a model that was trained
without their fold (``evaluation_items``, ``isolation_problems``).
"""

import hashlib
import json
from pathlib import Path

import numpy as np

RATE, CLIP, COLUMNS = 32000, 320000, 32
LABELS = ("guitar", "guitar_distorted", "guitar_clean", "guitar_acoustic")
TYPES = {"distorted_electric": 1, "clean_electric": 2, "acoustic": 3}
AUTOMATIC = ("automatic:strong_agreement", "automatic:moderate_agreement")
ARMS = {"H": ("human",), "H+A": ("human", *AUTOMATIC)}
WEIGHTS = {"human": 1.0, "automatic:strong_agreement": 0.5, "automatic:moderate_agreement": 0.25}
BASE_SEED = 20261122
STREAMS = ("plan", "moisesdb", "review_absent", "review_present", "augment", "torch")
META = ("id", "label", "type", "source", "weight", "start", "length", "pool", "selection")


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_of(values):
    """One hash for a set of ids, independent of order."""
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def load_segments(folder, audio=True):
    """Cached segments: [{id, recording, group, fold, label, type, source, weight, start,
    length, pool, selection, audio}]. ``audio`` holds the 10 s before the segment and the
    segment itself, mono at the model rate."""
    items = []
    for path in sorted(Path(folder).glob("r-*.npz")):
        with np.load(path, allow_pickle=False) as data:
            group, fold = str(data["group"]), int(data["fold"])
            same_audio = str(data["same_audio"])
            for index, row in enumerate(data["meta"]):
                item = dict(zip(META, (str(v) for v in row), strict=True))
                item.update(
                    recording=path.stem,
                    group=group,
                    fold=fold,
                    same_audio=same_audio,
                    weight=float(item["weight"]),
                    start=float(item["start"]),
                    length=float(item["length"]),
                    audio=data[f"audio_{index}"] if audio else None,
                )
                items.append(item)
    return items


def training_items(items, arm, hold):
    """Segments a run of ``arm`` may train on when fold ``hold`` is kept out."""
    allowed = ARMS[arm]
    chosen = [i for i in items if i["fold"] != hold and i["source"] in allowed]
    for item in chosen:
        if item["label"] not in ("present", "absent"):
            raise ValueError(f"{item['id']}: not a usable label ({item['label']})")
        if item["source"] != "human" and item["label"] != "present":
            raise ValueError(f"{item['id']}: an automatic segment may only be 'present'")
        if item["weight"] != WEIGHTS[item["source"]]:
            raise ValueError(f"{item['id']}: weight {item['weight']} does not match its source")
    return chosen


def evaluation_items(items, hold=None):
    """Human answers that are scored: of one fold, or of every fold for the baseline."""
    return [
        i
        for i in items
        if i["source"] == "human"
        and i["pool"] in ("primary", "secondary")
        and i["label"] in ("present", "absent")
        and (hold is None or i["fold"] == hold)
    ]


def isolation_problems(train, held):
    """Anything a training set shares with the excerpts it will be scored on."""
    problems = []
    for key, name in (("recording", "recording"), ("group", "leakage group"), ("fold", "fold")):
        shared = {i[key] for i in train} & {i[key] for i in held}
        if shared:
            problems.append(f"{name} on both sides: {sorted(map(str, shared))[:3]}")
    same = ({i["same_audio"] for i in train} & {i["same_audio"] for i in held}) - {""}
    if same:
        problems.append(f"same audio on both sides: {sorted(same)[:3]}")
    if any(i["source"] != "human" for i in held):
        problems.append("an automatic segment is among the scored excerpts")
    return problems


def seeds(fold):
    """One seed per random stream of a fold. The arm is deliberately not part of it."""
    return {
        name: int(np.random.SeedSequence([BASE_SEED, fold, index]).generate_state(1)[0])
        for index, name in enumerate(STREAMS)
    }


def generators(fold):
    return {name: np.random.default_rng(seed) for name, seed in seeds(fold).items() if name != "torch"}


def pools(items):
    """{label: [segments of one group, ...]}; groups in name order, segments in id order."""
    out = {}
    for label in ("absent", "present"):
        by_group = {}
        for item in sorted(items, key=lambda i: i["id"]):
            if item["label"] == label:
                by_group.setdefault(item["group"], []).append(item)
        out[label] = [by_group[g] for g in sorted(by_group)]
    return out


def plan(steps, batch, review_share, silence, rng):
    """Decided before training and shared by both arms: for every clip of every batch,
    whether it comes from the owner's recordings, whether that clip is guitar-free, and
    whether its lead-in is silenced."""
    return {
        "review": rng.random((steps, batch)) < review_share,
        "absent": rng.random((steps, batch)) < 0.5,
        "silenced": rng.random((steps, batch)) < silence,
    }


def gain(rng, low, high):
    return 10 ** (rng.uniform(low, high) / 20)


def review_clip(pool, rng, augment):
    """One clip from the owner's recordings: groups equally often, then a segment of the
    group, then an end point inside the segment. Columns inside the segment carry the label
    with the segment's weight; columns before it are not used.

    Returns (audio, target, weight, segment)."""
    group = pool[int(rng.integers(len(pool)))]
    item = group[int(rng.integers(len(group)))]
    lead = CLIP  # the segment begins 10 s into the cached audio
    end = int(rng.integers(lead + RATE, len(item["audio"]) + 1))
    audio = item["audio"][end - CLIP : end].astype(np.float32)
    inside = (np.arange(COLUMNS) * (CLIP // COLUMNS) + (end - CLIP)) >= lead
    target = np.zeros((len(LABELS), COLUMNS), np.float32)
    weight = np.zeros_like(target)
    if item["label"] == "absent":
        weight[:, inside] = item["weight"]
    else:
        target[0, inside], weight[0, inside] = 1, item["weight"]
        if item["source"] == "human" and item["type"] in TYPES:
            row = TYPES[item["type"]]
            target[row, inside], weight[row, inside] = 1, item["weight"]
    scale = gain(augment, -12, 0) / max(1.0, float(np.abs(audio).max()))
    return audio * scale, target, weight, item


def silence_before(audio, target, weight, augment):
    """Silence everything before a random point; those columns are guitar-free for certain."""
    keep = int(augment.integers(1, 9)) * RATE  # 1 to 8 s of audio stay
    audio = audio.copy()
    audio[: CLIP - keep] = 0
    empty = (CLIP - keep) * COLUMNS // CLIP
    target[:, :empty], weight[:, :empty] = 0, 1
    return audio, target, weight


def draw_batch(step, planned, pool, rngs, make_clip):
    """The clips of one step: [(audio, target, weight)] and what each one was.

    ``make_clip(rng)`` makes a MoisesDB clip. Each kind of decision has its own stream:
    MoisesDB clips, guitar-free clips, gains and silenced lead-ins consume theirs in the
    same order whichever arm is running, so they come out identical in both."""
    clips, kinds = [], []
    for slot in range(planned["review"].shape[1]):
        if planned["review"][step, slot]:
            label = "absent" if planned["absent"][step, slot] else "present"
            *clip, item = review_clip(pool[label], rngs[f"review_{label}"], rngs["augment"])
            kinds.append(item["source"] + ":" + label)
        else:
            clip = list(make_clip(rngs["moisesdb"]))
            kinds.append("moisesdb")
        if planned["silenced"][step, slot]:
            clip = list(silence_before(*clip, rngs["augment"]))
        clips.append(tuple(clip))
    return clips, kinds


def weighted_bce(logits, target, weight):
    """Reference for the training loss: mean over every label and column of
    weight x binary cross-entropy. A column with weight 0 contributes nothing."""
    logits, target, weight = (np.asarray(v, np.float64) for v in (logits, target, weight))
    each = np.maximum(logits, 0) - logits * target + np.log1p(np.exp(-np.abs(logits)))
    return float((each * weight).mean())


def composition(items):
    out = {}
    for item in items:
        key = f"{item['source']} / {item['label']}"
        entry = out.setdefault(key, {"segments": 0, "seconds": 0.0, "groups": set(), "weight": item["weight"]})
        entry["segments"] += 1
        entry["seconds"] += item["length"]
        entry["groups"].add(item["group"])
    return {
        k: {**v, "seconds": round(v["seconds"], 1), "groups": len(v["groups"])}
        for k, v in sorted(out.items())
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1), encoding="utf-8")
