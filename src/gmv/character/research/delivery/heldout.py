"""The pre-registered held-out test of clean against harsh vocals.

The plan, fixed before any held-out answer was read: docs/VOCAL_DELIVERY_HELDOUT_PLAN.md.
Two separate steps:

1. ``predict``: train the frozen model (the Phase A4 random forest) on the development
   excerpts, measure every held-out excerpt, and write each prediction to a private file.
   **No held-out answer is read.** The development reader decodes development lines only,
   and this step never calls the coarse reader below. The SHA-256 of the predictions file
   (the seal) is then committed and pushed, so the predictions are fixed in public before
   any answer is read.
2. ``score``: only when the owner authorises it. Checks that the predictions still match
   the committed seal, records that the held-out answers have been opened (before reading
   them, so a crash still counts), reads them **coarsely**, and scores. It refuses to run a
   second time.

The coarse reader turns each held-out answer into clean, harsh or other the moment it is
read, and keeps nothing finer. Which harsh answers are High or Low Screams, and which
others are Gutturals or Sing-Screams, is never stored, so these artists stay usable for a
later High-against-Low test.
"""

import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..labeling import SCHEME, SKIPS
from ..labeling import dataset as labelling
from . import SUPPORTED, data, features, models
from . import run as experiment

PREDICTIONS = "heldout-predictions.json"
MANIFEST = "heldout-manifest.json"
OPENED = "heldout-opened.json"
FROZEN_SEED = (20261125, 1000)
FROZEN_COUNTS = {"cleans": 170, "high_screams": 45, "low_screams": 51}  # Phase A4's training set
COARSE = {"cleans": "clean", "high_screams": "harsh", "low_screams": "harsh",
          "gutturals": "other", "sing_screams": "other"}

# The verdict and the interval, as fixed in the plan (and its Amendment 1).
MIN_EXCERPTS = 10
CONFIRMED_BA, CONFIRMED_RECALL, CONFIRMED_LOW = 0.80, 0.70, 0.60
NOT_CONFIRMED_BA, NOT_CONFIRMED_RECALL = 0.65, 0.50
BREADTH_EXCERPTS, BREADTH_BA = 5, 0.70
INTERVAL_SEED, VALID_ROUNDS, MAX_DRAWS = 20261125, 2000, 40000
MIN_GROUPS, MAX_REJECTED = 3, 0.25
STEPS = ("confirmed", "weaker than in development", "not confirmed")

LINE_EXCERPT = re.compile(r'"excerpt"\s*:\s*"([^"\\]+)"')
LINE_EVENT = re.compile(r'"event"\s*:\s*"([a-z_]+)"')
LINE_VALUE = re.compile(r'"value"\s*:\s*"([a-z_]+)"')
LINE_REASON = re.compile(r'"reason"\s*:\s*"([a-z_]+)"')
LINE_SCHEME = re.compile(r'"scheme"\s*:\s*(\d+)')


class SealError(Exception):
    """The predictions do not match their seal, or the step would overwrite them."""


class AlreadyOpened(Exception):
    """The held-out answers have already been opened once."""


class ConsistencyError(Exception):
    """The sealed predictions do not belong to the dataset supplied for scoring. Raised
    before anything is opened, so the one-time scoring stays available."""


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_fingerprint(examples):
    """SHA-256 of the training set (excerpt and answer, one per line, sorted). Private
    inputs, but the fingerprint reveals nothing about them."""
    text = "\n".join(sorted(f"{e['excerpt']}:{e['label']}" for e in examples))
    return hashlib.sha256(text.encode()).hexdigest()


def precaution(dataset, inventory):
    """Held-out recordings whose music-inventory leakage group is also the group of a
    development recording, or a development artist group itself (the plan's precaution)."""
    rows = {}
    with Path(inventory).open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            rows[row.get("sha256", "")] = row
    development_groups = {a for a, entry in dataset["artists"].items() if entry["split"] == "development"}
    development_leakage = set(development_groups)
    for recording in dataset["recordings"].values():
        if dataset["artists"][recording["artist"]]["split"] == "development":
            group = (rows.get(recording["sha256"]) or {}).get("leakage_group")
            if group:
                development_leakage.add(group)
    flagged = set()
    for identity, recording in dataset["recordings"].items():
        if dataset["artists"][recording["artist"]]["split"] != "evaluation":
            continue
        group = (rows.get(recording["sha256"]) or {}).get("leakage_group")
        if group and group in development_leakage:
            flagged.add(identity)
    return flagged


def train_frozen(X, y):
    """The frozen model: preprocessing, class weights and the forest, all from the
    development examples."""
    prepare = models.Preprocess().fit(X)
    weights = models.class_weights(y, len(SUPPORTED))
    model = models.Forest(list(FROZEN_SEED)).fit(prepare.transform(X), y, weights, len(SUPPORTED))
    return prepare, model


def predict(folder, out, inventory, note=None, processes=1, progress=None, say=print, code=None,
            expected_counts=FROZEN_COUNTS):
    """Step 1. Returns the manifest; writes the predictions and the manifest to ``out``."""
    out = Path(out)
    if (out / OPENED).exists():
        raise AlreadyOpened("the held-out answers have already been opened; predictions cannot be made again")
    if (out / PREDICTIONS).exists():
        raise SealError("held-out predictions already exist here and are never overwritten")
    found = data.load(folder, note=note, inventory=inventory)
    dataset = found["dataset"]
    counts = {k: sum(1 for e in found["examples"] if e["label"] == k) for k in SUPPORTED}
    if expected_counts is not None and counts != dict(expected_counts):
        raise data.DataError(f"the development training set is not Phase A4's ({counts}); the plan froze {dict(expected_counts)}")
    X, examples, dropped, _measured = experiment.extract(found, out, True, processes, progress)
    if dropped:
        raise data.DataError(f"development audio could not be measured ({dropped}); the frozen training set is incomplete")
    y = np.array([SUPPORTED.index(e["label"]) for e in examples], dtype=int)
    prepare, model = train_frozen(X, y)
    say(f"Frozen model trained on {len(examples)} development excerpts: "
        + ", ".join(f"{k} {v}" for k, v in counts.items()))

    side = data.sides(dataset)
    flagged = precaution(dataset, inventory)
    held = sorted((e for e in dataset["excerpts"] if side[e["id"]] == "evaluation"), key=lambda e: e["id"])
    jobs = [tuple(str(v) if isinstance(v, Path) else v for v in data.source(dataset, e["id"])) for e in held]
    measured = list(experiment._measure_all(jobs, processes, progress or (lambda n, t: None)))
    rows, unreadable = [], {}
    for excerpt, value in zip(held, measured, strict=True):
        row = {
            "excerpt": excerpt["id"],
            "recording": excerpt["recording"],
            "group": dataset["recordings"][excerpt["recording"]]["artist"],
            "precaution": excerpt["recording"] in flagged,
        }
        if isinstance(value, str):
            row.update(status=value, probabilities=None, said=None)
            unreadable[value] = unreadable.get(value, 0) + 1
        else:
            proba = model.predict_proba(prepare.transform(np.asarray(value, dtype=float)[None, :]))[0]
            row.update(status="measured", probabilities=[round(float(p), 6) for p in proba],
                       said="clean" if int(np.argmax(proba)) == 0 else "harsh")
        rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    (out / PREDICTIONS).write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    seal = sha256_of(out / PREDICTIONS)
    groups = {}
    for row in rows:
        groups[row["group"]] = groups.get(row["group"], 0) + 1
    largest = max(sorted(groups), key=lambda g: groups[g])
    manifest = {
        "plan": "docs/VOCAL_DELIVERY_HELDOUT_PLAN.md",
        "made_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code,
        "feature_version": features.FEATURE_VERSION,
        "model": models.SETTINGS["forest"] + f"; seed {list(FROZEN_SEED)}",
        "training": {"counts": counts, "fingerprint_sha256": training_fingerprint(examples)},
        "held_out": {
            "excerpts": len(rows),
            "measured": sum(1 for r in rows if r["status"] == "measured"),
            "unreadable": unreadable,
            "precaution_excerpts": sum(1 for r in rows if r["precaution"]),
            "artist_groups": len(groups),
            "largest_group_excerpts": groups[largest],
            "said_clean": sum(1 for r in rows if r["said"] == "clean"),
            "said_harsh": sum(1 for r in rows if r["said"] == "harsh"),
        },
        "largest_group": largest,  # private: the manifest stays local
        "predictions_sha256": seal,
        "held_out_answer_values_decoded": found["audit"]["held_out_values_decoded"],
        "answers_opened": False,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    say(f"Held-out excerpts predicted: {manifest['held_out']['measured']} of {len(rows)}; "
        f"held-out answer values decoded: {manifest['held_out_answer_values_decoded']}")
    say(f"Predictions SHA-256: {seal}")
    return manifest


# -- step 2 ---------------------------------------------------------------------------------

FIELDS = frozenset({"excerpt", "recording", "group", "precaution", "probabilities", "said", "status"})


def parse_predictions(text):
    """The sealed prediction records, checked field by field as ``predict`` writes them.
    Anything else raises ``ConsistencyError``."""
    try:
        rows = json.loads(text)
    except ValueError as exc:
        raise ConsistencyError("the predictions file cannot be parsed") from exc
    if not isinstance(rows, list) or not rows:
        raise ConsistencyError("the predictions file is not a list of prediction records")
    for row in rows:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ConsistencyError("a prediction record does not have exactly the fields predict writes")
        if not all(isinstance(row[k], str) and row[k] for k in ("excerpt", "recording", "group", "status")):
            raise ConsistencyError("a prediction record has an empty or non-text identifier or status")
        if not isinstance(row["precaution"], bool):
            raise ConsistencyError("a prediction record's precaution flag is not true or false")
        if row["status"] == "measured":
            values = row["probabilities"]
            if not (isinstance(values, list) and len(values) == len(SUPPORTED) and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values
            )):
                raise ConsistencyError("a measured prediction does not carry three finite probabilities")
            if row["said"] not in ("clean", "harsh"):
                raise ConsistencyError("a measured prediction is neither clean nor harsh")
        elif row["probabilities"] is not None or row["said"] is not None:
            raise ConsistencyError("an unmeasured excerpt carries a prediction")
    return rows


def expected_assignments(dataset, inventory):
    """excerpt -> {"recording", "group", "precaution"} for every evaluation excerpt of the
    supplied dataset, derived exactly as ``predict`` derives them: the excerpt's recording,
    that recording's artist key in the dataset, and the plan's precaution
    (``precaution(dataset, inventory)``)."""
    side = data.sides(dataset)
    flagged = precaution(dataset, inventory)
    return {
        e["id"]: {
            "recording": e["recording"],
            "group": dataset["recordings"][e["recording"]]["artist"],
            "precaution": e["recording"] in flagged,
        }
        for e in dataset["excerpts"] if side[e["id"]] == "evaluation"
    }


def validate_predictions(rows, dataset, inventory, manifest=None):
    """Check the sealed predictions against the dataset supplied for scoring, exactly: the
    same evaluation excerpts (none missing, none extra, none duplicated, none from the
    development side or unknown to the dataset), and for each its recording, its artist
    group and its precaution flag. With the manifest, also its count and the largest artist
    group that scoring uses for the breadth rule. Raises ``ConsistencyError`` listing every
    kind of problem (counts only, no identifiers). Reads no answer."""
    expected = expected_assignments(dataset, inventory)
    side = data.sides(dataset)
    ids = [row["excerpt"] for row in rows]
    problems = []
    duplicated = [k for k, n in Counter(ids).items() if n > 1]
    if duplicated:
        problems.append(f"{len(duplicated)} excerpt identifier(s) appear more than once")
    unknown = {i for i in ids if i not in side}
    if unknown:
        problems.append(f"{len(unknown)} prediction(s) name excerpts that are not in the dataset")
    development = {i for i in ids if side.get(i) == "development"}
    if development:
        problems.append(f"{len(development)} prediction(s) name development excerpts")
    missing = set(expected) - set(ids)
    if missing:
        problems.append(f"{len(missing)} evaluation excerpt(s) of the dataset have no prediction")
    wrong = Counter()
    for row in rows:
        want = expected.get(row["excerpt"])
        if want is not None:
            for key in ("recording", "group", "precaution"):
                if row[key] != want[key]:
                    wrong[key] += 1
    for key in ("recording", "group", "precaution"):
        if wrong[key]:
            problems.append(f"{wrong[key]} prediction(s) carry a {key} that differs from the dataset's")
    if manifest is not None:
        groups = Counter(row["group"] for row in rows)
        largest = max(sorted(groups), key=lambda g: groups[g])
        if manifest.get("largest_group") != largest:
            problems.append("the manifest's largest artist group does not match the predictions")
        if (manifest.get("held_out") or {}).get("excerpts") != len(rows):
            problems.append("the manifest's count of held-out excerpts does not match the predictions")
    if problems:
        raise ConsistencyError("the sealed predictions do not match the supplied dataset: " + "; ".join(problems))
    return {"excerpts": len(rows), "artist_groups": len({row["group"] for row in rows}),
            "precaution": sum(1 for row in rows if row["precaution"])}


def read_held_out_coarse(folder, side):
    """excerpt -> "clean", "harsh", "other", "no_vocals", "later" or "unanswered", for
    held-out excerpts only, with the labelling tool's own rules (only current-scheme
    answers count, a later answer replaces an earlier one, a skip counts only while there is
    no answer). Each answer is turned into its coarse class as soon as it is matched; the
    finer answer is never stored. Development lines are not read."""
    state = {e: {"answer": None, "skip": None} for e, s in side.items() if s == "evaluation"}
    path = Path(folder) / data.LABELS
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
        found = LINE_EXCERPT.search(line)
        if not found or found.group(1) not in state:
            continue
        current = state[found.group(1)]
        kind = LINE_EVENT.search(line)
        kind = kind.group(1) if kind else None
        if kind == "label":
            scheme = LINE_SCHEME.search(line)
            value = LINE_VALUE.search(line)
            coarse = COARSE.get(value.group(1)) if value else None
            if scheme and int(scheme.group(1)) == SCHEME and coarse:
                current["answer"], current["skip"] = coarse, None
        elif kind == "skip" and current["answer"] is None:
            reason = LINE_REASON.search(line)
            current["skip"] = reason.group(1) if reason and reason.group(1) in SKIPS else "later"
    return {e: s["answer"] or s["skip"] or "unanswered" for e, s in state.items()}


def two_class(truth, said):
    """Balanced accuracy and recalls of clean (0) against harsh (1); None where a class is
    absent. Computed for exactly two classes, never averaged over whatever is present."""
    truth, said = np.asarray(truth), np.asarray(said)
    clean = float((said[truth == 0] == 0).mean()) if (truth == 0).any() else None
    harsh = float((said[truth == 1] == 1).mean()) if (truth == 1).any() else None
    ba = (clean + harsh) / 2 if clean is not None and harsh is not None else None
    confusion = [[int(((truth == t) & (said == s)).sum()) for s in (0, 1)] for t in (0, 1)]
    precision_clean = confusion[0][0] / max(1, confusion[0][0] + confusion[1][0])
    precision_harsh = confusion[1][1] / max(1, confusion[1][1] + confusion[0][1])
    return {"balanced_accuracy": ba, "recall_clean": clean, "recall_harsh": harsh,
            "precision_clean": precision_clean, "precision_harsh": precision_harsh,
            "confusion": confusion, "clean": int((truth == 0).sum()), "harsh": int((truth == 1).sum())}


def guarded_interval(groups, truth, said, rounds=VALID_ROUNDS, seed=INTERVAL_SEED, max_draws=MAX_DRAWS,
                     min_groups=MIN_GROUPS, max_rejected=MAX_REJECTED):
    """95% interval of the two-class balanced accuracy, resampling artist groups. A draw
    that lacks clean or harsh excerpts is rejected and drawn again (Amendment 1). Returns the
    counts, and either the interval ("ok") or no interval at all ("inconclusive", with the
    reasons)."""
    groups, truth, said = np.asarray(groups), np.asarray(truth), np.asarray(said)
    names = sorted(set(groups.tolist()))
    rows = {g: np.flatnonzero(groups == g) for g in names}
    with_clean = sum(1 for g in names if (truth[rows[g]] == 0).any())
    with_harsh = sum(1 for g in names if (truth[rows[g]] == 1).any())
    rng = np.random.default_rng(seed)
    values, rejected, draws = [], 0, 0
    while names and len(values) < rounds and draws < max_draws:
        draws += 1
        picked = np.concatenate([rows[names[i]] for i in rng.integers(0, len(names), len(names))])
        t = truth[picked]
        if not ((t == 0).any() and (t == 1).any()):
            rejected += 1
            continue
        values.append(two_class(t, said[picked])["balanced_accuracy"])
    reasons = []
    if with_clean < min_groups:
        reasons.append(f"only {with_clean} artist group(s) contribute clean excerpts (at least {min_groups} needed)")
    if with_harsh < min_groups:
        reasons.append(f"only {with_harsh} artist group(s) contribute harsh excerpts (at least {min_groups} needed)")
    share = rejected / draws if draws else 1.0
    if share > max_rejected:
        reasons.append(f"{share:.0%} of draws lacked a class (at most {max_rejected:.0%} allowed)")
    if len(values) < rounds:
        reasons.append(f"only {len(values)} valid resamples in {draws} draws ({rounds} needed)")
    return {
        "status": "inconclusive" if reasons else "ok",
        "reasons": reasons,
        "interval": None if reasons else [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
        "draws": draws,
        "rejected": rejected,
        "valid": len(values),
        "groups_with_clean": with_clean,
        "groups_with_harsh": with_harsh,
        "seed": seed,
    }


def verdict(primary, interval, others):
    """The plan's verdict: (verdict, breadth note, reasons)."""
    reasons = []
    if primary["clean"] < MIN_EXCERPTS or primary["harsh"] < MIN_EXCERPTS:
        reasons.append(f"{primary['clean']} clean and {primary['harsh']} harsh scored excerpts (at least {MIN_EXCERPTS} of each needed)")
    if interval["status"] != "ok":
        reasons += interval["reasons"]
    if reasons:
        return "inconclusive", None, reasons
    ba = primary["balanced_accuracy"]
    worst = min(primary["recall_clean"], primary["recall_harsh"])
    if ba < NOT_CONFIRMED_BA or worst < NOT_CONFIRMED_RECALL:
        step = "not confirmed"
    elif ba >= CONFIRMED_BA and worst >= CONFIRMED_RECALL and interval["interval"][0] >= CONFIRMED_LOW:
        step = "confirmed"
    else:
        step = "weaker than in development"
    enough = others["clean"] >= BREADTH_EXCERPTS and others["harsh"] >= BREADTH_EXCERPTS
    broad = enough and others["balanced_accuracy"] is not None and others["balanced_accuracy"] >= BREADTH_BA
    if enough and not broad:
        step = STEPS[min(len(STEPS) - 1, STEPS.index(step) + 1)]
        reasons.append("lowered one step: the other artist groups reach a balanced accuracy below "
                       f"{BREADTH_BA} on their own")
    return step, (None if broad else "mainly on one artist"), reasons


def score(folder, out, expected_sha256, inventory, say=print):
    """Step 2: once only, with the owner's authorisation.

    Order: (1) not opened before; (2) the predictions and manifest exist; (3) the
    predictions match the committed seal and the manifest; (4) the supplied dataset loads
    and passes its checks; (5, 6) the sealed predictions match that dataset exactly
    (``validate_predictions``, with the plan's inventory for the precaution flags). Only
    then (7) is the opening recorded, (8) are the held-out answers read, coarsely, and (9)
    scored. Every failure before (7) leaves the answers unopened and the test available."""
    out = Path(out)
    if (out / OPENED).exists():
        raise AlreadyOpened("the held-out answers were already opened once; the test is not run again")
    predictions_path, manifest_path = out / PREDICTIONS, out / MANIFEST
    if not predictions_path.is_file() or not manifest_path.is_file():
        raise SealError("no sealed predictions here; run the predict step first")
    actual = sha256_of(predictions_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sealed = manifest["predictions_sha256"]
    except (ValueError, KeyError, TypeError) as exc:
        raise SealError("the manifest cannot be read") from exc
    if actual != expected_sha256 or actual != sealed:
        raise SealError("the predictions do not match their committed seal")
    try:
        dataset = labelling.load(folder)
    except labelling.DatasetError as exc:
        raise data.DataError(str(exc)) from exc
    data.check_split(dataset)
    if inventory is None or not Path(inventory).is_file():
        raise ConsistencyError("the music inventory the plan's precaution uses was not found")
    rows = parse_predictions(predictions_path.read_text(encoding="utf-8"))
    validate_predictions(rows, dataset, inventory, manifest)
    side = data.sides(dataset)
    (out / OPENED).write_text(json.dumps({
        "opened_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "predictions_sha256": actual,
    }, indent=1) + "\n", encoding="utf-8")
    coarse = read_held_out_coarse(folder, side)
    for row in rows:
        row["answer"] = coarse.get(row["excerpt"], "unanswered")

    def scored(row):
        return row["status"] == "measured" and row["answer"] in ("clean", "harsh")

    def pack(chosen):
        truth = np.array([int(r["answer"] == "harsh") for r in chosen], dtype=int)
        said = np.array([int(r["said"] == "harsh") for r in chosen], dtype=int)
        return truth, said, [r["group"] for r in chosen]

    primary_rows = [r for r in rows if scored(r) and not r["precaution"]]
    truth, said, groups = pack(primary_rows)
    primary = two_class(truth, said)
    interval = guarded_interval(groups, truth, said)
    largest = manifest["largest_group"]
    others_rows = [r for r in primary_rows if r["group"] != largest]
    others = two_class(*pack(others_rows)[:2])
    step, breadth, reasons = verdict(primary, interval, others)

    sizes = {}
    for row in primary_rows:
        sizes.setdefault(row["group"], []).append(row)
    numbered = sorted(sizes, key=lambda g: (-len(sizes[g]), g))
    per_group = []
    for number, group in enumerate(numbered, 1):
        chosen = sizes[group]
        per_group.append({
            "group": number,
            "clean": sum(1 for r in chosen if r["answer"] == "clean"),
            "harsh": sum(1 for r in chosen if r["answer"] == "harsh"),
            "right": sum(1 for r in chosen if r["answer"] == r["said"]),
            "largest": group == largest,
        })
    recordings = {}
    for row in primary_rows:
        recordings.setdefault(row["recording"], []).append(row)
    mixed = [r for rs in recordings.values() if {x["answer"] for x in rs} == {"clean", "harsh"} for r in rs]
    within = two_class(*pack(mixed)[:2]) if mixed else None

    def described(kind):
        chosen = [r for r in rows if r["answer"] == kind and r["status"] == "measured"]
        return {"excerpts": len(chosen), "said_clean": sum(1 for r in chosen if r["said"] == "clean"),
                "said_harsh": sum(1 for r in chosen if r["said"] == "harsh")}

    flagged = [r for r in rows if r["precaution"] and scored(r)]
    summary = experiment.clean({
        "plan": manifest["plan"],
        "predictions_sha256": actual,
        "verdict": step,
        "breadth": breadth,
        "verdict_reasons": reasons,
        "primary": primary,
        "interval_95": interval,
        "without_largest_group": others,
        "per_group": per_group,
        "within_recordings_with_both": {"recordings": len({r["recording"] for r in mixed}), **(within or {})},
        "other_answers": described("other"),
        "no_vocals": described("no_vocals"),
        "review_later": described("later"),
        "unanswered": described("unanswered"),
        "precaution": {"excerpts": len(flagged), "right": sum(1 for r in flagged if r["answer"] == r["said"])},
        "unreadable_audio": manifest["held_out"]["unreadable"],
        "fine_answers_stored": False,
    })
    public = json.dumps(summary, indent=1)
    experiment.check_public(public, {"dataset": dataset, "folder": str(folder)})
    (out / "heldout-summary.json").write_text(public + "\n", encoding="utf-8")
    (out / "heldout-results.json").write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    say(f"Verdict: {step}" + (f" ({breadth})" if breadth else ""))
    return summary
