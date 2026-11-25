"""The whole experiment: read and check the data, measure every usable excerpt, validate
each model by leaving out one artist group at a time, and write the reports.

Everything written goes to the output folder (by default the ignored
``local-evidence/sonic-character/vocal-delivery-classifier/``):

* ``report.json``: everything, including artist groups and excerpt identifiers (private);
* ``predictions.csv``: every excerpt's out-of-fold probabilities (private);
* ``feature-cache.npz``: the measured features, keyed by excerpt and audio (private);
* ``summary.json`` and ``summary.md``: counts and metrics only. Checked before writing:
  no excerpt, recording, artist, title or path may appear in them.
"""

import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from . import NAMES, SUPPORTED, VERSION, data, evaluate, features, models

SEED = 20261125
CACHE = "feature-cache.npz"


class PrivacyError(AssertionError):
    """A public summary would have carried private information."""


def fingerprint(dataset, excerpt_id):
    """What a cached feature vector depends on: the audio's checksum, the excerpt's times
    and the feature version."""
    excerpt = next(e for e in dataset["excerpts"] if e["id"] == excerpt_id)
    recording = dataset["recordings"][excerpt["recording"]]
    text = f"{recording['sha256']}:{excerpt['start_s']:.6f}:{excerpt['end_s']:.6f}:{features.FEATURE_VERSION}"
    return hashlib.sha256(text.encode()).hexdigest()


def measure_one(job):
    """Features of one excerpt, or the reason it could not be measured (a short phrase,
    never a path)."""
    from ..labeling import dataset as labelling

    path, start, end, size = job
    path = Path(path)
    try:
        if not path.is_file() or path.stat().st_size != size:
            return "audio_unavailable"
        audio, rate = labelling.read_excerpt(path, start, end)
    except Exception:
        return "audio_unreadable"
    try:
        vector = features.measure(audio, rate)
    except Exception:
        return "measurement_failed"
    if np.isnan(vector).all():
        return "no_measurable_audio"
    return vector


def _measure_all(jobs, processes, progress):
    if processes > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context

        from ...diagnose import THREAD_LIMITS

        kept = {name: os.environ.get(name) for name in THREAD_LIMITS}
        os.environ.update(dict.fromkeys(THREAD_LIMITS, "1"))
        try:
            with ProcessPoolExecutor(min(processes, len(jobs)), mp_context=get_context("spawn")) as pool:
                for number, found in enumerate(pool.map(measure_one, jobs), 1):
                    progress(number, len(jobs))
                    yield found
        finally:
            for name, value in kept.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    else:
        for number, job in enumerate(jobs, 1):
            found = measure_one(job)
            progress(number, len(jobs))
            yield found


def extract(found, out, use_cache=True, processes=1, progress=None):
    """Feature matrix for the usable examples, the examples kept, and what was dropped."""
    progress = progress or (lambda number, total: None)
    dataset = found["dataset"]
    examples = found["examples"]
    keys = [fingerprint(dataset, e["excerpt"]) for e in examples]
    cached = {}
    cache = Path(out) / CACHE
    if use_cache and cache.is_file():
        with np.load(cache, allow_pickle=False) as stored:
            if str(stored["version"]) == features.FEATURE_VERSION and list(stored["names"]) == list(features.NAMES):
                cached = dict(zip(stored["keys"].tolist(), stored["values"], strict=True))
    todo = [i for i, k in enumerate(keys) if k not in cached]
    jobs = [tuple(str(v) if isinstance(v, Path) else v for v in data.source(dataset, examples[i]["excerpt"])) for i in todo]
    results = dict(zip(todo, _measure_all(jobs, processes, progress), strict=True))
    vectors, kept, dropped = [], [], {}
    for i, example in enumerate(examples):
        value = cached.get(keys[i]) if i not in results else results[i]
        if isinstance(value, str):
            dropped[value] = dropped.get(value, 0) + 1
            continue
        vectors.append(np.asarray(value, dtype=float))
        kept.append(example)
    new = {keys[i]: results[i] for i in todo if not isinstance(results[i], str)}
    if new or (use_cache and not cache.is_file()):
        merged = {**cached, **new}
        Path(out).mkdir(parents=True, exist_ok=True)
        np.savez(
            cache, version=features.FEATURE_VERSION, names=np.array(features.NAMES),
            keys=np.array(list(merged), dtype=str),
            values=np.array(list(merged.values()), dtype=float).reshape(len(merged), len(features.NAMES)),
        )
    X = np.array(vectors, dtype=float).reshape(len(vectors), len(features.NAMES))
    return X, kept, dropped, len(todo)


def clean(value):
    """JSON-safe: NaN becomes null, numpy numbers plain numbers, int keys strings."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def private_words(found):
    """Every string that must never appear in a public summary."""
    dataset = found["dataset"]
    words = set()
    for excerpt in dataset["excerpts"]:
        words |= {excerpt["id"], excerpt["recording"]}
    for identity, recording in dataset["recordings"].items():
        words |= {identity, recording["artist"], recording["sha256"], str(recording["path"]), Path(recording["path"]).name}
    for artist in dataset["artists"]:
        words |= {artist, artist.split("-", 1)[-1]}
    from ..labeling import titles as labelling_titles

    folder = found.get("folder")
    if folder:
        for entry in labelling_titles.load(folder).values():
            if entry.get("title") and len(entry["title"]) >= 4:
                words.add(entry["title"])
    return {w for w in words if w and len(w) >= 4}


def check_public(text, found):
    """Raise if the public text carries any private word (case-insensitive)."""
    lower = text.lower()
    leaked = sorted(
        w for w in private_words(found)
        if re.search(r"(?<![a-z0-9])" + re.escape(w.lower()) + r"(?![a-z0-9])", lower)
    )
    if leaked:
        raise PrivacyError(f"{len(leaked)} private value(s) would have been published")


def run(folder, out, note=None, inventory=None, use_cache=True, processes=1, progress=None, say=print):
    started = time.perf_counter()
    found = data.load(folder, note=note, inventory=inventory)
    found["folder"] = str(folder)
    counts = data.summary(found)
    say(f"Usable development examples: {len(found['examples'])} "
        + ", ".join(f"{NAMES[k]} {v['examples']}" for k, v in counts.items()))
    say("Left out: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in found["excluded"].items() if v))
    say(f"Held out (totals only): {found['held_out']['answered']} answered, {found['held_out']['no_vocals']} No vocals; "
        f"answer lines decoded: {found['audit']['held_out_values_decoded']}")

    measuring = time.perf_counter()
    X, examples, dropped, measured = extract(found, out, use_cache, processes, progress)
    feature_s = time.perf_counter() - measuring
    y = np.array([SUPPORTED.index(e["label"]) for e in examples], dtype=int)
    say(f"Measured {measured} excerpts in {feature_s:.1f} s ({len(examples) - measured} from the cache); dropped: {dropped or 'none'}")

    results, private_results, predictions = {}, {}, {}
    for name in models.MODELS:
        fitting = time.perf_counter()
        fitted = []
        proba, facts = evaluate.cross_validate(X, y, examples, name, SEED, fitted=fitted)
        prediction = evaluate.predict(proba)
        per = evaluate.per_group(examples, y, prediction)
        result = {
            "settings": models.SETTINGS[name],
            "metrics": evaluate.metrics(y, prediction),
            "interval_95": evaluate.bootstrap(examples, y, prediction, SEED),
            "consistency": evaluate.consistency(per),
            "collapsed": evaluate.collapsed(y, prediction),
            "within": evaluate.within(examples, y, prediction),
            "abstention": evaluate.abstention(y, proba),
            "folds": len(facts),
            "all_converged": all(f["converged"] for f in facts),
            "seconds": time.perf_counter() - fitting,
        }
        if name == "logistic":
            columns = len(features.NAMES)
            weights = np.array([np.abs(m.W[:columns]).sum(axis=1) for _g, _p, m, _w in fitted])
            order = np.argsort(-weights.mean(axis=0))
            result["strongest_features"] = [
                {"feature": features.NAMES[i], "mean_abs_weight": float(weights[:, i].mean())} for i in order[:10]
            ]
        results[name] = result
        private_results[name] = {"per_group": per, "folds": facts}
        predictions[name] = proba
        m = result["metrics"]
        say(f"{name:9s} balanced accuracy {m['balanced_accuracy']:.3f}  macro F1 {m['macro_f1']:.3f}  "
            f"recall {', '.join(f'{r:.2f}' for r in m['recall'])}  ({result['seconds']:.1f} s)")

    summary = {
        "version": VERSION,
        "feature_version": features.FEATURE_VERSION,
        "seed": SEED,
        "answers": [NAMES[k] for k in SUPPORTED],
        "data": {
            "per_answer": {NAMES[k]: v for k, v in counts.items()},
            "usable_after_audio": len(examples),
            "dropped_for_audio": dropped,
            "excluded": found["excluded"],
            "held_out_totals": found["held_out"],
            "audit": found["audit"],
            "artist_groups": len({e["group"] for e in examples}),
            "unknown_feature_values": {
                name: int(np.isnan(X[:, i]).sum()) for i, name in enumerate(features.NAMES) if np.isnan(X[:, i]).any()
            },
            "development_credit_links": len(found["private"]["credit_links_development"]),
            "credit_links_across_sides": len(found["private"]["credit_links_across_sides"]),
            "inventory_checked": found["private"]["inventory_checked"],
        },
        "features": [{"name": n, "family": f, "what": w} for n, f, w in features.FEATURES],
        "models": results,
        "timing": {"feature_seconds": feature_s, "excerpts_measured": measured, "total_seconds": time.perf_counter() - started},
    }
    summary = clean(summary)
    public_text = json.dumps(summary, indent=1, ensure_ascii=False)
    markdown = render(summary)
    check_public(public_text + markdown, found)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(public_text, encoding="utf-8")
    (out / "summary.md").write_text(markdown, encoding="utf-8")
    report = {
        "summary": summary,
        "private": clean({**found["private"], "models": private_results}),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["excerpt", "group", "answer"] + [f"{m}:{k}" for m in models.MODELS for k in SUPPORTED])
        for i, example in enumerate(examples):
            writer.writerow([example["excerpt"], example["group"], example["label"]]
                            + [f"{predictions[m][i, c]:.6f}" for m in models.MODELS for c in range(len(SUPPORTED))])
    say(f"Done in {summary['timing']['total_seconds']:.1f} s. Reports in {out}")
    return summary


def number(value, digits=3):
    """A metric for the summary; "n/a" when it could not be computed (no examples)."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def render(summary):
    """The public summary as Markdown: counts and metrics only."""
    names = summary["answers"]
    lines = [f"# Vocal delivery recognition ({summary['version']}): development, leave one artist group out", ""]
    d = summary["data"]
    lines += ["| Answer | Examples | Artists | Recordings |", "|---|---:|---:|---:|"]
    for name in names:
        row = d["per_answer"][name]
        lines.append(f"| {name} | {row['examples']} | {row['artists']} | {row['recordings']} |")
    unknown = d.get("unknown_feature_values") or {}
    lines += ["", f"Artist groups: {d['artist_groups']}. Excluded: "
              + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in d["excluded"].items() if v) + ". "
              + ("Unknown feature values: " + ", ".join(f"{k} {v}" for k, v in unknown.items()) if unknown
                 else "No feature value was unknown.") + "", ""]
    lines += ["| Model | Balanced accuracy (95% interval) | Macro F1 (95% interval) | "
              + " | ".join(f"{n} recall" for n in names) + " |", "|---|---|---|" + "---:|" * len(names)]
    for model, result in summary["models"].items():
        m, ci = result["metrics"], result["interval_95"]
        lines.append(
            f"| {model} | {number(m['balanced_accuracy'])} ({number(ci['balanced_accuracy'][0])} to {number(ci['balanced_accuracy'][1])}) "
            f"| {number(m['macro_f1'])} ({number(ci['macro_f1'][0])} to {number(ci['macro_f1'][1])}) | "
            + " | ".join(number(r, 2) for r in m["recall"]) + " |"
        )
    lines.append("")
    for model, result in summary["models"].items():
        if model == "majority":
            continue
        c, w = result["collapsed"], result["within"]
        lines.append(
            f"{model}: clean against harsh, balanced accuracy {number(c['clean_vs_harsh']['balanced_accuracy'])}; "
            f"within the {w['clean_vs_harsh_within_recordings']['units']} recordings that have both, "
            f"{number(w['clean_vs_harsh_within_recordings']['balanced_accuracy'])}; high against low among harsh "
            f"predicted harsh, {number(c['high_vs_low_among_harsh']['balanced_accuracy'])}."
        )
    lines.append("")
    for model, result in summary["models"].items():
        lines.append(f"Confusion, {model} (rows true {', '.join(names)}; columns predicted): {result['metrics']['confusion']}")
    lines.append("")
    return "\n".join(lines) + "\n"
