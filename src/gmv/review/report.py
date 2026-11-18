"""Scoring of the model against human excerpt labels, and the batch report.

Pools are never mixed: core excerpts give the primary result; feature-selected excerpts
have their own table; model-flagged excerpts are diagnostic only. Within a pool, "present
throughout" hops are positives and "absent" hops are negatives; intermittent excerpts are
reported on their own, uncertain and unreviewed ones are only counted.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import timelines as tl
from .annotations import AnnotationStore
from .excerpts import CATEGORIES
from .pipeline import load_tables

POOL_TITLES = {
    "core": "Primary result: core excerpts (unbiased positions)",
    "feature": "Feature-selected excerpts (separate, not part of the primary result)",
    "model_flagged": "Model-flagged excerpts (diagnostic only, never an evaluation result)",
}


def _hop_states(timeline, start, end, hop=0.5):
    ends = timeline["t_end"]
    inside = (ends - hop >= start - 1e-6) & (ends <= end + 1e-6)
    return timeline["guitar_state"][inside]


def evaluate(workdir, store=None):
    workdir = Path(workdir)
    _, excerpts = load_tables(workdir)
    store = store or AnnotationStore(workdir)
    cache = {}

    def states(e):
        rid = e["recording_id"]
        if rid not in cache:
            path = workdir / "timelines" / f"{rid}.npz"
            cache[rid] = tl.load(path) if path.is_file() else None
        if cache[rid] is None:
            return np.array([], np.int8)
        return _hop_states(cache[rid], float(e["source_start_s"]), float(e["source_end_s"]))

    pools = {}
    for category in CATEGORIES:
        rows = [e for e in excerpts if e["excerpt_category"] == category]
        counts = Counter(
            (store.values(e["excerpt_id"])["guitar_presence"] or "unreviewed") for e in rows
        )
        pool = {"excerpts": len(rows), "labels": dict(counts), "levels": {}}
        for level, name in ((2, "detected"), (1, "detected_or_candidate")):
            tp = fn = fp = tn = 0
            pos_ok = pos_n = neg_any = neg_major = neg_n = 0
            inter = []
            for e in rows:
                presence = store.values(e["excerpt_id"])["guitar_presence"]
                s = states(e)
                if not len(s):
                    continue
                hit = s >= level
                if presence == "present_throughout":
                    tp += int(hit.sum())
                    fn += int((~hit).sum())
                    pos_n += 1
                    pos_ok += int(hit.mean() >= 0.5)
                elif presence == "absent":
                    fp += int(hit.sum())
                    tn += int((~hit).sum())
                    neg_n += 1
                    neg_any += int(hit.any())
                    neg_major += int(hit.mean() >= 0.5)
                elif presence == "present_intermittent":
                    inter.append(float(hit.mean()))
            recall = tp / (tp + fn) if tp + fn else None
            fpr = fp / (fp + tn) if fp + tn else None
            precision = tp / (tp + fp) if tp + fp else None
            f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
            pool["levels"][name] = {
                "positive_hops": tp + fn,
                "negative_hops": fp + tn,
                "recall": recall,
                "false_positive_rate": fpr,
                "precision": precision,
                "f1": f1,
                "throughout_excerpts": pos_n,
                "throughout_excerpts_mostly_on": pos_ok,
                "absent_excerpts": neg_n,
                "absent_excerpts_any_on": neg_any,
                "absent_excerpts_mostly_on": neg_major,
                "intermittent_excerpts": len(inter),
                "intermittent_mean_share_on": float(np.mean(inter)) if inter else None,
            }
        pools[category] = pool
    disagreements = []
    for e in excerpts:
        presence = store.values(e["excerpt_id"])["guitar_presence"]
        s = states(e)
        if not len(s) or presence not in ("present_throughout", "absent"):
            continue
        share = float((s == 2).mean())
        if presence == "absent" and share >= 0.5:
            disagreements.append((e, "model detected guitar where the reviewer heard none", share))
        elif presence == "present_throughout" and share < 0.2:
            disagreements.append(
                (e, "model mostly missed guitar the reviewer heard throughout", share)
            )
    return pools, disagreements


def fmt(value, pct=True):
    if value is None:
        return "–"
    return f"{value:.0%}" if pct else f"{value}"


def table(headers, rows):
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for row in rows:
        out.append(
            "| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in row) + " |"
        )
    return "\n".join(out)


def write_report(workdir):
    workdir = Path(workdir)
    batch, excerpts = load_tables(workdir)
    store = AnnotationStore(workdir)
    store.write_csv(excerpts, workdir / "Human_Annotations.csv")
    pools, disagreements = evaluate(workdir, store)
    build = (
        json.loads((workdir / "build.json").read_text(encoding="utf-8"))
        if (workdir / "build.json").is_file()
        else {}
    )
    verification = (
        json.loads((workdir / "verification.json").read_text(encoding="utf-8"))
        if (workdir / "verification.json").is_file()
        else None
    )
    labelled = sum(1 for e in excerpts if store.values(e["excerpt_id"])["guitar_presence"])
    by_cat = Counter(r["selection_category"] for r in batch)
    lines = [
        "# Validation batch report",
        "",
        "Local file: names the owner's recordings; not for the repository. **Model output in this "
        "report is model output, not a label. Only the human review section reports labels.**",
        "",
        "## Summary",
        "",
        table(
            ["Measure", "Value"],
            [
                ["Recordings", len(batch)],
                *[[f"… {c.replace('_', ' ')}", n] for c, n in sorted(by_cat.items())],
                ["Artists (folders)", len({r["top_folder"] for r in batch})],
                ["Excerpts", len(excerpts)],
                *[
                    [f"… {c}", sum(1 for e in excerpts if e["excerpt_category"] == c)]
                    for c in CATEGORIES
                ],
                ["Excerpts labelled by the reviewer", f"{labelled} of {len(excerpts)}"],
                ["Selection problems", "; ".join(build.get("selection_problems", [])) or "none"],
                [
                    "Sources and excerpts verified",
                    "–"
                    if not verification
                    else (
                        "passed"
                        if verification["passed"]
                        else "FAILED: " + json.dumps(verification)
                    ),
                ],
            ],
        ),
        "",
        "## Recordings and why each was selected",
        "",
        "Selection used metadata and stated reasons only. The `inventory model` column is the earlier "
        "inventory observation, copied for comparison; it played no part in choosing.",
        "",
        table(
            [
                "Category",
                "Artist",
                "Title",
                "Album",
                "Method",
                "Reason",
                "Min",
                "Inventory estimate",
            ],
            [
                [
                    r["selection_category"].replace("_candidate", ""),
                    r["artist"],
                    r["title"],
                    r["album"],
                    r["selection_method"],
                    r["selection_reason"],
                    f"{float(r['duration_s'] or 0) / 60:.1f}",
                    r["preliminary_guitar_estimate"],
                ]
                for r in batch
            ],
        ),
        "",
        "## Diversity",
        "",
        table(
            ["Artist folder", "Recordings", "Categories"],
            [
                [
                    f,
                    n,
                    ", ".join(
                        sorted(
                            {
                                r["selection_category"].replace("_candidate", "")
                                for r in batch
                                if r["top_folder"] == f
                            }
                        )
                    ),
                ]
                for f, n in Counter(r["top_folder"] for r in batch).most_common()
            ],
        ),
        "",
        "Genre tags: "
        + ", ".join(
            f"{g or '(none)'}: {n}" for g, n in Counter(r["genre"] for r in batch).most_common()
        )
        + ".",
        "",
        "Instrumental / vocal (inventory, metadata-based): "
        + ", ".join(
            f"{k}: {v}" for k, v in Counter(r["instrumental_or_vocal"] for r in batch).most_common()
        )
        + ".",
        "",
        "## Excerpt method",
        "",
        "- **core** (primary, unbiased): five 12 s excerpts per recording at the start, middle and end "
        "and at two random positions seeded from the file's SHA-256. No model or audio feature is consulted.",
        "- **feature**: the free 12 s window whose level and brightness differ most from the rest of the "
        "track (robust z-scores). Audio measurements only.",
        "- **model_flagged**: at most one window where the model's own output suggests a possible miss "
        "(guitar expected) or false activation (guitar not expected). Diagnostic only.",
        "- Excerpts never overlap. Each is the decoded source samples written unchanged as 32-bit float "
        "WAV at the source sample rate and channel count; every excerpt was compared sample-for-sample "
        "with the decoded source.",
        "- Model preprocessing (channel mean, resample to 32 kHz, log-mel) happens only inside the model "
        "path. Timelines use the application's own `Recognizer` with 10 s of context ending at each hop, "
        "so excerpt readouts come from the full track, not from the excerpt alone. Readouts for excerpts "
        "starting before 10 s are marked `model_context_complete = false`.",
        "",
        "## Source quality and integrity",
        "",
    ]
    issue_counts = Counter(
        i for r in batch for i in (r["source_integrity_issues"] or "").split(";") if i
    )
    lines += [
        table(["Issue", "Recordings"], issue_counts.most_common())
        if issue_counts
        else "No issues found.",
        "",
    ]
    flagged = [
        r
        for r in batch
        if r["source_integrity_issues"]
        and any(
            k in r["source_integrity_issues"]
            for k in (
                "clipping",
                "decoded_length",
                "decoder",
                "band_limited",
                "identical",
                "interior_low",
                "beyond",
                "low_bitrate",
            )
        )
    ]
    if flagged:
        lines += [
            table(
                ["Artist", "Title", "Rate", "Codec / kbit/s", "Bandwidth", "Peak dBFS", "Issues"],
                [
                    [
                        r["artist"],
                        r["title"],
                        r["source_sample_rate_hz"],
                        f"{r['source_codec']} / {r['source_bitrate_kbps']}",
                        r["source_effective_bandwidth_hz"],
                        r["source_peak_dbfs"],
                        r["source_integrity_issues"],
                    ]
                    for r in flagged
                ],
            ),
            "",
        ]
    mismatch = [e["excerpt_id"] for e in excerpts if e["excerpt_matches_source_samples"] != "true"]
    lines += [
        f"Excerpts identical to the decoded source samples: {len(excerpts) - len(mismatch)} of {len(excerpts)}"
        + (f" (mismatch: {', '.join(mismatch)})" if mismatch else "")
        + ".",
        "",
    ]

    lines += [
        "## Existing model behaviour (model output, not labels)",
        "",
        "Per recording, over the whole track. *Possible* passages are hints for listening; "
        "they are not labels.",
        "",
        table(
            [
                "Category",
                "Artist",
                "Title",
                "Detected",
                "Detected or cand.",
                "Longest on (s)",
                "Longest off (s)",
                "Possible misses (≥10 s off)",
                "Possible false activations (≥4 s on)",
                "Quiet hops off",
            ],
            [
                [
                    r["selection_category"].replace("_candidate", ""),
                    r["artist"],
                    r["title"],
                    fmt(float(r["model_share_detected"])) if r["model_share_detected"] else "–",
                    fmt(float(r["model_share_detected_or_candidate"]))
                    if r["model_share_detected_or_candidate"]
                    else "–",
                    r["model_longest_detected_s"],
                    r["model_longest_inactive_s"],
                    r["possible_missed_passages"] or "",
                    r["possible_false_activations"] or "",
                    f"{r['quiet_hops_model_inactive']}/{r['quiet_hops']}",
                ]
                for r in batch
            ],
        ),
        "",
    ]
    by_category = defaultdict(list)
    for r in batch:
        if r["model_share_detected"]:
            by_category[r["selection_category"]].append(float(r["model_share_detected"]))
    lines += [
        "Mean share of time detected, by selection category: "
        + ", ".join(
            f"{c.replace('_candidate', '')}: {np.mean(v):.0%} (n={len(v)})"
            for c, v in sorted(by_category.items())
        )
        + ".",
        "",
    ]

    lines += ["## Human review results", ""]
    if not labelled:
        lines += [
            "No excerpts have been labelled yet. Open the review window (`gmv review open`) and rerun "
            "`gmv review report` afterwards.",
            "",
        ]
    for category in CATEGORIES:
        pool = pools[category]
        lines += [
            f"### {POOL_TITLES[category]}",
            "",
            "Labels: " + ", ".join(f"{k}: {v}" for k, v in sorted(pool["labels"].items())) + ".",
            "",
        ]
        rows = []
        for level, m in pool["levels"].items():
            rows.append(
                [
                    level.replace("_", " "),
                    m["positive_hops"],
                    m["negative_hops"],
                    fmt(m["recall"]),
                    fmt(m["false_positive_rate"]),
                    fmt(m["precision"]),
                    fmt(m["f1"]),
                    f"{m['throughout_excerpts_mostly_on']}/{m['throughout_excerpts']}",
                    f"{m['absent_excerpts_any_on']}/{m['absent_excerpts']}",
                    f"{m['intermittent_excerpts']} ({fmt(m['intermittent_mean_share_on'])} on)",
                ]
            )
        lines += [
            table(
                [
                    "Model state counted as on",
                    "Guitar hops",
                    "No-guitar hops",
                    "Recall",
                    "FPR",
                    "Precision",
                    "F1",
                    "Throughout excerpts mostly on",
                    "Absent excerpts with any on",
                    "Intermittent (mean share on)",
                ],
                rows,
            ),
            "",
        ]
        if category == "model_flagged":
            lines += [
                "These excerpts were chosen *because* of the model's output, so their numbers describe "
                "how often its suspicious passages are real errors, not its accuracy.",
                "",
            ]
    if disagreements:
        lines += [
            "### Disagreements between reviewer and model",
            "",
            table(
                [
                    "Excerpt",
                    "Category",
                    "Artist",
                    "Title",
                    "Position",
                    "What",
                    "Model share detected",
                ],
                [
                    [
                        e["excerpt_id"],
                        e["excerpt_category"],
                        e["artist"],
                        e["title"],
                        f"{e['source_start_clock']}–{e['source_end_clock']}",
                        what,
                        f"{share:.0%}",
                    ]
                    for e, what, share in disagreements
                ],
            ),
            "",
        ]
    lines += [
        "## Limitations",
        "",
        "- About 65 recordings from one collection; heavily weighted to metal because the collection is.",
        "- Excerpt labels are 12 s long. Intermittent excerpts keep only approximate, reviewer-written spans and "
        "are left out of hop-level scores.",
        "- Hops in an excerpt are not independent; recall and false-positive rates are descriptive, not "
        "confidence-bounded.",
        "- Every excerpt is labelled by one person; there is no second annotator yet.",
        "- Lossy and band-limited sources (see the integrity table) may behave differently from lossless audio.",
        "- The batch is evaluation material. It is not a training set, and the model's own output was never "
        "used to choose recordings or core excerpts.",
        "",
        "## Recommended next steps",
        "",
        "1. Label the core excerpts first (the queue starts there), then feature-selected, then model-flagged.",
        "2. Work through the uncertain queue once; leave genuinely ambiguous excerpts uncertain.",
        "3. `gmv review export` writes the core labels in the corpus annotation format so "
        "`scripts/score_recognition.py` can use them.",
        "4. Decide which whole artists to hold out for testing before any of this material is used for training.",
        "5. Add more guitar-free material (piano, orchestral, electronic) from outside this collection; it has little.",
        "",
    ]
    (workdir / "Validation_Batch_Report.md").write_text("\n".join(lines), encoding="utf-8")
    (workdir / "evaluation.json").write_text(json.dumps(pools, indent=2), encoding="utf-8")
    return pools
