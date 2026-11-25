"""What the experimental measurements are worth across a corpus, with the groups that set
the scales kept apart from the groups they are judged on.

The corpus is a table the user writes: one row per recording with an ``id`` to show in
reports, a ``path``, a ``group`` (the artist, or whatever must not straddle the split) and
any number of ``label:<name>`` columns holding 1, 0 or nothing. Reports use the ids only.

**The split.** Groups, not recordings, are divided in two by a hash of the group's name:
*calibration* groups set every scale (means and standard deviations for standard scores,
the regression that predicts a new measurement from the Phase A profile), and
*evaluation* groups are what every figure is reported on. No artist is on both sides. A
row can force its group to one side with ``split``. None of the measurements is fitted to
anything: there are no learned weights, so the split protects the scales, not a model.

What is asked of each measurement:

* is it **available** (how often it can be stated at all, and how confidently);
* is it **new**: how much of it the seven Phase A dimensions already predict (R² on the
  evaluation groups of a regression fitted on the calibration groups);
* is it a property of **the recording**: the share of its block-to-block variation that
  lies between recordings instead of inside them;
* is it a property of **the artist**: the same between artist groups. Music by one artist
  is not all alike, so this is a weak, label-free indication, not a target;
* does it follow a **label** (area under the ROC curve on the evaluation groups);
* does it survive a change of **level**.
"""

import csv
import hashlib
import math
import warnings
from pathlib import Path

import numpy as np

from ..profile import DIMENSION_NAMES
from .profile import FAMILIES, NAMES
from .stems import auc

PHASE_A = tuple(f"phase_a.{name}" for name in DIMENSION_NAMES)
ALL = PHASE_A + NAMES
SETS = {
    "Phase A": PHASE_A,
    "tonal": tuple(n for n in NAMES if n.startswith("tonal.")),
    "lead": tuple(n for n in NAMES if n.startswith("lead.")),
    "rhythm": tuple(n for n in NAMES if n.startswith("rhythm.")),
    "Phase A + tonal": PHASE_A + tuple(n for n in NAMES if n.startswith("tonal.")),
    "Phase A + lead": PHASE_A + tuple(n for n in NAMES if n.startswith("lead.")),
    "Phase A + rhythm": PHASE_A + tuple(n for n in NAMES if n.startswith("rhythm.")),
    "Phase A + all": ALL,
}


def read_table(path):
    """Rows of the corpus table as dicts; labels as {name: 1.0 | 0.0 | None}."""
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if not (row.get("path") or "").strip():
                continue
            labels = {}
            for key, value in row.items():
                if key and key.startswith("label:"):
                    labels[key[6:]] = float(value) if (value or "").strip() in ("0", "1") else None
            rows.append(
                {
                    "id": row["id"].strip(),
                    "path": row["path"].strip(),
                    "group": (row.get("group") or row["id"]).strip(),
                    "split": (row.get("split") or "").strip(),
                    "labels": labels,
                }
            )
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Every recording needs its own id")
    return rows


def assign(rows, seed="sonic-character-a2"):
    """group -> "calibration" | "evaluation". A group any of whose rows names a side goes
    there; the rest are divided by a hash of the group name, so adding recordings later
    never moves a group."""
    forced = {row["group"]: row["split"] for row in rows if row["split"] in ("calibration", "evaluation")}
    out = {}
    for group in sorted({row["group"] for row in rows}):
        digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
        out[group] = forced.get(group) or ("calibration" if digest[0] % 2 == 0 else "evaluation")
    return out


def features(found):
    """One recording's values in the order of ``ALL`` (NaN where unknown)."""
    values = [
        math.nan if d["value"] is None else d["value"] for d in found["phase_a"]["overall"]["dimensions"]
    ]
    for name in NAMES:
        family, key = name.split(".", 1)
        value = found["families"][family][key]["value"]
        values.append(math.nan if value is None else value)
    return np.array(values, dtype=float)


def between_share(groups):
    """Share of the variance of a measurement that lies between groups (lists of values)
    rather than inside them; None with fewer than three groups of two or more."""
    groups = [np.asarray(g, dtype=float) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 3:
        return None
    within = float(np.mean([g.var(ddof=1) for g in groups]))
    size = float(np.mean([len(g) for g in groups]))
    between = max(0.0, float(np.var([g.mean() for g in groups], ddof=1)) - within / size)
    return between / (between + within) if between + within > 0 else None


def retrieval(scores, groups):
    """Share of recordings whose nearest other recording (Euclidean, standard scores,
    unknowns at the mean) is of the same group, among recordings that have a group mate;
    and the share expected by chance."""
    scores = np.where(np.isfinite(scores), scores, 0.0)
    groups = np.asarray(groups)
    hits, chance, counted = 0, 0.0, 0
    for index in range(len(scores)):
        mates = int((groups == groups[index]).sum()) - 1
        if mates < 1:
            continue
        distance = np.linalg.norm(scores - scores[index], axis=1)
        distance[index] = np.inf
        hits += int(groups[int(np.argmin(distance))] == groups[index])
        chance += mates / (len(scores) - 1)
        counted += 1
    return (hits / counted, chance / counted, counted) if counted else (None, None, 0)


def rounded(value, digits=3):
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def evaluate(rows, results, level_runs=None, pairs=()):
    """All the corpus statistics. ``results`` maps id -> research profile; ``level_runs``
    maps gain in dB -> {id -> profile}; ``pairs`` are (id, id) to compare in detail."""
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        # A measurement nobody could state is a column of unknowns: expected, not an error.
        warnings.simplefilter("ignore", RuntimeWarning)
        return _evaluate(rows, results, level_runs, pairs)


def _evaluate(rows, results, level_runs, pairs):
    rows = [row for row in rows if row["id"] in results]
    sides = assign(rows)
    ids = [row["id"] for row in rows]
    group = np.array([row["group"] for row in rows])
    side = np.array([sides[g] for g in group])
    matrix = np.array([features(results[i]) for i in ids])
    calibration, evaluation = side == "calibration", side == "evaluation"
    mean = np.nanmean(matrix[calibration], axis=0)
    spread = np.nanstd(matrix[calibration], axis=0)
    spread = np.where(spread > 0, spread, 1.0)
    scores = (matrix - mean) / spread
    index = {name: i for i, name in enumerate(ALL)}
    out = {
        "split": {
            "calibration_groups": sorted(g for g, s in sides.items() if s == "calibration"),
            "evaluation_groups": sorted(g for g, s in sides.items() if s == "evaluation"),
            "calibration_recordings": int(calibration.sum()),
            "evaluation_recordings": int(evaluation.sum()),
            "recording_split": dict(zip(ids, side.tolist(), strict=True)),
        },
        "scales": {name: {"mean": rounded(mean[i], 4), "sd": rounded(spread[i], 4)} for name, i in index.items()},
        "measurements": {},
    }
    # Regression of each new measurement on the Phase A profile, fitted on calibration.
    design = np.column_stack((np.ones(len(scores)), np.where(np.isfinite(scores[:, :7]), scores[:, :7], 0.0)))
    for name, column in index.items():
        values = matrix[:, column]
        known = np.isfinite(values)
        entry = {
            "available_share": rounded(known.mean()),
            "evaluation_sd_in_calibration_sds": rounded(np.nanstd(scores[evaluation, column])),
        }
        if not name.startswith("phase_a."):
            family, key = name.split(".", 1)
            entry["unit"], entry["kind"] = FAMILIES[family][key][0], FAMILIES[family][key][1]
            entry["mean_confidence"] = rounded(
                np.mean([results[i]["families"][family][key]["confidence"] for i in ids])
            )
            fit, test = known & calibration, known & evaluation
            beyond = None
            if fit.sum() >= 12 and test.sum() >= 8:
                weights = np.linalg.lstsq(design[fit], scores[fit, column], rcond=None)[0]
                beyond = scores[:, column] - design @ weights  # what Phase A does not predict
                residual = beyond[test]
                total = scores[test, column] - scores[test, column].mean()
                entry["predicted_by_phase_a_r2"] = rounded(1.0 - float(residual @ residual) / max(float(total @ total), 1e-12))
            entry["label_auc_beyond_phase_a"] = {}
            for label in sorted({k for row in rows for k in row["labels"]}):
                truth = np.array([row["labels"].get(label) for row in rows], dtype=float)
                keep = evaluation & known & np.isfinite(truth)
                if beyond is not None and keep.sum() >= 12:
                    entry["label_auc_beyond_phase_a"][label] = rounded(auc(beyond[keep], truth[keep] == 1.0))
            blocks = [
                results[i]["evolution"][family][key] for i, keep in zip(ids, evaluation, strict=True) if keep
            ]
            entry["between_recordings_share"] = rounded(
                between_share([[v for v in b if v is not None] for b in blocks])
            )
        by_group = [values[evaluation & (group == g)] for g in sorted(set(group[evaluation]))]
        entry["between_artists_share"] = rounded(between_share(by_group))
        labels = {}
        for label in sorted({k for row in rows for k in row["labels"]}):
            truth = np.array([row["labels"].get(label) for row in rows], dtype=float)
            keep = evaluation & known & np.isfinite(truth)
            if keep.sum() >= 12:
                labels[label] = rounded(auc(values[keep], truth[keep] == 1.0))
        entry["label_auc"] = labels
        out["measurements"][name] = entry
    out["retrieval_same_artist"] = {}
    for label, names in SETS.items():
        columns = [index[name] for name in names]
        rate, chance, counted = retrieval(scores[evaluation][:, columns], group[evaluation])
        out["retrieval_same_artist"][label] = {"hit_rate": rounded(rate), "chance": rounded(chance), "recordings": counted}
    # Correlation between everything, on the evaluation groups.
    filled = np.where(np.isfinite(scores[evaluation]), scores[evaluation], 0.0)
    if evaluation.sum() >= 8:
        matrix_r = np.corrcoef(filled.T)
        strongest = {}
        for name, column in index.items():
            others = [(abs(matrix_r[column, other]), ALL[other], matrix_r[column, other]) for other in range(len(ALL)) if other != column and np.isfinite(matrix_r[column, other])]
            if others:
                _, partner, value = max(others)
                strongest[name] = {"with": partner, "r": rounded(value, 2)}
        out["strongest_correlation"] = strongest
    # Level: change of each value, in calibration standard deviations.
    out["level"] = {}
    for gain, runs in sorted((level_runs or {}).items()):
        moved = np.array([np.abs(features(run) - features(results[i])) / spread for i, run in runs.items() if i in results])
        lost = np.array([np.isfinite(features(results[i])) & ~np.isfinite(features(run)) for i, run in runs.items() if i in results])
        if len(moved):
            out["level"][f"{gain:g}"] = {
                "recordings": len(moved),
                "measurements": {
                    name: {
                        "median": rounded(np.nanmedian(moved[:, i])) if np.isfinite(moved[:, i]).any() else None,
                        "p90": rounded(np.nanpercentile(moved[:, i], 90)) if np.isfinite(moved[:, i]).any() else None,
                        "largest": rounded(np.nanmax(moved[:, i])) if np.isfinite(moved[:, i]).any() else None,
                        "became_unknown": int(lost[:, i].sum()),
                    }
                    for name, i in index.items()
                },
            }
    # Pairs: where two recordings differ, against how recordings differ in general.
    out["pairs"] = []
    pool = scores[evaluation]
    upper = np.triu_indices(len(pool), 1)
    for first, second in pairs:
        if first not in results or second not in results:
            continue
        a, b = scores[ids.index(first)], scores[ids.index(second)]
        entry = {"a": first, "b": second, "measurements": {}, "distance": {}}
        for name, column in index.items():
            if not np.isfinite([a[column], b[column]]).all():
                entry["measurements"][name] = None
                continue
            apart = np.abs(pool[:, column][:, None] - pool[:, column][None, :])[upper]
            apart = apart[np.isfinite(apart)]
            entry["measurements"][name] = {
                "a": rounded(matrix[ids.index(first), column], 4),
                "b": rounded(matrix[ids.index(second), column], 4),
                "apart_sd": rounded(abs(a[column] - b[column]), 2),
                "more_apart_than_share_of_pairs": rounded((apart < abs(a[column] - b[column])).mean(), 2) if len(apart) else None,
            }
        for label, names in SETS.items():
            columns = [index[name] for name in names]
            usable = [c for c in columns if np.isfinite([a[c], b[c]]).all()]
            if not usable:
                continue
            fill = np.where(np.isfinite(pool[:, usable]), pool[:, usable], 0.0)
            apart = np.sqrt(((fill[:, None, :] - fill[None, :, :]) ** 2).mean(axis=2))[upper]
            own = float(np.sqrt(((a[usable] - b[usable]) ** 2).mean()))
            entry["distance"][label] = {"rms_sd": rounded(own, 2), "more_apart_than_share_of_pairs": rounded((apart < own).mean(), 2)}
        out["pairs"].append(entry)
    held = [i for i, keep in zip(ids, evaluation, strict=True) if keep]
    out["timescales_between_recordings_share"] = timescales(results, held)
    out["blocks_with_intensity"] = with_intensity(results, held)
    family = lambda prefix: [index[n] for n in NAMES if n.startswith(prefix)]  # noqa: E731
    reliable = lambda names: [index[n] for n in names]  # noqa: E731
    out["contrasts"] = {
        "similar_pace_different_rhythm": contrast(
            scores[evaluation], held, [index["phase_a.pace"]],
            reliable(("rhythm.pulse_clarity", "rhythm.on_beat_share", "rhythm.events_per_beat", "rhythm.offbeat_weight", "rhythm.bar_repetition")),
        ),
        "similar_harmony_different_lead": contrast(
            scores[evaluation], held,
            reliable(("tonal.harmonic_share", "tonal.harmonic_movement", "tonal.pitch_class_focus", "tonal.tonal_stability")),
            reliable(("lead.pitched_share", "lead.line_range_st", "lead.line_motion_st_s", "lead.held_share")),
        ),
        "different_texture_similar_music": contrast(
            scores[evaluation], held, family("tonal.") + reliable(("rhythm.pulse_clarity", "rhythm.events_per_beat", "lead.pitched_share")),
            [index[n] for n in PHASE_A],
        ),
    }
    out["transitions"] = transitions(results, held)
    return out


# -- over time ----------------------------------------------------------------------------


def series(found, name):
    """One measurement's block values of a recording, with the block centres."""
    family, key = name.split(".", 1)
    evolution = found["evolution"][family]
    centres = (np.array(evolution["start_s"]) + np.array(evolution["end_s"])) / 2
    values = np.array([np.nan if v is None else v for v in evolution[key]], dtype=float)
    return centres, values


def timescales(results, ids, counts=(1, 2, 4, 8)):
    """Per measurement: the share of the variation that lies between recordings when its
    blocks are averaged ``count`` at a time. A measurement is a property of the recording
    at the scale where this stops rising."""
    out = {}
    for name in NAMES:
        entry = {}
        for count in counts:
            groups = []
            for identity in ids:
                _, values = series(results[identity], name)
                values = values[: len(values) // count * count].reshape(-1, count)
                with np.errstate(invalid="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    groups.append(np.nanmean(values, axis=1))
            entry[str(count)] = rounded(between_share(groups))
        out[name] = entry
    return out


def with_intensity(results, ids):
    """Per measurement: the mean correlation, inside a recording, between its blocks and
    MusicalIntensity over the same blocks. Near zero: it does not follow intensity."""
    out = {}
    for name in NAMES:
        found = []
        for identity in ids:
            evolution = results[identity]["phase_a"]["evolution"]
            times, intensity = np.array(evolution["time"], dtype=float), np.array(evolution["intensity"], dtype=float)
            centres, values = series(results[identity], name)
            family = name.split(".", 1)[0]
            starts = np.array(results[identity]["evolution"][family]["start_s"])
            ends = np.array(results[identity]["evolution"][family]["end_s"])
            level = np.array([intensity[(times >= s) & (times < e)].mean() if ((times >= s) & (times < e)).any() else np.nan for s, e in zip(starts, ends, strict=True)])
            keep = np.isfinite(values) & np.isfinite(level)
            if keep.sum() >= 8 and values[keep].std() > 0 and level[keep].std() > 0:
                found.append(np.corrcoef(values[keep], level[keep])[0, 1])
        out[name] = {"mean_r": rounded(np.mean(found), 2) if found else None,
                     "mean_abs_r": rounded(np.mean(np.abs(found)), 2) if found else None, "recordings": len(found)}
    return out


def contrast(scores, ids, alike, differ, most_alike=0.1):
    """The pair of recordings that is among the most alike on the ``alike`` columns and
    furthest apart on the ``differ`` columns (both known for both)."""
    best = None
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = scores[i], scores[j]
            if not np.isfinite(np.concatenate((a[alike], b[alike], a[differ], b[differ]))).all():
                continue
            pairs.append((float(np.sqrt(((a[alike] - b[alike]) ** 2).mean())), float(np.sqrt(((a[differ] - b[differ]) ** 2).mean())), i, j))
    if len(pairs) < 20:
        return None
    limit = np.quantile([p[0] for p in pairs], most_alike)
    best = max((p for p in pairs if p[0] <= limit), key=lambda p: p[1])
    return {"a": ids[best[2]], "b": ids[best[3]], "alike_rms_sd": rounded(best[0], 2), "differ_rms_sd": rounded(best[1], 2),
            "pairs_considered": len(pairs)}


def transitions(results, ids, ahead_s=10.0, behind_s=30.0, seed=7):
    """Could a change of track be told from a change inside a track, by the Phase A
    short-window profile alone? ``change`` is the mean difference per dimension between
    the profile of the next ``ahead_s`` and of the last ``behind_s``. Inside each recording
    its largest value is what a detector must not fire on; across a boundary (the end of
    one recording followed by the start of another, in a fixed shuffled order) it is what
    the detector must catch."""
    def active(identity):
        evolution = results[identity]["phase_a"]["evolution"]
        keep = np.array(evolution["active"], dtype=bool)
        values = np.array([[np.nan if v is None else v for v in row] for row in evolution["short"]], dtype=float)
        return np.array(evolution["time"], dtype=float)[keep], values[keep]

    ahead, behind = int(ahead_s), int(behind_s)
    inside, tracks = [], {}
    for identity in ids:
        _, values = active(identity)
        if len(values) < behind + ahead + 20:
            continue
        tracks[identity] = values
        changes = [
            np.nanmean(np.abs(np.nanmean(values[t : t + ahead], axis=0) - np.nanmean(values[t - behind : t], axis=0)))
            for t in range(behind + 10, len(values) - ahead - 10)
        ]
        inside.append(float(np.nanmax(changes)))
    order = sorted(tracks)
    np.random.default_rng(seed).shuffle(order)
    across = []
    for first, second in zip(order, order[1:] + order[:1], strict=True):
        # The short window needs a few seconds of the new track before it shows it.
        start = np.nanmean(tracks[second][5 : 5 + ahead], axis=0)
        across.append(float(np.nanmean(np.abs(start - np.nanmean(tracks[first][-behind:], axis=0)))))
    if len(inside) < 10:
        return None
    inside, across = np.array(inside), np.array(across)
    out = {"recordings": len(inside), "ahead_s": ahead_s, "behind_s": behind_s,
           "largest_change_inside_a_recording_median": rounded(np.median(inside)),
           "change_across_a_boundary_median": rounded(np.median(across)), "thresholds": []}
    for allowed in (0.02, 0.1, 0.25):
        threshold = float(np.quantile(inside, 1 - allowed))
        out["thresholds"].append({"recordings_with_a_false_reset": allowed, "threshold": rounded(threshold),
                                  "boundaries_caught": rounded(float((across > threshold).mean()))})
    return out
