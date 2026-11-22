"""The two-arm experiment on the owner's recordings: what each arm may train on, what is
scored, how randomness is shared between arms, segment weights, fixed thresholds and the
paired comparison. No torch, no real recordings and no training: MoisesDB clips are faked."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "recognition"))
import cache_review  # noqa: E402
import review_data as rd  # noqa: E402
import review_stats as rs  # noqa: E402
import review_summary  # noqa: E402


def segment(number, group, fold, label="present", source="human", pool="primary", kind="", seconds=12):
    rng = np.random.default_rng(number)
    return {
        "id": f"s{number:03d}",
        "recording": f"r-{group}-{number // 2}",
        "group": group,
        "fold": fold,
        "same_audio": "",
        "label": label,
        "type": kind,
        "source": source,
        "weight": rd.WEIGHTS[source],
        "start": 30.0,
        "length": float(seconds),
        "pool": pool if source == "human" else "",
        "selection": "unbiased" if pool == "primary" else "model_selected:random_audit",
        "audio": (rng.standard_normal(rd.CLIP + seconds * rd.RATE) * 0.1).astype(np.float16),
    }


def library():
    """Five folds, two groups each: human guitar and guitar-free excerpts everywhere, and
    automatic positives in some groups, one of which has no human answer at all."""
    items, number = [], 0
    for fold in range(5):
        for half in "ab":
            group = f"g{fold}{half}"
            for label, pool, kind in (
                ("present", "primary", "distorted_electric"),
                ("absent", "primary", ""),
                ("present", "secondary", "acoustic"),
                ("absent", "secondary", ""),
            ):
                items.append(segment(number, group, fold, label, pool=pool, kind=kind))
                number += 1
            if half == "a":
                for _ in range(3):
                    items.append(segment(number, group, fold, source="automatic:moderate_agreement", seconds=20))
                    number += 1
        items.append(segment(number, f"g{fold}auto", fold, source="automatic:moderate_agreement", seconds=20))
        number += 1
    return items


# ----------------------------------------------------------------- what each arm may use


def test_arm_h_holds_human_segments_only_and_arm_ha_adds_automatic_positives_only():
    items = library()
    human = rd.training_items(items, "H", hold=2)
    both = rd.training_items(items, "H+A", hold=2)
    assert {i["source"] for i in human} == {"human"}
    assert {i["id"] for i in human} == {i["id"] for i in both if i["source"] == "human"}
    extra = [i for i in both if i["source"] != "human"]
    assert extra and all(i["label"] == "present" and i["weight"] == 0.25 for i in extra)
    assert not any(i["fold"] == 2 for i in both)


def test_an_automatic_guitar_free_segment_or_a_wrong_weight_stops_the_run():
    items = library()
    bad = dict(segment(900, "g0a", 0, source="automatic:moderate_agreement"), label="absent")
    with pytest.raises(ValueError, match="only be 'present'"):
        rd.training_items(items + [bad], "H+A", hold=2)
    rd.training_items(items + [bad], "H", hold=2)  # arm H never looks at automatic segments
    heavy = dict(segment(901, "g0a", 0, source="automatic:moderate_agreement"), weight=1.0)
    with pytest.raises(ValueError, match="weight"):
        rd.training_items(items + [heavy], "H+A", hold=2)
    undecided = dict(segment(902, "g0a", 0), label="present_intermittent")
    with pytest.raises(ValueError, match="usable label"):
        rd.training_items(items + [undecided], "H", hold=2)


def test_the_cache_refuses_a_table_with_automatic_negatives_or_scored_automatic_rows():
    row = {
        "segment_id": "x", "recording_id": "r-1", "usable": "True", "label": "present",
        "source": "automatic:moderate_agreement", "weight": "0.25", "evaluation_pool": "",
        "leakage_group": "g", "fold": "0",
    }  # fmt: skip
    assert list(cache_review.usable([row])) == ["r-1"]
    for change in ({"label": "absent"}, {"evaluation_pool": "primary"}, {"weight": "0.5"}):
        with pytest.raises(SystemExit):
            cache_review.usable([dict(row, **change)])
    with pytest.raises(SystemExit, match="different groups or folds"):
        cache_review.usable([row, dict(row, segment_id="y", fold="1")])
    assert cache_review.usable([dict(row, usable="False", label="present_intermittent")]) == {}


# ------------------------------------------------------------------- what is scored, and by whom


def test_only_human_answers_of_the_held_out_fold_are_scored():
    items = library()
    held = rd.evaluation_items(items, hold=3)
    assert held and all(i["source"] == "human" and i["fold"] == 3 for i in held)
    assert {i["pool"] for i in held} == {"primary", "secondary"}
    assert len(rd.evaluation_items(items)) == 5 * len(held)  # the baseline scores every fold


def test_training_and_scored_data_of_a_fold_share_nothing():
    items = library()
    for arm in rd.ARMS:
        for hold in range(5):
            assert rd.isolation_problems(rd.training_items(items, arm, hold), rd.evaluation_items(items, hold)) == []
    train, held = rd.training_items(items, "H+A", 0), rd.evaluation_items(items, 0)
    assert "leakage group" in " ".join(rd.isolation_problems(train + [dict(held[0], fold=1)], held))
    copy = dict(train[0], same_audio="same-audio-001")
    assert "same audio" in " ".join(rd.isolation_problems([copy], [dict(held[0], same_audio="same-audio-001")]))
    automatic = next(i for i in items if i["source"] != "human" and i["fold"] == 0)
    assert "automatic segment" in " ".join(rd.isolation_problems(train, held + [automatic]))


# ----------------------------------------------------------------------- seeds and batches


def test_seeds_depend_on_the_fold_and_the_stream_never_on_the_arm():
    assert rd.seeds(0) == rd.seeds(0)
    assert set(rd.seeds(0)) == set(rd.STREAMS)
    assert len(set(rd.seeds(0).values())) == len(rd.STREAMS)  # one stream per kind of decision
    assert len({json.dumps(rd.seeds(f)) for f in range(5)}) == 5
    assert rd.seeds(3)["plan"] == int(np.random.SeedSequence([rd.BASE_SEED, 3, 0]).generate_state(1)[0])


def fake_moisesdb(rng):
    audio = rng.standard_normal(rd.CLIP).astype(np.float32) * 0.1
    return audio, np.zeros((4, rd.COLUMNS), np.float32), np.ones((4, rd.COLUMNS), np.float32)


def run_batches(items, arm, hold, steps=12, batch=16):
    rngs = rd.generators(hold)
    planned = rd.plan(steps, batch, 0.4, 0.15, rngs["plan"])
    pool = rd.pools(rd.training_items(items, arm, hold))
    return planned, [rd.draw_batch(step, planned, pool, rngs, fake_moisesdb) for step in range(steps)]


def test_the_two_arms_of_a_fold_differ_only_in_the_guitar_clips_of_the_owners_recordings():
    items = library()
    plan_h, human = run_batches(items, "H", hold=1)
    plan_ha, both = run_batches(items, "H+A", hold=1)
    for key in plan_h:
        assert (plan_h[key] == plan_ha[key]).all()
    same = different = automatic = 0
    for (clips_h, kinds_h), (clips_ha, kinds_ha) in zip(human, both, strict=True):
        for clip_h, kind_h, clip_ha, kind_ha in zip(clips_h, kinds_h, clips_ha, kinds_ha, strict=True):
            assert not kind_h.startswith("automatic")  # arm H can never draw one
            if kind_h == "human:present":
                assert kind_ha.endswith(":present")
                automatic += kind_ha.startswith("automatic")
                different += not np.array_equal(clip_h[0], clip_ha[0])
            else:  # MoisesDB and guitar-free clips, their gains and silenced lead-ins
                assert kind_h == kind_ha
                for a, b in zip(clip_h, clip_ha, strict=True):
                    assert np.array_equal(a, b)
                same += 1
    assert same > 100 and different > 10 and automatic > 5
    again = run_batches(items, "H+A", hold=1)[1]
    assert all(np.array_equal(a[0], b[0]) for (x, _), (y, _) in zip(both, again, strict=True) for a, b in zip(x, y, strict=True))
    other_fold = run_batches(items, "H+A", hold=2)[1]
    assert not np.array_equal(both[0][0][0][0], other_fold[0][0][0][0])


def test_groups_are_drawn_equally_often_whatever_their_size():
    items = library()
    big = [segment(1000 + n, "g0a", 0, source="automatic:moderate_agreement") for n in range(60)]
    pool = rd.pools(rd.training_items(items + big, "H+A", hold=4))
    rng, augment = np.random.default_rng(0), np.random.default_rng(1)
    drawn = [rd.review_clip(pool["present"], rng, augment)[3]["group"] for _ in range(1200)]
    share = drawn.count("g0a") / len(drawn)
    assert abs(share - 1 / len(pool["present"])) < 0.04  # not 60-odd segments' worth


# ---------------------------------------------------------------------------- segment weights


def test_a_segments_weight_reaches_the_loss_columns_it_labels():
    rng, augment = np.random.default_rng(0), np.random.default_rng(1)
    auto = segment(1, "g", 0, source="automatic:moderate_agreement", seconds=20)
    _, target, weight, _ = rd.review_clip([[auto]], rng, augment)
    assert set(np.unique(weight[0])) <= {0.0, 0.25} and (weight[0] == 0.25).any()
    assert (target[0][weight[0] > 0] == 1).all()
    assert (weight[1:] == 0).all()  # an automatic segment says nothing about the guitar type
    person = segment(2, "g", 0, kind="clean_electric")
    _, target, weight, _ = rd.review_clip([[person]], rng, augment)
    assert set(np.unique(weight[0])) <= {0.0, 1.0} and (weight[2] == weight[0]).all() and (weight[1] == 0).all()
    free = segment(3, "g", 0, label="absent")
    _, target, weight, _ = rd.review_clip([[free]], rng, augment)
    assert (target == 0).all() and (weight.max(axis=1) == 1).all()  # guitar-free for every label


def test_the_loss_scales_with_the_weight_and_ignores_unweighted_columns():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((3, 4, rd.COLUMNS))
    target = (rng.random(logits.shape) < 0.5).astype(np.float32)
    full = rd.weighted_bce(logits, target, np.ones_like(logits))
    assert rd.weighted_bce(logits, target, np.full_like(logits, 0.25)) == pytest.approx(full / 4)
    weight = np.zeros_like(logits)
    weight[:, 0, 10:] = 0.25
    kept = rd.weighted_bce(logits, target, weight)
    noisy = logits.copy()
    noisy[:, 1:, :], noisy[:, 0, :10] = 50.0, -50.0  # nonsense where the weight is zero
    assert rd.weighted_bce(noisy, target, weight) == pytest.approx(kept)
    assert rd.weighted_bce(logits, target, weight * 2) == pytest.approx(kept * 2)


def test_the_training_script_uses_the_weighted_loss_and_needs_approval():
    torch = pytest.importorskip("torch")
    import finetune_review

    rng = np.random.default_rng(0)
    logits = rng.standard_normal((2, 4, rd.COLUMNS)).astype(np.float32)
    target = (rng.random(logits.shape) < 0.5).astype(np.float32)
    weight = rng.choice([0.0, 0.25, 1.0], logits.shape).astype(np.float32)
    got = float(finetune_review.loss_of(*(torch.from_numpy(v) for v in (logits, target, weight))))
    assert got == pytest.approx(rd.weighted_bce(logits, target, weight), abs=1e-5)


# ----------------------------------------------------------------- fixed thresholds, comparison

SHIPPED = {"detected": 0.926, "candidate": 0.627}


def test_fixed_thresholds_are_applied_as_given():
    pos = np.array([0.95, 0.93, 0.7, 0.6, 0.2])
    neg = np.array([0.94, 0.65, 0.3, 0.1])
    found = rs.figures(pos, neg, SHIPPED)
    assert found["recall_at_detected"] == pytest.approx(0.4)
    assert found["false_positive_rate_at_detected"] == pytest.approx(0.25)
    assert found["recall_at_candidate"] == pytest.approx(0.6)
    assert found["false_positive_rate_at_candidate"] == pytest.approx(0.5)
    assert rs.at_threshold(np.array([0.926]), np.array([0.9259]), 0.926) == (1.0, 0.0)


def pool(groups, arms, shift, seed=0, hops=24):
    """Excerpts whose guitar hops score higher than guitar-free ones; ``shift[arm]`` moves
    that arm's guitar scores."""
    rng = np.random.default_rng(seed)
    out = []
    for group in range(groups):
        for guitar in (True, False, True):
            base = rng.normal(0.75 if guitar else 0.45, 0.2, hops)
            out.append(
                {
                    "group": f"g{group}",
                    "guitar": guitar,
                    "scores": {arm: np.clip(base + (shift[arm] if guitar else 0), 0, 1) for arm in arms},
                }
            )
    return out


ARMS = ["baseline", "H", "H+A"]


def test_every_arm_is_measured_at_the_same_thresholds_and_changes_are_paired():
    excerpts = pool(12, ARMS, {"baseline": 0.0, "H": 0.0, "H+A": 0.1})
    found = rs.compare(excerpts, ARMS, SHIPPED, replicates=200)
    assert found["counts"]["groups"] == 12 and found["counts"]["excerpts_guitar_free"] == 12
    assert found["counts"]["hops_guitar"] == 12 * 2 * 24
    for arm in ARMS:
        assert set(found["arms"][arm]) >= {"recall_at_detected", "false_positive_rate_at_detected", "recall_at_candidate", "false_positive_rate_at_candidate"}
    same = found["changes"]["H minus baseline"]["auc"]
    assert same["change"] == 0 and same["interval_95"] == [0.0, 0.0]  # identical scores: no noise at all
    gain = found["changes"]["H+A minus baseline"]["auc"]
    assert gain["change"] > 0 and gain["interval_95"][0] > 0
    assert found["changes"]["H+A minus H"]["auc"]["change"] == pytest.approx(gain["change"])
    # Only guitar scores moved, so the false-positive rate is unchanged in every replicate.
    assert found["changes"]["H+A minus baseline"]["false_positive_rate_at_detected"]["interval_95"] == [0.0, 0.0]


def test_too_few_groups_give_no_interval_and_no_pass():
    excerpts = pool(3, ARMS, {"baseline": 0.0, "H": 0.0, "H+A": 0.1})
    found = rs.compare(excerpts, ARMS, SHIPPED, replicates=50)
    assert not found["interval_possible"]
    assert found["changes"]["H+A minus baseline"]["auc"]["interval_95"] is None
    control = {"baseline": {"auc": 0.95}, "H+A": {"auc_mean": 0.95, "auc_min": 0.94}}
    result = rs.verdict(found, found, control)
    assert result["criteria"]["auc_gain_on_primary_pool"]["result"] == "cannot be evaluated"
    assert result["overall"] != "pass"


def test_the_verdict_fails_on_each_criterion_on_its_own():
    better = rs.compare(pool(12, ARMS, {"baseline": 0.0, "H": 0.0, "H+A": 0.1}), ARMS, SHIPPED, replicates=200)
    control = {"baseline": {"auc": 0.95}, "H+A": {"auc_mean": 0.948, "auc_min": 0.94}}
    assert rs.verdict(better, better, control)["overall"] == "pass"
    lost = {"baseline": {"auc": 0.95}, "H+A": {"auc_mean": 0.93, "auc_min": 0.92}}
    assert rs.verdict(better, better, lost)["criteria"]["moisesdb_control"]["result"] == "fail"
    assert rs.verdict(better, better, lost)["overall"] == "fail"
    worse = rs.compare(pool(12, ARMS, {"baseline": 0.0, "H": 0.0, "H+A": -0.1}), ARMS, SHIPPED, replicates=200)
    result = rs.verdict(better, worse, control)
    assert result["criteria"]["no_regression_on_model_selected_pool"]["result"] == "fail"
    assert rs.verdict(worse, better, control)["criteria"]["auc_gain_on_primary_pool"]["result"] == "fail"
    # More false positives at the shipped threshold fails, even with a better AUC.
    louder = pool(12, ARMS, {"baseline": 0.0, "H": 0.0, "H+A": 0.1})
    for excerpt in louder:
        if not excerpt["guitar"]:
            excerpt["scores"]["H+A"] = np.where(np.arange(24) < 6, 0.99, excerpt["scores"]["H+A"])
    found = rs.compare(louder, ARMS, SHIPPED, replicates=200)
    result = rs.verdict(found, better, control)
    assert result["criteria"]["false_positives_at_shipped_detected_threshold"]["result"] == "fail"
    assert result["overall"] == "fail"
    assert rs.verdict(better, better, None)["criteria"]["moisesdb_control"]["result"] == "cannot be evaluated"


# --------------------------------------------------------------------- the runs, cross-checked


def fake_run(items, arm, fold, folder):
    train = rd.training_items(items, arm, fold) if arm != "baseline" else []
    held = rd.evaluation_items(items, None if arm == "baseline" else fold)
    return {
        "folder": folder,
        "arm": arm,
        "held_out_fold": None if arm == "baseline" else fold,
        "status": "complete",
        "seeds": {} if arm == "baseline" else rd.seeds(fold),
        "initial_weights": {"sha256": "abc"},
        "config": dict.fromkeys(review_summary.SAME_CONFIG, 1) | {"steps": 0 if arm == "baseline" else 500},
        "dataset": {
            "cache_manifest": {"segments_table_sha256": "table"},
            "training_ids": sorted(i["id"] for i in train),
            "scored_ids": sorted(i["id"] for i in held),
        },
    }


def test_runs_are_cross_checked_before_any_comparison():
    items = library()
    cache = {"segments_table_sha256": "table"}
    baseline = fake_run(items, "baseline", None, "baseline")
    runs = {(arm, f): fake_run(items, arm, f, f"{arm}/hold{f}") for arm in rd.ARMS for f in range(5)}
    assert review_summary.check_runs(baseline, runs, items, cache) == []

    def broken(key, change):
        copy = {k: json.loads(json.dumps(v)) for k, v in runs.items()}
        change(copy[key])
        return " | ".join(review_summary.check_runs(baseline, copy, items, cache))

    automatic = next(i["id"] for i in items if i["source"] != "human" and i["fold"] == 1)
    assert "arm H" in broken(("H", 0), lambda r: r["dataset"]["training_ids"].append(automatic))
    scored = runs[("H+A", 0)]["dataset"]["scored_ids"][0]
    assert "trained on" in broken(("H+A", 0), lambda r: r["dataset"]["training_ids"].append(scored))
    assert "seeds" in broken(("H+A", 2), lambda r: r["seeds"].update(plan=1))
    assert "starting weights" in broken(("H", 3), lambda r: r["initial_weights"].update(sha256="other"))
    assert "setting learning_rate" in broken(("H+A", 4), lambda r: r["config"].update(learning_rate=2))
    assert "steps differ" in broken(("H", 4), lambda r: r["config"].update(steps=600))
    assert "segment table" in broken(("H", 1), lambda r: r["dataset"]["cache_manifest"].update(segments_table_sha256="x"))
    assert "status" in broken(("H", 1), lambda r: r.update(status="started"))
    assert "should have" in broken(("H+A", 1), lambda r: r["dataset"]["training_ids"].pop())


def test_the_experiment_file_gives_both_arms_the_same_settings_and_trains_only_when_approved():
    import run_review_experiment as runner

    config = json.loads((Path(runner.HERE) / "review_experiment_v2.json").read_text(encoding="utf-8"))
    listed = dict(runner.commands(config))
    assert list(listed)[0] == "baseline" and list(listed)[-1] == "summary" and len(listed) == 12
    for fold in range(5):
        human, both = listed[f"H/hold{fold}"], listed[f"H+A/hold{fold}"]
        assert [a for a in human if a != "H"] == [a for a in both if a != "H+A"]  # the arm is the only difference
        assert "--owner-approved" not in human
    settings = listed["H+A/hold0"]
    for flag, value in (("--steps", "500"), ("--batch", "16"), ("--rate", "5e-05"), ("--review-share", "0.4"), ("--silence", "0.15")):
        assert settings[settings.index(flag) + 1] == value
    assert all("--owner-approved" in c for n, c in runner.commands(config, approved=True) if "/" in n)
    assert all("--dry-run" in c for _, c in runner.commands(config, dry_run=True))
    assert "--model-config" in listed["summary"]
