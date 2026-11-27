"""A5.2 Step 1, S1.7: the timing harness, on synthetic records and sleep-only toy jobs.

No performance is measured. Expected values are computed independently in each test from
explicit synthetic fixtures; none of the earlier illustrative figures is reused.
"""

import random
import sys
from pathlib import Path

import pytest

from gmv.character.research.presence_model import timing

TOY = Path(__file__).with_name("timing_toy_job.py")


def record(epochs, setup=1.0, startup=0.5, teardown=0.25, prediction=2.0, gap=0.0, size_class="outer",
           units=100, predictions=10, gate_eval=0.0):
    """A synthetic JobRecord (nanosecond marks) with the given phase durations in seconds."""
    t = 0.0
    intervals = []

    def add(label, seconds):
        nonlocal t
        intervals.append((label, round(t * 1e9), round((t + seconds) * 1e9)))
        t += seconds

    t = startup
    add("setup", setup)
    t += gap
    for e in epochs:
        add("epoch", e)
    if gate_eval:
        add("gate_eval", gate_eval)
    add("prediction", prediction)
    return timing.JobRecord("job", size_class, units, predictions, 0, round((t + teardown) * 1e9), tuple(intervals))


def test_every_second_from_exec_to_exit_is_accounted_for():
    r = record([3.0, 2.0, 2.0], gap=0.4, gate_eval=0.6)
    a = timing.accounting(r)
    total = a["startup"] + a["setup"] + sum(a["epochs"]) + a["gate_eval"] + a["prediction"] + a["other"] + a["teardown"]
    assert total == pytest.approx(r.elapsed_s, abs=1e-9)
    assert a["other"] == pytest.approx(0.4) and a["gate_eval"] == pytest.approx(0.6)
    assert not timing.complete(r, max_other_s=0.1) and timing.complete(record([3.0, 2.0]), max_other_s=0.1)


def test_components_treat_epoch_1_separately_and_exclude_gate_evaluation():
    s, d1, rest, q = timing.components(record([3.0, 2.0, 2.2], gap=0.1, gate_eval=0.6))
    assert s == pytest.approx(0.5 + 1.0 + 0.25 + 0.1)
    assert d1 == 3.0 and rest == pytest.approx([2.0, 2.2]) and q == 2.0


def test_the_projection_is_s_plus_d1_plus_e_minus_1_k_u_plus_p_p():
    records = [record([3.0, 2.0, 2.0], units=100, predictions=10), record([2.5, 1.5, 1.7], units=80, size_class="inner")]
    m = timing.summarise(records)
    assert m.k == pytest.approx({"outer": 0.02, "inner": 1.6 / 80}) and m.d1 == {"outer": 3.0, "inner": 2.5}
    job = timing.Job("outer", units=150, epochs=30, predictions=9)
    expected = m.S["outer"] + 3.0 + 29 * 0.02 * 150 + (2.0 / 10) * 9
    assert timing.process_time(job, m) == pytest.approx(expected)
    # the old form E·k·U would miss epoch 1's extra second here
    assert timing.process_time(job, m) - (m.S["outer"] + 30 * 0.02 * 150 + 0.2 * 9) == pytest.approx(3.0 - 0.02 * 150)


def test_h_proc_over_a_synthetic_plan_matches_a_closed_form():
    m = timing.Measured(S={"inner": 8.0, "outer": 10.0, "final": 11.0}, d1={"inner": 7.0, "outer": 8.0, "final": 9.0},
                        k={"inner": 0.04, "outer": 0.04, "final": 0.04}, p=0.2)
    sizes = {"inner": [(120, 30)] * 95, "outer": [(150, 9)] * 19, "final": [(160, 8)]}
    jobs = timing.plan_jobs(sizes, {"inner": 2, "outer": 5, "final": 5}, epochs=30)
    assert len(jobs) == 190 + 95 + 5
    closed = (190 * (8 + 7 + 29 * 0.04 * 120 + 0.2 * 30) + 95 * (10 + 8 + 29 * 0.04 * 150 + 0.2 * 9)
              + 5 * (11 + 9 + 29 * 0.04 * 160 + 0.2 * 8))
    assert timing.h_proc(jobs, m, preprocessing_s=600) == pytest.approx(600 + closed)
    d_jobs = timing.plan_jobs(sizes, {"inner": 1, "outer": 3, "final": 3}, epochs=30)
    assert len(d_jobs) == 95 + 57 + 3


def test_the_estimate_applies_a_factor_per_component():
    m = timing.Measured(S={"outer": 10.0}, d1={"outer": 8.0}, k={"outer": 0.04}, p=0.2)
    job = timing.Job("outer", 150, 30, 9)
    s, t, q = timing.split(job, m)
    assert s + t + q == pytest.approx(timing.process_time(job, m))
    factors = {"S": {"outer": 2.0}, "T": {"outer": 1.25}, "P": 3.0}
    assert timing.estimate(job, m, factors) == pytest.approx(2.0 * s + 1.25 * t + 3.0 * q)


def test_list_scheduling_with_fixed_durations_respects_the_classical_bound():
    rng = random.Random(1)
    for _ in range(500):
        durations = [rng.uniform(0.1, 10) for _ in range(rng.randint(1, 40))]
        for longest_first in (True, False):
            makespan = timing.list_schedule(durations, longest_first=longest_first)
            assert makespan <= timing.h_plan(durations) + 1e-9
            assert makespan >= max(max(durations), sum(durations) / 4) - 1e-9


def test_contention_dependent_durations_are_simulated_directly():
    """Each job runs at 1/slowdown(n) speed while n jobs run. With slowdown at most c, every
    realised duration stays below c times the solo time in this simulation; with a model
    that slows beyond the factor used for the estimate, the estimate is exceeded. This shows
    behaviour, it does not prove a bound."""
    rng = random.Random(2)
    work = [rng.uniform(1, 5) for _ in range(30)]
    gentle = {1: 1.0, 2: 1.05, 3: 1.15, 4: 1.3}
    realised, makespan = timing.contention_schedule(work, gentle.__getitem__)
    estimated = [1.3 * w for w in work]
    assert all(r <= e + 1e-9 for r, e in zip(realised, estimated, strict=True))
    assert makespan <= timing.h_plan(estimated) + 1e-9
    assert not timing.stops(realised=realised, estimated=estimated, makespan=makespan, plan=timing.h_plan(estimated))
    harsh = {1: 1.0, 2: 1.5, 3: 2.5, 4: 3.5}
    realised, makespan = timing.contention_schedule(work, harsh.__getitem__)
    assert any(r > e for r, e in zip(realised, estimated, strict=True))
    assert "a job took longer than its estimate" in timing.stops(realised=realised, estimated=estimated)


def test_throughput_diagnostics_from_synthetic_epoch_logs():
    single = [6.0] + [5.0] * 29
    concurrent = [[7.0] + [6.0] * 29, [7.0] + [6.0] * 10 + [6.3] * 9 + [6.6] * 10, [6.5] * 30, [6.2] * 30]
    t = timing.throughput(single, concurrent)
    assert t["tau1"] == 5.0
    assert t["tau4"] == 6.5  # the slowest process over epochs 11-30 (the third)
    assert t["phi"] == pytest.approx(4 * 5.0 / t["tau4"])
    expected_drift = 100 * (6.6 - (6.0 + 9 * 6.3) / 10) / ((6.0 + 9 * 6.3) / 10)
    assert t["drift_percent"][1] == pytest.approx(expected_drift) and t["drift_percent"][2] == 0
    assert timing.cv([9.0, 2.0, 2.0, 2.0]) == 0.0


def test_stop_rules_use_given_thresholds_only():
    assert timing.stops(ratios=[(4.2, 3)]) == ["a slowdown exceeds 4: four at once would be slower than one at a time"]
    assert timing.stops(ratios=[(2.5, 1)]) == []  # the m + 1 rule is a recommendation, off unless asked
    assert timing.stops(ratios=[(2.5, 1)], per_count_limit=True) == ["a slowdown exceeds its co-runner count plus one"]
    assert timing.stops(cvs=[0.3], drifts=[8.0]) == []  # pending thresholds: no default
    assert len(timing.stops(cvs=[0.3], drifts=[8.0], cv_threshold=0.15, drift_threshold=5.0)) == 2


@pytest.mark.parametrize(("gap", "complete"), ((0.0, True), (0.2, False)))
def test_a_toy_job_is_timed_from_exec_to_exit(tmp_path, gap, complete):
    marks = tmp_path / "marks.json"
    r = timing.run_job("toy", [sys.executable, str(TOY), str(marks), str(gap)], marks, "outer", units=10,
                       predictions=2, env=timing.thread_environment())
    a = timing.accounting(r)
    assert len(a["epochs"]) == 4 and a["startup"] > 0 and a["teardown"] >= 0
    assert timing.complete(r, max_other_s=0.05) is complete
    total = a["startup"] + a["setup"] + sum(a["epochs"]) + a["gate_eval"] + a["prediction"] + a["other"] + a["teardown"]
    assert total == pytest.approx(r.elapsed_s, abs=1e-6)
