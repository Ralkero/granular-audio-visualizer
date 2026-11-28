"""A5.2 Step 1, S1.7: the timing harness, on synthetic records and sleep-only toy jobs.

No performance is measured. Expected values are computed independently in each test from
explicit synthetic fixtures; none of the earlier illustrative figures is reused.
"""

import dataclasses
import os
import random
import subprocess
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


@pytest.mark.skipif(not hasattr(os, "wait4"), reason="the platform does not report a child's CPU time")
def test_cpu_time_is_recorded_separately_and_used_by_no_formula(tmp_path):
    marks = tmp_path / "marks.json"
    # one thread: with many, idle numeric-library threads can spin, and CPU time then exceeds
    # elapsed time (observed: 2.6 s of CPU in 0.55 s), which is why no formula uses it
    r = timing.run_job("toy", [sys.executable, str(TOY), str(marks), "0.2"], marks, "outer", units=10,
                       predictions=2, env=timing.thread_environment(1))
    assert r.cpu_s is not None and r.cpu_s >= 0.045  # at least the busy loop
    assert r.cpu_s < r.elapsed_s  # with one thread, the sleeps (over 0.3 s) cost elapsed time, not CPU time
    without = dataclasses.replace(r, cpu_s=None)
    assert timing.components(r) == timing.components(without) and timing.accounting(r) == timing.accounting(without)
    assert record([3.0, 2.0]).cpu_s is None  # synthetic records carry none


def test_a_failing_or_overrunning_job_stops(tmp_path):
    marks = tmp_path / "marks.json"
    with pytest.raises(subprocess.CalledProcessError):
        timing.run_job("fail", [sys.executable, "-c", "raise SystemExit(4)"], marks, "outer", 1, 0)
    with pytest.raises(subprocess.TimeoutExpired):
        timing.run_job("slow", [sys.executable, "-c", "import time; time.sleep(30)"], marks, "outer", 1, 0, timeout=0.5)


def probe(component, size_class, configuration, m, repetition, seconds):
    return timing.Probe(component, size_class, configuration, m, repetition, seconds)


def test_contention_factors_follow_their_definition_on_synthetic_probes():
    probes = [
        probe("T", "outer", "solo", 0, 0, 10.0), probe("T", "outer", "solo", 0, 1, 12.0),  # solo mean 11
        probe("T", "outer", "T+T", 2, 0, 15.0), probe("T", "outer", "T+T", 2, 1, 17.0),  # mean 16
        probe("T", "outer", "S+P+T", 3, 0, 20.0), probe("T", "outer", "S+P+T", 3, 1, 24.0),  # mean 22
        probe("T", "inner", "solo", 0, 0, 4.0), probe("T", "inner", "T", 1, 0, 5.0),
        probe("S", "final", "solo", 0, 0, 2.0), probe("S", "final", "T+T+T", 3, 0, 3.0),
        probe("P", None, "solo", 0, 0, 0.5), probe("P", None, "P+P", 2, 0, 0.6), probe("P", None, "S", 1, 0, 0.55),
    ]
    factors, ratios = timing.contention_factors(probes)
    assert factors["T"] == pytest.approx({"outer": 22 / 11, "inner": 5 / 4})
    assert factors["S"] == pytest.approx({"final": 1.5}) and factors["P"] == pytest.approx(1.2)
    expected = [(16 / 11, 2), (2.0, 3), (1.25, 1), (1.5, 3), (1.2, 2), (1.1, 1)]
    assert [(round(r, 9), m) for r, m in sorted(ratios)] == [(round(r, 9), m) for r, m in sorted(expected)]
    # T2's sustained 4-way slowdown sets a floor under c_T(outer) only
    raised, _ = timing.contention_factors(probes, sustained=(5.0, 12.5))
    assert raised["T"] == pytest.approx({"outer": 2.5, "inner": 1.25}) and raised["S"] == factors["S"]
    job = timing.Job("outer", 150, 30, 9)
    m = timing.Measured(S={"outer": 10.0}, d1={"outer": 8.0}, k={"outer": 0.04}, p=0.2)
    s, t, q = timing.split(job, m)
    plan_factors = {"S": {"outer": 1.4}, "T": raised["T"], "P": raised["P"]}
    assert timing.estimate(job, m, plan_factors) == pytest.approx(1.4 * s + 2.5 * t + 1.2 * q)


def test_contention_factors_refuse_incomplete_or_malformed_probes():
    with pytest.raises(ValueError):
        timing.contention_factors([probe("T", "outer", "T+T", 2, 0, 15.0)])  # no solo time
    for bad in (probe("X", "outer", "solo", 0, 0, 1.0), probe("T", "outer", "solo", 0, 0, 0.0),
                probe("T", "outer", "solo", 0, 0, float("nan"))):
        with pytest.raises(ValueError):
            timing.contention_factors([bad])


def test_the_cross_check_stop_applies_only_with_a_given_tolerance():
    pairs = [(1.00, 1.02), (0.040, 0.050)]  # (synthetic, real): 2% and 20% apart
    assert timing.stops(cross_checks=pairs) == []  # the tolerance is pending: no default
    assert timing.stops(cross_checks=pairs, cross_check_tolerance=0.25) == []
    reason = "synthetic and real timing at the same size differ by more than the cross-check tolerance"
    assert timing.stops(cross_checks=pairs, cross_check_tolerance=0.10) == [reason]
    assert timing.stops(cross_checks=[(1.1, 1.0)], cross_check_tolerance=0.10) == []  # exactly at the tolerance
    assert timing.stops(cross_checks=[(0.5, 1.0)], cross_check_tolerance=0.10) == [reason]  # either direction
