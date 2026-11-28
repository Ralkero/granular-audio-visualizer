"""Step 4, S4.6: the training-length gate, the 16 h gate and the caps (protocol §15; Amendment 3,
A3.1, A3.2 B18 with S3 and B19, A3.8 B11 and V1-V6). Built and tested on generated values only;
the real values come from step 5's gate runs.

**The matched complete-recipe rule** (A3.2), per arm a ∈ {C, B1}, with E (30 or 60), E⁺ = 1.5E and
E⁺⁺ = ⌈2.25E⌉, sizes z ∈ {outer, inner} and seeds s ∈ {0, 1}:

* ``Run``: one complete recipe's records: its horizon, the final gate loss L(H), its per-epoch
  gate losses (for r), and whether every loss, parameter and gradient stayed finite and its
  replay and matched-start checks passed.
* Δ_{z,s} = (L(E) − L(E⁺)) / L(E); ε_z = |L_{z,0}(E) − L_{z,1}(E)| / ½(L_{z,0}(E) + L_{z,1}(E));
  β_z = max(ε_z, δ) with δ = 0. EXT_z: both seeds have Δ > β_z; S4_z: both seeds have Δ < −β_z.
  Stage 2 uses the same with (E⁺, E⁺⁺).
* r = (m₁ − m₂)/m₁ over the means of the last two 5-epoch windows of one run.
* Checks on every run: S1 non-finite → stop; S2 any L ≤ 0, m₁ ≤ 0 or a non-positive two-seed
  mean → stop; S5 a failed replay or matched-start check → stop; S3 both seeds of one size and
  horizon with r < −0.05 → stop, and any single run with r < −0.02 is a warning.
* Stage 1 (E, E⁺): S4 at either size → stop; otherwise EXT at any size → stage 2; otherwise
  adopt E. Stage 2 (E⁺⁺): S4′ at either size → stop; otherwise EXT′ at any size → stop (the
  confirmation failed); otherwise adopt E⁺.
* A stop always means the owner's review; nothing is selected automatically.

**The 16 h gate** (A3.1): H_proc for C under policy C above 57,600 s → policy D for both arms.
The **policy-flip safeguard**: if adopting C's E⁺ is what moves the gate from C to D, stop for the
owner's review before Stage 3.

**Caps** (A3.8), exec to exit: C 0.1 h per epoch, B1 1/30 h per epoch; 4/3 of the solo cap for
jobs running beside others by design; 1 h for a T3 probe or a rehearsal run; inner-size runs use
the outer caps. **Timing stops** use the registered values (V1 0.15, V2 5%, the per-count slowdown
limit, V3b 20%).
"""

import math
from dataclasses import dataclass

from . import timing

LENGTHS = {"C": (30, 45, 68), "B1": (60, 90, 135)}
SIZES = ("outer", "inner")
SEEDS = (0, 1)
DELTA = 0.0
S3_STOP, S3_WARNING = -0.05, -0.02
WINDOW = 5
GATE_SECONDS = 57_600  # 16 h of process-time
CV_THRESHOLD, DRIFT_THRESHOLD, CROSS_CHECK_TOLERANCE = 0.15, 0.05, 0.20
CAP_PER_EPOCH_S = {"C": 360.0, "B1": 120.0}  # 0.1 h and 1/30 h
CONCURRENT_FACTOR = 4 / 3
PROBE_CAP_S = 3600.0


@dataclass(frozen=True)
class Run:
    arm: str
    size: str
    seed: int
    horizon: int
    loss: float  # L(H)
    epoch_losses: tuple  # the gate loss after each epoch
    finite: bool = True  # S1
    checks_passed: bool = True  # S5: replay and matched-start


@dataclass(frozen=True)
class Verdict:
    outcome: str  # "adopt", "stage 2" or "stop"
    epochs: int | None
    reasons: tuple  # fixed texts
    warnings: tuple


def r_of(epoch_losses):
    """r = (m₁ − m₂)/m₁ over the last two 5-epoch windows."""
    losses = list(epoch_losses)
    if len(losses) < 2 * WINDOW:
        raise ValueError("r needs at least two 5-epoch windows")
    m1 = sum(losses[-2 * WINDOW : -WINDOW]) / WINDOW
    m2 = sum(losses[-WINDOW:]) / WINDOW
    return m1, m2, (m1 - m2) / m1 if m1 > 0 else None


def _index(runs):
    out = {}
    for run in runs:
        key = (run.size, run.seed, run.horizon)
        if key in out:
            raise ValueError("two gate runs share a size, seed and horizon")
        out[key] = run
    return out


def _run_checks(runs):
    """S1, S2 (per run), S5 and S3; returns (stop reasons, warnings)."""
    stops, warnings = [], []
    for run in runs:
        if not run.finite or not all(math.isfinite(x) for x in (run.loss, *run.epoch_losses)):
            stops.append("S1: a loss, parameter or gradient was not finite")
            continue
        if not run.checks_passed:
            stops.append("S5: a replay or matched-start check failed")
        if run.loss <= 0:
            stops.append("S2: a gate loss is not positive")
        m1, _m2, r = r_of(run.epoch_losses)
        if m1 <= 0:
            stops.append("S2: a window mean is not positive")
        elif r < S3_WARNING:
            warnings.append("S3: a single run's loss rose by more than 2% over its last window")
    index = _index(runs)
    for size in SIZES:
        for horizon in {run.horizon for run in runs}:
            pair = [index.get((size, s, horizon)) for s in SEEDS]
            if all(pair) and all(p.finite for p in pair):
                rs = [r_of(p.epoch_losses)[2] for p in pair]
                if all(r is not None and r < S3_STOP for r in rs):
                    stops.append("S3: both seeds' losses rose by more than 5% over their last window")
    return sorted(set(stops)), sorted(set(warnings))


def _deltas(index, size, short, long):
    pair = [(index[(size, s, short)], index[(size, s, long)]) for s in SEEDS]
    a, b = pair[0][0].loss, pair[1][0].loss
    mean = (a + b) / 2
    if mean <= 0:
        return None
    epsilon = abs(a - b) / mean
    beta = max(epsilon, DELTA)
    deltas = [(x.loss - y.loss) / x.loss for x, y in pair]
    return deltas, beta


def _stage(index, short, long):
    """(stop reasons, extended sizes) for one stage."""
    reasons, extended = [], []
    for size in SIZES:
        result = _deltas(index, size, short, long)
        if result is None:
            reasons.append("S2: a two-seed mean is not positive")
            continue
        deltas, beta = result
        if all(d < -beta for d in deltas):
            reasons.append("S4: both seeds at one size got materially worse with the longer recipe")
        elif all(d > beta for d in deltas):
            extended.append(size)
    return reasons, extended


def decide(arm, runs):
    """The training-length decision for one arm from its gate runs (stage 1 always; stage 2 when
    its runs are present)."""
    e, e1, e2 = LENGTHS[arm]
    runs = [r for r in runs if r.arm == arm]
    index = _index(runs)
    stage1 = [(z, s, h) for z in SIZES for s in SEEDS for h in (e, e1)]
    if any(k not in index for k in stage1):
        raise ValueError("stage 1 needs E and E⁺ at both sizes and both seeds")
    stops, warnings = _run_checks([index[k] for k in stage1])
    if stops:
        return Verdict("stop", None, tuple(stops), tuple(warnings))
    reasons, extended = _stage(index, e, e1)
    if reasons:
        return Verdict("stop", None, tuple(sorted(set(reasons))), tuple(warnings))
    if not extended:
        return Verdict("adopt", e, (), tuple(warnings))
    stage2 = [(z, s, e2) for z in SIZES for s in SEEDS]
    if any(k not in index for k in stage2):
        return Verdict("stage 2", None, ("EXT: the longer recipe was materially better at one size or more",),
                       tuple(warnings))
    stops2, warnings2 = _run_checks([index[k] for k in stage1 + stage2])
    warnings = tuple(sorted(set(warnings) | set(warnings2)))
    if stops2:
        return Verdict("stop", None, tuple(stops2), warnings)
    reasons, extended = _stage(index, e1, e2)
    if reasons:
        return Verdict("stop", None, tuple(sorted(set(reasons))), warnings)
    if extended:
        return Verdict("stop", None, ("EXT′: the confirmation failed: E⁺⁺ was still materially better",), warnings)
    return Verdict("adopt", e1, (), warnings)


def policy(h_proc_seconds):
    """The 16 h gate (A3.1)."""
    if not math.isfinite(h_proc_seconds) or h_proc_seconds < 0:
        raise ValueError("H_proc must be a finite, non-negative number of seconds")
    return "D" if h_proc_seconds > GATE_SECONDS else "C"


def policy_with_safeguard(h_proc_at_e, h_proc_adopted, adopted_epochs):
    """(policy, stop reason or None): the policy-flip safeguard of A3.2."""
    before, after = policy(h_proc_at_e), policy(h_proc_adopted)
    if adopted_epochs != LENGTHS["C"][0] and before == "C" and after == "D":
        return after, "adopting E⁺ moves the 16 h gate from policy C to policy D: owner review before Stage 3"
    return after, None


def h_proc_c(sizes, measured, epochs, preprocessing_s):
    """H_proc for the C arm under policy C (A3.1, A3.8 B8): ``sizes`` as ``timing.plan_jobs``."""
    jobs = timing.plan_jobs(sizes, {"inner": 2, "outer": 5, "final": 5}, epochs)
    return timing.h_proc(jobs, measured, preprocessing_s)


def cap_seconds(arm, epochs, concurrent=False):
    """A job's exec-to-exit cap (inner-size runs use the same, outer, caps)."""
    cap = CAP_PER_EPOCH_S[arm] * epochs
    return cap * CONCURRENT_FACTOR if concurrent else cap


def timing_stops(**measurements):
    """``timing.stops`` with the registered thresholds."""
    return timing.stops(cv_threshold=CV_THRESHOLD, drift_threshold=DRIFT_THRESHOLD, per_count_limit=True,
                        cross_check_tolerance=CROSS_CHECK_TOLERANCE, **measurements)
