# Phase 1.1 — real-world validation and musical accuracy

Starting commit: `6d30da28242b4a34b56a221cfc6513fa9cd62bc7`. The existing checkout and architecture are preserved. This report is updated at each published increment.

## Milestone 1: baseline and evaluation infrastructure

Reproduced in this run: 30 original tests pass, Ruff passes, .NET Release helper builds with zero warnings/errors. Rediscovered Windows build 26300, Python 3.14.3, G6 48 kHz/eight-channel output. Dependencies: NumPy 2.5.4, SciPy 1.18.1, Pydantic 2.14.2, Qt Essentials 6.12.0, PyAudioWPatch 0.2.12.9, pycaw 20260927, psutil 7.2.2. Earlier Phase One numbers remain historical; new captures are reported separately.

Preserved the exact baseline DSP algorithm in `gmv.evaluation.baseline_dsp` (import path adjusted only) and added 13 development plus 13 held-out original synthetic fixtures. Holdout patterns AND seeds differ. The corpus includes dense subdivisions, simultaneous bands, irregular fills, syncopation/missing beats, tempo changes, bass sustain, sustained chords, quiet/loud transitions, empty references, opposite-phase stereo and eight-channel embedding. Reference sources remain available in memory. These are synthetic frequency/percussion proxies, not proof of instrument recognition on commercial music. Fixture audio is original CC0-1.0 procedural material; no streamed music is saved.

Scoring uses pinned mir_eval 0.8.2 one-to-one matching, mixed-onset deduplication, ±25/50 ms scores, FP/FN/duplicates and signed timing errors. Per-class and negative-fixture scores stay visible. Tempo accuracy, octave tolerance, stable-estimate delay and beat continuity are separate. Empty/empty gets F1=0 and an explicit correct_empty flag rather than inflating detection aggregates. The 50-frame stability criterion assumes the default 20 ms hop; evaluator configuration is recorded.

Baseline development onset F1: **0.9029** at both tolerances, with 479 TP, 17 FP and 86 FN. Dense fixture F1 is **0.8000** and irregular fills **0.7907**; isolated bass produces nine false positives. Those development results motivate candidate work. Baseline holdout is frozen in a separate JSON file; candidate thresholds will be frozen before evaluating it. Reports include source hashes, fixture hashes, configuration and code commit.

No annotated real-music corpus is claimed. A bounded local-import workflow requires verified permission, provenance, recording identity, hash, annotation method and uncertainty. It does not download Spotify tracks. Local imports and reports containing recording metadata must remain in ignored local-evidence; split recordings by source identity, not adjacent excerpts.

Read-only soak tooling captures resources for Python and owned helper descendants, bounded whole-session reservoirs and per-second signed clock-age samples. Quantiles explicitly identify approximation/scope. Detailed capture samples and identifiers remain local. Manual capture checks are tracked in [CAPTURE_VALIDATION.md](CAPTURE_VALIDATION.md).

Reproduce:
```powershell
uv sync --locked --extra dev --extra evaluation
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m gmv.evaluation.run --detector baseline --split development --output local-evidence\baseline-development.json
.venv\Scripts\python.exe -m gmv.evaluation.run --detector baseline --split holdout --output local-evidence\baseline-holdout.json
.venv\Scripts\python.exe -m gmv.evaluation.run --detector baseline --split development --output local-evidence\fixtures-report.json --export-fixtures local-evidence\fixtures
.venv\Scripts\python.exe scripts\validate_capture.py --seconds 600 --output local-evidence\soak.json
```

Next increment: complete available Spotify observations and fault-injection tests; then improve the detector using development material only. No renderer or automatic live stem recording is added.

## Milestone 2: capture validation and targeted fixes

Observed on Windows: Spotify process-tree capture plus simultaneous default system capture; pause/resume while the browser remained audible; user-reported seek and skip; actual process-root replacement after user quit/reopened Spotify, followed by activity in a new segment. Starting a fresh process capture while paused produced zero signal until resumption at approximately 20.3 seconds. No audio or stems were saved. Three real start/stop cycles for each source ended without capture threads or owned child processes.

The 600.03-second paired baseline produced 29,812 process and 29,966 system feature frames, zero queue/subscriber/logger drops, and clean shutdown. Whole-session approximate p95 DSP was 2.10 / 2.35 ms (process / system); estimated capture-center age was 30.25 / 368.85 ms. Parent plus owned helper peak sampled RSS was 120.28 MiB; p95 CPU 40.3% of one core. UI off, logging on, advanced off; developer activity and user playback transitions occurred during the run. These are workload observations, not a controlled before/after comparison. Seven process discontinuity resets occurred during the user transition sequence; they are not counted as queue drops or assumed to be only the restart.

**Failed timing goal:** system fallback sample-clock age grew from roughly 28 ms to a maximum of 409 ms across ten minutes, despite zero reported callback/queue drops. Its driver supplied no usable ADC timestamp. Cause remains unresolved (clock-rate mismatch, callback buffering and unreported loss are not distinguished). Do not treat this clock as precise synchronization or claim the 50 ms goal. Signed discrepancies remain visible and values exceeding 100 ms tag timestamp quality with `drift_warning`; no rebasing conceals the error. Process QPC timing did not show this trend. Physical sound-to-display latency is unmeasured.

Fixes: recovered analysis now returns from discontinuity status to capturing; explicit output selection binds a unique endpoint identity and refuses replacement by another endpoint sharing the name; startup failure cleanup always signals completion even if finalization also fails. Diagnostics include owned-helper resources and separate queue/logger/native counters. Fault injection covers logger write failure, cleanup failure, helper launch failure, ambiguous restart roots and repeated synthetic lifecycle. **38 tests and Ruff pass.** Physical USB/default-device changes remain unrun; existing simulated checks do not substitute for them.

Reproduce lifecycle with `python scripts/lifecycle_probe.py --pid PID` and without `--pid` for system audio. Use the capture matrix for user-assisted checks. Next: development-only detector improvements, followed by frozen holdout evaluation and a second ten-minute advanced-enabled run.

## Milestone 3: measured musical-analysis changes

Promoted mixed-audio attack v2: fixed 45 ms suppression combined with an independent positive energy-rise gate (8% of current analysis RMS), retaining adaptive spectral-flux evidence. This does not simply remove the 90 ms guard. Transient strength now reports energy rise; onset strength retains flux, and seven band-flux measurements are exposed. Candidate labels remain uncalibrated frequency heuristics; simultaneous observations retain their onset group.

One development candidate was evaluated, then frozen before one holdout evaluation. Development F1 rose 0.9029 to 0.9841, with FP 17 to 9 and FN 86 to 9. Holdout F1 rose **0.8801 to 0.9825** at ±50 ms and **0.8764 to 0.9825** at ±25 ms: TP 562, FP 0, FN 20, duplicates 0. Dense holdout F1 rose 0.7692 to 0.9970; irregular fills 0.7354 to 0.9218. Holdout signed timing bias remains **-12.03 ms** (window-center convention), median absolute error 14.95 ms, p95 19.42 ms. Timing errors are distinct from publication latency. Nine development sustained-bass false positives remain. No real-music accuracy is established. Full per-fixture/per-class JSON and [holdout comparison](evaluation/holdout_comparison.md) retain failures.

A circular-phase prototype distinguishes observed phase-aligned onsets from explicitly predicted future beats, requires coherent recent evidence and unlocks after evidence expires. **Not promoted to default:** holdout observed beat F1 on dense / syncopation / tempo-change fixtures was 0 / 0 / 0.7297, versus baseline candidate-beat scores 0.3281 / 0.6154 / 0.8941. Abstention prevents labeling every onset a beat, but recall is worse. The unchanged autocorrelation tempo estimator fails strict dense tempo and selects half tempo on syncopation. Tempo-change strict accuracy after four seconds is 1.0 then 0.7262; stable estimate times 4.00 and 5.78 seconds. Phase tracking is opt-in via `experimental_beat_phase=True`; default produces no beat claims. Candidate JSON measured the enabled prototype before this default-off gate was added; no detector thresholds were retuned after holdout. Reproduce that experiment with `--beat-phase`.

Raw RMS/band power retain channel-mean semantics. Separate `analysis_rms` uses fixed nonnegative channel-power coefficients: mono 1, otherwise 0.5 per channel, or explicit configured coefficients. Stereo embedded in eight channels has the same analysis RMS and half raw RMS; genuine eight-channel energy remains represented. Opposite-phase channels cannot cancel. Coefficients and versioned semantics travel in each frame. No dynamic quiet-channel selection occurs.

Schema 1.1 adds evidence clocks, segments, provenance, intended use and confidence interpretation. Strict 1.0 migration and a synthetic legacy fixture preserve replay. Replays retain original publication time and are excluded from live-impact eligibility; delayed/unknown/foreign-segment evidence also fails closed. The baseline DSP algorithm is unchanged; formatting and current-model defaults do not alter its computations.

**44 tests and Ruff pass**, including dense attacks, fixed-power layout, signed age, legacy parsing and stale-event rejection. Onset gains justify promoting the energy-gated detector; they do not justify promoting beat prediction or claiming instrument identities. Matched performance comparisons and advanced-on capture follow.

Reproduce: `python -m gmv.evaluation.run --detector candidate --split development --beat-phase --output local-evidence/candidate-dev.json`, then `--split holdout` without tuning against its output. Compare with `python -m gmv.evaluation.compare docs/evaluation/baseline_holdout.json docs/evaluation/candidate_holdout.json --output local-evidence/comparison.md`. Next: bounded CPU separation experiment.

## Milestone 4: real isolated CPU separation benchmark

Executed HTDemucs 4.1.0 / PyTorch 2.9.1+cpu on the Ryzen 9 5950X, Python 3.12.9 in a separate environment and owned subprocess tree. Exact weights, dependency lock, bounded setup, supervisor and reproduction commands are in [experiments/separation](../experiments/separation/README.md). Code is MIT; upstream says weight licensing is unresolved. No weights, live audio or separated audio are published.

New-process model load took **0.334 s** excluding imports (OS disk cache not cleared). Five six-second synthetic contexts took **5.088–5.296 s** each, offline RTF **0.848–0.883**. With one-second overlap, each job emits only five seconds of new audio: RTF **1.018–1.059**, so continuous processing would accumulate backlog. The earliest samples require six seconds of context plus inference: **11.09–11.30 s** old at completion. Peak sampled worker-tree RSS **1,388.93 MiB**, mean sampled CPU **350.29% of one core**, total experiment wall time **31.04 s**. Model segment is 7.8 seconds with padding; no real future context is invented. The resource monitor was corrected to include the Windows launcher child; an earlier launcher-only measurement was rejected and rerun.

Known synthetic references show failure on bass/drum assignment: bass SI-SDR **-32.78 to -25.58 dB**, drums **-5.62 to -4.87 dB** on attack mixtures. Much synthetic bass was assigned to drums. Sustained harmony assigned to other reached **40.89 dB**, which cannot compensate for failed attack separation. With no vocal reference present, estimated vocal power was **0.00098%–0.17956%** of mix power. Drums cross-correlation peaks aligned at 0 ms; tiny bass estimates gave weak best-lag readings about 0.57–1.32 ms. Shared one-second output disagreement RMS was 0.000935 drums, 0.000203 bass, 0.000498 other, 0.000198 vocals. Full per-stem projection leakage/absent-source scores are in [separation_cpu.json](evaluation/separation_cpu.json); these are synthetic, not real-music quality claims.

Decision: retain a bounded **offline diagnostic**, with no live model adapter enabled. Throughput with overlap, multi-second delay, poor synthetic attack separation and unresolved distribution terms do not justify integration. Optional HPSS remains explicitly a proxy. The concurrent advanced-on capture measures resource contention separately in Milestone 5. Tests validate checksum rejection without downloading weights; the actual model benchmark completed and left no owned worker. Next: finalize delayed-result invalidation, UI diagnostics, performance and readiness documentation.
