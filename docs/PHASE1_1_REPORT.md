# Phase 1.1 â€” real-world validation and musical accuracy

Starting commit: `6d30da28242b4a34b56a221cfc6513fa9cd62bc7`. The existing checkout and architecture are preserved. This report is updated at each published increment.

## Milestone 1: baseline and evaluation infrastructure

Reproduced in this run: 30 original tests pass, Ruff passes, .NET Release helper builds with zero warnings/errors. Rediscovered Windows build 26300, Python 3.14.3, G6 48 kHz/eight-channel output. Dependencies: NumPy 2.5.4, SciPy 1.18.1, Pydantic 2.14.2, Qt Essentials 6.12.0, PyAudioWPatch 0.2.12.9, pycaw 20260927, psutil 7.2.2. Earlier Phase One numbers remain historical; new captures are reported separately.

Preserved the exact baseline DSP algorithm in `gmv.evaluation.baseline_dsp` (import path adjusted only) and added 13 development plus 13 held-out original synthetic fixtures. Holdout patterns AND seeds differ. The corpus includes dense subdivisions, simultaneous bands, irregular fills, syncopation/missing beats, tempo changes, bass sustain, sustained chords, quiet/loud transitions, empty references, opposite-phase stereo and eight-channel embedding. Reference sources remain available in memory. These are synthetic frequency/percussion proxies, not proof of instrument recognition on commercial music. Fixture audio is original CC0-1.0 procedural material; no streamed music is saved.

Scoring uses pinned mir_eval 0.8.2 one-to-one matching, mixed-onset deduplication, Â±25/50 ms scores, FP/FN/duplicates and signed timing errors. Per-class and negative-fixture scores stay visible. Tempo accuracy, octave tolerance, stable-estimate delay and beat continuity are separate. Empty/empty gets F1=0 and an explicit correct_empty flag rather than inflating detection aggregates. The 50-frame stability criterion assumes the default 20 ms hop; evaluator configuration is recorded.

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
