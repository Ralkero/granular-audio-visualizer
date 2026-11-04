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
