# Evaluation methodology

Two kinds of evaluation exist and their results are never combined. Every report states its `data_kind`.

| | Synthetic | Real corpus |
|---|---|---|
| Material | 13 procedural CC0 fixtures per split, generated in code | Recordings registered in `corpus/manifest.json`, audio local only |
| Reference | Exact synthesis positions | Dataset annotations, stem-derived activity, or reviewed human annotations (see [DATASETS.md](DATASETS.md), [ANNOTATION.md](ANNOTATION.md)) |
| Splits | `development`, `holdout` | `train`, `validation`, `test` (held out), by recording group |
| `data_kind` | `synthetic` | `real_corpus` |
| What it shows | Whether the detector solves a controlled attack-detection problem | How the engine behaves on actual music |

A synthetic score is not evidence of real-music accuracy.

## Commands

```powershell
# Synthetic (unchanged)
.venv\Scripts\python.exe -m gmv.evaluation.run --detector candidate --split development --output local-evidence/dev.json

# Real corpus: iterate on train/validation
.venv\Scripts\python.exe -m gmv.evaluation.run --detector candidate --corpus corpus/manifest.json `
  --corpus-split validation --output local-evidence/validation.json --report-md local-evidence/validation.md

# Real corpus: held-out test, only for a final evaluation
.venv\Scripts\python.exe -m gmv.evaluation.run --detector candidate --corpus corpus/manifest.json `
  --corpus-split test --final --output local-evidence/test.json
```

Options: `--dataset NAME` restricts the run to one imported dataset. `--annotator ID` uses one annotation file; the default `auto` takes the best reference tier per layer (human-reviewed, then dataset-provided, then stem-derived; two reviewed human annotators are first reduced to their consensus). `--allow-unreviewed` also admits drafts and exploratory labels and is stamped in the report; do not quote such results as measurements. `--beat-phase` enables the experimental beat tracker.

Reports go to `local-evidence/` (ignored) by default because they contain per-recording detail. Commit a report only if it contains nothing you would not publish.

## Reproducibility

Each corpus report records: code commit, detector name and source hash, analysis configuration, corpus version, corpus and annotation schema versions, a SHA-256 over the manifest and the annotation files used, the split, the annotator selection, whether unreviewed annotations were allowed, and every skipped recording with its reason (audio missing, audio changed, no reviewed annotation, annotation for a different version of the audio). A recording is never scored against audio whose hash differs from the manifest.

When nothing can be evaluated the report says `real_music_status: "pending"`.

## What is scored, per class

Only annotated layers are scored, and only inside the annotation's covered time ranges. Each class has its own entry; nothing is folded into an overall accuracy.

| Class | Reference layer | Detector output today | Measures |
|---|---|---|---|
| Full-mix onsets | `onset` | `onset` events | TP/FP/FN, precision, recall, F1, duplicates, timing bias, median and p95 absolute error at ±25 ms, ±50 ms and ±(50 ms + annotation uncertainty); algorithmic latency |
| Beats | `beat` | `beat_observed` (and `beat_prediction` when the experimental tracker is on), scored separately | F-measure at ±70 ms, continuity (CMLc/CMLt/AMLc/AMLt), estimated-to-reference tempo ratio and metrical level (same / half-time / double-time / other) |
| Downbeats | `downbeat` | none | reported as `no_detector_output` with reference counts |
| Tempo | `tempo` | per-frame tempo estimate | strict (±4%) and octave-tolerant accuracy per annotated tempo span, time to a stable estimate |
| Instrument attacks | `hits.<label>` | none per instrument | reported as `no_detector_output`, plus **recall of the full-mix onset detector on those attacks** (every annotated attack is an acoustic onset; precision is not measurable when other instruments' onsets are unannotated); see candidates below |
| Band-ratio candidates | `hits.kick`/`hits.bass`, `hits.snare`, `hits.cymbals` | `bass_transient_candidate`, `snare_like_candidate`, `high_frequency_percussion_candidate` | each candidate scored against the attacks it is compared with, plus a coincidence table showing which annotated attacks each candidate actually fires on |
| Instrument activity | `activity.<label>` | none | reported as `no_detector_output` with reference active seconds; when a detector supplies intervals: frame-level precision, recall, F1, IoU, false-positive and false-negative rates, entrance and exit timing error |

**Candidates against activity.** Where only instrument activity is known, each candidate is checked against the instruments its name suggests (for example `snare_like_candidate` against drums and snare activity). An event fired while none of them is sounding within 50 ms is *refuted*. The refuted fraction is a lower bound on false positives; an active instrument does not confirm the event, so this is not an accuracy.

**Reference tier.** Each scored class carries the `reference_kind` it was scored against, and the report summarises tiers per layer. Stem-derived activity is never presented as annotated ground truth.

The candidates are uncalibrated frequency-band ratios. Comparing them with instrument attacks measures how misleading it would be to read them as instruments; it does not make them instrument detectors.

**Uncertain labels.** A point label marked uncertain is abstained: it is not a required detection, and a detection that matches only an uncertain label is ignored rather than counted as a false positive. Uncertain activity regions are excluded from interval scoring. Abstained counts and seconds are reported.

**Not annotated versus no output.** `not_annotated` means the layer has no reference, so nothing can be said. `no_detector_output` means a reference exists and the engine produces nothing for that class. Both are shown instead of being omitted.

**Latency.** Algorithmic latency is the end of the analysis window that produced an event minus the reference time, measured offline. It excludes capture, queueing and publication, which are measured separately on live capture.

**Conditions.** Aggregate onset scores are also broken down by the recording `tags` in the manifest (for example `dense`, `sparse`, `blast_beat`).

## Rules

- Tune only on `train` and `validation`. `test` requires `--final` and the report is stamped as a final evaluation.
- Do not retune the detector against the synthetic holdout.
- Report per-class results. Do not report a single overall score.
- Report measured numbers only, with the corpus version and content hash they came from.
