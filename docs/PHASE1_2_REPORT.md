# Phase 1.2 — real-music analysis and musical-intelligence validation

Status: **in progress.** This phase builds the corpus, annotation and evaluation foundation needed to measure the engine on real music. No real-music accuracy is claimed anywhere in this report until an annotated, held-out set of authorized recordings has actually been evaluated. Synthetic results are always labelled as synthetic.

## Milestone A: repository audit and starting state

Audited at commit `a4ae9c5` (2026-11-15) on Windows 11, Ryzen 9 5950X, Python 3.14, CPU only. `main` matched the remote and the working tree was clean.

### Reproduced baseline (measured in this audit)

| Check | Result |
|---|---|
| Automated tests | 63 passed |
| Ruff lint | clean |
| Native helper build (.NET 9, Release) | 0 warnings, 0 errors |
| Synthetic development onsets, detector v3 | 565 TP / 0 FP / 0 FN, F1 1.0000 at 25 and 50 ms; median absolute error 3.22 ms |
| Synthetic holdout onsets, detector v3 | 577 TP / 0 FP / 5 FN, F1 0.9957 at 25 and 50 ms; median absolute error 3.51 ms |
| Synthetic CPU benchmark (`scripts/benchmark.py`, 60 s of audio) | 3.85 s wall (0.064 × real time), DSP p50 1.09 ms / p95 1.35 ms, 0 dropped blocks |

These match the figures reported in Phase 1.1. The holdout was only re-run to confirm reproduction; nothing was tuned against it. Capture benchmarks were not repeated here (the paired native-capture runs are from the same day; see PHASE1_1_REPORT.md Milestone 7).

### What exists for evaluation

- `gmv.evaluation.corpus`: 13 procedural CC0 fixtures per split (development / holdout), separated by seeds, tempi and patterns.
- `gmv.evaluation.metrics`: one-to-one onset matching (`mir_eval`), precision/recall/F1, duplicates, timing error; tempo accuracy per annotated span; beat F1 and continuity.
- `gmv.evaluation.run`: offline detector run over a fixture and JSON report with commit, detector hash and configuration.
- A single-file local import (`--import-manifest`) for a user-supplied recording with full-mix onsets.

### Discrepancies and gaps found

| Finding | Action in this milestone |
|---|---|
| `docs/schemas/*.json` still described schema 1.0 (no `provenance`, `evidence_kind`, `intended_use`); nothing generated or checked them | Added `gmv.schema_export` (generate / `--check`), regenerated the files, added a test that fails when they drift |
| `.gitignore` covered only `*.wav` among audio formats | Added FLAC, MP3, M4A, OGG, Opus, AIFF and the local corpus path map directory |
| `--import-manifest` handles one file, forces every label to `unknown`, does not validate `split`, takes the report's split from a CLI flag, labels the corpus `synthetic-v1`, and has no tests | Superseded by the corpus work in Milestones B–C; kept working as a legacy path |
| No way to carry instrument, beat, downbeat or tempo references for real audio; no interval metrics; no annotation tooling; nothing prevents one recording appearing in two splits | Milestones B–C |
| The engine emits full-mix onsets and three uncalibrated band-ratio candidates; it produces **no instrument activity output** | Recorded as the starting capability; evaluation will report these classes as "no detector output" rather than omit them |
| Tempo autocorrelation has octave ambiguity; beat phase is experimental and off by default; no downbeat output | Unchanged; to be measured against references before any change |
| `MusicalEvent.event_type` is a free string with no registry | Noted; not changed (no schema change is justified yet) |

No regression was found against the Phase 1.1 documentation.

## Milestone B: corpus and annotation foundation

New package `gmv.corpus` and the `gmv corpus` commands. Specification: [CORPUS.md](CORPUS.md); format and workflow: [ANNOTATION.md](ANNOTATION.md); generated schemas: [CorpusManifest.json](schemas/CorpusManifest.json), [AnnotationFile.json](schemas/AnnotationFile.json).

| Capability | Implementation |
|---|---|
| Versioned manifest with stable ids | `corpus/manifest.json`; `recording_id` is `r-` + 12 hex digits of the file SHA-256; `corpus_version` x.y.z |
| No audio or identifying metadata in Git | Manifest holds hashes, format, licence statement, permission basis, group, split, tags; file locations live in ignored `corpus-local/paths.json` |
| Registration without copying audio | `gmv corpus register` hashes the file, reads its header, records optional stems by hash |
| Recording-level splits with leakage protection | Split is a property of the group (album / session / dataset song); a manifest with one group in two splits fails to load; `assign-splits` is deterministic and never moves an existing assignment |
| Changed or missing audio | Detected by hash on `validate` and at evaluation; reported and skipped, never substituted |
| Annotation schema 1.0 | Full-mix `events` (onset, beat, downbeat, tempo change, transition, section change) kept apart from instrument `activity` intervals (overlapping) and `instrument_events` (point attacks); certainty per label; `unknown` label; declared coverage (time ranges and layers) so "not annotated" is never read as "absent" |
| Provenance | Annotator id, tool, tool version, provenance text, status, timing uncertainty, audio hash, schema version in every file |
| Local-first annotation workflow | Audacity label tracks, one file per layer; `import-labels` / `export-labels` round-trip; `review` marks an annotation usable as reference |
| Drafts from multitrack stems | `draft-from-stems` derives activity intervals from stem RMS; written as `machine_draft` and excluded from reference until a person reviews them |
| Annotator disagreement | One file per annotator; `compare` reports per-layer agreement and lists disagreements; consensus keeps only what both marked |
| Validation | `gmv corpus validate` checks manifest, leakage, annotation/audio hash agreement, label bounds, stray files and local audio |

Deviation from the plan: instead of an `--accept-stem-derived` evaluation flag, a stem draft becomes reference only through the ordinary import-and-`review` step under a human annotator id. That keeps one rule for what counts as reference.

Tests: 9 new (schema and leakage validation, interval helpers, Audacity round trip with overlap, points, uncertainty and empty layers, malformed label files, stem drafting, comparison and consensus, the full register → split → draft → export → import → review CLI flow, and changed / missing / mismatched audio reporting). All use procedurally generated audio in temporary directories.

**The committed corpus is empty.** No recording has been registered or annotated, so nothing in this milestone is a real-music result.

## Milestone C: reproducible evaluation

Methodology and commands: [EVALUATION.md](EVALUATION.md).

- `gmv.evaluation.detect.run_detector` is the single offline detector loop, now shared by the synthetic and corpus paths. After this refactor the synthetic scores are identical to Milestone A, including every predicted onset time, class score and rhythm measure (development 565/0/0, holdout 577/0/5).
- `gmv.evaluation.metrics` gained `interval_score` (frame-level activity precision/recall/F1/IoU, false-positive and false-negative rates, entrance and exit timing error, abstained seconds), `beat_score` (F-measure at ±70 ms, continuity, half/double-time diagnosis) and `drop_uncertain` (abstention for uncertain labels).
- `gmv.evaluation.real` evaluates a corpus split: `python -m gmv.evaluation.run --corpus corpus/manifest.json --corpus-split validation ...`, with JSON and optional Markdown reports. Each class is reported separately; layers without reference are `not_annotated`; annotated classes the engine does not produce are `no_detector_output`.
- Guards: the `test` split requires `--final` and is stamped; only `reviewed` annotations are used unless `--allow-unreviewed` (stamped); recordings with missing audio, changed audio, or annotations for a different version of the audio are skipped and listed with a reason. Reports carry `data_kind`, corpus version, schema versions and a content hash.
- `gmv.evaluation.synthetic_corpus` expresses a synthetic fixture in the annotation schema so the whole corpus path can be tested without real audio.
- The legacy single-file `--import-manifest` path is unchanged and now has a regression test.

### Verification performed

| Check | Result |
|---|---|
| Tests | 82 passed (63 at the start of the phase; 19 new) |
| Ruff lint | clean |
| Native helper build | 0 warnings, 0 errors (helper not changed in this phase) |
| Synthetic scores after refactor | identical to Milestone A |
| Command-line dry run on the synthetic `dense` fixture: export WAV → `corpus register` → `assign-splits` → `export-labels` → `import-labels` → `validate` → `review` → corpus evaluation | Onset TP/FP/FN and every timing error equal the direct synthetic run (174/0/0); algorithmic onset latency p50 23.1 ms, p95 33.1 ms |
| Changed / missing audio, unreviewed annotation, unknown annotator | Each skipped with its reason (tests) |
| One group in two splits | Manifest rejected (test) |
| `--corpus-split test` without `--final` | Refused |
| Committed (empty) corpus | Report says `real_music_status: "pending"` |

The dry run used a synthetic fixture registered as if it were a recording. It verifies the plumbing and is **not** a real-music result. That same run illustrates why the band-ratio candidates are not instrument detectors: on the synthetic dense fixture `high_frequency_percussion_candidate` fired on 114 onsets of which 58 were high-frequency attacks (precision 0.51).

### What the engine can and cannot be measured on today

| Class | Engine output | Status |
|---|---|---|
| Full-mix onsets | yes | measurable as soon as `onset` layers exist |
| Tempo | per-frame estimate | measurable against `tempo` layers; octave ambiguity expected |
| Beats | experimental tracker, off by default | measurable against `beat` layers; synthetic dry run found 0 of 44 beats with the tracker off |
| Downbeats | none | reference can be stored; nothing to score |
| Kick / snare / cymbal / bass attacks | three uncalibrated band-ratio candidates only | compared and cross-tabulated, not instrument detection |
| Instrument activity (drums, bass, vocals, guitars, other) | none | reference can be stored; reported as no detector output |

## Direction change: no manual annotation

After Milestone C the project owner clarified that hand-labelling recordings is not something they will do. The data strategy was revised: reference data comes from existing research datasets, their supplied annotations and their isolated stems. The Audacity workflow stays as an optional capability. Milestones A–C are kept unchanged in purpose: hashing, group-level split isolation, the held-out guard, uncertainty handling, the separation of full-mix and instrument truth, and the reports all still apply.

## Milestone D1: dataset research and selection

Full comparison, licences and access conditions: [DATASETS.md](DATASETS.md).

| Selected | Why | Reference it provides | Access |
|---|---|---|---|
| MDB Drums | Only openly obtainable real full mixes with individual drum attacks and beats | kick, snare, hi-hat, tom, cymbal attacks; beats; drum stem | public git repository, ~2 GB |
| MUSDB18-HQ | Largest confirmed metal subset with stems (12 heavy-metal tracks of 150) | drums, bass, vocals, other stems | Zenodo, 22.7 GB |
| MoisesDB | Separate kick, snare, tom, cymbal and distorted-guitar stems | stems only | web form submitted by a person |
| MedleyDB (later) | Only set with dataset-annotated instrument activity | activity annotations, stems | approved access request |

Findings that shape what can be claimed:

- **No obtainable real-music dataset annotates full-mix onsets of all instruments.** Full-mix onset precision on real music therefore has no reference. Onset *recall* can be measured on annotated drum attacks.
- **No metal-specific beat or drum-transcription dataset with obtainable audio exists.** The Harmonix Set has 32 metal tracks with beats and downbeats but distributes no audio.
- The stems-only datasets (MUSDB18, MoisesDB) give instrument *activity*, not attacks.
- MUSDB18 and MDB Drums reuse MedleyDB recordings and three artists straddle the MUSDB18 train and test sets, so splits are assigned per artist across all datasets.

## Milestone D2: dataset import and reference integration

- **Reference tiers** (annotation schema 1.1): every annotation states its `reference_kind` — `human_annotation`, `dataset_annotation`, `stem_derived`, `pseudo_label` or `model_prediction`. Only the first three can be reference; pseudo-labels and model predictions never are, so a label generated by the detector under test cannot count as ground truth. Schema 1.0 files still load.
- **Stem-derived activity is automatic.** `gmv.corpus.stems.reference` measures activity from a dataset's isolated stems, as its own tier; stems flagged with bleed are marked uncertain. This replaces the Milestone B rule that stem drafts needed human review.
- **Best tier per layer.** `gmv.corpus.compare.combine` builds one reference per recording from the best available tier of each layer and records the source of each.
- **Dataset adapters** in `gmv.corpus.datasets`: `mdbdrums`, `musdb18hq`, `moisesdb`. `gmv corpus import-dataset NAME FOLDER` hashes and registers each track and its stems, assigns the cross-dataset artist group and split, writes references and validates. `gmv corpus datasets` lists licences and access. Imports are resumable.
- **New labels:** `hi_hat` and `toms` (children of `drums`).
- **Evaluation additions:** `--dataset` filter; per-class `reference_kind` and a per-layer tier summary in every report; recall of the full-mix onset detector on annotated instrument attacks; band-ratio candidates *refuted* by instrument activity (fired while none of the suggested instruments is sounding), reported as a lower bound on false positives.

Tests: 93 passed (82 after Milestone C; 11 new for tiers, best-tier selection, the three adapters, cross-dataset artist groups, and dataset evaluation). Ruff clean.

**Not yet run against real downloads.** The adapters were written from the documented layouts and tested on miniature stand-in folders. Until a dataset has actually been imported, its adapter is unverified against the real files, and no real-music number exists.

## Status of the phase

Done: A, B, C, D1, D2. Outstanding:

- **D3 — automated evaluation.** Needs the selected datasets on disk. Downloading them requires the owner to accept each dataset's terms.
- **D4 — identify the next detection improvements** from D3's measurements.
- **Capture reliability** (USB disconnect/reconnect, default-output switch on the native backend) and **source-separation research**: not started in this phase.

Real-music analysis is **not validated**.
