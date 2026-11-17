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

The `mdbdrums` and `musdb18hq` adapters have since been run against the real datasets (Milestone D3). The `moisesdb` adapter has only been tested on a miniature stand-in folder and is unverified against the real files.

## Milestone D3: first automated real-music evaluation (MDB Drums)

The owner approved downloading MDB Drums (CC BY-NC-SA 4.0). It was cloned to a local folder outside the repository, imported with one command and evaluated with no manual labelling. Dataset-derived manifests and annotations are kept in the ignored `corpus-data/` directory, not committed.

| | |
|---|---|
| Material evaluated | 23 real full mixes, 21.8 minutes, 44.1 kHz stereo |
| Genres | rock, punk, grunge, one speed-metal excerpt, jazz styles, reggae, disco, gospel, country |
| Reference: drum attacks | 7,898 distinct attack times (kick 1,539; snare 2,654; hi-hat 2,639; cymbals 1,002; toms 90) — `dataset_annotation` |
| Reference: beats | 2,694 beats, with downbeats — `dataset_annotation` |
| Reference: drums activity | 1,231.8 s active, from the drum-only stem — `stem_derived` |
| Import | 23 imported, 0 failed, validation clean; the real layout and file formats matched the adapter |
| Split | All 23 tracks are by one artist, so they form one group and sit in one split (`train`). **This dataset cannot provide a held-out set.** These are development measurements |
| Summary table | [evaluation/mdbdrums_v3.md](evaluation/mdbdrums_v3.md) |

### Results (detector v3, default configuration)

**Full-mix onset detector — recall on annotated drum attacks, ±50 ms.** Precision is not measurable here because other instruments' onsets are not annotated.

| Attacks | Annotated | Detected | Recall |
|---|---:|---:|---:|
| All drum attacks | 7,898 | 5,455 | **0.691** |
| Kick | 1,539 | 1,256 | 0.816 |
| Cymbals | 1,002 | 698 | 0.697 |
| Toms | 90 | 62 | 0.689 |
| Hi-hat | 2,639 | 1,749 | 0.663 |
| Snare | 2,654 | 1,714 | 0.646 |

- By tolerance: 0.587 at ±25 ms, 0.691 at ±50 ms, 0.747 at ±100 ms, 0.902 at ±200 ms. Widening the window recovers little, so most misses are attacks the detector did not report at all rather than attacks it reported late.
- Hit spacing is not the main cause: recall is 0.740 for attacks less than 120 ms after the previous one and 0.721 for attacks at least 250 ms after it.
- Per track, recall ranges from 0.36 (free jazz) to 0.97 (80s rock). The heavier, faster styles are below average: punk 0.57, speed metal 0.62. Grunge is 0.96.
- 85.9% of the 4,769 onset events fall within 50 ms of an annotated drum attack. The other 14.1% are either other instruments' onsets or false positives; this reference cannot tell which.
- For comparison, the same detector scores recall 0.991 on the synthetic holdout. The synthetic fixtures overstate real-music recall by a wide margin.
- The frozen v1 detector has higher recall on these attacks (0.751) than v3 (0.691). v3's per-band gating was selected on synthetic material. Without a precision reference it is not known whether v1's extra detections are correct, so this is a warning sign, not a verdict.

**Band-ratio candidates against annotated attacks, ±50 ms.**

| Candidate | Compared with | Events | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| `bass_transient_candidate` | kick | 2,607 | 0.385 | 0.652 | 0.484 |
| `snare_like_candidate` | snare | 2,105 | 0.413 | 0.328 | 0.366 |
| `high_frequency_percussion_candidate` | cymbals, hi-hat | 219 | 0.772 | 0.046 | 0.088 |

What they actually coincide with (an event can coincide with several): of 2,607 bass-transient events, 1,043 are within 50 ms of a kick, 923 of a snare, 854 of a hi-hat, and 321 of no drum attack. Of 2,105 snare-like events, 898 are near a snare and 913 near a hi-hat. The high-frequency candidate fires on 219 events against 3,641 cymbal and hi-hat attacks. **These candidates are not usable as kick, snare or cymbal detectors on real music.** Bass-guitar attacks are not annotated, so some bass-transient "false positives" may be real bass notes.

**Candidates refuted by drum activity** (fired with the drum stem silent): 0.9% of bass-transient, 1.2% of snare-like and 2.7% of high-frequency events. Drums play through most of this material, so this lower bound says little here.

**Beats, ±70 ms, 2,694 annotated.**

| Configuration | Events | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| v3 default (beat tracker off) | 0 | — | 0.000 | 0.000 |
| v3 with the experimental beat-phase tracker, observed beats | 15 | 0.867 | 0.005 | 0.010 |
| v3 with the experimental tracker, predicted beats | 25 | 0.880 | 0.008 | 0.016 |
| Frozen v1 (labels onsets as beat candidates) | 1,621 | 0.382 | 0.230 | 0.287 |

The engine has, in practice, no beat tracking on real music. Downbeats are annotated (dataset) and the engine has no downbeat output. Tempo was not scored: the dataset gives beats, not tempo marks.

**Instrument activity.** Drums: reference available (stem-derived), no detector output. No other instrument has a reference in this dataset.

**Performance.** Offline DSP p95 stayed at or below 1.6 ms per frame on every track.

### What this evaluation does not show

- Nothing about bass, vocals or guitars: no reference in this dataset.
- Nothing about full-mix onset precision.
- Little about metal: one 36-second speed-metal excerpt.
- Not a held-out result. All tracks are one artist group and the numbers are development measurements. Nothing was tuned on them in this milestone.

## Milestone D4: where detection needs to improve

Ordered by what the measurements support.

1. **Beat tracking is effectively absent** (recall 0.5–0.8% with the experimental tracker, none by default). A new causal beat and tempo tracker is needed; rhythm is the largest gap between what the visualizer needs and what the engine provides. Reference: MDB Drums beats for development. A held-out beat set with obtainable audio (RWC, Ballroom) is needed before any claim.
2. **Onset recall on real drums is about 0.69**, far below the synthetic 0.99, and lowest on snare and hi-hat and on punk and speed metal. The misses are unreported attacks, not late ones. Before changing thresholds, a precision reference is required, otherwise recall can be raised simply by firing more. Options that need no labelling: exact-onset rendered multitracks (Tiny AAM, BabySlakh) as a separate "rendered" tier, and stem-based checks on MUSDB18-HQ and MoisesDB (an onset while every stem is silent is a definite false positive).
3. **Drum-class labels need a real method.** Band-energy ratios reach F1 0.48 (kick), 0.37 (snare) and 0.09 (cymbals and hi-hat). A kick/snare/hi-hat classifier on detected onsets should be developed against MDB Drums, with ENST-Drums or MoisesDB kick and snare stems for a held-out check.
4. **There is no instrument-activity output at all.** The reference side now exists (stem-derived drums here; drums, bass, vocals from MUSDB18-HQ; kick, snare, cymbals, guitar from MoisesDB). A first activity estimator for drums, bass and vocals is the next capability to build, and whether lightweight DSP suffices or a separation model is needed is exactly what the MUSDB18-HQ stems can decide.
5. **v3 versus v1 on real music** should be re-examined once a precision reference exists; v3 was selected on synthetic data and loses recall on real drums.

Data steps that unblock the above, none needing manual labelling: import MUSDB18-HQ (22.7 GB, open download) for drums/bass/vocals activity and the 12 heavy-metal tracks; add a rendered-multitrack adapter for onset precision; add a beat dataset with a different artist pool for a held-out rhythm set; request MoisesDB.

## Milestone D3, continued: MUSDB18-HQ

The owner approved downloading MUSDB18-HQ (22.66 GB, academic / non-commercial terms). It was extracted outside the repository and imported with one command.

| | |
|---|---|
| Import | 150 of 150 tracks, 9.82 hours, 0 failed, validation clean; the real layout matched the adapter |
| Reference | drums, bass, vocals and other activity for every track — `stem_derived`. No attacks, onsets, beats or tempo: none are annotated in this dataset |
| Splits (by artist) | train 81 tracks / 4.78 h, validation 15 / 1.29 h, test 54 / 3.76 h. All 50 official test tracks are in test, plus 4 official train tracks whose artists also appear in the official test set |
| Shared artist | The 13 MUSDB18-HQ tracks by the MDB Drums artist share its group and split (train) |
| Evaluated | train and validation: 96 tracks, 363.6 minutes. **The test split was not evaluated and remains held out** |
| Summary tables | [evaluation/musdb18hq_train.md](evaluation/musdb18hq_train.md), [evaluation/musdb18hq_validation.md](evaluation/musdb18hq_validation.md) |

**Instrument activity.** Reference active time: train — drums 14,384 s, bass 14,281 s, vocals 10,873 s, other 16,047 s; validation — drums 3,580 s, bass 3,418 s, vocals 2,939 s, other 4,472 s. The engine has **no instrument-activity output**, so every class is reported as no detector output. This is now the largest block of reference the engine cannot be scored against.

**Band-ratio candidates refuted by activity** (fired while none of the instruments the name suggests was sounding within 50 ms; a lower bound on false positives):

| Candidate | Checked against | Train: events / refuted | Validation: events / refuted |
|---|---|---:|---:|
| `bass_transient_candidate` | drums, bass | 34,468 / 798 (2.3%) | 7,868 / 326 (4.1%) |
| `snare_like_candidate` | drums | 16,281 / 1,576 (9.7%) | 6,301 / 908 (14.4%) |
| `high_frequency_percussion_candidate` | drums | 1,881 / 203 (10.8%) | 547 / 29 (5.3%) |

At least one in ten snare-like events on this material fires with no drums playing at all. The true false-positive rate is higher, because drums being active does not make an event a snare (see the MDB Drums precision of 0.41).

Not broken down by genre: the importer does not yet tag MUSDB18 tracks with the genre list published by the dataset, so the 12 heavy-metal tracks are not reported separately.

## Milestone E1: causal beat and tempo tracker

D4's first priority. The circular-phase prototype behind `experimental_beat_phase` is replaced by a new tracker (`gmv.beat.BeatTracker`, event method `causal_beat_tracker_v1`). It is still **opt-in and off by default**: the default stream and the synthetic onset results are unchanged (development 565/0/0, holdout 577/0/5).

How it works:

- **Its own onset-strength signal.** The rise in log magnitude between consecutive analysis windows, averaged inside each of the seven bands and then summed, so the many high-frequency bins do not outvote the few bass bins.
- **Tempo.** Autocorrelation of the last 6 seconds of that signal, scored at the first four multiples of each candidate beat period (60–240 BPM), with a broad preference for 120 BPM and a penalty on sudden tempo jumps.
- **Phase.** A running cumulative score (the method of Ellis 2007 and of Stark, Davies and Plumbley's real-time tracker, reimplemented). Each beat is announced ahead of time as a `beat_prediction` event carrying the tracker's tempo in its `tempo_bpm` measurement.
- **Gates.** No beats until 3 seconds of audio have been heard, and none unless the onset detector has reported at least 4 attacks in the last 4 seconds. This is what stops a sustained tone from producing beats.
- `beat_observed` keeps its meaning: a detected onset within 50 ms of a predicted beat.
- The frame field `tempo_bpm` still comes from the older autocorrelation estimate; the two are not yet unified.

### Result on MDB Drums (development data, 23 tracks, 2,694 annotated beats, ±70 ms)

| Configuration | Events | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Previous prototype, predicted beats | 25 | 0.880 | 0.008 | 0.016 |
| Frozen v1 (onsets labelled as beat candidates) | 1,621 | 0.382 | 0.230 | 0.287 |
| **New tracker, predicted beats** | 2,431 | **0.726** | **0.656** | **0.689** |
| New tracker, observed beats (onset on a predicted beat) | 1,855 | 0.762 | 0.524 | 0.621 |

- Median absolute timing error of matched beats: 11 ms. Continuity, mean over tracks: CMLt 0.53, AMLt 0.67.
- Announced a median of 247 ms before the beat (5th percentile 120 ms). The first beat comes about 3.3 s into a track, so every track's opening beats count as misses; these tracks are 13 to 125 seconds long.
- Per track: median F1 0.81; 12 of 23 tracks at 0.8 or above; 5 below 0.4.
- **Main failure: locking onto the off-beat.** Punk (F1 0.00) and the Beatles-style track (0.15) have the right tempo and the wrong phase (AMLt 0.89 and 0.69). Latin jazz (0.13), the Zeppelin-style track (0.26) and free jazz (0.36) are also poor.
- Metrical level: 19 tracks at the annotated level, 2 at half time (bebop and gospel, annotated near 220 BPM), 1 at double time, 1 other.
- Cost: offline DSP p95 stayed at or below 1.7 ms per frame per track.
- Summary table: [evaluation/mdbdrums_beat_v1.md](evaluation/mdbdrums_beat_v1.md).

### What this result does not show

- **It is not a held-out result.** The onset-strength signal was chosen from six candidates, and the history length set, on these same 23 tracks. All are by one artist group.
- **The tempo range is narrow.** 20 of the 23 tracks are annotated near 110 BPM, close to the tracker's 120 BPM preference, which flatters it. Behaviour at metal tempos is unmeasured: the only metal is one 36-second excerpt (F1 0.94, annotated near 110 BPM).
- **Tempo changes and dense patterns are weak** on the synthetic development fixtures: tempo-change fixture precision 0.70 / recall 0.51; the dense fixture precision 0.39 / recall 0.27 at the wrong metrical level. The synthetic holdout was not run with the tracker.
- No downbeats, bars or time signature.
- Not yet tried on live capture.

Before the tracker can be turned on by default it needs a held-out beat set from other artists and tempos (RWC or Ballroom; neither is imported), and the off-beat failure needs work.

## Milestone E2: instrument-activity estimator (drums, bass, vocals)

The engine can now report, per analysis frame, whether drums, bass and vocals are sounding. This is the first output that can be scored against the stem-derived activity reference. It is **opt-in**: it needs a weights file, and without one the stream is unchanged.

How it works (`gmv.activity`, method `activity_mlp_v1`):

- **Features.** For each frame, 20 log-spaced bands from 40 Hz to 16 kHz are described by their share of the energy, how much they rose since the previous frame, how peaky (tonal) they are, and their stereo width, plus the level relative to the last few seconds. All are independent of playback volume. Causal averages over 0.25 s and 1.5 s and a measure of fluctuation are added.
- **Model.** One hidden layer of 64 units, about 21,000 weights, plain NumPy. Output is one uncalibrated score per instrument, turned into on/off with fixed thresholds (on at 0.6, off at 0.4). Frames below the silence level are always inactive.
- **Training.** `scripts/train_activity.py` fits it on the MUSDB18-HQ **train** split (81 tracks, 4.8 h) with the stem-derived activity as labels, weighting active and inactive time equally so that "always active" is not a solution. No hand labelling.
- **Output.** Frame fields `instrument_scores`, `instruments_active` and `activity_method`.
- **Cost.** Offline DSP p95 per frame rose from about 1.5 ms to about 1.7 ms (worst track 2.2 ms).

### Result on the MUSDB18-HQ validation split (15 tracks by other artists, 77 minutes)

These tracks were not trained on. Drums, bass and vocals play most of the time in this music, so an estimator that always says "active" already scores a high F1; that baseline and the false-positive rate (share of truly silent time reported as active) are shown beside the score.

| Instrument | Precision | Recall | F1 | Always-active F1 | False-positive rate |
|---|---:|---:|---:|---:|---:|
| Drums | 0.963 | 0.934 | **0.948** | 0.872 | 0.122 |
| Bass | 0.967 | 0.888 | **0.926** | 0.850 | 0.086 |
| Vocals | 0.883 | 0.813 | **0.847** | 0.777 | 0.187 |

- Ranking quality at frame level (AUC, 0.5 is chance): drums 0.96, bass 0.96, vocals 0.90.
- Per track, median F1 is 0.97 (drums), 0.96 (bass), 0.86 (vocals). The worst tracks are poor: one track scores 0.00 for bass and one 0.40 for vocals, and on the worst tracks half or more of the silent time is reported as active.
- Entrances are detected late: a median of roughly 0.2 s from the reference entrance.
- "Other" instruments (guitars, keyboards and everything else in one stem) are not estimated.
- Summary table: [evaluation/musdb18hq_validation_activity_v1.md](evaluation/musdb18hq_validation_activity_v1.md).

### What this result does and does not show

- **Lightweight analysis is enough for a useful first drums and bass signal; vocals are noticeably weaker.** Roughly one second in five without vocals is still reported as vocals. Whether a separation model is worth its cost should be judged mainly on vocals.
- **The validation split was used to choose the model.** Four network sizes were compared on it (they scored alike) and one was kept. It is therefore a development result. **The MUSDB18-HQ test split (54 tracks, 3.76 h) has still not been read**; it is the place for a final number.
- **The model overfits.** On the tracks it was trained on it scores AUC 0.99; on new tracks 0.90 to 0.96. Eighty-one tracks is little data. More stems (MoisesDB, MedleyDB) are the obvious improvement.
- **The reference is stem energy, not a musical judgement.** Reverb tails and bleed count as activity.
- **No per-genre figure.** Metal is not reported separately because MUSDB18-HQ genre tags are still not imported.
- **Not tried on live capture.** Streaming services alter loudness and stereo image; the features ignore level but that is untested.
- No kick, snare, cymbal or guitar activity: MUSDB18-HQ has no stems for them.

### The weights file is not in the repository

The weights are derived from MUSDB18-HQ audio, whose terms are academic and non-commercial. Publishing them is the owner's decision, so the file stays in the git-ignored `.model-cache/` directory. Anyone with the dataset can rebuild it:

```powershell
.venv\Scripts\python.exe scripts\train_activity.py --corpus corpus-data\corpus\manifest.json --dataset musdb18hq --cache <cache folder> --output .model-cache\activity_v1.npz
.venv\Scripts\python.exe -m gmv.evaluation.run --detector candidate --activity-model .model-cache\activity_v1.npz --corpus corpus-data\corpus\manifest.json --corpus-split validation --dataset musdb18hq --output local-evidence\activity.json
```

Also in this milestone: corpus audio is now read into memory before decoding. On the dataset drive used here that cut the load time of one track from about a minute to about a second.

## Status of the phase

Done: A, B, C, D1, D2, D3 (MDB Drums, MUSDB18-HQ train and validation), D4, E1 (beat tracker, development result only), E2 (drums, bass and vocals activity, validation result only).

Outstanding:

- More reference material: genre tags for MUSDB18-HQ, MoisesDB import (adapter unverified on real files), a rendered-multitrack tier, a held-out beat set, MedleyDB adapter.
- Detection work from D4: a held-out check and off-beat fix for the beat tracker, onset recall with a precision reference, drum classes; for instrument activity a final test-split evaluation, better vocals, and guitar and drum-part activity (needs MoisesDB).
- Capture reliability (USB disconnect/reconnect, default-output switch on the native backend) and source-separation research: not started in this phase.
- Final validation and report.

**Real-music analysis is not validated.** What exists is a development measurement on 21.8 minutes of real music for drum attacks and beats (including an opt-in beat tracker at F1 0.69 on the data it was developed on), and an opt-in drums, bass and vocals activity estimator scored on 77 minutes of validation tracks (F1 0.95, 0.93, 0.85). The MUSDB18-HQ test split (3.76 h) is reserved and untouched.
