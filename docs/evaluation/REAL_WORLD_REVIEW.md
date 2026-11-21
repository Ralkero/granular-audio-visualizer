# Real-world guitar review: method

A way to turn ordinary music files the owner has into trustworthy, excerpt-level guitar labels, and to see how the existing model behaves on them. It does **not** train anything. Code: `src/gmv/review/`; command: `gmv review`.

Nothing personal goes into Git: the selection file, excerpts, labels, timelines and the batch report all live in a local, ignored work folder (`review-local/<batch>/`). This document names no recordings.

## Steps

```
gmv review --workdir review-local/pilot-v1 build --inventory Complete_Music_Inventory.csv --selection review-local/pilot-v1/selection.json
gmv review --workdir review-local/pilot-v1 verify
gmv review --workdir review-local/pilot-v1 open
gmv review --workdir review-local/pilot-v1 report
gmv review --workdir review-local/pilot-v1 export            # core labels -> corpus format
```

`build` is resumable: each recording's results are cached, so a rerun only does what is missing, and adding recordings to the selection file adds them to the batch.

## Choosing recordings

From a collection inventory (one row per file with tags, hashes and duplicate groups), a local selection file lists:

- **picks**: named recordings with a category and a written reason;
- **quotas**: "N recordings from this artist folder", drawn reproducibly (seeded by the file hash), spread over albums (at most a set number per album), never two members of one duplicate or alternate-version group.

Categories: positive candidate, mixed candidate, hard-negative candidate, negative candidate. They are preliminary and come from metadata and stated reasons. **No model score is used to choose.** The inventory's earlier model observation is copied into the batch table for comparison only.

Each recording is registered in a corpus manifest as `owned_local`. The leakage group is the whole artist folder, so alternate versions, duplicates and every excerpt of an artist's recordings always share a split. Splits stay unassigned during review.

## Excerpts

12-second excerpts in three categories that are never mixed in results:

| Category | How it is placed | Use |
|---|---|---|
| `core` | five per recording: start, middle, end, two random positions seeded by the file's SHA-256 | **primary, unbiased evaluation** |
| `feature` | the free window whose level and brightness differ most from the rest of the track | separate results |
| `model_flagged` | at most one window where the model's own output suggests a miss (guitar expected) or a false activation (guitar not expected) | **diagnostic only**, never in evaluation |

Excerpts never overlap. Recordings too short for five non-overlapping excerpts get fewer, and the number is recorded.

Each excerpt is the decoded source samples written unchanged as 32-bit float WAV at the source's sample rate and channel count; it is compared sample-for-sample with the decoded source when written and again by `verify`. Start and end are kept in seconds and in samples. The only transformation is decoding (MP3 and other lossy formats by libsndfile/mpg123); there is no resampling, gain, normalisation, filtering or dither. The model's own preprocessing (channel mean, resample to 32 kHz, log-mel) happens only inside the model path and is documented in each row.

## Source quality and integrity

Per recording, kept apart from excerpt properties: container and subtype as decoded; codec, bit rate and bit depth from the inventory; sample rate and channels; declared and decoded length; SHA-256 checked on the same read that decodes; peak; samples beyond full scale; runs of at least three samples at full scale (clipping); RMS; DC offset; identical channels; channel correlation; share and longest stretch below -50 dBFS and interior low-volume sections; effective bandwidth (highest frequency within 50 dB of the 1–4 kHz level); decoder messages captured from the native decoder. Per excerpt: peak, RMS, full-scale samples, clipping runs, low-volume share, and whether it matches the source exactly.

Raw excerpt measurements (spectral centroid, roll-off, flatness, onset density, harmonic / percussive energy split) are descriptive. They do not show that any instrument is present.

## Model timelines

The existing model is run over each whole recording through `run_recognition`, the offline twin of the application's recognition worker: 10 s of context ending at each 0.5 s hop, smoothing over two estimates, the description's thresholds (detected, candidate). Output: score and state per hop for guitar and the three sub-labels (`Guitar_Activity_Timelines.csv`, `output_kind = model_prediction`). Excerpt readouts are taken from the full-track timeline, so the model always had its real context; readouts for excerpts starting before 10 s are marked `model_context_complete = false`. Per recording the batch table lists the model's longest on and off stretches, *possible* missed passages (at least 10 s inactive where guitar is expected) and *possible* false activations (at least 4 s detected where it is not). These are observations to check by listening, never labels.

## Review window

`gmv review open`: a PySide6 window in the same environment as the application (a separate window because the visualizer is built around live capture; reviewing files needs neither capture nor rendering). Playback goes through PortAudio (PyAudioWPatch, already a dependency); Qt Multimedia would add about 200 MB.

- Excerpt list with filters: review queue (unreviewed, core first), uncertain queue (uncertain or flagged to revisit), all, per category, labelled.
- Artist, title, album, excerpt category and position, original start and end in time and samples; play / pause (Space), replay (R), seek bar with the original time.
- Annotation controls: presence (present throughout T, present intermittently I, absent A, uncertain U), an optional approximate note of when guitar is audible for intermittent excerpts, guitar type (D, C, K, M, unknown), role, other prominent instruments, notes, a difficulty note, a revisit flag; previous (B), save and next (N).
- **Nothing is prefilled.** Every control starts empty for an unlabelled excerpt.
- The model's output and the recording's selection category (a metadata prior) are in a separate grey diagnostics box, hidden until a presence answer has been saved for that excerpt (an "always show" option exists).
- Recordings are queued in a fixed shuffled order, so expected-positive and expected-negative recordings are interleaved rather than reviewed in blocks.
- Saved automatically shortly after each change, when moving on and on closing; reopening returns to the last excerpt.

## Annotation storage

- `annotations.jsonl`: append-only; every save is a full snapshot with excerpt and recording id, revision number, UTC time, annotator and source `human:gmv-review-window`. Revisions never overwrite earlier answers.
- `Human_Annotations.csv`: rebuilt from the log; one row per excerpt, blank for unreviewed ones; no model columns.
- The store accepts only the annotation fields and their allowed values; an unknown field (such as a model score) is rejected. The build never writes to it.

## Corpus export

`gmv review export` writes one corpus annotation per recording (`reference_kind = human_annotation`, `status = reviewed`), from **core excerpts only** unless other categories are asked for explicitly:

| Answer | Export |
|---|---|
| present throughout | excerpt covered, guitar active over it |
| absent | excerpt covered, no guitar activity |
| present intermittently | excerpt covered, guitar activity marked *uncertain* (left out of frame scoring); the reviewer's approximate spans are kept in the notes, never as exact boundaries |
| uncertain or unreviewed | not covered: unknown |

Time outside the covered excerpts is unknown. `scripts/train_activity.py:frame_labels` was changed to honour coverage ranges for this: before, everything outside an annotation's activity counted as absent. References that cover whole recordings (every existing dataset reference) give identical frame labels; this was checked on all 904 annotated recordings.

## Scoring

`gmv review report` scores the model per excerpt category, never pooled:

- **primary**: core excerpts; hops in "present throughout" excerpts are positives, hops in "absent" excerpts negatives; recall, false-positive rate, precision, F1 for "detected" and for "detected or candidate"; plus excerpt-level counts;
- feature-selected excerpts in their own table;
- model-flagged excerpts in a section marked diagnostic;
- intermittent excerpts: the share of hops the model had on, reported separately; uncertain and unreviewed excerpts are only counted.

The report also lists excerpts where reviewer and model disagree strongly.

## Safeguards

1. Original files are opened read-only; their hash is checked when decoded and again by `verify`.
2. Every excerpt is tied to its recording id, sample span and hash in `Excerpt_Manifest.csv`.
3. Model output and human labels are in different files and fields; the label store rejects anything else.
4. Model output never chooses recordings or core excerpts, and never fills in a label.
5. A song is never treated as continuously guitar-positive: only excerpts are labelled, and only "present throughout" excerpts count as certain guitar time.
6. Uncertain answers stay uncertain (unknown in exports).
7. Duplicates and alternate versions: one per group in a batch; groups are whole artists.
8. Splits are by artist group, so excerpts of one recording cannot land in different splits.
9. Whole artists can be held out later; the batch report recommends deciding this before any of this material is used for training.
10. A review batch is evaluation material first. The pilot batch has since been used for a fine-tuning attempt ([review_finetune_v1.md](review_finetune_v1.md)), measured with whole artists held out. Weights trained on a batch (`guitar_mn10_rv1`) must not be scored on it; `guitar_mn10_v1` has never seen it and can still be.

## Dependencies and cost

No new dependencies: numpy, scipy, soundfile, PySide6 Essentials, PyAudioWPatch and ONNX Runtime (CPU) are all existing. No CUDA, no GPU, no source separation. Building a batch of about 65 recordings (decoding, integrity checks, model timelines at about 25 ms per estimate, excerpts) takes about 25 minutes over six processes on a 16-core CPU; excerpts take about 4.6 MB each at 48 kHz stereo.
