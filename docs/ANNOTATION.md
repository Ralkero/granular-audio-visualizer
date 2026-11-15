# Annotation format and optional manual workflow (annotation schema 1.1)

**Manual annotation is optional.** The corpus is normally built from research datasets and their stems with no labelling at all (see [DATASETS.md](DATASETS.md)). This document describes the annotation file format, which every reference uses, and the optional Audacity workflow for anyone who wants to add or correct labels by hand.

Hand annotations are made in [Audacity](https://www.audacityteam.org/) label tracks and imported into a versioned JSON file per recording and annotator. Nothing is uploaded anywhere.

## Two kinds of ground truth

They are separate on purpose and are never merged.

**A. Full-mix events** — what happens in the mix as a whole.

| Layer file | Label type | Meaning |
|---|---|---|
| `onset.txt` | points | An audible acoustic onset in the full mix |
| `beat.txt` | points | Beat positions (the pulse you would tap) |
| `downbeat.txt` | points | First beat of each bar |
| `tempo.txt` | points, text = BPM | Tempo from that time on, e.g. `174` |
| `sections.txt` | points or regions, text = name | Section or energy change; the text `transition` marks a major transition |

**B. Instrument-specific truth** — one file per instrument.

| Layer file | Label type | Meaning |
|---|---|---|
| `activity.<label>.txt` | regions | The instrument is sounding. Regions in different files overlap freely |
| `hits.<label>.txt` | points | Individual attacks of that instrument (a kick hit, a snare hit, a pick attack) |

Labels: `drums`, `kick`, `snare`, `cymbals`, `hi_hat`, `toms`, `bass`, `guitar`, `rhythm_guitar`, `lead_guitar`, `vocals`, `vocals_clean`, `vocals_harsh`, `other_harmonic`, `ambient`, `other`, `unknown`. Parents and children (`drums` ⊃ `kick`/`snare`/`cymbals`/`hi_hat`/`toms`, `guitar` ⊃ `rhythm_guitar`/`lead_guitar`, `vocals` ⊃ `vocals_clean`/`vocals_harsh`) are separate layers; annotate at whatever level you can hear reliably.

A full-mix onset is not an instrument attack. A kick, a crash and a vocal entrance at the same instant are **one** label in `onset.txt` and one label each in `hits.kick.txt`, `hits.cymbals.txt` and (as a region start) `activity.vocals_clean.txt`. A kick is not automatically a beat, and a beat needs no kick: `beat.txt` is the pulse, `hits.kick.txt` is the drum.

## Uncertainty, abstaining and unknown sounds

- **Not annotated ≠ absent.** A layer file that is missing means "I did not annotate this". A file that exists but is empty means "I listened for this and it is not there". Only annotated layers are scored.
- **Uncertain labels:** end the label text with `?` (for example `?`, or `174?` in `tempo.txt`). An uncertain point is neither required nor penalised: the detector gets no credit and no blame for it. An uncertain region is excluded from interval scoring.
- **Unknown sounds:** use `activity.unknown.txt` / `hits.unknown.txt` for audible material you cannot classify. Do not guess an instrument.
- **Partial coverage:** if you only annotated part of a recording, add `coverage.txt` with region labels over the stretches you covered. Without it the whole recording counts as covered for every layer you provide.
- `--uncertainty` at import records your timing precision in seconds (default 0.02). It is used as an additional matching tolerance in reports.

## Step by step

1. Register the recording and note its id (see [CORPUS.md](CORPUS.md)).
2. Open the audio in Audacity. Optional: switch the track to spectrogram view for transients.
3. Add one label track per layer (**Tracks ▸ Add New ▸ Label Track**) and name it after the layer. Point label: click and press `Ctrl+B`. Region label: drag a selection and press `Ctrl+B`. During playback `Ctrl+M` drops a label at the playhead, which is convenient for a first pass of beats; go back and correct positions afterwards.
4. Export each label track separately: select only that label track, then **File ▸ Export Other ▸ Export Labels…**, and save it into one folder per recording as `<layer>.txt` (for example `hits.kick.txt`). Exporting with several label tracks selected merges them into one file, which loses the layer.
5. Import, check and mark as reviewed:

```powershell
gmv corpus import-labels r-0123456789ab "D:\labels\r-0123456789ab" --annotator jm --tool-version "Audacity 3.7"
gmv corpus validate
gmv corpus review r-0123456789ab --annotator jm
```

A human annotation is used as reference only once its status is `reviewed`. Importing again replaces that annotator's file and resets it to `in_progress`.

To edit later, write the stored annotation back out as label files, import them into Audacity (**File ▸ Import ▸ Labels…**), edit, export and import again:

```powershell
gmv corpus export-labels r-0123456789ab --annotator jm --out "D:\labels\r-0123456789ab"
```

## Reference from stems (multitrack recordings)

Dataset imports do this automatically. For a recording registered by hand with stems, `gmv corpus reference-from-stems <id>` writes activity intervals for each stem under the annotator id `stem-energy`. A stem counts as active where its 50 ms RMS exceeds both −60 dBFS and 40 dB below that stem's loudest passage; gaps under 0.2 s are joined.

This is a measurement of stem energy, not a judgement. Bleed, reverb tails and noise count as activity, and a stem named "other" says nothing about which instrument is in it. The result is its own reference tier, `stem_derived`: it is used as reference without review, and is always reported separately from dataset-provided and human annotations. Stems flagged as containing bleed are marked uncertain and excluded from scoring. Only stems that ship with a dataset or a multitrack session qualify; the output of a source-separation model is a model prediction and is never reference. A human annotation of the same layer, once reviewed, takes precedence.

## More than one annotator

Each annotator has their own file, so disagreement is preserved rather than overwritten.

```powershell
gmv corpus compare r-0123456789ab jm second
```

The comparison lists, per shared layer, how many point labels agree within tolerance and which exist for only one annotator, and the overlap (IoU) of activity regions. Evaluation with `--annotator consensus` uses only what both marked: matched points and the intersection of regions, over layers and time ranges both covered. Everything they disagree on is left out of scoring and stays visible in the comparison. Tempo and section layers are not merged automatically. With more than two reviewed annotators, name one explicitly.

## File format

`corpus/annotations/<recording_id>/<annotator_id>.json`, validated against [AnnotationFile.json](schemas/AnnotationFile.json): schema version, recording id and audio hash, annotator id, tool and tool version, provenance, `reference_kind`, status, timing uncertainty, coverage (time ranges and layers), `events` (full-mix), `instrument_events` (point attacks) and `activity` (regions), each with a certainty and an optional intensity (0–1). Intensity and free-text notes can be added by editing the JSON; the Audacity round trip does not carry them.

## Reference tiers

Every annotation file states where its content came from (`reference_kind`), and every number in a report names the tier it rests on.

| Tier | Meaning | Counts as reference? |
|---|---|---|
| `human_annotation` | A person labelled it (Audacity workflow) | yes, once marked `reviewed` |
| `dataset_annotation` | Shipped with a research dataset | yes |
| `stem_derived` | Measured from the dataset's own isolated, time-aligned stems (activity from stem energy) | yes, reported separately from the two above |
| `pseudo_label` | Produced by an algorithm from the mix | **never** |
| `model_prediction` | Output of a separation or detection model | **never** |

When a recording has several references, evaluation takes the best available tier for each layer (human, then dataset, then stem-derived) and records which was used. The output of the detector being evaluated, or of any separation model, is never accepted as reference. Where a dataset has stems but no attacks, onsets or beats, those classes are reported as not annotated; nothing is invented to fill them.
