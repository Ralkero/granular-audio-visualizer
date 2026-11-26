# Vocal delivery recognition: a first supervised baseline (Sonic Character, Phase A4)

**Research only.** This is an offline experiment on the owner's labelled excerpts. Nothing
here is part of the visualizer, Radiant or any live analysis, and nothing in the
application imports it.

## The answer

**Can GMV tell Cleans, High Screams and Low Screams apart in artists it has never seen?
Partly. It can tell clean singing from harsh vocals. It cannot tell high from low
screams.**

- **Clean against harsh works well on unseen artists.** Each model was fitted without the
  artist it was tested on. Collapsing High and Low Screams into "harsh", the random forest
  separates clean from harsh with a balanced accuracy of **0.92** (Cleans 95% right, harsh
  89%), and logistic regression with **0.89**.
- **It is the voice, not the band.** In the 23 recordings that contain both a clean and a
  harsh excerpt, the forest still gets 21 of 23 of each right (0.91). The kind of band and
  the production are the same within a recording, so a model that only told a metal mix
  from another mix would fail there.
- **High against low does not work.** Among harsh excerpts, the models do no better than a
  coin, and slightly worse (balanced accuracy 0.36 to 0.40). High Screams are recognised
  only 13% (forest) to 27% (logistic) of the time; most are called Low Screams.
- **Three-way, the best models reach a balanced accuracy of about 0.54**, against 0.33 for
  always answering Cleans. The 95% intervals over unseen artists run from about 0.46 to
  0.63. All of the gain comes from clean against harsh.

These are **development results with exploratory model selection** (three models compared
on the same predictions). They are not an untouched final test: the held-out evaluation
artists were not used at all.

## What was tested

- **Supported answers:** Cleans, High Screams, Low Screams, the owner's own definitions
  ([VOCAL_DELIVERY_LABELING.md](VOCAL_DELIVERY_LABELING.md)).
- **Not supported:** Gutturals (10 development excerpts, all from one artist) and
  Sing-Screams (5, from 3 artists). They are not trained, and never relabelled as one of the
  three. A model that has learned three answers cannot recognise a fourth.
- **The prototype's two outcomes:** a prediction among the three, or an abstention when its
  confidence is below a threshold. An abstention means "not confident among the three". It
  does not mean "another technique": this does not detect Gutturals or Sing-Screams.

## The data, after exclusions

Only development excerpts are used. Both reference artists, and every recording of theirs,
are on the held-out side and are not used.

| Answer | Usable excerpts | Artists (= artist groups) | Recordings | Largest share from one artist |
|---|---:|---:|---:|---:|
| Cleans | 170 | 17 | 114 | 23% |
| High Screams | 45 | 7 | 40 | 27% |
| Low Screams | 51 | 10 | 41 | 28% |
| **Total** | **266** | **20** | | |

Development excerpts left out (77):

| Reason | Excerpts |
|---|---:|
| No vocals | 61 |
| Gutturals (not supported) | 10 |
| Sing-Screams (not supported) | 4 |
| Known guest-vocalist overlap | 2 |
| Put aside for review, never answered, or only an answer from the first answer set | 0 |

- **The guest-vocalist exclusions.** One development recording features the vocalist of a
  held-out reference artist. A private note names its two excerpts. The loader verifies the
  note against the dataset: each named excerpt exists and is on the development side; its
  recording carries the song title the note gives; and the note names every excerpt of that
  recording. If any check fails, the experiment stops. The answers on record for the two
  excerpts are one Low Screams and one Sing-Screams. That is why 51 Low Screams and 4
  Sing-Screams remain of the 52 and 5 counted before.
- **No audio was missing.** All 266 excerpts were read and measured, and no feature value
  was unknown. The pitch measurements can be unknown when there is too little pitched
  sound, but this did not happen here.
- **Artist groups.** Each artist is its own group. The music inventory has no named vocalist
  credits: its instrument credits list instruments, not people. Its artist credits (who is
  credited on a recording, including "feat." credits) were checked for any person credited
  on recordings of two artist groups. None links two development groups. One person, a
  composer and not a vocalist, is credited on development recordings and on one held-out
  recording; none of those development excerpts is among the 266 used.

**Held-out protection.** The side of each excerpt comes from `dataset.json` before any
answer is read. Each line of `labels.jsonl` is first matched only for its excerpt
identifier, without decoding it:

- **development lines** (349) were decoded;
- **held-out lines** (73) were never decoded. Only their event kind and skip reason were
  matched, to count 59 answered and 10 No vocals. No held-out answer value was parsed,
  stored, logged or reported.

The split rule was also re-applied to every artist before use: a hand-edited split would
stop the experiment.

## Features

There are 30 measurements of each ten-second excerpt, fixed before any model was fitted
(`gmv.character.research.delivery.features`). They come from the audio alone. Most are
taken from the centre of the stereo image, where a lead voice almost always sits. The
centre also holds the snare, kick, bass and any centred instrument, so nothing here
isolates the voice.

| Family | Measurements |
|---|---|
| Envelope (where the centre's energy sits) | Shares of the centre's 80 Hz to 8 kHz energy in six bands (80 to 250 Hz, 250 to 500 Hz, 500 Hz to 1 kHz, 1 to 2 kHz, 2 to 4 kHz, 4 to 8 kHz); the centroid and its spread over time; the 85% roll-off; the share of the 300 Hz to 3.4 kHz energy that is in the centre |
| Harshness (noise against harmonic structure) | Spectral flatness of the voice band (median, 90th percentile) and of 2 to 6 kHz; harmonic share of the voice band from the existing harmonic/percussive split (mean, spread); cepstral peak prominence (median, 90th percentile), a standard measure of breathy and rough voice; peak-to-valley contrast in three voice sub-bands |
| Pitch (the existing vocal analyzer, as Phase A2 measured it) | Centre line level; share of the music with one clear pitch; pitch clarity; range; motion; held-note share; median pitch |
| Temporal | Level spread; spectral flux; syllable-rate modulation (2 to 8 Hz); roughness: envelope modulation from 30 to 150 Hz in the centre, 300 Hz to 1 kHz and 1 to 4 kHz |

A value that cannot be measured is left unknown, never set to zero. No artist, title,
album, genre, file name, path, identifier or split enters. A test checks that renaming
every identifier leaves the feature matrix unchanged. The activity model is not used: its
weights are not shipped with GMV, so the experiment can be reproduced from the repository
and the labelled data alone. Nothing was downloaded, and no new dependency was added.

## Validation

- **Leave one artist group out.** There are 20 folds. Each fold holds out every excerpt of
  one artist group, fits on the other 19, and predicts the held-out group. Each excerpt is
  predicted once, by a model that never saw its artist or its recording. Each fold checks
  that no artist group, artist or recording is on both sides.
- **Inside each fold, everything learned comes from the training groups only:** median
  imputation of unknown values (with an indicator column for each feature that had any),
  centring and scaling, class weights, and the model.
- **Class weights:** `n / (3 × n_class)` from the training fold, so each answer counts
  equally despite 170 Cleans against 45 High Screams.
- **Intervals:** 95% intervals come from resampling whole artist groups 2,000 times. The
  intervals describe how results vary between unseen artists, not between excerpts.
- **Settings were fixed before evaluation and not tuned:**
  - **Majority:** always the training fold's most common answer.
  - **Logistic:** multinomial logistic regression, L2 penalty, C = 1.0, class-weighted,
    fitted with L-BFGS from zero.
  - **Forest:** a random forest of 300 trees, bootstrap samples, weighted Gini,
    √(features) candidates per split, at least 3 excerpts per leaf, class-weighted, fixed
    seed.
- **Model choice is exploratory.** Three models were compared on the same out-of-fold
  predictions, so whichever looks best is slightly flattered. An honest final number needs
  one pre-chosen model tested once on the held-out artists.

## Results

| Model | Balanced accuracy (95% interval) | Macro F1 (95% interval) | Accuracy |
|---|---|---|---:|
| Majority (always Cleans) | 0.333 | 0.260 (0.221 to 0.284) | 0.639 |
| Logistic regression | 0.538 (0.455 to 0.628) | 0.532 (0.451 to 0.604) | 0.695 |
| Random forest | 0.539 (0.479 to 0.629) | 0.525 (0.460 to 0.611) | 0.733 |

Accuracy is shown only for completeness. Always answering Cleans already scores 0.64.

**Per answer** (precision / recall / F1):

| Answer | Logistic | Forest |
|---|---|---|
| Cleans | 0.94 / 0.88 / 0.91 | 0.94 / 0.95 / 0.94 |
| High Screams | 0.27 / 0.27 / 0.27 | 0.21 / 0.13 / 0.16 |
| Low Screams | 0.38 / 0.47 / 0.42 | 0.42 / 0.53 / 0.47 |

**Confusion matrices** (rows: the owner's answer; columns: predicted Cleans, High, Low):

| Owner's answer | Logistic | Forest |
|---|---|---|
| Cleans (170) | 149 / 8 / 13 | 162 / 2 / 6 |
| High Screams (45) | 6 / 12 / 27 | 8 / 6 / 31 |
| Low Screams (51) | 3 / 24 / 24 | 3 / 21 / 27 |

**Two readings of the same predictions:**

| | Logistic | Forest |
|---|---|---|
| Clean against harsh, all 266 (balanced accuracy; Cleans / harsh right) | 0.891 (0.88 / 0.91) | 0.919 (0.95 / 0.89) |
| The same, only in the 8 artist groups with both (237 excerpts) | 0.890 | 0.919 |
| The same, only in the 23 recordings with both (46 excerpts) | 0.848 (17 of 23 / 22 of 23) | 0.913 (21 of 23 / 21 of 23) |
| High against low, among harsh excerpts predicted harsh | 0.404 (87 excerpts) | 0.362 (85 excerpts) |
| The same, only in the 6 artist groups with both | 0.444 | 0.404 |

### Across unseen artists

Of the 20 artist groups, 9 have only Cleans, 3 have only harsh answers and 8 have both.
Results per group, numbered by size (no names):

| Group | Cleans | High | Low | Logistic: accuracy | recall C / H / L | Forest: accuracy | recall C / H / L |
|---|---:|---:|---:|---:|---|---:|---|
| 1 | 39 | 9 | 12 | 0.67 | 0.90 / 0.00 / 0.42 | 0.75 | 0.95 / 0.11 / 0.58 |
| 2 | 31 | 12 | 5 | 0.71 | 0.84 / 0.50 / 0.40 | 0.62 | 0.87 / 0.00 / 0.60 |
| 3 | 37 | 3 | 6 | 0.85 | 0.86 / 0.33 / 1.00 | 0.91 | 0.97 / 0.00 / 1.00 |
| 4 | 18 | 0 | 14 | 0.72 | 1.00 / – / 0.36 | 0.69 | 1.00 / – / 0.29 |
| 5 | 15 | 1 | 0 | 0.69 | 0.73 / 0.00 / – | 0.88 | 0.93 / 0.00 / – |
| 6 | 3 | 12 | 1 | 0.19 | 0.33 / 0.17 / 0.00 | 0.31 | 1.00 / 0.08 / 1.00 |
| 7 | 7 | 4 | 2 | 0.77 | 1.00 / 0.50 / 0.50 | 0.77 | 1.00 / 0.75 / 0.00 |
| 8 | 0 | 4 | 7 | 0.45 | – / 0.25 / 0.57 | 0.36 | – / 0.25 / 0.43 |
| 9 | 5 | 0 | 1 | 0.67 | 0.80 / – / 0.00 | 0.83 | 1.00 / – / 0.00 |
| 10 to 20 | 1 or 2 each | 0 | 0 to 2 | 1.00, except group 13: 0.00 | | 1.00 | |

Groups 10 to 20 hold 18 excerpts between them: 15 Cleans and 3 Low Screams.

- **Cleans are consistent.** With the forest, at least 87% of every group's Cleans are
  right (17 of 17 groups at least half right; logistic: 16 of 17).
- **High Screams fail almost everywhere.** With the forest, only 1 of the 7 groups with
  High Screams has at least half of them right, and 3 have none right (logistic: 2 of 7
  and 2). Three groups supply 33 of the 45 High Screams. The forest recognises 2 of those
  33, and logistic regression 8. The answer currently rests on very few voices.
- **Low Screams are mixed.** With the forest, 6 of 10 groups have at least half right and
  2 have none (logistic: 4 of 10 and 3).
- **Group accuracy:** forest median 1.00, quartiles 0.76 to 1.00, minimum 0.31. Logistic
  median 0.92, quartiles 0.68 to 1.00, minimum 0.00. The groups with mostly harsh answers
  are where the models fail.

### What the logistic model leans on

These are the largest mean absolute weights over the 20 folds, on scaled features. They
describe the model, not proof of cause:

1. voice-band harmonic share
2. voice-band contrast
3. median pitch
4. voice-band level spread
5. 2 to 6 kHz flatness
6. 250 to 500 Hz share
7. 80 to 250 Hz share
8. harmonic-share spread
9. 1 to 4 kHz roughness
10. roll-off

The first two are the harmonic-against-noise measures, as expected for clean against
harsh.

## Abstention

Confidence is the highest of the three probabilities. Withholding the least confident
predictions, on the development out-of-fold predictions:

| Kept | Logistic: accuracy / balanced / macro F1 | Withheld: Cleans / High / Low | Forest: accuracy / balanced / macro F1 | Withheld: Cleans / High / Low |
|---:|---|---|---|---|
| 100% | 0.695 / 0.538 / 0.532 | 0 / 0 / 0 | 0.733 / 0.539 / 0.525 | 0 / 0 / 0 |
| 90% | 0.732 / 0.542 / 0.536 | 6% / 13% / 22% | 0.774 / 0.545 / 0.533 | 3% / 24% / 22% |
| 80% | 0.761 / 0.525 / 0.518 | 10% / 31% / 43% | 0.831 / 0.573 / 0.553 | 5% / 49% / 43% |
| 70% | 0.806 / 0.537 / 0.513 | 15% / 42% / 71% | 0.871 / 0.594 / 0.532 | 10% / 62% / 69% |
| 60% | 0.856 / 0.495 / 0.481 | 20% / 62% / 86% | 0.925 / 0.533 / 0.488 | 14% / 80% / 92% |
| 50% | 0.887 / 0.458 / 0.457 | 31% / 78% / 90% | 0.962 / 0.500 / 0.463 | 25% / 91% / 96% |

- **The rule fixed in advance** keeps 80%. Its threshold is chosen on the same development
  predictions, so it is optimistic. Forest: threshold 0.49, coverage 80%, accuracy 0.83,
  balanced accuracy 0.57, macro F1 0.55. Logistic: threshold 0.62, coverage 80%, accuracy
  0.76, balanced accuracy 0.52.
- **Abstention mostly removes harsh excerpts.** At 80% coverage the forest withholds 49% of
  High Screams and 43% of Low Screams, but only 5% of Cleans. Accuracy rises because what
  is left is mostly Cleans. Balanced accuracy and macro F1 barely move. Confidence does not
  separate right from wrong among the harsh answers. It mainly flags "harsh", which the
  models cannot split.
- **Abstention does not detect unknown techniques.** It was not tested on Gutturals or
  Sing-Screams, and should not be expected to catch them.

## Reproducibility and cost

- **Two complete runs gave identical results**, the second measuring every excerpt again
  without the feature cache. All 266 feature vectors, every out-of-fold probability and
  every metric matched exactly, so no numerical tolerance was needed. Both runs were on the
  same machine and software.
- **Time per run** on CachyOS (AMD Ryzen 9 5950X, 16 cores; 6 worker processes; audio on
  an external USB drive), over the repeated runs: 30 to 31 s to measure the 266 excerpts
  (about 0.7 s of CPU each), and 62 to 64 s in total. The forest's 20 folds take 31 to
  33 s; logistic regression takes 0.3 s.
- **CPU only.** Nothing uses a GPU or CUDA.

Reproduce with:

```
gmv character research delivery --inventory <music inventory CSV> --processes 6
```

This reads `local-evidence/sonic-character/vocal-labeling/` and the private leakage note
beside it, and writes into the ignored `local-evidence/sonic-character/vocal-delivery-classifier/`:

- `summary.json` and `summary.md`: counts and metrics only, checked before writing for any
  identifier, artist, title or path;
- `report.json` and `predictions.csv`: private, including per-artist results;
- `feature-cache.npz`: private.

`--no-cache` measures everything again. Without `--inventory` the credit check is skipped
(the summary records whether it ran).

## Limitations

- **Small and concentrated.** There are 45 High Screams from 7 artists, three of which
  supply 33. The intervals are wide (balanced accuracy about 0.46 to 0.63), and a few
  artists can move every number.
- **Full mixes only.** No feature isolates the voice. Guitars, drums and production share
  every measurement. The clean-against-harsh result survives within recordings, but how
  much of it is the voice and how much is what changes in the arrangement when the
  vocalist screams (heavier sections) cannot be separated with these data.
- **Ten-second excerpts, one dominant answer each.** The owner labelled the dominant
  technique, so mixed excerpts carry one answer for several deliveries.
- **One listener.** Every answer is the owner's; there is no second opinion on how
  consistent the High/Low boundary is.
- **Development only, with exploratory model choice.** No number here is a final test.
- **Vocalist identity.** Artist-disjoint folds do not prove that no singer appears in two
  groups. The inventory has no named vocalist credits. Artist credits show no shared
  person between development groups, but an uncredited guest or a shared member would not
  show. The one known case is excluded.
- **For a later held-out test.** In the music inventory, one held-out recording belongs to
  a leakage group whose other recordings are on the development side under another artist
  name. Those development recordings are by other performers and share no audio with it.
  One of their excerpts is among the 266 used here; that does not affect development
  validation, which never uses the held-out side. It should be reviewed before the
  held-out side is ever opened.

## Does this justify going further?

- **Clean against harsh: yes.** It holds on unseen artists and within recordings. A
  sensible next step is to freeze one model (the forest, or the simpler logistic
  regression) and this feature set, then test it **once** on the held-out artists, with
  the decision and the number to beat written down before looking. Only after that should
  live use be considered, as a separate phase.
  That test is planned, with every decision fixed before any held-out answer is read:
  [VOCAL_DELIVERY_HELDOUT_PLAN.md](VOCAL_DELIVERY_HELDOUT_PLAN.md).
- **High against low: not with these features or this data.** It needs both:
  - High Screams from more vocalists (at least 6 to 8 groups with several excerpts each,
    not three groups carrying most of them);
  - features that look at the voice more directly. Options include an offline separated
    vocal stem; register measures robust to noise, such as formant-like peaks of the
    harsh voice; and longer context.

  Adding models or tuning on the current features is not likely to help: the models do
  no better than a coin on this pair.
- **Gutturals and Sing-Screams** need labelling before they can even be tried. Use the
  same bar as before: about 20 development excerpts from at least 6 different vocalists
  each. Today Gutturals have 10 from one artist, and Sing-Screams 4 usable from 3.
