# Second training experiment with the owner's recordings: plan

**Status: proposed, awaiting the owner's approval. Nothing has been trained.**
`guitar_mn10_v1` remains the model the application starts with; `guitar_mn10_rv1` stays a
separate local alternative. What was checked before asking for approval is in
[PRE_TRAINING_VALIDATION_V2.md](PRE_TRAINING_VALIDATION_V2.md). Recording and artist names
stay local (`research-local/evidence-v1/trainset-v1/`).

## What the review queue showed

The owner answered 167 of 168 excerpts (12 s each) chosen by `gmv evidence fuse`.

| Automatic class audited | Excerpts | Right | Precision | 95% interval |
|---|---|---|---|---|
| Guitar, both models agree | 71 | 70 | 98.6% | 92 to 99.8% |
| No guitar, both models agree | 40 | 32 | 80% | 65 to 90% |

- The single wrong "guitar" candidate is a synthesizer piece. On artists not used in the
  first experiment, 19 of 19 were right.
- "No guitar" is wrong one time in five, as on the pilot labels. The misses are electric
  guitars under bass and synthesizer. It is not used as a label, in any class.
- Where the recogniser reported guitar and PANNs heard none (39 excerpts), guitar was
  audible in 32, twelve of them acoustic.

When the queue was built, the 71 "guitar" excerpts were drawn as two classes (31 "strong",
40 "moderate"). The "strong" class of that time rested on recording-level credits and no
longer exists (next section), so the audit is reported for the one class they now share.

## Correction to the evidence classes

A recording-level MusicBrainz guitar credit used to raise every stretch on which the two
models agreed from moderate to strong. A credit establishes that guitar is on the recording
somewhere, not that it plays in a given stretch, so this is no longer done:

- Only evidence measured over the stretch itself sets or raises its class: the two models'
  scores, and the separated guitar stem's level over at least 80% of that stretch.
- Credits of any scope (recording, release, artist) are kept beside a candidate as context
  (`recording_level_metadata`) and change neither its class nor its weight.
- Automatic "no guitar" is weak evidence in every case, never a training label.

Effect: all 196 stretches that were "strong" (0.89 h) owed the class to a credit alone and
are now moderate, at weight 0.25 instead of 0.5. Separation was only run around reviewed
excerpts, so no stretch outside them qualifies as strong today.

## The sets (`gmv evidence trainset`)

| Source | Segments | Guitar | No guitar | Hours | Groups | Weight |
|---|---|---|---|---|---|---|
| Human, chosen without model output (pilot batch) | 223 | 159 | 64 | 0.74 | 19 | 1.0 |
| Human, chosen because of model output (queue) | 162 | 121 | 41 | 0.54 | 33 | 1.0 |
| Automatic, strong agreement | 0 | 0 | 0 | 0 | 0 | 0.5 |
| Automatic, moderate agreement | 427 | 427 | 0 | 1.72 | 29 | 0.25 |
| Weak, conflicting or unknown evidence | not in the set | | | | | 0 |

Rules, enforced in code and tested:

- A person's answer wins. Every part of an automatic candidate that overlaps a reviewed
  excerpt is cut away; remainders shorter than 6 s are dropped.
- Automatic segments are "guitar present" only, and are never scored against.
- "Intermittent" answers (14) are left out: they are not timed precisely enough.
- The 45 leakage groups (artist folders joined by duplicate audio, alternate versions and
  shared recording identities) are placed whole into 5 folds. Every fold holds 10 to 16
  guitar-free excerpts of the unbiased pool.

| Fold | Groups with segments | Unbiased pool: guitar / none | Model-selected pool: guitar / none | Automatic segments (hours) |
|---|---|---|---|---|
| 0 | 7 | 18 / 16 | 15 / 8 | 53 (0.19) |
| 1 | 7 | 25 / 13 | 26 / 6 | 179 (0.87) |
| 2 | 7 | 47 / 12 | 22 / 8 | 54 (0.16) |
| 3 | 7 | 42 / 13 | 28 / 7 | 64 (0.23) |
| 4 | 8 | 27 / 10 | 30 / 12 | 77 (0.28) |

Nine further groups have no segment of either kind. The folds are a device for scoring
every excerpt with a model that did not train on its group; they are not five datasets.
Groups are the unit of leakage protection and of every interval reported.

## The experiment

Fixed in advance. No other configuration will be tried and nothing is chosen afterwards.

- **Two arms.** H: human-confirmed segments only. H+A: the same human segments plus the
  427 automatic "guitar present" segments at weight 0.25. The arms differ in nothing else.
- **Starting weights.** Both arms, every fold: the weights `guitar_mn10_v1` was exported
  from (`guitar_mn10_ft_all.pt`, SHA-256 `eb009fa0…29df`). The baseline is that model.
- **Folds.** Each arm is trained five times, each time without one fold. An excerpt is
  scored only by the run that kept its fold out.
- **Settings, identical in both arms:** 500 steps, batch 16, learning rate 5e-5 (ten times
  that for the output layer), 40% of clips from the owner's recordings, 15% silenced
  lead-ins. Clips of the owner's recordings are drawn guitar or guitar-free equally often,
  then by leakage group (every group equally often), then a segment of the group. A
  segment's weight multiplies the loss of the columns it labels.
- **Seeds.** One per fold and per kind of random decision, derived as
  `SeedSequence([20261122, fold, stream])` for the streams *plan* (which clips come from the
  owner's recordings, which of those are guitar-free, which lead-ins are silenced),
  *moisesdb*, *review_absent*, *review_present*, *augment* (gains, silence points) and
  *torch* (dropout). The arm is not part of any seed. Because the streams are separate, the
  two arms of a fold see the same MoisesDB clips, the same guitar-free clips, the same gains
  and the same silenced lead-ins; only the guitar clips of the owner's recordings differ.
  There is no random initialisation: every weight is loaded.
- **Record of each run** (`run.json`): arm, fold, command line, every setting, the seeds,
  the starting weights' hash, the hash of the segment table and fold assignment, the ids of
  every training segment and every scored excerpt, the code commit, library versions, the
  clips actually drawn by kind, and the hash of the weights written.
  `experiments/recognition/review_experiment_v2.json` holds the whole experiment;
  `run_review_experiment.py` runs it from that file and resumes after an interruption.

## Scoring

Out-of-fold predictions only, on the path the application uses: each answer is about the
10 s ending at a hop, quiet audio is raised to ordinary level first, and scores are averaged
over two hops. Baseline, H and H+A are scored on the same excerpts by the same code.

Reported apart, never combined into one figure:

1. **Primary pool:** the pilot excerpts, chosen without model output (159 guitar, 64
   guitar-free, 19 groups).
2. **Secondary pool:** the queue excerpts, chosen because of the baseline model's output
   (121 / 41, 33 groups), also per selection reason. The baseline is at its ceiling or
   floor on them by construction: this pool can show a regression, never an unbiased gain.
3. Primary pool by guitar type (distorted, clean, acoustic) and by start in the first 10 s.
4. **MoisesDB control:** artist fold 0 with guitar-free twins, kept out of the training mix.

For each: AUC; recall at 5% and 10% false alarms; and, at the shipped thresholds of
`guitar_mn10_v1` read from its description (**detected 0.926, candidate 0.627**), recall
and false-positive rate. The same two values are applied to every arm. No threshold is
fitted, moved or chosen in this experiment; calibration would be a separate experiment.

Changes are computed against the baseline on the same excerpts. Intervals come from 2,000
bootstrap replicates that resample **leakage groups**; each replicate uses the same groups
for every arm, so the comparison is paired. Every figure is reported with its numbers of
groups, excerpts and hops.

## What counts as a gain

All of the following, for H+A against the baseline:

1. The 95% interval of the change in AUC on the primary pool lies above zero.
2. Recall at 5% false alarms on the primary pool does not decline.
3. MoisesDB fold 0 loses no more than 0.01 AUC (mean of the five runs; the lowest run is
   reported too).
4. The secondary pool shows no regression: fails if the whole 95% interval of its AUC
   change is below zero. Passing this is not evidence of a gain.
5. False positives at the shipped "detected" threshold on the primary pool do not rise
   meaningfully: fails if the 95% interval of the change lies above zero, or if the rise
   exceeds 0.02 of guitar-free time whatever the interval says. The "candidate" threshold
   result is reported beside it.

A criterion that cannot be evaluated (fewer than five groups with guitar or with
guitar-free excerpts, or too few valid replicates) is reported as such and never counted as
passed. Recall at "detected" or "candidate", or false positives at "candidate", moving the
wrong way outside their interval are flagged even if every criterion passes. The same
table is produced for H, to show what the human segments alone do.

If it counts: final weights would be trained on everything from `guitar_mn10_v1` and kept
local as `guitar_mn10_rv2`, selectable, not the default; that step and any change of the
default need separate approval after listening in the application. If it does not: reported
as measured.

## Limits known before the run

- The primary pool is 19 groups (18 with guitar, 12 with guitar-free excerpts). The
  baseline's AUC there is 0.854 with a 95% interval of 0.79 to 0.90; small gains will not
  be distinguishable from chance.
- Guitar-free material is concentrated: three groups supply half of the 105 guitar-free
  excerpts; 18 of 45 groups have any.
- One artist supplies 44% of automatic time, and every automatic segment is one the
  baseline already scores above "detected". Group-wise sampling limits the first; the
  second means the automatic segments may teach little, which is what H+A against H shows.
- The baseline was trained on all MoisesDB train and validation songs, including the
  control fold. The control therefore checks that the model keeps what it had there; it is
  not a test on unseen MoisesDB artists. The fold stays out of this experiment's training.
- Criterion 5 rests on 64 guitar-free excerpts from 12 groups, of whose hops the baseline
  puts 3.7% above "detected". It can detect a clear rise, not a small one.
- No "strong" automatic class exists until separation is run more widely.
- Not available: AcoustID results, MedleyDB, audited guitar-type labels for automatic
  segments.
