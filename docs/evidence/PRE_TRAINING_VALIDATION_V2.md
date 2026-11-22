# Pre-training validation of the second experiment

What was corrected, rebuilt and checked before asking for approval of the experiment in
[TRAINING_PLAN_V2.md](TRAINING_PLAN_V2.md). **No training was done:** no optimiser step, no
gradient, no new weights file. `guitar_mn10_v1` and the application's defaults are
untouched. Names of recordings and artists are in local files only.

## 1. What changed in evidence fusion

`src/gmv/evidence/fusion.py` let a recording-level MusicBrainz guitar credit (on a reliably
matched recording) raise every stretch on which both models agreed from `moderate_agreement`
to `strong_agreement`. A credit says guitar is on the recording, not when. Now:

- A stretch is `strong_agreement` only if the separated guitar stem, measured over at least
  80% of that same stretch, supports the two models. Otherwise agreement stays moderate.
- Credits of any scope are written to a separate column, `recording_level_metadata`, and
  affect neither `confidence_class` nor `suggested_training_weight`.
- Every automatic "absent" is `weak_evidence`, weight 0, not training-suitable. Before, two
  "absent" stretches supported by separation were marked moderate and training-suitable;
  the training table already ignored them, and now the candidates table agrees.

Other uses of credits were reviewed and left as they are, because none acts on a stretch:
the recording-level row `present_somewhere` (weak evidence, no time, weight 0), the
inventory's descriptive columns, and the review queue's "credit but models hear no guitar"
reason, which only chooses what a person listens to.

Tests (`tests/test_evidence.py`): agreement plus a recording-level credit stays moderate at
weight 0.25; release- and artist-level credits likewise; agreement plus separation over the
stretch is strong at 0.5; separation covering 35% of the stretch, ending before it, or at an
in-between level does not raise it; no automatic "absent" is training-suitable with or
without credits or separation.

## 2. Candidate counts, before and after

| Class (guitar present) | Before: stretches, hours, recordings | After |
|---|---|---|
| strong agreement | 196, 0.89 h, 80 | 0, 0 h, 0 |
| moderate agreement | 276, 1.13 h, 138 | 472, 2.02 h, 218 |

Every former "strong" stretch owed its class to a credit alone. Automatic "absent": 450
stretches, 1.56 h, all weak evidence (2 of them were "moderate" before). Conflicts and
human references are unchanged. The answered review queue was not regenerated (identical
file hash before and after).

## 3. Rebuilt training set

| Source | Label | Segments | Hours | Leakage groups | Weight |
|---|---|---|---|---|---|
| Human | guitar present | 280 | 0.93 | 33 | 1.0 |
| Human | guitar absent | 105 | 0.35 | 18 | 1.0 |
| Automatic, strong agreement | guitar present | 0 | 0 | 0 | 0.5 |
| Automatic, moderate agreement | guitar present | 427 | 1.72 | 29 | 0.25 |
| Automatic, any class | guitar absent | 0 | | | not admitted |

Before the correction the automatic part was 173 segments at 0.5 and 254 at 0.25: the same
427 segments and 1.72 h, with a weighted total of 0.62 h against 0.43 h now. 14
"intermittent" human answers are excluded. 315 recordings contribute segments.

## 4. Folds and leakage audit

| Fold | Groups with segments | Human segments | Unbiased pool: guitar / none | Model-selected pool: guitar / none | Automatic segments (hours) |
|---|---|---|---|---|---|
| 0 | 7 | 57 | 18 / 16 | 15 / 8 | 53 (0.19) |
| 1 | 7 | 70 | 25 / 13 | 26 / 6 | 179 (0.87) |
| 2 | 7 | 89 | 47 / 12 | 22 / 8 | 54 (0.16) |
| 3 | 7 | 90 | 42 / 13 | 28 / 7 | 64 (0.23) |
| 4 | 8 | 79 | 27 / 10 | 30 / 12 | 77 (0.28) |

The fold assignment is identical to the one built before the correction (it depends on
human excerpts and automatic time, neither of which changed).

Checked by `gmv evidence trainset` (no problems found): no recording in two folds; no pair
of identical audio in two folds; no automatic segment overlapping a reviewed excerpt; no
overlapping reviewed excerpts with different answers; no automatic segment that is not
"present"; no automatic segment in an evaluation pool.

Checked again when the audio was cached (`cache_review.py`): every source file's SHA-256
matched the inventory; a table with an automatic "absent" row, a scored automatic row, a
weight that does not match its source, or one recording in two folds is refused.

Checked again by each run before it loads a model, and by the summary before it compares
anything: a run's training segments and the excerpts it scores share no recording, no
leakage group, no fold and no same-audio group.

## 5. Automatic negatives

None, at three levels: fusion marks every automatic "absent" as weak evidence with weight
0; the training table admits automatic segments only as "present" of an agreement class;
the cache and the run refuse a table or a segment list that says otherwise. All 105
guitar-free segments were confirmed by ear.

## 6. Seeds and reproducibility

- Seed of a stream: `numpy.random.SeedSequence([20261122, fold, stream_index])`, streams in
  the order plan, moisesdb, review_absent, review_present, augment, torch. The arm is not an
  input, so H and H+A of a fold have the same six seeds; the five folds have different ones.
- There is no random initialisation. Every run loads `guitar_mn10_ft_all.pt` (SHA-256
  `eb009fa091f15c491d40db762301b4494c80be9ce51cb305f25b53afd02429df`), the file
  `guitar_mn10_v1` was exported from; a run with other weights is refused by the launcher
  and rejected by the summary.
- Tested with faked MoisesDB clips: over 12 batches of a fold, every MoisesDB clip, every
  guitar-free clip, every gain and every silenced lead-in is bit-identical between the
  arms; only the guitar clips of the owner's recordings differ; a repeat of an arm is
  bit-identical; another fold differs.
- Each run writes `run.json` (settings, seeds, hashes of starting weights, segment table and
  fold assignment, ids of training segments and scored excerpts, code commit, versions,
  clips drawn by kind, hash of the weights written).
- The summary maps every excerpt to the one run of each arm that kept its fold out, names
  those runs in its report, and refuses to compare if any run is incomplete, used other
  weights, settings, seeds or segment table, trained on something its arm may not have, or
  scored something it trained on.
- Dry runs of H and H+A for fold 0 (model loaded, two batches drawn, loss computed without
  gradient, three excerpts scored, nothing written but a report): equal seeds, equal
  settings, equal scored excerpts (57), equal scores from the shared starting weights; H
  loaded 247 guitar and 81 guitar-free human segments from 28 and 16 groups; H+A loaded the
  same plus 374 automatic segments from 24 groups at weight 0.25; the training loss computed
  by torch matched the reference (`review_data.weighted_bce`) to 1e-7, so segment weights
  act on the loss and are not merely stored.
- A training command without `--owner-approved` stops before loading anything; a run never
  overwrites a weights file.

Full bit-for-bit repeatability of the trained weights additionally depends on the same
machine, library versions and thread count (recorded per run); it is expected, not proven.

## 7. Fixed thresholds and what will be reported

Read from `guitar_mn10_v1`'s description at report time: **detected 0.926, candidate 0.627**
(guitar label). Applied unchanged to baseline, H and H+A. For each pool and split in the
plan: AUC; recall at 5% and 10% false alarms; recall and false-positive rate at "detected";
recall and false-positive rate at "candidate"; with the numbers of groups, excerpts and hops
behind them; paired changes against the baseline with 95% intervals from 2,000 resamples of
leakage groups. No threshold is fitted or chosen.

### Baseline on the experiment's own path

The starting weights were scored on every human-labelled excerpt and on the control fold
(inference only, 23 minutes). On 120 contexts the starting checkpoint and the shipped ONNX
`guitar_mn10_v1` gave the same scores to five decimals, so "baseline" is the shipped model.
Intervals are 95% over resampled leakage groups.

| | Primary pool | Secondary pool | MoisesDB control |
|---|---|---|---|
| Guitar / guitar-free excerpts | 159 / 64 | 121 / 41 | 45 songs and their guitar-free twins |
| Groups (with guitar / with guitar-free) | 19 (18 / 12) | 33 (32 / 15) | |
| Hops, guitar / guitar-free | 3,813 / 1,533 | 2,904 / 984 | |
| AUC | 0.854 (0.79 to 0.90) | 0.891 (0.83 to 0.95) | 0.986 |
| Recall at 5% false alarms | 0.585 (0.43 to 0.72) | 0.533 (0.34 to 0.89) | 0.935 |
| Recall at 10% false alarms | 0.680 (0.55 to 0.79) | 0.750 (0.51 to 0.94) | 0.960 |
| Recall at detected (0.926) | 0.546 (0.42 to 0.65) | 0.880 | 0.749 |
| False positives at detected | 0.037 (0.01 to 0.08) | 0.171 | 0.002 |
| Recall at candidate (0.627) | 0.720 (0.63 to 0.79) | 0.914 | 0.898 |
| False positives at candidate | 0.138 (0.06 to 0.27) | 0.218 | 0.022 |

Primary pool by guitar type (AUC against all 64 guitar-free excerpts; recall at detected):
distorted 0.834, 0.485 (71 excerpts); clean 0.885, 0.639 (68); acoustic 0.854, 0.431 (10).
Excerpts starting in the first 10 s: AUC 0.756, recall at detected 0.335 (26 guitar, 16
guitar-free); later excerpts: 0.874 and 0.587 (133, 48).

The secondary pool's threshold figures reflect how it was selected (the "models disagree"
excerpts were picked where this model exceeds 0.926) and say nothing about the model in
general. These figures are lower than the excerpt-level ones quoted earlier for the same
excerpts (AUC 0.89) because they count every hop and use the application's levelling.


## 8. Pass and fail

For H+A against the baseline, all of:

| # | Criterion | Fails when |
|---|---|---|
| 1 | AUC gain on the primary pool | the 95% interval of the change does not lie above zero |
| 2 | Recall at 5% false alarms, primary pool | it declines |
| 3 | MoisesDB control fold | mean AUC of the five runs falls by more than 0.01 |
| 4 | No regression on the secondary pool | the whole 95% interval of its AUC change is below zero |
| 5 | False positives at shipped "detected", primary pool | the 95% interval of the change lies above zero, or the rise exceeds 0.02 |

A criterion that cannot be evaluated is reported as "cannot be evaluated" and the overall
result is then "cannot be decided", never "pass". Wrong-way movement of recall at either
threshold or of false positives at "candidate" outside its interval is flagged regardless.
Criterion 4 can only fail or not fail; it is never evidence of improvement.

## 9. Tests

`pytest -q` in the application environment: **231 passed, 1 skipped**. The skipped test
compares the training script's torch loss with the reference and needs torch, which that
environment does not have; the same comparison ran inside both dry runs and matched. `ruff
check .` is clean. New tests: 4 for the fusion scope rules, 2 for the set assembly, 17 in
`tests/test_review_experiment.py` (arm-specific data, refusal of automatic negatives, fold
isolation, seeds, identical shared randomness between arms, group-wise sampling, segment
weights in targets and loss, fixed thresholds, paired group bootstrap, each pass criterion
failing on its own, cross-checks of runs, the launcher's commands).

## 10. Running it after approval

Everything is in `experiments/recognition/review_experiment_v2.json`. From the checkout:

```
.venv\Scripts\python experiments\recognition\run_review_experiment.py --list
.venv\Scripts\python experiments\recognition\run_review_experiment.py --owner-approved
```

`--list` prints the twelve commands without running them: the baseline scoring (already
done), ten training runs in the order H then H+A for folds 0 to 4, and the summary. Each
training command is

```
.separation-venv\Scripts\python experiments\recognition\finetune_review.py
    --jobs local-evidence/guitar/jobs_moisesdb.json
    --parts E:/datasets/gmv-cache/recognition/parts
    --labels E:/datasets/gmv-cache/recognition/labels
    --source E:/datasets/gmv-cache/recognition/EfficientAT-src
    --init .model-cache/guitar_mn10_ft_all.pt
    --review E:/datasets/gmv-cache/recognition/trainset_v1
    --runs research-local/evidence-v1/trainset-v1/runs
    --weights-dir .model-cache/v2-experiment
    --fold 0/4 --steps 500 --batch 16 --rate 5e-05 --review-share 0.4 --silence 0.15
    --hop 0.5 --threads 4
    --arm ARM --hold FOLD --owner-approved
```

with ARM `H` or `H+A` and FOLD 0 to 4, followed by

```
.separation-venv\Scripts\python experiments\recognition\review_summary.py
    --review E:/datasets/gmv-cache/recognition/trainset_v1
    --runs research-local/evidence-v1/trainset-v1/runs
    --model-config .model-cache/guitar_mn10_v1.json
    --report research-local/evidence-v1/trainset-v1/runs/summary.json
```

Fold weights go to `.model-cache/v2-experiment/` (ignored by Git). They are experiment
checkpoints, not a release: no `guitar_mn10_rv2` is produced by these commands.

Expected cost: about half an hour per training run on this machine (the first
experiment's 500 steps took 17 minutes; scoring a fold and the control took 23 minutes for
the baseline, which scores five times as many excerpts), so roughly five hours for the ten
runs. This is an estimate, not a measurement of this configuration.

## Remaining limitations

- 19 groups in the primary pool; only large effects will be visible.
- No "strong" automatic class exists; H+A adds moderate candidates only.
- Automatic positives are stretches the baseline already detects.
- The MoisesDB control fold was part of the baseline's own training data.
- Criterion 5 rests on 64 guitar-free excerpts from 12 groups (baseline 3.7% of their
  hops above "detected"); it can detect a clear rise, not a small one.
- The secondary pool cannot support a claim of improvement.
- Real-time behaviour in the application is not tested by this experiment.
