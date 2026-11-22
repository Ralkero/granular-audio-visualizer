# Second training experiment with the owner's recordings: results

**Status: run once, as planned. No model was produced or installed.** `guitar_mn10_v1`
remains the model the application starts with. The ten sets of fold weights are local
intermediate files, not a model. The plan and its fixed rules are in
[TRAINING_PLAN_V2.md](TRAINING_PLAN_V2.md); what was checked beforehand is in
[PRE_TRAINING_VALIDATION_V2.md](PRE_TRAINING_VALIDATION_V2.md).

## What was run

- Ten training runs (arms H and H+A, five folds each), 500 steps, 32 to 35 minutes each,
  5.5 hours in total on CPU, from `review_experiment_v2.json` with no change of settings.
- All ten run records are complete. The summary's cross-checks found no problem: every run
  used the planned starting weights, segment table, fold assignment and settings, the two
  arms of a fold share their seeds, and every excerpt was scored only by the run that kept
  its fold out.
- Scoring is hop-level on the application's path at the shipped thresholds (detected 0.926,
  candidate 0.627). No threshold was fitted. Intervals are 95%, from 2,000 paired bootstrap
  replicates over leakage groups.

## Verdict against the five criteria (H+A against the baseline)

| # | Criterion | Measured | Result |
|---|---|---|---|
| 1 | AUC change on the primary pool, interval above zero | +0.017 (+0.005 to +0.028) | pass |
| 2 | Recall at 5% false alarms on the primary pool does not decline | +0.045 (-0.003 to +0.107) | pass |
| 3 | MoisesDB control loses at most 0.01 AUC | 0.986 to 0.982 (lowest run 0.980) | pass |
| 4 | No regression on the model-selected pool | +0.022 (0.000 to +0.053) | pass |
| 5 | False positives at "detected" on the primary pool do not rise | -0.010 (-0.029 to +0.001) | pass |

All five pass; none was unevaluable. Arm H passes the same five (AUC +0.021, +0.005 to
+0.036). The gain is real by the fixed rule and small.

## Primary pool (chosen without model output: 159 guitar, 64 guitar-free excerpts, 19 groups)

| | AUC | Recall at 5% FA | Recall at 10% FA | Detected: recall / FPR | Candidate: recall / FPR |
|---|---|---|---|---|---|
| Baseline | 0.854 | 0.585 | 0.680 | 0.546 / 0.037 | 0.720 / 0.138 |
| H | 0.875 | 0.641 | 0.722 | 0.577 / 0.031 | 0.763 / 0.151 |
| H+A | 0.871 | 0.630 | 0.713 | 0.545 / 0.027 | 0.730 / 0.116 |

3,813 guitar hops and 1,533 guitar-free hops.

**At the shipped thresholds H+A finds no more guitar than the baseline** (recall at
"detected" -0.000, interval -0.021 to +0.020). Its better ranking shows only as slightly
fewer false positives, and that change's interval includes zero. To turn the AUC gain into
more detected guitar, the thresholds would have to be recalibrated, which is a separate
experiment on separate data.

## What the automatic segments did (H+A against H, primary pool)

| Figure | Change | Interval |
|---|---|---|
| AUC | -0.004 | -0.012 to +0.005 |
| Recall at "detected" | -0.032 | -0.054 to -0.016 |
| Recall at "candidate" | -0.033 | -0.048 to -0.020 |
| FPR at "candidate" | -0.035 | -0.057 to -0.015 |

The 427 automatic "guitar present" segments did not improve ranking. They shifted the
operating point: less guitar found and fewer false positives at both thresholds. The gain
over the baseline comes from the human-confirmed segments.

## Primary pool by guitar type and position (H+A against the baseline)

| Subset | Guitar excerpts | AUC baseline / H+A | AUC change (interval) | Recall at "detected" change (interval) |
|---|---|---|---|---|
| Distorted electric | 71 | 0.834 / 0.859 | +0.025 (+0.006 to +0.038) | +0.019 (-0.002 to +0.042) |
| Clean electric | 68 | 0.885 / 0.899 | +0.015 (+0.000 to +0.032) | -0.007 (-0.026 to +0.023) |
| Acoustic | 10 | 0.854 / 0.843 | -0.011 (-0.028 to +0.006) | -0.067 (-0.136 to +0.028) |
| Starts in the first 10 s | 26 | 0.756 / 0.769 | +0.013 (-0.019 to +0.043) | +0.018 (+0.002 to +0.037) |
| Starts later | 133 | 0.874 / 0.891 | +0.017 (+0.006 to +0.029) | -0.004 (-0.027 to +0.020) |

Each subset is scored against the same 64 guitar-free excerpts (16 for the first 10 s, 48
later). Acoustic guitar shows no gain; with 10 excerpts in 6 groups nothing more can be
said. The first 10 s remain the weakest condition in every arm.

## Secondary pool (chosen because of the baseline's output: 121 guitar, 41 guitar-free, 33 groups)

| | AUC | Detected: recall / FPR | Candidate: recall / FPR |
|---|---|---|---|
| Baseline | 0.891 | 0.880 / 0.171 | 0.914 / 0.218 |
| H | 0.913 | 0.836 / 0.076 | 0.919 / 0.180 |
| H+A | 0.914 | 0.824 / 0.068 | 0.914 / 0.174 |

This pool cannot show an unbiased gain: its excerpts were picked where the baseline was
confident or in conflict. By selection reason, for H+A:

- Random audit of agreed candidates (77 / 33): AUC 0.915 to 0.912 (change -0.003, -0.019 to
  +0.006).
- Models disagreed (30 / 7, 13 groups, too few guitar-free groups for an interval): AUC
  0.706 to 0.832; false positives at "detected" 0.80 to 0.26; recall at "detected" 0.92 to
  0.75.
- Suspected distorted guitar (12 / 1): one guitar-free excerpt, not interpretable.

## MoisesDB control (45 songs and their guitar-free twins, not trained on in this experiment)

| | AUC | Recall at 5% FA | Detected: recall / FPR | Candidate: recall / FPR |
|---|---|---|---|---|
| Baseline | 0.986 | 0.935 | 0.749 / 0.002 | 0.898 / 0.022 |
| H, mean of 5 runs | 0.981 | 0.916 | 0.697 / 0.002 | 0.878 / 0.023 |
| H+A, mean of 5 runs | 0.982 | 0.920 | 0.690 / 0.002 | 0.874 / 0.019 |

The baseline had these songs in its own training data, so this checks retention only.

## Fixed-threshold regressions to weigh

The automatic check for wrong-way movement covers the primary pool and raised no flag.
Outside it, recall at the shipped "detected" threshold fell in both arms:

- **MoisesDB control: 0.749 to 0.690 for H+A** (runs 0.623 to 0.726), 0.697 for H. AUC
  passes criterion 3, but at the shipped threshold the retrained models report "detected"
  on about six points less of the guitar time there.
- **Secondary pool: -0.056 for H+A** (-0.088 to -0.029), -0.044 for H. Part of this is
  expected, since the pool was selected where the baseline scored high, and it comes with
  a larger fall in false positives (-0.103, -0.221 to -0.023).

Recall at "candidate" is unchanged in both places. Read together: the retrained models
score more cautiously, and with the shipped thresholds unchanged they trade some
"detected" time for fewer false positives.

## What this does and does not show

- Shown: training on the owner's human-confirmed excerpts improves ranking on unseen
  groups of the owner's music by about 0.02 AUC, mainly on distorted electric guitar,
  without raising false positives and within the allowed loss on MoisesDB.
- Not shown: more guitar detected at the shipped thresholds; any gain on acoustic guitar
  or in the first 10 s; any benefit from the automatic segments; behaviour on music
  outside the owner's library and MoisesDB. Nobody has listened to a retrained model in
  the application.
- The folds share one library and one annotator. The intervals reflect variation between
  groups of that library, not between libraries.

## Not done

No final weights were trained on all data, no `guitar_mn10_rv2` exists, and no default
changed. Each of these needs the owner's separate approval, as the plan states.
