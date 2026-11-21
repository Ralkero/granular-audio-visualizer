# Fine-tuning the guitar model with reviewed excerpts: first attempt

Result in one line: 223 labelled excerpts from 19 artists did **not** make the guitar model
better on artists it was not shown. `guitar_mn10_v1` stays the model the packaged build
starts with; the fine-tuned weights (`guitar_mn10_rv1`) are kept locally as an alternative
and are not the default anywhere.

## Material

The owner labelled excerpts of their own recordings in `gmv review`. Used here: core
excerpts (chosen without looking at model output) answered "present throughout" (159) or
"absent" (64), from 47 recordings by 19 artists. "Intermittent" (9) and uncertain answers
were not used. Titles, labels and audio stay local.

## Method

- `experiments/recognition/cache_review.py` stores each labelled excerpt with the 10 s
  before it; `finetune_review.py` continues the MoisesDB fine-tuning for 500 steps with 40%
  of the clips taken from reviewed excerpts (guitar and guitar-free equally often,
  recordings equally often) and 15% of all clips silenced before a random point, as at the
  start of a recording.
- Reviewed artists were split into three folds. Three models were trained, each without one
  fold, starting from weights that never saw MoisesDB artist fold 0. Each was scored on its
  kept-out reviewed excerpts and on MoisesDB fold 0 (songs and guitar-free twins); "before"
  is the starting model on the same material.
- Scores are averaged over two hops as in the application. Because a threshold trades recall
  for false alarms, models are compared at equal false-alarm rates.
  `review_summary.py` makes the tables.

## Results (guitar label)

| Held-out pool | Model | AUC | Recall at 5% false alarms | at 10% |
|---|---|---|---|---|
| Reviewed excerpts, artists not trained on | before | 0.844 | 0.59 | 0.68 |
| | after | 0.837 | 0.54 | 0.62 |
| MoisesDB fold 0 with guitar-free twins | before | 0.953 | 0.81 | 0.88 |
| | after | 0.953 | 0.80 | 0.87 |

Resampling the 19 artists gives a 95% interval for the change on reviewed excerpts of
-0.031 to +0.015 in AUC and -0.14 to +0.03 in recall at 5% false alarms: no gain, and the
data cannot rule out a small loss. By fold the change in AUC was +0.021, -0.017 and -0.003.

- Excerpts that begin in the first 10 s of a recording stay the weak case: recall at 5%
  false alarms 0.37 before and after (0.64 and 0.57 for later excerpts). The silenced
  lead-in clips did not fix it.
- Training raised scores of guitar-free reviewed excerpts (+0.06 on average) more than
  scores of guitar excerpts (+0.01).
- Guitar types, against the same guitar-free excerpts: distorted AUC 0.90 before, 0.89
  after; clean 0.40 before, 0.71 after (68 excerpts); acoustic 0.74 and 0.75 (10 excerpts).
  The clean-guitar gain is the one clear change and rests on few artists.

## What this means

- The model's ranking of guitar against no guitar is much worse on ordinary recordings
  (AUC 0.84) than on MoisesDB mixes (0.95). That gap is the problem to solve, and this
  amount of labelled material does not close it.
- 19 artists is too few both to train on and to measure with: only large effects would be
  visible. More artists, and more guitar-free excerpts in particular (64 so far), are
  needed before another attempt is worth scoring.
- No second configuration was tried: with this interval, choosing between configurations
  would be choosing by noise.

## The weights that exist

`guitar_mn10_rv1` was trained on all 223 excerpts, starting from `guitar_mn10_v1`'s
weights. Its thresholds are the scores exceeded in 5% ("detected") and 20% ("candidate") of
guitar-free time in the out-of-fold scores above, the two pools weighted equally. It has
seen every reviewed excerpt, so `gmv review report` on the pilot batch is no longer a test
of it, and no held-out figure exists for these exact weights. It is local only (derived
from MoisesDB, CC BY-NC-SA, and from the owner's recordings).
