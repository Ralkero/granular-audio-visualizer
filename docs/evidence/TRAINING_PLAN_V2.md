# Second training experiment with the owner's recordings: audited sets and plan

**Status: proposed. Nothing has been trained.** `guitar_mn10_v1` remains the model the
application starts with; `guitar_mn10_rv1` stays a separate local alternative. This page
records what the review of the evidence queue showed, the sets built from it, their audit,
and the experiment that would be run on approval. Recording and artist names stay local
(`research-local/evidence-v1/trainset-v1/`).

## What the review queue showed

The owner answered 167 of 168 excerpts (12 s each) chosen by `gmv evidence fuse`.

| Automatic class audited | Excerpts | Right | Precision | 95% interval |
|---|---|---|---|---|
| Guitar, strong agreement | 31 | 31 | 100% | 89 to 100% |
| Guitar, moderate agreement | 40 | 39 | 97.5% | 87 to 99.6% |
| No guitar, weak evidence | 40 | 32 | 80% | 65 to 90% |

- Both "guitar" classes were right on all 19 audited excerpts by artists not used in the
  first experiment. The single error is a synthesizer piece.
- "No guitar" is wrong one time in five, as on the pilot labels, and 5 of 9 right on
  artists not used before. The misses are electric guitars under bass and synthesizer.
  It is not used as a label.
- Where the recogniser reported guitar and PANNs heard none (39 excerpts), guitar was
  audible in 32, twelve of them acoustic. Where the recogniser reported distorted guitar
  and PANNs was silent (15), guitar was audible in 14.

## The sets (`gmv evidence trainset`)

One table of timed segments, each with label, source, weight, leakage group and fold.

| Source | Segments | Guitar | No guitar | Hours | Groups | Weight |
|---|---|---|---|---|---|---|
| Human, chosen without model output (pilot batch) | 223 | 159 | 64 | 0.74 | 19 | 1.0 |
| Human, chosen because of model output (queue) | 162 | 121 | 41 | 0.54 | 33 | 1.0 |
| Automatic, strong agreement | 173 | 173 | 0 | 0.76 | 15 | 0.5 |
| Automatic, moderate agreement | 254 | 254 | 0 | 0.97 | 24 | 0.25 |

Rules, enforced in code and tested:

- A person's answer wins. Every part of an automatic candidate that overlaps a reviewed
  excerpt is cut away; remainders shorter than 6 s are dropped.
- Automatic segments are "guitar present" only. Every guitar-free segment was confirmed by
  ear.
- "Intermittent" answers (14) are left out: the answer is not timed precisely enough.
- The 45 leakage groups are placed whole into 5 folds, largest first, each into the fold
  that keeps the counts most even, with confirmed guitar-free excerpts weighted highest.
  Every fold holds 10 to 16 guitar-free excerpts of the unbiased pool.
- Only human answers are scored against. Automatic segments are never evaluation material.

## Audit

Hard checks, all passed: no recording and no same-audio pair in two folds; no automatic
segment overlapping a reviewed excerpt; no automatic "absent"; no overlapping reviewed
excerpts with different answers; every recording has a leakage group.

Findings that limit what the sets can show:

1. **The unbiased evaluation pool is still 19 groups.** The queue added 15 groups, but its
   excerpts were chosen by the models' output, so on them the current model is at its
   ceiling or floor by construction. They can show a regression, not a gain.
2. **Guitar-free material is concentrated.** Three groups, two of them game soundtracks, supply half of the
   105 guitar-free excerpts; 18 of 45 groups have any. Two of the new ones are not music
   (a silent stretch, a spoken outro).
3. **Automatic positives are concentrated and easy.** One instrumental artist supplies 44%
   of automatic time, and every automatic segment is one the current model already scores
   above its "detected" threshold. They may teach it little; the experiment measures that
   instead of assuming it.
4. Expected wrong labels among automatic segments: at most about 13% for the moderate
   class and 11% for the strong class (upper ends of the audit intervals); observed 1 of 71.
5. 44 reviewed excerpts start in the first 10 s of a recording, the known weak case.
6. 11 groups have no human answer at all; they contribute automatic positives only.

## Baseline (`guitar_mn10_v1`, stored timelines, no training)

| Pool | Guitar / no guitar | AUC (95% over groups) | Recall at 5% false alarms |
|---|---|---|---|
| Unbiased | 159 / 64 | 0.89 (0.82 to 0.94) | 0.62 (0.49 to 0.81) |
| Model-selected | 121 / 41 | 0.90 (0.83 to 0.96) | 0.56 (0.39 to 0.94) |

At the shipped "detected" threshold the unbiased pool gives recall 0.47 with no false
alarms; the model-selected pool is not meaningful there (it was selected at that
threshold). These intervals are wide: a change smaller than about 0.03 in AUC will not be
distinguishable from chance.

## The experiment (to run only on approval)

Fixed in advance; no other configuration will be tried and none will be chosen afterwards.

- **Two arms, same settings.** H: human segments only. H+A: human segments plus automatic
  positives at weights 0.5 and 0.25. The difference between the arms is the value of the
  automatic candidates; the difference between H and "before" is the value of the larger
  human set (385 excerpts in 34 groups, against 223 in 19 last time).
- **Five-fold, by leakage group.** Each arm is trained five times, each time without one
  fold, from the weights that never saw MoisesDB artist fold 0 (as in the first
  experiment). "Before" is that starting model on the same excerpts.
- **Settings as before:** 500 steps, batch 16, rate 5e-5, 40% of clips from the owner's
  recordings, 15% silenced lead-ins. Two changes, both to counter the concentration found
  in the audit: clips are drawn group by group (every group equally often) instead of
  recording by recording, and each column's loss is multiplied by its segment weight.
- **Scoring.** Out-of-fold only. Primary: the unbiased pool (AUC, recall at 5% and 10%
  false alarms, by guitar type, first 10 s separately), with the change and its 95%
  interval from resampling groups. Secondary, reported apart and per selection reason: the
  model-selected pool. Control: MoisesDB fold 0 with guitar-free twins.
- **A gain counts only if** the 95% interval of the change in AUC on the unbiased pool lies
  above zero, recall at 5% false alarms does not fall, MoisesDB fold 0 loses no more than
  0.01 AUC, and the model-selected pool shows no drop outside its interval.
- **If it counts:** final weights trained on everything from `guitar_mn10_v1`, kept local
  as `guitar_mn10_rv2`, selectable, not the default. Whether it becomes the default is a
  separate decision after listening in the application.
- **If it does not:** reported as measured. The next step would then be confirmed
  guitar-free material from more artists (OpenMIC's human-labelled clips are the cheapest
  source), not more automatic positives.

Work before the first run: `cache_review.py` reads the segment table (automatic segments
included); `finetune_review.py` takes the fold column, segment weights and group-wise
sampling; `review_summary.py` reports five folds and both pools. Ten training runs on CPU.

## Not available

AcoustID results (needs the owner's key), MedleyDB (needs the owner's access request),
guitar-type labels for automatic segments (the models' type output was not audited).
