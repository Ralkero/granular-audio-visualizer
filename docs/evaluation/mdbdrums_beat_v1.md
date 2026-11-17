# Corpus evaluation — train split, mdbdrums

- Status: **evaluated**
- Corpus version 0.1.0, content hash `d5461b26543169dc`; detector `candidate` at commit `96562b5`
- Recordings evaluated: 23; skipped: 0
- Annotated audio evaluated: 21.8 min

## Event classes (aggregate over evaluated recordings)

| Class | TP | FP | FN | Precision | Recall | F1 | Median abs. error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full-mix onset, ±25 ms | — | — | — | — | — | — | — |
| Full-mix onset, ±50 ms | — | — | — | — | — | — | — |
| Beats, observed, ±70 ms | 1413 | 442 | 1281 | 0.762 | 0.524 | 0.621 | 6.8 ms |
| Beats, predicted, ±70 ms | 1766 | 665 | 928 | 0.726 | 0.656 | 0.689 | 11.1 ms |
| `bass_transient_candidate` vs annotated attacks | 1003 | 1604 | 536 | 0.385 | 0.652 | 0.484 | 6.8 ms |
| `snare_like_candidate` vs annotated attacks | 870 | 1235 | 1784 | 0.413 | 0.328 | 0.366 | 6.8 ms |
| `high_frequency_percussion_candidate` vs annotated attacks | 169 | 50 | 3472 | 0.772 | 0.046 | 0.088 | 6.8 ms |

A dash means the layer was not annotated in any evaluated recording.

## Instrument activity

| Instrument | Recordings | Reference | Reference active | Detector | F1 |
|---|---:|---|---:|---|---:|
| drums | 23 | stem_derived | 1231.8 s | no detector output | — |

## Full-mix onset detector: recall on annotated instrument attacks (±50 ms)

Recall only. Precision cannot be measured where other instruments' onsets are not annotated.

| Instrument attacks | Recordings | Annotated | Detected | Recall | Reference |
|---|---:|---:|---:|---:|---|
| cymbals | 23 | 1002 | 698 | 0.697 | dataset_annotation |
| drums | 23 | 7898 | 5455 | 0.691 | dataset_annotation |
| hi_hat | 23 | 2639 | 1749 | 0.663 | dataset_annotation |
| kick | 23 | 1539 | 1256 | 0.816 | dataset_annotation |
| snare | 23 | 2654 | 1714 | 0.646 | dataset_annotation |
| toms | 23 | 90 | 62 | 0.689 | dataset_annotation |

## Band-ratio candidates refuted by instrument activity

Events fired while none of the instruments the candidate's name suggests was sounding. This is a lower bound on false positives, not an accuracy.

| Candidate | Events | Refuted | Fraction | Reference |
|---|---:|---:|---:|---|
| `bass_transient_candidate` | 2607 | 23 | 0.009 | stem_derived |
| `snare_like_candidate` | 2105 | 26 | 0.012 | stem_derived |
| `high_frequency_percussion_candidate` | 219 | 6 | 0.027 | stem_derived |

## Reference used per layer (recordings)

- `activity.drums`: stem_derived ×23
- `beat`: dataset_annotation ×23
- `downbeat`: dataset_annotation ×23
- `hits.cymbals`: dataset_annotation ×23
- `hits.drums`: dataset_annotation ×23
- `hits.hi_hat`: dataset_annotation ×23
- `hits.kick`: dataset_annotation ×23
- `hits.snare`: dataset_annotation ×23
- `hits.toms`: dataset_annotation ×23

