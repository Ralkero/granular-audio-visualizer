# Corpus evaluation — validation split, musdb18hq

- Status: **evaluated**
- Corpus version 0.1.0, content hash `c9740c25f572e0ef`; detector `candidate` at commit `f466d2b`
- Recordings evaluated: 15; skipped: 0
- Annotated audio evaluated: 77.1 min

## Event classes (aggregate over evaluated recordings)

| Class | TP | FP | FN | Precision | Recall | F1 | Median abs. error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full-mix onset, ±25 ms | — | — | — | — | — | — | — |
| Full-mix onset, ±50 ms | — | — | — | — | — | — | — |
| Beats, observed, ±70 ms | — | — | — | — | — | — | — |
| Beats, predicted, ±70 ms | — | — | — | — | — | — | — |
| `bass_transient_candidate` vs annotated attacks | — | — | — | — | — | — | — |
| `snare_like_candidate` vs annotated attacks | — | — | — | — | — | — | — |
| `high_frequency_percussion_candidate` vs annotated attacks | — | — | — | — | — | — | — |

A dash means the layer was not annotated in any evaluated recording.

## Instrument activity

| Instrument | Recordings | Reference | Reference active | Detector | Precision | Recall | F1 | Always-active F1 | False-positive rate |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|
| bass | 15 | stem_derived | 3418.3 s | scored | 0.967 | 0.888 | 0.926 | 0.850 | 0.086 |
| drums | 15 | stem_derived | 3579.7 s | scored | 0.963 | 0.934 | 0.948 | 0.872 | 0.122 |
| other | 15 | stem_derived | 4472.0 s | no detector output | — | — | — | — | — |
| vocals | 15 | stem_derived | 2939.0 s | scored | 0.883 | 0.813 | 0.847 | 0.777 | 0.187 |

## Band-ratio candidates refuted by instrument activity

Events fired while none of the instruments the candidate's name suggests was sounding. This is a lower bound on false positives, not an accuracy.

| Candidate | Events | Refuted | Fraction | Reference |
|---|---:|---:|---:|---|
| `bass_transient_candidate` | 7868 | 326 | 0.041 | stem_derived |
| `snare_like_candidate` | 6301 | 908 | 0.144 | stem_derived |
| `high_frequency_percussion_candidate` | 547 | 29 | 0.053 | stem_derived |

## Reference used per layer (recordings)

- `activity.bass`: stem_derived ×15
- `activity.drums`: stem_derived ×15
- `activity.other`: stem_derived ×15
- `activity.vocals`: stem_derived ×15

