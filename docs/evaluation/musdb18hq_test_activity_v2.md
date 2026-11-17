# Corpus evaluation — test split, musdb18hq

- Status: **evaluated** (FINAL held-out test evaluation)
- Corpus version 0.1.0, content hash `5639526fb4024c85`; detector `candidate` at commit `46b8370`
- Recordings evaluated: 54; skipped: 0
- Annotated audio evaluated: 225.7 min

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
| bass | 54 | stem_derived | 10246.2 s | scored | 0.927 | 0.917 | 0.922 | 0.861 | 0.223 |
| drums | 54 | stem_derived | 11306.9 s | scored | 0.973 | 0.920 | 0.946 | 0.910 | 0.127 |
| other | 54 | stem_derived | 12648.3 s | no detector output | — | — | — | — | — |
| vocals | 54 | stem_derived | 8198.8 s | scored | 0.858 | 0.777 | 0.815 | 0.754 | 0.197 |

## Band-ratio candidates refuted by instrument activity

Events fired while none of the instruments the candidate's name suggests was sounding. This is a lower bound on false positives, not an accuracy.

| Candidate | Events | Refuted | Fraction | Reference |
|---|---:|---:|---:|---|
| `bass_transient_candidate` | 28008 | 752 | 0.027 | stem_derived |
| `snare_like_candidate` | 16648 | 1681 | 0.101 | stem_derived |
| `high_frequency_percussion_candidate` | 4110 | 253 | 0.062 | stem_derived |

## Reference used per layer (recordings)

- `activity.bass`: stem_derived ×54
- `activity.drums`: stem_derived ×54
- `activity.other`: stem_derived ×54
- `activity.vocals`: stem_derived ×54

