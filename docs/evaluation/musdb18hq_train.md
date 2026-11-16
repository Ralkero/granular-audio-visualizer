# Corpus evaluation — train split, musdb18hq

- Status: **evaluated**
- Corpus version 0.1.0, content hash `19e7917900169291`; detector `candidate` at commit `9def999`
- Recordings evaluated: 81; skipped: 0
- Annotated audio evaluated: 286.5 min

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

| Instrument | Recordings | Reference | Reference active | Detector | F1 |
|---|---:|---|---:|---|---:|
| bass | 81 | stem_derived | 14281.3 s | no detector output | — |
| drums | 81 | stem_derived | 14384.1 s | no detector output | — |
| other | 81 | stem_derived | 16047.1 s | no detector output | — |
| vocals | 81 | stem_derived | 10872.5 s | no detector output | — |

## Band-ratio candidates refuted by instrument activity

Events fired while none of the instruments the candidate's name suggests was sounding. This is a lower bound on false positives, not an accuracy.

| Candidate | Events | Refuted | Fraction | Reference |
|---|---:|---:|---:|---|
| `bass_transient_candidate` | 34468 | 798 | 0.023 | stem_derived |
| `snare_like_candidate` | 16281 | 1576 | 0.097 | stem_derived |
| `high_frequency_percussion_candidate` | 1881 | 203 | 0.108 | stem_derived |

## Reference used per layer (recordings)

- `activity.bass`: stem_derived ×81
- `activity.drums`: stem_derived ×81
- `activity.other`: stem_derived ×81
- `activity.vocals`: stem_derived ×81

