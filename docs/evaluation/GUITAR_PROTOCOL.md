# Guitar recognition: test protocol

Fixed on 2026-11-17, before any new model was scored. Changes after that date are listed at the end with their reason.

## Targets

| Measure | Target | Measured on |
|---|---:|---|
| Recall where guitar is present | ≥ 0.90 | guitar-present time of real mixes |
| False-positive rate where guitar is absent | ≤ 0.05 | guitar-free recordings and guitar-free time, real audio |
| F1 | ≥ 0.85 | real mixes |
| Delay of a guitar entrance | preferably ≤ 150 ms | entrances after ≥ 1 s without guitar |
| Sustained false activation | none | longest false run per guitar-free recording |

A model that misses a target is reported as missing it. It is not enabled by default.

## Material and roles

| Source | Audio | Reference | Role |
|---|---|---|---|
| MoisesDB, train + validation (160 songs, 33 artist groups) | real | stems (`stem_energy_v1`); bleed-flagged guitar time left out | training and model selection, by artist folds |
| MoisesDB **twins**: each song summed without its guitar files | real stems, re-summed | guitar absent by construction | guitar-free counterpart of every song; stays in the song's split |
| MoisesDB, held-out artists (48 songs + twins) | real | stems | test. **Already read three times** by earlier detectors; reported as "previously read" |
| MedleyDB 1.0 + 2.0 (pending the owner's access request) | real | typed stems, bleed flag | **final test, never used for training or selection** |
| MusicNet, URMP, The Spheres | real | no guitar in the dataset | guitar-free orchestral, chamber and piano; split by piece, part for selection and part for test |
| Slakh2100 | **synthesized** | MIDI program per stem + stem energy | training help and a synthetic-only test; **never pooled with real audio** |
| The owner's two folders (guitar-heavy, guitar-free) | real | folder = whole-track yes / no | real-world check only; reported apart, nothing committed |

Rules:

- Splits are by artist (or piece). A song, its twin and any excerpt of it are always in the same split.
- Selection (model, window length, features, thresholds, smoothing) uses only training and validation material. Because MoisesDB validation is 13 songs by 6 artists, selection uses four artist folds over the 160 train + validation songs.
- A test set is read once for the selected model and once for each baseline, with everything fixed.
- Twins can contain faint guitar where a non-guitar file has bleed (125 vocal files are flagged). This makes the twin false-alarm figure slightly pessimistic; it is not corrected.

## Two references

- **Stem energy**: guitar is present where the summed guitar files are above max(-60 dBFS, their peak - 40 dB). Same definition as all earlier reports.
- **Audible**: present where the guitar files are within 15 dB of the mix; absent where more than 30 dB below; time in between is not scored. Reported next to the first to show how much a quiet, masked guitar changes the numbers.

## Pools, always reported separately

1. **Guitar-present**: real mixes, time where the reference says guitar. Gives recall and false-negative rate.
2. **Guitar-absent, same songs**: guitar-free time inside real mixes and the twins.
3. **Guitar-absent, other music**: guitar-free songs of the datasets, orchestral / chamber / piano sets, the owner's guitar-free folder.
4. **Synthetic** (Slakh), present and absent.

"Always on" and "always off" rows are included in every table.

## Conditions

Tagged automatically from the stems each recording has and from the reference:

| Condition | Rule |
|---|---|
| Guitar-dominant | guitar within 6 dB of the mix for most of its active time |
| Full rock mix | guitar + bass + drums present |
| Dense distorted | distorted guitar active together with bass and drums |
| Clean guitar with other instruments | only clean or acoustic guitar files, other stems present |
| Guitar-free, bass-heavy | twin or guitar-free song with bass and drums |
| Guitar-free, keys / synth | guitar-free with piano or other keys stems |
| Guitar-free, orchestral / sustained | bowed strings or wind stems; orchestral datasets |
| Guitar enters or exits | windows within 2 s of a reference change after ≥ 1 s of the other state |
| Low guitar level | guitar 15 to 30 dB below the mix |

Genre comes from the dataset where it has one. There is no metal-specific stem dataset; "dense distorted" and the owner's folder are the stand-ins and are named as such.

## Measures

Per pool and condition: precision, recall, F1, false-positive rate, false-negative rate, ROC-AUC, PR-AUC; entrance delay (median and 90th percentile); state changes per minute; longest false activation per guitar-free recording; share of guitar-free recordings lit more than 20% of the time. Recordings are also averaged one by one (macro) so that long, guitar-heavy songs do not decide the result.

Detected / candidate / absent: the "detected" threshold is the one that gives at most 5% false alarms on the validation guitar-absent pools; "candidate" is a lower threshold fixed on validation as well. Scores are reported for "detected" alone and for "detected or candidate".

## Not covered

- No listening test is part of this protocol.
- Real metal with stem-level truth does not exist in the material.
- Cambridge-MT based sets are not used (terms unclear for research).

## Changes after 2026-11-17

None yet.
