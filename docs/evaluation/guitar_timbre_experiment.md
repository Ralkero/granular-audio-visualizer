# Guitar detection with finer features: experiment, negative result

**Verdict: no usable detector.** Three variants were compared on folds split by artist. None is clearly better than the first detector ([moisesdb_guitar_v1.md](moisesdb_guitar_v1.md)), so nothing was added to the application and the held-out test songs were not read again.

## What was tried

Code: `experiments/guitar/`. Data: the 160 MoisesDB train and validation songs (33 artist groups), four folds by artist; each fold trains on three quarters of the artists and scores the real mixes of the rest.

- **Features:** 96 semitone bands from a 93 ms window every 20 ms, overall level removed causally.
- **Model:** four causal dilated convolutions over time (about 1.2 s of history, 185,000 parameters), trained with pitch-shift augmentation by sliding the band axis.
- **A, plain:** trained on the real mixes and their guitar-less remixes.
- **B, cross-artist mixes:** 70% of training excerpts are built on the fly from one recording's backing and another recording's guitar (band power adds approximately), at random levels, or with no guitar at all.
- **C, audible target:** as B, but the target is "guitar within 15 dB of the whole mix"; more than 30 dB below counts as absent and in-between frames are not used. Scored against both definitions.

## Results per fold: AUC / recall / false-positive rate at score 0.5

Reference: stem energy (the same definition as the first detector).

| Variant | Fold (songs) | Guitar | Distorted | Clean | Acoustic |
|---|---|---|---|---|---|
| A | 0 (45) | 0.760 / 0.68 / 0.28 | 0.846 / 0.72 / 0.16 | 0.423 / 0.46 / 0.56 | 0.701 / 0.66 / 0.35 |
| A | 1 (37) | 0.886 / 0.80 / 0.17 | 0.892 / 0.85 / 0.22 | 0.707 / 0.60 / 0.31 | 0.726 / 0.65 / 0.29 |
| A | 2 (66) | 0.643 / 0.59 / 0.43 | 0.734 / 0.49 / 0.18 | 0.446 / 0.33 / 0.42 | 0.682 / 0.62 / 0.36 |
| A | 3 (12) | 0.747 / 0.72 / 0.27 | 0.971 / 0.94 / 0.12 | 0.228 / 0.04 / 0.35 | 0.891 / 0.86 / 0.26 |
| B | 0 | 0.758 / 0.64 / 0.23 | 0.880 / 0.74 / 0.12 | 0.486 / 0.52 / 0.51 | 0.703 / 0.64 / 0.32 |
| B | 1 | 0.859 / 0.78 / 0.22 | 0.871 / 0.83 / 0.24 | 0.649 / 0.47 / 0.27 | 0.765 / 0.57 / 0.18 |
| B | 2 | 0.770 / 0.65 / 0.26 | 0.756 / 0.46 / 0.14 | 0.480 / 0.49 / 0.51 | 0.740 / 0.71 / 0.35 |
| B | 3 | 0.719 / 0.72 / 0.34 | 0.982 / 0.97 / 0.13 | 0.184 / 0.02 / 0.29 | 0.785 / 0.78 / 0.28 |
| C | 0 | 0.685 / 0.32 / 0.13 | 0.893 / 0.55 / 0.04 | 0.463 / 0.25 / 0.30 | 0.702 / 0.41 / 0.17 |
| C | 1 | 0.815 / 0.45 / 0.07 | 0.834 / 0.67 / 0.15 | 0.571 / 0.18 / 0.11 | 0.766 / 0.33 / 0.09 |
| C | 2 | 0.715 / 0.32 / 0.15 | 0.712 / 0.28 / 0.08 | 0.488 / 0.14 / 0.20 | 0.725 / 0.52 / 0.21 |
| C | 3 | 0.722 / 0.51 / 0.10 | 0.981 / 0.92 / 0.05 | 0.290 / 0.00 / 0.09 | 0.737 / 0.50 / 0.20 |

Variant C against its own "audible" definition: guitar AUC 0.761, 0.846, 0.746, 0.795; distorted 0.904, 0.865, 0.790, 0.984; clean 0.578, 0.626, 0.539, 0.278; acoustic 0.766, 0.792, 0.759, 0.745.

For comparison, the first detector (broad-band features, one hidden layer) had guitar AUC 0.80 on its validation songs and 0.85 on the held-out songs. The folds here are different songs, so the comparison is rough.

## Reading the numbers

- **Guitar (any kind)** stays around AUC 0.75 to 0.85 whatever the features, the training mixes or the target. Cross-artist mixing evened out the worst fold (0.64 to 0.77) and nothing more.
- **Distorted guitar** is the only label with a consistent signal (0.71 to 0.98), and results still swing widely with which artists are held out.
- **Clean electric guitar** is at or below chance in most folds. In fold 3 it is far below chance, meaning the held-out artists' clean guitar looks like what the training artists use for something else.
- **The audible target** cut false alarms roughly in half and cut recall by about as much: a more cautious detector, not a better one.
- Results vary more between folds than between methods. With about 25 training artists per fold, which artists are held out matters more than the model.

## What this suggests

A small network trained from scratch on about thirty artists does not learn "guitar" in a way that carries to new artists. Options that could, none of them tried here:

- **A pretrained separation model with a guitar output** (for example the six-source HTDemucs variant) in the delayed-analysis path, like the drums / bass / vocals comparison in [musdb18hq_test_separation_activity.md](musdb18hq_test_separation_activity.md): seconds late, heavy on the CPU, and its guitar quality would have to be measured first.
- **A pretrained audio embedding** as the front end, with a small classifier trained on MoisesDB: also a delayed-path cost, and each model's licence would need checking.
- **More multitrack data** with typed guitar stems (MedleyDB needs an access request).

## Limits of this experiment

- Model selection was not exhaustive: one architecture, one training length. The loss was still falling at the end of training in variants B and C.
- Mixing in the band-power domain ignores phase; real mixes were still 30% of the training excerpts.
- The 32 MoisesDB songs with unequal file lengths were not used.
- Nothing here was listened to.
