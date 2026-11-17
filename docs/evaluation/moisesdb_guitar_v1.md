# Guitar activity detector, first attempt (MoisesDB)

**Verdict: not good enough to show as a guitar detector.** The weights are kept locally as `guitar_experimental_v1.npz`, are not loaded by default, and the guitar lane still says "no detector output" unless a file is chosen in Settings.

## Data

- MoisesDB v0.1 (CC BY-NC-SA 4.0, obtained by the owner through the Music.ai form, 240 songs, 47 artists). Audio and weights stay on this machine.
- 208 songs imported. 32 were rejected because their stem files differ in length by more than 0.5 s; whether those files are aligned was not checked, so they are left out.
- The dataset has no official split. One artist in five is held out as test and one in five is validation, decided by a hash of the artist name: 147 train, 13 validation, 48 test songs. Validation is small (6 artists).
- Reference: activity measured on the dataset's own guitar files (`stem_derived`), with the track type giving distorted, clean or acoustic. 50 guitar files are flagged by the dataset as containing bleed; time where only such a file is active is left out of training and scoring.
- Not handled: bleed in non-guitar files (125 vocal files are flagged). A flagged vocal file can contain faint guitar while the reference says no guitar.

## Model

Same lightweight features and one-hidden-layer network as the drums / bass / vocals detector (`activity_mlp`), four outputs. Training used each train song three times: as is, with all guitar files removed, and with random stem gains and omissions. Five settings (hidden 8 to 128, weight decay 0.01 to 0.3) were compared on validation only; hidden 32 with decay 0.1 was kept. The held-out songs were read once, after that choice.

## Held-out songs (48 songs, 10 artists), per 20 ms frame, real analyzer with hysteresis

Every label is scored on every song; a song without a file of that type counts as "absent".

| Label | Scored | Active share | Precision | Recall | F1 | Always-active F1 | False-positive rate | AUC | Lane lit in songs without it |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| guitar | 157 min | 0.85 | 0.960 | 0.719 | 0.823 | 0.918 | 0.167 | 0.854 | 37% (1 song) |
| distorted | 169 min | 0.37 | 0.648 | 0.773 | 0.705 | 0.535 | 0.241 | 0.837 | 25% (21 songs) |
| clean | 168 min | 0.40 | 0.644 | 0.590 | 0.616 | 0.570 | 0.216 | 0.758 | 24% (19 songs) |
| acoustic | 183 min | 0.22 | 0.511 | 0.740 | 0.605 | 0.355 | 0.195 | 0.834 | 18% (35 songs) |

Validation (13 songs): AUC 0.797 guitar, 0.936 distorted, 0.649 clean, 0.616 acoustic. On the training songs the same model reaches AUC 0.87 to 0.95: it fits songs it has seen far better than new artists.

## Reading the numbers

- **Guitar (any kind):** guitar plays in 85% of the scored time, so "always on" scores F1 0.92 and the detector 0.82. It misses 28% of guitar time and fires in 17% of guitar-free time. Only one held-out song has no guitar at all, so behaviour on guitar-free music is barely tested.
- **Distorted / clean / acoustic:** each lane would be lit about a fifth to a quarter of the time in songs that do not contain that kind of guitar. That is too often to label a lane with the instrument's name.
- Making the network smaller or more regularised did not close the gap between seen and unseen songs. The limit is the features (coarse band energies and their changes), which were designed for drums, bass and vocals and do not separate a guitar from keyboards or strings in the same register.
- MoisesDB is mostly rock, pop and singer-songwriter material. No metal-specific test exists.

## Reproduce

```
gmv corpus --manifest corpus-data/corpus/manifest.json import-dataset moisesdb C:\Datasets\moisesdb_v0.1
python scripts/train_activity.py --corpus corpus-data/corpus/manifest.json --dataset moisesdb --dataset-root C:\Datasets\moisesdb_v0.1 --cache CACHE --output .model-cache/guitar_experimental_v1.npz --labels guitar,guitar_distorted,guitar_clean,guitar_acoustic --name guitar_mlp_v1 --hidden 32 --decay 0.1
python scripts/score_activity_frames.py --corpus corpus-data/corpus/manifest.json --dataset moisesdb --split test --weights .model-cache/guitar_experimental_v1.npz --output OUT.json
```
