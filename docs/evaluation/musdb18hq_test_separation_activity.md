# Instrument activity from HTDemucs stems — MUSDB18-HQ test tracks

- Separator: `htdemucs/955717e8` (Demucs 4.1.0, CPU, `shifts=0`). Weights not in the repository; licence unresolved upstream.
- Tracks: the 50 test tracks from the dataset's own test folder (207.8 min). The 4 project test tracks from the train folder were seen by the separator and are excluded.
- Reference: stem-derived activity (`stem_energy_v1`).
- Margin: chosen per instrument on the validation split (tracks the separator was trained on), applied unchanged.
- Lightweight estimator: `activity_mlp_v2` final-run results (commit `46b8370`) on the same 50 tracks.
- Live-style: 7.8 s segment ending at the present, newest 2 s kept, margin measured from the stem's loudest level so far.

| Instrument | Method | Precision | Recall | F1 | False-positive rate | Worst track F1 |
|---|---|---:|---:|---:|---:|---:|
| Drums | Lightweight estimator | 0.974 | 0.918 | 0.945 | 0.124 | 0.70 |
| Drums | Separation, whole file | 0.978 | 0.977 | 0.977 | 0.113 | 0.83 |
| Drums | Separation, live-style | 0.984 | 0.977 | **0.981** | **0.078** | 0.88 |
| Bass | Lightweight estimator | 0.921 | 0.912 | 0.916 | 0.234 | 0.00 |
| Bass | Separation, whole file | 0.941 | 0.970 | 0.956 | 0.180 | 0.63 |
| Bass | Separation, live-style | 0.956 | 0.971 | **0.963** | **0.134** | 0.62 |
| Vocals | Lightweight estimator | 0.858 | 0.786 | 0.820 | 0.212 | 0.25 |
| Vocals | Separation, whole file | 0.876 | 0.954 | 0.913 | 0.220 | 0.38 |
| Vocals | Separation, live-style | 0.885 | 0.955 | **0.918** | 0.203 | 0.45 |

Always-active F1: 0.910 (drums), 0.856 (bass), 0.765 (vocals).

## Margin sensitivity, whole file (read off the test tracks afterwards; not a result)

| Margin below the stem's own peak | Drums recall / false-positive rate | Bass | Vocals |
|---|---:|---:|---:|
| 25 dB | 0.84 / 0.01 | 0.93 / 0.07 | 0.87 / 0.04 |
| 30 dB | 0.92 / 0.02 | 0.95 / 0.11 | 0.89 / 0.05 |
| 35 dB | 0.95 / 0.05 | 0.96 / 0.14 | 0.92 / 0.10 |
| 40 dB (chosen for drums, bass) | 0.98 / 0.11 | 0.97 / 0.18 | 0.94 / 0.14 |
| 45 dB (chosen for vocals) | 0.99 / 0.18 | 0.97 / 0.20 | 0.95 / 0.22 |
| Lightweight estimator, for comparison | 0.92 / 0.12 | 0.91 / 0.23 | 0.79 / 0.21 |

## Cost (Ryzen 9 5950X, CPU only)

| Threads | Whole file | One live-style pass per 2 s of new audio |
|---:|---:|---:|
| 4 | 0.41 s per second of audio | 2.33 s |
| 8 | 0.25 s per second of audio | 1.41 s |
