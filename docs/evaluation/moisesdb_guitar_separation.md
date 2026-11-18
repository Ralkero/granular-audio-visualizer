# Guitar activity from a separation model's guitar output (MoisesDB)

**Verdict: finds much more of the guitar than the trained detectors, and is still not specific enough to light a lane named "guitar".** Nothing was added to the application.

## What was measured

- Separator: six-source HTDemucs (`htdemucs_6s/5c90dfd2`, Demucs 4.1.0, CPU, `shifts=0`). Weights are not in the repository; their licence is unresolved upstream.
- Songs: the same 13 validation and 48 held-out MoisesDB songs as [moisesdb_guitar_v1.md](moisesdb_guitar_v1.md), **first 150 seconds of each**. Only level curves of the separated outputs were kept, no separated audio.
- Two passes: the whole excerpt at once ("whole track"), and a live-style pass that every 2 s separates only the last 7.8 s and keeps the newest 2 s.
- The guitar output counts as active where its level is above the dataset's floor (-60 dBFS) and within a margin of an anchor (its own peak, its running peak, or the mix level). The margin was chosen on the validation songs and applied unchanged to the held-out songs.
- Reference: activity of the dataset's own guitar files; time where only a bleed-flagged guitar file is active is left out (22 of 120 minutes).

Code: `experiments/separation/activity_worker.py --six-source`, `experiments/separation/score_guitar.py`.

## Held-out songs (48 songs, 10 artists, 98 scored minutes, guitar active 82% of the time)

| Pass / anchor | Margin | Precision | Recall | F1 | Always-active F1 | False-positive rate | Balanced accuracy | Lit in the one song without guitar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Whole track / own peak | 40 dB | 0.944 | 0.925 | 0.935 | 0.902 | 0.252 | 0.837 | 66% |
| Whole track / mix | 45 dB | 0.944 | 0.926 | 0.935 | 0.902 | 0.252 | 0.837 | 66% |
| Live-style / running peak | 35 dB | 0.944 | 0.920 | 0.932 | 0.902 | 0.254 | 0.833 | 62% |
| Live-style / mix | 40 dB | 0.944 | 0.921 | 0.932 | 0.902 | 0.255 | 0.833 | 61% |

Validation (13 songs): balanced accuracy 0.896 / 0.892, false-positive rate 0.16. Median song F1 on the held-out songs is 0.99; two songs are below 0.5.

How the margin trades recall against false alarms (held-out songs, whole track / own peak; computed after the fact, not a result):

| Margin | Recall | False-positive rate |
|---:|---:|---:|
| 10 dB | 0.608 | 0.037 |
| 15 dB | 0.789 | 0.083 |
| 20 dB | 0.868 | 0.136 |
| 25 dB | 0.899 | 0.188 |
| 40 dB | 0.925 | 0.252 |

The first trained detector ([moisesdb_guitar_v1.md](moisesdb_guitar_v1.md)) had recall 0.719 and false-positive rate 0.167 on the same songs. That figure covers whole songs per 20 ms frame, this one the first 150 s per interval, so the comparison is rough; still, at a 20 dB margin the separation output finds 87% of the guitar at a lower false-alarm rate (14%).

## Reading the numbers

- **Recall is high and the live-style pass costs almost nothing in accuracy** (0.92 against 0.93).
- **It fires in a quarter of the guitar-free time** at the chosen margin, and in the only held-out song without any guitar the output is above the floor for about two thirds of the song. The model puts other instruments (most likely keyboards and similar) into its guitar output at a low level. One song is far too little to put a number on this.
- The selected margins are so wide that the rule is effectively "anything above the floor". Balanced accuracy on validation rewarded that because guitar plays 82% of the time there. A tighter margin gives far fewer false alarms at lower recall, but that choice was not made in advance, so it is not reported as a result.
- Because guitar is almost always present in this dataset, "always on" already scores F1 0.90. The dataset cannot show how the output behaves on guitar-free music.
- No distinction between distorted, clean and acoustic guitar: the model has one guitar output.
- Whether these songs overlap with what the six-source model was trained on is not known. If they do, the figures are optimistic.

## Cost

- Whole-track pass: about 0.25 s of computation per second of audio (30 s probe, 8 threads, Ryzen 9 5950X).
- Live-style pass: about 1.1 s of computation per 2 s step with 8 threads and nothing else running, so it keeps up, using roughly half of those 8 cores continuously. During the full run (three workers at 5 threads each) it took 3.2 s per 2 s step and would not have kept up.
- Activity would appear 2 to 4 seconds after the sound. This can only be a delayed lane.

## Not done

- Nothing was listened to.
- Only the first 150 s of each song; the 32 songs with unequal file lengths are unused.
- No test on music without guitar beyond one song, and no metal-specific test.
- The model's piano output was not scored.
