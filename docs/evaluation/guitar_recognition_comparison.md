# Guitar recognition: comparison of candidates (selection stage)

**Outcome: a fine-tuned EfficientAT `mn10` is selected. It is clearly better than every earlier detector and does not reach the targets.** At the threshold that allows 5% false alarms it finds about two thirds of the guitar. Everything here was measured on training and validation songs split by artist; no test recording was read. Protocol: [GUITAR_PROTOCOL.md](GUITAR_PROTOCOL.md). Final test: [../GUITAR_RECOGNITION_REPORT.md](../GUITAR_RECOGNITION_REPORT.md).

## Material

160 MoisesDB train + validation songs (33 artist groups) and the guitar-free twin of each song that has guitar and something else (143 twins). Four artist folds; each model is scored on songs by artists it never saw. Scores are compared at one operating point, the threshold that gives 5% false alarms over all guitar-free time (guitar-free passages of real mixes and the twins), because the target is a false-alarm rate.

## Candidates

| # | Candidate | What it is |
|---|---|---|
| 0 | AudioSet "Guitar" output, untrained | the pretrained EfficientAT `mn10_as` class output as it comes |
| 1 | Frozen `mn10_as`, isolated 1 s / 2 s windows + linear head | the model sees only the window |
| 2 | Frozen `mn10_as`, 10 s context ending now + head | newest part of its feature map, whole-context average and AudioSet outputs as features |
| 3 | **Fine-tuned `mn10_as`, 10 s context** | the network itself trained on mixes made from stems; one output per 0.3 s column, the newest column is used |
| – | PaSST-S | measured cost only: 2.5 s of one core per 10 s clip, about 70 times `mn10`; not scored |
| – | BEATs | not tried |

Training mixes for candidate 3 (`experiments/recognition/finetune.py`): a song with its own guitar parts at changed levels; the same song without guitar; the song's backing with another song's guitar at a random level; and, in any of these, another song's keyboards on top. A column counts as guitar when the guitar part is within 20 dB of the mix.

## Results, all four folds (160 songs, 143 twins)

| Candidate | ROC-AUC (real + twins) | PR-AUC | Recall at 5% false alarms | F1 there |
|---|---:|---:|---:|---:|
| 0 AudioSet "Guitar", untrained | 0.696 | 0.564 | 0.110 | – |
| 1 frozen, isolated 1 s window | 0.853 | 0.810 | 0.456 | – |
| 1 frozen, isolated 2 s window | 0.881 | 0.843 | 0.519 | – |
| 2 frozen, 10 s context, linear head | 0.919 | 0.897 | 0.671 | – |
| 2 frozen, 10 s context, 64-unit head | 0.923 | 0.899 | 0.665 | 0.765 |

"Always on" F1 on the real mixes is 0.888 (guitar plays 80% of the scored time).

An observation that shaped the design: `mn10_as` gives unusable outputs on clips shorter than about 5 s (it was trained on 10 s clips). Feeding it the last 10 s and reading the newest part of its feature map is what moved recall from 0.46 to 0.67. It also means the model's answer always rests on several seconds of audio.

## Frozen against fine-tuned, folds 0 and 2 (111 songs, 107 twins)

Fine-tuning takes about 80 minutes per fold on the CPU, so two of the four folds were run.

| | Frozen + head | Fine-tuned |
|---|---:|---:|
| ROC-AUC (real + twins) | 0.919 | 0.919 |
| Recall at 5% false alarms | 0.626 | **0.672** |
| F1 there | 0.738 | **0.771** |
| Twins lit more than 20% of the time | 11 of 107 | 8 of 107 |
| Recall, "audible" reference (guitar within 15 dB of the mix) | 0.666 | 0.718 |
| Distorted guitar: recall at 5% false alarms | 0.568 | 0.636 |
| Clean electric guitar | 0.123 | 0.190 |
| Acoustic guitar | 0.513 | 0.565 |

Averaging the score over the last two estimates (1 s), as the application does, gives 0.689 recall at 5% false alarms; over four (2 s) 0.700. Longer averaging does not help further.

## Where it fails (fine-tuned, folds 0 and 2)

**Quiet guitar.** Recall at the 5% threshold by guitar level relative to the mix:

| Guitar level | Share of guitar time | Recall |
|---|---:|---:|
| more than 20 dB below | 3% | 0.08 to 0.12 |
| 15 to 20 dB below | 5% | 0.34 |
| 10 to 15 dB below | 22% | 0.57 |
| 5 to 10 dB below | 51% | 0.78 |
| within 5 dB or louder | 20% | 0.76 |

**Whole songs, not moments.** Per-song recall: median 0.78, a quarter of songs above 0.89, and 15 of 111 songs below 0.30. Songs are mostly either found or missed as a whole.

| Songs whose guitar is | Songs | Mean recall |
|---|---:|---:|
| clean electric only | 12 | 0.24 |
| distorted + clean | 3 | 0.48 |
| distorted + acoustic | 14 | 0.69 |
| clean + acoustic | 21 | 0.70 |
| distorted only | 11 | 0.72 |
| acoustic only | 29 | 0.76 |
| all three | 9 | 0.85 |

By genre: rap 0.08 (4 songs), reggae 0.16 (2), rock 0.64 (27), pop 0.73 (32), singer-songwriter 0.75 (26), country 0.85 (3). The rap and reggae songs have a clean, sparse rhythm guitar.

**False alarms come with keyboards.** With the frozen model at the 5% threshold, twins that contain piano are lit 9.4% of the time, twins without piano 1.2%; plucked instruments other than guitar 13%.

## Selection

- **Model: fine-tuned `mn10_as`** (candidate 3). Better than the frozen variant at the operating point, one network instead of a network plus a head, 36 ms per estimate on one core.
- **Estimate every 0.5 s from the last 10 s**, scores averaged over two estimates.
- **Thresholds**, from the held-out fold scores averaged that way: "detected" is the score exceeded in 5% of guitar-free time, "candidate" the score exceeded in 20%.

| Label | Detected | Candidate | Recall as detected | Recall as detected or candidate |
|---|---:|---:|---:|---:|
| guitar | 0.926 | 0.627 | 0.689 | 0.868 |
| distorted | 0.264 | 0.030 | 0.636 | 0.796 |
| clean | 0.307 | 0.099 | 0.196 | 0.497 |
| acoustic | 0.913 | 0.418 | 0.578 | 0.786 |

The final model is trained on all 160 songs with the same settings; its scores may sit slightly differently from the fold models the thresholds came from.

## Slakh2100 as extra training data (after the test, selection folds only)

Added on 2026-11-18, after the test was read. Only the selection folds were used; the test was **not** read again, because the result did not justify a new model.

- Material: every fourth Slakh2100 training track (312 synthesized songs), cut to their middle 90 s and turned into the same parts as MoisesDB songs (`experiments/recognition/cache_slakh.py`): guitar by General MIDI program (distorted 122 songs, clean 202, acoustic 167, other 6), keyboards (piano, organ, synth lead and pad classes; all 312) and everything else.
- Training: identical to the fine-tuned model above (same folds, steps, seed, rate) except that 35% of clips, guitar donors and keyboard donors come from Slakh (`finetune.py --extra-parts`).
- Scoring: the same held-out real MoisesDB songs and their twins; Slakh is never scored or pooled.

| Label (folds 0 + 2, 111 songs, 107 twins) | ROC-AUC without / with Slakh | Recall at 5% false alarms without / with | F1 without / with |
|---|---:|---:|---:|
| guitar | 0.919 / 0.923 | 0.672 / 0.675 | 0.771 / 0.773 |
| distorted | 0.889 / 0.904 | 0.636 / 0.653 | 0.633 / 0.645 |
| clean | 0.732 / 0.715 | 0.190 / 0.095 | 0.265 / 0.142 |
| acoustic | 0.887 / 0.905 | 0.565 / 0.609 | 0.661 / 0.694 |

Per fold (guitar, recall at 5% false alarms): fold 0 0.808 → 0.814, fold 2 0.630 → 0.608. The two folds move in opposite directions by about as much as the pooled change, so the guitar result is no change within noise. Distorted and acoustic gain a little; clean, already unusable, halves. With the main label unchanged, no model was trained on all folds and `guitar_mn10_v1` stays the selected model.

Reading: synthesized guitars rendered from MIDI do not add the variety that real recordings by more artists would. Slakh's clean and jazz guitar presets may also teach a sound that differs from real clean guitar, which fits the drop on that label; this is an inference, not tested.

## Against the targets, before the test

| Target | Selection-stage result |
|---|---|
| Recall ≥ 0.90 | 0.69 at the 5% false-alarm point |
| False-positive rate ≤ 0.05 | 0.05 by construction of the threshold |
| F1 ≥ 0.85 | 0.77 |
| Entrance delay ≤ 150 ms | not possible: estimates every 0.5 s, averaged over two |

## Not done at this stage

- Slakh2100 (synthetic) was not available in time; it was tried afterwards (section above) and did not help.
- PaSST and BEATs were not scored. The frozen and fine-tuned small model end at the same AUC, which points at the training data (115 to 120 songs by about 25 artists per fold) rather than the size of the network; that is an inference, not a measurement.
- Folds 1 and 3 were not run for the fine-tuned model.
- No hybrid with the separation model's guitar output was built.
