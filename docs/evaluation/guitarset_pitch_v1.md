# Pitch tracker on GuitarSet

- Tracker: `harmonic_salience_v1` (`gmv.pitch`), causal, 20 ms hop, 85 ms window, at most four notes.
- Data: GuitarSet microphone recordings (real solo acoustic guitar), reference notes from the dataset's own annotations. Notes are rounded to the nearest semitone; onsets are matched one to one with the same pitch.
- Split by player: 00 to 03 development (constants chosen here), 04 and 05 held out.
- The held-out players were scored twice: once before and once after two changes (overtone rejection and re-attack detection) that were prompted by synthetic unit tests, not by held-out results. The earlier held-out figures were frame F1 0.725 (solo) and 0.555 (chords).
- Reproduce: `python scripts/evaluate_pitch.py E:/datasets/GuitarSet --split heldout`

| Split | Playing | Minutes | Notes at once (mean) | Frame precision | Frame recall | Frame F1 | Pitch-class F1 | Note onset F1, ±50 ms | Note onset F1, ±100 ms | Frames with notes while the reference is silent |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Development | single notes | 60.8 | 0.96 | 0.798 | 0.776 | 0.787 | 0.811 | 0.701 | 0.758 | 8.7% |
| Development | chords | 60.8 | 2.64 | 0.811 | 0.412 | 0.547 | 0.690 | 0.373 | 0.480 | 5.1% |
| **Held out** | single notes | 30.4 | 0.71 | 0.759 | 0.836 | **0.795** | 0.805 | **0.771** | 0.815 | 13.0% |
| **Held out** | chords | 30.4 | 2.45 | 0.782 | 0.422 | **0.548** | 0.659 | 0.419 | 0.515 | 6.9% |

Held-out note onsets at ±50 ms: precision 0.713, recall 0.840 on single notes; precision 0.464, recall 0.383 on chords.

Reading the numbers:

- On single-note playing about three reported notes in four are right and most played notes are found.
- On chords, what is reported is mostly right (precision 0.78) but fewer than half of the sounding notes are reported.
- "Notes while the reference is silent" includes strings still ringing after the annotated note end; how much of it is ringing and how much is invention was not separated.
- This is acoustic guitar played alone. Nothing here measures distorted guitar, bends, or a guitar inside a full mix.
