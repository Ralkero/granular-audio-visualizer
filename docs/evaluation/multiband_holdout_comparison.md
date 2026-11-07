# Frozen attack v2 versus multiband v3

Split: holdout. One evaluation after the development result was frozen. Synthetic scores do not establish real-music instrument accuracy. The first number is `mixed_audio_attack_v2`; the second is `mixed_audio_attack_v3`.

| Fixture | F1 ±50 ms v2 → v3 | F1 ±25 ms v2 → v3 | FP v2 → v3 | FN v2 → v3 |
|---|---:|---:|---:|---:|
| isolated_bass | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0 → 0 | 0 → 0 |
| snare_noise | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0 → 0 | 0 → 0 |
| high_percussion | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0 → 0 | 0 → 0 |
| simultaneous | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0 → 0 | 0 → 0 |
| dense | 0.9970 → 0.9910 | 0.9970 → 0.9910 | 0 → 0 | 1 → 3 |
| irregular_fills | 0.9218 → 0.9923 | 0.9218 → 0.9923 | 0 → 0 | 19 → 2 |
| syncopation | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0 → 0 | 0 → 0 |
| tempo_change | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0 → 0 | 0 → 0 |
| quiet_loud | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0 → 0 | 0 → 0 |
| sustain_negative | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0 → 0 | 0 → 0 |
| silence_negative | 0.0000 → 0.0000 | 0.0000 → 0.0000 | 0 → 0 | 0 → 0 |
| opposite_phase | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0 → 0 | 0 → 0 |
| embedded_eight | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0 → 0 | 0 → 0 |

Aggregate counts and timing errors:
```json
{
  "attack_v2": {
    "25": {
      "tp": 562,
      "fp": 0,
      "fn": 20,
      "duplicates": 0,
      "precision": 1.0,
      "recall": 0.9656357388316151,
      "f1": 0.9825174825174825,
      "timing_bias_ms": -12.031586299026426,
      "median_absolute_error_ms": 14.94825000014366,
      "p95_absolute_error_ms": 19.4178500000023
    },
    "50": {
      "tp": 562,
      "fp": 0,
      "fn": 20,
      "duplicates": 0,
      "precision": 1.0,
      "recall": 0.9656357388316151,
      "f1": 0.9825174825174825,
      "timing_bias_ms": -12.031586299026426,
      "median_absolute_error_ms": 14.94825000014366,
      "p95_absolute_error_ms": 19.4178500000023
    }
  },
  "multiband_v3": {
    "25": {
      "tp": 577,
      "fp": 0,
      "fn": 5,
      "duplicates": 0,
      "precision": 1.0,
      "recall": 0.9914089347079038,
      "f1": 0.9956859361518551,
      "timing_bias_ms": -2.025801126610517,
      "median_absolute_error_ms": 3.5102500000414594,
      "p95_absolute_error_ms": 16.781250000002856
    },
    "50": {
      "tp": 577,
      "fp": 0,
      "fn": 5,
      "duplicates": 0,
      "precision": 1.0,
      "recall": 0.9914089347079038,
      "f1": 0.9956859361518551,
      "timing_bias_ms": -2.025801126610517,
      "median_absolute_error_ms": 3.5102500000414594,
      "p95_absolute_error_ms": 16.781250000002856
    }
  }
}
```

Per-class negatives, tempo ambiguity and beat continuity are in the corresponding JSON files; aggregate onset F1 is not a substitute for those checks.
