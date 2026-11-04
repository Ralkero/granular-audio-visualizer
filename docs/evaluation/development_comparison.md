# Frozen baseline versus candidate

Split: development. Synthetic scores do not establish real-music instrument accuracy.

| Fixture | F1 ±50 ms baseline → candidate | F1 ±25 ms baseline → candidate | FP baseline → candidate | FN baseline → candidate |
|---|---:|---:|---:|---:|
| isolated_bass | 0.8657 → 0.8657 | 0.8657 → 0.8657 | 9 → 9 | 0 → 0 |
| snare_noise | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0 → 0 | 0 → 0 |
| high_percussion | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0 → 0 | 0 → 0 |
| simultaneous | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0 → 0 | 0 → 0 |
| dense | 0.8000 → 1.0000 | 0.8000 → 1.0000 | 0 → 0 | 58 → 0 |
| irregular_fills | 0.7907 → 0.9508 | 0.7907 → 0.9508 | 8 → 0 | 28 → 9 |
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
  "baseline": {
    "25": {
      "tp": 479,
      "fp": 17,
      "fn": 86,
      "duplicates": 0,
      "precision": 0.9657258064516129,
      "recall": 0.8477876106194691,
      "f1": 0.9029217719132894,
      "timing_bias_ms": -10.907861691117857,
      "median_absolute_error_ms": 16.781249999999524,
      "p95_absolute_error_ms": 16.781250000208914
    },
    "50": {
      "tp": 479,
      "fp": 17,
      "fn": 86,
      "duplicates": 0,
      "precision": 0.9657258064516129,
      "recall": 0.8477876106194691,
      "f1": 0.9029217719132894,
      "timing_bias_ms": -10.907861691117857,
      "median_absolute_error_ms": 16.781249999999524,
      "p95_absolute_error_ms": 16.781250000208914
    }
  },
  "candidate": {
    "25": {
      "tp": 556,
      "fp": 9,
      "fn": 9,
      "duplicates": 0,
      "precision": 0.984070796460177,
      "recall": 0.984070796460177,
      "f1": 0.984070796460177,
      "timing_bias_ms": -12.686525179949795,
      "median_absolute_error_ms": 16.781249999998636,
      "p95_absolute_error_ms": 16.781250000208914
    },
    "50": {
      "tp": 556,
      "fp": 9,
      "fn": 9,
      "duplicates": 0,
      "precision": 0.984070796460177,
      "recall": 0.984070796460177,
      "f1": 0.984070796460177,
      "timing_bias_ms": -12.686525179949795,
      "median_absolute_error_ms": 16.781249999998636,
      "p95_absolute_error_ms": 16.781250000208914
    }
  }
}
```

Per-class negatives, tempo ambiguity and beat continuity are in the corresponding JSON files; aggregate onset F1 is not a substitute for those checks.
