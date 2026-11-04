# Frozen baseline versus candidate

Split: holdout. Synthetic scores do not establish real-music instrument accuracy.

| Fixture | F1 ±50 ms baseline → candidate | F1 ±25 ms baseline → candidate | FP baseline → candidate | FN baseline → candidate |
|---|---:|---:|---:|---:|
| isolated_bass | 0.8750 → 1.0000 | 0.8750 → 1.0000 | 6 → 0 | 0 → 0 |
| snare_noise | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0 → 0 | 0 → 0 |
| high_percussion | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0 → 0 | 0 → 0 |
| simultaneous | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0 → 0 | 0 → 0 |
| dense | 0.7692 → 0.9970 | 0.7692 → 0.9970 | 0 → 0 | 63 → 1 |
| irregular_fills | 0.7354 → 0.9218 | 0.7175 → 0.9218 | 10 → 0 | 49 → 19 |
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
      "tp": 468,
      "fp": 18,
      "fn": 114,
      "duplicates": 0,
      "precision": 0.9629629629629629,
      "recall": 0.8041237113402062,
      "f1": 0.8764044943820225,
      "timing_bias_ms": -10.980641025736885,
      "median_absolute_error_ms": 16.78124999999797,
      "p95_absolute_error_ms": 20.296000000069693
    },
    "50": {
      "tp": 470,
      "fp": 16,
      "fn": 112,
      "duplicates": 0,
      "precision": 0.9670781893004116,
      "recall": 0.8075601374570447,
      "f1": 0.8801498127340824,
      "timing_bias_ms": -10.664898936265951,
      "median_absolute_error_ms": 16.781249999998415,
      "p95_absolute_error_ms": 21.30434999999874
    }
  },
  "candidate": {
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
  }
}
```

Per-class negatives, tempo ambiguity and beat continuity are in the corresponding JSON files; aggregate onset F1 is not a substitute for those checks.
