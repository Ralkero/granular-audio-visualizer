# Frozen attack v2 versus multiband v3

Split: development. This is the split used to choose the gate. Synthetic scores do not establish real-music instrument accuracy. The first number is `mixed_audio_attack_v2`; the second is `mixed_audio_attack_v3`.

| Fixture | F1 ±50 ms v2 → v3 | F1 ±25 ms v2 → v3 | FP v2 → v3 | FN v2 → v3 |
|---|---:|---:|---:|---:|
| isolated_bass | 0.8657 → 1.0000 | 0.8657 → 1.0000 | 9 → 0 | 0 → 0 |
| snare_noise | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0 → 0 | 0 → 0 |
| high_percussion | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0 → 0 | 0 → 0 |
| simultaneous | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0 → 0 | 0 → 0 |
| dense | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 0 → 0 | 0 → 0 |
| irregular_fills | 0.9508 → 1.0000 | 0.9508 → 1.0000 | 0 → 0 | 9 → 0 |
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
  },
  "multiband_v3": {
    "25": {
      "tp": 565,
      "fp": 0,
      "fn": 0,
      "duplicates": 0,
      "precision": 1.0,
      "recall": 1.0,
      "f1": 1.0,
      "timing_bias_ms": -0.9995384956688821,
      "median_absolute_error_ms": 3.2187500000020464,
      "p95_absolute_error_ms": 16.781250000058456
    },
    "50": {
      "tp": 565,
      "fp": 0,
      "fn": 0,
      "duplicates": 0,
      "precision": 1.0,
      "recall": 1.0,
      "f1": 1.0,
      "timing_bias_ms": -0.9995384956688821,
      "median_absolute_error_ms": 3.2187500000020464,
      "p95_absolute_error_ms": 16.781250000058456
    }
  }
}
```

Per-class negatives, tempo ambiguity and beat continuity are in the corresponding JSON files; aggregate onset F1 is not a substitute for those checks.
