# Vocal presence model experiments (Sonic Character, Phase A5.2)

This is offline research tooling, separate from the application. The protocol, fixed
before anything was run, is [docs/VOCAL_PRESENCE_MODEL_PROTOCOL.md](../../docs/VOCAL_PRESENCE_MODEL_PROTOCOL.md).

## Environment (Linux, CPU only)

```
bash experiments/presence/setup_environment.sh
```

This creates `~/.venvs/gmv-presence-experiments`, outside the repository, containing:

- Python 3.12;
- PyTorch `2.9.1+cpu` from PyTorch's CPU index (no CUDA, no ROCm);
- the other packages at the versions in `requirements-lock.txt`;
- this checkout's `gmv` package (editable, without its application dependencies).

The application never imports this environment, and the application's own `.venv` is
untouched.

Checked on 2026-11-26 (CachyOS, Ryzen 9 5950X): Python 3.12.16, torch 2.9.1+cpu, CUDA not
available, not a ROCm build, 16 CPU threads by default, and the protected presence loaders
import cleanly.

## Data

All inputs and outputs stay in the ignored `local-evidence/`. That covers features, fold
files, weights, predictions and per-group results. Tracked files hold aggregates only. No
external audio or dataset is downloaded.

## Stage 1: the synthetic harness and its tests

Code:
- `src/gmv/character/research/presence_model/`: NumPy and SciPy only. It holds the front
  end, labels and collars, training weights, inner folds, post-processing, metrics,
  screening, the descriptive bootstrap and the seeds.
- `experiments/presence/cnn.py`: model C, forward pass and streaming only, no training loop
  (needs PyTorch).

Every test uses generated audio or generated projects. Run them in the experiments
environment:

```
~/.venvs/gmv-presence-experiments/bin/python -m pytest -v -p no:cacheprovider \
    tests/test_presence_model_alignment.py tests/test_presence_model_objective.py \
    tests/test_presence_model_folds.py tests/test_presence_model_metrics.py \
    tests/test_presence_model_cnn.py
```

In the application's own environment, the same command (with `.venv/bin/python`) skips
`test_presence_model_cnn.py`, because PyTorch is not installed there.

## Step 1: the synthetic implementation

Added for Step 1 (generated data only):
- `presence_model/`: `seedplan`, `b0check`, `grid`, `enginepass`, `integrity`, `foldgen`,
  `gateway`, `timing`, `normalise`, `augment`;
- `experiments/presence/rng.py`;
- `cnn.SeededDropout`.

Run every data-safe test:

```
~/.venvs/gmv-presence-experiments/bin/python -m pytest -q -p no:cacheprovider tests/test_presence_model_*.py
```

The spy test needs `strace` (layer 1). Without it, two tests fail by design.

## Status

- **Stage 0 is done:** the protocol is registered and the environment set up.
- **Stage 1 is done for generated data only:** the harness and the protocol §16 tests.
- **Still outstanding:** the real fold files with their hashes, and the aggregate MUSDB18-HQ
  overlap check. Both need separate authorisation to use real project data.
- **Not yet run:** nothing has been run on the owner's music.
