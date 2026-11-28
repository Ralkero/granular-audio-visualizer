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

Added for Step 1b (generated data only): `presence_model/musdb.py`, the MUSDB18-HQ overlap
matcher on in-memory inputs, and a spy test of step 2's access profile. Running it on the real
corpus is step 2, which is not implemented.

Run every data-safe test:

```
~/.venvs/gmv-presence-experiments/bin/python -m pytest -q -p no:cacheprovider tests/test_presence_model_*.py
```

The spy test needs `strace` (layer 1). Each control is checked per recorder, so without strace
the 15 tests that require strace's own detections fail by design.

## Step 4: Stage 2's synthetic implementation

Added for Step 4 (generated data only; Amendment 3, A3.9):
- `presence_model/`: `b1train` (B1's trainer), `cbatches` (model C's inputs), `matched` (the
  matched-start check), `scoring`, `auc`, `gates`, `recorders`, `launcher`, `seal` (N2), and the
  disclosure check for results: `exact`, `disclosure`, `results`;
- `seedplan.RunLedger.claim_horizon` (the `horizon` relations) and policy D in `screening`;
- `experiments/presence/ctrain.py` (model C's training loop) and `synthetic_run.py` (B3's synthetic
  end-to-end run); `cnn.SeededDropout` can record its mask digests.

`tests/test_presence_model_ctrain.py` and `tests/test_presence_model_endtoend.py` need PyTorch and
run in this environment; the end-to-end test trains 102 small models three times (about a minute).

## Status

- **Stage 0 is done:** the protocol is registered and the environment set up.
- **Stage 1 is done for generated data only:** the harness and the protocol §16 tests.
- **Still outstanding:** the real fold files with their hashes, and the aggregate MUSDB18-HQ
  overlap check (its matcher exists since Step 1b; running it is step 2). Both need separate
  authorisation to use real project data.
- **Step 4 is built for generated data only;** review RG1b is next.
- **Not yet run:** nothing has been run on the owner's music.
