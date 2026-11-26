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

## Status

Stage 0 is done: the protocol is registered and the environment set up. Nothing has been
run on the owner's music yet.
