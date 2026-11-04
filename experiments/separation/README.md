# Isolated CPU separation experiment

This is an explicitly invoked, offline research benchmark, never an application-startup download or live worker. Inputs are original CC0 procedural fixtures; no Spotify waveform is stored or uploaded. Estimated stems stay in memory and are not saved.

Primary references checked 2026-11-03: [Demucs maintainer repository](https://github.com/adefossez/demucs), [model repository](https://huggingface.co/adefossez/HTDemucs). Demucs 4.1.0 supports Python 3.10+ and modern CPU PyTorch; upstream warns of limited active development. Code is MIT. **Upstream explicitly states that weight licensing is unclear** because of training-data terms. This local benchmark neither resolves that issue nor grants redistribution permission; weights are excluded from Git and must not be bundled in the product.

## Reproduce from the repository root

```powershell
uv venv --python 3.12 .separation-venv
uv pip install --python .separation-venv/Scripts/python.exe torch==2.9.1+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .separation-venv/Scripts/python.exe -r experiments/separation/requirements-lock.txt
.venv\Scripts\python.exe experiments/separation/setup_model.py
.venv\Scripts\python.exe experiments/separation/benchmark.py
```

Tested with Python 3.12.9. The lock records all installed experiment dependencies; the main Python 3.14 environment remains separate. Only `setup_model.py` downloads weights. Pinned revision: `cbc8a9b1a87023b7fd74e7b3412e6321c0eab003`; file `955717e8.safetensors`, 84,025,440 bytes; SHA256 `d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd`. Download streams have a size cap and timeout; every model load verifies the file. No model training or GPU dependency is involved.

The supervisor caps the owned subprocess tree at 300 seconds / 4 GiB sampled RSS. CPU execution uses four torch threads and no parallel inference workers. Process-tree accounting includes Windows virtual-environment launchers and their Python descendants. Kill operations target only this owned experiment tree. RSS is sampled at 100 ms, so short peaks may be missed; this is not an OS-enforced allocation limit. The first resource trial incorrectly sampled only the launcher; it was discarded and rerun after correcting the monitor.

Five six-second jobs cover dense attacks, a shifted overlapping dense chunk, quiet activity, quiet/loud transition and sustained harmony. Audio converts from 48 kHz to 44.1 kHz with `resample_poly(147,160)`; timestamps retain original sample positions. Full chunks are available before inference, including future context relative to early samples. HTDemucs pads to its 7.8-second model segment; padding does not make unavailable future audio observable. `shifts=0`, one model, four source categories; no claim of guitar/piano identity.

Metrics include load time (excluding Python imports), each inference, offline RTF and RTF per five seconds of new audio under one-second overlap, sampled resources, SI-SDR against known synthetic sources, cross-correlation lag, projection leakage, absent-source energy, and disagreement between two estimates of the same overlapping second. SI-SDR is scale-invariant; projection leakage is confounded by correlated sources, and low-energy lag estimates are weak evidence. These are not professional-music quality scores. The cold load measurement uses a new process but the OS disk cache was not flushed.

Results: [machine-readable benchmark](../../docs/evaluation/separation_cpu.json), [phase report](../../docs/PHASE1_1_REPORT.md). Mean inference exceeded new audio duration, oldest-sample delay exceeded 11 seconds, and synthetic bass separation failed. Continuous integration is not justified; use immediate mixed-audio evidence for impacts and optional HPSS only as delayed context. Model licensing is an additional unresolved product-distribution issue.
