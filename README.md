# Granular Music Visualizer — Phase One

A Windows 11 desktop control center for **live audio capture, CPU analysis, timestamped events and session logging**. No visualization renderer is included.

## Launch

On this development machine, double-click **Launch Control Center.cmd**. Select **Default system output**, leave debug recording off, and click **Start analysis**. Play audio normally. **Stop analysis** finalizes the local session package.

Sources:
- **Default system output:** follows the Windows multimedia output; attempts recovery when output devices change.
- **Specific output device:** chooses a WASAPI loopback device. Duplicate device names are rejected rather than guessed.
- **Application / process tree:** refresh sources, select the app, then start. Captures its process and children across endpoints. If unavailable, the error explains how to select system output. It never silently switches to capturing all apps.
- **Audio file:** uses the same DSP engine, paced for the control center. CLI file analysis can run faster than real time.

Logging is enabled by default. Audio recording is disabled. The optional debug setting explicitly enables **at most 60 seconds** of float WAV recording with a timeline sidecar. Exports omit debug audio. Advanced analysis is also off by default; enabling it runs a delayed spectral harmonic/percussive estimate, **not vocals/drums/bass stem separation**.

## Fresh development checkout

Install Python 3.14 and `uv`, then run `./Setup.ps1`. Dependency versions are pinned in `uv.lock`. For application capture, install the .NET 9 SDK and run `./Build Process Capture.ps1`; the resulting helper uses the installed .NET 9 runtime. System capture uses the same helper when it is built (recommended: it timestamps every packet with the Windows audio clock); without it, system capture falls back to a PortAudio backend whose timestamps drift under load. File analysis does not require .NET. Python project metadata supports 3.11+, but this checkout was validated with 3.14.3 only.

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check src tests scripts
.venv\Scripts\python.exe -m gmv.cli devices
.venv\Scripts\python.exe -m gmv.cli capture --seconds 30
.venv\Scripts\python.exe -m gmv.cli capture --pid 1234 --seconds 30
.venv\Scripts\python.exe -m gmv.cli file "C:\path\track.flac"
.venv\Scripts\python.exe -m gmv.cli file "C:\path\track.flac" --realtime
.venv\Scripts\python.exe -m gmv.cli replay "sessions\SESSION\messages.jsonl" --speed 2
.venv\Scripts\python.exe scripts\benchmark.py
```

Each run produces an independent session under `sessions/`: manifest, capture information, feature CSV, realtime event JSONL, replayable message JSONL, performance JSON, summary, advanced output and error log. Session files, recordings, helper binaries and machine-specific evidence are ignored by Git. The canonical repository is [Ralkero/granular-audio-visualizer](https://github.com/Ralkero/granular-audio-visualizer); completed development milestones are committed and pushed there as the project progresses.

## Analysis interface

```python
from gmv.engine import Engine
from gmv.sources.system_native import system_audio_source

engine = Engine(system_audio_source())
subscription = engine.bus.subscribe(capacity=256)
engine.start()
message = subscription.get(timeout=1.0)  # immutable, versioned Pydantic model
# Consume on your own worker; never block capture from a renderer.
engine.stop()
engine.done.wait(15)
engine.bus.unsubscribe(subscription)
```

Subscribers have independent bounded queues; slow consumers lose their oldest messages and can inspect `subscription.dropped`. There is no network listener and no need to poll a JSON file. See [schema contract](docs/SCHEMA.md), [architecture](docs/ARCHITECTURE.md), [dependencies](docs/DEPENDENCIES.md), and [milestone report](docs/MILESTONES.md).

## Current limitations

This is an initial Phase One implementation, not a claim that every acceptance criterion has been validated. User-assisted Spotify playback, pause/resume with audible browser audio, start while paused, seek/skip and quit/restart were observed using the Sound BlasterX G6. Physical device unplug/replug, default-device switching, alternate devices and Linux capture remain unverified. The PortAudio system fallback silently lost audio under load and drifted; the native system backend showed no drift in two paired ten-minute runs on one device (limits in the Phase 1.1 report, Milestone 7). Physical sound-to-screen latency is unmeasured. Musical accuracy is validated with synthetic fixtures, not an annotated music corpus. Percussion labels and tempo are preliminary heuristics with uncalibrated confidence scores.

No Spotify login, playback routing, system volume changes, protected-content bypass, GPU or NVIDIA dependency is used. The application does not control the player. Unsupported/protected audio may yield silence. Short-term loudness is RMS dBFS, not LUFS. See [measured performance and validation](docs/VALIDATION.md) for precise scope.


## Phase 1.1 results and evaluation

[Phase 1.1 report](docs/PHASE1_1_REPORT.md) records the validation milestones and their limits. On the frozen synthetic holdout, onset F1 is 0.996 at 50 ms for per-band attack detection, compared with 0.880 for the original detector and 0.983 for the previous global gate. That is not real-music instrument recognition. Experimental beat-phase tracking stays off by default. Confidence values are uncalibrated scores. The control center shows relative band energy, separate drop counters, delayed-analysis age, and the pending physical device checks.

```powershell
uv sync --locked --extra dev --extra evaluation
.venv\Scripts\python.exe -m gmv.evaluation.run --detector baseline --split development --output local-evidence/baseline.json
.venv\Scripts\python.exe -m gmv.evaluation.run --detector candidate --split development --beat-phase --output local-evidence/candidate.json
.venv\Scripts\python.exe scripts/compare_performance.py --output local-evidence/performance.json
.venv\Scripts\python.exe scripts/validate_capture.py --pid PID --pair-system --seconds 600 --output local-evidence/soak.json
.venv\Scripts\python.exe scripts/clock_probe.py --engine --advanced --pair-app APP.exe --seconds 600 --output local-evidence/clock.json
```

Real-music evaluation uses a versioned corpus whose audio stays on your machine: see the [corpus specification](docs/CORPUS.md), [annotation workflow](docs/ANNOTATION.md), [evaluation methodology](docs/EVALUATION.md) and the [Phase 1.2 report](docs/PHASE1_2_REPORT.md). The corpus is currently empty, so no real-music accuracy is claimed.

```powershell
gmv corpus register PATH --source-kind owned_local --license "..." --permission-basis "..." --group GROUP
gmv corpus assign-splits
gmv corpus import-labels RECORDING_ID LABEL_FOLDER --annotator YOU
gmv corpus review RECORDING_ID --annotator YOU
.venv\Scripts\python.exe -m gmv.evaluation.run --detector candidate --corpus corpus/manifest.json --corpus-split validation --output local-evidence/validation.json
```

See [capture test matrix and legacy single-file import format](docs/CAPTURE_VALIDATION.md), [schema 1.1 / replay / stale-event policy](docs/SCHEMA.md), and [isolated Demucs benchmark](experiments/separation/README.md). Demucs is an offline research experiment: overlap throughput, latency, synthetic quality and unresolved weight licensing did not justify live integration. The control center distinguishes observed/candidate/predicted/delayed events, reports separate losses and helper resources, and shows pending hardware checks. Raw recording stays off.
