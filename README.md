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

Install Python 3.14 and `uv`, then run `./Setup.ps1`. Dependency versions are pinned in `uv.lock`. For application capture, install the .NET 9 SDK and run `./Build Process Capture.ps1`; the resulting helper uses the installed .NET 9 runtime. System capture and file analysis do not require .NET. Python project metadata supports 3.11+, but this checkout was validated with 3.14.3 only.

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
from gmv.sources.windows import SystemAudioSource

engine = Engine(SystemAudioSource())
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

This is an initial Phase One implementation, not a claim that every acceptance criterion has been validated. Live system and application capture were tested with current browser playback and the Sound BlasterX G6. Spotify playback/restart, physical device unplug/replug, alternate devices and Linux capture remain unverified. Musical accuracy is validated with synthetic fixtures, not an annotated music corpus. Percussion labels and tempo are preliminary heuristics with uncalibrated confidence scores.

No Spotify login, playback routing, system volume changes, protected-content bypass, GPU or NVIDIA dependency is used. The application does not control the player. Unsupported/protected audio may yield silence. Short-term loudness is RMS dBFS, not LUFS. See [measured performance and validation](docs/VALIDATION.md) for precise scope.
