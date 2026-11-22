# Granular Music Visualizer

A Windows 11 desktop application that **captures live PC audio, analyses it on the CPU and visualizes the result**: a control center for sources, diagnostics and session logs, and a visualizer window that draws what the analysis publishes. The visualizer is a first prototype (Phase 2A); the analysis underneath it is still being improved.

## Launch

Double-click **Launch Visualizer.cmd**. One window opens: the visualizer. Press **Start** to visualize what is playing on this PC, **Stop** to end the session.

**Settings** (top right) slides a drawer over the right edge without stopping anything:

- **Audio source**: default system output, a specific output device, one application and its child processes, or an audio file.
- **Analysis**: weights for drums / bass / vocals and for guitar, the experimental beat tracker, and note candidates. A summary line says what is available with the current choices.
- **Visualizer**: bars or radial layout, which lanes and timeline rows to show, smoothing, intensity, timeline length, reset, full screen (F11).
- **Sessions**: analysis logging (never audio), open or export the last session, replay a recorded one.
- **Diagnostics**: capture state, audio format, DSP time, latency estimate, drops, CPU and memory, drawing time.
- **Advanced**: window and hop, band edges, delayed harmonic/percussive analysis, debug recording.

Display settings apply immediately. Source and analysis settings changed while running show **Apply changes**, which restarts the analysis; the application itself never needs restarting. Choices are remembered between runs.

What the picture shows:

- **Spectrum** with level, onset flashes and, with the beat tracker on, a pulse on each predicted beat.
- **Timeline**, placed by audio timestamp. Solid marks are observed, hollow marks are heuristic candidates, amber dashed marks are predictions, dotted boxes are delayed analysis.
- **Instrument lanes** for drums, bass and vocals when their weights file is chosen. Weights are local files and are not in this repository.
- **Guitar lane**: shows a detector's output when guitar weights are chosen in Settings. A first detector was trained on MoisesDB and is not reliable enough to load by default ([results](docs/evaluation/moisesdb_guitar_v1.md)); without weights the lane says so. A separation model's guitar output was also measured: it finds more of the guitar and still fires too often without one ([results](docs/evaluation/moisesdb_guitar_separation.md)); it is not used by the application. A pretrained audio model fine-tuned for guitar (`guitar_mn10_v1`) can be chosen in the same field as a `.json` file: it reports about a second late as detected, candidate or absent. On held-out songs it finds 72% of guitar and reports guitar 4% of the time on guitar-free recordings, which is short of the targets, so it is off by default ([report](docs/GUITAR_RECOGNITION_REPORT.md)).
- **Notes lane**: pitch candidates between E2 and E6, as a piano roll. They show which pitches are present, not which instrument plays them; they turn amber only while a guitar detector reports guitar. Reliable for single-note playing, weak on chords (see the [Goal B report](docs/GOAL_B_REPORT.md)).

**Packaged build.** `.\"Build App.ps1" -GuitarModel guitar_mn10_v1` makes `dist\Granular Music Visualizer\Granular Music Visualizer.exe`, which runs without Python or an installed .NET: the build carries its own copy of the .NET 9 runtime for the capture helper, downloaded from Microsoft and checksum-verified by the build script. Models are copied from `.model-cache`; the named guitar model is switched on the first time that build runs and can be switched off in Settings. It remains experimental: on the owner's reviewed recordings it finds roughly half to two thirds of guitar at a 4 to 10% false-alarm rate ([results](docs/evaluation/review_finetune_v1.md)). Sessions are written to `%LOCALAPPDATA%\GranularMusicVisualizer\sessions`. The [0.1.1 release](https://github.com/Ralkero/granular-audio-visualizer/releases/tag/v0.1.1) carries `activity_v2` and `guitar_mn10_v1` for non-commercial use only (CC BY-NC-SA 4.0, attribution in `models/MODEL-LICENCES.txt`); weights trained on the owner's own recordings (`guitar_mn10_rv1`, `guitar_mn10_rv2`) stay local. `guitar_mn10_rv2` is an experimental alternative for listening comparisons ([how it was made](docs/evidence/TRAINING_RESULTS_V2.md)): include it with `-AlsoInclude guitar_mn10_rv2` and pick `guitar_mn10_rv2.json` with **Choose…** beside *Guitar* in the settings drawer; the build still starts with the model named by `-GuitarModel`.

A recorded session replays with `gmv visualizer --replay sessions\SESSION` or from Settings; replay uses logged messages and their original timestamps, not audio.

Sources:
- **Default system output:** follows the Windows multimedia output; attempts recovery when output devices change.
- **Specific output device:** chooses a WASAPI loopback device. Duplicate device names are rejected rather than guessed.
- **Application / process tree:** refresh sources, select the app, then start. Captures its process and children across endpoints. If unavailable, the error explains how to select system output. It never silently switches to capturing all apps.
- **Audio file:** uses the same DSP engine, paced at playback speed in the application. CLI file analysis can run faster than real time.

Logging is enabled by default. Audio recording is disabled. The optional debug setting explicitly enables **at most 60 seconds** of float WAV recording with a timeline sidecar. Exports omit debug audio. Advanced analysis is also off by default; enabling it runs a delayed spectral harmonic/percussive estimate, **not vocals/drums/bass stem separation**.

## Instrument layers (experimental)

The main view is the selected visualizer, Radial or Spectrum Bars, with the instruments drawn inside it: a ring each round the radial spectrum, or a strip each across the bars. Each layer is drawn from its own analysis of one instrument, with its own picture and its own colour, not from the spectrum:

| Layer | Picture | State |
|---|---|---|
| Vocals | strands that open with a phrase, follow its pitch and close between phrases | experimental |
| Guitar | six strings that are struck, ring and die away; strokes, beamed when they repeat | accepted by the owner |
| Bass | one broad ribbon whose height is the note and whose thickness is its weight | experimental |
| Drums | marks that hit and are gone: a disc (low), a burst of rays (mid), a spark (high) | experimental |

Each instrument's lane shows the same activity in more detail.

- Every layer joins a fast analysis of articulation (every 20 ms) with a slower judgement of whether the instrument is present: the recognition model for guitar, the frame-wise activity model (`activity_v2`) for drums, bass and vocals. While that judgement says "absent" the layer is empty.
- **Top bar**: a button per instrument, in its colour, shows or hides its layer while analysis runs; Timeline, Details and Notes show or hide the sections under the visualizer (drag a section's upper edge to resize it).
- **Settings → Instrument layers**: sensitivity (what the analysis reports), intensity (how strongly it is drawn; the analysis is not affected) and style per layer, the whole-mix spectrum (dimmed, full or hidden), and Reset layout.
- Performance measurements and what was changed: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).
- None of this is source separation. On MoisesDB stems, removing an instrument from a mix leaves 12% (drums), 5% (bass) and 0% (vocals) of what its layer shows, but that is mostly the activity model's doing: on their own the analyzers follow short broadband hits, whatever is low and periodic, and the centre of the mix. Design, measurements and limits: [docs/INSTRUMENT_LAYERS.md](docs/INSTRUMENT_LAYERS.md).

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
.venv\Scripts\python.exe -m gmv.cli visualizer
.venv\Scripts\python.exe -m gmv.cli visualizer --replay "sessions\SESSION"
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

Real-music evaluation uses a versioned corpus whose audio stays on your machine: see the [corpus specification](docs/CORPUS.md), [annotation workflow](docs/ANNOTATION.md), [evaluation methodology](docs/EVALUATION.md) and the [Phase 1.2 report](docs/PHASE1_2_REPORT.md). Reference data comes from research datasets and their stems ([dataset comparison](docs/DATASETS.md)); no manual labelling is required. No real-music accuracy is claimed until a dataset has been imported and evaluated.

```powershell
gmv corpus datasets
gmv corpus import-dataset mdbdrums "E:\datasets\MDBDrums"
.venv\Scripts\python.exe -m gmv.evaluation.run --detector candidate --corpus corpus/manifest.json --corpus-split validation --output local-evidence/validation.json
```

Your own music can be reviewed excerpt by excerpt without editing audio: `gmv review` picks recordings from a local inventory, cuts short excerpts, runs the guitar model over the full tracks for comparison and opens a review window that stores your labels locally ([method](docs/evaluation/REAL_WORLD_REVIEW.md)). Excerpts, labels and reports stay in the ignored `review-local/` folder. `gmv evidence` gathers what can be known about a collection without listening (MusicBrainz identity and credits, local fingerprints, two audio models, dataset references), combines it into label candidates by evidence class and builds a short, targeted review queue ([method and results](docs/evidence/EVIDENCE_ENRICHMENT.md)); its output stays in the ignored `research-local/` folder.

```powershell
gmv review --workdir review-local\pilot-v1 build --inventory Complete_Music_Inventory.csv --selection review-local\pilot-v1\selection.json
gmv review --workdir review-local\pilot-v1 open
gmv review --workdir review-local\pilot-v1 report
```

See [capture test matrix and legacy single-file import format](docs/CAPTURE_VALIDATION.md), [schema 1.1 / replay / stale-event policy](docs/SCHEMA.md), and [isolated Demucs benchmark](experiments/separation/README.md). Demucs is an offline research experiment: overlap throughput, latency, synthetic quality and unresolved weight licensing did not justify live integration. The control center distinguishes observed/candidate/predicted/delayed events, reports separate losses and helper resources, and shows pending hardware checks. Raw recording stays off.
