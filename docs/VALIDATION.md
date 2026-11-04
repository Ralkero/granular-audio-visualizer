# Validation and measurements — 2026-11-03

Environment: Windows 11 Home build 26300, AMD Ryzen 9 5950X, Python 3.14.3. Live system capture used a Sound BlasterX G6 output at 48 kHz / eight channels. The application and tests used CPU processing only. Playback was already running in a browser; it was not restarted, rerouted, paused or modified for these tests. No raw live audio was saved.

## Live measurements

| Run | Duration | Feature frames | Audio drops | DSP p95 | Estimated capture-center → publication p95 | Sampled process RSS |
|---|---:|---:|---:|---:|---:|---:|
| System, initial logging test | 12 s | 593 | 0 | 1.87 ms | 29.56 ms | 57.96 MB |
| System, longer run | 120 s | 5,994 | 0 | 1.85 ms | 30.05 ms | 58.32 MB; peak 58.42 MB |
| Browser process tree | 10 s | 484 | 0 (including native queue) | 1.41 ms | 29.26 ms | 48.72 MB, Python only |
| System + optional delayed HPSS | 15 s | 744 | 0 | 1.88 ms | 30.77 ms | 118.53 MB |

Long-run statistics retain the last 4,096 feature frames, approximately 82 seconds; counters cover the whole session. Last sampled Python process CPU was 20.3% of one core for the 120-second run (not 20.3% of the whole 32-thread machine). Sampling is once per second. Native helper CPU/RSS is not included in the Python process metrics. These short runs establish an initial baseline, not long-duration reliability or a universal performance guarantee.

The optional HPSS run completed seven two-second chunks: processing median 59.25 ms, p95 60.50 ms, max 60.57 ms. Audio evidence is published after the chunk is available, so delay is approximately two seconds plus processing/queue time. This is a spectral harmonic/percussive balance estimate, **not neural stem separation**.

### Timing limits

Measured: Python queue delay, DSP duration, estimated capture-center age, bus publication timestamps and session resource samples. Not measured: acoustic/DAC presentation delay, calibrated output-device latency, future renderer or display latency. The tested system backend uses callback-anchored sample timestamps because PortAudio's ADC timestamp was unusable; process capture supplied WASAPI QPC timestamps. No sub-30 ms physical end-to-end synchronization claim is made. Window length is 46.44 ms and hop is 20 ms; half-window accumulation is included in center-age estimates.

### Process isolation

A simultaneous six-second comparison captured an inactive Steam process tree and the system mix. Steam-target frames had peak 0.0, while system frames reached 0.04635 full scale with current browser playback. A separate browser-target capture produced live nonzero features. This verifies exclusion of that unrelated browser playback in the tested configuration; it is not a comprehensive multiple-player isolation or Spotify test.

### Desktop verification

The actual Windows control-center window was inspected and operated. Start enabled capture, disabled source edits, displayed changing input/transient meters, source format, timestamped events and performance values. Stop returned to the ready controls and finalized logs. The initial GUI run showed zero audio/UI drops and roughly 116 MB process RSS. GUI performance includes Qt and its polling overhead and should not be compared directly to headless process RSS.

Final GUI smoke test also selected the browser process source and enabled delayed analysis: 2,206 feature frames, 22 advanced jobs, zero capture/UI/native drops and no errors. DSP p95 was 1.66 ms, estimated center-age p95 29.53 ms, and advanced job p95 42.99 ms for stereo input. Final process RSS was 208.46 MB (sampled peak 212.80 MB), including Qt and SciPy. Stop finalized the session, with debug recording confirmed off.

## Synthetic validation

Final automated suite: **30 tests passing**; Ruff checks pass; .NET Release build succeeds with zero warnings/errors. A Qt runtime smoke test covers native DLL/plugin availability in addition to core tests. Removing the initially installed full PySide bundle during dependency minimization removed shared Qt files in this development environment; reinstalling the locked Essentials/shiboken packages repaired it, and the actual window was relaunched successfully. Fresh setup installs only the Essentials package.

Automated tests cover source/bus contracts; bounded queues; slow-subscriber isolation; lossless file backpressure; exact file sample offsets and partial last blocks; 22.05/44.1/48/96 kHz tones; opposite-phase stereo; known RMS and band power; zero-crossing rate; silence/resumption; known onset times; synthetic 120 BPM tempo; intensity ordering; pipeline/session/replay consistency; malformed native packets; simulated device changes; format changes and gaps; startup failure; duration-capped opt-in debug recording; bounded slow advanced processing; and live overload accounting.

Pulse timing test: at least 21 of 23 known 0.5-second-spaced attacks must have a detected onset within 50 ms. Tone tests require centroid within 3 Hz of 1 kHz, known RMS within 1.5%, band mean-square power within 1%, and >99% of tracked energy in the expected band. These are fixture-specific assertions, not music-dataset precision/recall scores.

Reproducible CPU benchmark: `python scripts/benchmark.py --seconds 60`. A 60-second 48 kHz stereo fixture (tone, silence, 120 BPM bass pulses, opposite-phase channels) completed in **3.92 seconds**, about **15.3× real time**, with 2,998 feature frames, no drops, and DSP p95 **1.52 ms**. Fast offline queueing deliberately waits for space; its queue delay is not live capture latency. Logging was disabled for this isolated benchmark. Output lives in ignored `local-evidence/`.

## Hardware and musical validation still needed

- Spotify playback, pause/resume, process-tree identity, closure/restart, and multiple Spotify sessions.
- Actual USB unplug/replug and default-device switch while playing. Recovery control flow was fault-injection tested; no system playback settings were changed for validation.
- Other DACs, Bluetooth, HDMI, headphones/speakers, unusual driver channel layouts, exclusive-mode players and protected content.
- Longer runs, sustained background load, disk-full logging failures and helper-process resource profiling.
- An annotated, rights-cleared music corpus to measure onset/tempo/candidate precision and recall. Current browser audio demonstrates capture, not musical classification accuracy.
- Neural vocals/drums/bass/other separation: no Demucs weights were downloaded or benchmarked. The CPU HPSS timings must not be presented as Demucs performance.
- Linux/PipeWire capture implementation and validation.

Private session evidence is retained locally in `sessions/`; the public-safe source and documentation do not include audio, screenshots, app identifiers, process IDs or library metadata.
