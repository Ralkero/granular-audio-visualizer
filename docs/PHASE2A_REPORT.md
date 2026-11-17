# Phase 2A report: first live visualizer

The owner asked for a working visualizer now, built on the analysis that exists, with the engine improving underneath it. This is that first version. It is a prototype: it runs, it responds to music, and what it shows is only as good as the analysis behind it.

## Starting state (before any change)

| Check | Result |
|---|---|
| `pytest -q` | 106 passed |
| `ruff check .` | clean |
| `dotnet build native/ProcessLoopback -c Release` | 0 warnings, 0 errors |

## What was built

| Milestone | Delivered |
|---|---|
| 2A.1 Renderer foundation | `gmv.visualizer`: window with its own bounded EventBus subscription, GUI-thread refresh timer, bounded histories, clean attach, detach and close |
| 2A.2 Main visualization | 48-band spectrum as bars or radial, level meter with peak, glow that follows intensity, onset flashes; rises are immediate, only the fall is smoothed |
| 2A.3 Timeline | 15 seconds, placed by audio timestamp; onsets, beats, transients by frequency region, energy and spectral changes, delayed context; loudness as a backdrop; capture gaps shown |
| 2A.4 Lanes and rhythm | Drums, bass and vocals lanes with continuous score and active state; guitar and other harmonic shown as having no detector; beat pulse, tempo and uncalibrated score when the experimental tracker is on |
| 2A.5 Integration | "Open visualizer", activity weights chooser and "Replay session in visualizer" in the control center; `gmv visualizer` command; `Launch Visualizer.cmd`; documentation |

Engine change: feature frames now carry a 48-band display spectrum taken from the analyzer's existing FFT, because seven bands were too coarse to draw. Detection is unchanged; the synthetic regression gives the same counts as before (development 565 found, 0 false, 0 missed; holdout 577, 0, 5).

Nothing else in capture or DSP was touched. No second capture path, FFT or analyzer exists.

## How the requirements are met

- **Separate consumer.** The window subscribes with its own queue of 2048 messages. The control center, the session writer and the window each have their own queue. A test publishes 6,000 frames at a window that is not refreshing: the publisher finishes without waiting, the queue drops the oldest messages and the window continues from the newest.
- **Timestamps.** Marks keep the audio timestamp they were published with. A test sends a prediction for the future, an onset just heard and a late report about older audio, and checks each lands at its own time and that the prediction is not acted on before its time.
- **Observed, candidate, predicted, delayed.** Four drawing styles chosen from `evidence_kind` and `analysis_tier`, with a legend. Predicted beats are amber and dashed, sit to the right of "now" until reached, and never use the solid observed style. The band-ratio candidates are labelled low, mid and high-frequency transients, never kick, bass or cymbal.
- **No invented activity.** A lane has data only when a detector publishes that label. Without a weights file the three supported lanes read "no detector output" and point to the control center; guitar and other harmonic read "no detector for this instrument yet".
- **Future detectors.** Publishing a new label in `instrument_scores` creates its lane; a test does this with "kick" (nested under drums) and an unknown label.
- **Beat tracker.** Still opt-in in the control center and still labelled experimental. When it is off the window says so and works normally.
- **States.** Idle, waiting (no audio arriving), live, replay and stopped are shown; the refresh timer drops from 60 to 4 per second when nothing is moving.
- **Replay.** A session's logged messages replay through the window with their original timestamps and a REPLAY label. A test records a session from a file and checks the replayed onsets sit at the same audio times. No audio is saved for this.

## Measured results

All on the development machine (Ryzen 9 5950X, Radeon RX 9070 XT, Windows 11, Python 3.14, PySide6 6.12), beat tracker and activity model on.

| Run | Window | Frames per second | Drawing time per frame (median / p95) | Window drops | Capture drops |
|---|---|---:|---:|---:|---:|
| Music file, real on-screen window, 12 s | 1440x900 | 61.3 (slowest second 62.0) | 6.0 / 8.4 ms | 0 | 0 |
| Music file, real on-screen window, 12 s | 2560x1440 | 61.2 (slowest second 61.7) | 10.6 / 13.3 ms | 0 | 0 |
| Music file, offscreen, 30 s | 1440x900 | 61.9 | 4.8 / 5.8 ms | 0 | 0 |
| Live default system output, offscreen, 12 s | 1440x900 | not recorded | 4.1 ms median | 0 | 0 |

- **Live capture** used the native system backend (48 kHz, 8 channels). Audio happened to be playing on the PC: 551 frames, 161 timeline marks, all three lanes received scores, and both the waiting and live states were seen. Estimated capture-to-publish age p95 was 44 ms.
- **Memory** of the whole process: 238 MiB at 1440x900 and 276 MiB at 2560x1440 after 12 s. Histories are fixed-size, so this does not grow with running time; a long soak was not run.
- **CPU** of the whole process: 64% of one core at 1440x900, 127% at 2560x1440.
- **Automated tests**: 125 pass (19 new), lint clean. The native helper was not changed and still builds.

## Known limitations

1. **Drawing slows the analysis thread.** Painting is done in software on the GUI thread and shares Python's interpreter lock with the DSP thread. DSP time per frame (budget 20 ms) rose from 1.6 ms median / 1.9 ms p95 without the window to 2.4 / 10.0 ms at 1440x900 and 5.5 / 14.1 ms at 2560x1440 (worst 18.4 ms). No audio was dropped, but at full screen the margin is thin, and analysis results reach consumers a few milliseconds later. A quick trial of GPU drawing halved the drawing time, but its picture could not be verified and it used 630 MiB, so it was not kept. The real fix (GPU drawing done properly, or the renderer in its own process) is follow-up work.
2. **Not watched by a person yet.** Frame rates and pictures were checked by measurement and by saved frames, not by someone watching with sound. Whether the motion feels musical, and how the display lines up with what is heard, is unjudged. Sound-to-screen latency is not measured; the display follows the analysis, which trails the audio by roughly 45 ms plus output latency.
3. **Only default system output and file sources were run** through the visualizer. Specific-device and per-application capture publish through the same bus and should behave the same, but were not exercised here. Capture gaps, restarts and discontinuities are covered by synthetic tests only.
4. **Session replay was tested on a session made from a synthetic file**, not on a long real capture. Sessions recorded before this change have no display spectrum and show seven broad bars.
5. **The picture is only as good as the analysis.** Activity scores are uncalibrated and wrong some of the time (held-out false-positive rates 0.13 to 0.22). The beat tracker scored 0.69 on its development tracks and can lock to the off-beat. Changes and candidates are frequent and make their timeline rows busy in dense music.
6. **No detector exists** for guitar, other harmonic instruments or individual drum parts.
7. **The radial view** uses the same data as the bars; it is a second presentation, not a second analysis.
8. **Display scaling above 100%** and a second monitor were not tested.

## Backlog kept open (not part of this increment)

- Engine: real-music onset accuracy with a precision reference; held-out beat evaluation and the off-beat fix; fewer vocal and bass false positives; guitar and individual drum-part detection (needs MoisesDB); capture reliability on physical device changes; separation as a delayed layer or as a source of training labels; AMD-compatible model research.
- Evaluation scoring corrections identified earlier: one-to-one matching for instrument attacks, and correct handling of uncertain reference spans. Neither is done.
- Visualizer: GPU or separate-process drawing; a person-in-the-loop viewing pass; sound-to-screen latency; kick, snare and cymbal lanes when detectors exist; using delayed separation output as a context lane.
