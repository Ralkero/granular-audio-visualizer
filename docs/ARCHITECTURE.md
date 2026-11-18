# Architecture

Live Windows output is the primary source. No Spotify credentials, routing, volume or playback modifications are required. This repository contains capture/analysis and diagnostic meters only.

`AudioSource -> bounded audio queue -> window assembler -> Tier A DSP -> EventBus`

The bus feeds independent bounded queues for the desktop monitor, session writer, and visualizer. Logging never writes from the capture callback. Tier B uses a separate worker and bounded audio history. File sources reuse the assembler and DSP; fast offline producers wait for queue space so their analysis is lossless. Live capture drops the oldest queued block to favor freshness. Queue drops and sample discontinuities are counted; context is reset rather than splicing missing audio into a false transient.

Message schema 1.1 uses immutable Pydantic models. Times are seconds from the session audio origin; feature timestamps refer to the window center, with window_end supplied separately. Events report evidence time, not task-completion time. Detection/publication monotonic times are separate and only comparable within the same run. Reconnected sources have new segment IDs. Recovery gaps remain gaps, not fabricated silent audio. Schemas support event groups and relationships without asserting instrument identity.

Initial 46.44 ms windows and 20 ms hops trade frequency resolution for responsiveness. Native device rates are analyzed without compulsory resampling. Band powers aggregate channel spectra (avoiding cancellation in opposite-phase stereo). A short window cannot resolve closely spaced low bass notes. Onsets and percussion labels are heuristic candidates; tempo requires historical evidence. Short-term loudness is unweighted RMS dBFS, not standardized LUFS. Adaptive display intensity blends absolute energy with a slowly moving relative reference; it does not change audio gain.

Privacy: raw audio is never written by default. Session metrics can reveal listening patterns and device/app names; sessions and local evidence are excluded from Git. There is no network listener. The visualizer subscribes directly through EventBus. Replay uses persisted messages, not rewritten live JSON files.

Linux: preserve the source contract for a PipeWire monitor adapter. It must connect to a sink monitor without changing the default device. No Linux implementation or hardware validation is claimed in this Windows milestone.

Phase 1.1 adds strict legacy migration, consumer freshness policy, generation invalidation for delayed jobs and an isolated offline CPU model experiment. Immediate onsets are per-band: a different frequency region can register inside the short grouping window, and a weak ripple after a louder attack does not. Reported band edges are measurement spans, not instrument names. See SCHEMA.md and PHASE1_1_REPORT.md. No model inference runs in the immediate pipeline.

System output capture has two backends behind the same `AudioSource` contract. The default, when the .NET helper is built, is native WASAPI endpoint loopback in the helper process: packets arrive over a pipe with a sequence number and QPC position, so capture is not exposed to Python scheduling, and loss or clock disagreement becomes an explicit discontinuity instead of silent drift. Timestamps are a contiguous sample clock per run, anchored to QPC minus a lead measured at device open so that evidence is not stamped ahead of delivery. The PortAudio callback backend remains as the fallback when the helper is absent; its sample clock cannot see audio lost before the callback and must not be treated as precise over long sessions. `scripts/clock_probe.py` measures either backend.

## Visualizer (Phase 2A)

`EventBus -> the window's own bounded queue (2048) -> VisualState -> QPainter`

The visualizer (`gmv.visualizer`) is one more subscriber. It has no audio path, no FFT and no analysis of its own; everything it draws comes from published feature frames, events and status messages. The control center and the session writer keep their own queues, so the windows can run together and none can starve another.

- **Threads.** Capture and DSP are untouched. A GUI-thread timer drains the queue and repaints, 60 times a second while data is arriving and 4 times a second when nothing is moving. The publisher never waits for the window: a full queue drops its oldest message, so a stalled window resumes from the newest data.
- **Time.** Marks and lane samples are placed by their audio timestamps. Arrival times are used only to learn how the audio timeline maps onto the local clock (the smallest recent difference between publication time and frame timestamp), which lets the display advance smoothly between 20 ms frames. A capture gap, a stalled stream or a new segment makes the window learn that mapping again and shows the hole; a restarted timeline or a new session clears the history. Beat predictions lie in the future and are drawn to the right of the "now" line until they are reached.
- **Honesty rules in code.** Drawing style follows `evidence_kind` and `analysis_tier`, not event names: observed is solid, candidate is hollow, predicted is amber and dashed, delayed analysis is a dotted box. The band-ratio candidates are labelled by frequency region only. A lane exists as data only if a detector publishes that label in `instrument_scores`; otherwise it reads "no detector".
- **Extending.** A new detector needs no renderer change: publishing a new label creates its lane (`gmv.visualizer.lanes` only adds titles, colours and nesting for labels such as kick, snare or lead guitar). A new event type appears on the timeline under its own name, drawn by its evidence kind.
- **Bounded memory.** Fixed-size arrays hold frame and lane history (4096 rows each), 2048 timeline marks, 64 gaps.
- **Drawing cost.** Scrolling histories are kept in pixmaps anchored to audio time (`gmv.visualizer.strip`): each refresh draws only what arrived since the last one and copies the visible part. Without this, redrawing 15 seconds of lanes took 17 ms per frame; with it, 0.3 ms.
- **Display spectrum.** Feature frames carry `spectrum_db`, 48 log-spaced bands from 30 Hz to 18 kHz read off the analyzer's existing FFT (see SCHEMA.md). The seven `band_energy` values were too coarse for a spectrum display.
- **Replay.** `gmv.visualizer.replay.ReplaySession` publishes a recorded session's messages on a private bus through the existing `session.replay`, marked `replayed`. No audio is read or written.

## One application (Goal B)

`gmv.app.window.MainWindow` is the only window: a top bar (status, Start/Stop, Settings), the visualizer view and a settings drawer that slides over the right edge. The former control center is gone; `gmv ui`, `gmv visualizer`, `gmv-control` and `python -m gmv.ui` all open this window.

- `gmv.app.controller.SessionController` has no widgets. It turns a `Settings` value into an audio source and an `AnalysisConfig`, owns the engine and any replay, exports logs and reports diagnostics. The window polls it four times a second.
- The window attaches the visualizer's own subscription before the engine starts. Display settings act on the view at once; engine settings changed while running are applied by an explicit restart (`Apply changes`).
- Closing the window detaches the view, stops the replay and the engine and waits for the engine's threads.
- Headless use is unchanged: `gmv capture`, `gmv file`, `gmv replay` and the evaluation tools do not import the application.

## Drawing cost (Goal B)

Software painting shares Python's interpreter lock with the DSP thread, so every millisecond of painting can delay analysis. The paint path now only draws what moves:

- everything that changes rarely (background, labels, legends, row backgrounds, header text) is one cached picture, redrawn only when what it shows changes;
- bars and glow are copied from cached textures; scrolling histories are cached strips;
- no wide antialiased strokes, per-frame gradients or per-frame text remain (each of those was measured at several milliseconds).

`scripts/benchmark_visualizer.py` measures a real on-screen window beside a running engine. A file source wakes on the same Windows timer tick as the display, which makes analysis and painting collide far more often than live capture does; live capture is the representative case.

## Detectors and note candidates (Goal B)

- **Activity detectors are weights files.** `ActivityBank` runs any number of them on one set of features; each publishes its own labels in `instrument_scores`. `guitar_model_path` is a second weights file (labels such as `guitar`, `guitar_distorted`, `guitar_clean`); a new label gets a lane without renderer changes.
- **Note candidates** (`gmv.pitch`, opt-in `pitch_tracking`): up to four simultaneous notes between E2 and E6 from an 85 ms window, by matching spectral peaks to harmonic series, rejecting overtones of accepted notes, requiring a note to persist for three frames, and reporting a new attack when a sounding note's own harmonics get clearly louder again. They are published with the audio time they describe (`pitch_time`, about 20 ms earlier than the frame) and as `note_attack_candidate` events, always as candidates.
- **Attribution.** The pitch tracker cannot tell which instrument plays a note. The visualizer draws notes neutral, and amber only for frames in which a guitar detector reported guitar. Without guitar weights nothing is attributed to guitar.

## Context recognition (guitar recognition milestone)

A second optional worker beside Tier B. `gmv.recognition.RecognitionWorker` receives every raw block from the analysis thread (`offer`, never blocking), keeps the last seconds of mono audio, and every hop asks a backend for one score per label. Scores become `detected`, `candidate` or `absent` and are published as `InstrumentEstimate` messages whose interval, not arrival time, places them on the audio timeline.

- **Replaceable model.** A backend is anything with `labels`, `sample_rate`, `context_s`, `hop_s`, `thresholds` and `infer(context)`. A JSON description file next to the weights names the backend kind, labels, thresholds, checksums and origin. The only kind shipped is `onnx_mel_tagger` (ONNX Runtime on the CPU, an optional dependency).
- **Bounded.** The worker's queue drops the oldest block when full and then starts its context again. A model slower than its hop skips hops; an estimate is always about the newest audio.
- **Failure is local.** A missing package, an unreadable model or an error during inference ends recognition only: one estimate with `available` false carries the reason, the guitar lane shows it, and capture, analysis and drawing continue.
- **One code path.** `gmv.evaluation.detect.run_recognition` feeds the same `Recognizer` offline; `scripts/score_recognition.py` scores models through it.
- **Display.** A lane has one source (frame detector or recognition model). Lane rows carry how far the detector commits; candidate is drawn weaker than detected. Notes take their guitar colour from the guitar lane when drawn, so a late estimate recolours them.

## Excerpt review (real-world guitar review)

`gmv.review` is an offline tool beside the application, not part of capture or drawing. It selects recordings from a local inventory by metadata, registers them in a corpus manifest as `owned_local`, runs the model over each whole recording through `run_recognition` (model output only), cuts non-overlapping 12 s excerpts in three categories (core, feature, model-flagged) as unchanged decoded samples, and opens a separate PySide6 review window that writes human labels to an append-only log. Labels and model output live in different files; the label store accepts annotation fields only. Reviewed core labels export to the corpus annotation format, where time outside the labelled excerpts is unknown. Method: [evaluation/REAL_WORLD_REVIEW.md](evaluation/REAL_WORLD_REVIEW.md).
