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

## Analysis in its own process

By default the application does not run the `Engine` itself. `gmv.remote.RemoteEngine` starts one worker process per session that runs the unchanged engine (capture, analysis, recognition, session logging) and forwards everything its EventBus publishes over a pipe, in order and unaltered, to an EventBus in the window's process (`EventBus.forward`). The window, replay and tests see the same messages either way. Drawing and analysis then no longer share an interpreter. `Settings.analysis_process` (or `GMV_ANALYSIS=thread`) runs the engine in the window's process as before. Measurements: [PERFORMANCE.md](PERFORMANCE.md).

## Musical intensity

The visual state keeps one number for the music as a whole, 0 to 1 (`VisualState.musical`, `gmv.visualizer.intensity`), worked out from fields the feature frames already carry: level against adaptive quiet and loud references, the share of energy below 250 Hz, and onset density, eased by audio time. No engine or schema change; identical in replay. The Radial and Spectrum Bars spectrum is coloured by it, neutral to dark red; any mode can read it. See [MUSICAL_INTENSITY.md](MUSICAL_INTENSITY.md).

## Live instrument state

Beside the activity history, the visual state keeps what each instrument is doing now (`VisualState.live`, `gmv.visualizer.live.Live`): eased level and energy, the ringing envelope, a bounded list of recent attacks with their strength and place, the note, and the presence weight. It is fed by the same rows as the history and advanced once a frame to the display's audio time less the analyzer's own delay, so every decay and oscillation is a function of audio time: the same at any frame rate and in replay. A live state belongs to one audio segment: when a feature frame starts a new segment (a capture gap or format change, even with continuous timestamps), every live state starts afresh while the history keeps all rows, and a late row of a finished segment is refused. Animation phases (`Live.phase`) are integrals over the rows in audio time from the segment's first row, so a ring's motion does not depend on how often, or whether, it was drawn. The radial rings are drawn from it every frame (`Layer.paint_live`); the strips of Spectrum Bars, the timeline and the lanes are drawn from the history as before. See [LIVE_INSTRUMENT_RESPONSIVENESS.md](LIVE_INSTRUMENT_RESPONSIVENESS.md).

## Instrument layers

Fast instrument analyzers (`gmv.instruments`) run inside the analysis pass on the per-channel FFT it already computes and publish one `InstrumentActivity` per frame and instrument: ringing level, articulation, attacks, repetition. They say what is being articulated, not which instrument it is; `gmv.instruments.combine` is the single rule that joins them with the recognition model's `InstrumentEstimate`. The visualizer keeps one activity history per instrument (`VisualState.activity`) and draws it twice from the same rows: as a layer inside the selected visualizer (`gmv.visualizer.layers`, one `Layer` per instrument, drawn incrementally through a surface that the mode supplies: a strip across the bars, a ring round the radial spectrum) and in the instrument's lane. Sensitivity belongs to the analyzer and can be changed on the running engine; intensity, style and visibility belong to the layer and never reach the analysis. Guitar, drums, bass and vocals each have an analyzer and a layer; presence comes from the recognition model for guitar and from the frame-wise activity model for the other three. See [INSTRUMENT_LAYERS.md](INSTRUMENT_LAYERS.md).

## Context recognition (guitar recognition milestone)

A second optional worker beside Tier B. `gmv.recognition.RecognitionWorker` receives every raw block from the analysis thread (`offer`, never blocking), keeps the last seconds of mono audio, and every hop asks a backend for one score per label. Scores become `detected`, `candidate` or `absent` and are published as `InstrumentEstimate` messages whose interval, not arrival time, places them on the audio timeline.

- **Replaceable model.** A backend is anything with `labels`, `sample_rate`, `context_s`, `hop_s`, `thresholds` and `infer(context)`. A JSON description file next to the weights names the backend kind, labels, thresholds, checksums and origin. The only kind shipped is `onnx_mel_tagger` (ONNX Runtime on the CPU, an optional dependency).
- **Bounded.** The worker's queue drops the oldest block when full and then starts its context again. A model slower than its hop skips hops; an estimate is always about the newest audio.
- **Failure is local.** A missing package, an unreadable model or an error during inference ends recognition only: one estimate with `available` false carries the reason, the guitar lane shows it, and capture, analysis and drawing continue.
- **One code path.** `gmv.evaluation.detect.run_recognition` feeds the same `Recognizer` offline; `scripts/score_recognition.py` scores models through it.
- **Display.** A lane has one source (frame detector or recognition model). Lane rows carry how far the detector commits; candidate is drawn weaker than detected. Notes take their guitar colour from the guitar lane when drawn, so a late estimate recolours them.

**Level before the model.** Captured audio follows the player's volume. In the first packaged session it arrived about 30 dB below ordinary music level, and `guitar_mn10_v1` loses guitar there: on the 223 reviewed excerpts, played 40 dB down, "detected or candidate" fell from 70% to 48% of guitar excerpts. `Recognizer` therefore raises a quiet 10 s context to -20 dBFS RMS (at most +45 dB; near-silence and audio already at that level are untouched) before inference. With it the same excerpts score the same at 0, -30 and -40 dB (70%; guitar-free excerpts 12.5% in every case).

## Evidence enrichment

`gmv.evidence` is offline tooling beside the application, like `gmv.review`: it never runs during capture or drawing and the application does not import it. `musicbrainz` and `enrich` identify recordings and collect credits with their scope; `fingerprint` computes Chromaprint locally and holds the AcoustID lookup behind an explicit approval flag; `leakage` builds split groups; `openmic` and `datasets` describe external references; `scan` runs the application's own `Recognizer` over a collection; `fusion` turns model timelines, credits, separation levels and human labels into candidates by evidence class; `queue` chooses what a person should hear; `outputs` writes the tables. The second audio model and separation need torch and live in `experiments/evidence`, in the experiments environment. Method and results: [evidence/EVIDENCE_ENRICHMENT.md](evidence/EVIDENCE_ENRICHMENT.md).

## Where files are found

`gmv.paths` is the one place that knows where models, sessions and the capture helper are. From a checkout they are in the repository (`.model-cache`, `sessions`, `native/...`). In a packaged build models and the helper are beside the executable and sessions go to the user's local application data. `models/defaults.json` in a build names the weights it starts with; each is applied once, so a later choice in Settings is kept. A checkout has no such file and experimental models stay off.

## Excerpt review (real-world guitar review)

`gmv.review` is an offline tool beside the application, not part of capture or drawing. It selects recordings from a local inventory by metadata, registers them in a corpus manifest as `owned_local`, runs the model over each whole recording through `run_recognition` (model output only), cuts non-overlapping 12 s excerpts in three categories (core, feature, model-flagged) as unchanged decoded samples, and opens a separate PySide6 review window that writes human labels to an append-only log. Labels and model output live in different files; the label store accepts annotation fields only. Reviewed core labels export to the corpus annotation format, where time outside the labelled excerpts is unknown. Method: [evaluation/REAL_WORLD_REVIEW.md](evaluation/REAL_WORLD_REVIEW.md).
