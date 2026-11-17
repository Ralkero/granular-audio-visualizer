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
