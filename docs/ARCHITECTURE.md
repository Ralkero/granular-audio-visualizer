# Phase One architecture

Live Windows output is the primary source. No Spotify credentials, routing, volume or playback modifications are required. This repository contains capture/analysis and diagnostic meters only.

`AudioSource -> bounded audio queue -> window assembler -> Tier A DSP -> EventBus`

The bus feeds independent bounded queues for the desktop monitor, session writer, and future renderer. Logging never writes from the capture callback. Tier B uses a separate worker and bounded audio history. File sources reuse the assembler and DSP; fast offline producers wait for queue space so their analysis is lossless. Live capture drops the oldest queued block to favor freshness. Queue drops and sample discontinuities are counted; context is reset rather than splicing missing audio into a false transient.

Message schema 1.1 uses immutable Pydantic models. Times are seconds from the session audio origin; feature timestamps refer to the window center, with window_end supplied separately. Events report evidence time, not task-completion time. Detection/publication monotonic times are separate and only comparable within the same run. Reconnected sources have new segment IDs. Recovery gaps remain gaps, not fabricated silent audio. Schemas support event groups and relationships without asserting instrument identity.

Initial 46.44 ms windows and 20 ms hops trade frequency resolution for responsiveness. Native device rates are analyzed without compulsory resampling. Band powers aggregate channel spectra (avoiding cancellation in opposite-phase stereo). A short window cannot resolve closely spaced low bass notes. Onsets and percussion labels are heuristic candidates; tempo requires historical evidence. Short-term loudness is unweighted RMS dBFS, not standardized LUFS. Adaptive display intensity blends absolute energy with a slowly moving relative reference; it does not change audio gain.

Privacy: raw audio is never written by default. Session metrics can reveal listening patterns and device/app names; sessions and local evidence are excluded from Git. There is no network listener. A future renderer subscribes directly through EventBus. Replay uses persisted messages, not rewritten live JSON files.

Linux: preserve the source contract for a PipeWire monitor adapter. It must connect to a sink monitor without changing the default device. No Linux implementation or hardware validation is claimed in this Windows milestone.

Phase 1.1 adds strict legacy migration, consumer freshness policy, generation invalidation for delayed jobs and an isolated offline CPU model experiment. Immediate onsets are per-band: a different frequency region can register inside the short grouping window, and a weak ripple after a louder attack does not. Reported band edges are measurement spans, not instrument names. See SCHEMA.md and PHASE1_1_REPORT.md. No model inference runs in the immediate pipeline.
