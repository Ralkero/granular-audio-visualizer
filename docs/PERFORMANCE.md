# Visualizer performance: baseline, changes, results

Measured on the development machine (Ryzen 9 5950X, 32 GB, Windows 11, Python 3.14,
PySide6 raster painting on the CPU; no GPU path is used or required). Window 1440 x 900 at
device pixel ratio 1, all four instrument layers on. "Before" is the checkout at the end of
instrument layers phase 2; "after" is this work.

## How it was measured

| Benchmark | What runs | Use |
|---|---|---|
| `scripts/benchmark_ui.py` | A recorded session's messages (analysis only, never audio) replayed at playing speed through the real window, on screen. No capture, no analysis. | The visualizer's own cost: intake, every part of the paint, frame pacing. Repeatable. |
| engine + window on a fixed synthetic band (local script) | File source at playing speed, all four analyzers, activity model, recognition model, pitch tracking, and the window. Old and new code on the same file. | What analysis and drawing cost each other. |
| the analysis pass alone, profiled | 20 s of the same synthetic band, no window | Where analysis time goes. |

The replayed session is 40 s of the owner's playing with all four layers active: 50 feature
frames, 200 instrument-activity messages, 2 recognition estimates and about 11 events per
second. Each replay figure is from 32 s after a 4 s warm-up, four configurations: radial
and bars, each with all sections shown and with none.

The window keeps rolling timings of every part (`gmv.visualizer.perf`,
`VisualizerWindow.performance()`): tick, messages per tick, backlog, intake time, paint,
frame interval, spectrum, each layer, timeline, details, notes, with p50, p95, p99 and
maximum; strips count their slides and full redraws.

## Baseline findings

**Not a bottleneck (measured, left alone or only bounded):**

- Message intake. 5 messages per frame at the median, 15 at most, 0.13 ms; backlog never
  above 15. The old loop could take 2,048 messages in one frame after a stall; it is now
  bounded, but in normal playback it was never the problem.
- The analysis pass alone: 2.35 ms per 20 ms frame at the median, 3.3 ms p95, 5 ms max,
  of which the four instrument analyzers take 0.63 ms.
- The spectrum (1.3 ms radial, less for bars), timeline (0.25 ms), details (0.5 ms).

**Confirmed causes of uneven animation:**

1. **Every instrument layer redrew its whole history whenever any instrument's judgement
   changed.** One shared revision counter keyed all four caches: 36 full redraws per layer
   in 32 s (144 in all), all four in the same frame. That is the paint's tail: p99 13 ms,
   maximum 19 to 21 ms, against 3 to 4.5 ms at the median.
2. **The note section was redrawn in full on every recognition estimate** (twice a
   second, 65 times in 32 s), to recolour notes the estimate had not reached yet.
3. **Frames were lost outright.** With a fixed 16 ms timer, a wake-up that came late cost
   a whole frame: 39 to 53 frame intervals above 25 ms per 32 s run (p99 30.7 ms) in three
   of four configurations, although painting itself took 4 ms. The timer also ran at
   62.5 Hz, not 60.
4. **Analysis and drawing share one interpreter.** With the engine running, analysis
   frames take 3.6 ms at the median but 8 to 10 ms at p95 and up to 20 ms, because the
   drawing thread holds the interpreter part of the time, and the other way round.
5. Scrolling strips allocated a new picture and copied the old one into it at every slide
   (16 slides per layer in 32 s). Small next to the above, but avoidable.

## Changes

| Change | Where | Effect |
|---|---|---|
| A judgement revision per instrument; a layer's cache is keyed by its own | `lanes.LaneBoard.judged`, `layers.Layer.paint_stage` | Layer history redraws 144 → 33 to 36 per run; a change for one instrument redraws that instrument only |
| Notes take the held judgement and are redrawn when it changes | `window.draw_notes` | Note redraws 65 → 4 per run |
| Frames on a fixed 60 Hz grid; the timer wakes when the next frame is due, a late wake-up draws at once and the grid is kept | `VisualizerWindow.poll` | Frame intervals above 25 ms: 39 to 53 → 0 per run in replay; cadence exactly 60.0 fps |
| Message intake bounded per frame (6 ms or 1,024 messages), in order, nothing skipped; the rest stays queued | `VisualizerWindow.tick` | No effect in normal playback; a burst can no longer hold up a frame |
| Strips slide in place (`QPixmap.scroll`) instead of allocating and copying | `strip.Strip` | No allocation per slide |
| Radial instrument rings never move: a hand sweeps round and each ring keeps what was drawn until it returns | `strip.Sweep` | No slides or copies at all in radial mode |
| Hidden sections are not painted; Details and Notes are hidden by default | `window.Canvas.paintEvent` | Their paint time is zero when hidden |
| The interpreter hands over between threads every 1 ms instead of 5 | `app.window.application` | Shorter waits for whichever thread is waiting |
| Both quantiles of a band's spectral contrast from one call | `dsp.Analyzer.process` | Same values; about a third of a millisecond less per analysis frame |

Nothing was changed in what the analysis computes or publishes, and no message is dropped
or coalesced to gain speed.

## Results

### The visualizer alone (replay), before → after

| | Radial, all sections | Radial, none | Bars, all sections | Bars, none |
|---|---|---|---|---|
| Frames per second | 62.6 → 60.0 | 62.1 → 60.0 | 62.5 → 60.0 | 62.5 → 60.0 |
| Paint p50, ms | 4.54 → 4.57 | 3.73 → 4.78 | 3.65 → 3.69 | 2.94 → 4.25 |
| Paint p95, ms | 7.07 → 6.24 | 5.68 → 6.33 | 5.50 → 4.95 | 4.59 → 5.87 |
| Paint p99, ms | 12.99 → 7.73 | 13.22 → 7.45 | 11.72 → 7.36 | 12.52 → 7.03 |
| Paint max, ms | 19.43 → 18.38 | 20.59 → 17.97 | 18.04 → 13.06 | 17.19 → 11.51 |
| Frame interval p99, ms | 30.72 → 18.35 | 30.89 → 18.52 | 17.33 → 18.56 | 30.94 → 18.58 |
| Frame interval max, ms | 32.04 → 19.47 | 31.53 → 19.25 | 19.05 → 18.86 | 31.64 → 19.20 |
| Intervals above 25 ms | 39 → 0 | 53 → 0 | 0 → 0 | 41 → 0 |
| Layer history redraws | 144 → 36 | 144 → 33 | 144 → 36 | 140 → 33 |
| Tick p99, ms | 0.59 → 0.43 | 0.52 → 0.48 | 0.51 → 0.42 | 0.65 → 0.54 |
| Messages dropped by the window | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| CPU, % of one core | 43 → 37 | 40 → 35 | 30 → 30 | 29 → 32 |
| Memory, MiB | 223 → 222 | 215 → 234 | 218 → 226 | 217 → 241 |

The median paint is not lower, and with no sections it is higher: the instruments are now
drawn across the whole visualizer instead of in short bands, so there is more to fill. What
changed is the tail and the pacing. The targets (paint p95 under 12 ms, p99 under 16.67 ms
at 1440 x 900) are met with room.

### Engine and window together (synthetic band, 30 s), before → after

| | Radial | Bars |
|---|---|---|
| Frames per second | 62.0 → 59.6 | 62.0 → 59.5 |
| Paint p99 / max, ms | 12.74 / 33.55 → 9.55 / 18.74 | 11.65 / 27.84 → 7.85 / 18.34 |
| Frame interval p50 / p95 / p99 / max, ms | 15.99 / 16.94 / 22.94 / 34.63 → 16.63 / 18.26 / 21.29 / 30.44 | 16.01 / 16.73 / 21.35 / 33.98 → 16.69 / 18.14 / 21.02 / 29.55 |
| Intervals above 25 ms (of about 1,800) | 8 → 9 | 7 → 7 |
| Analysis frame p50 / p95 / p99 / max, ms | 3.75 / 9.56 / 14.05 / 20.20 → 3.07 / 8.65 / 10.76 / 15.23 | 3.44 / 7.23 / 11.35 / 16.26 → 2.98 / 7.84 / 9.96 / 13.84 |
| Capture blocks dropped | 0 → 0 | 0 → 0 |
| Capture queue high-water (of 16) | 8 → 6 to 7 | 8 → 7 |
| CPU, % of one core | 80 → 74 | 70 → 73 |

## What remains

- **A few frames in a thousand are still late with the engine running** (7 to 9 intervals
  above 25 ms in 30 s, worst about 30 ms). The cause is finding 4 above: analysis and
  drawing take turns in one interpreter, and an analysis frame can hold it for 8 to 15 ms.
  Painting is no longer the limit. Removing this needs analysis in a separate process (or
  an interpreter without the shared lock); that is an architectural change and was not
  made here.
- Frame intervals at p95 are 18 ms, not 16.7: the timer's resolution is a millisecond and
  a wake-up is often a little late. No frame is lost to it.
- Painting is on the CPU. At 2560 x 1440 the paint was 6.6 ms at the median in an earlier
  measurement of the four layers; it has not been re-measured at that size after this work.
- The live figures above are from a file source and a synthetic band. On real capture the
  owner's last session before this work had no dropped blocks; a session after it has not
  been recorded yet.

---

# Phase 3: interpreter contention, frame rate and the graphics card

Same machine. **All on-screen measurements of this phase were made on the second display
(1920 x 1080, 144 Hz, device pixel ratio 1), full screen**, at the owner's request not to
use the main one. Qt reports the main display as 2560 x 1440 at 180 Hz. Nothing here was
measured at 2560 x 1440.

Workload "engine + window": the fixed synthetic band from a file at playing speed, all
four analyzers, activity model, recognition model, pitch tracking, the window drawing all
four layers. 30 s per run, the first 240 frames left out of the interval figures. The
machine was in ordinary use by its owner during the runs, so single outliers (one frame of
40 to 50 ms in a run) come and go between identical runs.

## Findings

1. **Analysis and drawing wait for each other in one interpreter, and it gets worse the
   more often the window draws.** In one process, analysis frames took 3.6 ms at the median
   with the window at 60 fps, 7.9 ms at 120 fps and 16.2 ms at 144 fps, where the 95th
   percentile (21.6 ms) is longer than the 20 ms the analysis has per frame. 144 fps was not
   reached (127). The analysis is mostly short NumPy calls with Python between them
   (message objects, per-band loops): each call releases the interpreter only briefly, so
   the drawing thread and the analysis thread take turns. Handing over every 1 ms instead
   of 5 ms (phase 2) shortens each wait but does not make them run at once.
2. **In its own process the analysis does not notice the window at all**: 2.7 to 2.9 ms at
   the median and at most 7.5 ms at every frame rate tried.
3. **The frame rate should divide the display's.** 60 fps on a 144 Hz display cannot give
   every frame the same time on screen; 72 can. On a 180 Hz display it is 60 that divides.
4. **The graphics card is not the limit.** Painting takes 5 to 9 ms of CPU at 1080p at the
   95th percentile, inside a 13.9 ms frame; once the analysis is out of the way no frames
   are late.

## Frame rates, one process against two (radial, engine + window)

Frames later than 20 ms / 25 ms / 33.3 ms, of the frames in the run; interval and analysis
times in ms.

| Target | Engine | Achieved | Interval p50 / p95 / p99 / max | Late 20 / 25 / 33 | Analysis p50 / p95 / max | CPU, % of a core |
|---|---|---|---|---|---|---|
| 60 | same process | 59.4 | 16.63 / 18.23 / 25.68 / 74.58 | 27 / 18 / 2 of 1,543 | 3.62 / 11.86 / 28.86 | 82 |
| 60 | own process | 60 | 16.66 / 18.07 / 18.55 / 19.63 | 0 / 0 / 0 of 1,447 | 2.72 / 3.83 / 5.88 | 42 + 38 |
| 72 | same process | 71.4 | 13.92 / 15.12 / 20.42 / 26.14 | 22 / 3 / 0 of 1,903 | 3.31 / 9.30 / 21.61 | 93 |
| 72 | own process | 72 | 13.91 / 15.09 / 15.46 / 45.07 | 4 / 2 / 1 of 1,799 | 2.68 / 3.79 / 7.49 | 44 + 35 |
| 120 | same process | 116.7 | 8.31 / 10.15 / 17.92 / 49.60 | 23 / 8 / 6 of 3,258 | 7.87 / 19.11 / 53.16 | 120 |
| 120 | own process | 120 | 8.34 / 9.62 / 10.25 / 13.45 | 0 / 0 / 0 of 3,156 | 2.82 / 3.91 / 7.09 | 68 + 42 |
| 144 | same process | 127.0 | 7.40 / 10.51 / 14.22 / 23.75 | 14 / 0 / 0 of 3,570 | 16.21 / 21.62 / 28.55 | 152 |
| 144 | own process | 144 | 6.97 / 8.19 / 8.74 / 10.37 | 0 / 0 / 0 of 3,833 | 2.85 / 4.00 / 5.25 | 78 + 34 |

No capture blocks were dropped in any run. "Achieved" for the own-process rows is from the
median interval; the run's average is lower because the worker takes about two seconds to
start. 144 fps is sustainable with the analysis in its own process, for about three
quarters more CPU in the window's process than 72 fps.

## What was adopted

| Change | Where | Default |
|---|---|---|
| Capture, analysis, recognition and session logging run in a worker process; the window receives the same messages over a pipe | `gmv.remote.RemoteEngine`, `EventBus.forward`, `app.controller` | on (Settings → Advanced, or `GMV_ANALYSIS=thread`, to turn off) |
| Frame rate chosen from the display the window is on: its refresh rate divided by a whole number, as near 66 as that gets; chosen again when the window moves to another display | `VisualizerWindow.set_frame_rate`, `automatic_rate` | automatic (Settings → Visualizer → Frame rate offers "every refresh") |
| After a late wake-up the next frame is at least half a frame away: no catch-up burst | `VisualizerWindow.poll` | always |
| The canvas can be an OpenGL surface | `GMV_RENDERER=opengl` | off: experimental |

**The worker.** One persistent process per session runs the unchanged `Engine`. Messages
cross in batches gathered for 4 ms, in publication order, exactly as published: the same
objects, sequence numbers, timestamps and ages; nothing is merged, restamped or skipped.
The audio path is entirely inside the worker, so nothing in the window's process can delay
capture. Stop and sensitivity changes go the other way. If the worker fails to start or
dies, the session ends with an error saying so, the window carries on, and the next Start
makes a new worker. A session only counts as over once its worker has gone, and the worker
is a daemon process, so it cannot outlive the application. A shared-memory ring buffer for
audio was not built: the audio never crosses the process boundary, only the results do, at
about 260 small messages a second.

**Cost.** Memory 314 to 327 MiB in one process against 236 + 191 MiB in two (about 110 MiB
more). Starting a session takes about two seconds longer, during which the window shows
"Starting the analysis process". Total CPU is about the same (81 to 96% of a core before,
47 to 52% + 34 to 36% after), now on two cores.

## Before and after (engine + window, this phase's default against the last one's)

| | Radial | Bars, all sections |
|---|---|---|
| Frame rate | 60 → 72 | 60 → 72 |
| Interval p50, ms | 16.68 → 13.90 | 16.64 → 13.90 |
| Interval p99, ms | 25.36 → 15.63 | 22.40 → 15.80 |
| Interval max, ms | 30.75 → 17.48 | 42.17 → 21.53 |
| Frames later than 20 ms | 27 of 1,544 → 0 of 1,800 | 23 of 1,544 → 1 of 1,799 |
| Frames later than 25 ms | 18 → 0 | 7 → 0 |
| Paint p95 / p99, ms | 8.89 / 10.71 → 6.69 / 8.43 | 7.16 / 8.68 → 7.09 / 7.65 |
| Analysis p50 / p95 / max, ms | 4.37 / 12.79 / 22.68 → 2.72 / 4.05 / 6.60 | 3.11 / 9.57 / 32.49 → 2.74 / 3.99 / 7.40 |
| Capture blocks dropped | 0 → 0 | 0 → 0 |
| CPU, % of a core | 96 → 52 + 36 | 81 → 47 + 34 |
| Memory, MiB | 327 → 238 + 191 | 314 → 236 + 191 |

The visualizer alone (session replay, no engine) at the automatic 72 fps, 1080p: 72.0 fps,
paint 4.4 to 6.1 ms at the median, 6.3 to 8.7 ms p95, 9.2 to 10.9 ms p99.

## The graphics card experiment

With `GMV_RENDERER=opengl` the canvas is a `QOpenGLWidget` and the same QPainter code draws
through OpenGL. The context reported **AMD Radeon RX 9070 XT**, OpenGL 4.6 compatibility
profile, driver 26.11.2. Same workload, analysis in its own process, 72 fps:

| | CPU painting | OpenGL |
|---|---|---|
| Paint p95 / p99, ms (radial) | 6.48 / 7.46 | 5.30 / 5.82 |
| Paint p95 / p99, ms (bars) | 6.21 / 6.94 | 4.52 / 5.45 |
| Interval p99, ms (radial / bars) | 15.49 / 15.53 | 15.43 / 15.36 |
| Memory of the window's process, MiB | 238 to 247 | 434 to 475 |

About a fifth to a quarter less CPU time in the paint, the same frame pacing, about 200 MiB
more memory. "Paint" is the CPU time of the paint call; the card's own execution time and
its utilisation were not measured. The picture was not compared by eye (lines and
antialiasing can differ under the OpenGL paint engine). On this evidence it is not worth
making the default; it stays an opt-in experiment and CPU painting is unchanged.

## Not done, and what remains

- Nothing was measured at 2560 x 1440 or on the 180 Hz display in this phase. Paint time
  grows with the area; the phase 2 figure at that size was about 6.6 ms at the median.
- No measurement with real capture and real music after the change; the figures are from
  a file source. Whether it looks smooth is for the owner to see.
- Presentation was measured as the time between paints. When the desktop compositor
  actually showed each frame was not measured, and the frame grid is not locked to the
  display's vertical sync: at 72 fps on 144 Hz each frame covers two refreshes on average,
  but which two can slip by one now and then.
- Changing the refresh rate while running, minimising, and long sessions were not
  exercised by hand. Moving the window to another display chooses the rate again (the rate
  function is tested; a real window was not moved).
- Per-core utilisation, context-switch counts and graphics-card utilisation were not
  recorded.
- An occasional single late frame (one in a run of 1,800) still appears in some runs with
  the machine in other use.
