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
