# Live instrument responsiveness

Phase 1 (diagnosis) comes first; phase 2 (what was built from it) follows at the end.

## Phase 1: diagnosis

The question: what does each instrument analyzer produce, how quickly, and why do the four
radial instrument rings not look like live, synchronized animation?

Nothing user-facing was changed in this phase. It adds measurement tools, tests for them,
and this report. No model was retrained, no threshold changed, no renderer replaced.

## Summary of what was found

1. **The update rate is not the problem.** Every analyzer publishes 50 times a second, with
   timestamps exactly 20 ms apart (median and 95th percentile both 20.0 ms in three
   recorded live sessions). The window takes every row in and draws it 12 to 13 ms after
   publication at the median (19 ms at the 95th percentile). No row was dropped, merged or
   overwritten anywhere on the way.
2. **The rings are a history by construction.** Each ring is a cached picture of the last
   six seconds. A frame writes only the newest 20 ms slice at the hand and leaves the rest
   as it was. Measured on a recorded session, less than 1% of a ring's drawn pixels change
   from one frame to the next (median 0.6 to 0.8%; drums 0%), against 10% for the whole-mix
   spectrum (82% at the 95th percentile). Nothing in an instrument's picture moves after it
   has been drawn. This is the main cause of what was criticised, and it is a rendering
   design, not a defect and not an analysis limit.
3. **Two analyzers do not provide what a live picture of them needs.**
   - *Bass.* A note played again at the same pitch is mostly not reported as an attack
     (1 of 12 at two notes a second on a controlled signal; 38% of the bass stem's onsets
     in real mixes), and the level barely falls between notes (it gives back 3 to 23% of
     its peak). When it is reported, the same-note attack is stamped about 55 ms too
     early. The continuous level is known 105 to 125 ms after the sound.
   - *Every level signal* (bass, vocals, drums, guitar) is measured against its own peak
     of the last four seconds over a 30 dB range, and sits near the top whenever the
     instrument plays. A bass line growing by 25 dB under a steady band moves its level
     from 0.81 to 0.97. A gradual build cannot show in it.
4. **Guitar and vocals are fast and complete enough already.** Guitar attacks: all found on
   controlled signals, placed within 3 ms, known 46 ms after the stroke; 84% of the guitar
   stem's onsets found in real mixes. Vocal level and pitch are known 26 to 46 ms after
   the sound, every frame; only phrase starts (106 to 126 ms) and ends (about 340 ms)
   wait for confirmation.
5. **Drums are complete but uniformly late.** 90% of the drum stem's onsets are found in
   real mixes, placed within 3 to 9 ms. All three regions are reported 86 ms after the hit,
   because the 60 ms decay check that only the low region needs delays the one message all
   three share.
6. **The learned models gate; they do not animate.** They decide how much of an
   instrument's fast activity is shown (all, weakened, tentative, none). They never supply
   the signal that moves. In the recorded sessions they hid 20 to 33% of the frames in
   which the bass analyzer was active, 12 to 60% for vocals and 11 to 48% for drums;
   whether that was right cannot be told from a session (there is no reference in it).
   Retraining would not make the rings move.

So the next phase should begin with the rendering: a live state per instrument, read at
the display's frame rate, and live radial renderers on top of it. The bass analyzer needs
a small addition alongside. Details in the plan at the end.

## A. Current architecture

```
capture (10 ms blocks)                              one process (the analysis worker)
  -> WindowAssembler: 46 ms window, every 20 ms
  -> Analyzer.process (gmv.dsp), one pass, one FFT per channel
       -> FeatureFrame (whole mix; carries the activity model's per-frame presence
          for drums, bass, vocals: activity_v2, a small model on band features)
       -> four instrument analyzers (gmv.instruments), each handed the same spectrum
          (bass also the frame's new samples) -> one InstrumentActivity each, every frame,
          time-stamped at the moment it describes (frame time less the analyzer's delay)
  -> RecognitionWorker (own thread): guitar_mn10, 10 s context, every 0.5 s
       -> InstrumentEstimate (guitar: detected / candidate / absent)
  -> EventBus -> pipe (batches every 4 ms) -> EventBus in the window's process

window (frame timer, 72 or 60 fps)
  -> tick: take in every queued message, in order (VisualState.consume)
       InstrumentActivity -> VisualState.activity[instrument]: a Track of rows
                             (the analyzer's fields plus derived ones: a "ring" envelope,
                             time/place of the last attack)
       FeatureFrame / InstrumentEstimate -> LaneBoard: presence judgements over time
  -> paint: Canvas.draw_layers
       Radial: a Stage per layer with a Polar surface, RING_SECONDS = 6
               Layer.paint_stage -> Sweep.paint -> Layer._slice -> paint_history(rows
               that arrived since the last paint) into the ring's own cached pixmap;
               the pixmap is then copied to the screen unmoved; a "hand" line marks now
       Bars:   the same paint_history through a Linear surface into a Strip that slides
       Timeline / details lanes: paint_lane, separate pictures of the same rows
```

Where things are decided:

| What | Where |
|---|---|
| Six-second ring period | `RING_SECONDS` in `visualizer/window.py`, passed as `Stage.seconds` |
| The hand and the retained ring | `Sweep` in `visualizer/strip.py`; the hand line in `Canvas.draw_layers` |
| Clearing ahead of the hand | `Sweep.paint` (`gap` = `layers.GAP`, 6% of a turn) |
| Only-new-rows drawing | `Layer.paint_stage` / `Layer._slice` (`drawn_until`) |
| Time to band space, band space to ring | `Stage`, `layers/surface.py` (`Polar`, `Linear`) |
| How much activity is shown | `VisualState.attribution` -> `instruments.combine` |
| When a ring is redrawn whole | its presence judgement changed (`LaneBoard.judged`), style or intensity changed |

Answers to the questions asked of the renderer:

1. The six-second behaviour is one constant and the `Sweep` class.
2. The sweep is `Sweep` plus `Layer.paint_stage`'s polar branch.
3. The history behaviour is shared in the sense that Radial rings, Bars strips and the
   detail lanes all read the same `Track` rows; they are three separate pictures. The lanes
   use `paint_lane` and their own strips and do not depend on the rings.
4. A separate live radial presentation is possible without touching the lanes or the
   strips: it needs another paint path for polar stages. The `Stage` / surface split
   already separates "where" from "what".
5. The cached pixmaps are what makes the present design a history, but they are private
   to each layer (`layer.sweep`), so not using them for a live radial picture disturbs
   nothing else.
6. There is one frame timer for the whole canvas. Layers have no schedules of their own;
   each is painted every frame. No layer is tied to a slower update.

The guitar layer's own description says it plainly: "Nothing is animated on its own: with
no activity rows there is no picture." A string's wave is a function of time *along the
ring*, drawn once; it does not oscillate in place.

## B. Per-instrument findings

Latency terms used below, all in audio time (no computer in them):

- **placed**: reported time less true time: how accurately the event sits on the timeline.
- **known**: how much audio after the event had to exist before the analyzer reported it.
  It is the analysis window's half (23 ms) plus the analyzer's own delay.

### Bass (`bass_pitch_v1`)

| | |
|---|---|
| Inputs | The frame's new samples (mono), low-passed at 180 Hz, thinned to 3 kHz, 170 ms kept; and the shared spectrum, 38 to 300 Hz |
| Cadence | 50 messages a second; 0.19 ms a frame (0.29 ms p95) |
| Windows | 170 ms for pitch *and for the level* (its RMS); 46 ms for the "struck again" check |
| Back-dating | 4 frames (80 ms): every message describes the middle of the 170 ms window. Known 103 ms after the moment described |
| Published | `level` (periodic low-register weight against its 4 s peak, eased over 60 ms), `flux` (rise of that), `attack`, `pitch_midi`, `pitch_confidence`, `centre`, repetition |
| Recognition | Activity model per frame: shown or hidden. No delay of its own beyond the frame |

Measured, analyzer alone, controlled signals:

| Signal | Attacks found | Placed (p50) | Known (p50 / p95) | Level falls back between notes | Right pitch known |
|---|---|---|---|---|---|
| Same note, 2 a second | 1 of 12 | | | 23% | 106 / 115 ms |
| Same note, 4 a second | 12 of 24 | -57 ms | 46 / 82 ms | 10% | 101 / 106 ms |
| Same note, 6 a second | 24 of 36 | -50 ms | 53 / 53 ms | 6% | 106 / 113 ms |
| Same note, 8 a second | 35 of 48 | -57 ms | 46 / 51 ms | 3% | 104 / 111 ms |
| Changing notes, 4 a second | 24 of 24 (17 extra) | +33 ms | 136 / 146 ms | 11% | 136 / 146 ms |
| Plucked, dying away, 1 a second | 6 of 6 | +3 ms | 106 / 121 ms | 91% | 106 / 121 ms |
| One held tone, 4 s | 1 of 1 | +23 ms | 126 ms | level 1.00, no movement at all | 126 ms |
| Same note, 2 a second, in a band | 1 of 16 (18 extra) | | | 33% | 106 / 111 ms |
| Same note, 4 a second, in a band | 9 of 32 (9 extra) | | | 15% | 101 / 106 ms |

On 22 real songs with stems (1,531 onsets of the bass stem, 2.4 a second): an attack was
reported for 38% of them in the full mix and for 57% on the bass stem by itself, known
119 ms after the onset. The level rose by 0.1 at 74% of the onsets and fell back by 23% of
its peak between them.

What this shows:

- **Repeated same-pitch notes are mostly not attacks.** The "struck again" rule compares
  the newest frame with the three before it. A note that returns to about the level it had
  60 ms earlier does not clear it; and it is only consulted while the pitch is confident.
  Notes are found when they die away first (the plucked case) or change pitch.
- **Same-note attacks are stamped too early.** That check reads the newest 46 ms, but the
  message carrying it is back-dated by the 80 ms that belong to the pitch window. Those
  attacks land 50 to 57 ms before the note. New-note attacks, which come from the pitch
  path, land 23 to 33 ms late.
- **The energy is slow because it is measured on the pitch window.** The level is the RMS
  of the same 170 ms, then eased over 60 ms. The 170 ms are needed for the pitch (E1 and F1
  are 2.5 Hz apart) but not for energy: the 46 ms spectrum the analyzer already reads for
  its "struck" check would give a low-band energy 80 ms sooner. That faster signal exists
  inside the analyzer and is not published.
- **A held note produces no movement at all** (level 1.00, variation 0, flux 0), and
- **a build does not show**: growing 25 dB under a steady band moved the level from 0.81 to
  0.97; alone, quiet and strong notes both read 0.98.
- In a band the kick drum produces attacks of its own (18 extra for 16 notes).

The renderer draws thickness from `level` and a 2-pixel bright edge for an attack. With a
level that neither pulses nor grows, and attacks that mostly are not there, the ribbon is a
slab even before the history question.

### Guitar (`guitar_articulation_v1`, accepted; unchanged)

| | |
|---|---|
| Inputs | Shared spectrum, 200 Hz to 5 kHz: the stereo difference (mono band as fallback) |
| Cadence | 50 a second; 0.17 ms a frame (0.33 ms p95) |
| Windows | 46 ms; an attack is a local maximum, decided one frame later |
| Back-dating | 1 frame (20 ms). Known 43 ms after the moment described |
| Published | `level` (ringing), `flux` (articulation), `attack`, `centre`, `spread`, `rate_hz`, `regularity` |
| Recognition | `guitar_mn10` every 0.5 s on 10 s of context; held for 1.5 s between reports |

| Signal | Attacks found | Placed (p50 / p95) | Known (p50 / p95) | Level falls back |
|---|---|---|---|---|
| Strums, 10 a second | 60 of 60 | 3 / 3 ms | 46 / 46 ms | 18% |
| Strums, 3 a second | 18 of 18 | 3 / 10 ms | 46 / 53 ms | 34% |
| Single strokes dying away | 6 of 6 | 3 / 3 ms | 46 / 46 ms | 70% |
| Alternating strong and weak, 4 a second | 24 of 24 | -7 / 3 ms | 36 / 46 ms | 57% |
| Quiet alone, then with a band, 8 a second | 64 of 64 | -2 / 8 ms | 41 / 51 ms | 12% |
| Strums, 10 a second, mono recording | 50 of 60 | 3 / 3 ms | 46 / 46 ms | 18% |
| One held chord, 4 s | 1 of 1 | | | level 0.76 with movement (0.11), no false attacks |

On 18 real songs (880 onsets of the guitar stem): 84% found in the mix, 90% on the stem,
placed 4 ms, known 48 ms after the onset.

Attack strength is compressed: strokes 8 dB apart read 1.00 and 0.77. Several attacks
between two display frames did not occur in the trace (at most one per paint), and the
state keeps every row regardless. The analysis is timely and complete; what is missing is
entirely on the drawing side. The visual state already derives a `ring` envelope (jumps at
an articulation, dies away over 60 ms to half a second) and the time and place of the last
stroke for every row, which is most of what a live string needs.

Recognition is the guitar's weak side for liveness. In the longest recorded session 31% of
guitar attacks were drawn tentative and 32% weakened. The recognition report measured
entrances: shown as detected within 5 s for 24 of 60, median delay 1.0 s when detected.
That is a delay in *whether* the guitar is shown, not in how its strokes move.

### Vocals (`vocal_contour_v1`)

| | |
|---|---|
| Inputs | Shared spectrum, centre of the stereo image, 150 Hz to 4 kHz |
| Cadence | 50 a second; 0.13 ms a frame (0.19 ms p95) |
| Windows | 46 ms; level eased over 50 ms; pitch averaged with the frame before |
| Back-dating | None. Known 23 ms after the moment described |
| Published | `level`, `pitch_midi`, `pitch_confidence`, `centre`, `boundary` (start / end), `attack` at a start |
| Recognition | Activity model per frame; it also decides when a phrase may start or end |

Continuous, every frame: level, pitch, pitch confidence. Only at transitions: phrase start
(after 100 ms above threshold) and end (after 250 ms below).

| Measured | Result |
|---|---|
| Level rises at an onset, known | 26 to 46 ms |
| Right pitch known, voice alone / in a band | 26 ms / 126 to 142 ms |
| Pitch sweep, an octave in 3 s | a pitch in every frame, error 0.09 semitone (0.26 p95), lag 22 ms (65 ms p95) |
| Vibrato, 2 semitones peak to peak at 5.5 Hz | read as 1.54 semitones |
| Phrase start known | 106 to 126 ms after the onset; **stamped at the time of confirmation**, 100 ms late on the timeline |
| Phrase end known | 336 to 346 ms after the voice stopped |
| Level between short phrases | falls back fully |
| Quiet, then strong singing | level 1.00 and 1.00 |
| Growing 26 dB under a steady band | level 0.77 to 0.93 |

On 23 real songs: the level rose at 93% of the vocal stem's onsets; the activity model
reported vocals at 89% of them; after 1.5 s without the voice it took a median 0.68 s
(2.6 s p95, 24 entrances) for the model to report it again.

The analyzer already gives a renderer enough fast information to move strands
continuously without waiting for a phrase boundary. Its weaknesses for this purpose: the
level does not carry dynamics, and the model's judgement lags an entrance by most of a
second.

### Drums (`drum_impacts_v1`)

| | |
|---|---|
| Inputs | Shared spectrum: centre for 40 to 140 Hz (low) and 180 Hz to 2.5 kHz (mid), whole image for 6 to 14 kHz (high) |
| Cadence | 50 a second; 0.08 ms a frame (0.20 ms p95) |
| Windows | 46 ms; a hit is a local maximum (one frame); a low hit is confirmed by what the region does 60 ms later |
| Back-dating | 3 frames (60 ms) for the whole message. Known 83 ms after the moment described |
| Published | `attack`, `components` (strength per region), `flux`, `level`, `centre`, repetition |
| Recognition | Activity model per frame |

| Signal | Hits found | Placed (p50) | Known (p50 / p95) | Regions that answered |
|---|---|---|---|---|
| Kick, 2 a second | 12 of 12 (13 extra) | 3 ms | 86 / 86 ms | low 11, mid 12, high 7 |
| Snare, 2 a second | 12 of 12 | 3 ms | 86 / 86 ms | low 12, mid 12, high 12 |
| Hi-hat, 8 a second | 48 of 48 | -2 ms | 81 / 91 ms | high 48 only |
| Tom, 2 a second | 12 of 12 (13 extra) | 3 ms | 86 / 86 ms | mid 12, high 12 |
| Kick and snare together | 12 of 12 | 3 ms | 86 / 86 ms | low 11, mid 12, high 12 |
| Kick, 8 a second | 32 of 32 (11 extra) | 1 ms | 84 / 91 ms | |
| Snare, 12 a second | 48 of 48 (4 extra) | 5 ms | 88 / 96 ms | |
| Kick, 2 a second, over a bass line | 16 of 16 (17 extra) | 3 ms | 86 / 86 ms | low 0, mid 14, high 9 |

On 23 real songs (2,393 onsets of the drum stem, 3.8 a second): 90% found in the mix, 95%
on the stem, placed 9 ms, known 92 ms.

- **The confirmation delay is shared by all regions.** Only the low region uses the
  60 ms decay check. The mid and high regions are decided after one frame, but they travel
  in the same message, which is assembled three frames back. A hi-hat is known 81 to 91 ms
  after it sounds where 43 ms would do.
- Events are not published before confirmation and are never revised afterwards.
- **Regions are not kit pieces**, as the analyzer's own description says. On these signals
  a snare answered in all three regions and a kick in two or three; a kick under a held
  bass note lost its low region altogether (the low band did not "give back"). Only the
  hi-hat was isolated cleanly. Different animation per region is supportable as "low
  thump / mid crack / high tick"; calling them kick, snare and cymbal is not.
- "Extra" hits on kicks and toms are second answers to the same hit from another region a
  frame or two later, not false hits elsewhere.
- Hit strength saturates alone (0.997 for both quiet and strong) and separates over other
  instruments (0.76 against 0.99).

## C. Performance measurements

Machine: Ryzen 9 5950X, Radeon RX 9070 XT, Windows 11; CPU painting. On-screen runs on the
second display, 1920 x 1080 at 144 Hz, automatic 72 fps. The machine was in ordinary use.

**Method and tools** (all reproducible):

| Tool | Clock | What it measures |
|---|---|---|
| `scripts/diagnose_responsiveness.py` | audio time | Controlled signals with known onsets through the real analysis pass, one analyzer at a time, no presence model |
| `scripts/diagnose_responsiveness_stems.py` | audio time | Full mixes of multitrack songs; reference onsets from each instrument's own stem (an energy detector on the clean stem, not human labels) |
| `scripts/trace_responsiveness.py log` | wall clock | Recorded live sessions: what the engine stamped on each message |
| `scripts/trace_responsiveness.py replay` | wall clock | A session replayed into a real window, with hooks installed by the script on the state's intake and the layers' drawing |
| `scripts/measure_ring_motion.py` | simulated | Offscreen, frame by frame: share of each ring's pixels that change |

The two clock domains are kept apart. Audio-time figures contain no processing time;
wall-clock figures are differences of one monotonic clock on one machine.

**Rates and capture to publication** (three recorded live sessions, 49 s, 40 s and 12 min;
system audio capture):

| | Messages a second | Interval p50 / p95 | Gaps over 30 ms | Capture to publication, p50 / p95 |
|---|---|---|---|---|
| Feature frames | 49.8 to 50.0 | 20.0 / 20.0 ms | | 33 to 43 / 36 to 46 ms |
| Vocals | 49.8 to 50.0 | 20.0 / 20.0 ms | 1, 0, 68 | 34 to 43 / 36 to 46 ms |
| Guitar | 49.7 to 50.0 | 20.0 / 20.0 ms | 1, 0, 67 | 53 to 63 / 56 to 66 ms |
| Drums | 49.3 to 50.0 | 20.0 / 20.0 ms | 1, 0, 67 | 93 to 103 / 96 to 106 ms |
| Bass | 49.0 to 50.0 | 20.0 / 20.0 ms | 1, 0, 67 | 113 to 123 / 116 to 126 ms |
| Guitar recognition | 2.0 | 0.5 / 0.5 s | | 28 ms a judgement (36 ms p95) |

"Capture to publication" is the engine's `age_at_publication_ms`: from the capture of the
moment a message describes to its publication. It contains the analyzer's back-dating
(0, 20, 60, 80 ms), half the analysis window, the capture block and the processing
(analysis 3.3 to 4.4 ms a frame at the median, 5 to 11 ms p95, for everything). The
differences between instruments are exactly their back-dating. The gaps in the long
session are capture discontinuities (the same 67 or 68 for every instrument), not lost
messages. The time the sound takes to reach the loudspeakers is not in any of this.

**Publication to picture** (40 s replay of a recorded session into a window, each mode):

| | Radial | Bars |
|---|---|---|
| Frame interval p50 / p95 / p99 | 13.89 / 15.22 / 16.67 ms | 13.90 / 15.17 / 15.63 ms |
| Messages taken in per frame, p50 / p95 / max | 5 / 6 / 12 | 5 / 6 / 12 |
| Publication to state, p50 / p95 | 6.6 to 6.8 / 11.9 to 12.1 ms | 6.4 to 6.5 / 12.1 to 12.4 ms |
| Publication to drawn, p50 / p95 | 12.0 to 13.4 / 17.8 to 19.3 ms | 12.2 to 13.4 / 18.3 to 19.6 ms |
| Rows drawn per paint, p50 / max | 1 / 2 | 1 / 2 |
| Paints holding two or more attacks of one instrument | 0 | 0 |
| Rows refused by the state (out of order) | 0 | 0 |
| Messages dropped by the window's queue | 0 | 0 |
| Newest ink behind the hand: vocals / guitar / drums / bass, p50 | 9 / 29 / 69 / 89 ms | the same |

Rows that were first drawn by a whole-ring redraw (after a presence judgement changed) are
not in the "publication to drawn" figures: about 15 per instrument in each run.

**Estimated audio to visible response.** Capture to publication plus publication to drawn,
medians: vocals about 46 to 55 ms, guitar 66 to 75 ms, drums 106 to 116 ms, bass 126 to
136 ms; about 6 ms more at the 95th percentile of each part. This is a sum of two measured
parts from different runs, not one measurement, and it ends at the paint call: composition
and scan-out (up to a refresh or two at 144 Hz, 7 to 14 ms) and the audio output path are
not measured.

**What moves** (10 s of a recorded session, offscreen at 1080p, 72 frames a second):

| Picture | Share of its drawn pixels that change per frame, p50 / p95 |
|---|---|
| Vocals ring | 0.6% / 2.6% |
| Guitar ring | 0.8% / 0.8% |
| Bass ring | 0.7% / 1.4% |
| Drums ring | 0.0% / 3.1% |
| Whole-mix spectrum (alone) | 10.2% / 82.0% |

**Cost of the diagnostics.** The application contains no instrumentation from this phase:
the hooks exist only inside `trace_responsiveness.py` while it runs (a test checks that
nothing in `src/gmv` imports the diagnostic module). With the hooks the replay ran at the
same 13.9 ms frame interval as the earlier benchmark, 55 to 61% of a core; their own cost
and memory were not measured separately.

## D. Root causes, classified

| # | Issue | Class | Evidence |
|---|---|---|---|
| 1 | Rings show where activity was, not what is happening | **Renderer history behaviour** | `Sweep` keeps six seconds; under 1% of a ring changes per frame; nothing drawn ever moves again |
| 2 | Nothing animates between or after events (no decay in place, no oscillation) | **Renderer interpolation / smoothing** | Layers only draw rows as they arrive; there is no per-frame state. The `ring` envelope exists in the data but is drawn along the time axis |
| 3 | Bass: repeated same-pitch notes are not attacks | **Signal-processing limitation** | 1/12, 12/24, 24/36, 35/48 on controlled signals; 38% of stem onsets in mixes, 57% on the stem alone |
| 4 | Bass: same-note attacks placed 50 to 57 ms early | **Activity publication issue** | "Struck" is read from the newest 46 ms and stamped with the pitch window's 80 ms back-dating |
| 5 | Bass: energy known 105 to 125 ms late and nearly flat between notes | **Analysis window length** (for the energy only) | Level is the RMS of the 170 ms pitch window, eased 60 ms; falls back 3 to 23% |
| 6 | No level shows dynamics or a build | **Signal-processing limitation** | 25 dB growth: 0.81 to 0.97 (bass), 0.77 to 0.93 (vocals); quiet and strong alike alone. Reference is the instrument's own 4 s peak over 30 dB |
| 7 | Drums: mid and high hits wait for the low region's check | **Activity publication issue** | All regions known at 86 ms; only low needs the 60 ms |
| 8 | Vocals: phrase start stamped 100 ms late, phrase end known after 340 ms | **Confirmation delay** | Measured; by design of the confirmation |
| 9 | Presence follows an entrance late: guitar about 1 s when at all, vocals 0.7 s (2.6 s p95) | **Recognition / attribution delay** | Recognition report; stems here |
| 10 | Presence models hide part of the fast activity (bass 20 to 33% of active frames, vocals 12 to 60%, drums 11 to 48% in sessions) | **Recognition / attribution** (correctness unknown) | Sessions have no reference; on stem excerpts where drums and bass play throughout nothing was hidden |
| 11 | Drum regions are not kit pieces | **Signal-processing limitation** | Region table above |

Ruled out, with evidence:

- **Analysis cadence**: 50 a second, 20.0 ms spacing, for every instrument.
- **Event loss or coalescing**: none dropped, refused or merged; at most two rows and one
  attack per instrument per paint.
- **Frame scheduling**: one timer, every layer painted every frame, 13.9 ms interval,
  rows on the picture 12 to 13 ms after publication.
- **Timestamps in the renderer**: rows are placed by their own audio time; the display
  clock maps audio time through the smallest arrival offset seen.
- **Model or training-data fidelity as the cause of static rings**: the models only gate.

The four kinds of defect the task asked to keep apart:

1. *Analyzer cannot detect the event*: bass repeated notes (3); soft kicks under a bass
   note in the low region (11).
2. *Delayed confirmation*: drums' shared 60 ms (7); vocal phrase boundaries (8); bass
   pitch, inherently about 105 to 135 ms.
3. *Recognition suppresses a detected event*: (9), (10).
4. *Renderer does not animate a correctly detected event*: (1), (2) for all four
   instruments; this is the whole of the problem for guitar and most of it for vocals.

## E. Recommended plan for phase 2

Begin with the renderer and a shared live state; add the bass and drum analyzer changes
beside it. In this order:

**1. A live state per instrument in the visual state (shared).**
`VisualState` gains, per instrument, a small object fed by the same rows as the history
(no second analysis, one owner of the data: the `Track` stays the source for history; the
live state is a reduction of its newest rows). It holds what an animation needs *now*:

- an energy envelope with time-based attack and release;
- the recent attacks as a short bounded list of (time, strength, place, width), so two
  attacks between frames are both kept;
- the current pitch and its confidence, and for drums the last hit per region;
- the presence weight, eased so that a judgement arriving does not snap the picture;
- the audio time of its newest row.

It is sampled at paint time with the display's audio-time clock (`VisualState.now`), so
decay, ringing and oscillation are functions of elapsed audio time and identical at 60, 72
or 144 fps and in replay. Interpolation between 20 ms rows happens here. This is cheap:
a few numbers per instrument per frame. *Expected improvement: large. Confidence: high.
Complexity: moderate. Performance risk: low. Recognition risk: none.*

**2. Live radial renderers, one per instrument, on that state.**
A second paint path for polar stages (`paint_live`), fixed in place, redrawn every frame
from the live state; no `Sweep`, no hand. The six-second history stays in the timeline and
detail lanes, and the Bars strips stay as they are unless asked otherwise. Each instrument
keeps its own language and its own code:

- *Guitar*: the six strings as closed rings; the existing stroke placement (`centre`,
  `spread`), `ring` envelope and repetition, now oscillating in place and dying away.
  The analyzer is not touched.
- *Drums*: a pulse per hit from the attack list; low slower and heavier, mid sharp, high a
  light shimmer; overlapping freely.
- *Vocals*: strands opened by the level, bent by the pitch, a bloom at the phrase start.
- *Bass*: a ring whose radius and thickness follow the energy envelope and that pulses per
  attack; pitch shifts its shape slightly.

Drawing cost must be watched: four live rings redraw every frame where today almost
nothing is redrawn. Budget from the last benchmark: about 6 ms of a 13.9 ms frame is used.
*Expected improvement: this is the change the criticism asks for. Confidence: high.
Complexity: high (four pictures). Performance risk: moderate, measurable with the existing
benchmark. Recognition risk: none.* Guitar first (its data is ready and its look is the
reference), then drums, then bass and vocals.

**3. Bass analyzer: a fast energy path beside the pitch path** (needed for the bass ring
to pulse; can be developed in parallel with 2).

- Publish the low-band energy of the newest 46 ms frame as its own envelope, stamped at
  its own time rather than 80 ms back.
- Detect note onsets on that envelope (a rise against a short running minimum, so a note
  re-struck at the same level counts), independent of pitch confidence; keep the pitch
  path as it is for the note's height, arriving about 60 ms after the pulse.
- Stamp same-note attacks at their own frame.

The existing fields keep their meaning; the new ones are added to `InstrumentActivity`
(an additive schema change). Acceptance: the controlled signals of this phase (repeated
notes at 2 to 8 a second found, placed within 20 ms, known within 50 ms; the kick not
counted), and the stem measurement (38% found today). *Improvement: large for bass.
Confidence: high for energy and onsets on clean notes, moderate in dense mixes (the kick
shares the band). Complexity: moderate. Performance risk: negligible. Recognition risk:
none.*

**4. A dynamics measure with a longer memory** (shared helper, all four).
Beside each `level`, a strength measured against a reference that adapts over tens of
seconds instead of four (the approach of the global musical intensity), so that a quiet
passage is small and a build grows. Used by the live renderers for size; the existing
`level` stays for presence-like uses. *Improvement: needed for "stronger passage, more
movement". Confidence: moderate, needs tuning on recordings. Risk: low.*

**5. Drums: do not hold mid and high hits for the low region's check.**
Either per-region delay inside one message, or mid and high published at one frame and
low when confirmed. The confirmation itself stays. To be decided with a false-positive
measurement on stems before and after; if mid and high cannot leave early without changing
what is reported, leave it. *Improvement: 40 ms sooner for snare and cymbals. Confidence:
high that it is safe for mid and high (their rule does not use the check). Complexity:
low to moderate (message shape).*

**6. Presence gating, last and carefully.**
No threshold changes. Two things belong to the live path only: easing of the weight, and
showing fast activity as tentative for the first moments of an entrance instead of not at
all. Whether to go further needs a reference this phase did not have.

Not recommended now: raising the analysis rate, a faster or more frequent pitch estimate,
retraining a model, GPU drawing, more processes. None of them is where the measured
problem is.

## F. Unresolved

- Whether the presence models were right when they hid activity in the recorded sessions.
  There is no reference in a session; on stems the excerpts mostly had the instrument
  playing throughout, so entrances were few (bass 2, drums 4).
- Guitar recognition entrance delay was not measured again here; the figures are from the
  recognition report.
- Reference onsets on stems come from an energy detector on the clean stem. Soft legato
  notes are missed by it and bleed can add some; "found" percentages are against that
  reference, not against a transcription.
- The synthetic bass has a soft pluck and a steady tone; a real bass has a sharper attack.
  The stem figure (57% on the bass stem alone) is the better guide to real playing.
- Time from the paint call to light on the display, and from capture to loudspeaker, are
  not measured. The relation between what is heard and what is seen may differ from the
  figures here by the audio device's output latency.
- Nothing was measured at 2560 x 1440 or on the 180 Hz display.
- What a full-ring live redraw of four instruments costs is unknown until one exists.
- Whether the "level moved" share on stems (74% bass, 93% vocals) is inflated by chance
  rises within the 0.4 s it looks at was not controlled.
- The look and feel of any of this has not been judged; this phase measured, it did not
  design.

## Files added in this phase

`src/gmv/evaluation/responsiveness.py`, `scripts/diagnose_responsiveness.py`,
`scripts/diagnose_responsiveness_stems.py`, `scripts/trace_responsiveness.py`,
`scripts/measure_ring_motion.py`, `tests/test_responsiveness.py`, this document. No
existing source file was changed.

---

# Phase 2: live rings

Implemented from the plan above: items 1 (live state), 2 (live radial renderers, all four
instruments) and 3 (the bass fast path). Item 4 is covered only in part; items 5 and 6 are
not done. Status: implemented and covered by automated tests; **how it looks and feels
with music playing has not been judged by a person.** The pictures used during
development were automated offscreen frames.

## What changed

**Live state** (`gmv.visualizer.live.Live`, one per instrument in `VisualState.live`).
Fed by the same rows as the history, as they arrive. Holds the eased level and energy, the
ringing envelope, a bounded list of the recent attacks (time, strength, place, width, and
for drums the strength per region), the note and where it sits in the instrument's range,
and the presence weight (eased over 0.15 s). Once a frame it is advanced to the display's
audio time less the analyzer's own delay (vocals 0, guitar 20, drums 60, bass 80 ms), so
the newest row is "now" when it arrives and an attack's age starts near zero. Everything
that decays or oscillates is a function of that audio time: the same at any frame rate and
in replay. With no rows for a quarter of a second the instrument is taken to have stopped;
after a hole in the stream the old attacks are dropped.

**Live radial rings** (`Layer.paint_live`, one implementation per instrument). Radial rings
are now fixed objects redrawn every frame from the live state. No hand, no six-second turn.

| Instrument | Picture | Moves with |
|---|---|---|
| Guitar | Six strings as concentric rings | A stroke sets the strings it covered swinging (a standing wave in place) and flashes them; the swing follows the ringing envelope down, slowly under a held chord, quickly after a muted stroke; each stroke restarts it; a held chord keeps a small shimmer. Same stroke placement (`centre`, `spread`) as before. The guitar analyzer is untouched |
| Bass | One heavy ring | Thickness is the low register's energy now; every note onset pushes the ring outwards and brightens it and it falls back between notes; it flows faster the more energy there is; the note sets how far out it sits; paler when no note can be read |
| Vocals | Four strands round the ring | They open with the level, with a bloom where a phrase begins; the ribbon moves outwards with a higher note; the strands drift past one another while a voice sounds and close when it stops; dotted without a readable pitch |
| Drums | Impacts | `low`: a heavy ring that swells outwards and thins over half a second. `mid`: a burst of rays all the way round, gone in 0.2 s. `high`: a scatter of sparks at the outer edge, gone in 0.13 s. Any number overlap. Between hits only a faint circle |

Every ring's movement is scaled by the global musical intensity (0.55 at rest to 1.0), so
an intense passage moves more than a quiet one. An instrument judged absent is not drawn;
a tentative judgement is drawn pale and dashed, as before.

Unchanged: the strips of Spectrum Bars, the timeline, the detail lanes, the notes lane and
all activity history. Settings → Instrument layers → Radial rings → History brings the
six-second rings back (remembered).

**Bass: a fast path beside the pitch path** (`bass_pitch_v1`, additive). From the newest
46 ms frame alone, not back-dated: `energy` (the low register against its loudest of the
last half minute, over 36 dB, never measured against less than a fifth of the whole mix's
loudest) and `pulse` (a note onset: the low register rising 5 dB above the lowest it has
been in the last 120 ms, once per rise). `InstrumentActivity` gained `energy`, `pulse` and
`lead_s`; the existing fields keep their meaning and their values, and old sessions load
(the bass ring then pulses with the older attacks).

## Measured

Bass, controlled signals (analyzer alone), fast path against the older attack:

| Signal | Attacks before | Pulses now | Placed (p50 / p95) | Known (p50 / p95) | Energy falls back |
|---|---|---|---|---|---|
| Same note, 2 a second | 1 of 12 | 12 of 12 | 3 / 3 ms | 26 / 26 ms | 100% |
| Same note, 4 a second | 12 of 24 | 24 of 24 | 8 / 13 ms | 31 / 36 ms | 47% |
| Same note, 6 a second | 24 of 36 | 36 of 36 | 16 / 23 ms | 40 / 46 ms | 39% |
| Same note, 8 a second | 35 of 48 | 48 of 48 | 13 / 23 ms | 36 / 46 ms | 40% |
| Changing notes, 4 a second | 24 of 24 (17 extra) | 24 of 24 (0 extra) | 8 / 13 ms | 31 / 36 ms | 48% |
| One held tone, 4 s | 1 | 1 | | | energy 0.90 to 1.00 throughout |
| Same note, 2 a second, in a band | 1 of 16 (18 extra) | 16 of 16 (4 extra) | 3 / 8 ms | 26 / 31 ms | 37% |
| Same note, 4 a second, in a band | 9 of 32 (9 extra) | 32 of 32 (4 extra) | 13 / 18 ms | 36 / 41 ms | 33% |

On the 22 songs with stems: a pulse for 61% of the bass stem's onsets in the full mix
(attacks: 38%), known 26 ms after the onset (attacks: 119 ms), placed 2 ms. 46% of the
pulses are at a bass-stem onset (attacks: 25%); most of the rest are kick drums, which
share the band. So the bass ring also answers kicks. Where bass and kick play together
that reads as one pulse; it is not a clean bass signal.

What moves, per frame (offscreen, 8 s of a recorded session, one instrument in the picture
at a time; share of lit pixels that change):

| | Live rings, p50 / p95 | History rings, p50 / p95 |
|---|---|---|
| Vocals | 98% / 100% | 7% / 12% |
| Guitar | 69% / 98% | 3% / 4% |
| Bass | 7% / 12% | 1% / 3% |
| Drums | 99% / 100% | 1% / 3% |

(The whole-mix spectrum alone: 9% / 84%. The bass ring's share is small because most of a
filled ring is the same colour before and after it has moved; its edges all move.)

Drawing cost, replay of a recorded session, 1920 x 1080 on the second display (144 Hz,
automatic 72 fps), timeline shown, 36 s a run, each twice:

| | Frame rate | Paint p50 / p95 / p99, ms | Frames over 25 ms | CPU, % of a core |
|---|---|---|---|---|
| History rings | 72.0, 71.9 | 5.76 / 7.42 / 9.02 and 5.68 / 7.12 / 9.08 | 0, 3 | 54, 51 |
| Live rings | 71.8, 72.0 | 6.99 / 9.15 / 11.16 and 6.73 / 8.79 / 9.93 | 4, 1 | 66, 60 |

About 1.1 ms more per frame and a tenth of a core. The first version, with wide
antialiased curves, took 13 to 16 ms a frame and fell to 50 to 58 frames a second; the ring
curves are therefore drawn one device pixel wide and the fills without antialiasing under
antialiased edges. Not measured: 2560 x 1440, the 180 Hz display, the OpenGL canvas, live
capture.

## Not done, and limits

- **Not judged by eye.** Shapes, speeds and sizes are first choices; they need the
  owner's reaction with music playing.
- **Dynamics per instrument** (plan item 4) exists only for bass (`energy`). Guitar,
  vocals and drums still have self-referenced levels; their rings grow with the global
  intensity, not with their own loudness.
- The bass energy reference is what has been heard in the last half minute. A line that is
  quiet from the first note reads as full until something louder has been played; strong
  then quiet reads correctly (1.00 against about 0.35 for notes 23 dB down).
- **Drums still wait 86 ms in every region** (plan item 5); regions are still not kit
  pieces. A snare lights `low`, `mid` and `high` together.
- **Presence gating is unchanged** (plan item 6), apart from the eased weight. An entrance
  still appears as late as the model reports it.
- Sessions recorded before this phase have no fast bass fields; replayed, their bass ring
  pulses with the older attacks.
- Guitar repetition rate and regularity are not used by the live strings yet.
- Layer styles (guitar "strokes", vocals "strands only") apply to the history pictures;
  the live rings have one look each, except drums' "low and mid only".
- Vocal vibrato shows only as far as the eased pitch carries it.
- The accepted guitar capture was not replayed by eye; the guitar analysis is unchanged
  (its source and output fingerprints are pinned by a test), so the strokes are the same.

## Files (phase 2)

`src/gmv/visualizer/live.py` (new), `state.py`, `layers/__init__.py`, `layers/guitar.py`,
`layers/bass.py`, `layers/vocals.py`, `layers/drums.py`, `window.py`;
`src/gmv/instruments/__init__.py`, `bass.py`; `src/gmv/models.py`, `dsp.py`,
`docs/schemas/InstrumentActivity.json`; `src/gmv/app/window.py` (the setting);
`tests/test_live_rings.py` (new), `tests/test_unified_view.py` (three history tests now
select the history rings); the diagnostic scripts read the fast path and the live rings.

---

# Correctness pass: audio segments and the bass ring's clock

Two faults in the phase 2 implementation, found in review and corrected without changing
the look of the rings.

## Live states and audio segments

**Fault.** A capture stream is numbered in segments: the number changes at a gap or a
format change, and the engine starts its analysis afresh for the new segment. The live
states (`Live`) did not know about segments. They started afresh only when the timestamps
went back or jumped by more than half a second. When a segment ended at 45.00 s and the
next began at 45.02 s, attacks, bass pulses, the note, the envelopes and an open vocal
phrase of the old segment carried on into the new one. The visual state's derived ringing
envelope (fed to the live state) carried over too.

**Correction.**

- *Detection.* The visual state compares the `segment` of each feature frame, and of each
  activity row that states one, with the newest seen (`VisualState.segment`; see "Segment
  ordering" below). On a change every live state starts afresh
  (`Live.begin(segment)`), and `live_epoch` counts it. The audio-clock handling at a
  segment change is as before.
- *What starts afresh*, all of it in `Live.reset()`: recent attacks, bass pulses, level,
  energy and ringing envelopes, pitch, pitch position, the vocal phrase state, repetition
  rate, the analyzer's delay, the presence weight, and the bass ring's phase origin (below).
  Only the instrument's pitch range stays. The first row of the new segment then sets up
  the state as a first row always has (no glide from old values).
- *What is kept.* The activity history (`VisualState.activity`): every row of the old
  segment stays, so the timeline, the detail lanes, the Spectrum Bars strips and the
  history rings show it. The one difference in the history: its derived columns (the
  ringing envelope, the time and place of the last attack, the gap between attacks) start
  again at the first row of a new segment, as they already did after a hole. The rows
  themselves and their published values are unchanged.
- *Late rows of a finished segment.* The engine publishes each frame's activity rows right
  after the frame, and the analysis of a new segment starts from nothing, so after a
  frame of segment *n* no row of an earlier segment is published. Should one arrive
  anyway (out of order), it is refused, from the history as well as from the live state,
  the same way a row out of time order always has been: kept, it would block the current
  segment's next rows from the history. Rows of the old segment that arrived in order stay.
  A row of a *newer* segment than the newest frame is accepted and moves every live state
  into that segment; the frame that follows starts nothing a second time.
- *Restarted timeline.* When the timestamps go back, the existing full reset happens
  (history included). It also forgets the current segment, so a segment change in the same
  frame does not cause a second fresh start.
- *Compatibility.* No schema change: `InstrumentActivity.segment` and
  `FeatureFrame.segment` already existed. Every recorded session examined carries the
  segment on both. A row parsed from a log without the field is taken to belong to the
  current segment (`model_fields_set` tells a missing value from a stated 0), so it is
  never mistaken for a new or an old segment. Sessions where everything is segment 0 never
  start afresh. Messages from the analysis process keep their stated segment through the
  pipe.

**In practice** the engine's analyzers start again at a new segment and need a few frames
(bass 170 ms) before they report, so a live ring is absent for that moment and then
appears from the new segment's own rows.

## Segment ordering (follow-up)

**Fault.** The stream's segment was learned from feature frames only, while each live
state could also move itself to the segment of a row it was fed. A row of segment 2
arriving before the frame of segment 2 moved the bass live state on; a late row of segment
1 then still passed the visual state's check (which only knew segment 1) and moved the bass
state back; the frame of segment 2 then started everything afresh a second time.

**Contract now.**

1. *Which messages can tell of a newer segment:* feature frames and activity rows that
   state their segment. `VisualState.segment` is the newest segment seen in either; it
   only moves forward, except at a full reset. `VisualState._observe` is the one place it
   changes.
2. *A newer row before its frame* is that segment beginning. Every live state, all four
   instruments together, starts afresh in it (`live_epoch` grows once), the row is then
   taken in as the first of its instrument in that segment, and the frame that follows
   finds the stream already in its segment and starts nothing again.
3. *A late row of an older segment* is refused before it touches anything: not added to
   the history, not fed to a live state, no change to the segment, to the per-instrument
   segment of the history rows or to their derived columns, and the current segment's
   next rows are not blocked. This holds whatever its timestamp, even one newer than every
   row received.
4. *A late frame of an older segment* is refused the same way (it does not become the
   newest frame and does not reset anything) when it is less than `RESTART_S` (2 s) behind
   the newest audio time seen.
5. *A restarted timeline* is still recognised: a frame of the same segment that goes back
   in time, or a frame of an older segment 2 s or more behind the newest audio time seen
   (a replay played again from its start within the same session), causes the existing
   full reset, once. The engine numbers segments upwards for as long as a source runs and a
   new Start is a new session, which resets everything regardless; out-of-order delivery
   between frames and rows is at most a few milliseconds, far below 2 s.
6. *Rows without a segment* (logged before the field existed) belong to whatever segment
   is current and never move it. Rows of that kind received before any frame are kept;
   the first frame then gives those live states its segment without starting them afresh.
   Sessions in which everything is segment 0 never start afresh.
7. *Live states on their own:* `Live.feed` ignores a row of a segment older than its own
   and starts afresh for a newer one, so a live state cannot be taken back even if used
   without the visual state.
8. *History:* every row accepted in order stays, from every segment. Only refused rows
   are missing, and they are refused before reaching it.

**Limit.** A row of a restarted timeline that arrives before the frame announcing the
restart, and names an older segment, is refused; the restart is recognised at that frame
and later rows are taken in. At most a row per instrument is lost that way.

Tests: `tests/test_live_segments.py`, the five tests after "ordering".

## The bass ring's flow

**Fault.** The ring's travelling wave advanced by `live.dt * (0.5 + 3.0 * energy)` at each
paint, kept in the renderer. Its position therefore depended on how often the ring had been
drawn: hidden for a while, it resumed where it had stopped; at different frame rates the
sum came out slightly differently.

**Correction.** `Live.phase(base, per_energy)` gives the angle as a function of the rows and
the audio time alone: `base` times the audio time since the segment's first row, plus
`per_energy` times the integral of the energy over that time. The integral is accumulated
as rows arrive (each row's energy held until the next row, for at most a quarter of a
second, the same rule by which the state takes an instrument to have stopped) and
extrapolated the same way to the instrument's present at `advance`. The bass ring uses
`phase(0.5, 3.0)`: still 0.5 rad a second at rest and 3.5 at full energy. The renderer keeps
no state (`BassLayer.live_shape` is a pure function of the stage and the live state).

The rate now follows each row's reported energy rather than the eased energy used before.
The eased value trails the reported one by 15 ms when rising and 100 ms when falling; as an
integral the difference is a fraction of a radian over a note and not visible in the
automated frames compared.

**What is established by tests** (`tests/test_live_segments.py`):

- the phase at 2.0 s is the same at 60, 72, 120 and 144 frames a second (to 1e-9 rad) and
  equals the integral worked out independently;
- not advancing the state for 1.3 s of music (a hidden ring) gives the same phase and the
  same ring geometry; drawing twice changes nothing;
- the same rows give the same phase and geometry in two independent states;
- a new segment with continuous timestamps counts its phase from its own first row, exactly
  as a state that never saw the old segment;
- more energy flows faster (0.5, 2.0 and 3.5 rad a second at energy 0, 0.5 and 1); after
  the rows stop, the energy part stops after a quarter of a second and the resting part
  goes on.

**Limits.** At a segment boundary the phase starts again from zero, by design; it is not
seen because the bass ring is absent until the new segment's first rows (above). Phase is
computed in double precision from the segment's start: after many hours the angle is large
but its precision is still far below a visible fraction of a radian.

## Drum sparks

The 18 spark positions of a high-region hit were generated from the hit's seed every frame.
They are now generated once per seed (`drums.sparks`, a cache of at most 128 hits, read-only
arrays). The same generator and seed give the same positions; a test checks they are
identical to what was drawn before.

## Files

`src/gmv/visualizer/live.py`, `state.py`, `layers/bass.py`, `layers/drums.py`;
`tests/test_live_segments.py` (new); this document, `docs/ARCHITECTURE.md`.
