# Instrument layers

**Status: first phase. Guitar is the one implemented layer; it is experimental.** The main
view is now a composition of instrument layers beside the whole-mix spectrum. Each layer is
drawn from its own analysis of one instrument. Bass, drums and vocals have a design profile
and no layer yet: nothing is drawn for an instrument without its own analysis source.

What has and has not been checked is in [Validation](#validation). The short version: the
analysis and drawing are tested on synthetic audio, measured on MoisesDB stems, and
corrected against a captured recording of the practical test case, whose fast opening strum
is now found and drawn as a repeating figure. With the full band playing, that recording
gives no stroke pattern, and whether the layer is convincing to watch is the owner's
judgement, not yet given.

## Design

### Two kinds of information, kept apart

| | Says | How fast | Message |
|---|---|---|---|
| Recognition model (`gmv.recognition`) | whether guitar is present | 10 s of context, twice a second, late | `InstrumentEstimate` |
| Fast instrument analyzer (`gmv.instruments`) | what is being articulated right now | every 20 ms frame, one frame late | `InstrumentActivity` |

Neither can do the other's job. The recognition model cannot see a single stroke. The fast
analyzer cannot tell a guitar from another panned, articulated instrument (measured below).
`gmv.instruments.combine` is the one place where they meet:

| Recognition says | Fast activity is shown |
|---|---|
| detected | fully |
| candidate | weakened (0.65) |
| absent | not at all, however busy the articulation |
| absent, but the model has heard less than its context of this music | tentatively (0.4), pale and dashed |
| nothing (no model, or no estimate for that moment yet) | tentatively |

The newest estimate is held for up to 1.5 s, because estimates always arrive late; what was
drawn under a held judgement is redrawn when the model's level changes. The tentative case
exists so that the first seconds of a song are not blank: the model's context is then
mostly silence and it says "absent" for 3 to 6 s (measured on the owner's plays of one
song: 6 s for `guitar_mn10_v1`, 3 s for `guitar_mn10_rv2`). No threshold of either model
was changed.

### Data contract

`InstrumentActivity` (schema in `docs/schemas/InstrumentActivity.json`): `instrument`,
`method`, `timestamp` (centre of the frame it describes, on the audio timeline), and

- `level`: how much is ringing, 0 to 1;
- `flux`: how sharply it is being articulated at this frame, 0 to 1;
- `attack`: the strength of an attack placed at this frame, else 0;
- `rate_hz`, `regularity`: whether the articulation repeats, and how fast;
- `centre`, `spread`: where in the instrument's band the articulation sat and how wide;
- `basis`: which signal the analyzer could use (guitar: `stereo` or `mono`).

It is published on the engine's EventBus like every other message, logged to
`advanced_analysis/instrument_activity.jsonl` and replayed with its original timestamps.

### Where it enters

Analyzers run inside the existing analysis pass (`Analyzer.process`), on the per-channel
FFT that pass already computes. There is no second capture path, no second FFT and no new
thread. `AnalysisConfig.instrument_analyzers` names them; the application always runs every
registered analyzer. An analyzer that would be expensive belongs in its own worker with a
bounded queue (as recognition has) and publishes the same message.

### Composition

`gmv.visualizer.layers`: a `Layer` per implemented profile. The canvas gives the active
layers equal horizontal bands of the main area and each layer paints only its own band:

- `paint_static`: label and background, in the rarely redrawn picture;
- `paint_history`: activity rows into a scrolling strip. Only rows that arrived since the
  last paint are drawn; the strip is drawn again from its start only when the recognition
  model's level changes, or style or intensity do;
- `paint_live`: the newest edge and a one-line reading;
- `paint_lane`: the same rows in the detailed lane.

A layer reads `VisualState.activity[key]` and `VisualState.attribution(key, times)`. It does
not read the whole-mix spectrum and layers do not read each other. The layers show the last
4 s (short, so that single strokes separate); the lanes show the chosen 10 to 30 s. Both
place rows by their audio timestamps.

The whole-mix spectrum stays: below the layers (default), behind them and dimmed, or
hidden. With no active layer the main view is exactly what it was.

### Sensitivity and intensity

- **Sensitivity** (per instrument, 0 to 1) scales the analyzer's thresholds. It changes
  what is reported and logged. It is applied to the running engine at the next frame.
  It does not touch the recognition model or its thresholds.
- **Intensity** (per layer) scales how strongly the layer is drawn. It exists only in the
  layer; `AnalysisConfig` has no such field.
- **Show / hide** hides the layer. Analysis continues, and the lane (which has its own
  switch) keeps showing the activity.
- **Style** selects between renderings of the same rows.

The settings drawer builds one block of these controls per layer from the layer registry.

### Adding an instrument

1. An analyzer in `gmv.instruments` (`key`, `method`, `delay_s`, `sensitivity`, `reset()`,
   `step(spectrum, quiet) -> Activity | None`), registered in `analyzers()`; or a worker
   that publishes `InstrumentActivity` for its key.
2. A `Layer` subclass with `paint_history` and `paint_lane`.
3. `renderer=` (and `styles=`) on its profile in `gmv.visualizer.layers.profiles()`.

Settings, composition, lane, logging and replay follow from the registries.

### Profiles

| Instrument | Character to aim for | What its analyzer has to provide | State |
|---|---|---|---|
| Guitar | Articulated, string-like: strings that are struck, ring and die away; strokes for picking and strumming, beamed when they repeat | Attacks, ringing level, repetition rate, where a stroke sat | implemented, experimental |
| Bass | Deep, heavy, flowing: one broad slow wave whose weight follows the note and whose height follows the pitch | Low-register note level and pitch, note changes | planned |
| Drums | Sharp, percussive, impact-driven: bursts that hit and are gone, placed by the part of the kit | Hit times and strengths per kit region | planned |
| Vocals | Organic, flowing, phrase-oriented: a ribbon that swells and bends with a phrase and closes between phrases | Voiced level and pitch contour, phrase boundaries | planned |

There is no generic fallback renderer, on purpose: a second instrument drawn with the first
one's picture in another colour is what this phase is meant to end.

## Guitar

### Analysis (`guitar_articulation_v1`)

**The finding that shaped it.** No spectral measure of a mono mix that was tried follows
the guitar better than it follows the drums. On six MoisesDB songs, plain and peak-based
spectral flux, restricted to any band between 100 Hz and 12.8 kHz, with or without a veto
on broadband hits, correlated 0.2 to 0.4 with the guitar stem's attacks and 0.4 to 0.7 with
the drum stem's; detectors built on the prominence or persistence of spectral peaks
rejected the drums and lost the guitar with them (correlation below 0.2). Taking the guitar
out of the mix left every one of them as busy as before.

**What works better: the stereo difference.** Rhythm guitars are usually panned to the
sides, often as two takes; kick, snare, bass and lead vocal are in the centre. Left minus
right removes what both channels share. On 30 MoisesDB songs, 200 Hz to 5 kHz, the guitar
stem holds a median 62% of the energy of the difference signal against 27% of the mono mix.

**What is computed**, per 20 ms frame, from the FFT the analysis pass already has:

1. the rise of the log spectrum of the difference signal between 200 Hz and 5 kHz. A
   first version subtracted half of the same rise in the sum signal, to take out what a
   centre hit leaks into the sides. It was dropped: a guitar that is partly in the middle
   rises in both signals at once, and the correction removed it (on the captured test
   recording 8.0 instead of 9.6 of 10 strokes a second; on stems, precision 0.69 instead
   of 0.84);
2. `flux` from that rise on a fixed scale (it is not normalised to recent activity, so a
   quiet stretch is not stretched into activity);
3. an `attack` where the flux has a local maximum that clears the recent lower quartile,
   rose sharply (a slow swell is not an attack), and is not a much weaker echo within
   0.4 s of a strong one;
4. `rate_hz` and `regularity` from the self-similarity of the last 1.28 s of flux, between
   2 and 17 per second, reported only when at least three attacks fell in that time;
5. `level` from the level of the difference signal against its recent peak;
6. `centre` and `spread` from where in the band the rise was.

**Mono fallback.** A recording with the difference signal more than 26 dB below the sum has
no stereo image to use. Then the mono band is used, counting only rises concentrated in a
few partials and vetoing 80 ms around a strong, flat rise (a broadband hit). `basis` says
`mono`, and the layer says so on screen.

**Cost.** 0.2 ms per frame at the median, 0.4 to 0.5 ms at the 95th percentile, inside the
analysis pass (hop 20 ms). One frame of delay.

### Rendering

Six strings on the audio timeline, low string lowest.

- A string is displaced by the *ring* envelope: it jumps to the flux at an articulation
  and dies away over 60 ms when little is ringing, up to half a second while a chord is
  held. A sustained chord is a long calm wave; silence is a straight line.
- Each attack is a stroke across the strings it covered: low or high by `centre`, few
  strings or all by `spread`, heavier and whiter the stronger it was. Each string's wave
  starts again at the stroke.
- Attacks that follow each other at the measured repetition rate (regularity at least 0.5,
  gap within 35% of the period) are joined by a beam, as beamed notes are.
- Absent by the recognition model: flat strings, no strokes. Tentative: pale and dashed.
- A line of text gives the reading: repeating and how fast, sustained, unconfirmed, not
  detected, mono source.

The second style, *Strokes*, draws one ribbon whose half-height is the ring envelope, with
the same strokes and beams.

The **Guitar lane** draws the same rows: the ring envelope as a line and every attack as a
tick (heavier when beamed), over the recognition model's filled band. The lane does not
apply the attribution weight, because the band underneath already shows the judgement.

The wave frequency of each string is a fixed look, like a string's gauge. Everything else
that changes on screen is a column of the activity history.

## Validation

### Automated (16 tests, `tests/test_instrument_layers.py`; 249 in the suite)

Synthetic audio: two takes of a chord with random phases on the two channels; noise bursts
with a low thump in the centre as drums.

- Ten strokes a second: at least 90% found, each within 50 ms of its stroke (median under
  25 ms), rate reported as 10 per second, regularity above 0.6.
- A held chord: one attack, still ringing 2 s later, no repetition rate.
- Centre drum hits alone: at most 15% become attacks. The same rhythm on a guitar, panned
  or centred: at least 85% found.
- Guitar on the sides with drums in the centre on a different grid: at least 85% of
  strokes found, at most 20% of the drum hits that fall between strokes.
- Sensitivity changes the count; silence reports nothing; the first frame is analysed
  within 60 ms of the start.
- `combine` and `attribution` for every case of the table above, on the audio timeline.
- Drawing: guitar activity appears in the main view; much less is drawn when nothing is
  articulated; the layer and the lane are handed the same attacks; two layers draw at once,
  each from its own rows, and hiding either leaves the other intact; intensity and style
  change the picture and not one value of the activity; the strip filled in live equals a
  full redraw (within 8%), and is redrawn when the model's judgement changes.
- Engine: activity for every frame, no dropped blocks, bounded queue, logged and parsed
  back; sensitivity set on the running engine takes effect; no analyzers, no messages.

### Measured on stems (MoisesDB, 40 songs with guitar and drums, 30 s each from 0:45)

`scripts/diagnose_instrument_activity.py`. The reference is the same analyzer run on the
guitar stem alone, not human annotation. Medians over songs:

| | Stereo, as recorded | Same songs summed to mono |
|---|---|---|
| Stem attacks also found in the full mix | 0.71 | 0.40 |
| Attacks in the mix that match a stem attack | 0.82 | 0.36 |
| Correlation of flux in the mix with flux of the stem | 0.70 | 0.25 |
| Same correlation with the guitar taken out of the mix | 0.22 | 0.03 |
| Attacks per minute with / without the guitar in the mix | 178 / 184 | 166 / 183 |

Read together: in a stereo mix the signal follows the guitar's own articulation
(0.70 against 0.22 when the guitar is removed; what remains is the rest
of the band playing in time with it), and about seven in ten of its attacks are found. **It
is not a guitar detector**: without the guitar the signal is as busy as with it, driven by
whatever else is panned. That is why nothing is shown while the recognition model says
guitar is absent. In mono the analysis is much weaker.

### Measured live (development machine, system loopback, 8-channel device)

A 5 s test pattern (a different plucked tone on each side, five per second) played on the
default output and captured by the running engine with the window drawing:

| | |
|---|---|
| Attacks found | 25 of 25 |
| Repetition rate reported | 5.0 per second |
| Activity published after the sound it describes | 54 ms median, 59 ms p95, 81 ms max |
| Newest drawn row behind the display's "now" | 34 ms median, 42 ms p95 |
| Guitar analysis per frame | 0.20 ms median, 0.48 ms p95 |
| Analysis pass in total per frame | 2.2 ms median, 8.3 ms p95 (hop 20 ms) |
| Paint | 2.5 ms mean |
| Dropped capture blocks, native packets, window messages | 0, 0, 0 |
| Capture queue high-water | 3 of 16 |

Times are on the capture clock. Output-device and monitor latency are not included.

Playing a real song from the owner's collection through the engine into the window (file
source at playing speed, both display placements and both styles): no dropped blocks,
paint 3.2 to 3.6 ms mean, guitar analysis 0.4 to 0.5 ms p95. The pictures showed strings,
strokes and beams where expected; that is a check that it runs and draws, not a judgement
of whether the strokes are the guitar's.

### Measured on the practical test case (a 60 s capture made by the owner)

The recording is streamed, not a file, so the owner captured its first minute with the
application's opt-in debug recording (system loopback, after the sound device; the audio
stays in the local session folder). The analyzer was run over it offline, through the same
analysis pass. The opening is a guitar alone, strummed ten times a second, for about 10 s;
then the full band enters.

| | Strum alone (10 s) | Full band (36 s) |
|---|---|---|
| Strokes found per second | 9.6 of 10 | 0.0 to 0.2 |
| Gaps between strokes within 25 ms of 100 ms | 97% | none |
| Frames with the repetition rate reported | 100%, at 10.0 per second | 0 to 5% |
| Frames regular enough for beams | 100% | 0 to 3% |
| Stereo basis | yes (difference 8 dB below the sum) | yes (1 to 5 dB below) |

Played through the engine into the window, the opening is drawn as evenly spaced strokes
joined by beams with the reading "repeating, about 10 strokes per second"; the same picture
from the presence state alone would be a filled band. That is the difference this phase
was asked to show.

With the full band the layer shows strings ringing and no strokes. In the capture the
difference signal then has no repeating articulation at any band resolution tried
(self-similarity under 0.4 between 2 and 17 per second), while the sum signal follows the
drum pattern. Either the guitars hold a sustained wall there, or their strokes are not
recoverable from this signal; the capture cannot say which. The layer does not invent a
pattern from the drums.

Before the correction described under Analysis, the owner's live plays of the same opening
gave 2 to 7.5 strokes a second.

### Not done

- Nobody has yet watched the corrected layer against real music and judged whether it is
  convincing. One recording was examined closely; others were only checked to run.
- Stems tell how the analyzer behaves on MoisesDB mixes. The owner's recordings, and
  streamed audio after any processing by the sound device, may be narrower in stereo. On
  the development machine the captured left and right channels of a two-tone test
  correlated 0.87, so the device mixes them to some degree.
- The drum rejection of the mono fallback was tested on synthetic bursts only.

## Limitations

- Fast activity is derived from the mixed audio. It is not an isolated guitar signal, and
  a stroke on screen may be another panned instrument articulating while guitar is present.
- Whether anything is shown depends on the recognition model, which finds roughly half to
  two thirds of guitar on the owner's reviewed recordings. Where it misses guitar, the
  layer is empty.
- Centre-panned guitar (a solo, a single acoustic guitar) is analysed by the weaker mono
  fallback or not separated from the centre at all.
- Attacks closer together than 55 ms, or faster than 17 per second, are merged.
- In a dense section the articulation can sit at a steady level with no peak that counts as
  an attack: the layer then shows ringing strings without strokes.
- Isolating the guitar itself needs source separation or a trained guitar-onset model.
  The six-source HTDemucs model measured earlier in this project needs 1.4 s of CPU per 2 s
  of audio and answers 1.4 to 3.4 s late, so it is not a live path on CPU; a GPU path that
  does not depend on one vendor was not investigated in this phase.

## Files

| Area | Files |
|---|---|
| Analyzer and combination rule | `src/gmv/instruments/__init__.py`, `guitar.py` |
| Message, configuration | `src/gmv/models.py`, `contract.py`, `schema_export.py`, `docs/schemas/` |
| Engine | `src/gmv/dsp.py`, `engine.py`, `session.py`, `evaluation/detect.py` |
| Visual state | `src/gmv/visualizer/state.py`, `lanes.py` |
| Layers | `src/gmv/visualizer/layers/__init__.py`, `guitar.py`, `window.py` |
| Settings | `src/gmv/app/controller.py`, `window.py` |
| Diagnostic | `scripts/diagnose_instrument_activity.py` |
| Tests | `tests/test_instrument_layers.py`, `tests/conftest.py` |
