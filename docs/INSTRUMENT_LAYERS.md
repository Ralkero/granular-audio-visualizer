# Instrument layers

**Status: two phases. Guitar (phase 1) is accepted by the owner. Drums, bass and vocals
(phase 2) are implemented and experimental; the owner has not yet judged them.** The main
view is a composition of four instrument layers beside the whole-mix spectrum. Each layer
is drawn from its own analysis of one instrument and has its own picture.

What has and has not been checked is in [Validation](#validation-phase-1-guitar) and
[Phase 2 validation](#phase-2-validation). The short version for phase 2: each analyzer is
tested on synthetic instruments and measured on MoisesDB stems with the instrument present,
removed and alone. With the activity model deciding what is shown, removing an instrument
leaves a small fraction of its picture. On their own the three analyzers are not instrument-specific
(they follow short broadband hits in the centre, whatever is low and periodic, and the
centre); that is stated with numbers below and not hidden.

## Design

### Two kinds of information, kept apart

| | Says | How fast | Message |
|---|---|---|---|
| Recognition model (`gmv.recognition`) | whether guitar is present | 10 s of context, twice a second, late | `InstrumentEstimate` |
| Fast instrument analyzer (`gmv.instruments`) | what is being articulated right now | every 20 ms frame, one frame late | `InstrumentActivity` |

For drums, bass and vocals the presence judgement comes from the frame-wise activity model
(`activity_v2`, in `FeatureFrame.instruments_active`) instead of the recognition model. It
judges every frame as it comes, so there is no warm-up and no tentative start; without that
model loaded, those three layers are drawn tentatively throughout.

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
- `basis`: which signal the analyzer could use (guitar: `stereo` or `mono`; drums and
  vocals: `centre` or `mono`).

Phase 2 added fields for what the common ones cannot say, rather than giving guitar's fields
other meanings. They are optional, so sessions logged before them still parse:

- `components`: drums only. The strength of a hit placed at this frame in each region, as
  `(("low", s), ("mid", s), ("high", s))`. Regions of the spectrum, not kit pieces.
- `pitch_midi`, `pitch_confidence`: bass and vocals. The note as a fractional MIDI number
  when there is one, else null; and how periodic or harmonic the sound was either way.
- `boundary`: vocals only. `"start"` where a phrase begins, `"end"` where it has ended.

For bass, `attack` is a new note or the same note struck again. For vocals it marks a
phrase start. `level` is what each layer's weight or opening is drawn from.

It is published on the engine's EventBus like every other message, logged to
`advanced_analysis/instrument_activity.jsonl` and replayed with its original timestamps.

### Where it enters

Analyzers run inside the existing analysis pass (`Analyzer.process`), on the per-channel
FFT that pass already computes. There is no second capture path and no new thread. One
analyzer needs more than that FFT: the bass keeps 170 ms of low-passed, thinned samples of
its own (about 500), because 46 ms cannot tell low notes apart. The pass also hands each
analyzer the activity model's judgement for its instrument in that frame; the vocal
analyzer uses it to end phrases. `AnalysisConfig.instrument_analyzers` names them; the application always runs every
registered analyzer. An analyzer that would be expensive belongs in its own worker with a
bounded queue (as recognition has) and publishes the same message.

### Composition

`gmv.visualizer.layers`: a `Layer` subclass per instrument. The canvas stacks the active
layers top to bottom (vocals, guitar, bass, drums) and gives each a band sized by the
weight of its profile: the six strings get the most room, the bass ribbon the least. With
the spectrum shown below them, the layers take 50% of the main area for one layer, 62% for
two, 72% for three and 80% for four. Each layer paints only its own band:

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

| Instrument | Picture | What its analyzer provides | State |
|---|---|---|---|
| Vocals | Strands that open with a phrase, bend with its pitch and close between phrases | Level of the centre in the voice's range, pitch contour where there is one, phrase starts and ends | experimental |
| Guitar | Strings that are struck, ring and die away; strokes, beamed when they repeat | Attacks, ringing level, repetition rate, where a stroke sat | accepted by the owner |
| Bass | One broad ribbon: its height is the note, its thickness the weight; it moves in steps and holds | Low-register pitch, weight of the periodic low sound, new and repeated notes | experimental |
| Drums | Marks that hit and are gone: a disc (low), a burst of rays (mid), a spark (high) | Hit times and strengths in three regions | experimental |

There is no generic renderer, on purpose: an instrument drawn with another one's picture in
a different colour is what these layers exist to end.

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

## Drums, bass and vocals (phase 2)

All three work on the centre of the stereo image where they can: per bin, |L+R|/2 less
|L-R|/2. A kick, a snare, a bass and a lead voice are in the middle and survive; what is
panned to one side cancels. Two different takes left and right (double-tracked guitars)
cancel only on average, so some of them remains.

### Drums (`drum_impacts_v1`)

A hit is a sudden gain that covers a whole region at once and is gone again. Three regions:
`low` 40 to 140 Hz and `mid` 180 Hz to 2.5 kHz from the centre, `high` 6 to 14 kHz from the
whole image (overheads are stereo). A gain is measured against the region's own recent peak,
with a floor tied to the whole mix so that a nearly empty region cannot hit.

- `mid`, `high`: the gain must be spread evenly over the region's bins. A chord or a note
  raises a few partials and is discounted.
- `low`: too few bins for that. A gain counts in full only with a click between 1.5 and
  5 kHz in the same frames, and the region must have given back a third of the gain 60 ms
  later. A kick does both; a bass note does neither. Hence three frames of delay, and a
  soft kick with no click is missed.

**Picture.** One mark per hit, nothing joined and nothing ringing: a disc with a ring at the
bottom for `low`, a burst of rays in the middle for `mid`, a spark at the top for `high`;
size and brightness are the strength. The lane has a tick per hit in the third of the lane
that belongs to its region. A style shows low and mid only.

### Bass (`bass_pitch_v1`)

The pitch is the period of the low-passed (180 Hz) signal, by autocorrelation over 170 ms,
between E1 and C3. A note is named when the low register is periodic enough and heavy
enough against the whole mix; a new note must hold for 60 ms. During a kick the register is
briefly aperiodic and the last note is held, not replaced. `attack` is a new note, or the
same note struck again (the low register of the shared spectrum clearly above the three
frames before). `level` is periodic low sound against the recent peak and the whole mix.

**Picture.** A broad ribbon. Height is the note, in steps; thickness is the weight; a new
or struck note is a bright edge across it. Where no note can be read it stays at the last
note and is drawn paler. The lane draws the note as a stepped line with a tick per attack.

### Vocals (`vocal_contour_v1`)

On the centre between 150 Hz and 4 kHz: the pitch (A2 to A5) whose harmonic series explains
the spectral peaks best, preferring one near the pitch before. Its confidence is the share
of all peak strength, up to the top of that series, that lies on the series: one voice
explains most of it, a dense mix does not. `level` is the centre's loudness between 300 Hz
and 3.4 kHz against the whole mix, in full when pitched and at a third when not.

A phrase is centre sound *while the activity model reports a voice in that frame*. In a
full band the centre is never empty, so without the model's judgement a phrase would never
end. There is no lyric or speech recognition, and clean and harsh vocals are not told
apart: harsh vocals simply have no pitch and are drawn from level alone.

**Picture.** Five strands that open with the level and close to nothing between phrases, so
a phrase is a leaf shape. The ribbon's height is the pitch contour, eased so it glides;
without a pitch it drifts to the middle and the strands are dotted. A small circle marks a
phrase start, a tick its end. The lane draws the level and the phrase boundaries.

## Phase 2 validation

### Automated (17 tests, `tests/test_instrument_layers_phase2.py`; 266 in the suite)

Synthetic instruments (`tests/synth_instruments.py`): kick, snare, hi-hat, bass notes, a
sung line with vibrato, a wide pad, and phase 1's two-take guitar.

- **Drums.** Kicks, snares and hi-hats are each found in their own region, on the audio
  timeline; a snare is never a low hit. Strummed guitar, bass notes, a held bass note, a
  sung phrase and a pad: no more than a fifth become low hits, and played alone fewer than
  half of the strums leave any mark (in a mix far fewer). Taking the kit out of a synthetic
  band removes at least 70% of the hit strength.
- **Bass.** E1, A1, D2 and G2 are read within half a semitone and each new note is an
  attack; higher notes sit higher. A held note stays one note for 3.4 s; a note repeated
  four times a second is struck again at least 9 times in 12. Kicks alone give no note and
  no attack. With the bass in a mix its notes are read through the kit; without it, less
  than a third as often.
- **Vocals.** A held note is continuous and pitched within three quarters of a semitone;
  two phrases give start, end, start, end; a line rising five semitones moves the contour
  up by that much. A guitar or a pad panned to one side is no phrase and no level. Removing
  the voice from a mix halves the level at least. With the activity model's judgement:
  no phrase while it says nobody sings, and a phrase per stretch in which it says so.
- **Guitar is unchanged.** Its source file is byte-identical to the accepted one apart from
  one unused parameter (checksum in the test), its output on a fixed signal has a recorded
  checksum, and it is the same whether or not the other analyzers run beside it.
- **Together.** Four layers are drawn at once by four different renderers, each from its
  own rows, with pictures that differ pairwise; bands are sized by weight; it works with
  the spectrum beside, behind or hidden, in bars and radial. Hiding or dimming one layer
  leaves the others' pictures and all activity values untouched. Each lane is handed the
  same rows as its layer. Settings build show, sensitivity, intensity and (where there is
  one) style for every layer from the registry. Older messages without the new fields parse.
- **Engine.** Four analyzers in one pass on a synthetic band: no dropped blocks, bounded
  queue, every message logged and back-dated by its analyzer's own delay.

### Measured on stems (MoisesDB, 29 songs with all three stems, 30 s each from 0:45)

`scripts/diagnose_layers.py`. Each run is reduced to what its layer would draw: drums, hit
strength per minute; bass and vocals, mean level. "Raw" is the analyzer alone; "shown"
multiplies each frame by the activity model's judgement, which is what the application
displays. Medians over songs.

| | Drums | Bass | Vocals |
|---|---|---|---|
| Raw, instrument removed / full mix | 0.55 | 0.73 | 0.48 |
| **Shown, instrument removed / full mix** | **0.12** | **0.05** | **0.00** |
| Shown, stem alone / full mix | 1.08 | 1.18 | 1.03 |
| Shown, stem with one competitor / full mix | 1.16 | 0.96 | 1.02 |

How closely the reading of the full mix follows the reading of the clean stem:

| | |
|---|---|
| Drum hits of the stem also found in the mix | 0.81 |
| Drum hits in the mix that match a stem hit | 0.77 |
| Bass: frames with a note on the stem / of those, a note in the mix / of those, the same note | 0.95 / 1.00 / 0.80 |
| Vocals: frames with a pitch on the stem / of those, a pitch in the mix / of those, the same pitch | 0.57 / 0.69 / 0.76 |
| Vocals: correlation of level in the mix with level on the stem | 0.69 |
| Bass: correlation of level in the mix with level on the stem | 0.64 |

Read together:

- **None of the three is instrument-specific on its own.** With the instrument's stem taken
  out, the raw output is still 0.55 (drums), 0.73 (bass) and 0.48 (vocals) of what
  it was. For drums the remainder is bass plucks, consonants and sibilants that are short
  and broadband in the centre; for bass, other periodic low sound; for vocals, whatever
  else is in the centre.
- **What makes them a drum, a bass and a vocal layer is the activity model.** As shown,
  they fall to 0.12, 0.05 and 0.00.
- **When the instrument is present, the reading follows it.** 81% of the drum stem's hits
  are found in the mix; the bass note read in the mix is the stem's note 80% of the
  time; the vocal pitch, where one is given, is the stem's 76% of the time.
- The reference is the same analyzer on the clean stem, not human annotation, and removing
  a stem from a multitrack is cleaner than any real "without".

### Measured on local recordings and live

Nine 20 s excerpts from the owner's collection (three dense band passages with vocals, two
instrumentals, two soundtrack pieces without a band, two quieter songs), analysed offline:

- Dense band: 2 to 4 low hits and 1.5 to 3 mid hits a second; a bass note read 99 to 100%
  of the time, between about D#1 and A#1; vocals pitched 41 to 47% of the time. The activity
  model reports vocals almost continuously there, so each 20 s was one long phrase.
- Soundtrack without a band: the activity model reports bass 4 to 7% and vocals 2 to 5% of
  the time, so those layers stay empty although the raw analyzers find notes and a centre.
- One instrumental: the model reports vocals 45% of the time and five phrases are drawn.
  That is the activity model's false alarm, shown as it is.
- All four analyzers together: 0.49 ms per frame at the median on stems; on these
  recordings 0.83 ms median, 1.15 ms at the 95th percentile.

Live (system loopback, the window drawing, all four analyzers, a 5 s test pattern):

| | |
|---|---|
| Instrument analysis per frame | 0.75 ms median, 3.9 ms p95 |
| Analysis pass in total per frame | 3.2 ms median, 10.6 ms p95 (hop 20 ms) |
| Published after the sound: vocals / guitar / drums / bass | 35 / 55 / 95 / 115 ms (median) |
| Newest drawn row behind the display's "now" | 34 ms median, 43 ms p95 |
| Paint, four layers | 3.1 ms mean; 6.6 ms at 2560 x 1440 from a file |
| Dropped capture blocks, native packets, window messages | 0, 0, 0 |
| Capture queue high-water | 4 of 16 |

The drums are 60 ms later than the guitar because a low hit is confirmed by its decay; the
bass 80 ms because its window is long. Each message is back-dated by its own delay, so the
layers stay aligned on the timeline; they simply appear at slightly different distances
from its newest edge.

### Not done (phase 2)

- Nobody has watched the three new layers against music and judged them. The pictures were
  rendered from real recordings to check that they draw and differ, no more.
- No listening comparison of hits, notes or phrases against what is heard.
- The activity model was used as it is. Its vocal judgement has false alarms on
  instrumentals and holds through gaps in dense music; improving that means retraining it,
  which this phase did not do.

### Limitations (phase 2)

- Drums: about half of the low and a third to a half of the mid hits of a stem are found in
  a full mix; a soft kick without a click is missed; a bass pluck or a hard consonant can
  be a mid hit while drums are playing. Regions are not kit pieces.
- Bass: follows the lowest periodic sound in the mix. While the model says "bass", a low
  guitar riff or a tuned kick can be drawn as bass notes. Nothing above C3; octave errors
  happen.
- Vocals: the centre is not only the voice. While the model says "vocals", a centred
  guitar solo, a snare or a synthesizer contributes to level and sometimes to pitch. Harsh
  vocals have no pitch. Phrases are only as good as the activity model's frames.
- On a mono recording nothing can be taken out of the centre; `basis` says `mono`.
- Better separation of these instruments from each other needs source separation or
  retrained models, neither of which is in this phase.

## Validation (phase 1, guitar)

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
| Analyzers and combination rule | `src/gmv/instruments/__init__.py`, `guitar.py`, `drums.py`, `bass.py`, `vocals.py`, `common.py` |
| Message, configuration | `src/gmv/models.py`, `contract.py`, `schema_export.py`, `docs/schemas/` |
| Engine | `src/gmv/dsp.py`, `engine.py`, `session.py`, `evaluation/detect.py` |
| Visual state | `src/gmv/visualizer/state.py`, `lanes.py` |
| Layers | `src/gmv/visualizer/layers/__init__.py`, `guitar.py`, `drums.py`, `bass.py`, `vocals.py`; `visualizer/window.py` |
| Settings | `src/gmv/app/controller.py`, `window.py` |
| Diagnostics | `scripts/diagnose_instrument_activity.py` (guitar), `scripts/diagnose_layers.py` (drums, bass, vocals) |
| Tests | `tests/test_instrument_layers.py`, `tests/test_instrument_layers_phase2.py`, `tests/synth_instruments.py`, `tests/conftest.py` |
