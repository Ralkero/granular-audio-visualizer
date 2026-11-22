# Visual presentation: instrument-reactive background and Expressive Mode

Two settings that change only how the picture is drawn. Neither reaches an analyzer, a
model, a threshold, a timestamp or the list of events: the same recording gives the same
attacks, pulses, pitches, presence judgements and segments whatever they are set to.

| Setting | Default | Where |
|---|---|---|
| Instrument-reactive background | on | Settings → Visualizer |
| Expressive Mode | off | Settings → Visualizer, and the "Expressive" button in the top bar |

Both take effect at once, are remembered with the other settings, and are independent of
each other. Settings saved before they existed load with these defaults.

Status: implemented, covered by automated tests, and looked at in automated offscreen
renders (1920 x 1080 and 2560 x 1440, both modes, all four combinations). **Not yet judged
by a person with music playing.**

## Instrument-reactive background

The radial view has a soft background glow whose strength follows the music (the frame's
dynamic intensity and onsets). Its colour used to be the spectrum's neutral indigo. With
this setting the colour is that of the instrument that is most prominent at the moment:
pink for vocals, gold for guitar, light blue for bass, orange for drums. The strength,
shape and falloff of the glow are what they were; only the colour is new. Spectrum Bars
keeps its neutral glow. Instrument layers and the spectrum's own colouring are not touched.

### A common scale of prominence

The instruments' own levels cannot be compared: each analyzer measures its level against
its own recent peak, so 0.8 means "near this instrument's own loudest of the last
seconds". On 59 multitrack songs, choosing the instrument with the highest level (times its
presence weight) named the truly loudest of the four stems in about 40% of the clear
cases, the same as always answering "bass".

So every instrument gets a score that estimates one and the same thing: **the chance that
it is the loudest of the four**. `gmv.visualizer.dominance` computes it, for each
instrument, as a logistic function of five numbers the application already has:

| Input | From |
|---|---|
| Its sustained activity | its level (bass: its energy), averaged over 0.5 s |
| Its transient activity | its attacks (bass: note pulses), each dying away over 0.5 s |
| Share of the mix's energy below 250 Hz | the band energies in every feature frame, averaged over 0.5 s |
| Share between 250 Hz and 4 kHz | the same |

**Calibration.** The coefficients (`dominance.MODELS`) were fitted on the first 60 songs of
a multitrack collection: full mixes through the analysis pass and the recognition models
as in the application, against which stem really was the loudest (by at least 3 dB, over
0.4 s). Every instrument was given the same chance beforehand, so a score is evidence from
the signals, not how often an instrument happens to be the loudest in that collection.
This is why the four scores are comparable: they are probabilities of the same event, each
fitted against the same reference.

The score is multiplied by the presence weight (what the recognition models let through:
1, 0.65, 0.4 tentative, 0) and by a gate that is zero while the instrument's own analyzer
reports nothing. Presence decides whether an instrument can count; it never makes one
lead by itself.

**Measured** with `scripts/evaluate_dominance.py` on 40 further songs that were not used
for the fit (22.7 minutes), running the shipped class with its hold and margin:

| | |
|---|---|
| Leader is the truly loudest stem, clear cases | 69.7% (chance 25%) |
| When the loudest is bass / vocals / drums / guitar | 83% / 62% / 56% / 27% |
| Share of time leading: bass / vocals / drums / guitar / nobody | 41% / 30% / 24% / 4% / 0.3% |
| Changes of leader | 9.3 a minute |

Guitar is the loudest stem in under 4% of the clear cases in that collection and is the
least reliably recognised as such. "Loudest stem" is also not the same as "what a listener
would call the lead": a quiet lead vocal over a loud rhythm section counts as drums or
bass here. The score is an estimate from mixed audio.

### Choosing and colouring

- The instrument with the highest score takes the lead once it has been at least 0.08
  ahead of the current leader for 0.45 s. Two instruments that are nearly level do not
  trade the lead, and a single hit does not take it.
- Below a score of 0.2 nobody leads. In silence, and when an instrument's rows stop
  arriving, its activity dies away and the lead is given up after 0.45 s.
- The colour never switches. `Dominance.mix` holds how much of each instrument's colour
  the glow has, eased towards the leader with a time constant of 0.35 s; what is left over
  is the neutral colour. The glow is drawn as that much of each colour's own glow picture,
  all at the strength the neutral glow would have had, so its total strength is unchanged.
- When the stream is not running, the neutral glow is drawn.

### Clock and lifecycle

The scores, the leader and the colour mix live in the visual state (`VisualState.dominance`)
and advance with the timestamps of the frames and rows themselves. The renderer only reads
them. So the colour is the same at any frame rate, goes on while the window is hidden, a
layer is switched off or another mode is shown, and comes out the same in replay. A new
audio segment starts it afresh (nobody leads a segment because of the one before), as does
a restarted timeline.

It has nothing to do with the global musical intensity (`VisualState.musical`), which
colours the spectrum from indigo to dark red and scales how far the live rings move. That
signal says how intense the music is as a whole; this one says which instrument stands
out. Neither reads the other.

### Cost

About 17 microseconds per analysis frame on average: the scoring is 4, and the presence
weights (66) are looked up every fifth frame, 100 ms apart.
Drawing: 0.3 ms more per frame at 1920 x 1080, and about 35 MiB for the instrument-coloured
glow pictures (made when first needed, full size: stretching half-size pictures every
frame cost 4 ms a frame and was dropped).

## Expressive Mode

The same music, events and timing, drawn larger and brighter.

### One response profile

`gmv.visualizer.response.Response` is the one place for presentation gains. The canvas
holds the current one (`Canvas.response`) and hands it to the layers with each stage;
renderers keep no gains of their own. `BASELINE` is all ones and reproduces the picture as
it was; `EXPRESSIVE` is this mode.

| Value | Baseline | Expressive | What it scales |
|---|---|---|---|
| `motion` | 1 | 1.5 | Displacement of strings and strands, thickness and swell of the bass ring, reach of impacts; in Spectrum Bars the height of each instrument's strip picture |
| `impact` | 1 | 1.5 | Extra size and brightness at an attack, a bass pulse, a phrase start, a drum hit |
| `shine` | 1 | 1.35 | Brightness and opacity of instrument shapes |
| `spectrum_curve` | 1 | 0.78 | Exponent on bar heights (0..1): lifts quiet and middle bars, leaves 0 and 1 where they are |
| `flash` | 1 | 1.5 | How strongly an onset whitens the bars |
| `breath` | 1 | 1.8 | How far the radial core swells with the level and the beat |
| `heat` | 1 | 1.25 | How far the intensity colouring goes for the same intensity |
| `glow` | 1 | 1.7 | Strength of the background glow and the beat glow |

Per instrument, in the live radial rings:

- **Vocals**: strands open further and curl more; a stronger bloom at a phrase start;
  brighter.
- **Guitar**: strings swing further; a struck string flashes more strongly; brighter.
  The picture itself is the accepted one.
- **Bass**: a thicker ring for the same energy, a larger and brighter swell at each note.
  The flow (its audio-time phase) is not touched.
- **Drums**: larger low rings, longer rays, larger sparks, all brighter. The regions stay
  what they are (low, mid, high: not kit pieces).

In Spectrum Bars the bars are lifted by the curve (peak caps and reflection follow), the
onset flash and the glow are stronger, and the instrument strips are drawn at the larger
scale (their cached pictures are redrawn when the mode is switched).

### What it cannot change

- **Events and analysis.** A response is never passed to the engine, the analyzers or the
  visual state. Tests feed one stream, draw it both ways and compare the state: attacks,
  pulses, their times, the bass phase, the prominence scores, the leader and the history
  are identical.
- **Timing.** Speeds, lifetimes and start times are not part of a response: how fast a
  string swings, how long an impact lasts, when a pulse begins and the bass ring's phase
  stay with the audio clock.
- **Bounds.** Every value is a bounded multiplier (or an exponent below one). Bar heights
  stay within 0..1 and keep their order; every instrument shape is clipped to its own ring
  or strip; opacities are clamped. At a layer's own highest intensity with the mode on,
  the bass ring is still inside its ring (tested).
- **Colours.** Instrument colours and the leader's colour are the same in both responses;
  the mode scales the glow's strength, not its colour.

Off is exactly the earlier picture: switching the mode on and off again gives the same
image, pixel for pixel, for the same state.

### Cost

Replay benchmark, 1920 x 1080 on the second display, 72 frames a second in every case:

| | Paint p50 / p95, ms | Memory |
|---|---|---|
| Radial, both off | 6.35 / 8.82 | 246 MiB |
| Radial, reactive background | 6.64 / 9.03 | 280 MiB |
| Radial, Expressive Mode | 6.08 / 8.86 | 246 MiB |
| Radial, both | 6.42 / 9.03 | 281 MiB |
| Bars, both off | 4.52 / 6.65 | 273 MiB |
| Bars, Expressive Mode | 4.90 / 7.32 | 272 MiB |
| Bars, both | 5.05 / 7.10 | 276 MiB |

Not measured at 2560 x 1440 on a display (rendered there offscreen only), with live
capture, or on the OpenGL canvas.

## Limits

- Neither feature has been judged by eye with music playing; the multipliers are first
  choices from automated renders.
- The leader is right about seven times in ten against the loudest stem, least for guitar.
- The fit used one collection of multitrack songs; other kinds of music may behave
  differently.
- Guitar's presence comes from the recognition model, which reports an entrance late and
  often tentatively; a guitar passage can therefore take a while to colour the glow.
- Layer styles and the history rings are scaled by `motion` only.

## Files

`src/gmv/visualizer/dominance.py`, `response.py`, `state.py`, `window.py`,
`layers/__init__.py`, `layers/guitar.py`, `bass.py`, `vocals.py`, `drums.py`;
`src/gmv/app/window.py`; `scripts/evaluate_dominance.py`, `scripts/benchmark_ui.py`;
`tests/test_presentation.py`.
