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

## What has been validated, and how

| Kind of evidence | What it covers |
|---|---|
| Automated tests | Scoring, leader choice, band edges, segments, the rendered colour and strength of the glow (pixels of offscreen pictures), Expressive Mode's bounds and its leaving the state untouched, settings |
| Evaluation against stems | How often the leader is the truly loudest of four stems, on multitrack songs (below) |
| Replay benchmark on a display | 1920 x 1080 on the second display (144 Hz), recorded session replayed |
| Offscreen renders | 1920 x 1080 and 2560 x 1440, both modes, all four combinations, looked at by the developer agent as still images; and offscreen paint timing at 2560 x 1440 |
| The owner, with music playing | After trying the packaged build, the owner reported that both features look good. Which music, which mode, which combinations and which display were not recorded, so this is an informal approval, not a structured test |

Not done: any measurement on a 2560 x 1440 or 180 Hz display; live-capture performance
with these features; the OpenGL canvas; a listening test of whether the leader matches
what a listener hears as prominent.

## Instrument-reactive background

The radial view has a soft background glow whose strength follows the music (the frame's
dynamic intensity and onsets). Its colour used to be the spectrum's neutral indigo. With
this setting the colour is that of the instrument that is most prominent at the moment:
pink for vocals, gold for guitar, light blue for bass, orange for drums. The strength,
shape and falloff of the glow are what they were; only the colour is new. Spectrum Bars
keeps its neutral glow. Instrument layers and the spectrum's own colouring are not touched.

### What "prominent" means here

Every instrument gets a score that estimates one thing: **the chance that it is the
loudest of the four**. That is measurable against stems, which is why it was chosen. It is
not the same as what a listener would call the lead: a quiet lead vocal over a loud rhythm
section is, by this measure, bass or drums. Nothing here separates sources, and nothing
available to this project measures perceived prominence.

The instruments' own levels cannot serve: each analyzer measures its level against its own
recent peak, so 0.8 means "near this instrument's own loudest of the last seconds".

`gmv.visualizer.dominance` computes each score as a logistic function of five numbers the
application already has:

| Input | From |
|---|---|
| Its sustained activity | its level (bass: its energy), averaged over 0.5 s |
| Its transient activity | its attacks (bass: note pulses), each dying away over 0.5 s |
| Share of the mix's energy below 250 Hz | the band energies in every feature frame, averaged over 0.5 s |
| Share between 250 Hz and 4 kHz | the same |

The score is multiplied by the presence weight (what the recognition models let through:
1, 0.65, 0.4 tentative, 0) and by a gate that is zero while the instrument's own analyzer
reports nothing. Presence decides whether an instrument can count; it never makes one
lead by itself.

### Band edges are a setting

`band_energy` holds the energy between consecutive band edges, and the edges can be changed
in Advanced Settings. The two shares are therefore worked out from the edges in force, not
from positions in the list (`dominance.Bands`):

- A band wholly on one side of 250 Hz or 4 kHz counts there in full. With edges that
  include both cutoffs (the default does) this is exact, and with the default edges the
  shares are what the coefficients were fitted on.
- A band that lies across a cutoff is divided by how that band's power is spread in the
  same frame's display spectrum (48 narrower bands of the same power in the same units),
  each display band counted by how much of it, in octaves, lies on each side. This is an
  approximation to within those display bands, each about a fifth of an octave wide.
  Without a display spectrum such a band cannot be divided.
- No shares are given, nobody leads and the glow stays neutral when: the edges are not
  known; they do not reach down to 45 Hz and up to 8 kHz (the shares would not be shares
  of the same whole); a band lies across a cutoff and the frame has no display spectrum;
  or a frame has another number of bands than the edges say.

Where the edges come from: for live capture, the session's own settings; for a replay, the
recording's saved configuration (every recording examined has it). A recording without a
configuration has unknown edges and a neutral glow; no edges are assumed for it. A visual
state that is told nothing (tests, scripts) uses the analysis default. No message schema
changed.

The 45 Hz and 8 kHz limits are a stated assumption, not a measured one, and the
coefficients were not refitted or re-evaluated with other edges: with non-default edges
the inputs mean the same, but the accuracy figures below were only measured with the
default.

### Calibration and evaluation

All of this uses one multitrack collection, its song folders taken in sorted order and
numbered from 1. From each song a 45 s excerpt (35 s to 80 s) is used, and the first 10 s
of it are not scored. A song is left out when fewer than three of its stems cover the
excerpt. The reference is the loudest of the four stems over 0.4 s; a frame is a *clear
case* where that stem leads the next by 3 dB or more.

| Songs | Use | Usable |
|---|---|---|
| 1 to 60 | Fitting the coefficients (`scripts/gather_dominance_frames.py`, `scripts/fit_dominance.py`) | 59 |
| 61 to 100 | Development: looked at once while the hold and margin of the leader choice were set | 40 |
| 101 to 140 | Test: not used for any choice. Songs 101 and 102 were run once to check that the script worked; nothing was changed afterwards | 38 (126 and 132 left out) |

No song is in two sets. The coefficients evaluated are the shipped ones (`dominance.MODELS`).
In the fit, wins and losses weigh the same for every instrument, so a score is evidence
from the signals and not how often an instrument is the loudest in the collection. The fit
builds its inputs directly from analyzer rows with presence as the models' plain yes or
no; split in three with each third held out in turn, it named the loudest stem in 71% of
clear cases.

The evaluation (`scripts/evaluate_dominance.py`) runs the application's own path: every
message goes to a real `VisualState` in the engine's order, and the leader, scores and
presence weights are read from it (attribution with candidate and tentative states, the
100 ms refresh of the weights, segments, stale rows, hold and margin). Two things are
simulated: recognition estimates arrive without the model's 25 to 50 ms delay, and
messages arrive at their audio time.

**Test set (songs 101 to 140): 38 songs, 21.5 minutes, 64,600 frames, of which 24,658 are
clear cases and 39,942 are not.**

| | |
|---|---|
| Clear cases with no leader | 44 (coverage 99.8%) |
| Leader is the loudest stem, when there is a leader | 58.7% |
| Leader is the loudest stem, over all clear cases | 58.6% |
| Per song (35 songs with at least 5 s of clear cases): mean / median / quartiles / range | 56.7% / 56.9% / 40.1% to 71.3% / 8.6% to 95.1% |
| Chance | 25% |
| Always the commonest loudest (bass) | 35.8% |
| Always the instrument this method picks most (vocals) | 34.1% |
| Highest own level times presence weight (what was possible before) | 37.6% |
| Changes of leader | 10.4 a minute |

Confusion, clear cases (rows: the loudest stem; columns: the leader):

| Actual / Predicted | Vocals | Guitar | Bass | Drums | Nobody |
|---|---|---|---|---|---|
| Vocals | 6,479 | 322 | 1,083 | 510 | 25 |
| Guitar | 951 | 298 | 163 | 114 | 0 |
| Bass | 2,210 | 117 | 5,258 | 1,229 | 16 |
| Drums | 1,636 | 125 | 1,714 | 2,405 | 3 |

| Instrument | Clear cases (share) | Times selected | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Vocals | 8,419 (34.1%) | 11,276 | 0.575 | 0.770 | 0.658 |
| Guitar | 1,526 (6.2%) | 862 | 0.346 | 0.195 | 0.250 |
| Bass | 8,830 (35.8%) | 8,218 | 0.640 | 0.595 | 0.617 |
| Drums | 5,883 (23.9%) | 4,258 | 0.565 | 0.409 | 0.474 |

**Development set (songs 61 to 100)**, for comparison and not as independent evidence: 40
songs, 22.7 minutes, 30,973 clear cases; 69.0% over all clear cases, against 51.0% for
always answering bass and 43.2% for the highest own level; per song a median of 62.6%.
The set is half bass-led, which flatters the figure. An earlier report of about 70% was
this set, measured with presence weights built by the script itself; the figures here
replace it.

**What to take from it.** The leader is right clearly more often than any baseline, and
wrong in about four clear cases out of ten. It is uneven between songs. Six frames in ten
are not clear cases at all, and there no answer is right or wrong by this reference.

**Guitar** is the weak instrument and the aggregate hides it: recall 0.20 and precision
0.35 on the test set (0.27 and 0.24 on the development set). What the data shows:

- Its presence weight is not the cause in these excerpts: where the guitar stem is the
  loudest, the weight averages 0.99 and is zero in under 1% of cases. (The excerpts start
  35 s into a song and skip the recognition warm-up; at a guitar entrance the known delay
  of the recognition model still applies and is not measured here.)
- It is mostly taken for vocals (951 of 1,526 cases): guitar and voice both put the mix's
  energy between 250 Hz and 4 kHz, and that share is the main evidence either has.
- There is little to learn from: guitar is the loudest stem in 6.2% of clear cases, and
  only two test songs have as much as 5 s of them.
- "Loudest stem" may also be a poor stand-in for a guitar's prominence, which is often a
  riff under louder drums and voice.

The coefficients were not changed in response. Doing so on these songs would spend the
test set; better evidence for guitar needs more guitar-led material and probably an input
that tells a guitar from a voice.

### Choosing and colouring

- The instrument with the highest score takes the lead once it has been at least 0.08
  ahead of the current leader for 0.45 s. Two instruments that are nearly level do not
  trade the lead, and a single hit does not take it.
- Below a score of 0.2 nobody leads. In silence, and when an instrument's rows stop
  arriving, its activity dies away and the lead is given up after 0.45 s.
- `Dominance.mix` holds how much of each instrument's colour the glow has, eased towards
  the leader with a time constant of 0.35 s; what is left over is the neutral colour.

**One glow in one colour.** The canvas mixes the neutral colour and the instruments'
colours in those shares into a single colour (`palette.blend`: each channel made linear,
averaged, turned back, so a mixture is not darker than the colours it is between) and
draws one glow in it at the strength the neutral glow would have had. How far the glow
lies over the background is therefore the same whatever its colour, and a change of leader
moves the colour without dimming or brightening the picture. This is checked on rendered
pixels: for each pure colour and for a hand-over in twelve steps, the pixel equals the
background plus the same coverage of the mixed colour, to within two levels of 255.

The first version drew each colour's share as its own translucent glow, one over the
other. Shares that add up to the full strength do not add up to it when composited that
way: at a glow coverage of 0.3 split evenly between two colours the result is 0.2775, a
dip of 7.5% in the middle of every change of leader (by the compositing rule; it was not
measured in a picture before being replaced).

The shares are taken in twelve steps, so only so many colours occur; a mixed colour's
picture is made when first needed and the four most recent are kept. The beat glow is a
separate glow in its own colour and is not tinted. With the setting off, in Spectrum Bars,
and while no stream is running, the colour is the neutral one and the picture is the
earlier one exactly.

### Clock and lifecycle

The scores, the leader and the colour mix live in the visual state (`VisualState.dominance`)
and advance with the timestamps of the frames and rows themselves. The renderer only reads
them. So the colour is the same at any frame rate, goes on while the window is hidden, a
layer is switched off or another mode is shown, and comes out the same in replay. A new
audio segment starts it afresh, as does a restarted timeline and a change of band edges.

It has nothing to do with the global musical intensity (`VisualState.musical`), which
colours the spectrum from indigo to dark red and scales how far the live rings move.
Neither reads the other.

## Expressive Mode

The same music, events and timing, drawn larger and brighter. Unchanged by the correction
pass that revised the sections above.

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
- **Bass**: a thicker ring for the same energy, a larger and brighter swell at each note.
  The flow (its audio-time phase) is not touched.
- **Drums**: larger low rings, longer rays, larger sparks, all brighter. The regions stay
  what they are (low, mid, high: not kit pieces).

In Spectrum Bars the bars are lifted by the curve (peak caps and reflection follow), the
onset flash and the glow are stronger, and the instrument strips are drawn at the larger
scale.

### What it cannot change

- **Events and analysis.** A response is never passed to the engine, the analyzers or the
  visual state. Tests feed one stream, draw it both ways and compare the state: attacks,
  pulses, their times, the bass phase, the prominence scores, the leader and the history
  are identical.
- **Timing.** Speeds, lifetimes and start times are not part of a response.
- **Bounds.** Every value is a bounded multiplier (or an exponent below one). Bar heights
  stay within 0..1 and keep their order; every instrument shape is clipped to its own ring
  or strip; opacities are clamped.
- **Colours.** Instrument colours and the glow's colour are the same in both responses;
  the mode scales the glow's strength (in the rendered picture by its factor of 1.7, to
  within the tolerance of the test), not its colour.

Off is exactly the earlier picture: switching the mode on and off again gives the same
image, pixel for pixel, for the same state.

## Cost

**Scoring.** 6 microseconds per analysis frame, plus the presence weights (73 microseconds)
every fifth frame: about 21 on average, 50 frames a second.

**Replay benchmark on a display**: a recorded session, 1920 x 1080 on the second display
(144 Hz, automatic 72 fps), radial view, 30 s a run, the machine in ordinary use.

| | Frame rate | Paint p50 / p95, ms | Memory |
|---|---|---|---|
| Both off | 72.0 | 6.28 / 8.69 | 246 MiB |
| Reactive background | 72.0 | 6.20 / 8.49 | 293 MiB |
| Expressive Mode | 72.0 | 6.19 / 8.53 | 246 MiB |
| Both | 72.0 | 6.22 / 8.57 | 293 MiB |

The reactive background costs no measurable paint time at the median and 47 MiB: up to
four mixed-colour glow pictures of about 12 MiB each at this size. Making one takes 3.5 ms
(4.3 ms at most), which happened about once a second during changes of colour.

Spectrum Bars was measured before the glow was changed to a single colour and not again
(its glow is always the neutral one): 4.52 / 6.65 ms with both off, 4.90 / 7.32 ms with
Expressive Mode, 72 fps.

**Offscreen at 2560 x 1440** (no display involved; a paint into memory, 60 simulated frames
a second, radial): paint p50 / p95 9.05 / 12.94 ms with the reactive background off, 8.78 /
11.70 ms with it on. A mixed-colour picture is 21 MiB there (up to four kept, 84 MiB) and
takes 6.5 ms to make (8.3 ms at most). This says the cost at that size is of the same
order; it is not a measurement of frame rate on a 2560 x 1440 display.

## Limits

- The leader matches the loudest stem in about six clear cases out of ten on unseen songs,
  least reliably for guitar, and "loudest stem" is not "what a listener hears as the lead".
- One collection of multitrack songs was used for fitting and testing; other music may
  behave differently.
- Accuracy was measured with the default band edges only.
- Guitar's presence comes from the recognition model, which reports an entrance late and
  often tentatively; that delay is not in the evaluation.
- The multipliers of Expressive Mode are first choices; the owner's approval was informal.
- Layer styles and the history rings are scaled by `motion` only.

## Files

`src/gmv/visualizer/dominance.py`, `response.py`, `palette.py`, `state.py`, `window.py`,
`layers/__init__.py`, `layers/guitar.py`, `bass.py`, `vocals.py`, `drums.py`;
`src/gmv/app/window.py`, `controller.py`; `scripts/gather_dominance_frames.py`,
`fit_dominance.py`, `evaluate_dominance.py`, `benchmark_ui.py`; `tests/test_presentation.py`.
