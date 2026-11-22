# Musical intensity and the colour of the spectrum

The main spectrum keeps its indigo-to-lavender in quiet passages and turns towards a dark
red as the music as a whole becomes louder and fuller. One signal drives it, for the whole
mix; no instrument is involved and no instrument colour changes.

Status: implemented and covered by automated tests. **How it looks while music plays has
not been judged by a person yet**; the pictures used during development were automated
renders, which show that the colours are applied, not that the response feels right.

## The signal

`gmv.visualizer.intensity.MusicalIntensity` turns the stream of feature frames into one
number from 0 (quiet passage) to 1 (very energetic). It lives in the visual state as
`VisualState.musical` (`.value` is the smoothed number, `.raw` the unsmoothed one), so any
visualizer mode can read it. It is worked out from fields every `FeatureFrame` already
carries (`loudness_dbfs`, `band_energy`, `rhythmic_density`, `silence`, `timestamp`):

- no second capture path and no second analysis pass; the engine and the message schema
  are unchanged;
- the same frames give the same values live and in replay, whenever they arrive;
- it does not depend on any instrument being recognised or any layer being shown;
- it costs about 3 microseconds a frame (50 frames a second).

Three parts are blended:

| Part | Share | What it is |
|---|---|---|
| Level | 65% | Loudness of the mix in decibels, averaged over about 0.4 s, placed on a scale that adapts to what has been heard |
| Weight | 25% | Share of the energy below 250 Hz: 0 at a tenth, 1 at six tenths |
| Activity | 10% | Onsets per second: 0 at one, 1 at six |

**The level scale.** Two references follow the music: a loud one (rises within a second
or two, falls over about a minute) and a quiet one (falls within a few seconds, rises over
minutes). The scale runs from 6 dB under the quiet reference up to the loud one, never
narrower than 18 dB and never wider than 30 dB. Consequences:

- The playback volume cancels out: the same music 18 dB quieter gives the same values.
- Differences within a song are what count. A quiet opening followed by the full band reads
  low, then high.
- A recording that is evenly loud from start to finish sits about a third of the way up the
  level scale. With a heavy low end and many onsets that comes to roughly 0.6 overall, a
  partial colouring; it is not left fully red for the whole track. Without contrast there
  is no telling from the level whether a passage is intense or the recording merely loud.
- For the first four seconds of music both references simply follow the level, so a
  fade-in is not taken for a quiet passage.

**Weight** does not depend on volume either. A full band with bass and kick drum has most
of its energy below 250 Hz; a voice, a lone guitar or a break does not. It is what tells a
soft opening from a heavy one before anything else has been heard. (A first attempt used
the share of energy *above* 500 Hz as "brightness". On the owner's recorded sessions that
was the wrong way round: a lone strummed guitar is nearly all above 500 Hz, the full band
mostly below.)

**Smoothing.** The level is averaged in decibels, not as energy, so one drum hit 20 dB up
for 60 ms moves the result by less than 0.08. The blend is then eased with a time constant
of 0.7 s when rising and 2.5 s when falling, applied by elapsed audio time, so the result
is the same at any frame rate. Silence eases it to zero with a time constant of 1.2 s.

## The colour

`intensity.tint(value)` maps the signal to how far the spectrum is coloured: nothing below
0.3, complete at 0.9, eased at both ends. Below 0.3 the bars are drawn in exactly the
colours they had before this existed.

| | Low frequencies | Middle | High |
|---|---|---|---|
| Neutral (`palette.SPECTRUM`) | `#5b62a8` | `#8f8fd6` | `#cfd0f2` |
| Most intense (`palette.INTENSE`) | `#74101e` | `#9a1828` | `#b52f40` |

Every bar moves the same share of the way between its neutral and its intense colour, in
24 steps: the spectrum changes colour as one thing. The geometry, the height of the bars
and their motion are not touched. The canvas eases the share it draws with over about a
tenth of a second, and lets it go when the stream stops.

- **Radial**: the bars round the core. The core, the glow and the rings of predicted beats
  keep their colours.
- **Spectrum Bars**: the bars, their peak caps and their reflection. The same rule and the
  same steps.

What is not coloured: the instrument layers and their lanes (vocals pink, guitar gold,
bass blue, drums orange), predicted-beat markers (lime), the timeline, the notes. The red
is darker and bluer than the drums' orange (its green channel is at least 80 lower), and
with the spectrum hidden a change of intensity changes no pixel of the picture; both are
tested.

**Cost.** Colours for a step are worked out once and kept (at most 25 sets of 48). In bar
mode each step keeps one pair of narrow pictures, three pixels of width a bar. Nothing is
blurred and no shader is involved; it is the same QPainter drawing, so it works the same
on the CPU and on the optional OpenGL canvas (which was not run again for this change).

## Setting

Settings → Visualizer → "Colour the spectrum by musical intensity", on by default and
remembered. It only changes the drawing: with it off the signal is still worked out and
the analysis is untouched.

## Measurements

Replay of a recorded session, 1920 x 1080 on the second display (144 Hz, automatic 72 fps),
all four layers and the timeline shown, 36 s a run, each run made twice. The machine was in
ordinary use.

| | Frame rate | Paint p50 / p95 / p99, ms |
|---|---|---|
| Radial, colouring off | 71.9, 71.8 | 5.75 / 7.38 / 9.42 and 5.61 / 7.03 / 9.06 |
| Radial, colouring on | 72.0, 71.9 | 5.76 / 7.27 / 8.60 and 5.71 / 7.49 / 9.12 |
| Bars, colouring off | 72.0, 71.9 | 4.79 / 6.41 / 8.31 and 4.80 / 6.64 / 8.17 |
| Bars, colouring on | 72.0, 72.0 | 4.90 / 6.87 / 9.34 and 4.78 / 6.59 / 8.78 |

The differences are smaller than those between two identical runs. Each run had one to
four frames longer than 25 ms, with the colouring on or off alike. Offscreen, a paint in
which the colour step changes takes about 0.3 to 0.9 ms longer than one in which it does
not. Not measured: 2560 x 1440, the 180 Hz display, the OpenGL canvas, live capture.

On four recorded sessions of the owner's music (level and band energies as logged), quiet
openings read 0.2 to 0.35, passages building up 0.45 to 0.6, the full band 0.8 to 0.95, a
quieter break inside a loud song about 0.7.

## Limits

- The look has not been validated by eye with music playing.
- The estimate was tuned on a handful of recorded sessions of one kind of music. Bass-heavy
  quiet music will read higher than it should; dense music with little low end lower.
- A track that starts at full intensity is partly coloured from the start but reaches the
  deepest red only once something quieter has been heard, or never if nothing is.
- After several minutes without a quiet passage the quiet reference has crept up and the
  colour eases back towards the middle.
- The references carry over from one track to the next; a new track that is much quieter
  is adjusted to within a few seconds, a louder one at once.

## Files

`src/gmv/visualizer/intensity.py` (signal, `tint`), `state.py` (`VisualState.musical`),
`palette.py` (`INTENSE`), `window.py` (`spectrum_colors`, `Canvas.heat`, `colors`,
`bar_pictures`, `VisualizerWindow.set_intensity_colour`), `src/gmv/app/window.py` (the
setting), `tests/test_intensity.py`, `scripts/benchmark_ui.py` (`--plain-spectrum`).
