# Radiant: a renderer of its own, on the graphics card

Radiant is one of three ways the application draws the same detected music.

| Renderer | Draws the central view with | Geometry |
|---|---|---|
| Classic (radial) | QPainter, on the canvas | The original radial visualizer. Unchanged |
| Spectrum bars | QPainter, on the canvas | The original bars. Unchanged |
| **Radiant** | **GLSL shaders, on the graphics card** | **Its own. Nothing of Classic's** |

They share the detection core and its visual state (`VisualState`: frames, markers, live
instrument states, musical intensity, dominance, audio time). They share no drawing.

While the graphics card draws Radiant it draws **everything** in the central view:
atmosphere, smoke, formations, the spectrum, the four instruments, filaments, bolts, the
core, bloom. The canvas draws none of it: not the phase 1 spikes, rings or segments, not an
instrument layer. This page describes that renderer. The QPainter Radiant of phase 1
([RADIANT_THEME.md](RADIANT_THEME.md)) remains only as a reduced picture for machines
whose card cannot draw this one.

History: phase 2 put shader effects *behind* the phase 1 geometry. Phase 3, described
here, removed that layering: the geometry is gone from the GPU path and the shaders supply
their own.

## What has been checked, and how

| Kind of check | What it covers | What it does not |
|---|---|---|
| Tests without a graphics card (`tests/test_radiant_plasma.py`, 17 tests) | Who draws the central view (instrumented: no phase 1 routine runs, the canvas's area is clear); the composition; shader files; every number the shaders are given, from the live states; the audio-time spectrum; frame-rate independence; the 25 palette roles; the fall-back; settings | Any picture |
| Smoke tests (`tests/test_radiant_smoke.py`: 11 without the card, 11 on it via the same probe) | Which events count as beats and which do not; strength, the list of sixteen, expiry, Show beats, segments, replay from the start, rebuilding any moment; musical intensity; frame-rate independence at 60, 72, 90, 120 and 144 frames a second in numbers and in pixels; the wave, recovery, variety between regular beats, no jump at expiry, no smoke over the instruments, no clipping; colours; cost | Whether it looks good |
| Tests on the graphics card (same file, 16 tests, via `tests/plasma_gpu_probe.py`) | Compiling and linking; what each pass draws and where; the spectrum's shape; each instrument's structure, zone, colour and response; core detail and colour; bloom; saturation; error capture; set-up and tear-down six times over | Whether it looks good. Skipped by name, with the reason, where no OpenGL 3.3 context can be made |
| Stills and frame sequences (`scripts/render_radiant_plasma.py`) | The composed picture in seven palettes and eight musical states at 1920×1080 and 2560×1440; two frame sequences; looked at by the developer of this change | Real displays; judgement by the owner |
| Smoke sequences (`scripts/render_radiant_smoke.py`) | Thirteen synthetic sequences and six stretches of recorded sessions (regular, irregular and dense beats, a quiet passage, an intense passage into a quiet one), fed message by message; looked at by the developer of this change | Real displays; judgement by the owner |
| The application window, recorded | Screen recordings of the real window on the 1080p display while replaying recorded sessions (beats, Expressive Mode with Multicolor Energy, intense into quiet), and stills taken from them; looked at by the developer of this change | Judgement by the owner; the recordings keep about 55 of the 72 frames a second |
| Replay benchmark on a display | Frame rate, canvas time, time on the card, memory, at 1920×1080 on a 144 Hz display | 2560×1440 on a display |
| The owner, with music | **Not done at the time of writing.** | |

Whether Radiant now looks like a plasma system that visualizes music is for the owner to
say with music playing. Nothing here claims it.

## Ownership

The canvas is still the application's shell: it paints the header, the timeline, details
and notes, and it holds settings and state. For the central view it does one of four
things, recorded in `Canvas.path` at every paint:

| `Canvas.path` | Who draws the central view |
|---|---|
| `radiant-gpu` | The graphics card, all of it. The canvas calls `Radiant.present`, which works out the frame's numbers and **draws nothing**, and leaves its own area clear |
| `radiant-cpu` | The reduced QPainter Radiant (`Radiant.paint`), because the card cannot or the user switched effects off |
| `classic` | The canvas's own `draw_radial` and `draw_layers`, as ever |
| `bars` | The canvas's own `draw_bars` and `draw_layers`, as ever |

On the `radiant-gpu` path none of `Radiant.paint`, `Radiant.paint_static`,
`Canvas.draw_radial`, `Canvas.draw_layers`, `Canvas.draw_glow` or `Layer.paint_stage` is
called. A test wraps every one of them and counts: zero. The same test checks that every
pixel of the canvas inside the visualizer's area is fully transparent, so nothing of the
canvas can be over or under the card's picture there.

The widgets: `PlasmaView` (a `QOpenGLWidget`) fills the visualizer; the canvas lies over
it with a clear ground and draws the shell. Qt composes the two. Leaving Radiant hides
the view and the canvas is opaque again (tested by switching back and forth: Classic and
Spectrum bars come back pixel for pixel).

## The rendering contract

```
VisualState ──► Radiant.present ──► frame.values ──► uniforms ──► shaders
 (shared)        (no drawing)        (numbers only)
```

| Module (`gmv/visualizer/themes/radiant_gpu/`) | Holds |
|---|---|
| `scene.py` | The composition (where each structure is) and what each instrument is doing, from the live states. Numbers only |
| `frame.py` | The dictionary of numbers one frame's shaders are given. No Qt, no OpenGL |
| `resources.py` | Shader sources from files, `#include` resolved |
| `renderer.py` | OpenGL: programs, framebuffers, passes. Works in whatever context is current |
| `view.py` | The `QOpenGLWidget` |
| `offscreen.py` | A context without a window, for tests and stills |
| `shaders/` | The GLSL. No shader text is built in Python |

`frame.py`, `scene.py` and `renderer.py` import nothing from the QPainter layers or from
Classic (tested).

## The composition

Shares of the reach (half the smaller side of the visualizer's area):

| Structure | Where | Shader |
|---|---|---|
| Core | to 0.085 (0.13 at most) | `core.frag` |
| Spectrum formation | 0.20 to 0.44 | `formations.frag` |
| Waveform | 0.475 | `formations.frag` |
| Segments | 0.52 | `formations.frag` |
| Middle formation | 0.555 | `formations.frag` |
| Vocals | 0.60 to 0.68 | `vocals.glsl` |
| Guitar | 0.68 to 0.76 | `guitar.glsl` |
| Bass | 0.76 to 0.86 | `bass.glsl` |
| Drums | 0.86 to 0.94 | `drums.glsl` |
| Outer formation, tendrils | 0.965 and beyond | `formations.frag` |
| Plasma field (corona), points of light, hazes, glow | Hugging the outer formation; hazes everywhere | `plasma_field.frag` |
| Smoke | Outside the outer formation, to about twice the reach | `smoke.frag` |
| Standing filaments and event bolts | Between and beyond the formations | `filaments.frag` |

An instrument keeps its zone whether or not the others are shown: hiding one removes its
structure and moves nothing.

### The passes

1. **Scene**, at 0.7 of the view's size each way, into a 16-bit floating-point buffer:
   the plasma field, then smoke, formations, instruments, filaments and core added to it.
2. **Bloom, tight**: what is brighter than a threshold, at half the scene's size, blurred.
3. **Bloom, wide**: that again at half its size, blurred.
4. **Composite**, at full size: background + scene + both blooms, brought down to what a
   display shows and encoded as sRGB.

### Plasma field

Clouds are noise looked up through a warp that is itself noise, which folds and branches
them; veins are ridged noise; the field turns slowly, faster near the centre. It is a
corona hugging the outer formation, with dark voids. Intensity sets its brightness and how
far out it reaches; lows fold it and swell it; highs brighten the veins; an onset sends a
wave of light outwards through it (nothing is stretched). The clouds beyond it are the
smoke, below.

### Formations

- **Outer**: a broad shell (thickness about 0.03 to 0.07 of the reach), slow, clockwise,
  with ridged surface texture. Lows and the bass's energy swell and buckle it. Tendrils
  and finer branches leave it outwards and end within about 0.3 of the reach.
- **Middle**: thin, fast, fine-grained, anticlockwise; mids and highs.
- **Segments**: forty, each lit by its stretch of the spectrum, with a texture inside,
  ragged edges and light running from one to the next; a predicted beat pushes them out.
- **Waveform**: a thin ring displaced by the envelope of the last three seconds
  (0.35 × level + 0.65 × onset strength), newest at the top.

### The spectrum

There are no spikes. The spectrum is the **surface of the inner formation**: a plasma
annulus round the core. At each angle (lows at the top, highs at the bottom, mirrored) its
outer surface stands as high as that band is strong. Where a band is strong the plasma is
tall, dense, bright and full of fine filaments; where it is quiet there is only a low dim
floor.

- The 48 bands are the display spectrum, in its order. Between two bands the surface is
  the straight line between them; nothing is added.
- Tested on the card: lows light the top and leave the bottom dark, and the reverse; a
  rising spectrum gives light rising from top to bottom; one band alone lights its own
  place and nothing far from it; a stronger band stands higher (surface at 0.32, 0.37 and
  0.45 of the reach for bands at 0.25, 0.5 and 1.0) and is brighter; silence leaves a
  floor; along a circle through it the formation is one continuous body.
- The surface height uses the band to the power 0.8, which keeps the order of strengths.

### Instruments

Each is drawn from its own live state (`scene.instruments`), in its own zone and Radiant
colour, by its own shader. None uses a QPainter layer.

| Instrument | Picture | Driven by |
|---|---|---|
| **Vocals** | Four ribbons of plasma weaving past one another round the zone, with light running along them | Level: how far they open and how bright. Pitch: the bundle moves outwards with a higher note. A phrase start: a pulse of light runs from the top down both sides. No readable pitch, or the presence model unsure: paler |
| **Guitar** | Five thin live wires. A stroke flashes the ones it covered and sets them swinging as a standing wave, with electrical jitter | The ringing envelope (a held chord swings long, a muted stroke dies fast); the last stroke's time, place, width and strength; arcs jump between the wires while strumming repeats steadily (the analyzer's rate and regularity). Idle: faint, still wires |
| **Bass** | One thick annulus, the broadest instrument, with slow lobes travelling round it | Energy: thickness and light. Note: how far out it sits. The lobes move by the bass's running phase, the same audio-time integral the bass layer uses. Each reported note onset swells it and sends a pressure wave outwards through the plasma |
| **Drums** | Impacts only; a faint line between hits | Each reported hit, by region: **low** a broad shockwave leaving outwards; **mid** a burst of crooked sparks all round; **high** a scatter of fine points at the outer edge. Size and light by strength; lifetimes 0.5, 0.2 and 0.13 s; hits overlap |

The three drum regions are frequency regions of the kit's range, as the analyzer reports
them. They are not identified drums.

What reaches the shaders is scaled by the presence model's weight, as in the instrument
layers. A bass note onset comes only from the bass analyzer's own pulses; a drum hit does
not become a bass note.

### Filaments and bolts

A standing filament is a contour line of a noise field laid round the formations: such
lines wander, fork and close. A second, slower field decides which stretches are live, so
they fade in one place and reform in another. Three layers at different scales and
directions; highs and intensity bring more of them.

A bolt is one reported event: a whole-mix onset, a guitar attack, a bass note onset, a
drum hit, a vocal phrase start. It strikes across its instrument's zone along a crooked
path with a branch and is gone in about a third of a second. With no reported event there
is no bolt.

### Core

A small bright centre; a body shaded like a ball with cells and slow turbulence; a limb;
a corona in drifting layers; fine and broad rays. Size and reach follow level, intensity,
onset flash and beat pulse.

Two defects of phase 2 were corrected here and in the composite:

- **The flat white disc.** The centre is now small and capped, and the body carries
  texture. On the card at full intensity, none of the core's disc is white and its
  brightness varies across it.
- **Pastel colours.** Brightness is now brought down on the brightest channel and the
  others are scaled with it, so a colour keeps its hue and saturation however bright it
  is. On the card, hot-pink plasma at its brightest has a saturation of 0.88; a pink core
  at full intensity is pink to its middle (red 57%, green 8% of its light).

## Smoke

Round the outside of the composition there are clouds of smoke (`shaders/smoke.frag`, a
pass of its own). They drift and change on their own all the time, quiet passages
included, and every predicted beat pushes them about.

It is a **procedural field, not a fluid simulation**. Nothing is simulated and nothing is
kept from one frame to the next: each frame's smoke is worked out afresh from audio time,
the list of recent beats and how intense the music is. That is what makes it the same at
any frame rate, in replay, and when a moment is opened without playing up to it. It is
drawn to look as if sound were moving it; no physics is computed.

Before the smoke, the outer part of the plasma field stood in for it, and its only response
was one line that scaled the whole field's coordinates at an onset (not a beat). That line
is gone. The plasma field now only hugs the outer formation.

### The clouds

Density is built at three scales. Each scale turns round the centre at its own rate and in
its own direction, and is looked up through a warp that is itself a slowly changing noise
field:

| Scale | What it is | Turns at | Evolves at |
|---|---|---|---|
| Masses | Where there is smoke at all | 0.021 rad/s | slowly |
| Billows | The rounded lumps and curls | 0.034 rad/s, the other way | about twice as fast |
| Wisps | Fine texture that eats into thin smoke and frays edges | 0.058 rad/s | about four times as fast |

Because the scales slide past one another and the warps keep changing, a cloud does not
travel as a picture: it leans, opens, closes, thins in one place and gathers in another.
A second, slower field decides in which directions there is much smoke and in which
little, and those directions drift. Two layers at different depths are added, the far one
dimmer, smaller in scale and slower, which gives the smoke depth.

Measured on the card with no beat at all: after a quarter of a second the smoke is 98% like
itself; after one second 83%; after four seconds 59%; after twelve seconds 43%. Turned
back by whatever angle fits best, the picture four seconds later is still only 55% like
the first, so it is not one picture being rotated. The amount of change differs between
twelfths of the circle by a factor of more than three, and so does the amount of smoke.
About two fifths of the smoke's region is dark. Nothing loops: every scale has its own
rate, none a whole fraction of another, and the field is never reset.

Everything is looked up at a position in the plane, not at an angle, so there is no seam:
the step in brightness across the top of the circle (5 of 255) is like the steps elsewhere
(4 on average, up to 16).

The smoke starts at the outer formation and never reaches inside it, where the
instruments are, whether a beat pushes it or not: measured, not one pixel of smoke there,
with no beat and with sixteen beats in Expressive Mode in loud music.

### Intensity

How intense the music is as a whole (the slow whole-mix musical intensity, 0..1, which is
measured against the recording's own quiet and loud parts) makes the smoke more turbulent,
not just brighter. The warps reach further, the billows curl more against the masses and
the wisps fray the edges more, so loud passages look stirred and quiet ones calm. How fast
everything moves does not change: a speed that followed the music would make the picture
depend on its history, which a field worked out afresh each frame cannot have.

Measured on the card, quiet against loud music with no beat: fine structure (the mean
difference between neighbouring pixels) is 50% higher in loud music, mean brightness only
22% higher, and still two fifths of the region is dark, so it is not a fog. It is the same
clouds, more stirred (79% alike), not another picture. In quiet music the smoke still
changes by 11 (of 255) in a second with nothing struck.

Smoke close to the outer formation catches its light, more when the music is intense or
the lows are strong.

### Light

The smoke is lit from the centre. A second look at the density a little way towards the
centre says how much smoke the light had to pass. Thick smoke shades what is behind it
(Smoke shadow); thin smoke and the side facing the centre are lit (Smoke primary); the
thin fringe of a cloud catches the highlight (Smoke highlight). It is added to the picture,
so what is behind shows through.

### Beats

**Which events.** A beat is exactly what the beat tracker publishes as `beat_prediction`:
a prediction for a time a little ahead, which counts once audio time has reached it.
Nothing else is a beat here: not an onset, not a transient, not a drum hit, not a beat
candidate. The tracker publishes predictions only while it is switched on and locked, so
with it off there are none and the smoke only drifts. `radiant_geometry.beats` gathers
them; the tracker, its thresholds and the analysis were not touched, and the renderer has
no beat detection of its own.

**Show beats.** That setting already switched every beat-driven part of the picture (the
pulse, the push on the segments). It switches the smoke's beat response too. Off: the smoke
drifts and no beat disturbs it. The markers are not discarded; switching it back on brings
the recent beats back.

**Strength.** The tracker gives each prediction a confidence, which it caps at 0.7. That
is the beat's strength, scaled so that 0.7 is 1 and zero is 0.45: a predicted beat always
does something. Musical intensity is a separate thing (above); it is not multiplied into
the beat.

**The list.** The shader is told the beats of the last 2.6 seconds (`BEAT_LIFE`), newest
first, at most sixteen, each once (a prediction published twice counts once), and how long
that lifetime is. The list is rebuilt from the markers' timestamps every frame; a beat
older than the lifetime, or beyond the sixteenth, is simply not in it.

**The wave.** A beat is a pressure front leaving the centre (from 0.25 of the reach) at 3.2
reaches a second. It gets to the outer formation in about 0.22 s and is through the smoke
by about 0.5 s. Before it arrives nothing changes (measured: zero). Where the front is, and
for a while behind it, the smoke is moved. The smoke drawn at a point is looked up
somewhere else:

- **outwards** (expansion where it pushes, compression where the pushed smoke piles into
  the smoke beyond), strongest towards one side of the circle and in a pattern of lobes;
- **sideways**, one way here and the other way there;
- **in eddies**: a small-scale swirling warp that the waves switch on and leave behind
  them, so the smoke curls after the front has passed;
- **churned**: the wisps eat further into the clouds, so edges break up and clouds
  fragment.

The front also throws a brief light on the smoke it passes.

**No two beats alike.** Each beat has a number of its own, worked out from its timestamp
(to the millisecond) with a full integer mix, so that beats of a steady tempo do not get
numbers a constant step apart. From it come the side of the circle the beat hits hardest,
how broad or fine its lobes are, and its sideways pattern. Measured on the card, four beats
of a regular sequence half a second apart, each seen at the same age over the same smoke:
they move about as much as one another (16 to 19 of 255), but their change pictures are
only 55 to 76% alike, and how their change is spread round the circle only 24 to 86%.

**Measured, one beat of strength 0.9:** the smoke differs from its undisturbed self by 0 at
the beat and 0.1 s later (the front has not arrived), 8 (of 255) at a quarter of a second,
19 at 0.45 s, then 10 at 0.9 s, 5 at 1.6 s and 0.05 at 2.5 s. The change differs between
twelfths of the circle by a factor of more than five. Stretching the undisturbed picture
outwards by the best-fitting amount leaves 88% of the change unexplained, so it is not a
stretch. Mean brightness changes by under 4%, so it is not a flash. Fine structure rises by
a sixth while the wave passes and is back to rest by 2.5 s. A beat of the least strength
(confidence zero) changes the smoke half as much as one of full strength (10 against 21).

**Recovery.** A strong beat's wake lingers longer than a weak one's. Every wake dies away
smoothly and reaches exactly nothing at the lifetime, when the beat leaves the list, so
nothing jumps: the largest step in any pixel at that moment is 1 of 255 (rounding). A new
beat starting while an older wake goes on changes nothing before its front arrives (0).

**Several beats** add up: two beats disturb more than either alone (20 against 14 and 10),
and each still shows. The total displacement is bounded (0.30 of the reach) and so are the
churning and the eddies, so sixteen beats at once in loud music leave the smoke agitated
but present: brightest pixel 199, none white; in Expressive Mode 215, none white.

### Time, segments, pauses, replay

- The only clock is audio time, wrapped once a day so that there is no jump within a
  session; a 32-bit float still tells times 2 ms apart for four and a half hours and 8 ms
  apart at the end of a day.
- A stream with beats fed as it would arrive at 60, 72, 90, 120 and 144 frames a second,
  compared at the same audio times (with 3 to 7 beats in the air): not one pixel differs,
  in the smoke or in the whole picture (tested on the card).
- A state handed the recording up to a moment, without playing up to it, draws the same
  smoke as one that played through (tested: same 7 beats, identical pixels). The replay has
  no seek; this is what a seek would rely on.
- A new audio segment drops the old segment's beats. Playing a recording again from its
  start resets the state, and none of the first run's beats disturbs the second run's smoke
  (tested).
- If audio time stands still (paused, or before playback starts), so does the smoke: the
  same moment gives the same frame. Resuming continues from that audio time; nothing is
  caught up or replayed. This is intended: the owner chose a still picture before playback
  and while paused over smoke that drifts on a clock of its own.
- A beat whose time has not been reached does nothing; a stale one (beyond the lifetime)
  is not in the list.

### Colours and Expressive Mode

Three roles were added, under "Smoke" in the palette editor: **Smoke primary**, **Smoke
highlight**, **Smoke shadow**. The palette has twenty-five roles. They are saved, restored
and reset with the rest, each preset has its own three, and settings saved before get the
factory ones. Changing them changes no shape and no timing, and no other pass. Bright smoke
keeps its hue (hot pink at its brightest: saturation 0.91).

Expressive Mode scales how far a beat moves and churns the smoke (`impact`), how stirred
the music makes it (`plasma`) and how bright it is (`shine`). The beats are the same beats:
same count, same times, same strengths (tested). Measured, the same beat changes the smoke
about 1.6 times as much.

### Cost

The smoke is drawn in the scene's buffer at 0.7 scale; pixels inside the formations or
beyond the smoke's reach leave the shader at once. It adds no buffer and no texture.

**On the card without a window** (frames drawn into a buffer and finished, nothing read
back, the best of five runs of twenty; two runs of the probe agree within 0.1 ms). The
smoke pass is the difference between a frame with and without it.

| | Smoke pass, 1920×1080 | All passes, 1920×1080 | Smoke pass, 2560×1440 | All passes, 2560×1440 |
|---|---|---|---|---|
| No beat | 0.5 to 0.6 ms | 2.1 ms | 0.5 to 0.6 ms | 2.5 ms |
| One beat | 0.5 to 0.6 ms | 1.9 ms | 0.6 to 0.7 ms | 2.6 ms |
| Sixteen beats | 0.6 to 0.7 ms | 1.8 to 1.9 ms | 1.0 to 1.1 ms | 3.0 ms |
| Sixteen beats, Expressive, Multicolor Energy | 0.6 ms | 1.7 to 1.8 ms | 1.0 to 1.1 ms | 3.0 ms |

Sixteen beats at once add about half a millisecond at 2560×1440 and less at 1920×1080:
each beat is a few arithmetic operations and two small noise lookups per pixel, and the
eddy field is worked out only where a wake is.

**In the application** at 1920×1080 on the 144 Hz display, Ryzen 9 5950X and RX 9070 XT,
one 30 s replay each of the same recorded session (with beats), measured over its last
26 s. Classic and Spectrum bars from the same batch for reference. "On the card" is
Radiant's own timer query, sampled every 50 ms (about 416 samples a run); "frame
preparation" is the CPU time to work out a Radiant frame (scene and uniforms):

| | Frames a second | Canvas paint p50 / p95 | Frame interval p50 / p95 / p99 | On the card p50 / p95 | Frame preparation p50 / p95 | CPU, % of a core | Process memory |
|---|---|---|---|---|---|---|---|
| Classic | 71.9 | 7.23 / 10.21 ms | 13.9 / 15.4 / 16.0 ms | – | – | 69 | 293 MiB |
| Radiant, smoke only (Show beats off) | 71.9 | 1.70 / 2.47 ms | 13.9 / 15.5 / 16.1 ms | 4.4 / 5.5 ms | 0.72 / 1.18 ms | 62 | 416 MiB |
| Radiant, beats disturbing the smoke | 72.0 | 1.74 / 2.48 ms | 13.9 / 15.4 / 16.1 ms | 4.4 / 5.4 ms | 0.72 / 1.16 ms | 64 | 416 MiB |
| the same, Expressive Mode | 72.0 | 1.88 / 2.62 ms | 13.9 / 15.6 / 16.1 ms | 4.3 / 5.4 ms | 0.80 / 1.19 ms | 69 | 417 MiB |
| the same, Multicolor Energy | 71.9 | 1.79 / 2.55 ms | 13.9 / 15.4 / 16.1 ms | 4.3 / 5.4 ms | 0.73 / 1.14 ms | 66 | 417 MiB |
| the same, no timeline | 71.8 | 1.25 / 1.85 ms | 13.9 / 15.4 / 16.1 ms | 4.7 / 5.9 ms | 0.69 / 1.08 ms | 59 | 411 MiB |
| the same, drawn at every refresh | 138.8 | 1.80 / 2.45 ms | 7.1 / 9.9 / 12.2 ms | 4.0 / 4.8 ms | 0.75 / 1.15 ms | 122 | 417 MiB |
| Spectrum bars | 71.9 | 4.79 / 7.32 ms | 13.9 / 15.2 / 15.8 ms | – | – | 49 | 274 MiB |

No message was dropped by the window in the runs that report it. Frame pacing is as even as
Classic's. The renderer's buffers are 12.6 MiB at this size (22.4 MiB at 2560×1440);
video memory in use was not measured.

Against the runs before the smoke (same session and script: 71.4 frames a second, paint
1.93 / 2.68 ms, 4.8 to 5.3 ms on the card, 426 MiB) and against the first version of the
smoke (72.0, 2.08 / 2.86 ms, 5.5 ms, 432 MiB), nothing got measurably worse. The card time
in the application varies by about a millisecond from one batch to the next (the card's
clocks are not fixed at this low load), more than the smoke costs, so the figures above
are the better measure of the smoke itself. 2560×1440 was not run on a display: the only
display of that size is the main one, which these tests leave alone.

### Checking it

`tests/test_radiant_smoke.py` holds the smoke's tests; the figures above that were
measured on the card come from `tests/plasma_gpu_probe.py`, which those tests run once
(they are skipped where there is no OpenGL 3.3). `scripts/render_radiant_smoke.py` renders
sequences of the whole picture without a window: synthetic ones (idle, one beat, two,
regular, rapid, a strong beat in loud music, recovery, Expressive Mode, other palettes and
the presets), or a stretch of a recorded session (`--session`, `--at`, `--step`), fed
message by message as playback would. What it renders is a check, not a judgement: the
smoke has not yet been judged with music by the application's owner.

### Limits of the smoke

- It is not a simulation. Smoke does not carry momentum, does not pile up over many beats,
  and a wave does not reflect or interfere with another; overlapping beats simply add.
- A wave always starts at the centre and is round. Only the smoke's response is uneven.
- Eddies come from one shared field that the wakes switch on where they are, not one field
  per beat.
- The lighting is one extra look towards the centre, not a march through a volume.
- Musical intensity changes how stirred the smoke is, not how fast it moves.
- The reduced QPainter picture has no smoke at all.

## Colours

All twenty-five roles of the Radiant palette reach the shaders, one uniform each
(`c_<role>`), decoded from sRGB to linear light. Changing one role changes exactly one
uniform (tested for all 25).

| Role | Colours |
|---|---|
| Background base | The background |
| Atmosphere primary, secondary | The two hazes; primary is also the glow when no instrument leads |
| Core centre, halo, rays | The core |
| Spectrum body, highlights | The spectrum formation: its body, and its surface and filaments |
| Primary ring | The middle and outer formations and the tendrils |
| Secondary ring | The waveform |
| Segmented ring | The segments |
| Energy accents | Bolts of whole-mix onsets |
| Particle accents | The points of light |
| Plasma primary, secondary, highlights | The clouds; the veins, hottest ridges and stroke flashes |
| Filament core, glow | The standing filaments |
| Smoke primary, highlight, shadow | The smoke: its lit body, its bright edges and the light of a beat's wave, its shaded depths |
| Vocals, Guitar, Bass, Drums | Each instrument's structure and its bolts |

The palette editor, presets, saving and factory reset are unchanged. A colour change is in
the very next frame's numbers; it changes no geometry, timing or event (tested). The
canonical instrument colours used by Classic and Spectrum bars are not touched.

The leading instrument's colour tints the background glow and nothing else.

## Time and determinism

The shaders have audio time and numbers derived from the visual state. No clock of their
own, no frame counter, no random numbers.

- The same messages at 60, 72, 120 and 144 frames a second give, at the same audio time,
  exactly the same spectrum, waveform, segments, onsets, bolts, bass pulses and phase,
  drum hits and time (to 1e-6). The envelopes the live states ease between analysis rows
  agree to within 0.03 (they are stepped by elapsed audio time per paint, as in the
  instrument layers).
- The same numbers give identical pixels, for every pass and for the whole picture.
- **The spectrum is now eased in audio time for Radiant.** Classic's display spectrum is
  stepped once per paint. Radiant instead uses `VisualState.spectrum_at(now)`: the same
  easing stepped once per analysis frame by the audio time between frames, then fallen by
  the release for the audio time since the newest frame. It does not depend on how often
  anything is drawn (tested). Frames are only noted as they arrive and stepped when the
  heights are asked for, so a renderer that never asks pays nothing; after more than
  twelve seconds unasked it is built up again from the frames of that long. Classic and
  Spectrum bars keep their own display, unchanged.
- The glow's strength is taken from the scene, not from the canvas's per-paint easing.
- A new audio segment clears onsets, bolts, pulses and hits.
- Audio time is wrapped once a day for float precision (it was once an hour); the picture
  jumps once then.
- The smoke was checked further, at 90 frames a second as well and pixel by pixel on the
  card (see "Smoke").
- Before any audio arrives (the window open, nothing started) and while paused, audio time
  does not move, so the whole picture, smoke included, holds still. It moves again as soon
  as audio time does. This is the owner's choice; no second clock is to be added for an
  idle or paused screen.

## Expressive Mode

The response's gains scale the picture: `plasma` (field, formations, filaments), `bloom`,
`motion` (folding, buckling, ribbons, strands, lobes), `impact` (bolts, onset waves, stroke
flashes, bass swell, drum impacts), `shine`, `breath` (core), and the spectrum curve.
The events are the same events: same onsets, bolts, pulses, hits, at the same times
(tested).

## Requirements, fall-back, diagnostics

OpenGL 3.3 with floating-point framebuffers. Developed on an AMD Radeon RX 9070 XT, driver
26.11.2 (the view gets a 4.6 compatibility context; tests use 3.3 core). `#version 330`,
no extensions, nothing vendor-specific.

If Qt's OpenGL modules are missing, the platform has no OpenGL, a shader does not compile
or link, a framebuffer cannot be made, or a frame fails on the card: the view is hidden and
Radiant is drawn by the reduced QPainter picture. Settings → Diagnostics then shows
"Radiant renderer: CPU: reduced QPainter picture (no plasma): *reason*". With the card
drawing it shows "GPU: *card*, OpenGL *version*, GLSL *version* · card … ms, submission …
ms per frame · buffers … MiB". The two paths are never mixed in one frame.

Resources: programs compiled once per context; five buffers made once per size and reused;
everything released when the context goes; rebuilt if a new one comes. A context was set
up, drawn in at two sizes and torn down six times in a row in the tests without failure.

When the window closes it releases the card's resources itself, with the context current,
and lets the view go, instead of leaving that to the order in which Python and Qt destroy
things at exit. Before this was added, one run of the packaged self-check ended with an
access violation at exit after reporting everything as working. With it, the self-check
ended cleanly three times in a row, and a separate exercise (three windows one after
another in one process, each resized, effects switched off and on, theme switched away and
back, closed) ended cleanly five times in a row. One crash in one run is too few to say
the cause is certain.

## Cost

These are the figures of the renderer before the smoke was added; the smoke's own cost,
and the whole renderer measured again with it, are under "Smoke", "Cost", above.

Replay of one recorded session for 26 s at 1920×1080 on a 144 Hz display (automatic 72
frames a second), four instruments, timeline shown unless said, one run each.

| | Frames a second | Canvas paint p50 / p95 / p99 | Frame interval p95 / p99 | Frames over 25 ms | On the card | Process memory |
|---|---|---|---|---|---|---|
| Classic | 71.6 | 7.57 / 11.28 / 15.48 ms | 15.4 / 18.0 ms | 5 | – | 295 MiB |
| Radiant, reduced QPainter picture | 63.4 | 10.45 / 14.60 / 23.54 ms | 17.3 / 25.2 ms | 17 | – | 274 MiB |
| **Radiant, GPU** | 71.4 | 1.93 / 2.68 / 4.95 ms | 15.5 / 16.1 ms | 0 | 5.3 ms | 426 MiB |
| Radiant, GPU, Expressive Mode | 71.5 | 1.81 / 2.60 / 4.72 ms | 15.5 / 16.4 ms | 6 | 4.8 ms | 412 MiB |
| Radiant, GPU, Multicolor Energy | 72.0 | 1.87 / 2.63 / 4.48 ms | 15.4 / 16.1 ms | 0 | 4.9 ms | 413 MiB |
| Radiant, GPU, no timeline | 72.0 | 1.55 / 2.17 / 3.36 ms | 15.4 / 16.0 ms | 2 | 5.3 ms | 408 MiB |
| Spectrum bars | 72.0 | 5.42 / 8.05 / 11.27 ms | 15.3 / 15.9 ms | 0 | – | 274 MiB |

- **The canvas now paints the shell only**: about 1.9 ms a frame, against 7.6 ms for
  Classic. Of that, working out the frame for the card (`Radiant.present`) takes 0.75 ms
  at the median and 1.2 ms at p95.
- **The card takes about 5 ms a frame** (it draws more than in phase 2, which took about
  4 ms), and submitting the frame 0.6 to 1.0 ms of CPU.
- **Frame pacing is as even as Classic's** (interval p99 16 ms).
- No message was dropped by the window in the run that reports it.
- The process uses about 130 to 150 MiB more than a QPainter renderer: that is Qt composing
  the window through OpenGL. The renderer's own buffers are 12.6 MiB of video memory at
  this size. Video memory in use was not measured.

At 2560×1440 nothing was run on a display. Without a window, a frame took 10.9 ms on the
card including a forced finish and reading the picture back (6.5 ms at 1920×1080 by the
same measure, against about 5 ms by the timer in the application), with 22.4 MiB of
buffers. That is not a frame-rate measurement; the card's share at that size has to be
taken as roughly 8 to 9 ms until it is run on that display.

## Limits

- **Appearance is unjudged** by the owner.
- **The instrument structures are concentric.** Each has its own kind of picture, but they
  sit in rings one outside the other; the composition is still a set of rings round a core.
- **The spectrum formation is small** relative to the view (out to 0.44 of the reach).
- **The scene is at 0.7 scale**, so the finest lines are slightly soft.
- **Guitar arcs are faint**: steady strumming adds little visible beyond the strokes.
- **2560×1440 was not run on a display.**
- **Video memory in use was not measured**, only the renderer's own buffers.
- **Composition cost is not separated out**; it shows only in the frame rate.
- **Live-state envelopes** (levels, ringing) are eased per paint by elapsed audio time,
  not recomputed from audio time.
- **Other cards are untested.**
- **The reduced QPainter picture is a different picture**: it has none of the plasma,
  formations or GPU instrument structures.
