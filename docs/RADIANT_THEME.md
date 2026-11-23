# Radiant: a second radial theme, with a palette of its own

Phase 1 of the Radiant theme: its geometry, what moves it, its colour system and how it sits
in the application. It is a QPainter picture.

**This is no longer what Radiant normally looks like.** Where the graphics card has OpenGL
3.3, Radiant is drawn entirely by shaders, with a composition, a spectrum and instrument
structures of its own: see [RADIANT_PLASMA.md](RADIANT_PLASMA.md). None of the geometry
on this page is drawn then: not the spikes, the contour, the segments, the rings or the
instrument layers. This page describes the reduced picture shown when the card cannot
draw Radiant or the user switches effects off, and the palette, theme selection and
saving, which both share. The palette has since grown to twenty-five roles (five for
plasma and filaments, three for smoke, all drawn by the graphics card only); the
seventeen described here are unchanged. This reduced picture has no plasma and no smoke.

Radiant is a **radial theme**: another picture of the same visual state. The radial layout
now has two, chosen in Settings → Visualizer → Radial theme:

- **Classic** (the default): the radial view exactly as it was, drawn by the canvas.
- **Radiant**: concentric luminous structures round a core of light.

Spectrum bars is a layout, not a radial theme, and nothing here touches it. The chosen
radial theme is kept while the bars are shown.

## What has been checked, and how

| Kind of check | What it covers | What it does not |
|---|---|---|
| Automated tests (`tests/test_radiant.py`, 50 tests) | Geometry and its bounds, the spectrum mapping, every event path, frame-rate independence of the scene, the palette model, that every role reaches the picture, isolation from Classic, Spectrum bars and the canonical colours, the glow, settings | Whether it looks good |
| Offscreen renders (`scripts/render_radiant.py`) | That the picture is composed as intended in several palettes at 1920×1080 and 2560×1440; looked at by the developer of this change, not by the owner | Motion; real displays |
| Replay benchmark on a display | Paint time, frame rate and memory at 1920×1080 on a 144 Hz display | 2560×1440 on a display (offscreen paint time only) |
| The owner, with music | **Not done at the time of writing.** | |

The theme is a first prototype. It is not claimed to be visually finished.

## Structure

| Module | Holds |
|---|---|
| `gmv/visualizer/themes/__init__.py` | The list of radial themes and the default |
| `themes/radiant_geometry.py` | Where every structure is and what moves it. Numbers only: no Qt, no colours |
| `themes/radiant_palette.py` | The palette: roles, presets, validation, saving. Plain `#rrggbb` strings, no Qt |
| `themes/radiant.py` | The drawing. No colour literals (see "Light" below) and no state between frames beyond cached pictures |

The canvas asks `Canvas.themed()` whether a theme draws the radial view. Classic is "no":
the canvas draws it as before, through unchanged code. Radiant is handed the canvas and
reads what it already holds: `display.values` and `peaks`, `state.frames`, `state.markers`,
`state.musical`, `state.dominance`, `state.live`, `canvas.now`, `canvas.response`. It
subscribes to nothing and keeps no clock. Another theme would be another module with
`paint_static`, `paint`, `neutral`, `beat` and `instrument`.

Geometry and palette do not know each other: `radiant_geometry` returns radii and angles,
the palette returns colours, and only `radiant.py` puts the two together. A later shader
renderer can take both as they are.

## Geometry

Everything is concentric round the middle of the radial area. Radii are shares of the
`reach` (half the smaller side of the area). `zone` is the radius of the inner composition:
0.56 of the reach while instruments are drawn, 0.9 when none is.

| Structure | Where | Moved by |
|---|---|---|
| Core centre and its boundary | 0.15 to at most 0.21 of the zone | Level, onset flash, predicted beat |
| Halo | 0.34 to 0.78 of the zone | Musical intensity (sustained), onset flash |
| Rays (28, turning slowly) | Out to 0.72 to 1.3 of the zone | Intensity, onset flash; the turn is audio time |
| Spike ring | Baseline at 0.33 of the zone, tips to 0.73 | The display spectrum |
| Contour | 0.815 of the zone, up to 0.065 either side | Envelope of the last three seconds |
| Segmented ring (40 segments) | 0.915 of the zone | Display spectrum by region, onset flash, predicted beat |
| Primary ring, inner | The zone's edge | Musical intensity and onset flash light it |
| Instruments | A band each between the inner primary ring and 0.955 of the reach | Each instrument's own layer |
| Primary ring, outer, with a ring of ticks | 0.985 of the reach | As the inner one |
| Atmosphere: two hazes, seven faint arcs, 150 points of light | The whole view | The points drift by audio time |

The order above is tested for windows from 480 to 1440 pixels high and for four, two, one
and no instrument layers: no structure can reach into the next, in Expressive Mode either.

### Core

`geometry.core` gives the radius of the core, of the halo and of the rays from four numbers:
the level (mean of the last 0.1 s of analysis frames), the intensity (mean of the last
0.3 s), the onset flash and the beat pulse. Sustained intensity keeps the halo and rays up;
an onset throws them out for a moment. In silence or with no stream all four are zero and
the core is at rest, still drawn.

### Spikes

The display spectrum is the same 48 values Spectrum bars draws, in the same order: lows at
the top, highs at the bottom, down the right side, mirrored on the left. Each band has
three spikes. The first is the band's own value. The other two lie on the straight line to
the next band, shortened to 55% and 78% of it: that is the fine tooth pattern, and it adds
nothing that was not measured (tested: no spike is above that line). A spike also reaches
inwards by a fifth of its length. Tips of taller spikes get a larger point of light, and
peaks are marked on their own band's spike.

### Contour: what signal it is

The contour is **not** the spectrum again. Each analysis frame of the last three seconds
(`state.frames`) is one point: the newest at the top, older ones further down both sides,
so the shape flows downwards as the music goes on. How far a point stands off the circle
is that frame's envelope, `0.35 × level + 0.65 × onset strength`, to one side or the other
by turns. The alternation is what makes it look like a waveform. It is the envelope the
analysis published, not the audio waveform: the visualizer never has audio samples.

### Segments, streaks and beats

- **Segments**: each is lit by the mean of the display spectrum over the stretch it stands
  for, plus the onset flash. The ring is complete at rest with every segment dimly lit.
- **Predicted beats** push the segmented ring out a little and add to its light, only when
  the beat tracker is on, has predicted a beat, that beat's time has been reached, and
  "Show beats" is on. With the tracker off the ring is the same ring without the push.
- **Streaks**: a whole-mix onset marker gives two to six streaks from the spike tips towards
  the rim and a widening circle. A guitar attack, a bass note onset, a drum hit and the
  start of a vocal phrase give one to three streaks across that instrument's own band,
  scaled by how much of the instrument the presence model lets through. More, longer and
  wider for a stronger event; they fade over 0.36 to 0.45 s. At most 64 at once.
- **Where a streak points** is fixed by the event's timestamp (a seeded choice, cached), so
  it does not move between frames or between two replays.
- Nothing is struck that the analysis did not report: no timers, no invented beats.

## Time

Every moving quantity is a function of the visual state and one audio time, `canvas.now`:

- flash and pulse are computed from the markers' ages, not accumulated per frame;
- level and intensity are means over the analysis frames at or before `now`;
- the contour is placed by each frame's age; the rays and the points of light turn by `now`;
- streaks are placed by the age of their event.

Radiant keeps no animation state of its own. The scene at a given audio time is the same
whether the display has been stepping at 60, 72, 120 or 144 frames a second (tested by
feeding the same messages at those four rates and comparing every number of the scene).
Two inputs it shares with Classic are eased per frame by the canvas, as before: the
spectrum's bar heights and the intensity colouring. They were left as they are.

When a new audio segment starts, `VisualState.segment_start` records its time and markers
before it no longer flash or streak; the live instrument states already start afresh.
Switching theme, layout or colours changes nothing in the state (tested).

## The palette

Seventeen roles, each with a colour of its own. No role's colour is derived from another's.

| Group | Roles |
|---|---|
| Background and atmosphere | Background base · Atmosphere primary · Atmosphere secondary |
| Central core | Core centre · Core halo · Core rays |
| Spectrum | Spectrum body · Spectrum highlights |
| Concentric structures | Primary ring · Secondary ring (the contour) · Segmented ring · Energy accents (streaks, onset circles, beat glow) · Particle accents |
| Instruments | Vocals · Guitar · Bass · Drums |

Settings → Visualizer → Radial theme → **Radiant palette** (shown while Radiant is the
radial theme) has, per role, a swatch that opens Qt's colour dialog and a field for a
`#rrggbb` code. Any RGB colour is accepted. The dialog is Qt's own rather than the Windows
one because it has a hex field; the picture follows it while it is open, and Cancel puts
the colours back. A change is drawn from the next frame, with the stream running.

**Presets** are starting points: Electric Blue (the factory palette), Plasma Magenta, Aurora
Green and Multicolor Energy. Choosing one sets all seventeen colours; changing any colour
afterwards makes the palette "Custom colours" and changes that colour only. "Restore
Radiant defaults" puts Electric Blue back and touches nothing else.

**Saving**: the theme and all seventeen colours are saved with the other settings. On
loading, a missing or invalid colour gets its factory colour, role by role; settings saved
before this get Classic and the factory palette; nothing in saved data can stop the
application starting. The saved colours are what counts: a preset name is only kept while
the colours still are that preset's. Named user palettes and import or export are not
implemented.

**Isolation**: the instrument roles start at the canonical instrument colours and are
Radiant's own copies. A layer is told its colour for the stage being drawn
(`Stage.colour`, kept in `Layer.tone`); without one it uses its own. So Classic, Spectrum
bars, the timeline and the detail lanes always use the canonical colours, and
`palette.INSTRUMENTS` is never written to (tested, with rendered pictures of Classic and
of the bars compared before and after). The instrument buttons in the top bar take the
Radiant colour while Radiant is on screen, and the canonical one otherwise.

**Light**: the renderer contains no colour of its own. The one fixed thing is white as
light: a lit thing is its role's colour moved towards white (a flash on a ring, a bright
segment). Mixing two colours is done as light adds up (linear light), so a gradient
between unlike hues does not pass through a dark or grey middle (tested for three unlike
pairs). The halo, the rays and the spectrum's glow are added to what is under them rather
than laid over it.

### Background glow

The instrument-reactive glow works as before, with the palette's colours: the leader's
Radiant colour, mixed with the neutral one in the eased shares, as one colour at the
unchanged strength. The neutral colour is Atmosphere primary and the beat glow is Energy
accents. Which instrument leads is decided without any colour (tested: scores and leader
are identical whatever the palette).

## Expressive Mode

Radiant reads the same `Response` as everything else and has no switches of its own.
Expressive Mode lifts the spikes (spectrum curve), lets the core swell further (breath),
widens the contour (motion), lengthens streaks (impact), brightens rings, segments and
core (shine, flash) and strengthens the glow (glow). The events are the same events: the
same number of streaks in the same directions (tested). Every enlarged shape is clipped to
its own lane. No response parameter was added.

## Other settings in Radiant

- **Instrument layers**: show or hide, intensity, sensitivity and style work as before.
  A hidden layer's band goes to the others; with none shown the inner composition takes
  the circle.
- **Radial rings: live or history**: both work; the history rings are drawn in the Radiant
  colours.
- **Whole-mix spectrum**: "Hidden" hides the spike ring. "Dimmed behind" has no meaning in
  Radiant and is ignored there.
- **Colour the spectrum by musical intensity**: in Radiant, intensity moves the spikes'
  body towards the Spectrum highlights colour. The classic dark red is not used.

## Cost

Replay of one recorded session for 26 s at 1920×1080 on a 144 Hz display (automatic 72
frames a second), timeline shown, four instrument layers, one run each:

| | Frames a second | Paint p50 / p95 / p99 | Frames over 25 ms | Memory |
|---|---|---|---|---|
| Classic | 71.9 | 6.72 / 9.69 / 12.45 ms | 1 | 295 MiB |
| Radiant | 71.7 | 10.08 / 12.80 / 20.62 ms | 6 | 272 MiB |
| Radiant, Expressive Mode | 71.6 | 9.96 / 13.28 / 21.10 ms | 7 | 271 MiB |
| Radiant, Multicolor Energy | 71.7 | 9.72 / 12.83 / 20.42 ms | 6 | 272 MiB |
| Radiant, history rings | 71.9 | 8.46 / 11.40 / 19.55 ms | 1 | 278 MiB |
| Spectrum bars | 72.0 | 4.92 / 7.06 / 9.27 ms | 0 | 271 MiB |

Radiant costs about 3.4 ms a frame more than Classic and holds the frame rate, with less
room to spare: a frame at 72 a second has 13.9 ms, and Radiant's slowest one in a hundred
takes about 20 ms, so a few frames in a thousand are late. The palette makes no difference
to the cost. At 2560×1440 only an offscreen paint was timed (a made-up state, no display):
about 13.9 ms for Radiant against 10.3 to 10.9 ms for Classic. That is not a frame-rate
measurement on a 1440p display.

How it is kept down:

- The atmosphere, the soft light under every ring, the primary rings themselves and the
  tick ring never change, so they are part of the still picture the canvas already caches.
  It is redrawn when the size, the layers shown or the palette change.
- Halo, core, rays and the spectrum glow are small cached pictures, at most ten kept.
- Spikes are opaque thin lines. Measured alternatives: filled spikes as one polygon cost
  about 20 ms a frame, a gradient along the lines 6 ms, translucent lines 2.5 to 5 ms,
  opaque lines about 1 ms.
- Large circles and the contour are one device pixel wide: wide antialiased pens along
  curves that long cost milliseconds.
- Blended colours are cached (4096 entries); streak directions are cached (512 events).

Changing a colour redraws the still picture once (a few milliseconds) and rebuilds the
small pictures that used it.

## Limits

- **Not a shader.** No procedural plasma, turbulence or real bloom. Light is cached
  gradients and additive drawing.
- **Thin lines.** For cost, spikes and the contour are fine lines and the large circles are
  hairlines over a soft static light. The bold glowing bands of the reference pictures
  need the shader phase.
- **Instrument pictures are the existing ones** in new bands and colours, with streaks
  added. They were not redrawn for this theme.
- **Static light does not pulse.** The soft light under the rings is part of the cached
  picture, so only the crisp line over it brightens with the music.
- **The OpenGL canvas** (`GMV_RENDERER=opengl`) was not tested with this theme.
- **Lanes and timeline** keep the canonical instrument colours, since they are shared by
  all layouts.
- The spectrum's bar heights and the intensity colouring are eased per frame by the canvas
  (as in Classic), not computed from audio time.
- No picture or measurement here replaces looking at it with music.
