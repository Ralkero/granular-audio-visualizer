# Sonic Character (Phase A: analysis and diagnostics)

Sonic Character is GMV's measurable description of what kind of sound a piece of music is:
seven values between 0 and 1, each with the measurements it was made from and a
confidence. Phase A builds the analysis, the profile model and a diagnostic tool.

**Nothing draws from it yet.** No renderer, shader, colour, setting or visual behaviour was
changed, and the application does not compute it while it runs. Whether pictures driven by
it look right is a question for Phase B and for the owner's eyes; nothing here answers it.

It measures properties of the sound. It does not measure mood, emotion or aggression, and
it names no genre.

## Why it is separate from musical intensity

`MusicalIntensity` (docs/MUSICAL_INTENSITY.md) says how intense this moment is *for the
recording being heard*. Its loudness scale adapts to that recording's own quiet and loud
passages, which is what makes it good at following a build or a break, and what makes it
unable to tell one recording from another: an evenly loud metalcore track and an evenly
soft piano piece both settle in the middle of their own scale.

| | Musical Expression (`MusicalIntensity`, the live instrument states) | Sonic Identity (`SonicCharacter`) |
|---|---|---|
| Question | What is happening now? | What does this music sound like? |
| Reference | The recording itself: adapts to it | Fixed anchors: the same sound always gets the same value |
| Moves in | Tenths of a second to seconds | Tens of seconds; a drum hit does not move it |
| Comparable between recordings | No, by design | Yes |
| Changed by Phase A | No | New |

The measured difference is in [Results](#results-on-real-recordings).

## The seven dimensions

The brief proposed six. Rhythm is reported as two, because the brief itself asks that fast
and loose be told from slower and tight, and one number cannot hold both.

Each dimension is a weighted average of a few measurements. A measurement is first placed
on a 0..1 scale between two fixed anchors (values at or beyond an anchor give 0 or 1). The
anchors are round numbers near the 5th and 95th percentile of the calibration sample
described below, so 0 and 1 mean "about as far as real music goes".

| Dimension | 0 means | 1 means | Made from (measurement: value that gives 0 → value that gives 1, weight) |
|---|---|---|---|
| `force` | gentle | forceful and unrelenting | crest factor inside the frame: 10.8 → 8.2 dB (45%); how far the loud level stands above the middle level: 6 → 1 dB (35%); share of energy under 250 Hz: 0.25 → 0.65 (20%). The result is multiplied by spectral flux (0.13 → 0.21) and the square root taken |
| `brightness` | dark, warm | bright | spectral centroid on a ratio scale: 256 Hz → 1.1 kHz (60%); slope of the spectrum: -5 → -0.8 dB/octave (40%) |
| `pace` | few onsets | many onsets a second | onsets a second: 1 → 5 |
| `regularity` | loose, irregular | a tight steady pulse | strength of the strongest periodicity between 60 and 200 BPM: 0.2 → 0.65 (60%); spread of the time between onsets: 1.3 → 0.7 octaves (40%) |
| `roughness` | smooth, tonal | noise-like, percussive | peak-to-valley contrast inside the bands under 6 kHz: 19 → 11 dB (50%); share of the level that is a fresh rise: 0.02 → 0.07 (25%); spectral flux: 0.12 → 0.24 (25%) |
| `density` | open, sparse | crowded, filled | share of the 50 Hz–10 kHz spectrum within 25 dB of its strongest band: 0.45 → 0.97 (30%); how far the bands stand off the spectrum's slope: 11 → 4.8 dB (20%); how far single frames stand above the last 0.4 s: 1.8 → 0.4 dB (25%); loud level above middle level: 8 → 1 dB (25%) |
| `dynamics` | one even level | very quiet and very powerful passages | spread between the quiet tenth and the loud twentieth of the level: 2 → 22 dB |

Why these and not others:

- **Force is not loudness and not intensity.** The level of the recording cannot be used:
  it is the playback volume. What a forceful recording has at any volume is a saturated,
  compressed waveform (a low crest factor), a level that stays near its top, and weight in
  the low end. A held synthesizer pad has all three and is not forceful, so the result is
  gated by how much the sound keeps being struck anew (flux): without that it is 0.
- **Brightness does not use level at all**, so it cannot follow the volume. The centroid
  alone follows the bass line (one strong low note pulls it down); the slope of the whole
  spectrum is steadier.
- **Pace is onsets, not tempo.** The tempo figure is reported as context and never used:
  a fast tempo with few articulated attacks is not a busy rhythm, and the engine's tempo
  has octave errors.
- **Roughness** is the noise-like and the percussive, read from the spectrum. It is not
  perceived aggression.
- **Density** combines how much of the spectrum is filled at once with how little single
  sounds stand out of the rest. A loud recording of one instrument scores low on the
  second and usually on the first.
- **Dynamics** uses the level averaged over 0.4 s, in decibels, so one drum hit is not a
  "powerful passage". Passages more than 30 dB under the loud level are left out: a lower
  playback volume would push them under the silence threshold, and counting them only at
  high volume would make the value follow the volume.

Every measurement is listed with its unit in `gmv.character.measures.MEASURES` and
`profile.DERIVED`, and every profile carries them (`measurements`) beside the dimension
values, with each dimension's components and their normalised values. Where they come from
in the analysis is in the [feature inventory](SONIC_CHARACTER_INVENTORY.md).

## Independence from level

No dimension uses a level. They are built from shares of energy, frequencies, ratios and
differences in decibels, all unchanged when the audio is scaled. Two things are still
level-dependent, both inherited from the analysis pass and both reported:

- the silence threshold is absolute (-65 dBFS): music under it is not heard at all, so a
  very quiet capture loses its quiet passages. A profile whose middle level is under
  -50 dBFS carries a `quiet_capture` warning and half the confidence;
- onset detection has small absolute floors.

How much this matters on real recordings is measured below.

## Time: three windows and the whole

| Window | Time constant | Reads as | Use |
|---|---|---|---|
| `short` | 4 s | the texture of the last few bars | local character; the diagnostics find section boundaries with it |
| `medium` | 20 s | a passage or section | how the character evolves through a recording |
| `long` | 90 s | the recording's identity | the stable profile; this is the Sonic Identity |
| `whole` | everything since the last fresh start | the recording as one thing | offline summaries and comparisons |

A window forgets exponentially by audio time. Its mean is the weighted sum over the weight
it has gathered, so at the start it is the plain average of what has been heard, with no
warm-up bias, and its weight (seconds) says how much that is.

How the lengths were chosen (calibration sample, every second recording, values sampled
twice a second after the first 20 s):

| Time constant | Mean change of a value per second | Largest change in a second (99th percentile) | Share of the variation that is between recordings |
|---:|---:|---:|---:|
| 1 s | 0.054 | 0.32 | 0.51 |
| 2 s | 0.039 | 0.23 | 0.54 |
| **4 s** | 0.025 | 0.16 | 0.58 |
| 8 s | 0.015 | 0.10 | 0.64 |
| 12 s | 0.012 | 0.08 | 0.68 |
| **20 s** | 0.008 | 0.05 | 0.74 |
| 30 s | 0.006 | 0.04 | 0.78 |
| 45 s | 0.005 | 0.03 | 0.82 |
| 60 s | 0.004 | 0.03 | 0.84 |
| **90 s** | 0.003 | 0.02 | 0.86 |
| 120 s | 0.003 | 0.02 | 0.87 |
| 180 s | 0.003 | 0.02 | 0.89 |

- 4 s is the shortest window that makes sense: the analysis itself counts onsets over 4 s
  and reads the pulse from 12 s, so anything shorter would only repeat single frames. At
  4 s a value still moves by 0.025 a second on average: local texture.
- 20 s: sections found in the calibration sample have a median length of 27 s
  (quartiles 18 and 55 s). A 20 s window settles inside most of them.
- 90 s: the share of the variation that lies between recordings (instead of inside them)
  rises quickly up to about 90 s and slowly after; a longer window would hardly be more
  stable and would take most of a track to follow a real change.

Behaviour, all covered by tests:

- **One transient does not move the identity.** Three frames of an extreme attack move the
  long window by under 0.01 and the short one by under 0.1.
- **A real change is followed in order.** Twenty seconds into a different sound the short
  window has arrived, the medium one is about three quarters of the way, the long one a
  third.
- **Silence and pauses hold everything.** Silent frames add nothing and age nothing; while
  the player is paused the profile is exactly what it was, marked `active: false`. Before
  any music the values are empty (`None`) with confidence 0, not zero.
- **Quiet music is still music.** Anything above the silence threshold is analysed.
- **After three seconds or more of silence** the windows keep their averages and a quarter
  of their weight. The silence may have been the gap between two recordings: the old
  character is the best guess until the new music says otherwise, and it gives way about
  four times faster, with lower confidence meanwhile.

## Confidence and warnings

Each dimension's confidence (0..1) says how far the value can be relied on, not how strong
it is. It is the product of:

- **enough music**: `1 - exp(-seconds / needed)`, where `seconds` is the weight of the
  window and `needed` is 2 s for brightness and roughness, 4 s for pace and density, 8 s
  for force, 12 s for regularity and 20 s for dynamics. The short window can therefore
  never be confident about dynamics, which is right: four seconds do not hold them;
- **the measurements it needs**: the share (by weight) of its components that were
  available. Frames without a display spectrum (older sessions) lower brightness and
  density; a stream without onset events lowers regularity to 0.6;
- **enough onsets** (regularity only): `1 - exp(-onsets / 8)`;
- **0.5 for a quiet capture** (see above).

Confidence does not know whether a measurement means what the dimension's name says for a
particular kind of music. That is in [Limitations](#limitations-and-confounders).

Warnings a profile can carry: `little_music` (under 5 s heard), `held` (no music now),
`quiet_capture`, `no_display_spectrum`, `no_onset_events`. The diagnostics add
`low_sample_rate`, `short_recording` and `low_confidence` (naming the dimensions).

## Segments, seeking, restart

`SonicCharacter` follows the same rules as the visual state (`gmv.visualizer.state`):

- the newest segment seen only moves forward. A frame of a **newer segment** (a capture gap
  or format change) starts everything afresh: nothing of the earlier segment is kept;
- a frame or onset of a **segment that is over** is refused, unless it lies two seconds or
  more before the newest audio time seen: then the timeline has started again and
  everything starts afresh;
- **time going backwards, or jumping ahead more than 10 s**, inside one segment is a
  restart or a seek: everything starts afresh;
- a hole of up to 10 s counts as silence of that length;
- the same frame twice is taken once.

## Recording, replay and determinism

The profile is **rebuilt from the recorded frames, not recorded**. A session already holds
every feature frame and onset event; the profile is a pure function of them, in order, and
costs 45 µs a frame to rebuild. So the session schema, the message schema and the engine
are unchanged, every existing session can be profiled, and a later change to the
definitions needs no migration: `profile_version` (0.1) names the definition a profile was
made with.

What the tests check:

- the same stream gives byte-identical profiles every time;
- a live engine run, the same session read back from its log, and the application's
  `replay` at two speeds give identical profiles at every frame along the way, not only at
  the end;
- reading the profile (at any display frame rate, or never) changes nothing: there is no
  per-read state;
- frames 10 ms apart, 20 ms apart, or unevenly 8 to 45 ms apart give the same values within
  0.03: the windows forget by audio time, not by update count;
- the time is always the frames' own audio time. No clock is read anywhere in the package.

## Diagnostics

```
gmv character analyze PATH [PATH ...] [--list FILE] [--out DIR] [--gain-db DB] [--processes N]
```

A path is an audio file (WAV, FLAC, MP3, OGG, AIFF), a recorded session folder, or its
`messages.jsonl`. `--list` names a text file with one path per line, for a corpus of twenty
or thirty recordings. Files are decoded in memory and only read. **Nothing is searched for**:
only the paths given are analysed. Nothing is copied, uploaded or downloaded.

Outputs, in `--out` (default `local-evidence/sonic-character/report`, which Git ignores;
the outputs name the files they were made from, so they stay local):

- `report.md`: overall profiles of all recordings side by side with their mean intensity;
  how far each dimension spreads across them and how they correlate; the four comparisons
  with `MusicalIntensity` found automatically; the level check; then per recording: every
  dimension with its confidence, its value in each window and the measurements behind it,
  warnings, sections with their own profiles, and the distribution of the measurements
  over the frames.
- `<recording>.json`: the full profile model, the evolution sampled every second (short,
  medium and long windows with confidences, and intensity), sections, distributions, cost.
- `<recording>.svg`: the medium-window evolution of the seven dimensions with
  `MusicalIntensity` as a dashed line and the section boundaries.
- `summary.json`: definitions, all overall profiles, the comparisons, the level checks and
  anything that could not be analysed, with the reason.

`--gain-db -10` analyses each audio file a second time with its samples scaled in memory
and reports how far each dimension moved.

A file runs through the same analysis pass as live audio, as fast as it will go (about
20 times real time on the development machine), entirely offline. It never runs while the
application plays.

Example (generated test audio, not music):

```
| Recording | force | brightness | pace | regularity | roughness | density | dynamics | intensity (mean) | music (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| soft swelling tone | 0.00 | 0.00 | 0.00 | 0.89 | 0.00 | 0.00 | 1.00 | 0.62 | 30 |
| wall of noise and bass | 1.00 | 0.96 | 0.17 | 0.76 | 0.46 | 0.79 | 0.00 | 0.47 | 30 |
```

## Results on real recordings

Kind of evidence: **offline analysis of real recordings** through the real analysis pass,
on CachyOS. It is not a live capture and nobody listened: it stands in for real-hardware
validation and is not one. The recordings are the owner's own files, read where they are;
titles are kept out of this document, as the corpus rules require (docs/CORPUS.md).

### What was analysed

| Set | Recordings | Music | How chosen | Used for |
|---|---:|---:|---|---|
| Calibration | 39 | 2 h 42 min | The first recording, by identifier, of each of the 38 artist groups of the owner's registered local corpus (`owned_local`), plus reference track A | Choosing the measurements, anchors and window lengths |
| Held out | 28 | 2 h 2 min | The second recording of each of the 28 groups that have one | Run once, after the definitions were fixed; nothing was changed afterwards |
| Recorded sessions | 4 | 26 min | Sessions recorded by the application on Windows on 3 and 15 November 2026 | Whether old sessions can be profiled, and what the warnings say |

The collection is mostly metal, metalcore and rock, with pop-punk, classic rock, a few
quiet songs and ambient or orchestral game soundtracks. Nearly all the files are MP3.

**The two reference tracks of the brief.** Reference track A (the dense metalcore track)
was found in the music folder the registered corpus points to. Reference track B (described
in the brief as mellow and atmospheric) was not reachable at first; the owner then supplied
its path and it was analysed with the definitions already fixed. Neither track was used to
set an anchor after B was seen. The result is under
[The two reference tracks](#the-two-reference-tracks), and it is the most important
limitation found in this phase.

### Sonic Character separates recordings; intensity does not

Overall value of each recording (the `whole` window), least to most across the set, and
its standard deviation:

| | Calibration (39) | | | Held out (28) | | |
|---|---:|---:|---:|---:|---:|---:|
| | least | most | s.d. | least | most | s.d. |
| force | 0.00 | 0.96 | 0.23 | 0.00 | 0.93 | 0.28 |
| brightness | 0.00 | 1.00 | 0.24 | 0.00 | 0.99 | 0.29 |
| pace | 0.15 | 0.86 | 0.16 | 0.09 | 0.89 | 0.22 |
| regularity | 0.02 | 0.95 | 0.21 | 0.06 | 1.00 | 0.23 |
| roughness | 0.05 | 0.91 | 0.18 | 0.06 | 0.79 | 0.21 |
| density | 0.04 | 0.95 | 0.23 | 0.06 | 0.96 | 0.24 |
| dynamics | 0.02 | 1.00 | 0.24 | 0.04 | 1.00 | 0.28 |
| `MusicalIntensity`, mean | 0.49 | 0.82 | 0.08 | 0.50 | 0.84 | 0.09 |

Every dimension uses nearly the whole scale on recordings it was not calibrated on. The
mean intensity of every recording lies in one third of its scale, and its order has
nothing to do with how heavy the music is: correlation of mean intensity with the
dimensions is between -0.30 and +0.40 in the calibration sample and between -0.12 and
+0.31 held out. The eight most forceful recordings of the calibration sample (force 0.86 to
0.96) have mean intensities from 0.53 to 0.82, seven of them 0.60 or less; a solo piano
piece from a game soundtrack (force 0.00, density 0.04, dynamics 1.00) has 0.65.

The rankings read as expected: the top of `force` and `density` is metalcore, deathcore
and a heavy game soundtrack; the bottom is ambient pieces, a quiet song for voice and
piano, and classic rock ballads. Dark, sparse electronic pieces are at the bottom of
`brightness`; modern metalcore and pop-punk at the top.

### The four comparisons with intensity

Found automatically by the report (`comparison` in `summary.json`), then checked by hand.

1. **Similar intensity, different character.** An ambient piano piece and a pop-punk
   track: mean intensity 0.65 and 0.64. Force 0.00 against 0.82, density 0.04 against 0.90,
   dynamics 1.00 against 0.26; the profiles differ by 0.65 on average per dimension. Held
   out: another ambient piece and a metalcore track, intensity 0.03 apart, profiles 0.77
   apart.
2. **Different intensity, similar character.** Two metalcore tracks, one of them reference
   track A: mean intensity 0.82 and 0.54, profiles 0.05 apart. Held out: 0.84 and 0.50,
   profiles 0.09 apart.
3. **One recording, intensity changes, character holds.** A held-out metalcore track: a
   passage from 37 to 122 s has intensity 0.64, one from 176 to 211 s has 0.82. Their
   profiles differ by 0.02 on average, by 0.04 at most (force 0.97 and 0.93, density 0.95
   and 0.98, roughness 0.75 and 0.72).
4. **One recording, the texture really changes.** A calibration track keeps its intensity
   (0.83 from 20 to 131 s, 0.85 from 163 to 203 s) while the second passage is a quiet
   interlude: force 0.96 to 0.00, brightness 0.79 to 0.03, roughness 0.75 to 0.22, density
   0.94 to 0.71. Intensity reports no change; the profile changes completely.

Over all 1,101 pairs of sections inside the same recording, the difference in intensity
and the difference in character correlate at 0.39: related, far from the same thing.

### The two reference tracks

**The profile does not tell these two recordings apart as the brief expected.** Taken as
whole recordings they measure as two similar things: dense, forceful, loud-mastered modern
productions with a heavy low end.

| | force | brightness | pace | regularity | roughness | density | dynamics | intensity (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A (dense metalcore), 171 s | 0.88 | 0.74 | 0.42 | 0.40 | 0.70 | 0.91 | 0.30 | 0.54 |
| B (described as mellow, atmospheric), 257 s | 0.81 | 0.63 | 0.40 | 0.42 | 0.64 | 0.81 | 0.49 | 0.80 |

Every confidence is 1.00 for both. The profiles differ by 0.08 on average per dimension;
two recordings from opposite ends of the samples differ by 0.6 to 0.8. Among the 67
recordings of the samples, B's force is above 64% of them and its density above 40%: it
sits with the rock and metal recordings, far from the ambient and quiet ones.

What the measurements do see: B is somewhat darker (centroid 617 Hz against 739 Hz), a
little more tonal (in-band contrast 13.3 dB against 12.0), slightly less filled (0.86 of
the spectrum against 0.91) and more dynamic (level range 11.8 dB against 8.1). All in the
direction one would expect, none of them large. `MusicalIntensity` goes the other way: B
averages 0.80 and A 0.54.

Where they do differ clearly is in how they evolve:

| A, section | force | brightness | pace | regularity | roughness | density | dynamics | intensity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–24 s | 0.85 | 0.92 | 0.77 | 0.42 | 0.88 | 0.79 | 0.23 | 0.56 |
| 24–85 s | 0.92 | 0.80 | 0.55 | 0.59 | 0.80 | 0.97 | 0.01 | 0.60 |
| 85–104 s | 0.76 | 0.61 | 0.20 | 0.06 | 0.58 | 0.87 | 0.06 | 0.49 |
| 104–155 s | 0.86 | 0.80 | 0.37 | 0.27 | 0.70 | 0.93 | 0.16 | 0.55 |
| 155–171 s (the ending) | 0.00 | 0.12 | 0.00 | 0.39 | 0.12 | 0.48 | 0.70 | 0.27 |

| B, section | force | brightness | pace | regularity | roughness | density | dynamics | intensity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–27 s (the opening) | 0.78 | 0.00 | 0.39 | 0.11 | 0.60 | 0.12 | 1.00 | 0.79 |
| 27–127 s | 0.82 | 0.82 | 0.49 | 0.48 | 0.73 | 0.94 | 0.05 | 0.88 |
| 127–140 s | 0.22 | 0.88 | 0.00 | 0.29 | 0.30 | 0.60 | 0.49 | 0.54 |
| 140–177 s | 0.85 | 0.84 | 0.42 | 0.55 | 0.76 | 0.96 | 0.00 | 0.88 |
| 177–200 s | 0.76 | 0.52 | 0.52 | 0.59 | 0.45 | 0.67 | 0.11 | 0.78 |
| 200–212 s | 0.81 | 0.94 | 0.32 | 0.21 | 0.64 | 0.70 | 0.39 | 0.77 |
| 212–227 s | 0.94 | 0.69 | 0.60 | 0.60 | 0.88 | 0.87 | 0.03 | 0.90 |
| 227–244 s | 0.58 | 0.62 | 0.00 | 0.30 | 0.54 | 0.91 | 0.32 | 0.76 |
| 244–257 s (the ending) | 0.00 | 0.39 | 0.17 | 0.27 | 0.24 | 0.30 | 1.00 | 0.19 |

A holds one character through its four main sections (force 0.76 to 0.92, density 0.79 to
0.97). B opens dark and sparse (brightness 0.00, density 0.12), drops out twice, and moves
between dense and open passages: nine sections against five, and far wider swings. Its
long main passages (27–127 s, 140–177 s) measure almost exactly like A's.

What this means:

- Whatever makes B feel mellow to a listener beside A, it is mostly **not in these
  measurements**. Likely candidates are things GMV does not measure at all: how the voice
  is used (sung against screamed), harmony and melody, the feel of the tempo, the kind of
  distortion. Two recordings mixed and mastered in the same modern, dense, bass-heavy way
  look alike to a spectrum-and-level analysis whatever is played and sung on them.
- The profile does separate broad kinds of sound (ambient from metal, sparse from dense,
  dark from bright), as the samples show. It does not yet separate two dense modern
  productions by their musical personality, which is the case the phase was started for.
- The evolution over time (the section tables) carries more of the difference than the
  overall values do.
- Nothing was retuned to pull the two apart. Doing so on two recordings would have
  produced a difference without knowing whether it meant anything.

Analysed 20 dB quieter, neither track moved by more than 0.015 on any dimension.

### Which dimensions are redundant

Correlation between the overall values across recordings (calibration / held out):

| | force | brightness | pace | regularity | roughness | density | dynamics |
|---|---:|---:|---:|---:|---:|---:|---:|
| force | | 0.59 / 0.80 | 0.12 / 0.47 | -0.18 / 0.19 | 0.86 / 0.93 | 0.77 / 0.92 | -0.60 / -0.59 |
| brightness | | | 0.09 / 0.61 | -0.05 / 0.38 | 0.67 / 0.86 | 0.84 / 0.89 | -0.55 / -0.52 |
| pace | | | | 0.74 / 0.80 | 0.42 / 0.61 | 0.12 / 0.52 | -0.40 / -0.51 |
| regularity | | | | | 0.15 / 0.35 | 0.03 / 0.27 | -0.23 / -0.52 |
| roughness | | | | | | 0.79 / 0.88 | -0.73 / -0.54 |
| density | | | | | | | -0.77 / -0.63 |

**Between recordings there are about three independent things, not seven**: how full,
bright, rough and forceful the sound is (four dimensions that largely agree); how busy and
steady the rhythm is (two that largely agree); and how much the level varies, which runs
against the first group. That is a property of this music as much as of the measurements:
in this collection what is dense is also bright, distorted and loud-mastered.

Inside one recording the dimensions say much more different things. Mean correlation of
the medium-window values over time within a recording (calibration / held out):

| | brightness | pace | regularity | roughness | density | dynamics | intensity |
|---|---:|---:|---:|---:|---:|---:|---:|
| force | 0.18 / 0.31 | 0.44 / 0.30 | 0.32 / 0.12 | 0.56 / 0.51 | 0.64 / 0.69 | -0.51 / -0.60 | 0.27 / 0.10 |
| brightness | | 0.09 / 0.18 | -0.05 / -0.13 | 0.33 / 0.41 | 0.39 / 0.49 | -0.24 / -0.33 | -0.10 / 0.00 |
| pace | | | 0.61 / 0.47 | 0.46 / 0.52 | 0.23 / 0.12 | -0.29 / -0.25 | 0.28 / 0.25 |
| regularity | | | | 0.22 / 0.16 | 0.20 / -0.02 | -0.19 / -0.07 | 0.14 / 0.15 |
| roughness | | | | | 0.38 / 0.38 | -0.37 / -0.32 | 0.22 / 0.21 |
| density | | | | | | -0.64 / -0.69 | 0.10 / -0.08 |
| dynamics | | | | | | | -0.20 / -0.03 |

So as a description of how a recording evolves, all seven carry information of their own,
and none of them follows intensity (0.28 at most).

### Does the level matter

Each audio file was analysed a second time with its samples scaled in memory. Change of
each overall value:

| | force | brightness | pace | regularity | roughness | density | dynamics |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10 dB quieter, held out (28): median | 0.000 | 0.002 | 0.001 | 0.001 | 0.000 | 0.003 | 0.000 |
| ... largest | 0.007 | 0.013 | 0.006 | 0.020 | 0.014 | 0.028 | 0.018 |
| 20 dB quieter, calibration (39): median | 0.002 | 0.003 | 0.003 | 0.002 | 0.002 | 0.012 | 0.002 |
| ... 90th percentile | 0.007 | 0.016 | 0.011 | 0.015 | 0.013 | 0.025 | 0.023 |
| ... largest | 0.021 | 0.085 | 0.030 | 0.076 | 0.056 | 0.086 | 0.216 |

At 10 dB down nothing moved by more than 0.03. At 20 dB down 34 of 39 recordings stayed
within 0.05 on every dimension. The five that did not are the quietest recordings (ambient
and sparse pieces whose level is already low): 20 dB down, part of their music is under
the silence threshold and is not heard. The largest change was in dynamics (0.22), for the
reason given under [Independence from level](#independence-from-level).

A first version of dynamics did depend on the level (15 of 39 recordings moved by more
than 0.05 at 20 dB down). Leaving out passages more than 30 dB under the loud level
brought that to 3.

### Recorded sessions

Four sessions recorded by the application on Windows (two of system audio, two of one
application's audio, ten minutes each; one of schema 1.0) were read back and profiled. All
four predate the display spectrum and were captured at a low level, and the report says so:
`no_display_spectrum` and `quiet_capture` on each, with confidence at most 0.5 and 0.25 to
0.3 for brightness and density. Their values are therefore not presented as findings. What
this shows is that old sessions load and are flagged instead of being profiled as if
complete. **No session recorded with the current analysis during real playback was
available**, so a live capture has not been profiled end to end.

A real recording put through the engine into a session on CachyOS (reference track A, as
a file, not as live playback) and read back gave the same overall profile and the same
evolution, value for value, as analysing the file directly (8,559 frames).


## Limitations and confounders

- **Two dense modern productions with different musical personalities measure alike.**
  This is the case the phase was started for, and the two reference tracks show it
  ([above](#the-two-reference-tracks)): 0.08 apart on average. Vocal style, harmony,
  melody and the feel of the tempo are not measured by anything GMV has. Closing this gap
  needs new measurements, not new anchors.
- **Brightness, roughness, density and force go together across recordings.** Between
  recordings they are close to one axis (correlations of 0.6 to 0.9):
  in this music, what is dense is also bright, noise-like and forceful. They are computed
  from different measurements and separate much better over time inside a recording, but a
  renderer should not treat them as seven independent controls.
- **Pace and regularity go together** (0.74 to 0.80 between recordings): whole-mix onsets are
  fewer and less even in a dense distorted mix, where many attacks are masked. A fast,
  dense metal track reads a *lower* pace than a sparse acoustic one with clear attacks.
  Pace is "articulated attacks a second in the mix", not how fast the music feels.
- **Regularity is the weakest dimension.** Mixed subdivisions give a wide spread of onset
  intervals even in tightly played music, and the engine's periodicity strength is modest
  in dense mixes. Treat low regularity as "no clear pulse found", not "played loosely".
- **Force is fooled by a saturated, bass-heavy but sparse recording**: a sparse beat with a
  loud synthetic bass scored 0.83 in the calibration sample. Its density (0.30)
  is what tells it from a wall of sound; the two have to be read together.
- **Dynamics counts fade-outs and intros**, and it ignores passages more than 30 dB under
  the loud level. It still depends on the capture level when quiet passages fall under the
  silence threshold (figures above).
- **Lossy files and low sample rates**: nothing above the cut-off exists, so brightness
  reads a little lower; the band above 6 kHz is left out of roughness for that reason.
- **Band edges and spectrum bands are settings.** The anchors assume the defaults (the
  seven bands, 48 display bands). A session recorded with other settings gets missing or
  shifted measurements; with the display spectrum turned off, brightness and density rest
  on fewer measurements, and say so.
- **The anchors come from one collection**: the calibration sample is the owner's own
  library, heavy on metal and rock with game soundtracks, and little classical, jazz,
  electronic or spoken material. For other music 0 and 1 may sit in the wrong places.
- **No listening test.** Whether the values match what a person hears has been checked only
  by reading the rankings against what the recordings are known to be. Nobody has judged
  it by ear in a structured way.
- **Track changes are not detected.** In a continuous capture the identity follows from
  one track to the next over about a minute and a half, faster after a gap of silence.

## Performance

Measured with `scripts/benchmark_character.py` on the development machine (CachyOS,
Python 3.14, NumPy 2.5; 120 s of generated audio through the real analysis pass, best of
five passes):

| | |
|---|---:|
| Sonic Character, per feature frame | 41 µs |
| ... reading the frame's measurements | 18 µs |
| ... updating four windows | 10 µs |
| ... the level distribution | 3 µs |
| The analysis pass itself, per frame | 764 µs |
| Sonic Character as a share of the analysis pass | 5.3% |
| Share of one core at 50 frames a second | 0.2% |
| `MusicalIntensity`, per frame, for comparison | 2.3 µs |
| Building one profile (one window) | 171 µs |
| Building all four | 693 µs |
| Memory held by one tracker | 23 kB, constant |
| Memory kept per message after warm-up | 0 bytes |

On the real recordings the median was 45 µs a frame beside 0.74 to 0.82 ms for the analysis
pass. A profile is only built when it is asked for; the per-frame update builds none.

(Phase A.) Nothing in the application calls this yet, so **the running application's performance is
unchanged by construction**: the capture, the analysis pass, the visual state and every
renderer are byte-for-byte what they were. The cost above is what Phase B will add when it
runs the tracker live, to be measured again there. The diagnostics run offline only. Phase B now runs it
live and measures it there: see [Phase B](#phase-b-live-analysis-and-the-sonic-character-monitor).

## What Phase A did not do, and what remains for Phase B

Not done, by design: any renderer or shader change, any visual setting, any colour or
geometry driven by the profile, genre presets or classification, model weights, network
access, any change to capture, feature extraction, onset or beat detection, instrument
attribution, `MusicalIntensity`, the message or session schema, or the packaging.

For Phase B:

0. **Decide first whether the profile is enough.** It did not separate the two reference
   tracks. Before any visual is tied to it, either accept that it describes broad kinds of
   sound and how a recording evolves, or look for measurements of what it misses (the
   voice, harmony, the feel of the tempo) and test them on pairs like this one.
1. **Run it live.** Add a `SonicCharacter` to `VisualState` beside `MusicalIntensity`
   (`consume` already receives every frame and onset event), behind the same segment
   handling, and measure the frame cost in the running application.
2. **Decide which dimensions drive anything.** On the evidence here: brightness, density or
   force (one of them, or a blend), dynamics and regularity carry the most separate
   information between recordings; all seven move separately inside one.
3. **Have the owner judge the values against their ears** on tracks they know, before any
   visual is tied to them. The diagnostic report is the tool for that.
4. **Detect track changes**, or take them from the player, so the identity starts afresh
   with a new track instead of drifting over.
5. **Widen the calibration** beyond one collection before trusting the ends of the scales.
6. If roughness needs to be more than a proxy, try the delayed harmonic/percussive
   measure the engine already has (see the inventory).

## Phase B: live analysis and the Sonic Character Monitor

Phase B runs the Phase A tracker on the live stream and shows what it measures, so the
values can be judged against what is heard during playback. **No visual uses them**: Radiant,
Spectrum bars, Classic, the shaders, colours and particles are unchanged, and so is the visual
state. Deciding which dimensions deserve a visual comes after watching them on real music.

### How it runs

`gmv.character.live.LiveCharacter` is one more subscriber of the `EventBus`, beside the
visualizer and the session writer. There is no second capture or analysis pipeline:

- **Its own bounded queue** (4,096 messages, about a minute of frames). Publishing a message
  puts it on this queue and returns, as for every subscriber; nothing of the character
  analysis runs in the capture path or the analysis pass. If the worker ever fell a minute
  behind, the oldest messages would be dropped and counted, as for every subscriber.
- **Its own worker thread** takes the feature frames and onset events in publication order and
  hands them to an unchanged `SonicCharacter`. Given the same messages it gives exactly the
  profiles Phase A gives offline, whatever the sizes of the batches it drains (tested, and
  checked against `gmv character analyze` on the same file).
- **Snapshots, not a stream of results.** Once per 0.5 s of audio time the worker builds the
  short (4 s), medium (20 s) and long (90 s) profiles and keeps the seven values, their
  confidences and the profile's warnings: one latest snapshot and a rolling two-minute history
  (240 points). Nothing accumulates, so a stalled or closed monitor never sees stale results
  when it looks again.
- **Playback.** A new session (Start, a source change, Apply changes, a replay) starts
  everything afresh. Inside a session the tracker's own rules apply: a new capture segment, a
  restart or a seek starts afresh; silence and pauses hold the values without ageing them (the
  monitor says "No music at the moment (values held)"); a pause during which nothing is
  published changes nothing. Stop keeps the last values readable ("Stopped, last values").
- **Local only.** Nothing is written or sent: no audio, no per-frame measurements, no
  snapshots. Closing the application stops the worker.

In the application's default mode the engine runs in its own process and the live analysis
runs in the window's process, on the forwarded messages. With `GMV_ANALYSIS=thread` it shares
the engine's interpreter; the benchmark below measures that case, the closer of the two to
the analysis pass.

### The monitor

Open it with the **Character** button in the top bar or the **C** key; close it with its
Close button, the window's own close button, the Character button or C again. It is a
separate window: the visualizer's layout and drawing do not change while it is open, and it
can be moved to another screen.

- **Seven rows**, one per dimension: a bar on a labelled 0–1 scale with the value written
  beside it, and under it a thinner bar and the number for its confidence.
- **"not enough signal"** replaces the value when there is none yet or its confidence is under
  0.3. That threshold is a display choice of the monitor; the values and confidences are Phase
  A's, unchanged. Confidence is how much signal supports a value (enough music, the
  measurements it needs, enough onsets), not the chance that it is right.
- **Window**: Short (last 4 s) follows the music within seconds; Medium (20 s) a passage; Long
  (90 s) the track's identity, which takes a minute or more to settle on a new track.
- **History**: the last two minutes of audio, one line per dimension, faint and dashed where
  the confidence is under 0.3 and broken where there was no value.
- **State and warnings** in words: waiting for music, measuring (and how many seconds of music
  were heard), no music at the moment, stopped; "under five seconds of music", "the level is
  close to the silence threshold", and so on.
- **Counters** at the bottom: frames taken in, messages dropped (should stay 0), messages
  waiting, and the update cost per frame.

It refreshes four times a second while it is open and redraws only when a new snapshot has
arrived; closed, it does nothing.

### Performance (measured on CachyOS)

`scripts/benchmark_character_live.py`: generated audio (not a recording) played through the
real engine at real-time pace in one process, alternately without and with the live analysis,
two rounds of 60 s each; then 20 minutes of audio as fast as it goes; then the monitor drawn
offscreen. Ryzen 9 5950X, Python 3.14.10, 2026-11-28.

| | without | with the live analysis |
|---|---:|---:|
| Process time, share of one core (round 1 / round 2) | 7.4% / 7.2% | 7.9% / 8.0% |
| Analysis pass per frame, 95th percentile (ms) | 1.13 / 1.11 | 1.08 / 1.11 |
| Analysis pass per frame, worst (ms) | 1.61 / 1.71 | 1.49 / 1.62 |
| Estimated capture to publication, 95th percentile (ms) | 28.5 / 28.2 | 28.1 / 28.2 |
| Dropped audio blocks, discontinuities | 0, 0 | 0, 0 |

| Live analysis | |
|---|---:|
| Tracker update per frame (real time, on its thread, waits for the interpreter included) | 92–93 µs |
| The same, running as fast as it goes (20 min of audio) | 64 µs |
| Worst single update (20 min) | 0.67 ms |
| One snapshot (three profiles), about two per audio second | 0.81 ms |
| Messages waiting at most (queue of 4,096) | 8 |
| Messages dropped, errors | 0, 0 |
| Process memory over 20 minutes of audio | 122.0 MB to 122.4 MB |
| History kept | 240 points, fixed |
| Monitor: refresh and draw with two minutes of history | 3.4 ms (95th percentile 3.6 ms), 4 times a second while open |
| Monitor: refresh with nothing new | 0.005 ms |

So the live analysis adds about half a percentage point of one core, and the analysis pass,
the queues and the timing of publication are the same with it as without it. The per-frame
update is slower than Phase A's offline 41 µs because it is timed on a thread that shares the
interpreter with the analysis pass and includes onset events. These are generated-audio runs
on this machine, not live playback.

### Tests

`tests/test_character_live.py` (25) and `tests/test_character_monitor.py` (12): exact
agreement with Phase A for the same messages through the bus and worker, and with `gmv
character analyze` on the same file; batch sizes from one message to all; history points equal
the offline profile at each sample time; pause with silence or with no frames; stop and a new
start; new sessions, segments and restarts; empty input; too little signal and confidence
passed through unchanged; the bounded queue under a stalled worker (drops counted, newest
state wins); readers on other threads; clean attach and detach with no threads left; a
failing message counted and skipped; snapshots holding no audio or per-frame data; the
visualizer seeing the same messages and state with the live analysis attached; the monitor's
text, low-signal display, redrawing only on new data, fixed history, timer only while open;
the Character button and C key; a running session, stop and restart; the view's state the same
with the monitor open or closed; shutdown.

### Limits

- **Phase A's main finding stands.** The two reference tracks with different musical
  personalities measured alike as whole recordings; the dimensions describe broad production
  characteristics (dense or sparse, dark or bright, forceful or gentle, even or dynamic) better
  than musical personality. Phase B does not change the measurements and does not address
  this: it makes it possible to see how they behave during real playback.
- **Track changes are not detected.** With system capture, one session can span many tracks:
  the short window follows a new track within seconds, the long one takes a minute or two
  (faster after three seconds of silence). Pressing Stop and Start begins afresh.
- **Not yet run on real playback.** The tests and the benchmark use generated audio; nobody has
  watched the monitor against music they know. Not run on Windows.
- If messages were ever dropped (the counter shows it), the live profile would rest on fewer
  frames than an offline analysis of the same session.

### Testing it by ear

1. Start GMV. In **Settings**, under Audio source, choose **System audio (PipeWire)** on
   CachyOS (the system output on Windows), or an audio file. Press **Start**, then
   **Character** (or C). Move the monitor next to the visualizer or onto the other screen.
2. Play a track you know well. For the first seconds most rows read "not enough signal";
   brightness and roughness come first, dynamics and regularity last (they need 12 to 20 s).
3. Watch the **Short** window during the track and check:
   - **Force** and **density** rise when the full band comes in and fall in a quiet intro or a
     breakdown;
   - **brightness** rises with cymbals, distorted guitars or a bright mix and falls on a dark,
     bass-heavy passage;
   - **pace** follows how many notes and hits there are a second; **regularity** is high for a
     steady beat and low for rubato or free playing;
   - **roughness** is high for noisy, percussive, distorted sound and low for clean, tonal sound;
   - **dynamics** (Medium or Long) is high when quiet and loud passages alternate and low for a
     track mastered at one level.
4. Pause the player: the state should read "No music at the moment (values held)" and the
   values should not move. Resume: they should carry on.
5. After a few minutes, switch to **Long** and compare two tracks you hear as different. Note
   where the values agree with what you hear and where they do not; remember that two dense,
   modern productions may measure alike even when they feel different.
6. Report back: the tracks (as you like to name them), which rows matched what you heard,
   which did not and when, whether "not enough signal" appeared where you expected, and whether
   the counters showed any dropped messages.
