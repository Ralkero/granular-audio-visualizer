# Sonic Character, Phase A2: what the profile misses, and what could fill it

Research and offline experiments. **Nothing here is used by the visualizer**, the Phase A
profile is unchanged, and no renderer, shader, setting, schema or analysis pass was
touched. The experimental code lives in `src/gmv/character/research/` and is reached only
through `gmv character research`.

Short version:

- **Harmony can be measured, and it is new information.** How fast the harmony moves and
  how clearly the music sits in a key are robust (unchanged by level, tone, saturation and
  band-limiting; nearly unchanged by removing the drums), stable over a recording, and
  mostly not predictable from the Phase A profile. They are the strongest result.
- **Rhythmic feel can be measured in part.** How concentrated the rhythm is on its pulse,
  and how busy the low end is, hold up. Anything that depends on *which* metrical level is
  "the beat" does not: the pulse found is right about four times in five and otherwise
  off by a factor of two or three, and it changes from one passage to the next.
- **Vocal delivery cannot be measured with what GMV has.** Whether a voice is sounding can
  be estimated (with the existing activity model). Whether it is sung or screamed cannot:
  the one promising cue, how much of the voice has a pitch, works on ordinary mixes and
  fails exactly in dense, distorted ones, where the two reference tracks come out equal.
- **The two reference tracks** go from "closer than 96% of pairs" (Phase A alone) to "an
  ordinary distance apart" (Phase A with the harmony measurements). That is real progress
  and it is not a solution: nothing here says one is mellow and the other is not.
- **Recommendation:** one more research step, on the voice, before Phase B relies on a
  distinction between the two reference tracks; the harmony measurements and two rhythm
  measurements are ready to be candidates for Sonic Identity. Details at the end.

## 1. What Phase A left open

Phase A ([SONIC_CHARACTER.md](SONIC_CHARACTER.md)) measures the texture of the mix:
force, brightness, pace, regularity, roughness, density, dynamics. On two recordings
described as opposite in musical personality (reference track A, dense metalcore;
reference track B, mellow and atmospheric) the seven values differ by 0.08 on average.
Both are dense, loud-mastered, bass-heavy productions, and a spectrum-and-level analysis
sees that. Three things it does not look at were investigated here: the voice, the harmony
and melody, and the feel of the rhythm.

Phase A's own evaluation also had a weakness that this phase corrects: its held-out
recordings were by artists already in the calibration sample. Here no artist is on both
sides (section 5).

## 2. What GMV already has that bears on this

Read from the code, not from field names. The general inventory is in
[SONIC_CHARACTER_INVENTORY.md](SONIC_CHARACTER_INVENTORY.md); this is what matters for the
three questions, with what testing against isolated stems showed (section 6).

| Existing capability | What it actually does | Tested here | Verdict |
|---|---|---|---|
| Vocal analyzer (`gmv.instruments.vocals`): `level` | Level of the stereo centre between 300 Hz and 3.4 kHz against its own recent peak | Follows when the vocal stem really sounds inside a song (median AUC 0.84 over 135 songs) | Useful as "something is in the centre in the voice's range". Not proof of a voice: snare, bass and centred guitars are there too |
| Vocal analyzer: `pitch_midi`, `pitch_confidence` | The pitch whose harmonic series best explains the centre's peaks, A2 to A5 | When it reports a pitch a voice is sounding 84% of the time (median); it is within a semitone of the voice's real pitch 63% of the time | A rough pitch. Jitters from frame to frame: in the corpus it never holds a note for 0.3 s in a dense mix, so contour statistics built on it are noise (section 4) |
| Vocal analyzer: phrase boundaries | Starts and ends of centre sound, gated by the activity model | Stated for too few passages to judge (confidence 0.02) | Not usable as it is |
| Activity model (`gmv.activity`, weights not shipped): vocals | A small network's per-frame judgement that a voice sounds | On the 50 MUSDB test songs: finds 80% of vocal frames, fires on 18% of the others (medians). Cannot be built for 22 kHz files | Usable for presence, with a real false-alarm rate. Needs a weights file the installed application does not carry |
| Pitch tracker (`gmv.pitch`) | Up to four note candidates per frame in the guitar register; known octave and fifth confusions, distorted chords unreliable | Not used: its own documentation rules it out as polyphonic harmony or melody analysis | Chroma (below) answers the harmonic questions without naming notes |
| Harmonic/percussive balance (`gmv.advanced`) | Median-filter split of the spectrogram, every two seconds, off by default | The same method at a finer hop is the basis of the research front end | Sound method; reused |
| Tempo (`FeatureFrame.tempo_bpm`, `tempo_confidence`) | Strongest autocorrelation of 12 s of flux, 60–200 BPM | Phase A already found octave errors | Superseded here by a pulse estimate that reports its own ambiguity |
| Beat tracker (`gmv.beat`, opt-in) | Causal beat phase and tempo; development result F1 0.69 on MDB Drums | Not re-evaluated; the research needs only the pulse period and how sure it is | Remains the candidate for live beat phase |
| Onset events | Whole-mix attacks | Used by Phase A | The rhythm prototypes use a percussive onset signal instead, which keeps accents |
| Instrument activity (drums, bass, guitar analyzers) | Energy in a region, not instrument-specific on their own (docs/INSTRUMENT_LAYERS.md) | Not used | The low-band onset signal here measures the same region without claiming a kick |
| Sections | Phase A's diagnostics find section boundaries | Used for the track-change question (section 9) | – |

## 3. The experimental measurements

All are computed offline from one spectrogram of the recording (22.05 kHz, 93 ms windows,
43 frames a second; `research/frontend.py`), split into a harmonic and a percussive part by
median filtering. Each returns a value **and a confidence**, or **unknown**. Unknown is
never zero: no reliable pitch is not "no melody", no clear pulse is not "an irregular
rhythm", no model is not "no voice".

*Measured* values state what is in the signal. *Inferred* values interpret it (a key, a
chord change, a voice, a tempo) and are wrong wherever the interpretation does not apply.

### Harmony and tonality (`research/tonal.py`, read every 12 s)

Pitch classes (chroma) are taken from the harmonic part between 190 Hz and 2.1 kHz, on a
grid of thirds of a semitone so that the recording's own tuning is found first, and
averaged over 0.5 s.

| Measurement | Kind | Unit | What it is | Usual range |
|---|---|---|---|---|
| `harmonic_share` | measured | 0..1 | share of the power that is steady in time rather than struck | 0.2 – 0.9 |
| `pitch_class_focus` | measured | 0..1 | 1 − entropy of the passage's pitch classes: 0 all twelve equally, 1 a single one | 0.02 – 0.4 |
| `pitch_classes_at_once` | measured | 1..12 | effective number of pitch classes sounding together (exp of the frame entropy) | 4 – 11 |
| `harmonic_movement` | measured | distance/s | how fast the pitch-class content moves: distance per second in Harte's six-dimensional tonal-centroid space, between the chroma 0.37 s before and after | 0.1 – 1.5 |
| `tonal_stability` | measured | 0..1 | mean cosine similarity of each moment's pitch classes to the passage's own | 0.6 – 0.98 |
| `key_clarity` | inferred | −1..1 | correlation of the passage's pitch classes with the best of the 24 Krumhansl major and minor key profiles | 0.3 – 0.9 |
| `major_lean` | inferred | −1..1 | best major fit minus best minor fit; confidence falls to zero when no key fits | −0.3 – 0.3 |
| `chord_change_rate` | inferred | 1/s | peaks of harmonic movement per second; half confidence by construction | 0.2 – 2 |

Confidence: `1 − exp(−seconds/8)`, times `min(1, harmonic_share/0.3)` for everything read
from pitch classes (with little tonal content there is little to read).

No chord or key is named anywhere. `key_clarity` says how key-like the pitch content is,
not which key it is.

### The centre lead line and the voice (`research/lead.py`, read every 6 s)

No new signal processing: these summarise the existing vocal analyzer's rows, and the
activity model's judgement when its weights file is given.

| Measurement | Kind | Unit | What it is |
|---|---|---|---|
| `centre_line_level` | measured | 0..1 | mean level of the centre in the voice's range |
| `pitched_share` | measured | 0..1 | share of the music in which the centre has one clear pitch |
| `pitch_clarity` | measured | 0..1 | how much of the centre's peaks the best harmonic series explains, while the centre is loud |
| `line_range_st` | measured | semitones | spread of that pitch, 10th to 90th percentile |
| `line_motion_st_s` | measured | semitones/s | how fast the pitch moves inside pitched stretches |
| `held_share` | measured | 0..1 | share of pitched time on notes held within half a semitone for 0.3 s |
| `vocal_presence` | inferred | 0..1 | share of the music in which the activity model reports a voice (unknown without the model) |
| `vocal_pitched_share` | inferred | 0..1 | share of those frames with a clear centre pitch: the candidate for sung against unpitched delivery (unknown without the model) |
| `phrase_s` | inferred | s | median phrase length (unknown without the model) |

### Rhythmic feel (`research/rhythm.py`, read every 12 s)

The onset signal is the rise in magnitude of the percussive part (not its logarithm:
accents are what tell a beat from its subdivisions). Its autocorrelation gives the pulse
period, the best one between 50 and 220 a minute with a broad preference for 110; the
signal is then folded onto one beat in 24 steps.

| Measurement | Kind | Unit | What it is |
|---|---|---|---|
| `pulse_clarity` | measured | 0..1 | autocorrelation at the pulse period |
| `tempo_bpm` | inferred | BPM | the pulse as a tempo. **Half and double are equally valid readings**; summarised as a median, never a mean |
| `tempo_ambiguity` | measured | 0..1+ | strength of half or double the pulse relative to the pulse; near 1, the level is undecided |
| `on_beat_share` | measured | 0..1 | share of the onset signal within a twelfth of a beat of the pulse (0.17 if spread evenly) |
| `events_per_beat` | measured | count | onset peaks per pulse |
| `offbeat_weight` | measured | 0..1 | of the signal on the beat and half-way between, the share half-way |
| `triplet_share` | measured | 0..1 | of the signal on the subdivisions, the share on thirds rather than quarters; unknown when under a fifth of the signal is on subdivisions |
| `low_events_per_beat` | measured | count | onset peaks under 150 Hz per pulse |
| `bar_repetition` | measured | 0..1 | autocorrelation at four pulses |
| `accent_alternation` | measured | −1..1 | whether strong and weak pulses alternate, scaled by how much pulses differ at all |

Confidence: `1 − exp(−seconds/8)`, times how much of the percussive part is newly struck
(a held pad has no pulse to find: unknown), times `min(1, pulse_clarity/0.25)`. Below a
pulse clarity of 0.25 every beat-relative value is unknown.

### Windows and summaries

Families do not share a window: 6 s for the lead line, 12 s for harmony and rhythm, blocks
overlapping by half. A recording's value is the mean of its blocks weighted by confidence
(the median for a tempo); its confidence is the mean over all blocks, so blocks that could
say nothing count against it. Section 8 looks at which scale each measurement belongs to.

## 4. What was rejected or deferred, and why

- **Contour of the lead line** (`line_motion_st_s`, `held_share`, `line_range_st`):
  rejected. In the corpus the pitch moves 35 to 50 semitones a second for every dense
  recording and almost no note is ever "held": that is the tracker flickering between
  candidates, not melody. Between-recordings share 0.18 to 0.35 (section 8).
- **Vibrato, pitch stability, formants**: not attempted. They need a pitch track of the
  voice that is right most of the time, and the one available is within a semitone 63% of
  the time on ordinary mixes and worse in dense ones. Formant measurement on a full mix has
  no defensible basis without isolating the voice.
- **Clean against screamed delivery**: investigated, not achieved (section 7).
- **Chord and key naming, major or minor**: deliberately not done. `major_lean` is reported
  for study only; its confidence is low (0.51) and it separates nothing.
- **Phrase length**: the existing boundaries are stated too rarely to evaluate.
- **Swing**: `triplet_share` measures thirds against quarters. Swing as unequal eighths is
  not separately measured; nothing in the corpus let it be checked.
- **Syncopation in the musical sense** (accents against the metre) needs bars and
  downbeats, which nothing here finds. `offbeat_weight` is weight between pulses, whatever
  level the pulse is.
- **Melodic repetition and variation across passages**: deferred; it needs a melody
  representation that does not exist yet.
- **Source separation** would address most of the above. It was not added: GMV's own
  earlier comparison (docs/MILESTONES.md) found that a separation model needs 8 CPU threads
  and answers 1.4 to 3.4 s late, so it is not a live path, and its weights' licences were
  not cleared for shipping. It remains the
  honest answer to "how would one really measure the voice", and it is offline-only.

## 5. Corpus and split

Kind of evidence throughout: **offline analysis of real recordings and of research
datasets on CachyOS**. Not live playback, and nobody listened.

**The owner's registered corpus**: 197 recordings of 38 artist groups, plus the two
reference tracks (one in a registered group, one a group of its own): 199 recordings, 39
groups. Titles and artists stay in ignored local files; reports use ids such as `g07-03`.

**Group-disjoint split.** Groups, not recordings, are divided by a hash of the group id:
17 groups (104 recordings) are *calibration*, 22 other groups (95 recordings) are
*evaluation*. No artist is on both sides. Calibration sets every scale (means and standard
deviations for standard scores, and the regression that predicts a measurement from
Phase A). Every figure reported is on evaluation. Both reference tracks' groups were
forced into evaluation, so **neither reference track influenced any scale**.

Nothing is fitted: there are no learned weights and no anchors in the experimental
measurements. The constants in the definitions (window lengths, frequency ranges,
thresholds for "unknown") were set from signal-processing reasoning and the generated test
signals before the corpus was run, with two exceptions made after the first test run on
generated clicks, neither involving a reference track: the onset signal was changed from
log-compressed to plain magnitude, and a held sound was made "unknown" for rhythm.

Composition by metadata tags (calibration / evaluation): metalcore, deathcore and
post-hardcore 43 / 31; other metal 13 / 20; progressive and classic rock 20 / 0; other
rock, alternative, indie 0 / 11; punk and pop-punk 5 / 1; ambient, chillout and game
soundtracks 22 / 16; untagged 1 / 16.

Limits of the corpus: it is one person's library, heavy on metal. It has little
electronic dance music, almost no jazz, classical, hip-hop or folk, and few sparse acoustic
recordings. The split is unbalanced by genre because whole artists move together. Results
on rhythm in particular should not be assumed to hold for music with swing or complex
metres, of which there is very little.

**Labels.** Two, both from file metadata and neither verified by listening:
`vocal` (the inventory's vocal or instrumental field: 113 and 79) and `harsh_genre` (among
vocal recordings with genre tags, whether the tags include metalcore, deathcore,
post-hardcore, hardcore or death metal: 78 and 34). The second is a stand-in for "likely
to contain screamed vocals". It is a statement about genre tags, not about any recording,
and since those genres also differ in production it cannot isolate the voice.

**Research datasets**, for checks against known signals (section 6): MUSDB18-HQ (150
songs with separate vocal, drum, bass and other stems; 139 used, 11 shorter than the
excerpt; seconds 30 to 120 of each) and MDB Drums (23 mixes with annotated beats). The
activity model was trained on MUSDB's training songs, so its figures are given on the 50
test songs only.

## 6. Checks against isolated stems and annotated beats

A measurement made on a full mix claims to describe one part of it. Where the part exists
on its own, the claim was tested. The stems are references for what is in the signal, not
for how anything is perceived.

**Harmony: the mix against the same mix without its drums** (rank correlation over 139
songs). If drums disturbed these measurements the two would disagree.

| | |
|---|---:|
| `tonal_stability` | 0.97 |
| `harmonic_movement` | 0.95 |
| `key_clarity` | 0.95 |
| `major_lean` | 0.94 |
| `pitch_class_focus`, `pitch_classes_at_once` | 0.92 |
| `chord_change_rate` | 0.91 |
| `harmonic_share` | 0.76 |

The pitch-class measurements describe the harmony whether or not drums play.
`harmonic_share` is lower by design: removing the drums changes the share itself.

**Rhythm: the mix against the drum stem alone.**

| | |
|---|---:|
| `pulse_clarity` | 0.74 |
| `offbeat_weight`, `low_events_per_beat` | 0.73 |
| `events_per_beat`, `bar_repetition` | 0.72 |
| `on_beat_share` | 0.67 |
| `triplet_share` | 0.64 |
| `tempo_bpm` | 0.61 |
| `accent_alternation` | 0.59 |
| `tempo_ambiguity` | 0.56 |

The mix and its drums agree on the tempo within 6% for 83% of songs and disagree by a
factor of two for 9%. Against annotated beats (MDB Drums): the pulse is the annotated beat
in 18 of 23 mixes, half of it in 2, a third or two thirds of it in 3. Median
`tempo_ambiguity` there is 0.98: the measurement that says "the level is undecided" is
high, as it should be. So the rhythm of the mix is mostly the rhythm of the drums, the
pulse is right about four times in five, and when it is wrong it is wrong by a metrical
factor.

**The voice: the mix against the vocal stem.**

| | median | quartiles |
|---|---:|---|
| Share of the excerpt in which a voice sounds | 0.79 | 0.68 – 0.91 |
| `centre_line_level` follows when the voice sounds (AUC inside a song) | 0.84 | 0.78 – 0.89 |
| `pitch_confidence` follows when the voice sounds (AUC) | 0.58 | 0.52 – 0.64 |
| When the mix has a centre pitch, a voice is sounding | 0.84 | 0.73 – 0.92 |
| Centre pitch within a semitone of the voice's own pitch | 0.63 | 0.52 – 0.74 |
| ...an octave off | 0.03 | 0.02 – 0.05 |
| Activity model finds vocal frames (50 test songs) | 0.80 | 0.67 – 0.92 |
| Activity model fires where there is no voice (50 test songs) | 0.18 | 0.06 – 0.37 |

**Delivery.** The reference is the share of the vocal stem that has a pitch at all (the
same analyzer run on the stem alone; median 0.80, from 0.50 to 0.91 over the songs): high
for sung notes, lower for breathy, spoken or unpitched delivery. Rank correlation of the
mix-based estimates with it:

| Estimate from the mix | rank correlation |
|---|---:|
| `pitched_share`, whole excerpt | 0.61 |
| pitched share while the stem says a voice sounds (an oracle; not available in practice) | 0.64 |
| `vocal_pitched_share` (gated by the activity model; 49 test songs) | 0.69 |

So on these mixes, mostly sung pop and rock, how much of the voice is pitched can be read
from the mix moderately well. Section 7 shows where that stops.

## 7. Results on the owner's corpus (evaluation groups only)

Columns: how often the value could be stated; mean confidence; how much of it the seven
Phase A dimensions already predict (R², regression fitted on calibration groups; near 0 is
new information, negative means the calibration fit does not carry over at all); the share
of its block-to-block variation that lies between recordings (a property of the recording
when high); and the same between artists.

| Measurement | stated | confidence | in Phase A already (R²) | between recordings | between artists |
|---|---:|---:|---:|---:|---:|
| `tonal.harmonic_share` | 100% | 0.77 | 0.45 | 0.59 | 0.55 |
| `tonal.pitch_class_focus` | 100% | 0.77 | 0.45 | 0.50 | 0.46 |
| `tonal.pitch_classes_at_once` | 100% | 0.77 | 0.70 | 0.66 | 0.61 |
| `tonal.harmonic_movement` | 100% | 0.77 | **0.16** | 0.58 | 0.40 |
| `tonal.tonal_stability` | 100% | 0.77 | 0.48 | 0.61 | 0.53 |
| `tonal.key_clarity` | 100% | 0.77 | **0.10** | 0.51 | 0.38 |
| `tonal.major_lean` | 100% | 0.51 | −0.14 | 0.36 | 0.39 |
| `tonal.chord_change_rate` | 100% | 0.39 | 0.13 | 0.25 | 0.37 |
| `lead.centre_line_level` | 100% | 0.77 | 0.42 | 0.46 | 0.66 |
| `lead.pitched_share` | 100% | 0.77 | 0.38 | 0.38 | 0.46 |
| `lead.pitch_clarity` | 100% | 0.70 | 0.51 | 0.49 | 0.61 |
| `lead.line_range_st` | 100% | 0.49 | 0.14 | 0.18 | 0.42 |
| `lead.line_motion_st_s` | 100% | 0.49 | −0.05 | 0.35 | 0.33 |
| `lead.held_share` | 100% | 0.49 | −0.36 | 0.18 | 0.03 |
| `lead.vocal_presence` | 94% | 0.72 | 0.35 | 0.48 | 0.67 |
| `lead.vocal_pitched_share` | 85% | 0.29 | 0.10 | 0.40 | 0.30 |
| `lead.phrase_s` | 76% | 0.02 | −0.29 | 0.21 | 0.00 |
| `rhythm.pulse_clarity` | 100% | 0.77 | 0.46 | 0.24 | 0.12 |
| `rhythm.tempo_bpm` | 100% | 0.76 | −0.28 | **0.03** | 0.03 |
| `rhythm.tempo_ambiguity` | 100% | 0.77 | −0.14 | 0.19 | 0.04 |
| `rhythm.on_beat_share` | 100% | 0.73 | 0.36 | **0.65** | 0.65 |
| `rhythm.events_per_beat` | 100% | 0.73 | 0.26 | 0.35 | 0.44 |
| `rhythm.offbeat_weight` | 100% | 0.73 | 0.11 | 0.46 | 0.26 |
| `rhythm.triplet_share` | 100% | 0.71 | −0.04 | 0.39 | 0.00 |
| `rhythm.low_events_per_beat` | 100% | 0.73 | −0.04 | 0.32 | 0.32 |
| `rhythm.bar_repetition` | 100% | 0.73 | 0.35 | 0.40 | 0.21 |
| `rhythm.accent_alternation` | 100% | 0.73 | −0.12 | 0.29 | 0.08 |

For comparison, the Phase A dimensions' between-artists shares on the same recordings:
force 0.87, brightness 0.83, roughness 0.73, density 0.73, regularity 0.28, dynamics 0.19,
pace 0.03.

Reading it:

- **`harmonic_movement` and `key_clarity` are the clearest additions**: always available,
  hardly predicted by Phase A (0.16 and 0.10), and a property of the recording (over half
  their variation is between recordings, at 12 s).
- `pitch_classes_at_once` and `tonal_stability` largely restate density (R² 0.70 and 0.48;
  a dense mix occupies every pitch class). `harmonic_share` is the inverse of Phase A's
  roughness (r = −0.86). They duplicate.
- **`on_beat_share`** is the most stable rhythm measurement (0.65 between recordings), and
  two thirds of it is not in Phase A.
- **The tempo is not a property of a 12 s passage**: 3% of its variation is between
  recordings. From one block to the next it flips between metrical levels (reference
  track B reads 65, 87 and 130 in different passages: one pulse at three levels). Every
  "per beat" value inherits that.
- The lead-line contour values are unstable and are rejected (section 4).

**Does adding a family help find music by the same artist?** For each evaluation
recording, is its nearest neighbour by the same artist (87 recordings with a group mate;
chance 8%)? Music by one artist is not all alike, so this is a weak indication, not a goal.

| Measurements | nearest is the same artist |
|---|---:|
| Phase A | 40% |
| Phase A + harmony | **49%** |
| Phase A + lead line | 39% |
| Phase A + rhythm | 28% |
| Phase A + all | 41% |
| harmony alone / lead alone / rhythm alone | 32% / 32% / 18% |

Harmony adds identity information. Rhythm, as measured here, removes it: it varies more
from song to song and passage to passage than from artist to artist, which is what the
table above also says. That makes it a candidate for Musical Expression, not Identity.

### The voice: presence yes, delivery no

Labels are from metadata (section 5). AUC is the chance that a labelled-1 recording scores
above a labelled-0 one (0.5: no relation; values under 0.5 mean the relation runs the other
way). "Beyond Phase A" is the same for the part of the measurement that Phase A does not
predict.

| | `vocal` (vocal against instrumental) | beyond Phase A | `harsh_genre` | beyond Phase A |
|---|---:|---:|---:|---:|
| `phase_a.force` | 0.89 | | 0.90 | |
| `phase_a.density` | 0.86 | | 0.90 | |
| `lead.centre_line_level` | 0.88 | 0.82 | 0.91 | 0.86 |
| `lead.vocal_presence` | 0.83 | 0.59 | 0.92 | 0.76 |
| `lead.pitched_share` | 0.49 | 0.77 | 0.42 | 0.62 |
| `lead.vocal_pitched_share` | 0.70 | 0.67 | 0.25 | 0.26 |
| `lead.pitch_clarity` | 0.32 | 0.50 | 0.05 | 0.07 |

- **Presence.** In this library the instrumental recordings are mostly quiet soundtracks,
  so Phase A's force alone "detects vocals" (0.89): the label is confounded with
  production. What matters is the part beyond Phase A, where `centre_line_level` (0.82)
  holds up and the model's `vocal_presence` (0.59) mostly does not. Together with the stem
  check (finds 80% of vocal frames, fires on 18% of others), presence is estimable but
  noisy.
- **Delivery.** `vocal_pitched_share` is lower for the harsh-genre recordings (AUC 0.25,
  that is 0.75 the other way), in the expected direction, and `pitch_clarity` separates the
  genres almost perfectly (0.05). But `pitch_clarity` is mostly density (r = −0.78 with
  Phase A's density): a dense mix has less clear pitch whatever the voice does. And on the
  reference pair the measurement meant to carry delivery gives **0.58 for the screamed
  track and 0.59 for the sung one**. In a dense metal mix the centre is full of pitched,
  distorted guitar, so "the centre has a pitch" says nothing about the voice. The cue
  works where it is least needed.

**Conclusion on the voice:** GMV cannot tell sung from screamed delivery from the mixed
signal with its existing analysis, and no measurement here should be presented as doing so.

### Comparisons the brief asked for

1. **Similar production, different musical personality: the reference tracks.** Section 10.
2. **Similar texture, different vocal behaviour.** Not demonstrated; see above.
3. **Different texture, similar musical behaviour.** Shown two ways. Across the corpus, a
   metalcore recording and a chill-out electronic one that are 1.8 standard deviations
   apart in Phase A (brightness 0.78 against 0.00, pace 0.39 against 1.00) are 0.6 apart on
   harmony and rhythm (harmonic movement 0.16 and 0.18, key clarity 0.58 and 0.61). And
   directly, by changing the production of the same recording (section 11): making it
   duller moves Phase A's brightness by 2.7 standard deviations and the harmony
   measurements by 0.1 to 0.35.
4. **Similar rhythmic density, different rhythmic feel.** Two evaluation recordings with
   the same Phase A pace (0.39 and 0.37): a metalcore track with 3.7 events per beat, 18%
   of its onset signal on the beat and bar repetition 0.40, and a game-soundtrack piece
   with 1.7 events per beat, 39% on the beat and bar repetition 0.76. Phase A calls them
   equally busy; the rhythm measurements call one a dense stream and the other a clear,
   repeating pattern.
5. **Similar harmony, different melodic behaviour.** The pair found (two pieces by the same
   soundtrack composer, alike on harmony, apart on the lead line) differs on `pitched_share`
   (0.61 and 0.39) and on contour values that section 4 rejects. This comparison is **not
   supported**: there is no reliable melody measurement to make it with.
6. **Within-track evolution.** Section 8 and the reference tracks in section 10.
7. **Stable identity under ordinary intensity changes.** From block to block inside a
   recording, the measurements correlate with `MusicalIntensity` at 0.2 to 0.4 in absolute
   value on average, with no consistent sign (mean r between −0.27 and +0.30): they do not
   simply follow intensity.

## 8. Time scales

Share of each measurement's variation that lies between recordings when its blocks are
averaged 1, 2, 4 or 8 at a time (for 12 s blocks overlapping by half: about 12, 18, 30 and
54 s of music).

| | 1 block | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| `tonal.harmonic_movement` | 0.58 | 0.60 | 0.62 | 0.70 |
| `tonal.key_clarity` | 0.51 | 0.56 | 0.63 | 0.73 |
| `tonal.pitch_classes_at_once` | 0.66 | 0.69 | 0.73 | 0.78 |
| `rhythm.on_beat_share` | 0.65 | 0.69 | 0.74 | 0.80 |
| `rhythm.events_per_beat` | 0.35 | 0.40 | 0.51 | 0.66 |
| `rhythm.pulse_clarity` | 0.24 | 0.28 | 0.34 | 0.45 |
| `rhythm.tempo_bpm` | 0.03 | 0.05 | 0.09 | 0.51 |
| `lead.centre_line_level` (6 s blocks) | 0.46 | 0.48 | 0.53 | 0.60 |
| `lead.vocal_presence` (6 s blocks) | 0.48 | 0.50 | 0.54 | 0.60 |
| `lead.pitched_share` (6 s blocks) | 0.38 | 0.40 | 0.45 | 0.50 |

- **Harmony belongs to the long scale.** It is already a property of the recording at
  12 s and keeps settling to a minute: it fits Phase A's medium (20 s) and long (90 s)
  windows.
- **Rhythm is two things.** `on_beat_share` is identity-like. Pulse clarity, events per
  beat and above all the tempo are properties of the passage: half or more of their
  variation is inside a recording even at a minute. A tempo should never be read from one
  12 s passage.
- **The voice is a property of the moment.** Whether the centre carries a line changes
  phrase by phrase; over a whole recording it is half identity at most. It needs seconds,
  not tens of seconds.

That is the division the brief asks about: **Musical Identity** (slow: Phase A's texture,
harmony, how concentrated the rhythm is) and **Musical Expression** (now: vocal activity,
pulse and its clarity, how busy this passage is). They should stay two systems.

## 9. Track transitions

Phase A's 90 s window carries one track's character into the next. Could a change of
track be detected from the music alone? Tested with the Phase A short-window profile: the
mean change per dimension between the next 10 s and the last 30 s. Inside each of 94
evaluation recordings the largest such change is what a detector must not fire on; across
a boundary between two recordings (in a fixed shuffled order) it is what it must catch.

| Recordings that would be reset by mistake | Boundaries caught |
|---:|---:|
| 2% | 15% |
| 10% | 31% |
| 25% | 49% |

Largest change inside a recording: median 0.21. Change across a boundary: median 0.29.
**They overlap almost completely.** An ordinary breakdown, intro or quiet passage changes
the sound as much as a new track does. A reset triggered by change in the music would
either miss most track changes or destroy the identity of a quarter of all recordings.

Recommended strategy, in order of trust:

1. **An explicit signal** where one exists: a new capture segment (already a full reset
   in Phase A), and in future the player's own track-change notification (MPRIS on Linux,
   the system media session on Windows). Neither is read by GMV today; it is the only
   reliable source and would be new work.
2. **Silence**: Phase A already keeps the profile but only a quarter of its weight after
   three seconds of silence. Keep that. Fails for gapless albums and crossfades, and
   loosens (harmlessly) on a silent pause inside a track.
3. **Change in the music: as evidence, never as a trigger.** A large change may shorten
   the window for a while (the same loosening as after silence), so the identity follows
   faster if the change persists, and returns if it was a breakdown. It must never reset.

Failure modes that remain: gapless playback and crossfades between similar tracks are not
detected by 2 or 3 and drift over about a minute and a half; a DJ mix has no boundaries
at all. Nothing was implemented; this belongs to a later phase.

## 10. The two reference tracks

Both analysed with every definition already fixed. Distance between them in standard
deviations of the calibration recordings (root mean square over the measurements used),
and the share of all pairs of evaluation recordings that are closer together than these
two:

| Measurements | distance | further apart than |
|---|---:|---:|
| Phase A | 0.36 | 4% of pairs |
| harmony | 1.38 | 70% of pairs |
| lead line | 0.74 | 17% of pairs |
| rhythm | 0.71 | 10% of pairs |
| Phase A + harmony | 1.04 | 48% of pairs |
| Phase A + all | 0.88 | 18% of pairs |

On Phase A alone they are among the closest 4% of pairs. With the harmony measurements
they are an ordinary distance apart. **They are never far apart**, on anything.

Where they differ most:

| | A (metalcore) | B (described as mellow) | further apart than |
|---|---:|---:|---:|
| `tonal.harmonic_movement` | 0.12 | 0.25 | 82% of pairs |
| `tonal.key_clarity` | 0.51 | 0.66 | 73% of pairs |
| `tonal.chord_change_rate` (inferred, low confidence) | 1.29 | 0.84 | 95% of pairs |
| `rhythm.low_events_per_beat` | 3.06 | 1.68 | 85% of pairs |
| `lead.vocal_presence` (model) | 0.78 | 0.51 | 46% of pairs |
| `lead.vocal_pitched_share` | 0.58 | 0.59 | 7% of pairs |
| `rhythm.on_beat_share` | 0.17 | 0.18 | 23% of pairs |
| `rhythm.pulse_clarity` | 0.51 | 0.47 | 31% of pairs |

In words: **A sits on its harmony and B moves through chords** (twice the harmonic
movement, and B fits a key better); **A's low end is struck almost twice as often**; and
the model hears a voice through nearly all of A but only half of B, with clear gaps. Those
are real, interpretable musical differences that Phase A does not contain. The delivery of
the voice, the one a listener would name first, is not among them.

One derived figure, **found after looking at this pair and therefore not a test**:
low-band onsets per *second* (`low_events_per_beat` × tempo / 60, which cancels the
unreliable choice of metrical level) is 5.4 for A and 2.9 for B, further apart than 90% of
evaluation pairs; across the evaluation recordings it runs from 2.0 to 5.4 (5th to 95th
percentile) and correlates at most 0.45 with any Phase A dimension. It is the obvious
measurement to define properly and test on new material.

Over time (12 s blocks): A's harmonic movement stays between 0.07 and 0.14 until its
ending; B's runs from 0.11 in its opening to 0.57 in a late passage. The model's vocal
presence in A is 1.0 for almost every block from 6 s to 144 s; in B it is 0 for the first
24 s, then alternates between 0.9 and 0.1 with the song's sections. B's sections are
visible in the vocal and harmonic measurements; A is one continuous statement. That
matches Phase A's finding that the two differ most in how they evolve.

## 11. Level and other robustness

**Level.** 50 recordings (every fourth of the table) analysed 10 and 20 dB quieter. Change
of each whole-recording value in calibration standard deviations: the median is at most
0.014 for every measurement at both levels; the 90th percentile at most 0.10 at 20 dB;
the largest single change 0.35 (lead-line range), apart from `phrase_s` (1.6, one
recording; rejected anyway). No value became unknown. Every experimental measurement is
independent of level in practice. As in Phase A, music under the −66 dBFS silence threshold
is not heard at all, so a very quiet capture loses its quietest passages.

**Changes that leave the music as it was.** 19 evaluation recordings (first 90 s of each),
analysed again after each change. Median move in calibration standard deviations (0.1 is
negligible; 1 is the difference between two ordinary recordings):

| | mono | duller −3 dB/oct | brighter +3 dB/oct | band-limited 7 kHz | saturated | half sample rate |
|---|---:|---:|---:|---:|---:|---:|
| `phase_a.brightness` | 0.07 | **2.67** | **1.00** | 0.58 | 0.07 | 0.03 |
| `phase_a.force` | 0.39 | 0.27 | 0.87 | 0.06 | 0.64 | 0.03 |
| `phase_a.density` | 0.27 | 0.81 | 0.20 | 0.47 | 0.30 | 0.01 |
| `tonal.harmonic_movement` | 0.15 | 0.09 | 0.08 | 0.00 | 0.20 | 0.00 |
| `tonal.key_clarity` | 0.12 | 0.35 | 0.13 | 0.00 | 0.05 | 0.00 |
| `tonal.tonal_stability` | 0.13 | 0.13 | 0.11 | 0.00 | 0.24 | 0.00 |
| `tonal.pitch_class_focus` | 0.05 | 0.20 | 0.14 | 0.00 | 0.13 | 0.00 |
| `tonal.harmonic_share` | 0.11 | 0.51 | **1.28** | 0.04 | 0.10 | 0.00 |
| `tonal.chord_change_rate` | 0.48 | 0.36 | 0.26 | 0.00 | 0.48 | 0.00 |
| `rhythm.on_beat_share` | 0.07 | 0.14 | 0.10 | 0.05 | 0.25 | 0.00 |
| `rhythm.events_per_beat` | 0.10 | 0.34 | 0.19 | 0.09 | 0.41 | 0.00 |
| `rhythm.pulse_clarity` | 0.13 | 0.41 | 0.64 | 0.11 | 0.41 | 0.00 |
| `rhythm.tempo_ambiguity` | 0.24 | 0.64 | 0.34 | 0.10 | 0.32 | 0.00 |
| `lead.pitched_share` | **0.86** | 0.57 | 0.39 | 0.03 | 0.17 | 0.02 |
| `lead.line_motion_st_s` | **2.05** | 0.27 | 0.28 | 0.05 | 0.17 | 0.02 |
| `lead.centre_line_level` | 0.36 | 0.22 | 0.23 | 0.01 | 0.18 | 0.01 |

- The Phase A profile moves with the production, as it is meant to: it measures the sound.
- **The pitch-class measurements do not**: harmonic movement, key clarity, stability and
  focus move by 0.35 at most under every change. They describe the music, not the mix.
- `harmonic_share` and `chord_change_rate` are production-dependent and should not be used
  as musical measurements.
- **The lead-line measurements depend on a stereo image.** In mono there is no centre to
  extract and they change by up to two standard deviations. The tool warns (`mono`).
- Rhythm sits in between: stable under mono and band-limiting, moved by tone and
  saturation by a quarter to two thirds of a standard deviation.
- Band-limiting at 7 kHz (roughly what a low-bitrate file does) and halving the sample
  rate hardly move anything experimental. The corpus itself is nearly all lossy files, so
  lossy against lossless was not compared directly.

Where a measurement becomes unreliable rather than merely different: harmony in passages
with little tonal content (confidence falls with `harmonic_share`); rhythm with no clear
pulse or nothing struck (unknown); lead line in mono, in dense distorted mixes, and for
22 kHz files with the activity model (the model cannot be built at that rate; `vocal_*`
are then unknown, which is what happened for all 12 recordings of one artist group).

## 12. Cost

Offline, on the development machine (one core), for a 257 s recording:

| | seconds | |
|---|---:|---|
| Research spectrogram and harmonic/percussive split | 3.5 | 0.32 ms per 23 ms frame |
| The existing analysis pass with the vocal analyzer on | 10.5 | (9.0 without it) |
| Harmony, rhythm and lead measurements together | 0.2 | |
| **Total** | **14.2** | 3.3 s per minute of music; 18 times real time |

Peak memory 760 MB for that recording, of which 227 MB are the spectrogram arrays held at
once. That is an offline design: the whole recording is in memory.

**Live cost is not measured, because nothing here runs live.** What can be said: the
spectrogram and split cost about 1.4% of one core at this frame rate if computed
incrementally, but the median filter along time looks 0.2 s ahead, so a causal version
would either lag by that much or use a one-sided filter with a different behaviour that
would have to be validated again. The pitch-class measurements need only the harmonic
magnitudes and a 12-bin fold; they are cheap. A production design should be measured, not
extrapolated from these figures.

## 13. What worked, what failed, what remains unmeasured

**Worked**

- Harmonic movement and key clarity: new, robust, stable, interpretable.
- On-beat share: a stable description of how concentrated a rhythm is on its pulse.
- Low-band onset activity: separates heavy, kick-driven music from the rest, largely
  independently of Phase A (as a per-second figure it still needs a proper definition).
- Vocal presence, with the existing model: usable, noisy.
- Reporting uncertainty: unknown is kept apart from zero throughout, and the pulse states
  its own ambiguity.

**Failed**

- Vocal delivery (sung against screamed): not measurable from the mix with existing
  analysis. Fails in dense mixes, which is where it is needed.
- Lead-line contour, vibrato, pitch stability: the pitch track is not good enough.
- Tempo as a number: unreliable per passage; half/double ambiguity is inherent.
- Telling a track change from an arrangement change by the music alone.
- `harmonic_share`, `pitch_classes_at_once`, `tonal_stability`: valid but duplicate Phase A.

**Remains unmeasured**

- Anything about the voice beyond "probably present".
- Melody: contour, range, repetition.
- Metre, downbeats, swing, true syncopation.
- Chords, keys and mode as named things.
- Whether any of this matches what a person hears. Nobody listened. Every claim about
  "musical character" here rests on measurements agreeing with stems, with metadata, and
  with what the recordings are known to be.

## 14. Diagnostics

```
gmv character research analyze PATH...                 the experimental measurements beside Phase A
gmv character research corpus --table FILE [--gain-db -10 -20] [--pair ID ID]
gmv character research stems --musdb DIR --mdb-drums DIR
gmv character research robustness --table FILE --scales corpus.json
```

All take `--activity-model FILE` (for `vocal_*`), `--processes N` and `--out DIR` (default
`local-evidence/sonic-character/research`, ignored by Git). `gmv character analyze` works
as before.

- `analyze` writes `research.md` (Phase A and the experimental values per recording, with
  kind, confidence, how many blocks could state each, and warnings), one JSON per recording
  (every value, block by block) and one SVG chart of six measurements over time.
- `corpus` takes a table (`id`, `path`, `group`, optional `split`, optional `label:<name>`
  columns) and writes `corpus.md` and `corpus.json`: the split, every figure in sections 7
  to 10, and the level check. Reports and stored results use the ids only; file names
  appear nowhere in them.
- `stems` and `robustness` write the tables of sections 6 and 11.

## 15. Recommendation

**Sonic Identity** (slow, per recording), candidates on this evidence:

- the Phase A texture profile, unchanged;
- **`harmonic_movement`** and **`key_clarity`**, on the medium and long windows;
- **`on_beat_share`**;
- a properly defined **low-band onset rate per second**, once tested on material not used
  to find it.

**Musical Expression** (now, per passage), where these belong if they are used at all:

- **vocal presence** (activity model) and the centre-line level: phrase-scale, noisy,
  need the model's weights;
- **pulse clarity** and how busy the passage is;
- beat phase, from the existing opt-in beat tracker, which this phase did not re-evaluate.

**Optional or experimental only:** `tempo_bpm` (always with its ambiguity),
`offbeat_weight`, `triplet_share`, `bar_repetition`, `accent_alternation`, `major_lean`,
`chord_change_rate`.

**Dropped:** lead-line contour values, `phrase_s`, and the three harmony values that
duplicate Phase A.

**Is there enough to begin Phase B?** For a visual response to broad sonic texture and to
harmonic character, yes: those are measured, robust and understood. For the goal that
started this, a visibly different response to the two reference tracks because one is
aggressive and the other mellow, **no**. What separates them most for a listener is the
voice, and GMV cannot measure it. The measured differences that do exist (harmony moving
twice as fast, a low end struck half as often, a voice present half as much) are real and
could drive something, but they would make the two look somewhat different, not opposite.

**One more research step is justified, and it is narrow:** vocal delivery. Two routes,
neither tried here:

1. **Labels from the owner.** The owner already labels short excerpts in `gmv review`. A
   few hundred 12 s excerpts marked "sung / screamed or shouted / no voice" would allow a
   small classifier on existing features to be trained and, more importantly, honestly
   scored. Without labels nothing about delivery can be validated at all.
2. **Separation, offline.** Measure delivery on a separated vocal stem to learn what the
   answer looks like, then look for a cheap proxy that tracks it. Useful for research;
   not a path to a live feature.

Whichever is chosen, the two reference tracks should stay out of the fitting.

The decision on what to integrate, and how it looks, is the owner's.
