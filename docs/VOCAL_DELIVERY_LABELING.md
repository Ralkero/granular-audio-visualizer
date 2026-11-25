# Vocal delivery labelling (Sonic Character, Phase A3)

Phase A2 ([SONIC_CHARACTER_RESEARCH.md](SONIC_CHARACTER_RESEARCH.md)) found that GMV cannot
tell sung from screamed vocals with what it measures, and that nothing about it can even be
tested without labels made by a person. Phase A3 is a tool for making those labels: the
owner listens to ten-second excerpts and clicks what the voice does.

Research only. Nothing here changes the visualizer or the analysis, nothing is trained, and
no label is guessed by the software.

## For the owner: how to label

**Open the window.** In a terminal, run:

```
uv run --directory ~/granular-audio-visualizer/.claude/worktrees/research-vocal-delivery-labeling gmv character label
```

It works from any folder. The external drive that holds your music must be connected.

**What you see.** At the top, the excerpt's name (for example `VOCAL-0042`), the **song
title** of the recording it comes from, whether you have answered it, and how far you are:
labelled, skipped (no vocals / review later), still to label. On the right, the full
definitions of the five answers and your reference vocalists, and the list of all excerpts.

**Listen.** The excerpt plays as soon as it opens (ten seconds of a song; nothing else of
the song is played).

- **Play / Pause** starts and stops it.
- **Play again from the start** goes back to the beginning.
- **Repeat** plays it over and over until you pause.
- The bar shows where you are; drag it to jump.

**Answer.** Click the one technique that **dominates** the excerpt:

| Answer | In short (full definitions in the window) |
|---|---|
| **Cleans** | Conventional clean singing: a recognisable melodic performance without sustained harsh distortion. Loud or emotional singing is still Cleans if the technique stays clean |
| **High Screams** | Harsh delivery in a noticeably high register: sharp, piercing, shrill screams and shrieks |
| **Low Screams** | Harsh delivery in a lower or lower-middle register: thick, gritty screams or growls that are not distinctly guttural |
| **Gutturals** | Extremely low, throat-heavy vocals with a deep, wet, bubbling or cavernous texture |
| **Sing-Screams** | Harsh or distorted singing that keeps a clearly recognisable melodic line |

Label what you hear: the artist, the style and the song title never decide the answer.
Your answer is saved the moment you click ("Saved ✓" appears in green), and the next
excerpt opens.

**No voice in the excerpt?** Click **No vocals (skip)**. Instrumental passages are not
forced into a vocal answer; they are recorded as skipped for that reason.

**Several techniques with none dominating, or not sure?** Click **Skip / review later**.
Skipped excerpts stay in the list and are counted apart; **Next without an answer** brings
you back to them once everything else is done (excerpts to review before no-vocals ones).
You can also type a short **note** (optional); it is saved by itself.

**Change an answer.** Click **← Previous** (or click the excerpt in the list on the right)
and click a different answer. The new answer replaces the old one; the old one is kept in
the history.

**Stop and resume.** Close the window whenever you like. Nothing is lost. Run the same
command again and it opens where you stopped.

**Optional keys:** 1 to 5 answer, Space play/pause, R from the start, S review later,
N no vocals, ← → previous/next. The buttons do the same.

**Answers you gave with the first set of choices.** The answer choices changed on
2026-11-24 (from clean singing / screaming / shouting / mixed / no vocals to the five
above). Excerpts you had already answered keep that answer on record and show it ("earlier
answer, with the old choices: …", and "• earlier: …" in the list), but they count as still
to label until you give a new answer: an old answer is never translated into a new one.
The window says how many are waiting ("Earlier answers to redo"). Click them in the list to
redo them. Nothing was deleted.

**If an excerpt cannot be played** (the window says so in red), its drive is probably not
connected. Skip it; it will be there when the drive is back.

**Add more music** (optional): **Add recordings…** lets you choose more of your own audio
files and say who the artist is. Ten-second excerpts are taken from them and mixed into the
list. Your files are only read, never copied or changed.

**Back up your answers:** **Save a backup copy…** writes a zip of the excerpt list and your
answers into a folder you choose.

**Where your answers are kept:** only on this computer, in
`~/granular-audio-visualizer/.claude/worktrees/research-vocal-delivery-labeling/local-evidence/sonic-character/vocal-labeling/`.
Git ignores that folder; nothing in it is uploaded or published.

## What the window shows about an excerpt

The excerpt's anonymous name and, at the owner's request, the **song title** of its
recording, so there is no guessing which song is playing. Nothing else: no artist, album,
file name or folder, not how the excerpt was chosen, and not whether it belongs to the part
kept for the final test. An automated test opens every excerpt and checks that none of
these appears in any text, tooltip, list entry or title of the window, including when a
recording is missing.

**Where titles come from** (`titles.json`, beside the dataset; `gmv character label-titles`):
the owner's music inventory (the table `gmv evidence` built from the files' own tags),
matched by the recording's checksum; for a recording without metadata, a title the owner
gave; otherwise the title tag inside the file. Never a file name. In the first dataset, 82
recordings have their inventory title and one reference track, whose file carries no title
tag, has the title the owner gave it. Every inventory title was checked against a file with
the same checksum at the same path. Adding the titles did not change `dataset.json` (it is
byte-for-byte the same).

**Reference vocalists.** The owner's examples of vocalists and the techniques they use are
shown under the definitions, from `reference_vocalists.txt` beside the answers (kept out of
Git with the rest). They are reference points, not answers.

## How the excerpts were chosen

`gmv character prepare-labels` decides everything before anyone listens, and writes it to
`dataset.json`. It reads only the recordings named by the registered corpus (its manifest
and local path map) and the reference tracks named on the command line; nothing is searched
for, and nothing is copied.

1. **Recordings.** Every named recording is checked: it must exist, be readable audio, be at
   least 12 s long, and match the corpus's checksum where the corpus has one.
2. **Artist split, first.** An artist (a corpus group: one artist, or artists the corpus
   merged because they share recordings) is held out for the final evaluation if it owns a
   reference track, or if the first eight hex digits of
   `sha256("gmv-vocal-delivery-a3:" + artist)` divided by 2^32 are below 0.25. Everything
   else is development. No artist is on both sides. The rule needs nothing but the artist,
   so it never changes and adding music never moves anybody.
3. **Excerpts.** Artists take turns, one excerpt each per round, so none dominates (at most
   four excerpts per artist and two per recording). A reference track is offered first by
   its artist. Placements rotate through four methods:
   - **even**: one of a fixed set of positions through the recording (20%, 50%, 80%, 35%,
     65%);
   - **random**: a seeded random position;
   - **likely vocal** and **likely instrumental**: a random choice among the quarter of
     positions where the existing analysis (the activity model's vocal judgement, or the
     vocal analyzer's centre level where the model cannot run) thinks a voice most likely,
     or least likely.

   The analysis only proposes where to listen. It never decides or suggests a label, and the
   method is never shown in the window. Excerpts keep one second from either end of the
   recording, two seconds from each other, and are not offered if they average under
   −45 dBFS (near silence).
4. **Identifiers and order.** Excerpts are numbered `VOCAL-0001`… in a seeded random order,
   which is also the listening order; the number says nothing about the artist, recording,
   method or split.

Nothing balances the answers in advance: nobody knows them until the owner listens.

## What is kept, and where

In `local-evidence/sonic-character/vocal-labeling/` (ignored by Git):

| File | Holds |
|---|---|
| `dataset.json` | Every recording (path, artist, checksum, length, rate, channels, whether it is a reference track), every artist's side and why, every excerpt (identifier, recording, start and end in seconds, method, level), the listening order, the seed and the split rule |
| `labels.jsonl` | Every click, one line each, with its time: answers (with the number of the answer scheme they belong to), skips (with their reason: no vocals or review later), notes. Never rewritten, so a changed answer keeps the old one in the history, and a crash can at worst lose the line being written (it is then reported and left out). Answers saved before schemes were numbered belong to scheme 1 |
| `titles.json` | Each recording's song title and where it came from |
| `reference_vocalists.txt` | The owner's reference vocalists, shown in the window's guide |
| `session.json` | Which excerpt was open, and the two options |
| `labelling.lock` | Present while the window is open, so two windows cannot write at once |

Once anything has been labelled, `prepare-labels` refuses to run again in that folder: the
split and the excerpts are fixed. The only change allowed afterwards is adding recordings,
which appends new excerpts and new artists without touching anything that exists.

## Protecting the later evaluation

The split is fixed in `dataset.json` before the first label. When a classifier is built
later:

- develop and tune on excerpts of **development** artists only (the split is a field of
  every excerpt's artist; `store.export` writes it as a column so it can be filtered on);
- do not look at the evaluation labels while choosing features, thresholds or models;
- evaluate once, on the untouched evaluation artists, and report the class counts and
  per-class results;
- a class with only a handful of evaluation examples cannot be evaluated meaningfully;
  report it as such.

Both reference tracks, and every other recording by their two artists, are on the
evaluation side and must stay out of any fitting or tuning.

**Adding recordings later without contaminating the evaluation:** add them through the
window. A recording by a known artist joins that artist's side. A new artist gets the side
the hash rule gives it, so roughly a quarter of new artists are held out. Do not add music
after evaluation results have been looked at, or the held-out side is no longer unseen.

## Playback

On Linux the excerpt is sent to the default output through `pacat`, the PulseAudio client
that PipeWire systems carry (GMV's capture already uses its sibling `parec`). It opens an
ordinary playback stream at the file's own sample rate and channels (mono stays mono; more
than two channels plays the first two); no volume, device or routing is changed. Only the
excerpt's samples are sent. On Windows the PortAudio player of `gmv review` would be used;
that has not been tried with this tool.

## Relation to `gmv review`

`gmv review` is GMV's earlier excerpt-review tool, built for guitar labels: it writes
excerpt WAV copies into its work folder, its answers are guitar presence, type and role, and
its player uses PyAudioWPatch, which exists only on Windows. Reused here: its append-only
answer history (the same idea, with the five vocal answers), and on Windows its PortAudio
player. Not reused: its window and answer schema (guitar-specific), its excerpt copies
(this tool reads the seconds straight from the original), and its Linux playback (there is
none). The excerpt list and the split are new because the review tool has no artist split.

## The first dataset

Prepared on CachyOS from the owner's registered local corpus (198 recordings in 38 artist
groups) and the two reference tracks: 200 recordings named, all 200 usable (present,
readable, long enough, matching the corpus checksums).

| | development | evaluation (held out) |
|---|---:|---:|
| Artists | 27 | 12 (both reference artists among them) |
| Excerpts | 67 | 29 |
| Recordings with excerpts | 61 | 22 |

96 excerpts from 83 recordings and all 39 artists: two or three per artist, at most two per
recording, 24 by each placement method. Both reference tracks have two excerpts each (one
"likely vocal", one "even"), on the evaluation side. Excerpt starts spread through the
recordings (10th to 90th percentile at 17% to 77% of a recording's length; four start in
the first 15 s). Sample rates: 54 excerpts at 48 kHz, 39 at 44.1 kHz, 3 at 22.05 kHz; all
stereo. The held-out share is 30% of excerpts and 31% of artists rather than 25%, because
the hash rule and the two forced reference artists decide it, not a target count.

Limits: the corpus is one person's library, heavy on metal and metalcore, with game
soundtracks and some rock and pop-punk; little else. Twelve held-out artists is enough to
see whether a classifier works on artists it has not met; it is not enough to estimate
per-class accuracy precisely, especially for whichever answers turn out to be rare.

## Labelling results (the owner, 2026-11-24)

Kind of evidence: **the owner's own listening**, in the labelling window on CachyOS with
real playback. The answers stay on the owner's computer; only counts are given here.

**Every excerpt has an answer or a skip.** Of 96 excerpts, 58 have one of the five answers
and 38 were marked **No vocals**; none was left for review, and none is waiting for a
redo. The save file has 103 entries and no damaged line.

| | Development (27 artists) | Held out (12 artists) |
|---|---:|---:|
| Excerpts | 67 | 29 |
| Answered | 36 | 22 |
| No vocals | 31 | 7 |

**Answers on the development side** (the only side whose answers have been looked at):

| Cleans | High Screams | Low Screams | Gutturals | Sing-Screams |
|---:|---:|---:|---:|---:|
| 26 | 3 | 5 | 2 | 0 |

The 36 answered development excerpts come from 19 of its 27 artists.

**The held-out answers have deliberately not been read by answer.** Only whether each
held-out excerpt was answered or marked No vocals was counted. Their answers stay unseen
until a final evaluation, so nothing built later can be tuned to them, even by accident.

**The seven answers given with the first choices** were all redone with the current ones:
four received an answer and three were marked No vocals. They are still on record, and none
was translated automatically.

**How the labelling went.** The 96 current answers and skips were given in one sitting of
about 15 minutes (13 minutes of it in steady work). The median time between two answers
was 7.6 s, shorter than an excerpt (10 s), so many answers were given before the excerpt
had finished. No answer was changed afterwards and no note was written.

**What the existing vocal analysis was worth for choosing excerpts** (development side
only). Excerpts were placed four ways, and the owner's answers show how often a voice was
actually there:

| Placement | With a voice | No vocals |
|---|---:|---:|
| likely vocal (the existing analysis expected a voice) | 9 | 6 |
| even | 9 | 7 |
| random | 10 | 8 |
| likely instrumental (the analysis expected none) | 8 | 10 |

The analysis's guesses hardly differ from even or random placement. This matches Phase A2:
GMV's existing analysis cannot reliably tell from a full mix whether a voice is present.

**What this means for the next step.**

- **Two classes are usable on the development side; three are not.** Cleans (26) and Low
  Screams (5) could start a development experiment. High Screams (3) and Gutturals (2) are
  too few to learn or check anything, and there is no Sing-Screams example at all.
- **Answers are skewed:** about three in four development answers are Cleans.
- **Many excerpts had no voice:** 40% of all excerpts were instrumental, which spends
  listening time on passages with no vocal delivery to label.
- **Before any classifier, the dataset needs more harsh vocals.** Add recordings rich in
  high screams, gutturals and sing-screams, through **Add recordings…**, which keeps each
  artist's side fixed. A later round could also place excerpts only where the owner's own
  answers or the music suggests a voice, to waste fewer of them on instrumental passages.
- **The held-out side** (22 answered) can only show whether something works on unseen
  artists. It cannot measure per-class accuracy for the rare classes.

## Validation

- **Automated tests** (`tests/test_vocal_labeling.py`, 29, with generated audio and a
  stand-in for `pacat`): the dataset manifest, stable identifiers, exact sample-for-sample
  association of each excerpt with its recording and times, excerpt lengths and integrity
  checks, the artist split and its fixed rule, reference tracks held out, adding recordings
  without moving anyone, refusal to prepare again once labelled, missing, unreadable, empty,
  too short and changed recordings, saving, changing and skipping answers, notes, a damaged
  line after a crash, resuming, the reproducible order, the backup, playback sending exactly
  the excerpt's samples at its rate and channels, pause and resume, no sound program, the
  window's clicks, keys and counts, a second window refused, a missing recording, and that
  nothing about a recording or the split appears anywhere in the window.
- **Real machine** (CachyOS, KDE Plasma Wayland, PipeWire, Sound BlasterX G6 as the default
  output; the window on the HP Omen display), `scripts/validate_vocal_labeling.py`, on a
  copy of the real excerpt list in a temporary folder so the owner's answers were not
  touched: 28 of 28 checks passed under Wayland and again under X11 (XWayland). The window
  opened; clicking Play created a playback stream on the default output at the excerpt's
  own format (float32, 2 channels, 48 kHz); pausing after 2.5 s ended the stream at once
  and the position held; playing on took 7.53 s for the 7.46 s left (real time); the stream
  ended at the excerpt's end; "from the start", the position bar and Repeat worked;
  answers by click and by key were saved at once, changed, noted and skipped; the window
  closed with no stream left, and reopened on the same excerpt with every answer kept.
  Short pieces of excerpts at 48, 44.1 and 22.05 kHz each played as a stream at their own
  rate in real time.
- **Full test suite**, three runs: 665 or 666 passed, 3 skipped; the failures are the three
  known `tests/test_remote.py` failures in every run, plus two intermittent timing tests
  that fail in some runs and pass alone (details in [VALIDATION.md](VALIDATION.md)).
  `ruff check` clean.
- **Not checked:** that the sound was *heard*. The checks show the stream was created on
  the default output, consumed in real time and ended; whether it was audible at the
  owner's volume is for the owner to notice in the first seconds of labelling. Playback on
  Windows was not tried.

### After the change of answers and the song titles (same day)

- **Automated tests:** 39 in `tests/test_vocal_labeling.py` (10 added): earlier answers kept,
  shown and never translated, and replaced by a new one; skips with their reason, no vocals
  apart from review later, and the order they come back in; titles from the inventory by
  checksum, from the owner, from the file tag, never from a file name; title checks
  catching a wrong title, an unknown checksum and a file elsewhere; the titles command
  leaving `dataset.json` byte-for-byte unchanged; the title in the window, and "Song title
  not known" without one; the guide's definitions and the owner's reference list; nothing
  but the title shown about a recording; the export carrying the scheme and earlier answers.
- **Real machine,** the same script on a temporary copy (with the titles): 33 of 33 checks
  under Wayland, now also covering the song title shown for the excerpt and after reopening,
  the new answers, a changed answer, No vocals and Review later recorded apart, and a
  skipped excerpt still listed.
- **Full suite:** 676 passed, 3 skipped, 4 failed: the three known `tests/test_remote.py`
  failures and the known intermittent replay test. `ruff check` clean.
- **The owner's own folder:** `dataset.json` and `labels.jsonl` are byte-for-byte what they
  were; the 7 answers given with the first choices are kept and wait to be redone.

## Developer notes

```
gmv character prepare-labels --manifest corpus-data/owned/manifest.json \
  --paths corpus-data/corpus-local/paths.json --map 'E:\=/run/media/<user>/<drive>/' \
  --reference '<path>=<artist group>' ... --activity-model <weights> --target 96 --processes 8
gmv character label-titles --inventory <music inventory CSV> [--title '<path>=<title>'] \
  [--map 'E:\=/run/media/<user>/<drive>/']
gmv character label [--folder DIR]
```

Code: `src/gmv/character/research/labeling/` (`dataset`, `store`, `playback`, `window`,
`cli`). Tests: `tests/test_vocal_labeling.py`.
