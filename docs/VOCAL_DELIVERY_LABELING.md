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

**What you see.** At the top, the excerpt's name (for example `VOCAL-0042`), whether you have
answered it, and how far you are: labelled, skipped, still to label. On the right, what the
five answers mean, and the list of all excerpts.

**Listen.** The excerpt plays as soon as it opens (ten seconds of a song; nothing else of
the song is played).

- **Play / Pause** starts and stops it.
- **Play again from the start** goes back to the beginning.
- **Repeat** plays it over and over until you pause.
- The bar shows where you are; drag it to jump.

**Answer.** Click one of the five large buttons:

| Answer | Means |
|---|---|
| **Clean singing** | Melodic, pitched singing with a relatively clean vocal tone |
| **Screaming** | Harsh delivery: screams, growls, shrieks and similar aggressive techniques |
| **Shouting** | Forceful, yelled or speech-like delivery that is not mainly clean singing and not a sustained scream |
| **Mixed / ambiguous** | Several techniques in the excerpt, overlapping voices, or you cannot confidently pick one |
| **No vocals** | Instrumental: no audible voice |

Label what you hear, whatever the style of the song. Your answer is saved the moment you
click ("Saved ✓" appears in green), and the next excerpt opens.

**Not sure?** Click **Skip / come back later**. Skipped excerpts stay in the list and are
counted apart; **Next without an answer** brings you back to them once everything else is
done. You can also type a short **note** (optional); it is saved by itself.

**Change an answer.** Click **← Previous** (or click the excerpt in the list on the right)
and click a different answer. The new answer replaces the old one; the old one is kept in
the history.

**Stop and resume.** Close the window whenever you like. Nothing is lost. Run the same
command again and it opens where you stopped.

**Optional keys:** 1 to 5 answer, Space play/pause, R from the start, S skip, ← →
previous/next. The buttons do the same.

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

## Blind labelling

The window shows an excerpt's anonymous name and nothing else about it: no title, artist,
album, file name or folder, not how the excerpt was chosen, and not whether it belongs to
the part kept for the final test. An automated test opens every excerpt and checks that
none of these appears in any text, tooltip, list entry or title of the window, including
when a recording is missing.

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
| `labels.jsonl` | Every click, one line each, with its time: answers, skips, notes. Never rewritten, so a changed answer keeps the old one in the history, and a crash can at worst lose the line being written (it is then reported and left out) |
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

## Developer notes

```
gmv character prepare-labels --manifest corpus-data/owned/manifest.json \
  --paths corpus-data/corpus-local/paths.json --map 'E:\=/run/media/<user>/<drive>/' \
  --reference '<path>=<artist group>' ... --activity-model <weights> --target 96 --processes 8
gmv character label [--folder DIR]
```

Code: `src/gmv/character/research/labeling/` (`dataset`, `store`, `playback`, `window`,
`cli`). Tests: `tests/test_vocal_labeling.py`.
