# Vocal presence annotation and dataset foundation (Sonic Character, Phase A5.1)

**Research tooling.** This phase builds the annotation tool, the data layout and the
protections needed to develop a better vocal-presence detector. **No model has been trained
or evaluated in A5.1.** Nothing here is part of the visualizer, Radiant or any live
analysis. The existing activity model and the vocal analyzer are unchanged.

## Why: what A5.0 found

A private audit (A5.0) found that the existing vocal-activity detector rarely misses a voice
at the excerpt level. But it reports a voice in most of the owner's instrumental excerpts,
and every such false alarm is drawn as a vocal phrase. Measuring and fixing that needs:

- time-localised labels (where a voice starts and stops), which the A3 labels are not;
- deliberate instrumental negatives;
- a fresh artist-disjoint final test.

## Presence is not delivery

The Phase A3 project records one *delivery* answer per ten-second excerpt (Cleans, High
Screams, Low Screams, Gutturals, Sing-Screams, or No vocals). Phase A5.1 records something
different: *where* in the excerpt a human voice is audible, whatever its kind. A stretch can
be marked as a voice without saying how it is delivered.

The A3 project is read, never written. Its answers keep their meanings, and its history,
titles, backups and inventory are untouched (fingerprinted before and after every
preparation run).

## The annotation schema

Each partition (development, validation, final test) has its own append-only history,
`annotations.jsonl`, separate from the A3 files.

- **Intervals:**
  - Each has a start and end in seconds on the excerpt's own timeline, kept to the
    millisecond. Start must come before end, and both must lie inside the excerpt; marks
    shorter than 20 ms are refused as slips.
  - Each is either **confirmed** (a voice is audible) or **uncertain** (the annotator cannot
    tell), with an optional note.
  - Overlaps are allowed: several voices at once still mean "a voice". Binary targets use
    the union of confirmed intervals; the original intervals stay in the history.
- **Review state:**
  - **in progress**, the default;
  - **complete**: every audible voice is marked, as far as the annotator can tell. A voice is
    marked by listening. A no-voice answer can also rest on the annotator's knowledge of the
    recording: a band that never has vocals, a song known to have none, or an interlude. In
    that case not every second need have been played, and the window does not record how
    much was heard;
  - **unusable**: with a required reason.
- **Excerpt status:**
  - **positive**: complete, with at least one confirmed interval;
  - **negative**: complete, with no interval of either kind. This is an explicit No-vocals
    example: the annotator's judgement, from listening and from knowledge of the recording;
  - **uncertain**: complete, with only uncertain intervals. Its excerpt-level label is
    unknown;
  - **incomplete** or **unusable**: never training or evaluation data, and never a
    negative.
- **Frame targets for a complete excerpt:** 1 inside confirmed intervals, unknown inside
  uncertain-only time, 0 elsewhere. An incomplete or unusable excerpt has no known frames.
  Uncertain material never silently becomes a negative.
- **Confounder tags,** any number per excerpt: distorted rhythm guitar, guitar lead/solo,
  bass-focused, drums only, synthesizer/keyboard, orchestral/ambient, dense instrumental
  mix, quiet or buried vocals, layered vocals, other (see the note). Tags describe what
  makes an excerpt useful for testing a detector. **They are not presence labels:** a
  guitar-solo tag does not mean "no voice".
- **History integrity:**
  - Edits and removals are new events, so every revision is kept.
  - Each event carries a sequence number and a hash chained to the previous event, so a
    changed, reordered or deleted line is detected; the development loaders refuse such a
    history.
  - A line cut off by a crash is skipped and counted, and the next event chains on from the
    last good one.
  - Invalid annotations are refused before they are written.

## Preparing the project

`gmv character presence prepare --inventory <music inventory CSV> [--map <inventory prefix>=<mount point>] [--corpus-manifest <corpus manifest>] [--freeze-final-test]`
reads the A3 project (read only), the inventory and, with `--corpus-manifest`, the MUSDB18-HQ
test artists to exclude. `--map` rewrites the inventory's file paths when the inventory was
made on another system. It writes only inside the presence folder, and it can be run again:
existing excerpts are kept, and groups never move.

## The annotation window

`gmv character presence annotate --partition development` (or `validation`).

- **Reuse:** it uses the A3 tool's player, which plays exactly the excerpt's samples
  through the sound server and changes no volume, device or routing. It also reuses the A3
  excerpt reader and its private song titles; a title is never taken from a file name.
- **The timeline** shows the excerpt's loudness outline, confirmed (green) and uncertain
  (amber) stretches, an open mark and the playback position. Clicking it seeks.
- **Marking:** `[` starts a stretch at the playback position. `]` ends it as a voice, `U`
  ends it as uncertain, and `Esc` drops it.
- **Moving:**
  - `Space` plays or pauses; `Home` goes to the start.
  - Arrow keys move by 0.1 s, with Shift 0.02 s, with Ctrl 1 s.
  - PgUp and PgDn go to the previous or next excerpt.
- **Accuracy:**
  - A mark made while paused is exact.
  - A mark made while playing lands late, by reaction time plus output latency (about
    0.1 s).
  - Refine boundaries while paused: nudge the selected stretch's start or end by 0.05 s, set
    it to the current position, or **listen around** a boundary (1 s before to 0.5 s after).
- **Review:** complete (refused while a mark is open), reopen, unusable (with a reason),
  tags and a note.
- **Saving:** every decision is saved at once.
- **Other:** "Next not completed", resume on reopening, a lock so that one partition is
  open in one window only, and "Save a backup copy…" (history, manifests, fingerprints,
  titles; never audio).
- **Blind to selection:** the window never shows why an excerpt was chosen: no delivery
  answer, no "candidate negative". The annotation is not steered.

## Data layout and protection

Everything private lives in the ignored folder `local-evidence/sonic-character/vocal-presence/`:

| Folder | Contents |
|---|---|
| `manifests/` | Per-partition excerpt lists, plus the split registry |
| `fingerprints/` | SHA-256 of each manifest and of the registry |
| `identity/` | Private artist groups, titles, and the owner's optional alias file |
| `development/`, `validation/`, `final-test/` | Annotation histories, one per partition |
| `backups/` | Backup copies |
| `selection/` | The selection report and candidate lists |

- **Manifests:** each excerpt has a stable identifier, derived from the recording checksum
  and the start time. Each also records its recording (checksum, size, length, rate,
  channels), its times, its artist group and why it was selected. Manifests are written
  atomically with their fingerprints, and a manifest that does not match is refused.
- **Project check.** Before anything is used, every excerpt is checked against the
  registry:
  - its group exists and belongs to the excerpt's partition;
  - its recording is a member of that group;
  - no excerpt or recording appears in two partitions;
  - unknown groups, duplicate identifiers and mismatched assignments are rejected.
- **Final-test protection.** Final-test annotations are kept in their own folder, and only
  the annotation window reads them. The only development loader (`development_examples`)
  reads the development and validation histories. Asked for the final test, it raises an
  error, and it never opens a final-test path; tests prove this by failing on any such read.
  No final scoring exists in A5.1.

## Artist groups and the split

- **Conservative grouping.** Recordings are joined into one group when they share any of:
  - a credited name: artist, album artist, MusicBrainz credit and "(feat. …)" in titles,
    normalised for case, accents, punctuation and a leading "the", so a spelling variant
    never creates a new artist;
  - a MusicBrainz artist identifier;
  - the inventory's leakage, duplicate-audio or version group;
  - the A3 artist key;
  - the owner's private alias and shared-vocalist links.

  Generic credits ("Various Artists" and similar) link nothing. Where identity is unclear,
  groups merge. Vocalist identities are mostly not recorded, so performer disjointness is
  claimed only as far as these links go.
- **The registry (append-only).**
  - **Excluded:** groups touching consumed evaluation material, meaning the 12 Phase A4
    held-out groups, MUSDB18-HQ test artists (from the corpus manifest), or anything linked
    to them.
  - **Development and validation:** earlier development artists. New ones are walked in a
    fixed hash order; a group joins validation while validation holds under 20% of the
    recordings and would stay at or below 30%.
  - **Reserved:** never-used groups, held back for a future final test.
  - **Final test:** set only by an explicit freeze, which requires at least 20 eligible
    never-used groups.
  - Groups never move. A rebuilt group that would join two partitions, or newly link a used
    group to consumed material, stops preparation until resolved.
- Selection never uses model output. Everything is deterministic (seeded SHA-256 orders),
  and re-preparing keeps existing excerpts and adds only new ones.

## Current coverage (aggregate)

- **The library** is 45 artist groups after conservative grouping, which is the whole
  inventory. 952 audio files, 942 of them inventoried.
- **Registry:**

  | Partition | Groups | Recordings |
  |---|---:|---:|
  | Excluded | 12 | 295 |
  | Development | 20 | 484 |
  | Validation | 6 | 168 |
  | Reserved | 7 | 7 |

- **Prepared for annotation:** 199 excerpts from 26 artist groups and 178 recordings.
  Development has 171 (91 reused A3 excerpts, 80 new candidates); validation has 28 (9
  reused, 19 new).
  - **Reused A3 development excerpts (100):** 7 Gutturals, 4 Sing-Screams, 22 High Screams,
    22 Low Screams, 10 No vocals (to re-check as full negatives) and 35 Cleans. They span 23
    groups and 100 different recordings. The A3 answer is kept only as the reason for
    selection.
  - **Instrumental negative candidates (99),** from metadata only:

    | Category | Candidates |
    |---|---:|
    | Distorted rhythm guitar | 20 |
    | Guitar lead / solo | 20 |
    | Orchestral / ambient | 20 |
    | Other: instrumental soundtrack | 20 |
    | Other: song-intro windows | 19 |
    | Synthesizer / keyboard | **0** |
    | Bass-focused | **0** |
    | Drums only | **0** |

    The library's metadata supports none of the last three. Each category rests on 2 to 5
    artist groups (song intros: 19). Every candidate must be confirmed by the annotator.
  - **Harsh vocals:** the development side has High Screams from 7 artist groups, Low
    Screams from 10, Gutturals from 1 and Sing-Screams from 2. Only 42 unused vocal
    recordings exist in the 3 groups that have harsh answers. Broadening harsh coverage
    needs new artists.
- **Final test: not frozen.** Only 7 never-used artist groups exist, one recording each,
  against the 20 groups required. A defensible final test needs at least about 13 more
  never-used artist groups with vocal recordings. Ideally they would include harsh-vocal
  artists and instrumental material per confounder type, added to the library and
  inventory. Until then the 7 reserved groups stay unused.

## Development annotation (completed 2026-11-26, aggregate)

The owner annotated the development partition in the window on CachyOS (real hardware,
owner-evaluated), in two sittings and a short round of fixes. The checks below were read
from the saved history and the audio; nothing was changed.

- **Result:**
  - All 171 development excerpts are completed: 89 have a voice and 82 have none. None is
    uncertain or unusable.
  - They span 20 artist groups and 153 recordings. Voiced excerpts come from 16 groups, and
    no-voice excerpts from 18.
  - A voice is audible in 703 of the 1,710 annotated seconds (41%).
- **Marks:**
  - 112 voice stretches, 1 to 3 per voiced excerpt (69 have one, 17 two, 3 three). The
    median voiced excerpt has a voice in 92% of its length; 35 of the 89 are voiced
    throughout.
  - No boundary was adjusted after marking, so boundaries carry the live-marking lag.
  - No confounder tag or note was used.
- **History integrity:**
  - 289 events, chain intact, sequence contiguous, no damaged line, none rejected.
  - The project check passes, and the A3 answer file is unchanged.
- **Agreement with the A3 delivery answers** (the same windows):
  - Of the 84 reused excerpts with an A3 voice answer, 82 have a voice marked. Two (one
    Low Screams, one Cleans) are marked as having no voice. The two answers disagree, and
    both histories are kept as they are.
  - All 7 reused A3 No-vocals excerpts are negatives.
- **Metadata candidates:** 7 of the 80 instrumental candidates had an audible voice:

  | Category | With a voice |
  |---|---:|
  | Song-intro windows | 3 of 14 |
  | Distorted rhythm guitar | 2 of 20 |
  | Guitar lead / solo | 1 of 20 |
  | Orchestral / ambient | 1 of 10 |
  | Instrumental soundtrack | 0 of 16 |

  Two song intros have a voice entering 6.0 s and 8.4 s in. Metadata alone does not make a
  negative.
- **The 82 negatives:** 73 confirmed candidates, the 7 A3 No-vocals excerpts and the 2
  disagreements above.
- **Audio checks:** every completed excerpt's audio could be read. No voice mark lies over
  near-silence, and no negative is mostly silence.
- **How the negatives were judged:** the window does not record playback. On 50 of the 82
  negatives, Complete came sooner after the previous excerpt than the excerpt could play in
  full. The owner confirmed that such answers rest on their knowledge of the recordings:
  all-instrumental bands, songs known to have no vocals, and interludes.

## Validation annotation (completed 2026-11-26, aggregate)

The owner then annotated the validation partition in the same window (real hardware,
owner-evaluated). The checks were read-only.

- **Result:**
  - All 28 validation excerpts are completed: 6 have a voice and 22 have none. None is
    uncertain or unusable.
  - They span 6 artist groups and 25 recordings. Voiced excerpts come from 4 groups, and
    no-voice excerpts from 5.
  - A voice is audible in 52 of the 280 annotated seconds (19%).
- **Marks:**
  - 6 voice stretches, one per voiced excerpt, and no uncertain stretch. The median voiced
    excerpt has a voice in 98% of its length (lowest 44%); 3 of the 6 are voiced throughout.
  - No confounder tag or note was used.
- **Sources:**
  - **Voiced (6):** 3 A3 Cleans, 2 A3 Low Screams and 1 song-intro candidate.
  - **No voice (22):**
    - 18 confirmed candidates: 10 orchestral/ambient, 4 instrumental soundtrack and 4 song
      intros;
    - 3 A3 No-vocals excerpts;
    - 1 A3 Cleans excerpt now marked as having no voice. The answers disagree, and both
      histories are kept.
  - 1 of the 19 instrumental candidates had a voice.
- **History integrity:** 34 events. The chain is intact, checked both by the project's
  loader and by an independent recomputation. The sequence is contiguous, and no line is
  damaged or rejected.
- **Separation:**
  - The project check passes.
  - Development and validation share no excerpt, recording or artist group, and each
    history refers only to its own partition's excerpts.
  - All 6 validation groups are registered as validation. No final-test list exists, and
    the reserved groups are unchanged.
  - The development history is byte-identical to its state before validation was
    annotated. A verified local backup of both partitions was made.
- **Audio checks:** every excerpt's audio could be read. No voice mark lies over
  near-silence, and no negative is mostly silence.
- **What validation can show:** with 22 negatives from 5 groups, it can check false
  alarms. With only 6 voiced excerpts from 4 groups, and only 2 harsh ones, any estimate
  of missed voices will be very imprecise.

## Limitations

- Three reused excerpts (2 development, 1 validation) disagree with their A3 delivery answers:
  a voice in A3, none here.
- Live marks lag by reaction time; boundary accuracy depends on refining while paused.
- No-voice answers are the owner's judgement, partly from knowledge of the recordings; how
  much of each excerpt was played is not recorded.
- Confounder categories come from sparse metadata; three categories have no candidates.
- Validation has only 6 voiced excerpts (4 artist groups, 2 harsh), so missed-voice estimates
  on it will be imprecise. All negative categories rest on few artists.
- Identity grouping depends on credits; uncredited shared vocalists would not be caught.
  One development group was excluded through a composer credit, a conservative merge.
- The project folder lives in this worktree's ignored `local-evidence/`. Use **Save a backup
  copy…** regularly; a worktree removed without moving that folder would lose it.
