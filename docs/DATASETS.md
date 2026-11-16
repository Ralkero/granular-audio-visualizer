# Research datasets for real-music evaluation

Researched 2026-11-15 from primary sources (dataset pages, Zenodo records, official repositories, papers). The project owner does not annotate recordings by hand; reference data comes from existing datasets and their isolated stems. The licence of a loader package says nothing about the audio: the audio licences below are the ones stated for the audio itself.

**No dataset audio is committed.** The user obtains each dataset under its own terms; `gmv corpus import-dataset` then registers it from the local folder. All datasets below are non-commercial or research-only unless stated.

## Comparison

| Dataset | Tracks / hours | Metal relevance | Stems | Activity annotations | Attack / onset annotations | Beats | Audio licence | Access | Size |
|---|---|---|---|---|---|---|---|---|---|
| **MDB Drums** | 23 real full mixes | 1 speed-metal excerpt, punk, grunge | mix, drum-only, drum multitracks | no (derivable from drum stem) | **yes: kick, snare, hi-hat, toms, cymbals (6 and 21 classes, 7,994 onsets)** | **yes** | CC BY-NC-SA 4.0 | public git repository | ~2 GB |
| **MUSDB18-HQ** | 150 / ~10 h | **12 Heavy Metal** (8 train, 4 test), 17 Rock, 72 Pop/Rock | mix + drums, bass, vocals, other | no | no | no | mixed; academic / non-commercial ("educational purposes") | Zenodo record, open | 22.7 GB |
| **MoisesDB** | 240 / 14.4 h | "mostly rock and pop"; metal count not published | **kick, snare, toms, cymbals, overheads, distorted / clean guitar, bass, vocals**; no mixture file | no (the loader computes it from stem energy) | no | no | CC BY-NC-SA 4.0, research only | download form on music.ai, submitted by a person | not published |
| **MedleyDB 1 + 2** | 122 + 74 / 7.3 h (v1) | 31 Rock, 1 speed-metal excerpt | mix, stems, raw tracks | **yes: activation confidence and source-ID files** | no | no | CC BY-NC-SA 4.0 | Zenodo access request | not published |
| ENST-Drums | 3 drummers, ~3.75 h | rock kit | 8 microphone channels | no | yes | no | CC BY-NC-ND 4.0 | Zenodo, open | 9.6 GB |
| IDMT-SMT-Drums | 104 drum loops, 2.2 h | none | kick/snare/hi-hat (subset) | no | yes | no | CC BY-NC-ND 4.0 | Zenodo, open | 0.3 GB |
| STAR Drums | 6,185 / 124.5 h | "metal and punk" bucket | drums + non-drum | no | yes, 18 classes, **synthesised drums** | no | BSD-3 tag on Zenodo | open; part must be rebuilt from MUSDB18 | 181 GB |
| E-GMD / GMD | 444 h / 13.6 h | none | drums only (electronic kit) | no | yes (MIDI) | tempo | CC BY 4.0 | open | 90 / 4.8 GB |
| RWC | several collections | not checked | no | partial | MIDI-aligned | yes | CC BY-NC 4.0 | Zenodo, open (2026 re-release) | 13.4 GB |
| Ballroom | 698 excerpts | none | no | no | no | yes, with downbeats | not stated | open | 1.5 GB |
| Harmonix Set | 912 / 56 h | **32 Metal, 5 Prog** | no | no | no | yes, with downbeats | **no audio distributed** | — | — |
| Slakh2100 / BabySlakh | 2,100 / 145 h; 20 tracks | synthetic renderings | yes | from MIDI | exact (MIDI) | not explicit | CC BY 4.0 | open | 104 GB / 0.9 GB |
| AAM / Tiny AAM | 3,000; 20 tracks | synthetic renderings | yes | yes | yes | yes | CC BY 4.0 | open | 210 GB / 0.2 GB |
| ADTOF | 359 h | rhythm-game charts | no audio (spectrograms only) | no | 5 classes | — | — | access request | — |
| Böck onset_db | — | — | no | no | onsets (annotations only) | no | audio by contacting the authors | — | — |

Not verified: Cambridge-MT / Mixing Secrets licence terms (the site blocked automated reading); MoisesDB archive size and form fields; SMC and Hainsworth availability; RWC genre content. No metal-specific drum-transcription or beat dataset with obtainable audio was found.

## What each kind of reference can come from

| Needed reference | Best source | Honest limit |
|---|---|---|
| Kick / snare / hi-hat / tom / cymbal attacks on real mixes | **MDB Drums** | 23 tracks, little metal |
| Beats (and downbeats where numbered) on real mixes | **MDB Drums**; later RWC, Ballroom | no metal with obtainable audio; Harmonix has metal annotations but no audio |
| Drums / bass / vocals activity | **MUSDB18-HQ** stems (12 heavy-metal tracks) | derived from stem energy, not annotated |
| Kick / snare / cymbal / guitar activity | **MoisesDB** stems | derived from stem energy; drum microphones bleed |
| Dataset-annotated instrument activity | **MedleyDB** | needs an approved access request |
| Full-mix onsets of all instruments on real music | **none obtainable** | only synthetic renderings (AAM, Slakh) have exact onsets |

## Selection

1. **MDB Drums — first.** The only openly obtainable real full mixes with individual drum attacks and beats. It makes the drum-related candidates, the beat tracker and onset recall measurable today.
2. **MUSDB18-HQ — second.** Largest confirmed metal subset with isolated stems; gives drums, bass and vocals activity over about ten hours.
3. **MoisesDB — third.** Adds kick, snare, tom, cymbal and distorted-guitar stems. Needs a person to submit the download form.
4. **MedleyDB — when access is granted.** Adds dataset-annotated activity. No adapter yet.

Deferred: synthetic renderings (Tiny AAM, BabySlakh) as a separate "rendered" tier for full-mix onsets; RWC and Ballroom for more beat material; ENST-Drums and IDMT-SMT-Drums for isolated drum attacks.

**Overlap.** MUSDB18 (46 tracks) and MDB Drums (all 23) reuse MedleyDB recordings, and three artists appear in both the MUSDB18 train and test sets. Splits are therefore assigned per artist across all datasets: a group is the artist, one artist is in exactly one split everywhere, and an artist with any official test track goes to test as a whole.

## Suitability for splits

| Dataset | Train / validation | Held-out test |
|---|---|---|
| MDB Drums | yes (all 23 tracks are one artist group) | **no** |
| MUSDB18-HQ | official train tracks, divided 80/20 by artist | official test tracks plus any artist that overlaps them |
| MoisesDB | yes, by artist (no official split) | yes, by artist |

## Importing

```powershell
gmv corpus datasets                                   # supported datasets, licences, access
gmv corpus import-dataset mdbdrums  "E:\datasets\MDBDrums"
gmv corpus import-dataset musdb18hq "E:\datasets\musdb18hq"
gmv corpus import-dataset moisesdb  "E:\datasets\moisesdb"
```

Dataset imports belong in a local corpus that is not committed. Pass `--manifest corpus-data/corpus/manifest.json` (the `corpus-data/` directory is git-ignored) so that dataset track lists and dataset-derived annotations stay on your machine; pass the same manifest to the evaluation command with `--corpus`.

Each import hashes and registers every track, registers its stems, assigns the artist group and split, writes the references, and validates the corpus. It can be interrupted and run again; tracks already registered are skipped. `--limit N` imports only the first N new tracks.

| Adapter | Reference written | Tier |
|---|---|---|
| `mdbdrums` | attacks for drums, kick, snare, hi-hat, toms, cymbals; beats and downbeats | `dataset_annotation` |
| `mdbdrums` | drums activity from the drum-only stem | `stem_derived` |
| `musdb18hq` | drums, bass, vocals, other activity | `stem_derived` |
| `moisesdb` | drums, kick, snare, toms, cymbals, hi-hat, bass, vocals, guitar, other-harmonic activity; stems flagged `has_bleed` marked uncertain; mixture rendered locally as the sum of stems | `stem_derived` |

`mdbdrums` has been run against the real dataset (23 of 23 tracks imported). `musdb18hq` and `moisesdb` were written from the documented layouts and tested against miniature stand-in folders only.

**MDB Drums cannot be split.** All 23 tracks are by one artist, so they form one group and one split. It is development material; a held-out drum set has to come from another dataset.

## What still needs a person

- Accepting each dataset's terms and starting its download (MDB Drums and MUSDB18-HQ are direct downloads; MoisesDB needs its web form; MedleyDB needs an approved request).
- Nothing else. No labelling is required.
