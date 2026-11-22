# Evidence enrichment and automatic label candidates

Milestone result in one paragraph: every recording of the owner's collection (943
recordings, 59.4 h) now has an identity attempt, credits by scope, local fingerprints and
two independent model timelines. Combining them gives trustworthy "guitar present"
candidates for only **3.4% of the collection's time** (2.0 h in 218 recordings), and **no
trustworthy "guitar absent" candidates at all**. The reason is measured, not assumed: the
second audio model hears little guitar in this collection, and two models agreeing on "no
guitar" were right about four times in five on the owner's own labels. A person is still
needed, but for about an hour of targeted listening rather than for the collection.

No model was trained. `guitar_mn10_v1` is unchanged and remains the baseline;
`guitar_mn10_rv1` is kept apart and is not used as evidence, because it has seen the
reviewed excerpts.

Personal results (titles, credits per recording, label candidates, the review queue) are in
the ignored folder `research-local/evidence-v1/`; this page holds only aggregate numbers.
Licences and provenance: [LICENCES_AND_PROVENANCE.md](LICENCES_AND_PROVENANCE.md).

## Commands

```
gmv evidence enrich --inventory INVENTORY.csv            # MusicBrainz identity and credits
gmv evidence fingerprint --inventory INVENTORY.csv       # local Chromaprint, same-audio groups
gmv evidence acoustid --inventory INVENTORY.csv --owner-approved   # needs GMV_ACOUSTID_KEY
gmv evidence datasets --openmic ROOT --moisesdb ROOT     # dataset manifest, OpenMIC labels
gmv evidence scan --inventory INVENTORY.csv              # the project's recogniser, all files
python experiments/evidence/panns.py --weights W.pth scan --inventory INVENTORY.csv --output research-local/evidence-v1/panns/timelines
python experiments/evidence/separation.py pilot --inventory INVENTORY.csv --review review-local/pilot-v1 --output research-local/evidence-v1/separation
gmv evidence check                                       # automatic classes against human labels
gmv evidence fuse --inventory INVENTORY.csv              # candidates, queue, all tables
gmv review --workdir research-local/evidence-v1/review-queue build --inventory INVENTORY.csv --selection research-local/evidence-v1/review_queue_selection.json
gmv review --workdir research-local/evidence-v1/review-queue open
gmv evidence trainset --review review-local/pilot-v1 --review research-local/evidence-v1/review-queue   # segments, folds, audit
```

The two `python experiments/...` commands run in the experiments environment (torch); the
rest in the application environment. Every step skips what it has already done: MusicBrainz
and AcoustID answers are cached per request, scans per recording. `fuse` rewrites the tables
from whatever is present.

Cost on this machine (Ryzen 9 5950X, CPU only), for 59.4 h of audio: recogniser scan 94 min
with 7 processes; PANNs 95 min with 6 threads (36x real time while sharing the CPU, 72x
alone); MusicBrainz about 2,100 requests at one a second; fingerprints a few minutes; separation
1.0x real time with 4 threads, which is why it was run on 39 minutes of audio only.

## Identity and credits (MusicBrainz)

Matching order: recording id in the file's tags, ISRC, then a scored search on title,
artist, album and length. The ids in these files' tags turned out to be track ids, which
are resolved through a search by track id.

| Identity | Recordings |
|---|---|
| matched, high confidence | 744 |
| matched, medium or low confidence | 25 |
| ambiguous: candidates kept, none assigned | 81 |
| unmatched | 93 |

A match is refused when another kind of recording (live, remix, acoustic, ...) fits the
tags equally well apart from its version marker. 21 matched recordings carry a noted
conflict (length or title differing from MusicBrainz).

| Strongest guitar credit | Recordings | Used as |
|---|---|---|
| recording-level guitar credit | 254 | context about the recording: guitar is on it somewhere, never when. It does not change the class or weight of any stretch |
| release-level only | 11 | weak evidence; never applied to a track |
| artist-level only (a member plays guitar) | 270 | weak evidence |
| credits, none for guitar | 35 | nothing: a missing credit is not "no guitar" |
| no instrument credits | 373 | nothing |

Of the 254 recording-level credits, 158 name "guitar" without a kind, 96 electric, 1
acoustic and 1 steel or slide. Credits therefore cannot fill the gap in acoustic and clean
examples.

AcoustID lookups are implemented (fingerprint, duration and application key only; lookup
only; cached; only for unmatched or ambiguous recordings; a result is assigned only when it
names one recording of fitting length) and approved by the owner, but not run: the service
needs an application key from the owner's account. Discogs was not used: its search needs an
account token and its terms restrict reuse of part of its data.

## Leakage protection

Split groups are the connected components of: artist folder, identical audio by local
fingerprint, the inventory's duplicate and version groups, and identical reliably matched
MusicBrainz recordings (`gmv.evidence.leakage`). Fingerprints found four pairs of identical
audio; three lie inside one folder each, one joins two folders, which are now a single
group of 37 recordings. The collection has 45 groups. `gmv review build` reads the groups
through the selection file (`leakage_groups`) and never picks two copies of the same audio;
`cache_review.py --leakage` applies them to the training cache.

19 groups (571 recordings) contain an artist whose reviewed excerpts were used to train
`guitar_mn10_rv1`. 26 groups (372 recordings) are untouched and are the only place an
independent evaluation set can come from.

## The two audio models, each on its own

PANNs CNN14 with frame-wise output is a different architecture from the project's
EfficientAT model, but both were pretrained on AudioSet: agreement is supporting evidence,
not independent confirmation.

OpenMIC-2018 (human answers, 10 s clips, neither model trained on it). Guitar was put to
annotators for 1,650 of 20,000 clips (1,138 present, 512 absent); the other 18,350 are
unknown. Figures are for the dataset's test partition (436 clips; no artist shared with the
training partition); AUC involves no threshold.

| | AUC | Recall at 5% false alarms | at 10% |
|---|---|---|---|
| `guitar_mn10_v1` | 0.967 | 0.89 | 0.95 |
| PANNs guitar | 0.989 | 0.94 | 0.99 |

`guitar_mn10_v1` at its shipped thresholds, all 1,650 clips: "detected" finds 75% of guitar
clips and fires on 3.3% of guitar-free ones. Among guitar-free clips its mean score is
highest with organ (0.31), cello (0.27) and violin (0.25), and lowest with synthesizer
(0.04); PANNs stays below 0.01 on all of them.

The owner's reviewed excerpts (core excerpts, 159 with guitar throughout, 64 without, 19
artists), excerpt means:

| | AUC, all | distorted | clean electric | acoustic (10) |
|---|---|---|---|---|
| `guitar_mn10_v1` | 0.886 | 0.873 | 0.909 | 0.889 |
| PANNs guitar | 0.832 | 0.805 | 0.868 | 0.905 |
| mean of both ranks | 0.892 | | | |

Both models are far better on OpenMIC than on this collection, so the difficulty lies in
this music (dense, distorted, heavily produced), not in guitar detection in general. PANNs'
median guitar score on excerpts with guitar is 0.20 on OpenMIC and 0.018 here.

## Agreement between the models

Cut-offs were chosen on OpenMIC's training partition only, as the widest pair at least 99%
pure there, and then scored once on its test partition:

| Both models say | Rule | Test clips selected | Wrong | Precision | Coverage |
|---|---|---|---|---|---|
| guitar | recogniser >= 0.926 and PANNs >= 0.10 | 181 | 0 | 1.000 | 0.63 of guitar clips |
| no guitar | recogniser <= 0.05 and PANNs <= 0.003 | 42 | 1 | 0.976 | 0.27 of guitar-free clips |

The same rules on the owner's reviewed excerpts (46 recordings; one excerpt counts as
selected when at least half of its half-second hops are):

| Both models say | Excerpts selected | Correct | Wrong | Precision | Coverage |
|---|---|---|---|---|---|
| guitar | 10 | 10 | 0 | 1.00 | 10 of 155 guitar excerpts (6.5%) |
| no guitar | 13 | 11 | 2 | 0.85 | 11 of 64 guitar-free excerpts (17%) |

By hop: 338 hops called "guitar", 337 in guitar excerpts (99.7%); 392 hops called "no
guitar", 315 in guitar-free excerpts (80.4%). 3 excerpts were contradictions, all with
guitar; 193 were left undecided. The first ten seconds of a recording are always left
undecided, because the recogniser has no full context there.

What this means: "both say guitar" transfers to this collection but covers little of it;
"both say no guitar" does not transfer. Lowering PANNs' cut-off would widen coverage (the
recogniser's own "detected" was 97.7% right on these excerpts at 61% coverage), but that
would be tuning on the labels used to report it, and a label that only repeats the
recogniser's own confident output teaches a retrained recogniser little.

## Separation as a third source (pilot)

HTDemucs six-source on stretches around the reviewed excerpts of 20 recordings (90 excerpts:
65 with guitar, 25 without). The measure is the guitar stem's level relative to the mix.

| On the same 90 excerpts | AUC |
|---|---|
| guitar stem level | 0.874 |
| `guitar_mn10_v1` | 0.911 |
| PANNs guitar | 0.879 |
| mean rank, recogniser + PANNs | 0.931 |
| mean rank, all three | 0.939 |

Stem level at least -10 dB below the mix: 58 excerpts selected, 93% with guitar. At most
-40 dB: 14 selected, 86% guitar-free (two had guitar). Among the 81 excerpts the two models
leave undecided the stem separates as well as the recogniser does (0.887 against 0.872).

Verdict: about as informative as either model, with errors of its own, and little gain when
added (0.931 to 0.939, well inside what 20 recordings can resolve). It is kept as a
supporting source for chosen stretches. Running it over the whole collection would cost
about 60 CPU-hours and is not justified by this pilot.

## Evidence classes

| Class | What it rests on | Category | Proposed training weight |
|---|---|---|---|
| verified reference | a person's answer for an excerpt | A | 1.0 |
| strong agreement | both models say guitar, and the separated guitar stem measured over that same stretch (at least 80% of it) supports them | B | 0.5 |
| moderate agreement | both models say guitar, with no other source that covers the stretch | B | 0.25 |
| weak evidence | a credit alone (no time); and every automatic "no guitar", with or without the stem agreeing | C | none |
| conflict | the models, or a model and the stem, contradict each other | D | none |
| unknown | everything else, including all recordings without credits | D | none |

**Scope.** Only evidence measured over a stretch can set or raise that stretch's class: the
two models' scores and the separated stem's level. Credits describe a whole recording
(or a release, or an artist) and are written beside a candidate as
`recording_level_metadata`; they change neither class nor weight. An earlier version let a
recording-level guitar credit raise every agreed stretch of that recording from moderate to
strong. That was wrong, since the credit says nothing about when the guitar plays, and it
was corrected before any training: all 196 stretches that had been "strong" (0.89 h) owed
the class to a credit alone and are now moderate. No stretch outside the reviewed excerpts
has separation, so the collection currently has no "strong" candidate. Automatic
"no guitar" is never a training label in any class.

The weights are proposals for a later experiment, not calibrated probabilities. Stretches
that a person has labelled are never offered an automatic label. Intervals shorter than six
seconds are not turned into candidates; interval edges are model edges, not note onsets. A
human "intermittent" answer is kept as such and is not given times.

## What the collection yields

| | Recordings | Hours | Share of time |
|---|---|---|---|
| verified reference (human excerpts) | 47 | 0.77 | 1.3% |
| strong agreement, guitar present | 0 | 0 | 0% |
| moderate agreement, guitar present | 218 | 2.02 | 3.4% |
| weak evidence, possibly guitar-free | 192 | 1.56 | 2.6% |
| conflict | 75 | 0.62 | 1.0% |
| no candidate of any kind | 478 | | |

218 recordings have an automatic "present" candidate; 75 of them belong to artists not used
in earlier training. Recording by recording, the fused view calls 13 "guitar through most",
112 "guitar in parts", 10 "intermittent", 12 "guitar absent" (all soundtrack pieces, weak
evidence only) and 796 "uncertain".

## Review queue

170 excerpts of 12 s (168 in the built batch, after two were dropped as versions of recordings already in it; about an hour at the pilot's pace), chosen where listening changes
something: 40 where the models contradict each other, 15 where the recogniser reports
distorted guitar and PANNs hears none, 1 where PANNs hears guitar the recogniser misses, and
113 as a seeded audit of automatic candidates, split evenly between what were then strong
"present", moderate "present" and weak "absent" (after the scope correction above, the first
two are one class). The audit measures the candidates' real error rate on
this collection; its "absent" third is also the cheapest way to obtain confirmed guitar-free
examples, which are what the corpus lacks most (64 so far). At most two excerpts come from
one recording and eight from one artist folder; stretches with a human label are skipped.

The queue is built into an ordinary review batch whose excerpts have the category
`evidence`; the review window has a filter per reason. Model output is not shown while
labelling, as before. Labels from this batch are not an unbiased sample (they were chosen
from model behaviour) and must stay out of headline accuracy figures, like the existing
"model flagged" excerpts.

## What is trustworthy and what is not

Trustworthy: human excerpts; identities matched by tag id or ISRC; recording-level credits
as "guitar is on this recording somewhere"; "both models say guitar" as a high-precision,
low-coverage candidate (measured 100% on 181 OpenMIC test clips and on 10 of the owner's
excerpts, which is a small number).

Speculative: every "guitar absent" candidate; the fused recording-level views; identities
matched by search with low confidence; guitar type from the models (not evaluated here);
the proposed training weights.

Not obtained: MedleyDB (access request needs the owner's account); AcoustID results (needs
a key); any guitar-type breakdown of the candidates.

## After the review queue

The owner answered the queue. Its outcome, the training and evaluation sets built from it
(`gmv evidence trainset`), their audit and the proposed experiment are in
[TRAINING_PLAN_V2.md](TRAINING_PLAN_V2.md). Nothing has been trained.
