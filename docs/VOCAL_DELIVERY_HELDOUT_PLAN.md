# Held-out test of clean against harsh vocals: the plan, fixed before looking

**Status: plan only.** No held-out answer has been read; nothing described here has run.
This document is committed **before** the test so that every decision below can be seen to
have been made without knowing the outcome. Any change to it after the test starts would
have to be stated as such.

## Why, and what it costs

Phase A4 ([VOCAL_DELIVERY_RECOGNITION.md](VOCAL_DELIVERY_RECOGNITION.md)) found that a
model can tell clean singing from harsh vocals in development artists it was not trained
on: balanced accuracy 0.92 for the random forest. That number comes from comparing three
models on the same predictions, so it is slightly flattered. The held-out artists have
never been used, and they are the only honest check.

**The cost.** The test is a one-time use. Once each held-out excerpt's answer has been
compared with a prediction, those artists can no longer serve as an untouched test **for
clean against harsh**. To keep them useful for later questions, the test reads only
whether each held-out answer is clean, harsh or something else. **It never reads whether a
harsh answer is a High or a Low Scream, nor whether an "other" answer is a Guttural or a
Sing-Scream.** A later High-against-Low test on these artists therefore stays possible.

## What is frozen

| Decision | Fixed as |
|---|---|
| Question | Is an excerpt's vocal delivery clean (Cleans) or harsh (High Screams or Low Screams)? |
| Features | `delivery_features_v1`: the 30 measurements of Phase A4, unchanged |
| Model | The random forest of Phase A4: 300 trees, bootstrap samples, weighted Gini, √(features) per split, at least 3 excerpts per leaf, seed `[20261125, 1000]` |
| Classes it learns | The same three as in development: Cleans, High Screams, Low Screams |
| Clean/harsh decision | The highest of the three probabilities: Cleans means clean, High or Low Screams means harsh. No threshold, no abstention |
| Training data | All 266 development excerpts used in Phase A4, with preprocessing (median imputation, indicators, scaling) and class weights fitted on them. The two guest-vocalist excerpts stay out |
| Code | The commit that adds the test command, after this plan is approved. Any later change to features or model makes this pre-registration void |

The forest was chosen over logistic regression on development results: 0.92 against 0.89
overall, and 0.91 against 0.85 within recordings that have both deliveries. That choice
is the exploratory step; the held-out test measures the chosen model honestly.

## Which held-out excerpts count

The held-out side has 12 artist groups and 69 excerpts: 59 answered and 10 No vocals.
These counts come from the excerpt list and answer totals only.

- **Scored (primary):** held-out excerpts whose answer is Cleans (truth: clean) or High
  or Low Screams (truth: harsh).
- **Left out of the score, reported separately:**
  - **Gutturals and Sing-Screams** (together, "other"): the model never learned them. The
    report gives only how many were called clean or harsh.
  - **No vocals:** the same, as a hint of what happens on instrumental passages.
  - **The held-out recording that shares an inventory leakage group with development
    recordings.** That is 2 excerpts. The development recordings are by other performers
    and share no audio with it, but the precaution costs little. It is scored separately.
- **Unplayable audio:** a held-out excerpt whose audio cannot be read is counted and
  reported, never guessed.

**One artist group dominates.** A reference artist accounts for 43 of the 69 held-out
excerpts (40 answered). The other 11 groups have 2 or 3 excerpts each, 19 answered in
all. Whatever the result, it will mostly describe that one vocalist. Its vocalist also
guests on the one development recording that was excluded, so the model has not been
trained on that voice.

## How it runs: predictions sealed before answers are read

1. **Predict, blind.** The model is trained on the 266 development excerpts. It measures
   every held-out excerpt and writes each prediction (clean or harsh, with the three
   probabilities) to a private file. No held-out answer is read in this step; the reader
   used for development cannot decode held-out lines, and this step does not call any
   other reader.
2. **Seal.** The SHA-256 of that predictions file is committed to this branch and pushed,
   so the predictions are fixed in public before step 3. The fingerprint reveals nothing
   about them.
3. **Score, once.** A separate step checks the file still matches its sealed fingerprint.
   It then reads the held-out answers through a reader that turns each answer into
   **clean**, **harsh**, **other**, **No vocals** or **unanswered**, and keeps nothing
   finer. It scores, and records that the held-out answers have been opened. It refuses to
   run a second time, so a quiet rerun after a change is not possible.
4. **Report.** Public: counts and metrics only, with artist groups numbered. Private:
   per-excerpt results in the ignored local-evidence folder.

Before step 1, the test command gets automated tests on generated data. They cover the
two-step order, the coarse reader, the seal check, the refusal to run twice, and that no
High/Low identity of a held-out answer is ever stored.

## What will be reported

Primary, on the scored excerpts:

- balanced accuracy of clean against harsh, recall of clean and recall of harsh,
  precision, the 2 × 2 confusion matrix;
- a 95% interval from resampling held-out artist groups 2,000 times (seed 20261125). With
  one dominant group this interval will be lumpy, and the report will say so.

Secondary, described but not used in the verdict except where stated:

- the same metrics **without the dominant group** (the other scored groups pooled: 10,
  after the precaution above);
- per artist group, numbered by size (excerpts, clean and harsh counts, right answers);
- within held-out recordings that have both a clean and a harsh excerpt;
- the "other" and No vocals excerpts: how many were called clean or harsh;
- the 2 leakage-precaution excerpts: right or wrong.

## The verdict, decided now

Let **BA** be the primary balanced accuracy, and **recalls** the clean and harsh recalls.
The rows are applied in this order: inconclusive, not confirmed, confirmed; anything else
is "weaker than in development".

| Verdict | Condition |
|---|---|
| **Inconclusive** | Fewer than 10 scored clean or fewer than 10 scored harsh excerpts. Numbers are still reported |
| **Confirmed** | BA ≥ 0.80, both recalls ≥ 0.70, and the lower end of the 95% interval ≥ 0.60 |
| **Weaker than in development** | Everything else: BA from 0.65 to 0.80, or a recall from 0.50 to 0.70, or a low interval |
| **Not confirmed** | BA < 0.65, or a recall < 0.50 |

**Breadth.** The verdict is stated as "mainly on one artist" unless the other scored groups
together have at least 5 clean and 5 harsh scored excerpts and reach BA ≥ 0.70 on their
own. If they have enough excerpts and fall below 0.70, the verdict is lowered one step
(confirmed to weaker, weaker to not confirmed).

For comparison, always answering one class gives BA = 0.50, and the development estimate
was 0.92.

## What happens after

- **Confirmed (broadly):** the clean-against-harsh distinction is worth a separate phase on
  live use, which would start with its own plan for latency and CPU cost. Nothing is
  integrated as part of this test.
- **Confirmed mainly on one artist, or weaker:** these held-out artists cannot be tested
  again for clean against harsh. The next step would be more development artists, then a
  test on **new** held-out artists.
- **Not confirmed:** report it plainly. The development result was then flattered by the
  artists it happened to contain, and features closer to the voice are needed.
- **In every case:** no model or feature is tuned on these results. The held-out artists
  count as used for clean against harsh. High against Low on them stays untouched.

## Not part of this test

- High against Low, Gutturals, Sing-Screams: not scored.
- Live or real-time use, the visualizer, Radiant: unchanged.
- The owner's dataset and answer history: read only, never changed.
