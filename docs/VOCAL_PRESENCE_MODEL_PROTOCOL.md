# Vocal presence models: the pre-registered protocol (Sonic Character, Phase A5.2)

**Status: pre-registered on 2026-11-26, before anything was run.** At the time of this
commit:

- no model has been trained, and no prediction has been made;
- no feature has been extracted;
- no fold file exists.

The owner approved the design (proposal revision 4) on 2026-11-26. Every rule below was
fixed without seeing any result. A later change is an **amendment**: appended at the end
with its date and reason, and made before the predictions it could influence exist.

**Amended ~~once~~ ~~twice~~ ~~three times~~ ~~four times~~ five times since registration** (Amendments 1
and 2, both 2026-11-26, before Stage 1; Amendments 3 and 3A, 2026-11-27, after the synthetic Stage 1
and before any real-data step; Amendment 3B, 2026-11-28, after Step 4's synthetic implementation and
before any real-data step; see the end). Superseded wording is struck through and kept visible.

**This is an exploratory comparison.** It measures and describes how three vocal-presence
detectors differ on the owner's development artists. It does **not** test hypotheses:
the cross-validated results are dependent (§12), so no claim of statistically confirmed
superiority, non-inferiority or equivalence is made from them. Confirmation needs a fresh,
untouched artist-disjoint final test, which does not exist yet (§14).

## 1. The question

The existing vocal-activity model reports a voice in most of the owner's instrumental
excerpts, and each false report is drawn as a vocal phrase (Phase A5.0 audit; see
[VOCAL_PRESENCE_ANNOTATION.md](VOCAL_PRESENCE_ANNOTATION.md)). The experiment asks:

1. How much false vocal activity does each detector produce on no-voice material?
2. How much of the confirmed voice does each find?
3. Does retraining the existing small network on the owner's interval labels already
   help (B1)?
4. Does a small convolutional network add anything beyond that (C)?

## 2. Data (aggregate; the private project stays in ignored `local-evidence/`)

- **Annotations:** the Phase A5.1 presence annotations are the only labels.
  - Confirmed intervals are voice; completed excerpts without marks are no voice; uncertain
    time is masked (none exists today).
  - Phase A3 delivery answers are never used as labels; they may only group results
    descriptively.
  - The three reused excerpts that disagree with A3 keep their presence label.
  - No annotation is changed during A5.2.
- **The paired development set:** 163 excerpts from 19 artist groups (the A5.1 registry's
  conservative groups, used as they are).
  - 15 groups have voice and all 19 have no-voice time.
  - 694 s with voice and 936 s without; the three largest no-voice groups hold 51% of the
    no-voice time.
- **The low-rate group** (8 excerpts, 22.05 kHz): the existing model cannot be built below
  about 23.7 kHz, so it is left out of training for every arm and out of every paired
  statistic. Model C alone is ~~reported on it, descriptively~~ scored on it, descriptively; its
  results stay private (Amendment 3, A3.5).
- **Validation:** 28 excerpts from 6 groups (52 s of voice from 4 groups), ~~used once and
  descriptively~~ scored once by A5.2, descriptively, after the owner's separate
  authorisation (Amendment 2) (~~§13~~ §14; editorial correction, Amendment 1, A1.4). Its
  provenance, including its earlier inspection during A5.1: Amendment 1, A1.2.
- **Final test:** none. The 7 reserved groups stay reserved and nothing is frozen.

## 3. The arms

| Arm | What it is |
|---|---|
| **B0** | `activity_mlp_v2` as shipped: its vocals output, on 0.6 / off 0.4. Weights `activity_v2.npz`, read-only from the owner's other checkout, never copied |
| **B0ₜ** | B0's raw vocals score with an operating point chosen by the §8 procedure |
| **B1** | The same 548 activity features (identical code and framing) → 128 ReLU → **1 sigmoid** ("voice present"): 70,401 parameters, trained from scratch on presence labels only |
| **C** | The causal log-mel CNN of §5: 154,641 parameters |

## 4. Audio, framing and pre-roll

- **Decoding:** `soundfile`, at the recording's native rate.
- **Pre-roll:** each arm is run from up to 5 s before the excerpt start, and scored only
  inside the excerpt.
- **B0 and B1:** the engine's own analysis pass (`gmv.character.research.gmvpass`).
  - Native rate; Hann window of 46.44 ms, 20 ms hop.
  - Frames below −65 dBFS give a score of 0.
- **C:** the mid channel (L+R)/2 is resampled to 22,050 Hz (`scipy.signal.resample_poly`).
  - Hann window of 1024 samples, hop 441 (20 ms).
  - 64 mel bands from 40 Hz to 11 kHz, log power with a floor.
  - Each band is standardised with the mean and spread of that model's training groups
    only.
- **Frame grid:** both front ends put frame centres 23.2 ms after each frame's start, every
  20 ms. A generated-audio test (§16) must confirm that the frames really align before
  any real audio is used.

## 5. Model C, exactly

Convolutions are causal in time.
- **2D layers:** time padding (2, 0), frequency padding (1, 1).
- **1D dilated layers:** left padding 2d.
- **Pooling:** max over 2 bands, stride 2, frequency only.

**FrameNorm** normalises each frame's activations over its own channels × frequency (2D)
or its own channels (1D), then applies a learnable per-channel scale and shift (ε = 1e-5).
It has no running statistics and no dependence on other frames or other excerpts. It
computes the same thing in training and in use, so the network is causal in training too.
It replaces BatchNorm, which mixes future frames and other excerpts into its statistics
while training.

| # | Layer | In→out | Kernel / dilation | Bias | Then |
|---|---|---|---|---|---|
| 1 | Conv2d | 1→16 | 3×3 / 1 | no | FrameNorm, ReLU |
| 2 | Conv2d | 16→16 | 3×3 / 1 | no | FrameNorm, ReLU, pool 64→32 bands |
| 3 | Conv2d | 16→32 | 3×3 / 1 | no | FrameNorm, ReLU |
| 4 | Conv2d | 32→32 | 3×3 / 1 | no | FrameNorm, ReLU, pool 32→16 |
| 5 | Conv2d | 32→64 | 3×3 / 1 | no | FrameNorm, ReLU |
| 6 | Conv2d | 64→64 | 3×3 / 1 | no | FrameNorm, ReLU, pool 16→8 |
| 7 | Conv1d, input 64×8 = 512 | 512→64 | 1 / 1 | no | FrameNorm, ReLU → h⁽⁰⁾ |
| 8–11 | Conv1d, residual | 64→64 | 3 / 2, 4, 8, 16 | no | h⁽ᵏ⁾ₜ = h⁽ᵏ⁻¹⁾ₜ + ReLU(FN(Σⱼ₌₀..₂ Wⱼ h⁽ᵏ⁻¹⁾ₜ₋ⱼ·dₖ)), zeros before the start |
| 12 | Conv1d | 64→1 | 1 / 1 | **yes** | dropout 0.2 before it; sigmoid |

**Why biases are on or off:**
- Layers 1–11 have no bias: the normalisation's per-channel shift supplies the offset.
- The output layer has a bias, because it sets the base log-odds and no normalisation
  follows it.

**Size:** 153,552 weights + 1 bias + 1,088 normalisation parameters = **154,641**.

**Receptive field:** 1 + 6·2 + 2·(2+4+8+16) = **73 frames**.
- That is 1.44 s of frame centres, or about 1.49 s of audio.
- It ends 23.2 ms after the newest frame centre, the same look-ahead as B0's window.

**Running cost:** 1,565,760 multiply-adds per frame.

## 6. Training (B1 and C share one objective)

**Which frames count.** For a model trained on groups 𝒯:
- **Label:** a frame (on the 20 ms grid inside the labelled 10 s) has y = 1 if its centre
  lies inside the union of confirmed intervals, and 0 otherwise.
- **Excluded frames:**
  - inside uncertain time;
  - within ±0.3 s of an interior boundary (§10);
  - below −65 dBFS (the engine's level rule);
  - in the pre-roll.
- **Per group and class:** F₍g,c₎ is a group's eligible frames of class c, and n₍g,c₎ their
  number. A group belongs to class c (Aᶜ) when n₍g,c₎ ≥ 100 frames (2 s).

**Loss:**

  L = ½ · mean over g in A¹ of [ mean over F₍g,1₎ of ℓ(p, 1) ]
    + ½ · mean over g in A⁰ of [ mean over F₍g,0₎ of ℓ(p, 0) ]

- ℓ is binary cross-entropy computed from logits.
- Equivalently, the frame weight is w = 1 / (2·|Aᶜ|·n₍g,c₎). Every group's total weight in a
  class is exactly 1/(2·|Aᶜ|).
- A group with one class counts only in that class. A group with less than 2 s of a class
  has weight 0 there and still counts in its other class.
- Excluded frames have weight 0 and are not counted in n₍g,c₎.

**Batches reproduce the objective exactly:**
- **C:** each epoch visits every training excerpt once, in a seeded order, in batches of 8
  whole excerpts. Each excerpt carries up to 1.5 s of preceding context, whose frames have
  weight 0. With J batches per epoch, the batch loss is J · Σ w·ℓ. This is not normalised
  per batch, because that would reweight groups by batch composition.
- **B1:** frames are independent. Each epoch is a seeded permutation of all frames with
  weight > 0, in batches of 1024, with the same formula.

**Optimisation:**

| | C | B1 |
|---|---|---|
| Optimiser | AdamW | The existing hand-written Adam |
| Learning rate | 1e-3, cosine to 1e-5 | 1e-3, then 3e-4 for the second half |
| Regularisation | Weight decay 1e-4 | L2 1e-2 |
| Epochs | 30 (about 600 steps) | 60 |
| Early stopping | None | None |
| Initialisation | | Fresh, by the existing scheme |

**Augmentation (C only, all label-preserving):**
- ±6 dB gain;
- a smooth tilt of up to ±3 dB across the band range;
- up to 6 masked mel bands;
- mixing two no-voice excerpts of the **same** group.

No pitch or tempo change, no random crops, no external data.

## 7. Seeds, determinism, environment

- **Training seeds:** each training seed is the first 8 hex digits of
  SHA-256(`gmv-a5.2:{arm}:{outer}:{inner}:{seed_index}`); a final model uses
  `gmv-a5.2:{arm}:final:final:{seed_index}` (Amendment 3, A3.3).
  - ~~Seed indices: 0–1 for inner models, 0–4 for outer and final models.~~ Seed indices: 0–1
    for inner models, 0–4 for outer and final models under policy C; 0 and 0–2 under policy D
    (§15).
  - Fold assignment uses `gmv-a5.2-inner-folds`, and the descriptive bootstrap uses
    `gmv-a5.2-bootstrap`.
  - Seed format, enumeration, collisions, replays and paired gate runs: Amendment 3, A3.3.
- **Determinism:** `torch.use_deterministic_algorithms(True)`, with fixed thread counts.
- **Recorded per run:** the commit, a fingerprint of the annotation histories, manifests
  and fold files, and the environment lock.
- **Environment:** CPU only (no CUDA, NVIDIA or ROCm), in a separate Linux experiments
  environment (`experiments/presence/README.md`, `requirements-lock.txt`). The application
  never imports it.

## 8. Nested, group-disjoint operating points

For each of the 19 outer groups g:
1. Set g aside completely.
2. Assign the other 18 groups to 5 inner folds (§9).
3. Train inner models on 4 folds and predict the fifth, so that all 18 groups get
   group-disjoint predictions.
4. ~~Choose t by maximising J(t) on those predictions.~~ Choose t by maximising J̄(t), the mean
   over the inner seed indices I of the policy in force (§15) of J(t) computed on each inner
   seed's group-disjoint predictions; under policy D, I has one index and J̄ = J. Scores are
   never averaged between models.
   - J(t) = ½ · (mean group recall + mean group specificity), each group counting once
     in its class (§11).
   - Candidates: on = t, off = t − 0.2, t ∈ {0.30, 0.35, …, 0.80}.
   - Ties go to the t closest to 0.5, then the higher one.
5. ~~Train the outer model on all 18 groups.~~ Train one outer model per seed index in O on all
   18 groups.
6. ~~Predict g once, and score it at t.~~ Each predicts g once and is scored on its own at t.

- **B0ₜ** uses J(t) on B0's scores for the 18 groups. B0 is fixed, so its scores are
  already group-disjoint.
- ~~**Final models** (for validation) are trained on all 19 groups. Their operating point
  comes from the 19 outer predictions.~~ **Final models** (for validation): one final model per
  seed index in O, each trained on all 19 groups. Their operating point t_final maximises the
  mean over s ∈ O of J computed on seed index s's 19 outer predictions (same candidates and tie
  rule).
- **The same post-processing for every arm:** frames below −65 dBFS forced to 0,
  hysteresis, then the unchanged `VocalContour` for phrases.

## 9. Inner folds: class-aware, capacity-limited, deterministic

**Inputs:** per group, voiced time V and no-voice time N, both after exclusions.
- **Type:** *both*, *voice-only* or *no-voice-only*, using the 2 s minimum.
- **Targets:** V* = ΣV/5 and N* = ΣN/5.
- **Capacity:** 4 groups per fold.

**Placement:**
1. **Order:** single-class groups first, by type and then by descending total time. Then
   two-class groups, by descending total time. Ties are broken by SHA-256 of the seed and
   the group key.
2. **Candidates for every placement:** the folds holding fewer than 4 groups.
3. **A single-class group** goes to the candidate fold with the fewest groups of its type,
   then the least time of its class, then the seeded fold order.
4. **A two-class group** goes to the candidate fold that minimises
   ((V_f + V)/V*)² + ((N_f + N)/N*)².

**Checks:**
- every inner training partition keeps at least 65% of the voiced time and at least 65% of
  the no-voice time;
- no fold holds more than ⌈n_type/5⌉ single-class groups of one type;
- no fold holds more than 4 groups.

**If a check fails:**
- Up to 99 retries; retry r orders the two-class groups by SHA-256 of (seed, r, key).
- If all fail, **stop before any training** and report in aggregate.
- Nothing is relaxed automatically; any change is an amendment.

**Records:** fold files are private; their SHA-256 values are recorded here as an
amendment before any training. Fold balance is published only in aggregate.

## 10. Label boundaries

The owner tapped boundaries live and never adjusted them, so marks are probably late by
about 0.1–0.3 s. That delay is not assumed to be the same everywhere, and no label is
shifted or edited.

- **Training:** frames within ±0.3 s of interior boundaries are excluded.
- **Main evaluation:** the same ±0.3 s collars are excluded for every arm. Excerpt edges are
  not boundaries.
- **Events:**
  - **References:** confirmed intervals, merged across gaps shorter than 0.25 s.
  - **Predictions:** `VocalContour` start and end.
  - **Onset collar:** ±0.5 s.
  - **Offset collar:** the larger of 0.5 s and 20% of the reference length.
  - **Timing** is scored only on interior boundaries.
- **Sensitivity** (reported, not used for screening):
  - frame collars 0, ±0.15 and ±0.5 s;
  - reference boundaries moved 0.1, 0.2 and 0.3 s earlier (computed on the fly);
  - event collars 0.25 s and 1.0 s.

## 11. Metrics

All metrics are computed after post-processing, with collars excluded, at each group's own
outer-fold operating point.

- **Recall** R_g: detected voice time divided by voice time, for groups with at least 2 s
  of voice.
- **False vocal activity** FPA_g: 60 × false vocal seconds ÷ no-voice seconds, in seconds
  per minute of no-voice time, for groups with at least 2 s of no-voice time.
- **Specificity** S_g = 1 − false vocal seconds ÷ no-voice seconds.
- **Group averages (the main view):** FPA_macro and R_macro. A group with one class counts in
  that class only and is never dropped.
- **Duration-weighted (pooled):** the same quantities over all frames, ~~always shown beside
  the group averages~~ shown beside the group averages for the main results, as ranges near
  their natural limits; the duration-weighted values of breakdown cells are private
  (Amendment 3, A3.5).
- **Also reported:**
  - missed voice time;
  - ROC-AUC and PR-AUC, as defined in Amendment 3, A3.5;
  - phrase metrics: detected and missed events, false phrases per minute of no-voice time,
    onset and offset error (onset and offset error published only as group averages of each
    group's mean absolute error; Amendment 3, A3.5);
  - no-voice material split ~~into vocal recordings and instrumental recordings,~~ by selection
    route (how each excerpt was chosen, which is not the same as whether its recording is
    instrumental; Amendment 3, A3.5), and by the metadata category of each excerpt's
    selection;
  - the fixed 0.6 / 0.4 operating point for every arm.

## 12. Reporting policy (exploratory)

**Why there are no tests.** In leave-one-group-out cross-validation each outer model shares
17 of its 18 training groups with every other. The per-group results are therefore
correlated, in a way that cannot be estimated from one cross-validation run (Bengio and
Grandvalet 2004; Nadeau and Bengio 2003). A sign-flip or bootstrap procedure that treats
them as independent does not control its error rate, and enumerating sign patterns does not
make it exact. **No p-values are reported** from the development results.

**What is reported:**
- **Effect sizes** (paired, group-averaged) for C vs B0ₜ, and descriptively for C vs B0 as
  shipped, B1 vs B0ₜ and C vs B1:
  - D_FA = FPA_macro(C) − FPA_macro(B0ₜ), and the ratio when §12.1 allows it;
  - Δ_R = R_macro(C) − R_macro(B0ₜ);
  - missed voice time.

  False activity and missed voice are always reported separately.
- **Both group-averaged and duration-weighted** results.
- **Per-group results** privately. In public: counts of groups where C is lower or higher,
  with no test~~, and quartiles of the per-group differences~~, shown as "0–2" or "17–19" at the
  ends and withheld when the disclosure check requires it (Amendment 3, A3.5).
- **Training-seed spread:** ~~the effect sizes for each of the 5 single-seed outer models.~~ ~~the
  effect sizes for each seed-index aggregate s ∈ O (5 under policy C, 3 under policy D).~~ whether
  the requirement held in every seed-index aggregate s ∈ O (5 under policy C, 3 under policy D):
  yes or no; each aggregate's effect sizes stay private (Amendment 3B).
  Aggregate s is the cross-validation result in which every outer fold uses its outer model
  with seed index s; it combines 19 models and is not a single trained model. The primary
  effect sizes are the means of these aggregates; the ratio is a ratio of these means.
  Individual outer models' results stay private; in public only the counts of models for which
  the requirement holds, fails, ties or is undefined.
- **Leave-one-group-out influence:** each effect size recomputed without each group, from the
  groups' means over O, with models and thresholds fixed. This removes the group's test result,
  not its influence on training. Published only as the number of recomputations in which the
  requirement holds, plus pass or fail; each recomputation stays private (Amendment 3, A3.5).
- **Operating-point stability:** the distribution of chosen t across folds, the agreement
  between the inner seeds (policy C only; policy D has one inner seed), and the effects at
  t ± 0.05.
- **Descriptive group-bootstrap intervals** (10,000 paired draws, percentile method, group
  as the unit), **kept private** (Amendment 3, A3.5), always with this text: *"Descriptive only. These treat dependent
  cross-validation results as independent, ignore training-set variation, and are
  probably too narrow. They are not evidence of superiority, non-inferiority or
  equivalence."*
- The report names the policy in force and states that policy D's seed criterion is less
  demanding.
- Every public figure passes the disclosure check of Amendment 3, A3.5.

**Wording:**
- Allowed: "C showed X% lower group-averaged false vocal activity than B0ₜ in development
  cross-validation."
- Not allowed: significant, confirmed, proven, non-inferior, equivalent, no difference,
  generalises.

### 12.1 When the baseline's false activity is low

The ratio FPA_macro(C) / FPA_macro(B0ₜ) is used only when FPA_macro(B0ₜ) ≥ 3 s/min, the
readiness target fixed in Phase A5.0.

Below that, including zero:
- the absolute difference D_FA is the main comparison;
- the 50% criterion is recorded as **not applicable**;
- no constant is added to make a ratio defined;
- per-group ratios are never used.

A ratio interval is ~~shown~~ computed only when the point-estimate denominator is at least
3 s/min, together with the share of bootstrap draws where it fell below; both stay private
(Amendment 3, A3.5).

## 13. Screening rule: what to do next (not an inference)

| Result | Rule (point estimates, development, C against B0ₜ) |
|---|---|
| **Promising: carry forward to confirmation** | All of: (a) the false-activity ratio is ≤ 0.50 — ~~or, under §12.1, FPA_macro(C) ≤ 3 s/min and D_FA ≤ 0~~ (the low-baseline case is replaced by Amendment 1, A1.1); (b) Δ_R ≥ −0.05; (c) ~~the direction of D_FA and the sign of Δ_R + 0.05 hold in all 5 seeds and in every leave-one-group-out recomputation~~ the direction of D_FA and the sign of Δ_R + 0.05 (D_FA < 0 and Δ_R + 0.05 ≥ 0) hold in every seed-index aggregate of the policy in force (5 under policy C, seed indices 0–4; 3 under policy D, 0–2) and in every leave-one-group-out recomputation |
| **Mixed** | Lower false activity than B0ₜ without meeting "promising" |
| **Not promising** | False activity not lower than B0ₜ, or Δ_R < −0.05 |

~~B1 is screened against B0ₜ in the same way.~~ B1 is screened against B0ₜ in the same way,
under the same policy as C (§15), so both are held to the same number of seed-index aggregates.
If B1 alone is promising, the cheaper retrain
is the candidate to carry forward. When FPA_macro(B0ₜ) is below 3 s/min, the
classification in Amendment 1, A1.1 applies instead of this table.

**The thresholds are owner decisions:**
- **50% reduction:** the smallest change expected to be clearly visible.
- **5-point recall margin:** a third of A5.0's 15% missed-voice budget, about 0.5 s per
  10 s of singing.

## 14. Validation and the future confirmation

- ~~**Validation is opened once,** only after the owner's separate authorisation:~~
  **A5.2 validation scoring occurs once, after the owner's separate authorisation.** This
  scoring use is distinct from the earlier A5.1 annotation-history inspection documented
  in Amendment 1, A1.2. In order (Amendment 2):
  1. final models and operating points are frozen;
  2. predictions for B0, B0ₜ, B1 and C are sealed with SHA-256;
  3. ~~the answers are read once.~~ the validation answers are then accessed for this
     scoring pass.
- Each arm's result is the mean over its final models (one per seed index in O: 5 under
  policy C, 3 under policy D), each scored on its own at t_final; each model's value and the
  range are also reported, subject to the disclosure check. Predictions are not averaged
  between models.
- **It is descriptive only:**
  - false activity on the material it contains (no guitar-type negatives);
  - recall and missed voice (52 s from 4 groups);
  - phrases;
  - per group and per excerpt privately.
- ~~**It cannot confirm anything,** even in principle. With 5 no-voice groups and 4 voice
  groups, the smallest attainable one-sided sign-flip p-values are 1/32 and 1/16, both
  above 0.025, even if every group favoured C.~~ Replaced by Amendment 1, A1.3.
- **Confirmation needs an untouched final test:** at least 20 eligible never-used groups
  (A5.1's rule), scored by fixed, frozen models.
  - Only 7 exist, so about 13 or more new artists must be collected first.
  - Its size and tests are planned separately, before it is frozen.
  - Until then the 7 reserved groups stay reserved.

## 15. Compute policy and gates

**Policy C:**
- inner: 19 × 5 × 2 seeds = 190 models;
- outer: 19 × 5 seeds = 95;
- final: 5;
- in total 290 CNN models, plus the same counts for B1.

**Estimates:**
- **Per CNN model:** 1.5–10 minutes, central about 3, so about 6–40 h of ~~CPU~~ process-time
  in total (Amendment 3, A3.8).
- **B1:** about 0.5–2 h.
- **Caches:** about 120 MB in ignored `local-evidence/`.
- **Memory:** under 8 GB with 4 processes.

**Gates** (decided from timing and training loss only, before any held-out prediction):
- **Timing:** if ~~the Stage 2 timing run projects more than 16 h for policy C, use
  **policy D** instead: inner 1 seed, outer and final 3 seeds, 155 CNN models.~~ H_proc for the
  C arm under policy C exceeds 57,600 s (16 h of process-time; Amendment 3, A3.1 and A3.8), both
  B1 and C use **policy D** instead: I = {0}, O = {0, 1, 2}; 155 models per arm.
- **Convergence:** ~~if training loss has not levelled off within 30 epochs on development
  training data, an amendment may raise the epoch count.~~ each arm's epoch count is decided at
  the gates by the matched complete-recipe rule of Amendment 3, A3.2; any outcome outside that
  rule stops for the owner's review.

**Visibility:** long runs are visible in a terminal.

## 16. Tests required before any real run

All on generated audio and generated projects:

- **Alignment:**
  - B0's engine pass and C's front end at 22.05, 44.1 and 48 kHz have the same frame
    counts and centres within 1 ms;
  - an impulse peaks in the same frame;
  - resampling keeps timing within one sample.
- **Causality and normalisation:**
  - changing input after frame t leaves outputs up to t bit-identical, in training and in
    inference mode;
  - training and inference modes give equal outputs;
  - output does not depend on batch composition;
  - streaming equals batch;
  - the same seed gives identical weights.
- **Loss:**
  - every group's total weight per class is exactly 1/(2·|Aᶜ|), whatever its length;
  - one epoch reproduces L;
  - single-class and excluded frames behave as specified.
- **Folds:**
  - deterministic;
  - every group appears exactly once, and the outer group never;
  - capacity, balance and spread hold;
  - retries are deterministic, and failure stops the run;
  - reports hold no group keys.
- **Metrics and reporting:**
  - collars, the 2 s minimum, single-class handling and event collars;
  - the low-denominator rule;
  - the bootstrap's caveat text is always attached.
- **Further required tests:** Amendment 3, A3.10.

## 17. Data protection

- ~~Only the development and validation presence histories are read.~~ Of the annotation
  histories, only the development and validation presence histories are read. The other project
  files each step may open are listed, as a ceiling, in Amendment 3, A3.4. ~~Validation is read
  only in Stage 4.~~ This was incomplete; see Amendment 1, A1.2 for the validation data's
  prior inspection and the access rule from Stage 1 on.
- No final-test path is opened.
- Audio, titles, identifiers, paths, features, weights, fold files, predictions and
  per-group results stay in ignored `local-evidence/`.
- Tracked documents carry aggregates only.
- Every public figure passes the disclosure check of Amendment 3, A3.5; withheld figures all
  look the same.
- The host paths of protected inputs appear only in private run records; tracked text uses
  placeholders.
- No external audio or dataset is downloaded. The only download is the CPU-only PyTorch
  build for the separate experiments environment.
- Annotation histories, manifests, the split, checksums and backups are never modified.

## 18. Stages

| Stage | What |
|---|---|
| 0 | This protocol, and the experiments environment |
| 1 | The harness and the §16 tests~~, fold files with their hashes recorded here, and an aggregate check for overlap with MUSDB18-HQ training material~~, on generated data. The aggregate check for overlap with MUSDB18-HQ training material and the fold files, with their hashes recorded here, follow as steps 2 and 3 (Amendment 3, A3.9) |
| 2 | B1 and C, and the timing and convergence gates |
| 3 | The nested development run: predictions sealed, then the §12 report |
| 4 | ~~Validation, descriptive,~~ The one-time A5.2 validation scoring, descriptive, after separate authorisation (Amendment 2) |
| 5 | Documentation and a draft pull request with the screening result |

The authorisation steps and review gates within and between these stages: Amendment 3, A3.9.

## Amendments

### Amendment 1 — 2026-11-26, before Stage 1

**State when made:**
- the protocol had been committed (694fad6);
- no model had been trained, no prediction made and no feature extracted;
- no fold file existed;
- nothing had been run on the owner's music.

The amendment was written from the protocol alone; no annotation history was read for it.
It does not change the model architecture, data partitions, seeds, fold policy, training
procedure, compute policy or any other acceptance criterion.

#### A1.1 The low-baseline screening rule

**Why:** when FPA_macro(B0ₜ) is below 3 s/min, the registered rule (§13) called C
"promising" when FPA_macro(C) ≤ 3 s/min and D_FA ≤ 0. That counted *identical* false
activity as an improvement.

**Replacement.** When FPA_macro(B0ₜ) < 3 s/min (including zero), C is classified against
B0ₜ by the first matching row:

| # | Class | Rule (point estimates, development) | Consequence |
|---|---|---|---|
| 1 | **Recall below margin** | Δ_R < −0.05 | Not promising: retain B0 |
| 2 | **Worsened** | D_FA > 0: C has more false vocal activity than B0ₜ | Not promising: retain B0 |
| 3 | **Improved** | D_FA < 0 (strictly lower), and FPA_macro(C) ≤ 3 s/min, and Δ_R ≥ −0.05, and the robustness requirement of §13 (c): D_FA < 0 and Δ_R + 0.05 ≥ 0 hold ~~in all 5 seeds~~ in every seed-index aggregate of the policy in force (5 under policy C, 3 under policy D) and in every leave-one-group-out recomputation | Promising: carry forward to confirmation |
| 4 | **Maintained** | D_FA = 0: the same false activity, both arms meeting the 3 s/min readiness target | **Not an improvement.** Retain the simpler baseline. C could be carried forward only for another benefit that was pre-declared and independently demonstrated. **This protocol pre-declares no such benefit, so "maintained" means retaining B0.** |
| 5 | **Mixed** | Anything else (for example, D_FA < 0 but the robustness requirement fails) | Not carried forward on this evidence |

**What stays the same:**
- The 50% relative-reduction criterion stays **not applicable** in this case (§12.1).
- No constant is added to any denominator.
- The 3 s/min readiness target is unchanged.
- "Same" means equal point estimates of FPA_macro, as measured on the 20 ms frame grid; no
  tolerance band is introduced.
- **B1 against B0ₜ** is classified with the same rows.
- When FPA_macro(B0ₜ) ≥ 3 s/min, the §13 table applies unchanged. There, equal false
  activity already counts as "not lower", so it is not promising.

#### A1.2 Validation provenance

**Why:** §17 said validation is read only in Stage 4. That is not a complete account of
earlier access.

**Record:**
- **Already inspected.** The 28-excerpt validation partition was complete before A5.2 was
  registered. Its annotation history was read during A5.1 completion and integrity checks,
  and its aggregate statistics were inspected:
  - counts of excerpts with and without voice;
  - artist groups;
  - voiced and no-voice seconds;
  - interval counts and coverage;
  - negative categories.
- **Used for planning.** Those aggregate facts informed the feasibility assessment and the
  design of A5.2, including its descriptive-only role for validation.
- **Not used for fitting.** Validation was not used, and will not be used, for model
  training, feature selection, threshold tuning or model selection.
- **Not pristine.** No claim is made that validation is unseen or pristine.
- **From Stage 1 through Stage 3,** the validation annotation history is not read and no
  individual validation label is used. Any loader call is restricted to the development
  partition.
- **Stage 4 is unchanged.** It is a one-time descriptive scoring pass:
  1. models and operating points are frozen;
  2. predictions are sealed with SHA-256;
  3. the validation answers are ~~read~~ accessed for this scoring pass only after the
     owner's separate authorisation (wording aligned by Amendment 2).

#### A1.3 Validation's role, without the withdrawn test framework

**Why:** revision 4 of the proposal withdrew formal significance testing. §14 still
justified validation's role through minimum attainable sign-flip p-values, as if a
sign-flip test were part of the plan. It is not.

**Replacement.** Validation is descriptive only. Its small number of artist groups and
positive excerpts cannot provide sufficiently reliable confirmation for the intended
claims, so no hypothesis test or confirmatory conclusion is planned. The descriptive-only
reporting restriction of §12 and §14 is unchanged.

§12's explanation of why development results are not tested still mentions sign-flip
procedures, as the reason they are *not* used. It is kept as written.

#### A1.4 Editorial

§2 referred to §13 for validation; the right section is §14. The original reference is
struck through and the correct one shown beside it.

### Amendment 2 — 2026-11-26, before Stage 1

**State when made:** unchanged from Amendment 1:
- no model had been trained, no prediction made and no feature extracted;
- no fold file existed;
- nothing had been run on the owner's music.

The amendment was written from the protocol alone; no annotation history was read for it.

**Why:** the review of Amendment 1 (commit b6bf3f5) found that Stage 4 was still described
as validation being "opened once" and its answers being "read once". That wording could
suggest the validation annotations had never been inspected, which contradicts the A5.1
inspection that A1.2 records.

**Clarification:** Stage 4 is the one-time **A5.2 validation-scoring pass**. It is not the
first inspection of the validation annotations.

**Replacement wording in §14:** "A5.2 validation scoring occurs once, after the owner's
separate authorisation. This scoring use is distinct from the earlier A5.1
annotation-history inspection documented in Amendment 1, A1.2."

**The order is unchanged:**
1. final models and operating points are frozen;
2. predictions for B0, B0ₜ, B1 and C are sealed with SHA-256;
3. the validation answers are then accessed for this scoring pass.

**Where the wording changed** (originals struck through, kept visible):
- the amendment note at the top;
- §2 (validation);
- §14 (the first bullet and its third step);
- §18 (the Stage 4 row);
- Amendment 1, A1.2 (the last bullet's third step).

**Unchanged:**
- models and operating points are frozen before scoring;
- predictions are sealed before the answers are accessed;
- the owner's separate, explicit authorisation is required;
- validation is descriptive only: no confirmatory claim and no claim of generalisation;
- the validation history is not read in Stages 1 to 3;
- every other part of the protocol.

### Amendment 3 — 2026-11-27, after the synthetic Stage 1, before any real-data step

**State when made:**
- the protocol and Amendments 1 and 2 had been committed;
- Stage 1's synthetic harness and the Step 1b MUSDB18-HQ matcher had been built and tested on
  generated data only, and accepted by the owner (review RG1 passed);
- no model had been trained on project data, no prediction made and no feature extracted from
  project audio;
- no fold file existed, and the MUSDB18-HQ overlap check had not been run on real inputs;
- nothing had been run on the owner's music.

**How it was made.** It records the owner's decisions of 2026-11-26 and 2026-11-27 on the
readiness proposals (revisions 3 to 4.4), reviewed in six groups. It was written from the
protocol, those proposals, the tracked code and synthetic tests; no annotation history,
manifest, registry, recording or other project data was read for it.

**Why.** The registered text left many operational details open (the clock behind the 16 h
gate, policy D's wording, normalisation, augmentation, convergence, seeds, data access,
disclosure control, the MUSDB check, timing). Each had to be fixed before the data it could
influence is seen.

**Standing qualifications.**
- **Operational thresholds.** Every numeric threshold introduced here is an operational choice,
  not a calibrated or empirically validated value. Synthetic simulations used to examine them
  illustrate rule behaviour only; they are not evidence about training or privacy.
- **Rules that depend on tests.** A rule marked *[tested before use]* takes effect only after
  the synthetic tests named for it in A3.10 pass and the review gate that covers it (A3.9) has
  reviewed them.
- **No authorisation.** Nothing in this amendment starts a step or permits access to project
  data. Each step needs its own authorisation (A3.9).

#### A3.1 Screening frame

- **The 16 h gate measures process-time** (decision C1). Process-time is the elapsed time on a
  monotonic clock from the start of a job's process to its exit, with the job running alone
  (A3.8) and the registered thread settings. It includes interpreter start, imports, loading,
  statistics, model building, training, predictions, checkpoints and teardown; it excludes the
  gate-only end-of-epoch loss evaluation. H_proc is the C arm's one-time preprocessing plus the
  sum of the process-times of all its policy-C models, projected from the step 5 measurements
  (A3.8). CPU time (user plus system) is recorded for information only and is never compared
  with the gate. If H_proc exceeds 57,600 s, policy D applies. *Why:* the registered estimate is
  elapsed-minute arithmetic, while CPU time is up to about 4× larger and unstable (a Step 1 toy
  job used 2.6 s of CPU in 0.55 s elapsed, from idle library threads).
- **B1 follows C's policy** (C7, option b). The policy the gate selects applies to both arms,
  with the same inner and outer seed sets for training, threshold selection, effect sizes,
  robustness, the final threshold and validation: policy C, I = {0, 1}, O = {0, …, 4}; policy D,
  I = {0}, O = {0, 1, 2}.
- **Robustness uses seed-index aggregates** (B12, option A): every seed-index aggregate under the
  policy in force (5 under policy C, 3 under policy D) and every leave-one-group-out
  recomputation. Individual outer-model results stay descriptive and private (§12).
- **Policy D wording** (B13): the replacements in §7, §8 (steps 4-6 and final models), §12
  (training-seed spread, leave-one-group-out influence, operating-point stability, the policy
  sentence), §13 (condition (c) and the B1 sentence), §14, §15 and Amendment 1, A1.1 row 3.
- **Unchanged:** the 50% and 5-point thresholds, the 3 s/min readiness target, the
  average-single-model estimand; no ensemble.

#### A3.2 Model definition and training length

- **B1's normalisation** (B14): B1 standardises its 548 features with B0's stored `mean` and
  `scale` (read only; the B0 file's SHA-256 is recorded), exactly as B0 does. Only the
  normalisation is inherited; the weights are trained from scratch. This is disclosed as
  MUSDB-derived.
- **C's normalisation** (B15): each C model computes its band statistics from its own training
  excerpts only, before augmentation: the pooled mean and the population standard deviation
  + 1e-6 over non-quiet frames centred inside the excerpt (collar and uncertain frames included;
  pre-roll and context excluded). Held-out groups, inner held-out folds, validation and the
  low-rate group never contribute.
- **Mixing** (B16): power-normalised mixing (M1) with probability 0.25 and gain G ~ U[−6, 0] dB;
  the partner is a no-voice excerpt of the same group from the same model's training set; mixing
  is skipped when E_a = 0 or E_b ≤ 1e-20·E_a; voiced excerpts are never mixed. *Limitation:*
  texture cues remain, and the C-versus-B1 comparison confounds architecture, front end,
  augmentation and optimiser.
- **C's optimiser** (B17): AdamW with β = (0.9, 0.999), ε = 1e-8, no AMSGrad; weight decay 1e-4
  on all parameters; a per-step cosine from 1e-3 to 1e-5 over the model's own K steps; no
  warm-up and no gradient clipping; PyTorch 2.9.1 default initialisation, seeded per §7.
- **Augmentation draws** (V7): gain U[−6, +6] dB and tilt U[−3, +3] dB (band b scaled by
  τ·(2b/63 − 1) dB) for every excerpt in every epoch; one masked block of width w ~ U{0…6}
  starting at s ~ U{0…64 − w}, set to 0 after standardisation; each draw has its own seeded
  stream; order: mix → gain → tilt → log → standardise → mask.
- **Numerics** (V8): C's statistics in float64 (population standard deviation + 1e-6), inputs in
  float32. B1 matches B0's existing arithmetic: the training script's float32 path in training,
  and the engine's activity-model path in inference.
- **Training length: the matched complete-recipe rule** (B18, with S3 and B19). Per arm a ∈ {C,
  B1}, with registered length E (30 for C, 60 for B1), E⁺ = 1.5E (45, 90) and E⁺⁺ = ⌈2.25E⌉ (68,
  135), gate sizes z ∈ {outer, inner} and gate seeds s ∈ {0, 1}:
  - L_{z,s}(H) is the §6 objective after the last epoch of the complete recipe with horizon H,
    on the run's own training data, in inference mode, without augmentation, accumulated in
    float64. The longer schedule is a **separate complete recipe with the same seed identity**,
    not a continuation of the shorter run.
  - Δ_{z,s} = (L(E) − L(E⁺)) / L(E). ε_z = |L_{z,0}(E) − L_{z,1}(E)| / ½(L_{z,0}(E) + L_{z,1}(E)),
    an operational reference from two seeds, not a confidence bound. β_z = max(ε_z, δ) with
    **δ = 0**. Stage 2 uses Δ′, ε′, β′ with (E⁺, E⁺⁺) in place of (E, E⁺).
  - EXT_z: both seeds at size z have Δ > β_z. S4_z: both seeds at size z have Δ < −β_z.
  - r = (m₁ − m₂)/m₁ over the means of the last two 5-epoch windows of one run.
  - **Checks on every run:** S1 any non-finite loss, parameter or gradient → stop; S2 any L ≤ 0,
    m₁ ≤ 0 or a non-positive two-seed mean → stop; S5 a failed exact replay or matched-start check
    (A3.3) → stop; S3 both seeds of the same arm, size and horizon with r < −0.05 → stop, and any
    single run with r < −0.02 is a warning reported at RG3.
  - **Stage 1** (E and E⁺, both seeds, both sizes): S4_z at either size → stop; otherwise EXT_z at
    any size → stage 2; otherwise adopt E. Seed disagreements are reported.
  - **Stage 2** (E⁺⁺ at both sizes): S4′_z at either size → stop; otherwise EXT′_z at any size →
    stop (the confirmation failed); otherwise adopt E⁺ for all of the arm's inner, outer and final
    models.
  - Then H_proc is recomputed with C's adopted length and the 16 h gate applied (A3.1); the policy
    applies to both arms. **Policy-flip safeguard:** if adopting E⁺ is what moves the gate from
    policy C to policy D, stop for the owner's review before Stage 3.
  - A stop always means owner review; nothing is selected automatically. No path adopts a recipe
    after both seeds at one size have shown material deterioration.
  - **Inner-size runs** (B19) use the diagnostic gate identities `…:inner-0:conv:0` and `…:1` at
    E, E⁺ (and E⁺⁺ if stage 2 runs), paired with their sources as described in A3.3. They change
    no production seed set, model count or policy, and decide together with the outer runs (one
    epoch count per arm). *Limitation:* training loss cannot show that thresholds chosen on inner
    models transfer to outer models.
  - Costs: stage 1 always costs 9.0 C-run equivalents (5.4 of them new), stage 2 costs 8.2 if
    triggered; one equivalent is 1.5-10 minutes.

#### A3.3 Seeds, fold order, boundary rules, B0's rate limits and the Stage 1 choices

- **Seeds** (B5), as built and tested in Stage 1:
  - The integer seed is n = int(SHA-256(UTF-8(string))[:8 hex], 16). Streams: stream k =
    `SeedSequence(n, spawn_key=(k,))`, with k = 0 initialisation, 1 data order, 2 augmentation,
    3 dropout. C seeds PyTorch from k = 0 and k = 3 with `generate_state(1, uint64)[0]` and builds
    NumPy `PCG64` generators for k = 1 and 2; B1 builds NumPy generators for k = 0 and 1.
  - Format `a5.2-seed-format-v1`. Enumerated identities: registered inner
    `gmv-a5.2:{arm}:{o}:{i}:{s}` (380), outer `gmv-a5.2:{arm}:{o}:outer:{s}` (190), final
    `gmv-a5.2:{arm}:final:final:{s}` (10); gate `gmv-a5.2-gate:{arm}:{set}:{purpose}:{g}` (22,
    including the unused `conv45` and `conv90` identities, which stay enumerated so the
    fingerprint does not change); sensitivity `gmv-a5.2-{sensT|sensV}:{arm}:{o}:{i}:{s}` (720) and
    `…:{o}:outer:{s}` (360). Total 1,682; enumeration fingerprint
    `0436bb715365843deac45c27a01be640d441850889ef9df12df8f6a6c81cb8f5`. Policy D's subsets apply to
    both arms whenever C is on policy D (A3.1).
  - o is the position in the private hashed outer order (below), never a group key; i is the
    fold number from `folds.assign`.
  - Before training, everything is enumerated for both policies and checked for distinct
    strings, integers and used derived values; any failure stops the run. A collision of
    registered seeds stops the run (no remapping). A gate or sensitivity string that collides
    gets `:c1`, `:c2`, … in sorted Unicode order until unique (dormant today).
  - **Declared relations** are not identities. Type `replay`: G-C3 and G-B1-3 repeat their source
    exactly (epoch-2 parameters byte-identical). Type `horizon` (the paired gate runs of A3.2):
    16 relations, 2 arms × 4 gate sources (`outer-0:conv:0`, `outer-0:conv:1`, `inner-0:conv:0`,
    `inner-0:conv:1`) × 2 longer lengths (E⁺, E⁺⁺; the E⁺⁺ ones used only if stage 2 runs). A claim
    needs a declared relation, its matching source, and that source already run at E; each
    relation is claimed once; gate identities only. Any other reuse of a string stops the run.
  - **Matched-start check** for `horizon` relations (a failure is S5; never the replay check):
    identical initial parameters; identical data order and augmentation draws for the shared
    epochs; identical dropout masks for the shared steps; for C, identical parameters after
    optimiser step 1 (the cosine schedule differs from step 2); for B1, identical parameters at
    the end of the shorter recipe's first half, before its learning rate drops (§6): epoch 30 for
    E against E⁺, epoch 45 for E⁺ against E⁺⁺.
  - Synthetic timing loads use fixed seeds outside every namespace and are not production
    identities. The random value of a public commitment (A3.6) is drawn from the operating system
    and is never derived from a seed.
  - Uniqueness establishes non-collision only, not statistical independence. Bit-identical
    results are expected only on the same machine, instruction set, environment lock and thread
    settings (4 intra-op and 1 inter-op threads; OMP, MKL and OpenBLAS at 4).
- **Hashed outer-fold order** (B6): the outer-fold position o is the rank of SHA-256 of
  `gmv-a5.2-inner-folds:outer:{key}` over the group keys. Seed strings carry only o. The
  sensitivity reruns use the same order restricted to P′ and renumbered.
- **B0's sample-rate limits** (B21), measured in Stage 1: at or below 23,700 Hz B0 cannot be
  constructed; from 23,720 to 32,000 Hz it constructs but cannot compute; from 32,001 Hz it is
  supported; 32 kHz itself is not. A rate counts as supported only after the four rate tests
  (L1-L4) pass at that exact rate. The low-rate group must fail L1, and any paired excerpt at an
  unsupported rate stops the run before folds.
- **Boundary and exclusion rules** (B22), as built, with ε = 1e-9 s:
  - R1 frame grid (§4); R2 centre times c = (first + i·hop + window/2)/rate − start, with
    first = round(max(0, start − 5 s)·rate) using Python's round-half-even, computed only in
    `frontend.segment`; R3 frames centred before 0 (pre-roll) excluded from training, evaluation
    and statistics, on the canonical segment and slice rule; R4 a frame counts if 0 ≤ c <
    duration, with no tolerance, and an excerpt ending more than one sample beyond its recording
    is refused;
  - R5 a frame is voice when c ≥ start − ε and c < end − ε for an interval of the confirmed union;
    R6 spans merge when start ≤ previous end + ε; R7 a boundary is interior if 0 < b < duration,
    with no tolerance; R8 a frame is excluded when |c − b| ≤ 0.3 + ε for an interior boundary b;
    R9 uncertain time uses R5's comparison on the uncertain union, for frames not already voice;
    R10 no collars at uncertain edges;
  - R11 only positive and negative excerpts are used, any other status stops the run; R12 a
    training frame is silent when 20·log₁₀(max(rms, 1e-6)) < −65 (strictly; unwindowed RMS over
    all channels at the native rate, float32); R13 silent frames stay in the evaluation mask with
    scores forced to 0 before hysteresis;
  - R14 training class membership: at least 100 training-eligible frames of the class; R15 metric
    class membership: at least 100 evaluation-eligible frames, silence kept; R16 fold typing: at
    least 2 s of a class, and a group under 2 s in both classes stops the run; R17 the low-rate
    group check;
  - R18 events: references are the union, then gaps merged when next start − previous end <
    0.25 − ε; predictions clipped to [0, duration]; a match needs |Δonset| ≤ 0.5 + ε and
    |Δoffset| ≤ max(0.5, 0.2·length) + ε, one-to-one, maximising matches and then minimising
    total onset difference; timing only at interior boundaries; R19 boundary-shift sensitivity
    moves only interior boundaries, clipped to [0, duration]; R20 merged events and frame collars
    are independent; R21 C's statistics frames as in A3.2 (collar and uncertain frames included);
    R22 mixed excerpts keep the primary excerpt's labels, mask and weights.
  - Fixed details: screening ε = 1e-12; the fold hash strings `seed:key`, `seed:fold:i`,
    `seed:r:key`; at most 2 samples of zero padding; operating points on a 2-decimal grid;
    hysteresis inactive at each segment's start; event-matching cost −10 + |Δonset|; the
    bootstrap seed from the first 16 hex digits of SHA-256(`gmv-a5.2-bootstrap`).
- **The nine Stage 1 choices** (B23): (1) HTK mel scale, triangular filters on the power
  spectrum, log(mel power + 1e-10); (2) C takes its frame count from the engine's rule, with at
  most 2 samples of zero padding; (3) interior boundaries are strictly inside the excerpt, with no
  tolerance; (4) no collars around uncertain intervals; (5) event collars apply to every reference
  event as written, timing errors only at interior boundaries; (6) a group with under 2 s of both
  classes is refused, voice-only single-class groups ordered before no-voice-only; (7) final
  models' seed fields are `final:final` (above); (8) undefined bootstrap draws are counted and
  reported, a ratio draw with a baseline of exactly 0 is undefined, and draws below 3 s/min stay
  in with their share reported (all private, A3.5); (9) PyTorch numbers convolution taps oldest
  first, the same family of functions as written.

#### A3.4 Data access and confinement

- **Each step's access is a ceiling** (B1): a step may open only the files listed for it, and
  only under its own authorisation and run declaration. Unknown steps are refused. Every input is
  validated against its registered schema; any missing or unexpected field or file stops the step
  with a fixed message. `source` never leaves its reader; raw `selection` is handled only inside
  the gateway at step 6.

| Step | May open (and pass on only the fields named) | Refused |
|---|---|---|
| 1, 4 | Generated data only | Every project file |
| 2 | The MUSDB corpus manifest (MUSDB records: track id, corpus split, checksum; other records: `dataset` only); the split registry (development and validation entries: partition and nodes; others: partition), with its fingerprint check; `aliases.json` (mandatory) and `titles.json` (assessed groups' titles only); the MUSDB credit-readings file and the review-decisions file (A3.6); the parent folders of these files, never listed | Development manifest and history; validation, final-test, `selection/`, `backups/`, `session.json`; other identity files; audio locations, owned-music manifest, corpus annotations, all audio; the rest of `local-evidence/` and the home folder; folder listings, child processes, network |
| 3 | The loader set: the development manifest through gateway fields, the development history (status, confirmed, uncertain), and the fingerprints, registry, validation manifest and final-test manifest for integrity only; the paired audio; step 2's private match list (group keys and match types) and the review-decisions file | Low-rate audio; validation history and audio |
| 5 | The loader set; fold files and fold index; paired audio of outer fold 0's training groups only; from the B0 file only `feature_version`, `mean` and `scale`; generated audio for synthetic loads | The held-out group's audio; low-rate audio; B0's scoring fields; tokens |
| 6 | The loader set; fold files; all paired audio; low-rate audio (final C models only); the B0 file; raw selection held in the gateway; tokens to the reporting module only after the N2 verification (A3.5) | Validation history and audio; final test; the match list |
| 7a (before sealing) | The validation manifest through a manifest-only reader (development fields minus `source` and `selection`); validation audio; frozen final models; the B0 file; t_final; the verified step 6 seal | Development audio; tokens; validation `selection` and `source` |
| 7b (after the step 7 seal verifies) | The validation history (status, confirmed, uncertain) | As 7a |

- **Parser contract** (B2), as built: `re.fullmatch` anchoring, exact string type, a fixed token
  table, and `missing` and `unrecognised` fallbacks. **Vocabulary** (C3, option a): the 6 A3
  answers and 8 negative categories derived from tracked code; fallbacks are reported as counts
  only, at RG4, through the disclosure check.
- **Output classes** (B3): computational and provenance outputs are kept apart, with the
  allowed-change lists for DU1, DU4 and DU5. This is built and tested for step 3's outputs; at
  RG1b the same classes and checks must be shown on a synthetic end-to-end run covering models,
  normalisation statistics, predictions, thresholds, metrics, screening, the seal, and run and
  gate records.
- **Access recording** (B4): every real-data step is accepted only through the spy test (checked
  per recorder, fail closed, `python -B`, folder listings counted as violations, strace reading
  input paths only).
  - **Live recording, steps 2, 3, 6 and 7:** `strace -f --kill-on-exit --seccomp-bpf -s 0` over
    the whole process tree, recording file, process and network calls, `getdents64`, `fchdir` and
    `io_uring_setup`; deny-by-default under personal-data roots, against an environment baseline
    from synthetic runs. Staged outputs are promoted only after a complete, fully classified trace
    with no violation (strace exit 0, the root exit line present, every traced process accounted
    for, no unparsable or truncated input path). Step 7 uses phase markers: the history only after
    the seal-verified marker. Any io_uring call fails the run.
  - **Step 5** is not live-recorded; instead a short traced rehearsal runs for each job type, and
    its timings are discarded. Traced runs never overlap timing runs, and traced durations are
    never used as timing data.
  - **Retention:** the raw trace is kept privately (folder mode 0700, file mode 0600, ignored by
    git, read only by the trace reader) until A5.2's final report is accepted, then deleted only
    with the owner's confirmation. Public: the trace's SHA-256, the strace version, the call set,
    counts by class and kind, the unresolved count and the verdict.
  - **File-level and field-level access are different things.** Tracing shows which complete
    files a process opens, not which fields it reads. Each step's record states which complete
    files it opens, what it retains, and what must never reach logs, terminal output or the
    conversation.
- **Confinement (step 2, and the pattern for later steps)** *[tested before use]:* bubblewrap,
  **as the proposed configuration for synthetic Step 2a testing, not a proven security
  boundary.**
  - Read-only inside the sandbox: the inputs at `/in/corpus/manifest.json`,
    `/in/presence/manifests/split-registry.json`, `/in/presence/fingerprints/manifests.json`,
    `/in/presence/identity/aliases.json` and `/in/presence/identity/titles.json` (and the step's
    readings and review-decisions files); GMV's `src` at `/gmv/src`; seven packages at
    `/gmv/site-packages` (numpy, numpy.libs, pydantic, pydantic_core, annotated_types,
    typing_inspection, typing_extensions); `/usr/bin/python3.14`; `/usr/lib` with `/lib` and
    `/lib64` links; `/etc/ld.so.cache`. **The host paths are placeholders here;** the exact paths
    of protected inputs are verified during the separately authorised preparation and recorded
    only in the private run record.
  - `/out` is the only writable path (created empty by the launcher, which refuses if it exists).
    Empty read-only root; a new `/proc`; bubblewrap's minimal read-only `/dev`; no `/sys`,
    `/tmp`, `/home`, `/root`, `/run`, `/var`, `/usr/share`, the rest of `/etc` or the rest of
    `/usr/bin`. A loopback-only network with no route out and no name lookup. Separate user,
    process, IPC, network, hostname and cgroup namespaces; no nested namespaces; no capabilities;
    the sandbox dies with its parent and is detached from the terminal; the environment is
    cleared except `OPENBLAS_NUM_THREADS=1`; command `/usr/bin/python3.14 -I -S -B -X utf8 -c
    "<fixed start-up>"`; the launcher confirms outside the sandbox that the tree is clean at the
    authorised commit.
  - Exactly three inherited handles (input from `/dev/null`, output and errors to a private
    pipe); the step lists its handles first and stops on any extra; crash dumps disabled and the
    process marked non-dumpable; strace runs outside the sandbox; the trace reader keeps
    bubblewrap's set-up separate from the step's actions; `/out` must hold exactly the match list,
    the owner's review file and the public report.

#### A3.5 Reporting and disclosure control

- **AUC** (B20) *[tested before use]:* the score is each arm's per-frame voice probability after
  the silence gate and before hysteresis (C's computed in float64 from its logit). Frames: the
  evaluation mask, quiet frames included (scored 0). Ranked by value, ties counting half (ROC) or
  one step (average precision, AP = Σ (R_k − R_{k−1})·P_k over distinct scores, no
  interpolation). A group is eligible with at least 2 s (100 frames) of each class; its AUC comes
  from the one model that predicted it; the macro is taken over eligible groups per seed index,
  then averaged over O. B0 and B0ₜ share one value. Public: the macro values, the eligible-group
  count and the mean voiced share over eligible groups. Private: every group's value and
  prevalence. No pooled AUC.
- **The low-rate group's C result** (B24): the mean over the frozen final C models at t_final,
  with §11's minimums, computed after N2 within step 6's own authorisation; private. Public text
  only: "C was scored on the low-rate group; its results are private."
- **The no-voice split** (C6, option d): reported "by selection route", A3 excerpts against
  negative candidates, with this caveat: route describes how an excerpt was chosen, not whether its
  recording is instrumental; `other_song_intro` candidates come from recordings flagged as vocal,
  and A5.1 found a voice in 7 of 80 metadata-chosen "instrumental" candidates.
- **Breakdown tables** (including the route table): the cell metric is the macro false activity
  over the cell's eligible groups (at least 2 s of evaluated no-voice time in the cell), per seed
  index then averaged; a cell is published only with at least 3 eligible groups, eligible-group
  counts shown as bands 3–4, 5–9, 10+; groups with a defined overall value but eligible in no
  published cell must number 0 or at least 3, otherwise more cells are suppressed (fewest groups
  first); fallback buckets are shown as counts only and treated as suppressed; duration-weighted
  and per-seed cell values are private.
- **Not public:** quartiles of per-group differences; bootstrap intervals, the ratio interval and
  the share of draws below 3 s/min (all computed and kept privately, with their caveat).
- **Fixed precision:** 3 decimals on 0-to-1 scales, 2 for s/min, 1 for seconds.
- **The disclosure check (N1)** *[tested before use]:* every public figure, in every table and
  release, passes this check before publication.
  - **N1.1 Reader.** The reader knows the cumulative public record (A5.1's reports and figures
    already in this protocol, such as "52 s from 4 groups", so differences between tables and
    releases count) and the registered structure. ~~When results (durations, metrics, effect sizes)
    are checked, the reader is also assumed, to be safe, to know each group's excerpt count,
    per-class eligibility, direction, leave-one-group-out outcomes, and G_T and G_V membership;
    these are cautious assumptions, not public facts.~~ The reader knows nothing else about
    individual groups (Amendment 3B). Which groups could be in G_T or G_V (the 19 paired groups) is
    structure; which groups are is private. ~~Directions and membership are also protected in their
    own right: when they are checked, the reader is assumed to know no group's fact of that kind,
    but may know the others.~~ Each group's direction, leave-one-group-out outcome, per-class
    eligibility and G_T and G_V membership are protected in their own right (N1.3; Amendment 3B).
  - **N1.2 Coverage.** Checked: group averages and effect sizes for every arm~~, seed index~~ and
    comparison; the seed-spread verdict (Amendment 3B); effects at t ± 0.05; the §10 sensitivity results (frame collars, boundary shifts,
    event collars); validation averages; duration-weighted main results; the false-activity ratio;
    breakdown tables and route rows; excerpt counts; onset and offset error (group averages of each
    group's mean absolute error only); direction counts; the leave-one-group-out count and pass or
    fail; §13 verdicts (as inequalities); step 2 and 3 counts, the sizes of G_T and G_V, feasibility
    statements and seconds; step 3 voiced and no-voice durations. Outside protection: the operating
    point chosen per fold, and inner-seed agreement.
  - **N1.3 Protected quantities.** Each group's own value behind each published figure; the sum
    and the difference of any two groups' values; missed voice through recall (a share of the
    group's voice), not in seconds; a group's direction (both must remain possible); a group's
    leave-one-group-out outcome and its per-class eligibility (both must remain possible;
    Amendment 3B); counts (at
    least two whole-number values must remain possible); each group's MUSDB match status for each
    split, and its G_T and G_V membership (at least two values, for single groups and pairs).
    Item D's exemption: A3.6.
  - **N1.4 Widths (operational):** 0.10 on 0-to-1 scales; 1 s/min for false-activity rates; 10 s
    for step 3 voiced and no-voice durations. *Fix 1 (prior-limited rule):* a step 3 duration
    whose possible range before publication is already narrower than 10 s (an ineligible class,
    or a one-excerpt group) must keep exactly that range. *Extension:* a group's match status for
    one MUSDB split counts as already known only when that split's item D figure fixes it
    completely (0 or all assessed groups). Any other quantity narrower than its width before
    publication gets no exemption: the run stops for the owner's review.
  - **N1.5 Rounding and limits.** Each figure enters as the interval its rounding allows. Natural
    limits: no negative time; shares in [0, 1]; a group's false seconds at most its no-voice
    seconds; voiced plus no-voice at most 10 s per excerpt.
  - **N1.6 Fixed zones** (public information only). An average near a natural limit is shown as
    "below z" or "above limit − z", with z = width ÷ the smallest group count its published band
    allows (for example, 1 s/min over 3–4 groups gives "below 0.34 s/min"); duration-weighted values
    and sums use z = the full width; step 3 totals below 10 s are shown as "below 10 s"; zone edges
    are rounded up to the reporting unit; differences and the ratio are coded consistently with the
    figures they come from; direction counts and other group counts are shown as "0–2" and
    "(n−2)–n" at the ends; ~~a sensitivity set's totals are published only when the removed set has
    at least 3 groups (rule 5), which does not replace the check~~ (rule 5 no longer applies:
    sensitivity sets' totals are private, Amendment 3A); false-activity averages are shown as
    numbers only up to 35 s/min (3–4 groups), 45 s/min (5–9) or 50 s/min (10 or more) and as
    "above [cap] s/min" beyond, and a difference or ratio whose source is shown as a code is shown
    as "coded with its sources" (Amendment 3B).
  - **N1.7 Suppression, withholding and stopping.** (1) The reader observes the numbers, which
    figures appear as numbers or as codes and which code, which are suppressed or withheld and
    where, whether a report was released, stopped or released after a stop, and every earlier
    release. (2) Every protected quantity keeps its width (or its prior range under Fix 1) across
    all values that would produce the same observable outcome under the registered rules.
    (3) Every decision to release, suppress, withhold, widen to a larger unit or stop is computed
    only from the registered rules, the public record and the figures already released in the
    same report, never from private values, including every input that can change a verdict (the
    repair point, the solver's starting point and settings); the checker is deterministic.
    (4) Release order: main figures; direction counts; other figures from most groups to fewest,
    then table order; a sensitivity set's voiced and no-voice totals count as one figure (Fix 2).
    (5) A figure is released only if every value it could show (each rounded number and each
    code), given what is already public, keeps every protected quantity at its width or prior
    range; its actual value decides only what it shows. An uncertain verdict counts as unsafe. A
    shortcut that checks only the extreme possible numbers and each code may be used only where
    its soundness has been established for the registered constraint structure (never for
    whole-number counts); otherwise every possible value is checked or a conservative fallback is
    used. (6) Structural suppression is part of (5). (7) A figure that cannot be decided from
    public information has its registered fixed unit withheld whole (units fixed in advance from
    the table layout); with no unit, the run stops. (8) ~~Reader knowledge that is not public
    (directions, leave-one-group-out outcomes, eligibility, membership) must not be taken from
    its actual values: decisions hold for every possibility consistent with the public record,
    and if that cannot be certified within a registered size limit, (7) applies.~~ Decisions
    never use private values, and no group's direction, leave-one-group-out outcome, eligibility
    or membership is assumed known to the reader (Amendment 3B). (9) The run stops
    for the owner's review when the already-public record fails on its own, a prior range is
    narrower than its width outside Fix 1 and its extension, or (7) finds no unit; these stops are
    decided from public information and their reasons are given in public terms; a later release
    uses only registered units and is checked together with the stop; a self-check failure is a
    software fault, fixed and rerun from the start. Withheld figures all look the same; registered
    pass conditions stay public and enter the check.
  - **N1.8 Exact verification.** (1) Every verdict is proven in exact rational arithmetic; a
    floating-point solver may only propose. (2) A pass needs exactly feasible values spanning the
    width (under Fix 1, reaching both ends of the prior range). (3) A fail needs an exact bound
    from non-negative multipliers and the variables' limits. (4) A range exactly equal to its width
    passes; the checker finds exact optimal corner points by solving the binding constraints
    exactly and verifies feasibility and the matching bound exactly. (5) Repairs move toward an
    exact rational point in the relative interior of the feasible region (every forced equality
    met exactly, positive slack on every other constraint). (6) Forced equalities (coinciding
    limits, a figure that pins its groups, inequalities that together leave no room) are identified
    exactly, each with an exact proof, and proposals are projected exactly onto them before repair;
    a quantity with a single possible value fails unless Fix 1 applies and it already had only that
    value. (7) Self-check: the true values satisfy every constraint exactly and the region is not
    empty, otherwise the run stops. (8) Uncertain counts as failure. (9) A count passes only with
    two exactly valid whole-number values.
  - **N1.9 Tests** before the checker is used: A3.10.
  - **N1.10 Limits.** The widths hold only under the stated reader and are not calibrated
    guarantees; the evidence for them is synthetic; collar limits, validation tables, effects at
    t ± 0.05, AUC tables and route rows were not modelled in the synthetic tests; the
    relative-interior handling is untested. Under Amendment 3B the widths hold only against a
    reader who knows the public record, and strict inequalities in verdicts enter as their closed
    forms.
- **The prediction seal (N2)** *[tested before use]:* a write-once seal (canonical JSON with
  format and version, run and step identity, code commit, integrity-record and fold-index hashes,
  seed format and enumeration hash, policy, adopted epochs, thresholds file hash and model file
  hashes), created exclusively and then made read-only, its digest recorded in the run record and
  the RG4 summary.
  - The verifier first builds the expected inventory only from the registered design (arms, roles,
    folds, units, the seed plan's models), the published decisions (Amendment 5's fold index,
    checked against its hash, and its sensitivity sets; Amendment 6's policy and epochs) and the
    run declaration (checked against those decisions). The seal is not an input; its file is opened
    only after the expectation's digest has been written to the run record.
  - The seal's inventory must equal the expectation exactly. Refused: a model identity outside the
    list; a seed outside the declared policy's set; an undeclared rerun or a declared rerun outside
    Amendment 5's sets; an evaluation unit not in the fold index; a relabelled entry; duplicates;
    missing entries; extra files on disk. RG4 compares the declaration with the owner's
    authorisation text. Tokens and low-rate metrics need the in-memory verified seal.

#### A3.6 The MUSDB18-HQ overlap check (steps 2 and 3)

- **Where the split is decided** (L1, option C): step 2 matches every assessed group and files
  development groups under one private "development" partition. Step 3, which already reads the
  development manifest, classifies groups as paired or low-rate with the rule that defines the
  training set (the B0 rate check, A3.3), checks that the registry's development groups equal the
  manifest's, and derives G_T, G_V and the paired counts from the private match list and the
  owner's review decisions. The low-rate group's cells stay private. *[tested before use: deciding
  the split later gives exactly the same matches as deciding it first.]*
- **Inputs:** the corpus manifest named exactly in step 2's authorisation (the git-ignored list
  B0's training command names; the tracked, documented-empty `corpus/manifest.json` must fail
  closed), the registry, `aliases.json`, `titles.json`, the credit-readings file and the
  review-decisions file. The link between the corpus manifest and B0's training data is an owner
  attestation unless a historical hash or other provenance record establishes it. The hashes of
  `aliases.json` and `titles.json` recorded at step 2 are a new baseline, not proof that they are
  unchanged since A5.1.
- **Matching** (readiness revision 4, §7), for each corpus record r with `dataset ==
  musdb18hq`: split(r) ∈ {train, validation, test} (GMV's corpus split, matching B0's training,
  which used train and validation only); artist and title from `dataset_track_id` split at the
  first " - "; exact keys E(r) = the credited names (aliases applied) plus exactly one step of link
  expansion; permissive keys (the whole artist, its pieces at lower-case " and ", " with " and
  brackets, and compact forms); an **exact match** when E(r) meets a group's names, a **checksum
  match** when the file's SHA-256 is one of the group's recordings; **title confirmation** and
  **permissive-only** matches are reported only, never excluding a group. G_T and G_V hold the
  paired groups with an exact or checksum match to a train or validation record. Any exact or
  checksum match from a test record to a development or validation group stops the check before
  any result exists. Reserved groups are not assessed.
- **Fail-closed safeguards** *[tested before use]:*
  - `aliases.json` is mandatory: a missing file stops step 2; a present file with an empty alias
    map is accepted. A strict alias reader stops on conflicting targets, chained aliases, entries
    that normalise to empty or generic names, malformed links, and a link whose end is an alias
    spelling (L5; the owner then corrects the file; no canonicalisation rule).
  - The registry is read through its fingerprint-checked reader; the corpus manifest is validated
    with the strict schema (unexpected fields forbidden).
  - **Unparsed records** (MUSDB records only; other datasets never trigger it): no " - "
    separator; an empty or generic artist; or an **ambiguous credit** without a registered reading.
    Any unparsed MUSDB record stops step 2 before G_T and G_V are fixed.
  - **Ambiguous credits** (L3): a MUSDB artist field containing, after standard normalisation,
    "and" or "with" as a separate word in any capitalisation, a bracket, or letters outside the
    Latin alphabet. (The exact tier never splits an artist field at "and" or "with", so "X And Y"
    could otherwise never match a group "Y".) Step 2 begins with a listing phase that reads only the
    MUSDB records and lists the ambiguous credits; it runs every time. Each gets a reading in the
    private, fingerprinted, always-present readings file: by default the whole name and each piece
    all count as credits; "single name" only with a stated public reason, never by checking the
    owner's library. Matching refuses an ambiguous credit with no reading, a reading for a credit
    not on the list, and duplicates. Matches from a piece are flagged in the review file.
  - The corpus's `unassigned` split is refused (L4).
  - **Documented limitations:** names transliterated or written in another script are not matched
    on the library side; uncredited performers are not matched; covers by other artists are not
    matched; checksums catch only byte-identical files; title confirmation and permissive-only
    matches never exclude; "(With …)" title brackets count as credits and can over-match (the
    review shows where each credit came from, L5); the report-only tier splits "and" and "with" in
    lower case only (L3; this limitation stands only if the L3 tests pass).
- **The owner's match review** (C4) *[tested before use]:* mandatory, after step 2's matching and
  before step 3 fixes G_T and G_V, in a private file the owner opens; Claude sees counts only.
  - Shown: every name-based pair (group name, MUSDB artist and title, and where the credit came
    from); file matches for information only.
  - The owner may mark a name-based pair "not the same artist"; no additions (a missed match is
    fixed by correcting `aliases.json` and re-running step 2); file matches cannot be rejected.
  - Decisions go into a private, fingerprinted review-decisions file that is always present (an
    empty list is valid). Step 2 applies its test-split rejections before the test-split check;
    step 3 applies the rest.
  - **A test-split stop** writes a review file for the test pairs only. If the owner rejects every
    pair, step 2 may re-run under a new authorisation with those rejections as a registered input.
    If the owner confirms any pair, that is a **research-safety stop**: A5.2 halts for a separate
    owner decision. It is decided from private data and the owner's judgment, before any figure
    exists, and is not a disclosure stop (N1.7).
  - **Public wording:** ~~every step 2 stop, whatever its cause: "Step 2 stopped for owner review. No
    figures from this run are published." Every re-run: "Step 2 re-run after the owner corrected a
    private input."~~ only step 2's final outcome is public (replaced by Amendment 3A). Never published: counts from before the review, the number of pairs shown or
    rejected, or any wording saying matches were found, reviewed, rejected or confirmed. These
    sentences, when they appear, whether a report is published, the public commitment and the
    item D figures enter the membership check (N1).
- **The public count report** (L6) *[tested before use]*, published only after RG2, together with
  step 3's figures:

| Figure | Treatment |
|---|---|
| **Item D:** for each MUSDB split (train, validation, test), the number of assessed groups (all of them together) with a name or file match after the review, zero included | Published as it is; exempt from the count rule; still known to the reader when other figures are checked |
| MUSDB record and unparsed counts per split; the number of credits read through a registered reading | Through N1 (they describe the MUSDB list only) |
| Assessed-group and partition totals | Through N1 |
| Validation-partition cells, breakdowns by match type, paired cells, the sizes of G_T and G_V and their overlap | Through N1, groups only, with the end codes |
| Matched-record counts | Through N1 |
| Low-rate cells | Always private ("withheld: one group") |
| A separate test-split line; review counts; counts from before the review | Never published |

  Release order: the item D figures; the G_T and G_V sizes and overlap; validation cells; paired
  cells; match-type breakdowns; record counts. Item D exempts nothing else: not denominators,
  unparsed counts, breakdowns or separately derived test-split statements. Validation-partition
  counts are published only if the combined publication passes the check.
- **Public fingerprints** *[tested before use]:* no public hash, or other repeatable calculation,
  of any private input or of anything derived from review decisions or membership. Private records
  keep each input's real SHA-256. The public report carries a hiding commitment instead: SHA-256
  over a fixed label, a fresh 32-byte secret random value from the operating system (new for every
  run, kept only in the private run record, never in logs, terminal output or the conversation)
  and the canonical list of private input hashes. The same rule covers every other public hash of
  content that depends on review decisions or membership, including the sensitivity reruns' fold
  files, their amendment entry and their sealed inventories.

#### A3.7 Sensitivity reruns

- **Definitions** (readiness revision 3, §4.2): G_T and G_V as in A3.6; a group in both is removed
  from both reruns; permissive-only matches never exclude. Feasibility: F0 the removed set is not
  empty; F1 every outer fold of P′ passes every registered check with 5 non-empty inner folds (at
  least 6 remaining groups); F2 every training set has a member group in each class with at least
  100 training frames; F3 J(t) is defined in every outer fold; F4 the macro metrics are defined;
  F5 every leave-one-group-out recomputation stays defined; F6 the projected compute fits the
  approved limit (A3.8). A rerun is a complete nested rerun on P′ with nothing reused: every inner
  and outer B1 and C model retrained on P′, normalisation per model, augmentation partners from
  the model's own training set, the `sensT` and `sensV` seed namespaces, the main run's policy and
  epochs, B0ₜ's threshold chosen again on P′, no final models, predictions sealed. Folds for each
  feasible P′ use the registered algorithm, the hashed outer order restricted to P′, the same
  counts and the independent check, and are recorded (as commitments, A3.6) before training.
- **Reporting:** descriptive, beside the main results, labelled "sensitivity: does not change the
  screening decision" (or "not evaluable"); any difference combines the removed exposure and the
  smaller sample. "Infeasible under the registered procedure" or "not run within the approved
  compute limit" when a check fails. No substitute analysis is ever run.
- **The rules** (C5, option a):
  1. **Trigger:** ~~a rerun may go ahead only if the published size code of its removed set shows at
     least three groups removed: an exact count of 3 or more, or the top end code "17–19". If the
     size is shown as "0–2" or withheld, that rerun is not run. The trigger never uses unpublished
     membership.~~ Replaced by Amendment 3A, rule 1.
  2. **Feasibility without membership:** ~~F2-F5 and F6 must pass for every possible removed set of
     the published size among the 19 paired groups, not only the actual one; F6 prices the most
     expensive case (removing the groups with the fewest excerpts). F1 fails from the size alone
     when 14 or more groups are removed, so "17–19" always ends as infeasible.~~ Replaced by
     Amendment 3A, rule 2.
  3. **Order:** G_T first; G_V only if F6 shows both fit.
  4. **Authorisation and checking:** the final go stays with the Stage 3 authorisation, which must
     name the reruns. ~~Whether a rerun ran, any "not run because…" statement, and every rerun
     figure that depends on which groups were removed (excerpt totals, eligible-group counts,
     compute and timing records) go through both the results check and the membership check (N1).~~
     The rest of this rule is replaced by Amendment 3A, rule 4: rerun results are private.
  - Cost: each rerun is about 78% of the main run's training.

#### A3.8 Timing, resources and running alone

- **Process-time** (B7): as in A3.1, the job running with no other GMV experiment or benchmark;
  host activity recorded; the machine need not be idle. CPU time is for information only.
- **Components** (B8): per size class (inner, outer, final), setup S, the first epoch d1 counted
  separately, cost per unit k from epochs 2 to E, and prediction cost p. A job is projected as
  S + d1 + (E − 1)·k·U + p·P; H_proc sums these over the policy's jobs plus the one-time
  preprocessing.
- **Measurements** (B9; C2 option R). T1: single-process components from B18's real-data gate
  runs (outer and inner size, both seeds, which run alone and serve as T1's measurements; the
  gate-only loss evaluation is timed separately and excluded), plus synthetic runs at the
  smallest, median and largest inner and outer training-set sizes and at the final size, with a
  synthetic-against-real cross-check at one size. T2: four concurrent 30-epoch outer-size runs
  (sustained slowdown and drift). T3: co-load probes in 13 configurations (all ten 3-co-runner
  mixes, one co-runner training, two co-runners training, and setup with prediction), 7 probe kinds,
  2 repetitions: 182 probe runs, in a hot machine state (immediately after T2, or after at least
  10 minutes of four-way load), in random order. T4: end-to-end checks at the adopted epoch count
  (T2's runs as complete jobs, and a synthetic mixed 16-job schedule run twice). Synthetic loads use
  generated data only, enforced by a runner that refuses project data.
- **Step 5 order:** the traced rehearsal (timings discarded) → T1 with B18 stage 1 → B18 stage 2
  if triggered → adopt the epoch count, recompute the 16 h gate, apply the policy-flip safeguard →
  T2 → T3 → T4 → the planning estimate, the 48 h check and F6 → RG3.
- **Planning estimate** (B10): each job's estimate scales its components by the largest measured
  slowdown for its size class; the phase estimate is (Σ D̂ + 3·max D̂)/4. It is conditional on
  stated assumptions and never called conservative or guaranteed; the scheduling bound is claimed
  only for fixed durations. In Stage 3, if any job exceeds its estimate or elapsed time exceeds the
  plan, the run pauses at the next model boundary for the owner; the policy never changes once
  Stage 3 has started.
- **Stops and values** (B11, V1-V6):

| Rule | Value |
|---|---|
| A slowdown above 4, or above its co-runner count plus one | Stop |
| Coefficient of variation of epoch times (epochs 2…E) | Stop above 0.15 |
| Throughput still falling (epochs 21–30 against 11–20) | Stop above 5% |
| Inner against outer cost per unit | Above 20%: the larger value is used in the planning estimate and F6 only, and reported; the 16 h gate keeps the measured per-class values |
| Synthetic against real timing at the same size | Stop above 20% |
| Any job above its estimate, or a schedule above its plan, in T4 | Stop |
| Wall-clock limit for the planning estimate (preprocessing, C, B1 and rerun phases) | 48 h (172,800 s). Main phases alone above it: stop for the owner. Only the reruns not fitting: they do not run (G_T first) |
| Caps, exec to exit | C 0.1 h per epoch (3 h at 30, 4.5 h at 45, 6.8 h at 68); B1 1/30 h per epoch (2 h at 60, 3 h at 90, 4.5 h at 135); 4/3 of the solo cap for jobs running beside others by design (4 h for T2's processes); 1 h for a T3 probe or a rehearsal run; inner-size runs use the outer caps. A cap hit is a stop |
| Memory | Stop if the experiment's whole process tree (launcher, jobs, every process they start, monitoring tools included), measured once a second as the sum of each process's proportional set size plus its proportional swap share (shared memory counted once), exceeds 8 GiB; the peak is recorded |
| Swap activity anywhere on the machine during a timing run | The run is marked disturbed (never counted towards memory) |

- **Running alone** (C1 procedure) *[tested before use]:* every GMV experiment and benchmark
  runner takes the **same exclusive** operating-system lock (never a shared lock) on one fixed
  file outside every working copy, before any work, held until all its work has finished;
  non-blocking (a second runner refuses at once and never queues). The launcher takes it outside
  any sandbox for the whole experiment; experiments that run several jobs by design are one
  holder; jobs stop if the launcher stops; the lock handle is never passed to a job; the operating
  system releases it on exit, even after a crash. A host-activity record (about once a second)
  keeps only aggregate counters: total processor use, the job's own use, other programs' use,
  available memory, swap activity and processor frequency; **never process names, command lines,
  window titles or file names.** A run is disturbed if other programs average more than one logical
  processor, or use more than four for over a tenth of the run, or any swap activity occurs; a
  disturbed run is repeated once, and disturbed twice means owner review. These thresholds are
  checked first by a generated-data baseline run. During step 5 the owner avoids heavy programs and
  GMV's own application; light use is recorded only as aggregate activity. The processor's power
  settings are never changed.
- **Timing privacy:** training-set sizes reveal groups' excerpt counts. Public: per-size-class
  costs, the preprocessing time, total process-time and the policy chosen, slowdown factors, the
  planning estimate and measured schedule lengths, stop counts, disturbance thresholds and counts.
  Private: each job's duration with its training-set size, the synthetic sweep's sizes, anything per
  fold, and the activity samples.

#### A3.9 Authorisation steps and review gates

**No review authorises the next step, and no authorisation implies another or permits private
data unless it says so.** Every real-data step needs its own authorisation and run declaration.
Access lists are ceilings, used only under that step's own authorisation; B18's automatic
adoption of a longer schedule happens only inside step 5; the match review happens only inside
steps 2 and 3. A separate authorisation is needed for any step 2 re-run, for anything after a
research-safety stop, and for including the sensitivity reruns in step 6.

| Gate or step | Permits | Closed by |
|---|---|---|
| **D1** | Drafting Amendment 3 in the conversation | The complete decision register (Groups 1-6) |
| **D2** | Approving the draft | The full draft with superseded text struck through, a map from each decision to its wording, the consistency checks |
| **P1** | Editing the protocol file, committing and pushing to PR #12's branch, which stays Draft; it starts no step | A protocol-only change, the commit on the remote branch, PR #12 still Draft, the header reading "amended three times" |
| **Step 2a**, then **RG1c** | Synthetic implementation: the step 2 and 3 runners, the sandbox, the strict readers, the listing phase, the review files, the commitments, the disclosure checker for step 2 and 3 outputs, and their tests | The commits, the synthetic test logs, the sandbox checks, the reviewer's verdict |
| **Step 4**, then **RG1b** | Synthetic implementation: Stage 2's trainers, normalisation, augmentation, dropout, single-model scoring, gate and projection code, the lock and recorders, the paired-run ledger and matched-start check, policy D in screening, the AUC code, the N2 verifier, the rest of the disclosure checker, and their tests | The commits, the synthetic test logs, the reviewer's verdict |
| **Step 2** (after RG1c) | The real MUSDB check: the listing phase, readings, matching, the private match list, the owner's review file | Listing and readings done; the trace verdict passes; private outputs staged; the owner's review file written; counts only in the conversation |
| **Step 3** | The split, G_T and G_V with the review decisions, the main and sensitivity folds, F0-F5, the Amendment 4 and 5 drafts | The equivalence check passes; folds verified; every public output passes the disclosure check |
| **RG2** | Review of Amendments 4 and 5 (aggregates), the public commitments and digests, and the access summaries; then a separate publish authorisation for each amendment | — |
| **Step 5** (after RG1b and RG2) | The gates in A3.8's order; private run records; the Amendment 6 draft (epochs, policy, F6) | — |
| **RG3** | Review of Amendment 6 and the gate aggregates; then a publish authorisation | — |
| **Step 6** | Stage 3: the main run, plus sensitivity reruns only if named in the authorisation and F6 passed; sealed development predictions; the report draft | — |
| **RG4** | Review, before validation scoring, of the Stage 3 report draft, the frozen final models and their hashes, t_final, the Stage 4 code, and the access summary showing the validation history was not read | — |
| **Step 7** | Stage 4: seal the validation predictions, then score them | — |

#### A3.10 Tests required before use (all on generated data)

Already built and passing in Stage 1 and Step 1b, and kept: the seed tests SC1-SC7 and RP1-RP5,
the rate tests L1-L4, the boundary tolerance tests, the spy-test controls, the parser tests, the
DU tests and the matcher tests. New:

- **Disclosure checker (N1):** the regression cases from the synthetic width tests (one group left
  out of the G_T row rebuilt to 0.275 s/min and suppressed; two groups' sum rebuilt to 0.34 s/min;
  ties at exactly 1.00 s/min passing; the earlier zone of width ÷ count failing at 0.0977; missed
  voice protected in seconds narrowed to 5.27 s and stopping; nested G_V sets suppressed; little
  voice in G_T resolved by Fix 2; prior-limited groups under Fix 1); true values on a constraint;
  coinciding limits; a figure pinning its groups; an equality forced by two inequalities; a
  single-valued quantity; an empty region; a deliberately unprovable case (withheld or stopped,
  never passed); whole-number counts; and:
  - T1 the earlier true-value rule shown to leak (positive control); T2 decisions identical across
    alternative private values with the same public information; T3 an access spy showing the
    decision code receives no private values; T4 exhaustive small worlds, grouped by full
    observable outcome; T5 stops depend only on public information and name only public facts,
    and a release after a stop passes T4; T6 a forced uncertifiable case withholds its unit
    identically, and without a unit the run stops; **T7** the extreme-value shortcut compared with
    exhaustive checking on small worlds (passing these does not prove the shortcut sound; its
    soundness must be established for the registered constraint structure before use, otherwise
    exhaustive checking or a conservative fallback); T8 the cost of the rules on the synthetic
    worlds (what is published); T9 determinism across runs and thread counts; T10 item D's scope.
- **Step 2 and 3:** the sandbox checks (a full walk finds exactly the mounted paths; writes fail
  except `/out`; loopback, outside, host-socket and abstract-socket connections fail; `/proc` shows
  only the sandbox and `/bin/sh` fails; a planted extra handle stops the run; the trace handle is
  not inherited; a forced crash stores no dump; killing strace or the launcher stops the step;
  io_uring is flagged; outputs byte-identical to a reference run outside the sandbox); a synthetic
  world mirroring the real layout, including the empty tracked corpus manifest failing closed;
  negative controls for every refused file class; the strict readers (missing alias file;
  conflicting, chained, empty or generic aliases; malformed links; a link end that is an alias
  spelling; malformed title keys; unexpected registry fields or node types; a registry fingerprint
  mismatch; an extra corpus field); the unparsed-record stop; deferred classification identical to
  classifying first; low-rate cells never public and not derivable by subtraction; the match-review
  file private and never printed; determinism.
- **L3:** "and" and "with" in lower, capitalised, all-capital and mixed case in artist fields;
  "(with …)" and "[with …]" in titles; unusual spacing and full-width letters; no false alarms on
  words merely containing "and" or "with"; each ambiguous credit without a reading counted as
  unparsed and stopping step 2 with a counts-only message; each reading giving exactly its
  credits; a piece-only match found under the default reading and flagged; no ambiguous credit
  ending with neither a match nor a refusal; supported separators identical in every
  capitalisation; missing, stale or duplicate readings stopping step 2.
- **C4-T:** small generated worlds (5-7 assessed groups, every case listed) that differ only in
  which groups match: no candidates; test-split candidates all rejected (re-run, report with a
  test-split count of 0); one confirmed (research-safety stop); stops and re-runs for other
  causes; train and validation candidates and rejections. Worlds that look identical are grouped;
  within each group, match status per split before and after the review, G_T membership and G_V
  membership must keep at least two possible values for single groups and for the sum and
  difference of every pair, except where an item D figure of 0 or all fixes a status completely
  (before-review statuses only by "all"). A leaky variant (a stop message naming its cause, a
  re-run sentence that differs by cause, a decisions file present only after rejections) must be
  caught.
- **Commitments:** an attacker who knows every input except the review-decisions file and tries
  every possible decisions file never reproduces the public value, and the rest of the public
  output is identical whatever the decisions; the same attacker recovers the decisions from a
  plain digest (positive control); identical inputs give different public values in two runs; the
  private record verifies the value and any change fails; the secret and the private hashes never
  appear in public output, logs, terminal output or errors; trying every removed set never
  reproduces the sensitivity fold-file values.
- **Report and review:** item D figures exact and after the review; breakdowns groups only; end
  codes applied; the Fix 1 extension applying at 0 and all only (negative tests at 1 and all minus
  one); the review file round trip; rejections applied before G_T and G_V; rejecting a file match
  refused; additions refused; the test-split path; review counts never public.
- **C5:** the trigger read from the published code only; feasibility over every removed set; G_T
  first when F6 allows one rerun; run or not-run and reasons through both checks.
- **Timing and resources:** two runners started together, exactly one runs; a shared-lock holder
  does not let an exclusive one in; killing the holder releases the lock; jobs die with their
  launcher; the same lock file from every working copy; no job inherits the lock handle; a memory
  block shared by four processes counted once; child and copy-on-write processes found and not
  double-counted; swapped-out memory still counted; the memory stop firing (lowered test limit);
  swap activity flagged separately; the synthetic runner refusing project data.
- **Seeds:** the `horizon` ledger (declared, matching source, source already run, claimed once,
  gate identities only, any other reuse stopping); the matched-start check passing for genuine
  pairs and failing for a perturbed initialisation, data order or B1 first half.
- **N2:** the expectation identical with or without a seal file; an access spy showing the
  expectation code never opens the seal; one refusal test per refused case; policy C against D; a
  declaration disagreeing with Amendment 6; a weights-hash mismatch.
- **B3 at RG1b:** the synthetic end-to-end run of A3.4.
- **Amendment 3B's rules:** the tests listed in Amendment 3B.

#### A3.11 Where the wording changed

Originals struck through and kept visible: the amendment note at the top; §2, the low-rate
group; §7; §8 steps 4-6 and final models; §11; §12; §12.1; §13; §14; §15; §16; §17; §18; Amendment 1,
A1.1 row 3.

#### A3.12 Unchanged

The question; the arms; the data partitions and their counts; the decoding, framing and
pre-roll; model C's architecture; the training objective and batching; the inner-fold policy; the
label-boundary rules of §10 and the metric definitions of §11, except where reporting is
restricted above; the no-p-values policy and the allowed and forbidden wording; the 50% and
5-point screening thresholds and the 3 s/min target; Amendment 1's low-baseline rows apart from
row 3's seed wording; validation's descriptive role and Stage 4's order (freeze, seal, then
access); the registered seed formula; the bootstrap's computation and its caveat; the
average-single-model estimand with no ensembles.

### Amendment 3A — 2026-11-27, after the synthetic Stage 1, before any real-data step

**State when made:** unchanged from Amendment 3; in addition, Step 2a's synthetic implementation
had been built and tested on generated data only (commits up to `d20ab59`). No project data was
read for it.

**Why.** Under Amendment 3's disclosure check (A3.5, N1.7), the item D exemption (A3.6) and the end
codes, the size of G_T can be published only when the item D MUSDB-train figure is 0–2 (shown as
"0–2") or 24–26 (shown as "17–19"); the same holds for G_V. For every value in between, one value
the size could show would reveal that the low-rate group and every validation group have no
match, so the check withholds it. Step 2a's exact count check proves this for every value. A3.7's
trigger (a published size of 3 or more) could therefore never reach a feasible rerun.

**Replacement of A3.7, rules 1, 2 and 4** (rule 3, G_T first, is unchanged):

- **Rule 1, trigger.** The G_T rerun is considered only from the published item D MUSDB-train figure d,
   and the G_V rerun only from the MUSDB-validation figure, never from membership or from the size
   of G_T or G_V. It is considered when d is at least 3. Given d, the removed set can hold from
   max(0, d − 7) to min(19, d) paired groups (7 is the number of assessed groups that are not
   paired: the low-rate group and the 6 validation groups).
- **Rule 2, feasibility.** F1–F5, and F6 at step 5, must pass for every possible removed set of every
   possible size of one or more in that range, among the 19 paired groups. F1 fails from the
   figure alone when d is 14 or more. F6 prices the most expensive possible set. If the actual
   removed set turns out to be empty, the rerun still runs and repeats the main run, so whether a
   rerun ran never depends on membership.
- **Rule 4, results are private.** Every rerun result and verdict, the actual size of the removed set,
   the rerun's compute and timing records, and every step 3 figure computed without the removed
   groups (the sensitivity sets' voiced and no-voice durations included) stay private, for the
   owner only. The only public facts are whether each rerun was considered and run, which follow
   from the public item D figures and the public feasibility outcome, and planning estimates
   priced at the most expensive possible set. The public report says only: "A sensitivity rerun
   without the groups exposed to MUSDB [training | validation] material was [run | not run]; its
   results are private."

**Replacement of A3.6's public wording for step 2 stops and re-runs** (a second finding, Step 2a
test C4-T): with the approved re-run sentence published, a report that lacks it tells a reader
that no assessed group had any MUSDB-test candidate before the owner's review, which fixes every
group's before-review test status. Instead, **only step 2's final outcome is public**: either the
published report (after RG2) or, if A5.2 halts, the single sentence "Step 2 stopped for owner
review. No figures from this run are published." Intermediate stops and re-runs, and their
authorisations, are recorded only in the private run records. C4-T shows that every
before-review status then stays uncertain.

**Consequences for other text.** A3.5's N1.6 rule 5 (publishing a sensitivity set's totals when
the removed set has at least 3 groups) no longer applies, since those totals are now private. The
sensitivity folds stay recorded only as hiding commitments (A3.6). The N2 seal verifier uses the
private sensitivity sets.

**Where the wording changed** (originals struck through and kept visible): the amendment note at
the top; Amendment 3, A3.5 (N1.6, rule 5), A3.6 (the public wording of step 2 stops and re-runs)
and A3.7 (rules 1, 2 and 4).

**Unchanged:** everything else in Amendment 3, including the screening rule, which the reruns
never change.

### Amendment 3B — 2026-11-28, after Step 4's synthetic implementation, before any real-data step

**State when made:** unchanged from Amendment 3A; in addition, Step 4's synthetic implementation had
been built and tested on generated data only (commits up to `7b88dff`, and the measurement of this
amendment's rules). No project data was read for it.

**Why.** Step 4's measurement of the disclosure check's cost (A3.10, T8), on generated worlds at the
registered size (15 groups with both classes and 4 with no-voice time only), found that the Stage 3
report would publish almost none of its main results:

1. *The cautious reader cannot be certified.* N1.1 assumes, when results are checked, that the reader
   knows every group's directions, leave-one-group-out outcomes and eligibility; N1.7 item 8 forbids
   taking them from their actual values, so every possibility must be checked within a registered
   size limit. Even after an exact reduction, one protected group would need about 4 × 10²⁵ cases, so
   N1.7 item 7 withholds every table: 0 of 27 figures under policy D, 0 of 35 under policy C.
2. *Wide natural limits.* Under N1.7 item 5 a figure is released only if every value it could show is
   safe. Once C's false-activity average is public, B0ₜ's average could in principle show a value near
   60 s/min, which would prove C lower in every group; so it is withheld whatever its true value, and
   D_FA and the ratio with it. A seed-index aggregate could show its largest possible value, which
   would pin every other seed's values, so the seed aggregates are withheld.

**Replacement 1, the reader (A3.5, N1.1 and N1.7 item 8).**
- N1.1's cautious assumption is withdrawn. The reader knows the cumulative public record (A5.1's
  reports, the figures already in this protocol, and every earlier table and release) and the
  registered structure, and nothing else about individual groups.
- Each group's **direction**, **leave-one-group-out outcome**, **per-class eligibility** and **G_T and
  G_V membership** stay protected in their own right (N1.3): each must keep both of its possible
  values, for single groups and, for membership, for pairs, whatever the public record shows.
- N1.7 item 8 becomes: decisions never use private values, and no group's direction,
  leave-one-group-out outcome, eligibility or membership is assumed known to the reader.
- *Groups that the public structure cannot tell apart* may be checked through one representative:
  when every published figure weighs the groups of a public class equally and they share the same
  limits, the public record is unchanged by swapping them, so a range shown for one stands for all
  of its class (and one representative pair for all pairs of the same classes). The checker verifies
  the condition on every table and checks every group where it does not hold.

**Replacement 2, display caps (A3.5, N1.6).**
- A false-activity average (any arm, any table) is shown as a number only up to a cap fixed by its
  table's group-count band: **35 s/min for 3–4 groups, 45 s/min for 5–9, and 50 s/min for 10 or
  more**. A larger value is shown as "above [cap] s/min". The main results tables (19 groups) use
  50 s/min.
- *Why these values.* With m the smallest count of the band, a code window above 60·(m − 1)/m s/min
  could prove that one arm is lower in every group (40, 48 and 54 s/min for 3, 5 and 10 groups); each
  cap lies below that with a margin. They are operational choices, measured on generated worlds only.
- A difference or ratio whose source figure is shown as a code is shown only as "coded with its
  sources", and adds nothing to the public record (N1.6: coded consistently with the figures it comes
  from).
- The §13 verdicts are unchanged: whether D_FA is below 0, Δ_R at least −0.05 and the ratio at most
  0.5 are still decided and checked as verdicts, whatever the numbers show.

**Replacement 3, the seed spread (§12 "Training-seed spread"; A3.5 N1.2).**
- Instead of "the effect sizes for each seed-index aggregate s ∈ O", the report publishes **whether the
  requirement held in every seed-index aggregate (D_FA < 0 and Δ_R + 0.05 ≥ 0): yes or no.** Each
  aggregate's effect sizes become private, like the individual outer models' results.
- The primary effect sizes stay the means of the seed-index aggregates, and §13 (c)'s robustness
  requirement is unchanged; only what is published changes.
- The yes/no verdict is checked with **union semantics**: a protected quantity's range over the "no"
  outcome is its range over every way the requirement could fail (any seed, either condition),
  computed exactly from each way's own range.

**Tests required before use (A3.10, added):** caps shown as codes above them, and a source shown as a
code making its difference and ratio "coded with its sources"; the cap of each band below its
direction threshold; the union verdict (each alternative pinning a quantity while the union keeps
its width, and a single pinning alternative failing); no figure decided after a verdict that leaves
the public record a union; the representative-group condition verified on every table and an
asymmetric table caught; the proposed rules needing the public-record reader; and T8 repeated under
the new rules.

**Measured (T8, generated worlds at the registered size; a comparison, not evidence about real data):**

| Policy | World | Under Amendment 3 | Under 3B (caps 20, 30 or 50 s/min) |
|---|---|---|---|
| D | typical | 0 of 27 | 11 of 11 main figures and verdicts at every cap |
| D | near zero | 0 of 27 | 9 of 9 (D_FA coded with C's code "below 0.1") |
| D | small effect | 0 of 29 | 11 of 11 |
| D | B0ₜ around 37 s/min | — | 11 of 11; at a 30 s/min cap B0ₜ shows "above 30" and D_FA and the ratio are coded; at 50 it shows the number |
| C | typical, near zero, small effect | 0 of 35-37 | 11 of 11 (9 of 9 near zero) |
| C | B0ₜ around 37 s/min | — | 11 of 11 at caps 20 and 30 (B0ₜ "above 30" at 30) |

(The counts are over fewer figures under 3B because the seed-index aggregates are replaced by one
verdict.) With baselines just above the cap and C near zero, B0ₜ's average was published at caps up
to 40 s/min for 3 groups, 45 for 4, 45 for 5 and 50 for 9-10, and withheld above those, as the
threshold predicts.

**Not covered by this amendment (still to be modelled before step 6):** the counts unit of the main
results (direction counts, the leave-one-group-out count and verdict, and the screening class, which
depends on the leave-one-group-out verdict) and the other tables (breakdowns and route rows, AUC
tables, effects at t ± 0.05, the §10 sensitivity results, onset and offset errors, validation). Until
they are modelled they are withheld whole (N1.7 item 7). Step 3's durations stay a fixed unit.

**Limits (N1.10, added).** The widths now hold only against a reader who knows the public record; a
reader with private knowledge of individual groups (for example, how a particular artist's recordings
sound) is not modelled. Strict inequalities in verdicts enter as their closed forms, as rounding
windows do.

**Where the wording changed** (originals struck through and kept visible): the amendment note at the
top; §12 (training-seed spread); Amendment 3, A3.5 (N1.1, N1.2, N1.3, N1.6, N1.7 item 8, N1.10) and
A3.10.

**Unchanged:** everything else, including the screening rule, the widths, the zones near the lower
limits, the effect sizes and their robustness requirement, and Amendment 3A.
