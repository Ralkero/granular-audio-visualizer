# Vocal presence models: the pre-registered protocol (Sonic Character, Phase A5.2)

**Status: pre-registered on 2026-11-26, before anything was run.** At the time of this
commit:

- no model has been trained, and no prediction has been made;
- no feature has been extracted;
- no fold file exists.

The owner approved the design (proposal revision 4) on 2026-11-26. Every rule below was
fixed without seeing any result. A later change is an **amendment**: appended at the end
with its date and reason, and made before the predictions it could influence exist.

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
  statistic. Model C alone is reported on it, descriptively.
- **Validation:** 28 excerpts from 6 groups (52 s of voice from 4 groups), used once and
  descriptively (§13).
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
  SHA-256(`gmv-a5.2:{arm}:{outer}:{inner}:{seed_index}`).
  - Seed indices: 0–1 for inner models, 0–4 for outer and final models.
  - Fold assignment uses `gmv-a5.2-inner-folds`, and the descriptive bootstrap uses
    `gmv-a5.2-bootstrap`.
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
4. Choose t by maximising J(t) on those predictions.
   - J(t) = ½ · (mean group recall + mean group specificity), each group counting once
     in its class (§11).
   - Candidates: on = t, off = t − 0.2, t ∈ {0.30, 0.35, …, 0.80}.
   - Ties go to the t closest to 0.5, then the higher one.
5. Train the outer model on all 18 groups.
6. Predict g once, and score it at t.

- **B0ₜ** uses J(t) on B0's scores for the 18 groups. B0 is fixed, so its scores are
  already group-disjoint.
- **Final models** (for validation) are trained on all 19 groups. Their operating point
  comes from the 19 outer predictions.
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
- **Duration-weighted (pooled):** the same quantities over all frames, always shown beside
  the group averages.
- **Also reported:**
  - missed voice time;
  - ROC-AUC and PR-AUC;
  - phrase metrics: detected and missed events, false phrases per minute of no-voice time,
    onset and offset error;
  - no-voice material split into vocal recordings and instrumental recordings, and by the
    metadata category of each excerpt's selection;
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
  with no test, and quartiles of the per-group differences.
- **Training-seed spread:** the effect sizes for each of the 5 single-seed outer models.
- **Leave-one-group-out influence:** each effect size recomputed without each group. This
  removes the group's test result, not its influence on training.
- **Operating-point stability:** the distribution of chosen t across folds, the agreement
  between the inner seeds, and the effects at t ± 0.05.
- **Descriptive group-bootstrap intervals** (10,000 paired draws, percentile method, group
  as the unit), always with this text: *"Descriptive only. These treat dependent
  cross-validation results as independent, ignore training-set variation, and are
  probably too narrow. They are not evidence of superiority, non-inferiority or
  equivalence."*

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

A ratio interval is shown only when the point-estimate denominator is at least 3 s/min,
together with the share of bootstrap draws where it fell below.

## 13. Screening rule: what to do next (not an inference)

| Result | Rule (point estimates, development, C against B0ₜ) |
|---|---|
| **Promising: carry forward to confirmation** | All of: (a) the false-activity ratio is ≤ 0.50 — or, under §12.1, FPA_macro(C) ≤ 3 s/min and D_FA ≤ 0; (b) Δ_R ≥ −0.05; (c) the direction of D_FA and the sign of Δ_R + 0.05 hold in all 5 seeds and in every leave-one-group-out recomputation |
| **Mixed** | Lower false activity than B0ₜ without meeting "promising" |
| **Not promising** | False activity not lower than B0ₜ, or Δ_R < −0.05 |

B1 is screened against B0ₜ in the same way. If B1 alone is promising, the cheaper retrain
is the candidate to carry forward.

**The thresholds are owner decisions:**
- **50% reduction:** the smallest change expected to be clearly visible.
- **5-point recall margin:** a third of A5.0's 15% missed-voice budget, about 0.5 s per
  10 s of singing.

## 14. Validation and the future confirmation

- **Validation is opened once,** only after the owner's separate authorisation:
  1. final models and operating points are frozen;
  2. predictions for B0, B0ₜ, B1 and C are sealed with SHA-256;
  3. the answers are read once.
- **It is descriptive only:**
  - false activity on the material it contains (no guitar-type negatives);
  - recall and missed voice (52 s from 4 groups);
  - phrases;
  - per group and per excerpt privately.
- **It cannot confirm anything,** even in principle. With 5 no-voice groups and 4 voice
  groups, the smallest attainable one-sided sign-flip p-values are 1/32 and 1/16, both
  above 0.025, even if every group favoured C.
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
- **Per CNN model:** 1.5–10 minutes, central about 3, so about 6–40 h of CPU in total.
- **B1:** about 0.5–2 h.
- **Caches:** about 120 MB in ignored `local-evidence/`.
- **Memory:** under 8 GB with 4 processes.

**Gates** (decided from timing and training loss only, before any held-out prediction):
- **Timing:** if the Stage 2 timing run projects more than 16 h for policy C, use
  **policy D** instead: inner 1 seed, outer and final 3 seeds, 155 CNN models.
- **Convergence:** if training loss has not levelled off within 30 epochs on development
  training data, an amendment may raise the epoch count.

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

## 17. Data protection

- Only the development and validation presence histories are read; validation is read
  only in Stage 4.
- No final-test path is opened.
- Audio, titles, identifiers, paths, features, weights, fold files, predictions and
  per-group results stay in ignored `local-evidence/`.
- Tracked documents carry aggregates only.
- No external audio or dataset is downloaded. The only download is the CPU-only PyTorch
  build for the separate experiments environment.
- Annotation histories, manifests, the split, checksums and backups are never modified.

## 18. Stages

| Stage | What |
|---|---|
| 0 | This protocol, and the experiments environment |
| 1 | The harness and the §16 tests, fold files with their hashes recorded here, and an aggregate check for overlap with MUSDB18-HQ training material |
| 2 | B1 and C, and the timing and convergence gates |
| 3 | The nested development run: predictions sealed, then the §12 report |
| 4 | Validation, descriptive, after separate authorisation |
| 5 | Documentation and a draft pull request with the screening result |

## Amendments

None yet.
