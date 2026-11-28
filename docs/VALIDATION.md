# Validation and measurements — 2026-11-03

Environment: Windows 11 Home build 26300, AMD Ryzen 9 5950X, Python 3.14.3. Live system capture used a Sound BlasterX G6 output at 48 kHz / eight channels. The application and tests used CPU processing only. Playback was already running in a browser; it was not restarted, rerouted, paused or modified for these tests. No raw live audio was saved.

## Live measurements

| Run | Duration | Feature frames | Audio drops | DSP p95 | Estimated capture-center → publication p95 | Sampled process RSS |
|---|---:|---:|---:|---:|---:|---:|
| System, initial logging test | 12 s | 593 | 0 | 1.87 ms | 29.56 ms | 57.96 MB |
| System, longer run | 120 s | 5,994 | 0 | 1.85 ms | 30.05 ms | 58.32 MB; peak 58.42 MB |
| Browser process tree | 10 s | 484 | 0 (including native queue) | 1.41 ms | 29.26 ms | 48.72 MB, Python only |
| System + optional delayed HPSS | 15 s | 744 | 0 | 1.88 ms | 30.77 ms | 118.53 MB |

Long-run statistics retain the last 4,096 feature frames, approximately 82 seconds; counters cover the whole session. Last sampled Python process CPU was 20.3% of one core for the 120-second run (not 20.3% of the whole 32-thread machine). Sampling is once per second. Native helper CPU/RSS is not included in the Python process metrics. These short runs establish an initial baseline, not long-duration reliability or a universal performance guarantee.

The optional HPSS run completed seven two-second chunks: processing median 59.25 ms, p95 60.50 ms, max 60.57 ms. Audio evidence is published after the chunk is available, so delay is approximately two seconds plus processing/queue time. This is a spectral harmonic/percussive balance estimate, **not neural stem separation**.

### Timing limits

Measured: Python queue delay, DSP duration, estimated capture-center age, bus publication timestamps and session resource samples. Not measured: acoustic/DAC presentation delay, calibrated output-device latency, future renderer or display latency. The tested system backend uses callback-anchored sample timestamps because PortAudio's ADC timestamp was unusable; process capture supplied WASAPI QPC timestamps. No sub-30 ms physical end-to-end synchronization claim is made. Window length is 46.44 ms and hop is 20 ms; half-window accumulation is included in center-age estimates.

### Process isolation

A simultaneous six-second comparison captured an inactive Steam process tree and the system mix. Steam-target frames had peak 0.0, while system frames reached 0.04635 full scale with current browser playback. A separate browser-target capture produced live nonzero features. This verifies exclusion of that unrelated browser playback in the tested configuration; it is not a comprehensive multiple-player isolation or Spotify test.

### Desktop verification

The actual Windows control-center window was inspected and operated. Start enabled capture, disabled source edits, displayed changing input/transient meters, source format, timestamped events and performance values. Stop returned to the ready controls and finalized logs. The initial GUI run showed zero audio/UI drops and roughly 116 MB process RSS. GUI performance includes Qt and its polling overhead and should not be compared directly to headless process RSS.

Final GUI smoke test also selected the browser process source and enabled delayed analysis: 2,206 feature frames, 22 advanced jobs, zero capture/UI/native drops and no errors. DSP p95 was 1.66 ms, estimated center-age p95 29.53 ms, and advanced job p95 42.99 ms for stereo input. Final process RSS was 208.46 MB (sampled peak 212.80 MB), including Qt and SciPy. Stop finalized the session, with debug recording confirmed off.

## Synthetic validation

Final automated suite: **30 tests passing**; Ruff checks pass; .NET Release build succeeds with zero warnings/errors. A Qt runtime smoke test covers native DLL/plugin availability in addition to core tests. Removing the initially installed full PySide bundle during dependency minimization removed shared Qt files in this development environment; reinstalling the locked Essentials/shiboken packages repaired it, and the actual window was relaunched successfully. Fresh setup installs only the Essentials package.

Automated tests cover source/bus contracts; bounded queues; slow-subscriber isolation; lossless file backpressure; exact file sample offsets and partial last blocks; 22.05/44.1/48/96 kHz tones; opposite-phase stereo; known RMS and band power; zero-crossing rate; silence/resumption; known onset times; synthetic 120 BPM tempo; intensity ordering; pipeline/session/replay consistency; malformed native packets; simulated device changes; format changes and gaps; startup failure; duration-capped opt-in debug recording; bounded slow advanced processing; and live overload accounting.

Pulse timing test: at least 21 of 23 known 0.5-second-spaced attacks must have a detected onset within 50 ms. Tone tests require centroid within 3 Hz of 1 kHz, known RMS within 1.5%, band mean-square power within 1%, and >99% of tracked energy in the expected band. These are fixture-specific assertions, not music-dataset precision/recall scores.

Reproducible CPU benchmark: `python scripts/benchmark.py --seconds 60`. A 60-second 48 kHz stereo fixture (tone, silence, 120 BPM bass pulses, opposite-phase channels) completed in **3.92 seconds**, about **15.3× real time**, with 2,998 feature frames, no drops, and DSP p95 **1.52 ms**. Fast offline queueing deliberately waits for space; its queue delay is not live capture latency. Logging was disabled for this isolated benchmark. Output lives in ignored `local-evidence/`.

## Hardware and musical validation still needed

- Spotify playback, pause/resume, process-tree identity, closure/restart, and multiple Spotify sessions.
- Actual USB unplug/replug and default-device switch while playing. Recovery control flow was fault-injection tested; no system playback settings were changed for validation.
- Other DACs, Bluetooth, HDMI, headphones/speakers, unusual driver channel layouts, exclusive-mode players and protected content.
- Longer runs, sustained background load, disk-full logging failures and helper-process resource profiling.
- An annotated, rights-cleared music corpus to measure onset/tempo/candidate precision and recall. Current browser audio demonstrates capture, not musical classification accuracy.
- Neural vocals/drums/bass/other separation: no Demucs weights were downloaded or benchmarked. The CPU HPSS timings must not be presented as Demucs performance.
- Linux/PipeWire capture implementation and validation.

Private session evidence is retained locally in `sessions/`; the public-safe source and documentation do not include audio, screenshots, app identifiers, process IDs or library metadata.

## 2026-11-15 addendum: system loopback clock

The callback-anchored PortAudio system backend lost whole 10 ms blocks without any counter or flag when a second engine ran in the same process (90 ms in 600 s; nine blocks), which is the drift reported in Phase 1.1. A native WASAPI loopback backend with per-packet QPC checks replaced it as the default when the helper is built: in the same paired 600 s configuration its timestamp-vs-arrival drift was 0.001 ms with zero drops, discontinuities and re-anchors, and estimated capture-center age p95 was 33.8 ms. See [Milestone 7](PHASE1_1_REPORT.md) for tables and limits. A second native run with the paired application audibly playing gave -0.017 ms drift, again with zero drops and discontinuities. This is one device and two runs; physical latency and physical device changes remain unmeasured/unrun. The "Timing limits" statement above about the system backend describes the PortAudio fallback only.

## 2026-11-24 addendum: Sonic Character, Phase A

Full results and limitations: [SONIC_CHARACTER.md](SONIC_CHARACTER.md). The three kinds of evidence, kept apart:

- **Automated tests** (`tests/test_character.py`, 41 tests): the measurements and their independence from level; normalisation at and beyond the anchors; missing and invalid values; each dimension; confidence; the three windows, single transients, real changes, silence, pauses, quiet captures; segments, stale messages, seeks and restarts; determinism across repeated runs, update intervals and reads; a live engine run against its own session log and the application's replay at two speeds; the diagnostic outputs. Full suite on CachyOS: 603 passed, 3 skipped, 3 failed. The three failures are the existing `tests/test_remote.py` ones (Python's multiprocessing resource tracker counted as a leftover worker), present on `main` before this phase; the baseline on `main` was 562 passed, 3 skipped, 3 failed.
- **Stand-in integration** (offline analysis of real recordings through the real analysis pass, CachyOS): 39 recordings used for calibration and 28 held out, each analysed a second time 20 or 10 dB quieter; both reference tracks of the brief, the second with the definitions already fixed (they measure alike as whole recordings: the main limitation found); one real recording put through the engine into a session and read back, identical to the direct analysis in all 8,559 frames; four older sessions recorded on Windows read back and correctly flagged as incomplete. Cost: 41 µs a frame beside 764 µs for the analysis pass.
- **Real-hardware validation: none.** Sonic Character has not been run during live playback, on Linux or Windows, and nobody has compared its values with what they hear. Not run on Windows at all: the code is plain Python and NumPy with no platform-specific part, but that is an expectation, not a test. The installed Arch package predates this phase and was left as it is; a wheel built from this branch contains the new package.

## 2026-11-24 addendum: Sonic Character, Phase A2 (research)

Full results: [SONIC_CHARACTER_RESEARCH.md](SONIC_CHARACTER_RESEARCH.md). Research and offline experiments only; nothing was integrated.

- **Automated tests** (`tests/test_character_research.py`, 23 tests, generated signals): the harmonic/percussive split; pitch classes, harmonic movement, key fit and tuning; the lead-line summaries and what is unknown without a pitch or without the activity model; pulse, its stated ambiguity, busy against spacious, triplets against straight, no pulse found; 20 dB quieter; determinism; that the Phase A profile made beside the experiments is the Phase A profile; the group-disjoint split; the corpus statistics; the commands, and that reports carry ids and never file names. Generated signals show the arithmetic is right, not that anything matches perception. Full suite on CachyOS: 626 passed, 3 skipped, 3 failed, the three being the existing `tests/test_remote.py` failures; the Phase A branch this is stacked on gave 603 passed, 3 skipped and the same 3 failed. `ruff check` clean. The project has no type checker configured.
- **Stand-in integration** (offline, CachyOS): 138 MUSDB18-HQ songs against their own stems (one more was excluded by the pairing check added in the correction below); 23 MDB Drums mixes against annotated beats; 199 of the owner's recordings (104 in 17 calibration groups, 95 in 22 other evaluation groups), 50 of them again 10 and 20 dB quieter, 19 again under six changes of production. Both reference tracks analysed, in evaluation groups.
- **Real-hardware validation: none**, and none applies: nothing here runs during playback. Not run on Windows.
- **Not validated at all:** whether any measurement matches what a listener hears. The two labels used (vocal or instrumental, harsh-vocal genre) come from file metadata and were not checked by listening. The fallback for sample rates at which the activity model cannot be built was exercised on real 22 kHz files but has no automated test, because the model's weights are not in the repository.
- **Correction, same day** (paired drum-removal check and confidence wording): the MUSDB18-HQ comparison of each mixture with the same mixture less its drums now reports paired, per-song differences (median, 90th percentile and largest absolute difference, mean signed difference, share of songs rising and falling, in each measurement's unit and in calibration standard deviations) instead of only rank correlations. On 138 valid pairs per measurement (none missing or invalid), the pitch-class measurements change by a median of 0.05 to 0.24 calibration standard deviations, but four of them shift consistently in one direction: harmonic movement rises in all 138 songs once the drums are removed, tonal stability falls in all 138. Key clarity is the least affected. The first version's "nearly unchanged by removing the drums" overstated this and was corrected. A pairing check (mixture and stems with one rate, length and channel count, and the stems adding up to the mixture within 40 dB) excluded one song whose stems do not add up to its mixture; it had been included before. Two complete re-runs of the stems check gave identical per-song values. Six tests were added (`tests/test_character_research.py`, 29 in all): paired differences with their sign and scale, missing and invalid values, the rank correlation's blind spot, order independence, harmonic share reported apart and the tonal-content split, and the pairing check. Full suite after the correction on CachyOS: 632 passed, 3 skipped, 3 failed; the three are the same existing `tests/test_remote.py` failures as before (a leftover multiprocessing resource-tracker process at `assert not children()`), so no new failure. `ruff check` clean.
- **Second correction, same day** (Spearman rank correlation): the research code's Spearman rank correlation ranked tied values by their position (two nested sorts), which is not Spearman's coefficient when values repeat. It now gives tied values their average rank, returns no value when either column has no variation among the pairs used, and still needs eight valid pairs and drops missing, non-numeric and non-finite ones. Checked against SciPy's implementation on generated data with many ties. The stems evaluation was re-run with the same datasets, activity model, calibration scales and options: every per-song value and every paired-difference figure was identical, two rank correlations changed in the fifth decimal (the mix's pitched share against the vocal stem, 0.59978 to 0.59983; chord-change rate with and without drums, 0.91324 to 0.91318), the other twenty were unchanged, and no reported figure changed at two decimals. Six tests were added (`tests/test_character_research.py`, 35 in all): average ranks of tied values, agreement with the textbook definition under ties, a perfectly negative relationship with ties, columns without variation, missing and invalid observations, and determinism. Full suite after it, run twice on CachyOS: 637 passed, 3 skipped, 4 failed both times. Three are the existing `tests/test_remote.py` failures. The fourth is the known intermittent `tests/test_app.py::test_replay_runs_inside_the_same_window_with_original_times` (replay finished before the view had taken in every message: 3 and then 4 of 5 onsets; the fix is PR #1, not on this branch). It passed in the two previous full runs on this branch, passed 8 of 8 times on its own and 3 of 3 times with the rest of its file, and nothing this correction changed is run by it. `ruff check` clean.

## 2026-11-24 addendum: Sonic Character, Phase A3 (vocal delivery labelling tool)

Details: [VOCAL_DELIVERY_LABELING.md](VOCAL_DELIVERY_LABELING.md). Research tooling; the visualizer and the analysis are unchanged.

- **Automated tests** (`tests/test_vocal_labeling.py`, 29; generated audio, a stand-in for `pacat`, offscreen Qt): dataset manifest, stable identifiers, sample-exact excerpt associations, excerpt lengths, artist-disjoint split and its fixed rule, reference tracks held out, adding recordings without moving anyone, refusal to rebuild once labelled, missing, unreadable, empty, too short and changed recordings, saving, changing and skipping answers, notes, a damaged line after a crash, resuming, the reproducible order, the backup, playback sending exactly the excerpt at its own rate and channels, pause and resume, no sound program, the window's controls and keys, a second window refused, and no artist, title, file name, folder, method or split anywhere in the window. All 29 pass.
- **Full suite** on CachyOS, run three times on this branch: 665 or 666 passed, 3 skipped, 4 or 5 failed. Every run has the three existing `tests/test_remote.py` failures. The known intermittent `tests/test_app.py::test_replay_runs_inside_the_same_window_with_original_times` (fixed by PR #1) failed in two of the three. `tests/test_recognition.py::test_a_slow_model_skips_hops_instead_of_queueing`, a timing race (it gives a 0.4 s model 50 ms to publish its last answer), failed in two of the three; it had not been seen failing before. It passed 10 of 10 times alone, 3 of 3 times with its neighbouring test files and the new one, and in a full run of this branch without the new test file; this branch changes no code it uses. Recorded as intermittent and not caused by this phase, though only by that evidence. For comparison, PR #8's head on the same day: 637 passed, 3 skipped, 4 failed (the three `test_remote` and the replay test). `ruff check` clean.
- **Real machine** (CachyOS, KDE Plasma Wayland, PipeWire, the Sound BlasterX G6 as default output, window on the HP Omen display): `scripts/validate_vocal_labeling.py` on a temporary copy of the real excerpt list, 28 of 28 checks under Wayland and under X11. Playback created a stream on the default output at the excerpt's own format, paused and resumed at once, ran in real time and ended at the excerpt's end; answers by click and key, changes, notes and skips were saved; the window resumed after being closed. Excerpts at 48, 44.1 and 22.05 kHz each played at their own rate. Whether the sound was audible was not checked by a person. The owner's own labelling folder was left with no answers.
- **Not tried:** Windows playback.
- **Revision, same day** (new answers, song titles, migration): the five answers are now Cleans, High Screams, Low Screams, Gutturals and Sing-Screams, with the owner's definitions in the window; "No vocals" and "Review later" are skips with a reason, not answers; each excerpt shows its song title (from the music inventory by checksum, one title given by the owner for a file without metadata; never from a file name), kept in `titles.json` beside an unchanged `dataset.json`. Answers given with the first choices are kept, shown as earlier answers and counted as still to label; none is translated. Tests: 39 in `tests/test_vocal_labeling.py` (10 added), all passing. Real machine: 33 of 33 checks under Wayland, including titles, the new answers, a changed answer and the two kinds of skip. Full suite: 676 passed, 3 skipped, 4 failed (the three `tests/test_remote.py` failures and the known intermittent replay test); the intermittent recognition test passed in this run. `ruff check` clean. The owner's `dataset.json` and `labels.jsonl` are unchanged byte for byte.
- **Owner's use** (real hardware, owner-evaluated, 2026-11-24): the owner listened to and labelled all 96 excerpts in the window on CachyOS. Answers were saved with every click and survived several closes and reopenings, and the seven answers given with the first choices were redone. The save file has 103 entries and no damaged line, and the excerpt list is unchanged. Results in counts only: [VOCAL_DELIVERY_LABELING.md](VOCAL_DELIVERY_LABELING.md), "Labelling results". The owner did not report any playback problem; whether every excerpt was clearly audible was not asked separately.
- **Backup correction (2026-11-25):** a backup now also holds `titles.json` and `reference_vocalists.txt` when they exist, taken by name, so no audio or lock file is ever copied. A missing optional file is left out without error. 4 tests were added (43 in `tests/test_vocal_labeling.py`, all passing), using temporary test data only; the owner's own files were not used. Full suite: 681 passed, 3 skipped, 3 failed (the three `tests/test_remote.py` failures; both intermittent tests passed this run). `ruff check` clean.
- **Counting fix (2026-11-25):** after the owner's second batch the window reported some excerpts as unanswered although every one had an answer or a No vocals mark. Cause: No vocals marks were counted as still to label, and old-choice answers redone as No vocals were still reported as waiting. Now No vocals counts as done, and only never-answered or review-later excerpts are offered or counted as still to do. The saved answers were not changed. 2 tests were added (45 in `tests/test_vocal_labeling.py`, all passing). Real machine: 33 of 33 checks under Wayland. Full suite: 682 passed, 3 skipped, 4 failed (the three `tests/test_remote.py` failures and the known intermittent replay test). `ruff check` clean.
- **No vocals needs playback (2026-11-25):** when a recording could not be played, the five answers were off but **No vocals** and its N key still worked. Now the No vocals button is off, the N key does nothing and the window's skip action refuses No vocals while the excerpt cannot be played; **Skip / review later** still works and does not replace an earlier No vocals decision; with playback available nothing changed. 6 tests were added (51 in `tests/test_vocal_labeling.py`, all passing): button off, N key, the action called directly, review later, decisions already saved, and normal labelling. Real machine (Wayland, temporary copy of the excerpt list): 36 of 36 checks, including 3 new ones on an excerpt whose recording was made unavailable in the window's memory only. Full suite: 688 passed, 3 skipped, 4 failed (the three `tests/test_remote.py` failures and the known intermittent replay test). `ruff check` clean.

## 2026-11-25 addendum: Sonic Character, Phase A4 (vocal delivery recognition, research)

Details: [VOCAL_DELIVERY_RECOGNITION.md](VOCAL_DELIVERY_RECOGNITION.md). Research only; nothing in the application uses it.

- **Automated tests** (`tests/test_vocal_delivery_recognition.py`, 30, generated audio and a generated dataset; the owner's answer files are never used): only development answer lines are decoded (held-out lines carry a poison value that never appears anywhere); held-out totals without answers; exclusions by reason, with the guest-vocalist excerpts' answers taken from the record; a missing, mismatched or partial leakage note, or a tampered split, stops the experiment; the answer folding matches the labelling tool's own rules on random histories; folds never share an artist group, artist or recording; a person credited on two artists' recordings joins their groups; features deterministic, unknown is never zero, renaming every identifier leaves them unchanged; unavailable or changed audio is dropped and counted; the cache is used only for the same audio and feature version; preprocessing and class weights come from the training fold only; median imputation with indicators; models deterministic and above the majority on a clear signal; metrics, per-group results, the within-recording reading and abstention against hand counts; the whole experiment twice with identical results and a public summary that names nothing; a public summary carrying a private value is refused; outputs go to an ignored folder; the command stops on unusable data.
- **Real data, development side only** (owner's 412 labelled excerpts; 266 usable): two complete runs on CachyOS, the second measuring every excerpt again without the cache. Features, all 266 out-of-fold probabilities and every metric identical; the pair was repeated after a reporting change, again identical. 30 to 31 s to measure (6 processes), 62 to 64 s per run. 0 held-out answer values decoded (349 development lines decoded, 73 held-out lines matched for totals only).
- **Results** (development, leave one artist group out, exploratory model choice): balanced accuracy 0.538 logistic and 0.539 forest against 0.333 majority; clean against harsh 0.89 and 0.92; within recordings that have both, 0.85 and 0.91; high against low 0.40 and 0.36.
- **Not done:** any test on the held-out artists (deliberately); listening checks of individual predictions; live or real-time use (out of scope).
- **Full suite:** 718 passed, 3 skipped, 4 failed (the three known `tests/test_remote.py` failures and the known intermittent replay test, as before this phase). `ruff check` clean.
- **Held-out test command (2026-11-25), built from the pre-registered plan and its Amendment 1** ([VOCAL_DELIVERY_HELDOUT_PLAN.md](VOCAL_DELIVERY_HELDOUT_PLAN.md)): `gmv character research heldout predict|score`. 21 tests on generated data (51 in `tests/test_vocal_delivery_recognition.py`, all passing). They cover: the guarded interval rejects and redraws resamples lacking clean or harsh and reports draws, rejections, valid resamples and the artist groups per class; it gives no interval with fewer than 3 groups for a class, with more than 25% rejected draws, or with a class missing; the two-class score is never taken over one class; the verdict table, including breadth; the predict step decodes no held-out line, never calls the answer reader, writes a prediction for every held-out excerpt, flags the precaution recording, seals the predictions with their SHA-256, gives the same seal on a second run, never overwrites, and stops if the training set is not Phase A4's; the coarse reader keeps only clean, harsh, other, No vocals, review later or unanswered; scoring refuses a wrong or broken seal without opening anything, records the opening before reading answers, stores no fine answer, and refuses to run twice; the command refuses to score without the seal and the explicit flag. Full suite, two runs: 738 passed, 3 skipped, 5 failed, then 740 passed, 3 skipped, 3 failed. The three `tests/test_remote.py` failures are known. The other two in the first run (`tests/test_radiant_plasma.py::test_custom_colours_reach_the_effects_they_belong_to_and_stay_saturated` and `tests/test_review_experiment.py::test_the_two_arms_of_a_fold_differ_only_in_the_guitar_clips_of_the_owners_recordings`) passed alone, twice in their own files and in the second full run. They are recorded as intermittent and were not investigated further; neither touches this phase's code, and both run before its tests. `ruff check` clean.
- **Held-out scoring consistency check (2026-11-26)**, a safety correction after the seal and before scoring ([VOCAL_DELIVERY_HELDOUT_PLAN.md](VOCAL_DELIVERY_HELDOUT_PLAN.md), "Implementation note"). Scoring now checks, before recording the opening or reading any answer, that the sealed predictions belong to the supplied dataset: records parse with exactly the predict step's fields, the evaluation excerpts are exactly the same set, and each record's recording, artist group and precaution flag match. The precaution flag is recomputed from the same inventory, which scoring now requires. The manifest's count and largest group also match. 16 tests on generated data, each with a properly re-sealed artifact so only the new check can catch the problem, and with the answer reader set to fail if called. They cover a missing, development, unknown or duplicated excerpt; recording, group and precaution mismatches; unparseable, malformed and mistyped records; a missing inventory; another labelling folder with as many excerpts (renamed recordings, swapped identifiers); no opening marker after any failure; and a valid attempt still scoring once afterwards. 67 in `tests/test_vocal_delivery_recognition.py`, all passing. Full suite: 756 passed, 3 skipped, 3 failed (the three known `tests/test_remote.py` failures: on Linux the multiprocessing resource tracker is counted as a leftover child process; the fix is on a separate branch outside this stack). `ruff check` clean. Dry check on the real sealed predictions, opening only them, the manifest, `dataset.json` and the inventory (an audit hook confirmed `labels.jsonl` was not opened): consistent, 69 excerpts, 12 artist groups, 2 precaution excerpts. The predictions file is byte-for-byte unchanged and scoring has not run.
- **Held-out scoring, once (2026-11-26 03:21 UTC, which is 2026-11-25 21:21 CST; owner-authorised; commit `39d7881`).** Kind of evidence: an offline evaluation on the owner's labelled held-out excerpts; no hardware or real-time aspect.
  - **Preflight, all passed before anything was opened:**
    - the scoring checkout was at `39d7881`, the remote head of PR #10, with the consistency check present;
    - the predictions SHA-256 was `48aee927…a6a3` and the manifest was unchanged;
    - the answers file SHA-256 was `f241d92f…bc6e`, byte-for-byte identical to the owner's latest verified backup;
    - the music inventory was present and there was no opening marker;
    - the consistency check of predictions against the dataset passed.
  - **One run.** The opening marker records a single opening at 2026-11-26 03:21:39 UTC (2026-11-25 21:21:39 CST). The scoring step read the held-out answers through the coarse mapping (clean, harsh, other or No vocals); the original fine-grained answers were not inspected or retained. There was no retry and no second invocation.
  - **Verdict: INCONCLUSIVE.** There were 16 clean and 4 harsh scored excerpts, against at least 10 of each needed; the thresholds were unchanged.
    - **Score:** clean recall 15/16 (0.94), harsh recall 3/4 (0.75), balanced accuracy 0.84.
    - **Guarded interval:** 0.59 to 1.00, from 2,009 draws with 9 rejected and 2,000 valid. 9 artist groups contributed clean excerpts and 4 harsh.
    - **Breadth:** the largest group had 4 of the 20 scored excerpts, all clean, 3 right. The remaining groups pooled reached balanced accuracy 0.88 on 12 clean and 4 harsh, short of the breadth rule's 5 harsh. Under the pre-registered rule the breadth classification is therefore "mainly on one artist": insufficient evidence of breadth, not an observed concentration. The scoring output leaves its breadth field empty, because the implementation attaches it only to a conclusive verdict.
    - **Precaution excerpts:** 2 of 2 right.
  - **Not scored, described only:** 37 excerpts answered "other" (24 called harsh, 13 clean) and 10 No vocals (9 called clean). The model has no No vocals answer, so the latter is not evidence of detecting a voice.
  - **Integrity afterwards:**
    - the predictions, manifest, answers file and backup were unchanged;
    - the output accounts for all 69 excerpts, and the per-group totals agree with the confusion matrix;
    - no fine answer (High/Low, Guttural/Sing-Scream) was stored or reported;
    - the per-excerpt results stay in the ignored local evidence folder.

## 2026-11-26 addendum: Sonic Character, Phase A5.1 (vocal presence annotation foundation)

Details: [VOCAL_PRESENCE_ANNOTATION.md](VOCAL_PRESENCE_ANNOTATION.md). Research tooling; no model trained or evaluated.

- **Automated tests:** `tests/test_vocal_presence.py`, 32, all passing; generated audio and generated projects only, never the owner's answer files. They cover:
  - schema: bounds and order, union, uncertain masking, statuses, explicit negatives, tags, revisions, the hash chain, crash recovery;
  - identity and registry: spellings, aliases, featured credits, duplicates, MusicBrainz IDs, generic credits, no movement, conflicts, links to consumed material, freeze refusal;
  - preparation: no held-out line decoded, A3 files byte-identical, deterministic and stable manifests;
  - protection: fingerprints, registry consistency, rejected assignments, development loaders unable to read final-test labels (guarded file reads), a final-test manifest unchanged by development changes, broken histories refused;
  - the window: marking at the position, seeking, listening around boundaries, edits, resume, completion guard, titles, backup and restore, unplayable excerpts, the lock, the A3 project untouched, the status command.
- **Real machine** (CachyOS, KDE Wayland, PipeWire, window on the HP Omen; generated audio): `scripts/validate_vocal_presence.py`, 19 of 19 checks:
  - playback stream created, position in real time (1.508 s over 1.508 s);
  - pause ends the stream and holds the position; exact paused seeks and marks; keyboard marking;
  - "listen around" stopped at 2.52 s for a 2.5 s target;
  - completion, backup, no stream after closing, resume, marks kept, history intact.
- **Real data (private):** the presence project was prepared from the A3 project and inventory.
  - The A3 files were unchanged, and no held-out answer was decoded.
  - New recordings' checksums were verified.
  - The final-test freeze was refused: 7 never-used artist groups against 20 required.
- **Full suite:** 787 passed, 3 skipped, 4 failed. The failures are the three known `tests/test_remote.py` failures (Linux resource tracker; the fix is on a separate branch) and the known intermittent replay test (its fix is PR #1, not in this stack). No new failure. `ruff check` clean. After the full run, a privacy correction removed local-path defaults from the `prepare` command (now a required `--inventory` and an optional `--map`); `tests/test_vocal_presence.py` was run again afterwards: 32 passed.
- **Owner's use** (real hardware, owner-evaluated, 2026-11-26), read from the saved history without changing it:
  - **Progress:** all 171 development excerpts completed (89 with a voice, 82 without), from 20 artist groups and 153 recordings.
  - **Integrity:** 289 events, chain intact, no damaged line, none rejected, project check passing, A3 answer file unchanged.
  - **Agreement with A3:** 82 of the 84 A3 voice excerpts have a voice marked; 2 (one Low Screams, one Cleans) are marked no voice, so those answers disagree. All 7 A3 No-vocals excerpts are negatives.
  - **Candidates:** 7 of 80 instrumental candidates had a voice (song intros 3 of 14, rhythm guitar 2 of 20, guitar solo 1 of 20, orchestral 1 of 10, soundtrack 0 of 16).
  - **Audio:** no voice mark over near-silence.
  - **How negatives were judged:** playback is not recorded. On 50 of the 82 negatives, Complete came sooner than the excerpt could play in full; the owner confirmed such answers rest on their knowledge of the recordings (all-instrumental bands, songs known to have none, interludes).
  - **Not reported:** the owner gave no separate feedback on position sync or boundary placement. No boundary was adjusted after marking.
- **Validation partition** (real hardware, owner-evaluated, 2026-11-26), read-only check:
  - **Progress:** all 28 excerpts completed (6 with a voice, 22 without), from 6 artist groups and 25 recordings. A voice fills 19% of the time.
  - **Marks:** 6 voice stretches, no uncertain stretch; median coverage per voiced excerpt 98%.
  - **Integrity:** 34 events; the chain is intact in both the loader and an independent recomputation, and no line is damaged or rejected.
  - **Agreement with A3:** 5 of 6 A3 voice excerpts have a voice marked; 1 Cleans is marked no voice. All 3 A3 No-vocals excerpts are negatives.
  - **Candidates:** 1 of 19 instrumental candidates had a voice.
  - **Separation:** no excerpt, recording or artist group is shared with development; each history holds only its own partition's excerpts; no final-test list exists; the reserved groups are unchanged.
  - **Development:** the history is byte-identical to before. A verified local backup of both partitions exists.
  - **Limit:** only 6 voiced excerpts (4 groups, 2 harsh), so missed-voice estimates on validation will be imprecise.
- **Not done:** Windows playback; any model training or evaluation.


### 2026-11-26 correction: final-test freeze persistence (Phase A5.1)

- **Root cause:**
  - Preparation with `--freeze-final-test` computed the frozen registry, then discarded it. The registry had already been saved before the freeze step, and no final-test manifest was written. A successful freeze would therefore have left no trace.
  - A later preparation run would also have dropped the freeze's timestamp, because the registry rebuild kept only the group entries.
  - The existing test called the freeze function directly, so the preparation path was never exercised.
- **Found while testing the fix:** re-preparing could crash when two rebuilt artist groups touched the same registered group, and it could merge two frozen groups. Both are fixed: linked groups are merged once, and a merge of frozen groups stops preparation.
- **The fix** (details in [VOCAL_PRESENCE_ANNOTATION.md](VOCAL_PRESENCE_ANNOTATION.md), "Freezing the final test"):
  - The freeze is worked out in full before anything is written; a refusal or an unreadable recording writes nothing.
  - The final-test manifest is written first and the registry last, as the commit point. The registry records the freeze time and a digest of the cohort's excerpts.
  - Both are read back. Any failure rolls back to the earlier registry and fingerprints.
  - The project check now requires the registry and the final-test manifest to agree, so an interrupted or altered freeze is refused.
  - A frozen cohort is never replaced or extended.
- **Automated tests** (generated audio and generated projects only): `tests/test_vocal_presence.py` now has 40 tests, 8 of them new. They run the real preparation path:
  - an insufficient cohort is refused, through the function and the command, with the project byte-identical afterwards;
  - a sufficient cohort persists a valid manifest. The project check passes, only eligible reserved groups move, and consumed and too-short groups stay out. The same inputs give the same cohort, and the status command and the annotation window recognise it;
  - repeated freezes, a direct rewrite, and a new never-used artist cannot change the cohort;
  - an interruption at the worst moment (the frozen registry written, its fingerprint not) rolls back byte for byte, and a later freeze still works;
  - half-written or altered freezes are refused by the check, the loaders, the status command and preparation: a manifest without a frozen registry, a frozen registry without its manifest, a manifest without its fingerprint, and a manifest rewritten with one excerpt fewer;
  - development loaders still cannot read final-test labels after a real freeze (guarded file reads);
  - a rebuilt identity that would merge frozen groups stops preparation with nothing written;
  - the deterministic, per-group-capped final-test selection;
  - the registry merge that previously crashed.
- **Full suite:** 796 passed, 3 skipped, 3 failed. The three failures are the known `tests/test_remote.py` failures (Linux resource tracker; fixed on a separate branch). The intermittent replay test passed in this run. `ruff check` is clean.
- **The owner's project:**
  - It was not prepared again, and no cohort was frozen. It has no final-test manifest, and its 7 reserved groups are unchanged.
  - The new project check passes on it read-only, and its files are byte-identical before and after.
  - The development (171) and validation (28) annotations are unchanged.
- **Not done:** a real freeze. It needs about 13 or more new never-used artist groups in the library first.

## 2026-11-26 addendum: Sonic Character, Phase A5.2 Stage 1 (synthetic harness and contract tests)

The protocol is [VOCAL_PRESENCE_MODEL_PROTOCOL.md](VOCAL_PRESENCE_MODEL_PROTOCOL.md). Stage 1 was authorised **for generated audio and generated projects only**:
- no recording, annotation, manifest, registry, private metadata or model weight was opened;
- no feature was extracted from real audio;
- no model was trained, no threshold fitted and no prediction made.

**Automated tests** (all on generated audio, generated projects or generated tensors):

| Area (protocol §16) | File | Tests | Result |
|---|---|---:|---|
| Alignment (engine frame path against model C's front end at 22.05, 44.1 and 48 kHz) | `tests/test_presence_model_alignment.py` | 15 | All pass |
| Loss weighting, plus the labels and collars it uses | `tests/test_presence_model_objective.py` | 14 | All pass |
| Inner folds | `tests/test_presence_model_folds.py` | 17 | All pass |
| Metrics, events, screening (with Amendment 1), bootstrap caveat, post-processing, seeds | `tests/test_presence_model_metrics.py` | 20 | All pass |
| Causality and normalisation of model C | `tests/test_presence_model_cnn.py` | 21 | All pass |

**Environments:**
- **Experiments environment:** Python 3.12.16, torch 2.9.1+cpu, NumPy 2.5.5, SciPy 1.18.1, 16 threads. **87 passed**, 0 skipped.
- **The application's own environment** (Python 3.14.10, no PyTorch): **66 passed**, with the CNN file skipped as designed.
- **Lint:** `ruff check` is clean. `ruff format` is not enforced in this repository; existing modules are not formatted with it either.

**What the tests establish:**
- **Alignment:**
  - The engine's frames and C's frames have equal counts for awkward lengths (one sample short of a window, exactly a window, a hop minus one, and others).
  - Their centres agree within 1 ms, and are 23.22 ms after each window's start, every 20 ms.
  - An impulse peaks in the same frame on both paths, including near the midpoint between frames.
  - Resampling keeps impulse timing within one sample.
  - Pre-roll and scored frames agree, including at the very start and end of a recording.
- **Causality:**
  - Changing input after frame t leaves C's outputs up to t bit-identical, in inference and in training (with the dropout mask held fixed).
  - The receptive field is exactly 73 frames.
  - Deliberately wrong variants (symmetric time padding, BatchNorm) are caught.
- **FrameNorm:**
  - It is identical in training and inference and has no running statistics.
  - The representation before the registered dropout is identical in both modes. Dropout itself is kept, so whole-network outputs differ in training mode as intended.
- **Batch independence:**
  - Outputs are bit-identical when only the companions in a batch change.
  - Across batch sizes they agree to rounding (float32 within 1e-5, float64 within 1e-12), because PyTorch picks a different kernel for a different batch size.
- **Streaming** equals batch output within 1e-10 (float64).
- **Seeds:** the same registered seed gives identical weights.
- **Size:** 154,641 parameters.
- **Loss:**
  - Every group's total weight per class is exactly 1/(2·|Aᶜ|).
  - The weighted sum equals the registered formula, and a longer group gains no weight (pooled weighting would have given it more than 3 times as much).
  - The 2 s minimum is inclusive and per class.
  - Excluded frames change nothing.
  - One epoch's batches reproduce L for any order and batch size; per-batch normalisation would not.
- **Folds:**
  - Assignments are deterministic and every check holds.
  - The outer group is never assigned.
  - Retries are deterministic, exercised with a stricter test-only share.
  - Impossible cases stop with an error.
  - Summaries and messages hold no group keys.
- **Metrics and screening:**
  - Collars are inclusive at 0.3 s.
  - Single-class groups count only in their class.
  - Event collars hold at their exact boundaries.
  - At exactly 3 s/min the ratio is used with nothing added; below it the absolute difference is used and the 50% criterion is "not applicable".
  - The improved, maintained, worsened, mixed and recall-below-margin classes are all tested.
  - Every bootstrap interval carries the caveat, which cannot be passed in or changed.

**Choices the protocol leaves open**, made for Stage 1 and to be reviewed before Stage 2:
1. **Mel and log details:** the HTK mel scale, triangular filters on the power spectrum, and log(mel power + 1e-10). The protocol says only "64 mel bands from 40 Hz to 11 kHz, log power with a floor".
2. **Frame counts:** C takes its frame count from the engine's rule. At 48 kHz its last frame can need up to one resampled sample past the end, which is zero-padded; more than 2 is refused.
3. **Collars near excerpt edges:** "interior" boundaries are those strictly inside the excerpt. A boundary a few milliseconds from an edge therefore gets a collar; the protocol gives no tolerance.
4. **Uncertain time:** no collars are added around the edges of uncertain intervals, which the protocol does not specify. None exist today.
5. **Events at excerpt edges:** the onset and offset collars apply to every reference event as written, including boundaries cut off by the excerpt edge. Timing errors are reported only for interior boundaries. Whether cut-off boundaries should be exempt from matching is not specified.
6. **Folds:** a group with less than 2 s of both classes is refused, because the protocol does not say where to place it. When ordering single-class groups, voice-only come before no-voice-only, the order the protocol lists them.
7. **Final models' seeds:** the outer and inner fields of a final model's training seed are not specified. That is left to Stage 2.
8. **Undefined bootstrap draws:** a draw where no group has a statistic is counted and reported, never dropped silently. In the ratio interval, a draw whose baseline mean is exactly 0 is undefined, and draws below 3 s/min stay in, with their share reported.
9. **Kernel indexing:** PyTorch's convolutions number the taps oldest first, while the protocol writes newest first. This is the same family of functions.

**Not done in Stage 1 (deferred):**
- **Data-dependent work, which needs separate authorisation:**
  - the project's real fold files and their hashes;
  - the aggregate MUSDB18-HQ overlap check.
- **Stage 2 work:**
  - frame silence from real audio (the engine's −65 dBFS rule), which Stage 1 takes as an input;
  - phrases from `VocalContour` gated by B1 and C;
  - any B0 scoring with weights;
  - training;
  - the timing and convergence gates.

**Correction after independent review (2026-11-26, Stage 1 contract):**
- **The problem:** the first Stage 1 commit (56e8801) checked screening robustness with `robust(recomputations)`. That accepted any non-empty list whose rows all passed, so a single passing recomputation could count as robust. That does not meet §13 (c): the direction must hold in **all five** single-seed recomputations **and every** leave-one-group-out recomputation.
- **The interface now:** `check_robustness(seed_recomputations, group_recomputations, expected_groups)`, with `robust(...)` returning its `passed` value.
  - Exactly 5 seed rows (D_FA, Δ_R) are required.
  - One leave-one-group-out row (group, D_FA, Δ_R) is required for every group in the caller-supplied expected set. A missing, duplicated or unexpected group makes the set incomplete.
  - Every row in both sets must have D_FA < 0 and Δ_R ≥ −0.05.
  - An empty or incomplete set is never robust; there is no fallback.
  - Malformed or non-finite rows raise an error.
  - Results and messages report counts only, never group keys.
  - `screen()` uses it in both the relative and the absolute mode, and records the reason in the result.
- **Tests:** 8 new synthetic tests:
  - a complete passing set;
  - one failed seed;
  - one failed leave-one-group-out row;
  - 4 or 6 seeds;
  - missing, duplicated, unexpected or swapped groups;
  - empty or incomplete inputs;
  - both modes held to the same requirement;
  - no group keys in results or errors.

  The tests that called the old single-list form now supply complete sets, and the Amendment 1 classes are still covered.
- **Counts now:** metrics and screening have 28 tests (was 20). In the experiments environment **95 passed** (was 87); in the application's environment 74 passed and the CNN file skips (was 66). `ruff check` is clean.

**Second correction after independent review (2026-11-26, seed identity):**
- **The problem:** the first robustness correction (13b6452) counted exactly 5 seed rows but gave them no identity. One seed's result supplied five times would have satisfied it.
- **The fix:** seed rows are now (seed_index, D_FA, Δ_R), and the set of indices must be exactly {0, 1, 2, 3, 4}, the registered outer and final seed indices (§7).
  - Missing, duplicated or unexpected indices are rejected, as are four or six rows.
  - A non-integer or boolean index, a malformed row or a non-finite value raises an error.
  - Every seed must still have D_FA < 0 and Δ_R ≥ −0.05.
  - The leave-one-group-out completeness checks are unchanged.
  - Results and messages count missing, duplicated and unexpected entries without naming seed indices or group keys.
- **Tests:** new or revised synthetic tests cover:
  - all five distinct indices, in any order;
  - a failed result at each index;
  - a missing, duplicated (including five rows of one seed) or unexpected index;
  - four or six rows;
  - malformed and non-finite rows;
  - both screening modes rejecting a repeated or unexpected seed;
  - no seed index or group key in results or errors.
- **Counts now:** metrics and screening have 32 tests (was 28). In the experiments environment **99 passed** (was 95); in the application's environment 78 passed and the CNN file skips (was 74). `ruff check` is clean.

## 2026-11-26 addendum: Sonic Character, Phase A5.2 Step 1 (synthetic implementation)

Authorised by the owner for **generated data only**, against the readiness proposal (revision 4.4) as a provisional baseline. Pending decisions are parameters, not settled choices. No owner recording, annotation, history, manifest, validation or final-test data was opened. Nothing was trained on project data, and no real timing was measured.

| Increment | What was built | Tests | Evidence (synthetic) |
|---|---|---|---|
| S1.1 seeds | `seedplan.py`: all 1,682 production seed identities, one stream per consumer, collision checks, `:cN` remapping for gate and sensitivity seeds only, replay ledger. `cnn.SeededDropout` (its own generator); `build()` inside `fork_rng`; `experiments/presence/rng.py` | 23 | The canonical serialisation reproduces the provisional enumeration hash `0436bb71…`. No collision among 1,682 integers, 1,682 PyTorch seeds or 5,887 NumPy generator states (C has 7 consumers, B1 has 2). SC1–SC7 and RP1–RP5 pass. Two optimiser steps on generated data replay bit for bit. The global random state is untouched |
| S1.2 B0 rates | `b0check.py`: L1–L4, warnings and floating-point errors as failures, the silent fallback detected | 43 | Observed on the scan: up to 23,700 Hz not constructible; 23,720–32,000 Hz construct but fail when computing; 32,001 Hz and up (44.1, 48, 88.2, 96 kHz) supported. At 32,000 Hz the top FFT bin is 15,999.999999999998 Hz, just below the 16 kHz edge, so 32 kHz itself is **not** supported |
| S1.3 grid | `grid.py` (canonical segment, f0/f1, C rows [max(0, f0−75), f1)); `enginepass.py` (the engine's quiet rule; B1's features) | 29 | T-grid values confirmed. On one 44.1 kHz example, a naive 1.5 s pre-roll cut lands 573 samples off the canonical grid. B1's features equal the engine's bit for bit. C's slice against the full segment: max difference 7.7e-7 (float32), 1.4e-15 (float64). An excerpt starting at the recording start has 499 excerpt frames, not 500 |
| S1.4 integrity | `integrity.py`: R-A–R-F, public digest, kind-and-count comparison, artifact trees with a provenance allow-list | 31 | Every field mutation is detected. Reports carry no identifiers |
| S1.5 folds | `foldgen.py` and `tests/presence_synth.py` (a synthetic project built with the A5.1 APIs) | 10 | Folds for every paired group; the low-rate and validation excerpts are not read; two runs are byte-identical; every stop condition fires |
| S1.6 privacy | `gateway.py` (Tier 2 fields only, `source` never passed, seal-gated tokens, `re.fullmatch` parser, no production vocabulary); spy harness `tests/privacy_*` | 31 | PC1–PC4, PC6–PC9, DU1, DU4 and DU5 pass with the audit-hook and stat-probe layers. **PC5 and the strace layer fail: strace is not installed. Privacy acceptance is not met** |
| S1.7 timing | `timing.py`: exec-to-exit accounting, projection S + d1 + (E−1)·k·U + p·P, per-component factors, fixed-duration list scheduling, direct contention simulation, throughput diagnostics, stop rules with explicit thresholds | 11 | Arithmetic against independently computed fixtures; toy sleep jobs; no performance measured |
| S1.8 normalisation and augmentation | `normalise.py`, `augment.py` (M1, M0; M2 refused) | 13 | M1 keeps the primary's power; the partner-to-primary ratio is −6 to 0 dB; masking comes after standardisation |

**Totals (data-safe suites `tests/test_presence_model_*.py` only):**
- experiments environment: 288 passed, 2 failed (PC5 and the strace layer);
- application environment: 258 passed, 3 skipped (PyTorch-only files), the same 2 failed.
- `ruff check` is clean.

**Not covered yet:**
- the MUSDB matcher (needed before step 2);
- DU4 and DU5 over Stage 3 artifacts (they need Stage 2 code; extended at RG1b);
- every empirical timing value (step 5).

### RG1 corrections (2026-11-26, generated data only)

The RG1 review (conditional pass) found two defects in the spy test and gaps in S1.5 and S1.7. They are corrected below, still on generated data only. strace is still not installed.

**Spy test, defect A (recorder attribution).**
- Each recorder writes its own evidence: the strace trace, the audit-hook log and the stat-probe log. Every detection names its recorder, its origin (the main process or a child) and its kind (open, stat, list, chdir, launch, exec and so on).
- `require` checks one recorder at a time. Another recorder's detection never counts, and a non-empty trace counts only if the required access was classified in it.
- PC3–PC6 are split by recorder, following revision 4.4:

  | Control | Python-recorder part | strace part |
  |---|---|---|
  | PC3 | audit hook: the open | strace: the open |
  | PC4 | stat probes: the stat and the existence check; audit hook: the listing | strace: all three |
  | PC5 | (none) | strace: the open inside libsndfile |
  | PC6 | audit hook: the launch | strace: the child's own open |

- The layer-1 test now requires strace to classify the permitted run's reads, with nothing left unresolved.
- PC4 follows revision 4's text literally: a relative stat and listing from the project root.

**Spy test, defect B (relative paths).**
- Relative paths are resolved against the calling process's directory at the time of the call:
  - for the Python recorders, the directory is recorded with each event;
  - in the trace, it is followed through `chdir`, `fchdir`, threads (which share it) and new processes (which copy it), and kernel-resolved descriptor paths are preferred.
- Paths are compared after `realpath`.
- A record that cannot be resolved is counted as `unresolved`, which is a violation. Examples: a descriptor, a `dir_fd`, a process whose directory is unknown, a truncated string, a shell command, an unreadable line.
- A listing of a project folder is also a violation, because it reveals Tier 3 names.
- The recorded process runs with `python -B`. Before this change, writing the bytecode cache after a source edit (which reopens files by descriptor) produced intermittent unresolved records.

**Guard.** `test_presence_model_guard.py` reads every A5.2 test source on every run, and checks that none refers to the private evidence folder.

**Negative controls.** All of these pass without strace.
1. A Python-only detection does not satisfy a strace requirement. On the PC5 target, Python sees only soundfile's stat, never libsndfile's open.
2. A non-empty, classified trace that lacks the injected access does not satisfy the requirement.
3. and 4. Ten relative-path cases are detected by the Python recorders: plain, dot segments, a double separator, a parent segment, a Tier 3 folder after changing into it, a stat, a listing, a glob, a child process, and a symbolic link outside the project. Changing into a Tier 3 folder is itself detected, and a glob pattern spanning project folders fails closed.
5. Unresolvable records are counted, and they are violations.
6. Two tests fail on purpose, on worlds that carry canaries. No canary appears in pytest's failure output or in the reports.

Three further tests run the strace reader on hand-written lines in strace's format: directory changes, threads, children, kernel paths, interrupted calls, escapes and unresolvable cases. They test the reader's logic only. The real format is confirmed only when strace runs.

**S1.5.**
- The fold index carries `public.development_manifest_sha256` and `chain.public_digest_sha256`, and the public digest is written as provenance.
- R-E carries the seed format and the enumeration fingerprint (`0436bb71…`). Fold generation stops if a caller supplies different seed fields.
- Per-excerpt records must have exactly the listed fields: `record` refuses others, and `compare` reports them.
- A declared replay can be claimed only once.
- DU4 and DU5 now allow exactly revision 4.3's seven provenance changes, and recompute every changed value from the files. DU1 also checks that public provenance is byte-identical.

**S1.7.**
- The synthetic-versus-real cross-check stop has a pending tolerance and no default.
- `contention_factors` works from probe records: r = mean time under a configuration ÷ mean solo time, and c = the largest r. T2's τ₄/τ₁ sets a floor for c_T(outer).
- CPU time per job (user + system, through `wait4`) is recorded for information only.
- Observed on the toy job with the default thread settings: 2.6 s of CPU time in 0.55 s elapsed, from idle numeric-library threads spinning. This is why no formula uses CPU time; the test pins the job to one thread.
- All synthetic. No timing was measured.

**Totals (data-safe suites):**
- experiments environment: 323 passed, 15 failed;
- application environment: 293 passed, 3 skipped, 15 failed.

The 15 failures are the strace-recorder tests: the strace parts of PC3, PC4, PC5 and PC6, the layer-1 test, and the 10 relative-path strace cases. `ruff check` is clean.

**RG1 privacy acceptance at that point: not met**, because strace was not installed. It was met later the same day; see the strace validation below.

### RG1 strace validation (2026-11-26): PASS

The owner installed strace (package `strace 7.3-1`; `strace -- version 7.3`). Every check below ran on generated projects with **real strace recordings on this machine**. No owner data was used.

**First run.**
- All 65 privacy tests passed in the application environment.
- In the experiments environment, 4 permitted-run tests failed closed: both PC1 tests, the strace permitted-run test, and the PC8 report stage.
- The cause was a reader defect. `readlink` returns a link's target in an output buffer, which strace prints truncated at 32 characters, and the reader treated every quoted argument as an accessed path. The truncated buffers became unresolved accesses: false alarms, with nothing missed. `getcwd`'s returned directory was misread the same way.
- The application environment was not affected, because its interpreter's links are short.

**Correction (`30dc7af`).**
- The reader reads only the input path positions of `readlink`, `readlinkat`, `getcwd` (none), the extended-attribute calls, `symlink` and `symlinkat`. Every other call is read as before.
- A regression test covers truncated `readlink` output and `getcwd` results, which are not accesses. It also checks that input paths of the same calls are still classified, and that a truncated input path still fails closed.
- Run through the previous reader, the test's lines leave 4 records unresolved and give 4 Tier 3 hits, where the test requires 1 and 1.
- Before the correction was applied, it was checked against every recorded run: no Tier 3 detection by strace was lost except the spurious `getcwd` one.
- No assertion was weakened.

**Final results (strace 7.3, data-safe suites, visible terminal):**

| Environment | Passed | Failed | Skipped |
|---|---|---|---|
| Experiments (Python 3.12, torch) | 339 | 0 | 0 |
| Application (Python 3.14) | 309 | 0 | 3 (environmental: the PyTorch-only files) |

`ruff check` is clean.

In both environments:
- **Permitted runs:** all 5 left no strace record unresolved and produced no violation.
- **Injected accesses:** all 21 runs still show their violations.
- **Negative controls:** all 5 pass.

**What strace itself recorded:**

| Control | Synthetic access | strace's own record | Python recorders |
|---|---|---|---|
| PC3 | Opening three Tier 3 files | `openat` by the main process, once per file | audit hook: open |
| PC4 | From the project root: a relative stat of a final-test file, a listing of the final-test folder, an existence check of validation audio | `newfstatat`; `openat` plus `getdents64`; `newfstatat` | stat probes: the stat and the existence check; audit hook: the listing |
| PC5 | Reading validation audio's details through the audio library | the `openat` inside the library itself | a stat only: Python cannot see the open |
| PC6 | A child process opening validation annotations | `openat` by the child process | audit hook: the launch |

- The 10 relative-path cases are detected in the trace, the child's through the directory it inherited.
- strace 7.3 labels every line with its process ID, and attaches the working directory to every relative lookup.
- Interrupted calls in the child-process runs were joined correctly.
- Raw traces and canary values stayed in local temporary folders.

**RG1 recommendation: PASS. RG1 privacy acceptance: met.**

**Still outside Step 1:**
- the MUSDB matcher (Step 1b, before step 2);
- DU4 and DU5 over Stage 3 artifacts (RG1b);
- every empirical timing value (step 5).

P1 is still required before any real-data step.

## 2026-11-26 addendum: Sonic Character, Phase A5.2 Step 1b (synthetic MUSDB18-HQ matcher)

Authorised by the owner for **generated data only**. Nothing real was read: no corpus manifest, registry, alias or title file, recording or annotation. The matcher was not run on project data, and no real fold file or hash exists.

**What was built.**
- `presence_model/musdb.py` is the overlap matcher of protocol §18 (Stage 1). Its rules are readiness proposal revision 4, §7 (unchanged to revision 4.4).
- It works on in-memory inputs and reads no file. It reuses `presence/identity.py` (`credited`, `normalise`, `TITLE_FEATURE`) and the registry's partitions.
- **Match types:**
  - exact credit, with aliases and exactly one step of link expansion;
  - checksum;
  - title confirmation and permissive-only, both reported only.
- **Exposure:** G_T and G_V hold paired groups only, from exact or checksum matches.
- **Test-split stop:** an exact or checksum match from a MUSDB-test record to any development or A5.1-validation group stops the check, before any result exists.
- **Public report:** every split × partition × match-type cell, holding distinct groups and distinct records (zeros included), plus:
  - record and unparsed counts per split;
  - the sizes of G_T and G_V and their overlap;
  - the commit;
  - a digest of the four input fingerprints;
  - the limitations.
- **Fail closed:** malformed, missing, duplicated or conflicting inputs stop the check, with counts-only messages.
- **Private part:** the match list, G_T and G_V never print, not even in a pytest failure diff.
- **Spy-test harness:** the step-2 inputs have their own categories (`ALLOWED_STEP2`), and none of them is allowed at step 3.

**Tests.** 63 matcher tests and 14 spy tests.

| Criterion | Covered by |
|---|---|
| 1. Parsing | the first spaced hyphen only; hyphens inside names; extra separators |
| 2. Unparsed and malformed records | unparsed counted and still matched; 10 malformed shapes refused; other datasets ignored |
| 3–6. Matching | exact credit, including featured artists, never generic names; aliases; exactly one link step, both directions, nothing transitive; identical checksums only |
| 7–8. Report-only types | title confirmation and permissive-only reported, never exposing; a checksum match with a matching title gets no title confirmation |
| 9–10. Exposure and multiplicity | G_T and G_V, paired groups only; groups across several records and match types counted once per cell |
| 11. Partitions | reserved, final-test and excluded entries never assessed; a misspelt partition refused |
| 12. Test-split stop | credit or checksum matches to paired, low-rate and validation groups stop; report-only matches do not |
| 13. Count dimensions | every cell present |
| 14. Privacy | no canary in public output, error messages or failure diffs |
| 15. Determinism | shuffled records, entries, links and categories give the same result |
| 16. Input faults | 20 missing, malformed, duplicated or conflicting cases, and malformed report inputs |

**The spy test, with strace 7.3:** the synthetic check reads exactly the corpus manifest, the registry and the two identity files. strace and the audit hook each record all four. An extra open of a Tier 3 identity file, the development manifest or the validation history is detected by both recorders. The step-2 inputs stay violations at step 3, and the public report holds no canary.

**Independent review:**
- Fixed:
  - an unknown partition was skipped rather than refused (it could hide a test-split match);
  - empty or unnormalised alias names were accepted (an empty alias could create an exposure);
  - one test claim was incomplete;
  - a small widening of the step-3 directory list (the corpus folder now has a step-2-only category);
  - the private part could appear in pytest's dataclass diffs.
- Reported, not resolved: see the open points below.

**Results (strace 7.3, data-safe suites, visible terminal):**

| Environment | Passed | Failed | Skipped |
|---|---|---|---|
| Experiments | 416 | 0 | 0 |
| Application | 386 | 0 | 3 (environmental: the PyTorch-only files) |

That is 77 more than the RG1 baseline, with no regression. `ruff check` is clean.

**Open points for the owner before step 2:**
1. **Paired versus low-rate.** The registered count table splits development groups into paired and low-rate. That split comes from the excerpts' sample rates in the development manifest, which is not among step 2's registered inputs, and it is not among the fingerprinted inputs either. The matcher takes it as an argument and resolves nothing.
2. **Literal readings, stated in the module:**
   - " and " and " with " are matched in lower case only, so the permissive tier misses "And" and "With" (report-only);
   - an empty title key never confirms;
   - an id such as " - Title" is parsed, with an empty artist;
   - the corpus's `unassigned` split is refused rather than classed;
   - each cell counts both groups and records.
3. **Consequences of the registered rules themselves:**
   - `credited` reads "(With …)" in a title as a featured credit, which can over-match and exclude;
   - a link whose end is an alias spelling never expands.
4. **For step 2's loader:**
   - `identity.read_aliases` returns nothing when the alias file is missing;
   - two spellings that normalise to the same key resolve silently.

**Owner review (2026-11-26): Step 1b accepted as delivered**, within its authorised synthetic scope and acceptance criteria. Every open point above stands, unresolved:
- **The paired versus low-rate classification stays open.** No data source for it has been chosen, and it has not been computed from any input.
- The literal readings and rule consequences in points 2 and 3 remain as documented.
- The step-2 loader notes in point 4 remain for step 2.

**Acceptance authorises nothing further.** Step 2 (the real MUSDB check), step 3 and P1 are not started, and each still needs its own authorisation. No real MUSDB or corpus data has been read.

## 2026-11-27 addendum: Sonic Character, Phase A5.2 Step 2a (synthetic step 2 and 3 implementation)

Authorised by the owner after P1 published Amendment 3 (`4848298`), for **generated data only**. Nothing real was read: no corpus manifest, registry, alias or title file, recording, annotation, history or weights, and nothing under the private evidence folder. Step 2 and step 3 have not run on project data; each still needs its own authorisation, after review RG1c.

**What was built** (`gmv.character.research.presence_model`, Amendment 3, A3.4-A3.10):

| Part | Module | What it does |
|---|---|---|
| Strict readers | `readers.py` | The corpus manifest through GMV's strict corpus schema (MUSDB fields only), the split registry through its fingerprint check, the mandatory `aliases.json` with its stops, titles of assessed recordings only, and the new credit-readings and review-decisions files. Repeated JSON keys are refused; messages carry counts only |
| Matcher | `musdb.py` | Ambiguous credits ("and"/"with" in any capitalisation, brackets, non-Latin letters) need a registered reading or step 2 stops; records without " - " or with an empty or generic artist stop it too (MUSDB records only); credit sources are kept for the review; the paired/low-rate split is deferred to step 3 (option C); test-split rejections apply before the test-split check; a file match can never be rejected |
| Review and wording | `review.py` | The private review file the owner opens; a counts-only summary; one neutral public sentence for every step 2 stop and one for every re-run |
| Commitments | `commitment.py` | Hiding commitments (a fresh 32-byte secret per run) in place of plain public hashes of private inputs |
| Step 3 | `step3.py`, `step3run.py` | The split from B0's rate check, the owner's train and validation rejections, G_T and G_V, the end codes, sensitivity-rerun feasibility over every possible removed set, sensitivity folds as commitments, and the public count report built only through the disclosure check |
| Disclosure check (counts) | `disclosure_counts.py` | Exact over every possible world (517,440 at the registered sizes): item D exempt; the sizes of G_T and G_V and their overlap released only if every value they could show keeps every group's and every pair's match status uncertain; decisions read public figures only |
| Sandbox and trace | `sandbox.py`, `trace.py`, `sandbox_checks.py` | The bubblewrap configuration (the proposed configuration for synthetic testing, not a proven security boundary), the traced launcher, and the live-trace reader (set-up kept apart, deny by default with a recorded baseline of interpreter start-up probes, no new programs, child processes, network, changes of directory or io_uring) |
| Runners and lock | `step2run.py`, `runlock.py` | The step 2 runner (listing and matching phases, sandboxed and traced, accepted only on a complete trace with no violation and exactly the expected files) and the one exclusive experiment lock |

**Tests** (generated data only; the sandbox and runner tests start bubblewrap 0.13.1 under strace 7.3 for real on this machine):

| Area | File | Tests |
|---|---|---|
| Strict readers | `test_presence_model_readers.py` | 45 |
| Matcher (Step 1b tests updated where Amendment 3 changed a rule) | `test_presence_model_musdb.py`, `test_presence_model_musdb_v2.py` | 64 + 61 |
| Review file and commitments | `test_presence_model_review.py` | 12 |
| Step 3 | `test_presence_model_step3.py`, `test_presence_model_step3run.py` | 17 + 8 |
| Disclosure check (counts) | `test_presence_model_disclosure_counts.py` | 7 |
| Sandbox and trace reader | `test_presence_model_sandbox.py`, `test_presence_model_trace.py` | 11 + 11 |
| Step 2 runner | `test_presence_model_step2run.py` | 11 |
| Lock | `test_presence_model_runlock.py` | 8 |

What the sandbox tests showed on this machine: a full walk finds exactly the mounted paths; writes fail everywhere except `/out`; loopback, outside, host-socket and abstract-socket connections fail and are flagged; `/proc` shows only the sandbox; `/bin/sh` cannot start; a clean run has exactly three handles (the trace's is not inherited) and is accepted; a planted handle stops the step; a forced crash stores no dump; io_uring is flagged; killing strace or the launcher stops the step. The step 2 runner's sandboxed outputs equal a reference run outside the sandbox, apart from the fresh commitment.

**Suites at `87ff50a`** (visible terminal run, data-safe files only): experiments environment (Python 3.12, torch CPU) **586 passed, 22 skipped** (the sandbox and runner tests, which need the application environment's packages); application environment (Python 3.14, no torch) **578 passed, 3 skipped** (the PyTorch-only files). `ruff check` is clean. An earlier run of the application suite showed 4 errors in the fold-generation tests' shared fixture ("an audio file differs from its manifest entry"); they did not reproduce alone, after the tests that precede them, or in two later full runs of the application suite, and no test shares those files. Recorded as an unexplained intermittent failure, seen once.

**Suites after the runner change for Amendment 3A** (the commit after `3daebef`; visible terminal run, data-safe files only): experiments environment **586 passed, 22 skipped**; application environment **578 passed, 3 skipped**; `ruff check` is clean. The intermittent fixture error did not recur.

**Findings for the owner** (not changed silently):
1. **No sensitivity rerun could ever run under Amendment 3** (A3.7 rule 1): the exact count check shows the size of G_T can be published only when the item D MUSDB-train figure is 0-2 or 24-26 (and likewise G_V), so the trigger never reaches a feasible rerun. The owner chose option B; the proposed **Amendment 3A** (reruns decided from the item D figure, results private) was drafted; the owner approved it and it is published (`3daebef`). Step 2a implements its rule (`feasibility_from_item_d`).
2. **The approved public re-run sentence leaks** (test C4-T): a published report without it tells the reader that no assessed group had a MUSDB-test candidate before the review. Publishing only the final outcome (the report, or a single stop) keeps every before-review status uncertain. Resolved by Amendment 3A (`3daebef`): only step 2's final outcome is public, and the runner now gives no public wording for intermediate stops or re-runs.

**Not built or not covered in Step 2a:**
- Step 3 is not sandboxed (B0's rate check needs SciPy and the engine code, outside the sandbox's seven packages), and a live-trace wrapper for step 3's host paths is not built; fold generation's own access was spy-tested in Step 1.
- The exact duration check: step 3's voiced and no-voice totals, the breakdowns by partition and match type, and the matched-record counts are withheld as one fixed unit (N1.7 item 7); the full checker for durations and metric tables comes in step 4.
- io_uring setup succeeds inside the sandbox; the trace reader rejecting it is the only safeguard.
- The exact host paths of protected inputs are not verified (separately authorised preparation).

## 2026-11-28 addendum: Sonic Character, Phase A5.2 Step 4 (synthetic Stage 2 implementation)

Authorised by the owner after Amendment 3A was published (`3daebef`), for **generated data only** (A3.4: step 4 may open generated data only). Nothing real was read: no recording, annotation, manifest, registry or B0 weight file, and nothing under the private evidence folder; B0's normalisation constants and scores are generated stand-ins in every test. No model was trained on project data. Step 5 needs reviews RG1b and RG2 and its own authorisation.

**What was built** (Amendment 3, A3.9's step 4 list):

| Part | Module | What it does |
|---|---|---|
| B1 trainer | `presence_model/b1train.py` | 548 → 128 ReLU → 1 sigmoid (70,401 parameters); B0's initialisation scheme and hand-written Adam (bias correction, L2 1e-2 on the weight matrices); 1e-3, then 3e-4 after `E // 2` epochs; a seeded permutation of every frame with weight > 0 in batches of 1024 with batch loss J·Σw·ℓ; the float32 path of B0's script; inference by the engine's activity-model formula; the float64 gate loss; a deterministic model file |
| C inputs | `presence_model/cbatches.py` | Rows with up to 1.5 s of context at weight 0; B15 statistics before augmentation; B16 mixing partners (same-group no-voice, voiced never mixed); V7 draws from four streams; the order mix → gain → tilt → log → standardise → mask; R22 |
| C trainer | `experiments/presence/ctrain.py` | Batches of 8 whole excerpts in seeded order (end padding at weight 0); AdamW (B17) with a closed-form per-step cosine 1e-3 → 1e-5; seeded dropout; float64 probabilities from the logit; the gate loss; matched-start records; a deterministic model file |
| Ledger and matched start | `seedplan.RunLedger.claim_horizon`, `presence_model/matched.py` | The 16 declared `horizon` relations and their rules; the matched-start check for B1 (initial parameters, data order, parameters at the end of the shorter recipe's first half) and C (initial parameters, data order, augmentation draws, dropout masks, parameters after optimiser step 1) |
| Scoring | `presence_model/scoring.py` | Post-processing over the whole segment, per-group counts, exact J(t), the tie rule, J̄ over the policy's inner seeds, seed-index aggregates, effect sizes as their means, leave-one-group-out rows, t ± 0.05, t_final |
| Policy D in screening | `presence_model/screening.py` | The seed-index set follows the policy (5 under C, 3 under D); the result names the policy and says when its seed criterion is less demanding |
| AUC | `presence_model/auc.py` | B20: ROC-AUC with ties half, average precision over distinct scores, exact; quiet frames scored 0; eligibility at 2 s of each class; macro per seed index then over O |
| Gates | `presence_model/gates.py` | B18's matched complete-recipe rule (stage 1 and stage 2, S1-S5), the 16 h gate, the policy-flip safeguard, the caps and the registered timing stops |
| Recorders and launcher | `presence_model/recorders.py`, `presence_model/launcher.py` | Host activity as aggregate counters only; V6 memory (Pss + SwapPss over the job's tree); the disturbance and memory stops; jobs in their own group dying with the launcher; the experiment lock; the synthetic-data guard |
| N2 | `presence_model/seal.py` | The write-once seal; the expectation built from the design, Amendment 5's fold index (hash-checked) and sets, Amendment 6 and the declaration, recorded before the seal is opened; every listed refusal |
| N1 for results | `presence_model/exact.py`, `presence_model/disclosure.py`, `presence_model/results.py` | Exact rational verification (HiGHS proposes; weak-duality bounds, exact vertices, an exact relative-interior point from public information, forced equalities, emptiness proofs); the release engine (every outcome a figure could show, codes and zones, ratios, verdicts, fixed units, stops, the self-check, the extreme-value shortcut with its soundness argument for linear figures); the main results tables with representative protected quantities justified by exchangeability |
| End-to-end (B3) | `experiments/presence/synthetic_run.py` | A whole synthetic development run under policy D: 102 models, operating points, metrics, screening, AUC, B1's registered gate pairs with the ledger and matched-start check, the seal and the run record, written by output class |

**Tests** (generated data only):

| Area | File | Tests |
|---|---|---|
| B1 trainer | `test_presence_model_b1train.py` | 17 |
| C inputs | `test_presence_model_cbatches.py` | 10 |
| C trainer and C's matched start (PyTorch) | `test_presence_model_ctrain.py` | 20 |
| Horizon ledger and B1's matched start | `test_presence_model_matched.py` | 14 |
| Scoring and policy D | `test_presence_model_scoring.py` | 16 |
| AUC | `test_presence_model_auc.py` | 12 |
| Gates | `test_presence_model_gates.py` | 14 |
| Recorders and launcher (Linux `/proc`) | `test_presence_model_recorders.py` | 11 |
| Lock (the launcher added to the registered runners) | `test_presence_model_runlock.py` | 9 |
| N2 seal | `test_presence_model_seal.py` | 20 |
| Exact verification | `test_presence_model_exact.py` | 14 |
| Release engine (regression mechanisms, T1-T7, T9) | `test_presence_model_disclosure.py` | 27 |
| Main results layout (T8) | `test_presence_model_results.py` | 9 |
| End-to-end, B3 (PyTorch) | `test_presence_model_endtoend.py` | 6 |

Highlights: B1's training reproduces B0's script's update loop bit for bit, and its inference equals the engine's `ActivityModel.scores` bit for bit; AUC and average precision equal brute-force definitions with ties; every branch of the gate rule; a memory block shared by four processes is counted once, copy-on-write children are not double-counted and swapped-out memory still counts (the tests' own pages, paged out); the recorder opens no name, command-line or environment file; one test per refused N2 case; ties at exactly the width pass and an unprovable case is never passed; the shortcut agrees with checking every outcome (T7); decisions are identical across thread counts (T9); the end-to-end run is byte-identical when repeated, and changing only metadata and the run's name changes only allowed provenance fields.

**T8 measurement** (generated worlds, 15 groups with both classes and 4 with no-voice time only, three scenarios, a scratch script outside the repository):

| Policy | World | Registered reader | Public-record reader (comparison only) |
|---|---|---|---|
| D (27 figures) | typical | 0 published | 6 |
| D | near zero | 0 | 6 |
| D (29 with the ratio) | small effect | 0 | 10 |
| C (35 figures) | typical | 0 | 8 |
| C | near zero | 0 | 8 |
| C (37 with the ratio) | small effect | 0 | 12 |

Each world took 9-20 s with the public-record reader; the registered reader decides from the case count alone (about 4 × 10²⁵ cases for one protected group).

**Suites at `c1435e7`** (visible terminal run, data-safe files only): experiments environment (Python 3.12, torch CPU) **777 passed, 22 skipped** (the sandbox and runner tests, which need the application environment's packages); application environment (Python 3.14, no torch) **743 passed, 5 skipped** (the PyTorch-only files); `ruff check` is clean.

**Findings for the owner** (not changed silently):
1. **The approved disclosure rules publish almost none of the main results** (test T8, generated worlds at the registered size, 15 + 4 groups). With the registered reader (N1.1, which assumes the reader knows every group's directions, leave-one-group-out outcomes and eligibility, together with N1.7 item 8, which forbids taking them from their actual values), one protected group needs about 4 × 10²⁵ cases even after the exact exchangeability reduction, so every unit is withheld: 0 of 27 (policy D) and 0 of 35 (policy C) figures in every world. With a reader who knows only the public record (a comparison alternative, not an approved rule), 6-8 figures are published in typical and near-zero worlds and 10-12 in a small-effect world; B0ₜ's false-activity average, D_FA, the ratio and nearly every seed-index aggregate stay withheld, because an extreme value each could show (B0ₜ's average near 60 s/min; one seed aggregate at its largest possible value) would reveal every group's direction or pin the other seeds. Options for an Amendment 3B are in the PR description; the owner decides.
2. **An unexplained one-off difference in C's training.** In the first run of the end-to-end test, one of the 102 models of the third run (a C inner model) differed from the first two runs in 47 bytes after its single optimiser step; the identical rerun before it matched exactly. It did not recur in 30 isolated repeats of that model, 40 repeats with shifted memory allocations, 6 further whole runs at 4 threads (306 C and 306 B1 trainings) or the next full run of the test. Recorded as unexplained, seen once. If it happens in step 5, the replay check (S5) stops the run, as designed; the end-to-end test stays strict.

**Not built or not covered in Step 4:**
- The counts unit of the results tables (direction counts, the leave-one-group-out count and verdict, the screening class) is not modelled by the checker and is withheld whole; other tables (breakdowns and route rows, AUC tables, effects at t ± 0.05, the §10 sensitivity results, onset and offset errors, validation tables) have no figure model yet. Step 3's durations remain the Step 2a fixed unit.
- The shortcut's soundness rests on a written concavity argument for linear figures, checked against exhaustive outcomes on small worlds (T7); it still needs RG1b's review.
- C's gate rule is tested on generated values; the end-to-end run makes gate records for B1 only.
- The relative-interior handling is tested on small constructed systems only (N1.10).
- Timing measurements, the T1-T4 schedule and the traced rehearsals are step 5's; nothing was timed here.
