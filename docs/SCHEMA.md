# Schema 1.1 and timeline contract

The authoritative definitions are `gmv.models`. JSON schemas can be generated with each model's `model_json_schema()` method. All model fields reject NaN/Infinity and unknown fields. Tuples become JSON arrays. Frozen objects can be shared by subscribers without mutation. Strict 1.0 readers cannot accept 1.1 additions. Use `gmv.contract.parse_message` for explicit version dispatch: 1.0 validates against frozen legacy models before migration, 1.1 validates against current models, other versions fail. Legacy event provenance/freshness stays unknown; do not invent timestamps.

Every message includes type, schema_version, session_id, source, timestamp, detected_monotonic, published_monotonic and a strictly increasing bus sequence. Sequence order is publication order, not necessarily evidence-time order: delayed analysis can refer to earlier audio. Monotonic values have no UTC meaning; session UTC start/stop live in the manifest.

Feature frames report:
- timestamp: center of a completed causal window; window_end is its right edge. Audio time is preserved through queue drops, reconnection and file decoding. Discontinuities reset spectral/rhythm history.
- sample_rate/channels/segment: native input format and capture epoch.
- rms/peak: linear full-scale amplitude, aggregated across input channels; unused surround channels reduce aggregate RMS. No gain is applied.
- loudness_dbfs: window RMS dBFS; short_term_loudness_dbfs: moving three-second mean-square dBFS. Both floor at -120; neither is BS.1770 LUFS.
- spectral_centroid_hz and bandwidth: power-weighted frequency moments; rolloff: 85% cumulative power.
- band_energy: seven absolute mean-square band powers, not 0–1 normalized proportions. Default edges: 20, 60, 250, 500, 2000, 4000, 6000, 24000 Hz. Bands above Nyquist are empty. Spectra aggregate channel power so opposite-phase stereo does not cancel.
- spectral_contrast_db: within-band 90th/10th percentile power ratio in dB; zero_crossing_rate: average per-channel sign-change rate per sample.
- spectral_flux/onset_strength: positive spectral change relative to current magnitude. transient_strength/attack_energy_ratio: independently measured positive analysis-RMS rise divided by current analysis RMS. band_flux: positive per-band magnitude changes divided by current per-band magnitudes.
- analysis_rms: square root of the sum of channel mean squares times fixed analysis_weights. Defaults: mono coefficient 1; every channel in a multichannel source coefficient 0.5. Optional explicit coefficients must be nonnegative, have positive sum, and match channel count. No amplitude cancellation or dynamic quiet-channel removal. Values may exceed 1 for many full-scale channels; raw RMS/band-energy semantics are unchanged. measurement_semantics identifies both conventions.
- dynamic_intensity: bounded 0–1 display signal blending 65% absolute dBFS mapping and 35% slowly referenced relative level; it never changes audio.
- tempo_bpm: null until at least four seconds of history and sufficient autocorrelation; bounded search 60–200 BPM. Octave ambiguity and speech false positives remain possible. confidence is a heuristic score, not calibrated probability.
- rhythmic_density: onsets in trailing four seconds divided by four (early sessions are biased low).
- silence: thresholded energy with 250 ms hold before a silence transition; playback resumes immediately when an active window arrives. Missing packets generate a waiting status rather than invented samples.
- queue_latency_ms: receipt-to-analysis-start delay for the latest block. dsp_ms: feature/event compute duration. capture_to_publish_ms: estimated window-center age near publication, including half-window accumulation and queue/compute; it is null for fast offline runs. timestamp_quality states whether origin used PortAudio ADC, callback arrival, WASAPI QPC, or exact file sample indexing.

PortAudio on the tested device provided unusable ADC timestamps; its sample clock is anchored to callback receipt minus a block duration. This estimates timing, but cannot establish physical speaker/DAC latency. Native process capture exposes WASAPI QPC positions in 100 ns units; these align with Windows Python perf_counter. Native system capture reports `wasapi_qpc_sample_clock`: a contiguous sample clock per run, anchored to and checked against QPC, offset by a measured per-device lead so evidence is not stamped ahead of delivery. Unknown timestamps fall back explicitly to pipe arrival estimates. Never interpret these measurements as calibrated sound-to-screen latency.

Musical events include event_id, event_type, intensity, confidence, method, analysis_tier, related_event_ids, group_id, optional interval_end and named measurement pairs. `mixed_audio_attack_v3` onset measurements `band_low_hz` and `band_high_hz` are the outer edges of the bands that crossed the attack test on that hop. They are not an instrument classification. Simultaneous candidate labels share a group and link to the onset. This represents related observations, not proof of separate instruments. The frozen 1.0 baseline beat_candidate merely labeled onsets during periodic evidence. Current default emits no beat claims. Opt-in experimental phase tracking emits beat_observed only for phase-aligned onset evidence, and beat_prediction for future estimated beat times; predictions are not observed attacks. beat_locked/beat_phase expose the provisional phase (fraction of period), otherwise false/null. Low recall and octave ambiguity prevent default promotion. Advanced HPSS emits a two-second interval with harmonic/percussive fractions and processing time; its evidence timestamp stays at the original interval start.

Status messages are diagnostic lifecycle observations. Their timestamp uses elapsed session monotonic time, not musical evidence time; this is deliberately separate from file sample positions during accelerated offline analysis.

Replay preserves original evidence and detection timestamps, schedules messages by their original publication spacing, and assigns new bus publication times/sequences. Compare source/session identifiers and evidence timestamps across replay; don't subtract a replay publication timestamp from an original detection time to infer latency.


## Freshness and permitted use

Messages retain evidence `timestamp`, optional interval_end, source segment, detection time and publication time. `evidence_monotonic` maps original evidence into the live capture clock; null means unavailable (including accelerated file analysis). `age_at_publication_ms` is signed publication minus evidence time, never clamped. Future predictions have unknown evidence clocks rather than pretending observation. Clock drift warnings appear in frame timestamp_quality. The PortAudio system fallback clock failed the long-run 50 ms goal because audio lost before its callback is invisible to it; the native system backend did not drift in the paired ten-minute run (see PHASE1_1_REPORT.md Milestone 7 for its limits).

Events declare `provenance` (mixed_signal_heuristic, harmonic_percussive_proxy, model_estimated_stem or legacy_unknown), `confidence_interpretation` (currently uncalibrated_heuristic_score), `evidence_kind` and `intended_use`. A 0.8 heuristic score is not an 80% probability.

| Evidence | Intended use | Live impact permitted? |
|---|---|---|
| Fresh mixed-audio onset/candidate, current session/segment | immediate | Only within the consumer age budget |
| Beat prediction | context | No; schedule explicitly as a prediction if a future renderer supports it |
| Two-second HPSS proxy | context | No; use slowly varying context |
| Offline model benchmark or legacy unknown event | retrospective | No |
| Replayed original event | replay inspection | No live impact; use a dedicated replay clock |

`eligible_for_live_impact(message, now=..., session_id=..., segment=..., max_age_ms=100)` fails closed for unknown/future evidence clocks, replay, wrong session/segment, advanced tier and non-immediate use. The 100 ms default is a rejection budget, not a measured latency guarantee. Consumers must also clear queued responses on capture discontinuity/status changes and apply stricter thresholds for their renderer. Correct historical timestamps cannot make late percussion drive a synchronized impact on arrival. Playback is never delayed to compensate.

Advanced work is generation-tagged. New source formats, segments, gaps and queue overflow invalidate in-flight results; shutdown discards pending work. The compute thread cannot publish old-generation results after invalidation. Queues remain bounded; skipped blocks and discarded results are separate diagnostics.

Replay supports committed 1.0 fixtures and 1.1 logs, preserves original evidence/detection times and relationships, schedules original publication spacing, sets replayed=true, stores original_published_monotonic and assigns new publication sequence/time. Replay age_at_publication_ms is null; original capture_to_publish_ms is historical. Never infer capture latency from a replay arrival.

## Related schemas

The JSON schema files in `docs/schemas/` are generated from the models (`python -m gmv.schema_export`; a test fails when they drift). Reference annotations for evaluation use a separate, versioned schema that is not part of the live message contract: see [CORPUS.md](CORPUS.md) and [ANNOTATION.md](ANNOTATION.md).

`beat_prediction` events (opt-in beat tracker, method `causal_beat_tracker_v1`) have a timestamp in the future of the audio analysed so far and a `tempo_bpm` measurement. `beat_observed` is a detected onset within 50 ms of a predicted beat. Frame fields `beat_locked` and `beat_phase` follow the same tracker.

Feature frames carry `instrument_scores` (label and uncalibrated score pairs), `instruments_active` (labels currently judged to be sounding) and `activity_method`. They are empty unless `activity_model_path` in the analysis configuration points to a weights file. The scores are not probabilities and not a separation.

## Display spectrum (Phase 2A)

Feature frames carry `spectrum_db`: band power in dB (same mean-square units as `band_energy`, floor -120, one decimal), low to high frequency, in `spectrum_bands` log-spaced bands (default 48, at most 96, 0 turns it off) between `spectrum_low_hz` (30) and `spectrum_high_hz` (18000, or Nyquist if lower). It is read off the same FFT as every other feature; no second transform runs. A band narrower than one FFT bin (the lowest few at the default window) takes the power interpolated at its centre, so neighbouring low bands are not independent measurements. The field is for display; detection does not use it.

These are additive optional fields inside schema 1.1, like the beat and activity fields before them. Sessions logged earlier have no `spectrum_db`; they still load and replay, and the visualizer falls back to the seven `band_energy` values. As with every 1.1 addition, a strict reader built from an older copy of the models rejects the new fields.

## Note candidates and guitar weights (Goal B)

Additive, optional fields inside schema 1.1:

- `AnalysisConfig.pitch_tracking` (default false) turns on note candidates; `AnalysisConfig.guitar_model_path` loads a second activity weights file.
- `FeatureFrame.pitch_candidates`: pairs of MIDI note number and uncalibrated score (0 to 1) for notes sounding at `pitch_time`; at most four, between E2 (40) and E6 (88). `pitch_method` names the tracker (`harmonic_salience_v1`) and is null when tracking is off. `pitch_time` is earlier than the frame timestamp because the pitch window is longer.
- `note_attack_candidate` events: `evidence_kind` candidate, timestamp back-dated to when the note began, measurements `midi` and `frequency_hz`.
- With two weights files, `instrument_scores` holds the labels of both and `activity_method` joins their names with `+`.

None of these identify an instrument by themselves: pitch candidates describe the mix, and only a detector's label in `instrument_scores` says an instrument is playing.

## Instrument activity

`InstrumentActivity` (`type` `instrument_activity`, [schema](schemas/InstrumentActivity.json)) is published by a fast instrument analyzer for every analysis frame while `AnalysisConfig.instrument_analyzers` names it.

- `timestamp` is the centre of the frame it describes; analyzers answer a frame or more late and back-date the message.
- `instrument`, `method` (guitar: `guitar_articulation_v1`), `basis` (guitar: `stereo` or `mono`).
- `level`, `flux`, `attack` (0 unless an attack is placed at this frame), `regularity`, `centre`, `spread`: 0 to 1, uncalibrated. `rate_hz` is null when the articulation does not repeat.
- It is instrument-oriented activity derived from the mixed audio, not an isolated instrument signal, and it does not say the instrument is present. Presence is `InstrumentEstimate`.
- Logged to `advanced_analysis/instrument_activity.jsonl` as well as `messages.jsonl`. Schema 1.0 has no such message.
- `AnalysisConfig.instrument_sensitivity`: pairs of analyzer name and sensitivity (0 to 1, 0.5 when not listed).
