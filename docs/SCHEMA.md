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

PortAudio on the tested device provided unusable ADC timestamps; its sample clock is anchored to callback receipt minus a block duration. This estimates timing, but cannot establish physical speaker/DAC latency. Native process capture exposes WASAPI QPC positions in 100 ns units; these align with Windows Python perf_counter. Unknown timestamps fall back explicitly to pipe arrival estimates. Never interpret these measurements as calibrated sound-to-screen latency.

Musical events include event_id, event_type, intensity, confidence, method, analysis_tier, related_event_ids, group_id, optional interval_end and named measurement pairs. `mixed_audio_attack_v3` onset measurements `band_low_hz` and `band_high_hz` are the outer edges of the bands that crossed the attack test on that hop. They are not an instrument classification. Simultaneous candidate labels share a group and link to the onset. This represents related observations, not proof of separate instruments. The frozen 1.0 baseline beat_candidate merely labeled onsets during periodic evidence. Current default emits no beat claims. Opt-in experimental phase tracking emits beat_observed only for phase-aligned onset evidence, and beat_prediction for future estimated beat times; predictions are not observed attacks. beat_locked/beat_phase expose the provisional phase (fraction of period), otherwise false/null. Low recall and octave ambiguity prevent default promotion. Advanced HPSS emits a two-second interval with harmonic/percussive fractions and processing time; its evidence timestamp stays at the original interval start.

Status messages are diagnostic lifecycle observations. Their timestamp uses elapsed session monotonic time, not musical evidence time; this is deliberately separate from file sample positions during accelerated offline analysis.

Replay preserves original evidence and detection timestamps, schedules messages by their original publication spacing, and assigns new bus publication times/sequences. Compare source/session identifiers and evidence timestamps across replay; don't subtract a replay publication timestamp from an original detection time to infer latency.


## Freshness and permitted use

Messages retain evidence `timestamp`, optional interval_end, source segment, detection time and publication time. `evidence_monotonic` maps original evidence into the live capture clock; null means unavailable (including accelerated file analysis). `age_at_publication_ms` is signed publication minus evidence time, never clamped. Future predictions have unknown evidence clocks rather than pretending observation. Clock drift warnings appear in frame timestamp_quality. The system fallback clock failed the long-run 50 ms goal; treat timing as uncertain until its backend is improved.

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
