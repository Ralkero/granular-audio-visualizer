# Schema 1.0 and timeline contract

The authoritative definitions are `gmv.models`. JSON schemas can be generated with each model's `model_json_schema()` method. All model fields reject NaN/Infinity and unknown fields. Tuples become JSON arrays. Frozen objects can be shared by subscribers without mutation. Additive defaulted fields may be added during Phase One; incompatible changes must bump schema_version.

Every message includes type, schema_version, session_id, source, timestamp, detected_monotonic, published_monotonic and a strictly increasing bus sequence. Sequence order is publication order, not necessarily evidence-time order: delayed analysis can refer to earlier audio. Monotonic values have no UTC meaning; session UTC start/stop live in the manifest.

Feature frames report:
- timestamp: center of a completed causal window; window_end is its right edge. Audio time is preserved through queue drops, reconnection and file decoding. Discontinuities reset spectral/rhythm history.
- sample_rate/channels/segment: native input format and capture epoch.
- rms/peak: linear full-scale amplitude, aggregated across input channels; unused surround channels reduce aggregate RMS. No gain is applied.
- loudness_dbfs: window RMS dBFS; short_term_loudness_dbfs: moving three-second mean-square dBFS. Both floor at -120; neither is BS.1770 LUFS.
- spectral_centroid_hz and bandwidth: power-weighted frequency moments; rolloff: 85% cumulative power.
- band_energy: seven absolute mean-square band powers, not 0–1 normalized proportions. Default edges: 20, 60, 250, 500, 2000, 4000, 6000, 24000 Hz. Bands above Nyquist are empty. Spectra aggregate channel power so opposite-phase stereo does not cancel.
- spectral_contrast_db: within-band 90th/10th percentile power ratio in dB; zero_crossing_rate: average per-channel sign-change rate per sample.
- spectral_flux/onset_strength/transient_strength: positive spectral change relative to current magnitude; onset/transient fields currently share the same spectral-flux proxy, not independent estimators.
- dynamic_intensity: bounded 0–1 display signal blending 65% absolute dBFS mapping and 35% slowly referenced relative level; it never changes audio.
- tempo_bpm: null until at least four seconds of history and sufficient autocorrelation; bounded search 60–200 BPM. Octave ambiguity and speech false positives remain possible. confidence is a heuristic score, not calibrated probability.
- rhythmic_density: onsets in trailing four seconds divided by four (early sessions are biased low).
- silence: thresholded energy with 250 ms hold before a silence transition; playback resumes immediately when an active window arrives. Missing packets generate a waiting status rather than invented samples.
- queue_latency_ms: receipt-to-analysis-start delay for the latest block. dsp_ms: feature/event compute duration. capture_to_publish_ms: estimated window-center age near publication, including half-window accumulation and queue/compute; it is null for fast offline runs. timestamp_quality states whether origin used PortAudio ADC, callback arrival, WASAPI QPC, or exact file sample indexing.

PortAudio on the tested device provided unusable ADC timestamps; its sample clock is anchored to callback receipt minus a block duration. This estimates timing, but cannot establish physical speaker/DAC latency. Native process capture exposes WASAPI QPC positions in 100 ns units; these align with Windows Python perf_counter. Unknown timestamps fall back explicitly to pipe arrival estimates. Never interpret these measurements as calibrated sound-to-screen latency.

Musical events include event_id, event_type, intensity, confidence, method, analysis_tier, related_event_ids, group_id, optional interval_end and named measurement pairs. Simultaneous candidate labels share a group and link to the onset. This represents related observations, not proof of separate instruments. A beat_candidate is an onset during periodic evidence; it is not a phase-locked beat tracker. Advanced HPSS emits a two-second interval with harmonic/percussive fractions and processing time; its evidence timestamp stays at the original interval start.

Status messages are diagnostic lifecycle observations. Their timestamp uses elapsed session monotonic time, not musical evidence time; this is deliberately separate from file sample positions during accelerated offline analysis.

Replay preserves original evidence and detection timestamps, schedules messages by their original publication spacing, and assigns new bus publication times/sequences. Compare source/session identifiers and evidence timestamps across replay; don't subtract a replay publication timestamp from an original detection time to infer latency.
