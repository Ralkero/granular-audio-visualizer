"""Causal, CPU-only mixed-audio features. Candidate events are not instrument identities."""

from collections import deque
from time import perf_counter
from uuid import uuid4

import numpy as np

from .models import AnalysisConfig, FeatureFrame, MusicalEvent


class WindowAssembler:
    def __init__(self, config):
        self.config = config
        self.pending = None
        self.expected = None
        self.format = None
        self.resets = 0
        self.reset_pending = False

    def push(self, block):
        fmt = (block.sample_rate, block.channels, block.segment)
        gap = (
            self.expected is None
            or block.discontinuity
            or fmt != self.format
            or abs(block.timestamp - self.expected) > 2 / block.sample_rate
        )
        audio = np.frombuffer(block.data, dtype="<f4").reshape(block.frames, block.channels)
        if not np.isfinite(audio).all():
            raise ValueError("Capture contains non-finite samples")
        if gap:
            self.pending = np.empty((0, block.channels), dtype=np.float32)
            self.start = block.timestamp
            self.mono_start = block.capture_monotonic
            self.resets += 1
            self.reset_pending = True
        self.format = fmt
        self.expected = block.timestamp + block.frames / block.sample_rate
        self.pending = np.concatenate((self.pending, audio), axis=0)
        size = round(block.sample_rate * self.config.window_ms / 1000)
        hop = round(block.sample_rate * self.config.hop_ms / 1000)
        while len(self.pending) >= size:
            yield (self.pending[:size], self.start, self.mono_start, self.reset_pending)
            self.reset_pending = False
            self.pending = self.pending[hop:]
            self.start += hop / block.sample_rate
            self.mono_start += hop / block.sample_rate


class Analyzer:
    def __init__(self, rate, channels, config: AnalysisConfig, session_id, source):
        self.rate, self.channels, self.config = rate, channels, config
        self.session_id, self.source = session_id, source
        self.size = round(rate * config.window_ms / 1000)
        self.hop = round(rate * config.hop_ms / 1000) / rate
        self.window = np.hanning(self.size).astype(np.float32)
        self.freq = np.fft.rfftfreq(self.size, 1 / rate)
        self.masks = [
            (self.freq >= a) & (self.freq < min(b, rate / 2 + 1))
            for a, b in zip(config.band_edges, config.band_edges[1:], strict=False)
        ]
        self.prev = None
        self.flux_history = deque(maxlen=max(8, round(12 / self.hop)))
        self.onsets = deque(maxlen=256)
        self.last_onset = -1e6
        self.last_tempo = -1e6
        self.tempo, self.tempo_confidence = None, 0.0
        self.energy = deque(maxlen=max(1, round(3 / self.hop)))
        self.reference_db = -30.0
        self.previous_db = -120.0
        self.silent = True
        self.silence_duration = 0.0
        self.last_intensity = 0.0
        self.last_change = -1e6
        self.previous_centroid = None
        self.previous_rms = 0.0
        self.previous_bands = np.zeros(7)
        weights = config.analysis_channel_weights
        if weights is not None and len(weights) != channels:
            raise ValueError("Analysis coefficient count must match source channels")
        # Fixed power coefficients, not amplitude mixing or dynamic quiet-channel selection.
        self.channel_weights = np.asarray(
            weights if weights is not None else [1 / min(channels, 2)] * channels
        )
        self.phase = None
        self.phase_period = None
        self.beat_locked = False
        self.last_prediction = -1e6

    def estimate_tempo(self, timestamp):
        if timestamp - self.last_tempo < 1 or len(self.flux_history) * self.hop < 4:
            return
        self.last_tempo = timestamp
        x = np.asarray(self.flux_history, dtype=np.float64)
        x -= x.mean()
        if np.dot(x, x) < 1e-6:
            self.tempo, self.tempo_confidence = None, 0.0
            return
        lo, hi = max(1, round(60 / 200 / self.hop)), min(len(x) // 2, round(60 / 60 / self.hop))
        lags = range(lo, hi + 1)
        scores = [
            float(
                np.dot(x[:-lag], x[lag:])
                / (np.linalg.norm(x[:-lag]) * np.linalg.norm(x[lag:]) + 1e-12)
            )
            for lag in lags
        ]
        index = int(np.argmax(scores))
        confidence = max(0.0, min(1.0, scores[index]))
        self.tempo_confidence = confidence
        self.tempo = 60 / ((lo + index) * self.hop) if confidence >= 0.2 else None

    def process(self, audio, start, capture_mono, block, queue_ms=0, live=True):
        began = perf_counter()
        timestamp = start + len(audio) / self.rate / 2
        end = start + len(audio) / self.rate
        rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
        analysis_rms = float(np.sqrt(np.dot(np.mean(audio**2, axis=0), self.channel_weights)))
        peak = float(np.max(np.abs(audio)))
        db = max(-120.0, 20 * np.log10(max(rms, 1e-6)))
        self.energy.append(rms * rms)
        short_db = max(-120.0, 10 * np.log10(max(float(np.mean(self.energy)), 1e-12)))
        spectrum = np.fft.rfft(audio * self.window[:, None], axis=0)
        power = np.mean(np.abs(spectrum) ** 2, axis=1)
        magnitude = np.sqrt(power)
        total = float(power.sum())
        weights = power / max(total, 1e-20)
        centroid = float(np.dot(self.freq, weights))
        bandwidth = float(np.sqrt(np.dot((self.freq - centroid) ** 2, weights)))
        rolloff = (
            float(self.freq[min(len(self.freq) - 1, np.searchsorted(np.cumsum(weights), 0.85))])
            if total
            else 0.0
        )
        # One-sided power, Hann power compensation -> mean-square units.
        energy_scale = 2 / (self.size * float(np.square(self.window).sum()))
        bands = tuple(float(power[m].sum() * energy_scale) for m in self.masks)
        contrast = tuple(
            float(
                10
                * np.log10(
                    (np.quantile(power[m], 0.9) + 1e-20) / (np.quantile(power[m], 0.1) + 1e-20)
                )
            )
            if np.any(m)
            else 0.0
            for m in self.masks
        )
        flux = (
            float(np.maximum(magnitude - self.prev, 0).sum() / max(magnitude.sum(), 1e-12))
            if self.prev is not None
            else 0.0
        )
        band_magnitude = np.array([magnitude[m].sum() for m in self.masks])
        band_flux = np.maximum(band_magnitude - self.previous_bands, 0) / np.maximum(
            band_magnitude, 1e-12
        )
        self.previous_bands = band_magnitude
        # Independent energy-rise measurement: sustain and spectral movement can have flux
        # without a new attack. Absolute/raw level semantics remain unchanged.
        attack_ratio = max(0.0, (analysis_rms - self.previous_rms) / max(analysis_rms, 1e-9))
        self.previous_rms = analysis_rms
        self.prev = magnitude
        recent = np.asarray(list(self.flux_history)[-max(1, round(1 / self.hop)) :])
        threshold = max(
            0.12,
            float(np.median(recent) + 3 * np.median(np.abs(recent - np.median(recent))))
            if len(recent)
            else 0.12,
        )
        is_quiet = db < self.config.silence_dbfs
        strength = min(1.0, flux) if not is_quiet else 0.0
        self.flux_history.append(strength)
        self.estimate_tempo(timestamp)
        # Slow reference plus absolute anchor prevents equalizing every passage.
        if not is_quiet:
            alpha = 1 - np.exp(-self.hop / 20)
            self.reference_db += alpha * (db - self.reference_db)
        absolute = np.clip((db + 65) / 65, 0, 1)
        relative = np.clip((db - self.reference_db + 18) / 36, 0, 1)
        intensity = float(0.65 * absolute + 0.35 * relative) if not is_quiet else 0.0
        events = []
        group = None

        def emit(kind, confidence, value=None, event_time=None):
            event = MusicalEvent(
                session_id=self.session_id,
                source=self.source,
                timestamp=timestamp if event_time is None else event_time,
                detected_monotonic=perf_counter(),
                event_id=uuid4().hex,
                event_type=kind,
                intensity=intensity if value is None else float(np.clip(value, 0, 1)),
                confidence=confidence,
                group_id=group,
                related_event_ids=tuple(e.event_id for e in events[:1]),
                method="mixed_audio_attack_v2",
                segment=block.segment,
                evidence_monotonic=capture_mono
                + (timestamp if event_time is None else event_time)
                - start
                if live and kind != "beat_prediction"
                else None,
                evidence_kind="predicted"
                if kind == "beat_prediction"
                else ("observed" if kind == "onset" else "candidate"),
                intended_use="context" if kind == "beat_prediction" else "immediate",
            )
            events.append(event)

        self.silence_duration = self.silence_duration + self.hop if is_quiet else 0.0
        if is_quiet and not self.silent and self.silence_duration >= 0.25:
            self.silent = True
            emit("silence_started", 0.95, 0)
        elif not is_quiet and self.silent:
            self.silent = False
            emit("audio_resumed", 0.95)
        if (
            not is_quiet
            and flux > threshold
            and attack_ratio > 0.08
            and timestamp - self.last_onset >= 0.045
        ):
            self.last_onset = timestamp
            self.onsets.append(timestamp)
            group = uuid4().hex
            emit("onset", min(0.9, 0.4 + 0.3 * flux / threshold), strength)
            band_sum = max(sum(bands), 1e-12)
            if sum(bands[:2]) / band_sum > 0.45:
                emit("bass_transient_candidate", 0.55, strength)
            if sum(bands[2:5]) / band_sum > 0.45 and centroid > 500:
                emit("snare_like_candidate", 0.35, strength)
            if sum(bands[4:]) / band_sum > 0.35:
                emit("high_frequency_percussion_candidate", 0.45, strength)
            if self.beat_locked and self.phase_period is not None:
                distance = abs(
                    (timestamp - self.phase + self.phase_period / 2) % self.phase_period
                    - self.phase_period / 2
                )
                if distance < 0.05:
                    emit("beat_observed", min(0.7, self.tempo_confidence), strength)
        # Circular concentration of recent attacks estimates beat phase separately from tempo.
        self.beat_locked = False
        if (
            self.config.experimental_beat_phase
            and self.tempo is not None
            and self.tempo_confidence >= 0.45
            and timestamp - self.last_onset < 1.25
        ):
            period = 60 / self.tempo
            recent_onsets = np.asarray([t for t in self.onsets if t > timestamp - 4])
            if len(recent_onsets) >= 6:
                vector = np.mean(np.exp(2j * np.pi * recent_onsets / period))
                if abs(vector) > 0.72:
                    self.phase = float(np.angle(vector) % (2 * np.pi) * period / (2 * np.pi))
                    self.phase_period = period
                    self.beat_locked = True
                    future = self.phase + (np.floor((end - self.phase) / period) + 1) * period
                    if future - self.last_prediction > period * 0.5:
                        emit(
                            "beat_prediction",
                            min(0.7, self.tempo_confidence * abs(vector)),
                            event_time=float(future),
                        )
                        self.last_prediction = future
        if not self.beat_locked:
            self.phase = self.phase_period = None
        if not is_quiet and db - self.previous_db > 6:
            emit("energy_rise", 0.65)
        if (
            self.previous_centroid is not None
            and not is_quiet
            and abs(centroid - self.previous_centroid) > 1500
            and strength > 0.2
        ):
            emit("spectral_change", 0.55)
        if abs(intensity - self.last_intensity) > 0.25 and timestamp - self.last_change > 1:
            emit("intensity_change", 0.6)
            self.last_intensity, self.last_change = intensity, timestamp
        self.previous_db, self.previous_centroid = db, centroid
        while self.onsets and self.onsets[0] < timestamp - 4:
            self.onsets.popleft()
        done = perf_counter()
        frame = FeatureFrame(
            session_id=self.session_id,
            source=self.source,
            timestamp=timestamp,
            detected_monotonic=done,
            window_end=end,
            sample_rate=self.rate,
            channels=self.channels,
            segment=block.segment,
            rms=rms,
            peak=peak,
            loudness_dbfs=float(db),
            short_term_loudness_dbfs=float(short_db),
            spectral_centroid_hz=centroid,
            spectral_bandwidth_hz=bandwidth,
            spectral_rolloff_hz=rolloff,
            spectral_contrast_db=contrast,
            zero_crossing_rate=float(np.mean(np.diff(np.signbit(audio), axis=0))),
            band_energy=bands,
            spectral_flux=flux,
            transient_strength=attack_ratio,
            onset_strength=strength,
            dynamic_intensity=intensity,
            silence=self.silent,
            tempo_bpm=self.tempo,
            tempo_confidence=self.tempo_confidence,
            rhythmic_density=len(self.onsets) / 4,
            queue_latency_ms=max(0, queue_ms),
            dsp_ms=(done - began) * 1000,
            capture_to_publish_ms=(done - capture_mono - len(audio) / self.rate / 2) * 1000
            if live
            else None,
            timestamp_quality=block.timestamp_quality,
            evidence_monotonic=capture_mono + len(audio) / self.rate / 2 if live else None,
            analysis_rms=analysis_rms,
            analysis_weights=tuple(self.channel_weights),
            band_flux=tuple(band_flux),
            attack_energy_ratio=attack_ratio,
            beat_locked=self.beat_locked,
            beat_phase=float((timestamp - self.phase) % self.phase_period / self.phase_period)
            if self.beat_locked
            else None,
        )
        return frame, events
