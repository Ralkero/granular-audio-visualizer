"""A fixed set of audio measurements of one excerpt, chosen before any model was fitted.

Everything is measured from the audio alone: no name, title, path, identifier or split
ever enters. Most values are taken from the **centre** of the stereo image (what both
channels share, ``frontend.Analysis.centre``), where a lead voice almost always sits. The
centre also holds the snare, kick, bass and any centred lead instrument, so none of these
values isolates the voice; whether they still carry the delivery is what the experiment
asks.

Four families (``FEATURES`` lists each value, its family and what it measures):

* **envelope**: where the centre's energy sits: band shares from 80 Hz to 8 kHz, centroid,
  roll-off, and how much of the voice band is in the centre. High against low delivery.
* **harshness**: noise against harmonic structure in the voice band: spectral flatness,
  the harmonic share of the harmonic/percussive split, cepstral peak prominence (the voice
  sciences' measure of breathy and rough voice) and spectral contrast. Clean against harsh.
* **pitch**: the existing vocal analyzer's lead line (``gmv.instruments.vocals`` through
  ``lead.measure``): how much of the centre has one clear pitch, how clear, its range,
  motion and held notes, and its median height. Phase A2 found these unreliable in dense
  distorted mixes; they are kept to find out whether they still help.
* **temporal**: level spread, spectral flux, syllable-rate modulation (2 to 8 Hz) and
  roughness: envelope modulation at 30 to 150 Hz in the centre, as in growled, fried or
  distorted voice.

A value that cannot be measured is NaN (unknown), never zero. Imputation, scaling and
everything else learned happens later, on training data only (``models.Preprocess``).
"""

from math import gcd

import numpy as np
from scipy.signal import resample_poly

from .. import frontend, gmvpass, lead

FEATURE_VERSION = "delivery_features_v1"
BANDS_HZ = (80.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
VOICE_HZ = (300.0, 3400.0)
CONTRAST_HZ = ((300.0, 800.0), (800.0, 1600.0), (1600.0, 3400.0))
CPP_F0_HZ = (60.0, 880.0)
MIN_MUSIC_S = 2.0  # less music than this: nothing is measured
FINE_SIZE, FINE_HOP = 256, 32  # 11.6 ms windows every 1.45 ms (689 per second), for roughness
ROUGH_HZ, MODULATION_HZ = (30.0, 150.0), (4.0, 150.0)
SYLLABLE_HZ, SLOW_HZ = (2.0, 8.0), (0.5, 16.0)
EPS = 1e-12

FEATURES = (
    ("centre_share_80_250", "envelope", "log share of the centre's 80 Hz to 8 kHz energy between 80 and 250 Hz"),
    ("centre_share_250_500", "envelope", "the same, 250 to 500 Hz"),
    ("centre_share_500_1k", "envelope", "the same, 500 Hz to 1 kHz"),
    ("centre_share_1k_2k", "envelope", "the same, 1 to 2 kHz"),
    ("centre_share_2k_4k", "envelope", "the same, 2 to 4 kHz"),
    ("centre_share_4k_8k", "envelope", "the same, 4 to 8 kHz"),
    ("centre_centroid", "envelope", "mean spectral centroid of the centre, 250 Hz to 8 kHz, in octaves"),
    ("centre_centroid_spread", "envelope", "standard deviation over time of that centroid"),
    ("centre_rolloff", "envelope", "mean frequency under which 85% of the centre's 250 Hz to 8 kHz energy lies, in octaves"),
    ("voice_band_centre_share", "envelope", "share of the 300 Hz to 3.4 kHz energy that is in the centre"),
    ("voice_flatness", "harshness", "median spectral flatness of the centre, 300 Hz to 3.4 kHz (1: noise-like, 0: peaky)"),
    ("voice_flatness_p90", "harshness", "90th percentile of that flatness"),
    ("upper_flatness", "harshness", "median spectral flatness of the centre, 2 to 6 kHz"),
    ("voice_harmonic_share", "harshness", "mean harmonic share (harmonic/percussive split) of the centre's voice band"),
    ("voice_harmonic_share_spread", "harshness", "standard deviation over time of that share"),
    ("cepstral_peak", "harshness", "median cepstral peak prominence of the centre (F0 60 to 880 Hz), dB"),
    ("cepstral_peak_p90", "harshness", "90th percentile of that prominence"),
    ("voice_contrast", "harshness", "mean peak-to-valley contrast of the centre in three voice sub-bands, dB"),
    ("centre_line_level", "pitch", "existing vocal analyzer: level of the centre in the voice's range"),
    ("pitched_share", "pitch", "existing vocal analyzer: share of the music with one clear pitch (A2 to A5)"),
    ("pitch_clarity", "pitch", "existing vocal analyzer: how much of the centre's peaks the best harmonic series explains"),
    ("line_range_st", "pitch", "existing vocal analyzer: spread of the pitch, 10th to 90th percentile, semitones"),
    ("line_motion_st_s", "pitch", "existing vocal analyzer: pitch motion inside pitched stretches, semitones a second"),
    ("held_share", "pitch", "existing vocal analyzer: share of pitched time on held notes"),
    ("pitch_median_midi", "pitch", "existing vocal analyzer: median pitch while pitched, MIDI note number"),
    ("voice_level_spread", "temporal", "standard deviation over time of the centre's voice-band level, dB"),
    ("voice_flux", "temporal", "mean rise of the centre's voice-band log spectrum from frame to frame"),
    ("syllable_modulation", "temporal", "share of the voice-band envelope's 0.5 to 16 Hz modulation between 2 and 8 Hz"),
    ("roughness_low", "temporal", "share of the centre's 300 Hz to 1 kHz envelope modulation (4 to 150 Hz) between 30 and 150 Hz"),
    ("roughness_high", "temporal", "the same for the centre's 1 to 4 kHz envelope"),
)
NAMES = tuple(name for name, _family, _what in FEATURES)


def _bins(freq, low, high):
    return slice(int(np.searchsorted(freq, low)), int(np.searchsorted(freq, high)))


def _flatness(power):
    return np.exp(np.mean(np.log(power + EPS), axis=1)) / (np.mean(power, axis=1) + EPS)


def _share(spectrum, rate, band, total):
    """Share of an envelope's modulation power in ``band`` against ``total`` (Hz)."""
    values = spectrum - spectrum.mean()
    if not np.any(values):
        return np.nan
    power = np.abs(np.fft.rfft(values * np.hanning(len(values)))) ** 2
    freq = np.fft.rfftfreq(len(values), 1.0 / rate)
    whole = power[(freq >= total[0]) & (freq < total[1])].sum()
    if whole <= 0:
        return np.nan
    return float(power[(freq >= band[0]) & (freq < band[1])].sum() / whole)


def _cpp(magnitude, rate):
    """Cepstral peak prominence of each frame: the peak of the real cepstrum of the dB
    spectrum in the quefrency range of a voice's F0, above the regression line there."""
    db = 20.0 * np.log10(magnitude + EPS)
    cepstrum = np.fft.irfft(db, axis=1)
    size = cepstrum.shape[1]
    low, high = int(rate / CPP_F0_HZ[1]), min(int(rate / CPP_F0_HZ[0]), size // 2)
    part = cepstrum[:, low:high]
    quefrency = np.arange(low, high, dtype=float)
    slope, intercept = np.polyfit(quefrency, part.T, 1)
    peak = part.argmax(axis=1)
    return part[np.arange(len(part)), peak] - (slope * quefrency[peak] + intercept)


def _fine_centre(audio, rate):
    """Centre magnitude on a fine time grid (for envelope modulation up to 150 Hz)."""
    if rate != frontend.RATE:
        divisor = gcd(int(rate), frontend.RATE)
        audio = resample_poly(audio, frontend.RATE // divisor, int(rate) // divisor, axis=0)
    window = np.hanning(FINE_SIZE)

    def spectrum(signal):
        frames = np.lib.stride_tricks.sliding_window_view(signal, FINE_SIZE)[::FINE_HOP]
        return np.fft.rfft(frames * window, axis=1)

    left = spectrum(audio[:, 0])
    right = spectrum(audio[:, 1]) if audio.shape[1] > 1 else left
    total, side = np.abs(left + right) / 2, np.abs(left - right) / 2
    return np.maximum(total - side, 0.0), np.fft.rfftfreq(FINE_SIZE, 1 / frontend.RATE)


def measure(audio, rate):
    """The feature vector (``NAMES`` order) of one excerpt: float64, NaN where unknown."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    out = dict.fromkeys(NAMES, np.nan)
    analysis = frontend.Analysis(audio, rate)
    music = analysis.music
    if music.sum() * frontend.FRAME_S < MIN_MUSIC_S:
        return np.array([out[n] for n in NAMES], dtype=np.float64)
    freq = analysis.freq
    magnitude = analysis.centre[music]
    centre = magnitude**2
    power = analysis.power[music]
    mask = analysis.harmonic_mask[music]

    # Envelope.
    whole = _bins(freq, BANDS_HZ[0], BANDS_HZ[-1])
    total = centre[:, whole].sum(axis=1) + EPS
    names = ("80_250", "250_500", "500_1k", "1k_2k", "2k_4k", "4k_8k")
    for name, low, high in zip(names, BANDS_HZ[:-1], BANDS_HZ[1:], strict=True):
        share = np.log10(centre[:, _bins(freq, low, high)].sum(axis=1) + EPS) - np.log10(total)
        out[f"centre_share_{name}"] = float(share.mean())
    upper = _bins(freq, 250.0, 8000.0)
    octave = np.log2(freq[upper])
    weight = centre[:, upper] + EPS
    centroid = (weight * octave).sum(axis=1) / weight.sum(axis=1)
    out["centre_centroid"] = float(centroid.mean())
    out["centre_centroid_spread"] = float(centroid.std())
    cumulative = np.cumsum(weight, axis=1)
    rolloff = (cumulative < 0.85 * cumulative[:, -1:]).sum(axis=1)
    out["centre_rolloff"] = float(octave[np.minimum(rolloff, len(octave) - 1)].mean())
    voice = _bins(freq, *VOICE_HZ)
    voice_centre = centre[:, voice]
    out["voice_band_centre_share"] = float(
        np.mean(voice_centre.sum(axis=1) / (power[:, voice].sum(axis=1) + EPS))
    )

    # Harshness.
    flatness = _flatness(voice_centre)
    out["voice_flatness"] = float(np.median(flatness))
    out["voice_flatness_p90"] = float(np.percentile(flatness, 90))
    out["upper_flatness"] = float(np.median(_flatness(centre[:, _bins(freq, 2000.0, 6000.0)])))
    harmonic = (voice_centre * mask[:, voice]).sum(axis=1) / (voice_centre.sum(axis=1) + EPS)
    out["voice_harmonic_share"] = float(harmonic.mean())
    out["voice_harmonic_share_spread"] = float(harmonic.std())
    cpp = _cpp(magnitude, frontend.RATE)
    out["cepstral_peak"] = float(np.median(cpp))
    out["cepstral_peak_p90"] = float(np.percentile(cpp, 90))
    contrast = []
    for low, high in CONTRAST_HZ:
        part = centre[:, _bins(freq, low, high)] + EPS
        contrast.append(10.0 * np.log10(np.percentile(part, 95, axis=1) / np.percentile(part, 5, axis=1)))
    out["voice_contrast"] = float(np.mean(contrast))

    # Pitch: the existing vocal analyzer, as Phase A2 measured it.
    _messages, rows, frames = gmvpass.run(audio, rate)
    line = lead.Rows(rows, frames)
    measured = lead.measure(line, 0.0, len(audio) / rate)
    for name in ("centre_line_level", "pitched_share", "pitch_clarity", "line_range_st", "line_motion_st_s", "held_share"):
        value = measured[name][0]
        out[name] = np.nan if value is None else float(value)
    if len(line.pitch) == len(line.music):
        pitched = line.pitch[line.music]
        pitched = pitched[~np.isnan(pitched)]
        if len(pitched) * line.hop >= 1.0:
            out["pitch_median_midi"] = float(np.median(pitched))

    # Temporal.
    level = 10.0 * np.log10(voice_centre.sum(axis=1) + EPS)
    out["voice_level_spread"] = float(level.std())
    every = analysis.centre[:, voice] ** 2
    log_spectrum = np.log10(every + EPS)
    pairs = music[1:] & music[:-1]
    if pairs.any():
        rise = np.maximum(np.diff(log_spectrum, axis=0), 0.0).mean(axis=1)
        out["voice_flux"] = float(rise[pairs].mean())
    envelope = 10.0 * np.log10(every.sum(axis=1) + EPS)
    out["syllable_modulation"] = _share(envelope, 1.0 / frontend.FRAME_S, SYLLABLE_HZ, SLOW_HZ)
    fine, fine_freq = _fine_centre(audio, rate)
    fine_rate = frontend.RATE / FINE_HOP
    for name, (low, high) in (("roughness_low", (300.0, 1000.0)), ("roughness_high", (1000.0, 4000.0))):
        band = fine[:, _bins(fine_freq, low, high)].sum(axis=1)
        out[name] = _share(band, fine_rate, ROUGH_HZ, MODULATION_HZ)
    return np.array([out[n] for n in NAMES], dtype=np.float64)
