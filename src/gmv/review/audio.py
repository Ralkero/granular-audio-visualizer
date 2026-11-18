"""Decoding, source-quality checks and raw excerpt measurements.

Everything here reads audio and reports numbers. Nothing changes a source file, and no
gain, normalisation, filtering or resampling is applied to any audio that is written.
The numbers are measurements, not evidence that any instrument is present.
"""

import hashlib
import io
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import welch

from ..corpus import registry

FULL_SCALE = 0.999  # a sample at or above this magnitude is counted as at full scale
LOW_VOLUME_DBFS = -50.0


@contextmanager
def captured_stderr():
    """Collect what native decoders print on the process's standard error (mpg123 and
    libsndfile report damaged frames and odd tags there)."""
    result = {"text": ""}
    try:
        sys.stderr.flush()
        saved = os.dup(2)
    except (OSError, ValueError):
        yield result
        return
    with tempfile.TemporaryFile() as sink:
        os.dup2(sink.fileno(), 2)
        try:
            yield result
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            sink.seek(0)
            result["text"] = sink.read().decode("utf-8", "replace").strip()


def decode(path):
    """(audio float32 [frames, channels], rate, info). ``info`` describes the source as the
    decoder sees it and lists any decoder messages. The decoded samples are returned
    unchanged: no resampling, gain or clipping."""
    path = Path(path)
    if path.stat().st_size > registry.MAX_BYTES:
        raise ValueError("Audio file exceeds 2 GiB")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    with captured_stderr() as messages:
        with sf.SoundFile(io.BytesIO(data)) as handle:
            if handle.frames * handle.channels * 4 > registry.MAX_DECODED_BYTES:
                raise ValueError("Decoded audio would exceed 1 GiB")
            info = {
                "container": handle.format,
                "subtype": handle.subtype,
                "rate": handle.samplerate,
                "channels": handle.channels,
                "frames_declared": handle.frames,
            }
            audio = handle.read(dtype="float32", always_2d=True)
    info["frames_decoded"] = len(audio)
    info["sha256"] = digest
    lines = [line.strip() for line in messages["text"].splitlines() if line.strip()]
    # keep the message, drop the build path prefix ("[C:\...\id3.c:...] error: ...")
    info["decoder_messages"] = [line.split("] ", 1)[-1] for line in lines]
    return audio, handle.samplerate, info


def dbfs(value):
    return float(20 * np.log10(max(float(value), 1e-9)))


def _runs(mask):
    """Lengths of runs of True."""
    if not len(mask):
        return np.array([], int)
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return edges[1::2] - edges[::2]


def level_profile(audio, rate, window_s=0.5):
    """RMS level (dBFS) of the channel mean per ``window_s`` window."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    size = max(1, round(window_s * rate))
    count = len(mono) // size
    if count == 0:
        return np.array([dbfs(np.sqrt(np.mean(mono.astype(np.float64) ** 2)) if len(mono) else 0)])
    blocks = mono[: count * size].astype(np.float64).reshape(count, size)
    return 20 * np.log10(np.maximum(np.sqrt(np.mean(blocks**2, axis=1)), 1e-9))


def effective_bandwidth(mono, rate):
    """Highest frequency whose long-term level is within 50 dB of the 1-4 kHz level; a
    steep low value marks a band-limited (low bit rate or low sample rate) source."""
    if len(mono) < 4096:
        return None
    freq, psd = welch(mono, fs=rate, nperseg=4096)
    psd_db = 10 * np.log10(psd + 1e-20)
    band = (freq >= 1000) & (freq <= 4000)
    if not band.any():
        return None
    above = np.flatnonzero(psd_db > np.median(psd_db[band]) - 50)
    return float(freq[above[-1]]) if len(above) else None


def integrity(audio, rate):
    """Integrity measurements of decoded audio (a whole source or one excerpt)."""
    frames, channels = audio.shape
    magnitude = np.abs(audio)
    peak = float(magnitude.max()) if frames else 0.0
    full = magnitude >= FULL_SCALE
    clipped_runs = int(sum(int((_runs(full[:, c]) >= 3).sum()) for c in range(channels)))
    levels = level_profile(audio, rate)
    low = levels < LOW_VOLUME_DBFS
    window = 0.5
    runs = _runs(low)
    # low-volume stretches of at least 2 s that do not touch the start or end (fades)
    interior = 0
    if low.any():
        padded = np.concatenate([[False], low, [False]]).astype(np.int8)
        edges = np.flatnonzero(np.diff(padded))
        for begin, end in zip(edges[::2], edges[1::2], strict=True):
            if (end - begin) * window >= 2.0 and begin > 0 and end < len(low):
                interior += 1
    mono = audio.mean(axis=1)
    out = {
        "peak_dbfs": round(dbfs(peak), 2),
        "samples_beyond_full_scale": int((magnitude > 1.0).sum()),
        "samples_at_full_scale": int(full.sum()),
        "clipped_runs": clipped_runs,
        "rms_dbfs": round(
            dbfs(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if frames else -180.0, 2
        ),
        "dc_offset": round(float(np.abs(audio.mean(axis=0)).max()) if frames else 0.0, 5),
        "low_volume_share": round(float(low.mean()) if len(low) else 0.0, 4),
        "longest_low_volume_s": round(float(runs.max() * window) if len(runs) else 0.0, 1),
        "interior_low_volume_sections": interior,
    }
    if channels == 2 and frames:
        left, right = audio[:, 0].astype(np.float64), audio[:, 1].astype(np.float64)
        out["identical_channels"] = bool(np.max(np.abs(left - right)) < 1e-6)
        denom = np.sqrt(np.sum(left**2) * np.sum(right**2))
        out["channel_correlation"] = (
            round(float(np.sum(left * right) / denom), 4) if denom > 0 else None
        )
    else:
        out["identical_channels"] = None
        out["channel_correlation"] = None
    bandwidth = effective_bandwidth(mono, rate)
    out["effective_bandwidth_hz"] = round(bandwidth) if bandwidth else None
    return out


def source_issues(quality, inventory_row=None):
    """Plain-language integrity flags for a source recording."""
    issues = []
    row = inventory_row or {}
    codec = (row.get("codec") or "").lower()
    if codec and not codec.startswith("pcm") and codec not in ("flac", "alac", "wavpack"):
        issues.append("lossy_source")
    bitrate = row.get("bitrate_kbps")
    if bitrate and codec and not codec.startswith("pcm") and float(bitrate) < 128:
        issues.append("low_bitrate")
    if quality["rate"] < 32000:
        issues.append("sample_rate_below_model_rate")
    if quality.get("effective_bandwidth_hz") and quality["effective_bandwidth_hz"] < 15000:
        issues.append("band_limited")
    if quality["samples_beyond_full_scale"]:
        issues.append("decoded_beyond_full_scale")
    if quality["clipped_runs"]:
        issues.append("clipping_runs")
    if quality.get("identical_channels"):
        issues.append("identical_channels")
    if quality["dc_offset"] > 0.01:
        issues.append("dc_offset")
    if quality["interior_low_volume_sections"]:
        issues.append("interior_low_volume_sections")
    if (
        quality["frames_declared"]
        and abs(quality["frames_declared"] - quality["frames_decoded"]) > quality["rate"] * 0.05
    ):
        # informational: libsndfile's pre-decode length estimate for MP3 is often off; the
        # decoded length is compared with an independent measurement (the inventory) below
        issues.append("header_length_estimate_differs")
    listed = row.get("duration_s")
    if listed and abs(float(listed) - quality["frames_decoded"] / quality["rate"]) > 0.5:
        issues.append("decoded_length_differs_from_inventory")
    if quality["decoder_messages"]:
        issues.append("decoder_messages")
    if row.get("extension") == "mp3" and codec and codec != "mp3":
        issues.append(f"mp3_extension_but_{codec}")
    return issues


def excerpt_measures(audio, rate):
    """Raw measurements of one excerpt: level, spectral shape, onset density and a
    harmonic / percussive energy split. Descriptive only."""
    mono = audio.mean(axis=1).astype(np.float32)
    n_fft, hop = 2048, 512
    if len(mono) < n_fft:
        return {}
    frames = np.lib.stride_tricks.sliding_window_view(mono, n_fft)[::hop]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(n_fft).astype(np.float32), axis=1))
    freq = np.fft.rfftfreq(n_fft, 1 / rate)
    power = spectrum**2
    total = power.sum(axis=1) + 1e-12
    centroid = (power * freq).sum(axis=1) / total
    cumulative = np.cumsum(power, axis=1)
    rolloff = freq[np.argmax(cumulative >= 0.85 * cumulative[:, -1:], axis=1)]
    flatness = np.exp(np.mean(np.log(power + 1e-12), axis=1)) / (np.mean(power, axis=1) + 1e-12)
    loud = 10 * np.log10(total / n_fft) > -70  # ignore silent frames in spectral means
    pick = loud if loud.any() else np.ones_like(loud)
    # onset density: peaks of half-wave rectified log-spectral flux
    log_mag = np.log1p(100 * spectrum)
    flux = np.maximum(np.diff(log_mag, axis=0), 0).sum(axis=1)
    if len(flux) > 3:
        threshold = np.median(flux) + 1.5 * np.median(np.abs(flux - np.median(flux))) + 1e-9
        gap = max(1, round(0.05 * rate / hop))
        onsets, last = 0, -gap
        for i in range(1, len(flux) - 1):
            if (
                flux[i] > threshold
                and flux[i] >= flux[i - 1]
                and flux[i] >= flux[i + 1]
                and i - last >= gap
            ):
                onsets, last = onsets + 1, i
    else:
        onsets = 0
    harmonic = median_filter(spectrum, size=(17, 1))
    percussive = median_filter(spectrum, size=(1, 17))
    h_mask = harmonic**2 / (harmonic**2 + percussive**2 + 1e-12)
    h_energy = float((h_mask * power).sum())
    return {
        "spectral_centroid_hz": round(float(np.mean(centroid[pick]))),
        "spectral_rolloff85_hz": round(float(np.mean(rolloff[pick]))),
        "spectral_flatness": round(float(np.mean(flatness[pick])), 4),
        "onset_density_per_s": round(onsets / (len(mono) / rate), 2),
        "harmonic_energy_ratio": round(h_energy / float(power.sum() + 1e-12), 3),
    }
