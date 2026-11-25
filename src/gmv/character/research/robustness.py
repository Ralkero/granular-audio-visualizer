"""Does a measurement describe the music, or the way it was recorded and played back?

Each recording is analysed again after a change that leaves the music what it was: a
lower level, a fold-down to mono, a duller or a brighter tone, a band limit like a lossy
file's, saturation, half the sample rate. A measurement of musical behaviour should stay
where it was; a measurement of the sound's texture (the Phase A profile) is expected to
move, and how far it moves gives the scale to read the others against.

The changes are made to the decoded samples in memory. No file is written.
"""

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

from .evaluate import ALL, features
from .frontend import load
from .profile import measure_audio


def tilt(audio, rate, db_per_octave):
    """A tone control: this many dB an octave around 1 kHz, by FFT."""
    spectrum = np.fft.rfft(audio, axis=0)
    freq = np.maximum(np.fft.rfftfreq(len(audio), 1 / rate), 20.0)
    gain = 10.0 ** (db_per_octave * np.log2(freq / 1000.0) / 20.0)
    return np.fft.irfft(spectrum * gain[:, None], len(audio), axis=0).astype(np.float32)


def lowpass(audio, rate, hz):
    return sosfilt(butter(8, hz, "low", fs=rate, output="sos"), audio, axis=0).astype(np.float32)


def saturate(audio, drive_db=12.0):
    """Driven into a soft clipper and brought back to its old level."""
    drive = 10.0 ** (drive_db / 20.0)
    peak = max(float(np.abs(audio).max()), 1e-9)
    out = np.tanh(audio / peak * drive)
    return (out * np.sqrt(np.mean(audio**2) / max(float(np.mean(out**2)), 1e-20))).astype(np.float32)


CHANGES = {
    "10 dB quieter": lambda audio, rate: (audio * np.float32(10 ** -0.5), rate),
    "20 dB quieter": lambda audio, rate: (audio * np.float32(0.1), rate),
    "mono": lambda audio, rate: (np.repeat(audio.mean(axis=1, keepdims=True), 2, axis=1), rate),
    "duller (-3 dB/octave)": lambda audio, rate: (tilt(audio, rate, -3.0), rate),
    "brighter (+3 dB/octave)": lambda audio, rate: (tilt(audio, rate, 3.0), rate),
    "band-limited at 7 kHz": lambda audio, rate: (lowpass(audio, rate, 7000.0), rate),
    "saturated (12 dB drive)": lambda audio, rate: (saturate(audio), rate),
    "half the sample rate": lambda audio, rate: (resample_poly(audio, 1, 2, axis=0).astype(np.float32), rate // 2),
}


def one(job):
    """One recording under every change: {change: values in the order of ``ALL``}."""
    path, seconds = job
    try:
        audio, rate = load(path, seconds=seconds)
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        out = {"original": features(measure_audio(audio, rate)).tolist()}
        for name, change in CHANGES.items():
            out[name] = features(measure_audio(*change(audio, rate))).tolist()
        return out
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def summarise(found, scales):
    """Per change and measurement: the median and largest move, in calibration standard
    deviations, and how often a value that was known became unknown or the reverse."""
    spread = np.array([scales[name]["sd"] or np.nan for name in ALL], dtype=float)
    out = {}
    for change in CHANGES:
        moved, lost, gained = [], [], []
        for values in found:
            before = np.array([np.nan if v is None else v for v in values["original"]], dtype=float)
            after = np.array([np.nan if v is None else v for v in values[change]], dtype=float)
            moved.append(np.abs(after - before) / spread)
            lost.append(np.isfinite(before) & ~np.isfinite(after))
            gained.append(~np.isfinite(before) & np.isfinite(after))
        moved, lost, gained = np.array(moved), np.array(lost), np.array(gained)
        out[change] = {}
        for index, name in enumerate(ALL):
            column = moved[:, index]
            known = np.isfinite(column)
            out[change][name] = {
                "median": round(float(np.median(column[known])), 3) if known.any() else None,
                "largest": round(float(column[known].max()), 3) if known.any() else None,
                "became_unknown": int(lost[:, index].sum()),
                "became_known": int(gained[:, index].sum()),
            }
    return out


def report(summary, count):
    lines = [
        "# Sonic Character research: robustness",
        "",
        f"{count} recordings, each analysed again after a change that leaves the music as it "
        "was. Median move of each value, in standard deviations of that value across the "
        "calibration recordings (0.1 is negligible, 1 is the difference between two ordinary "
        "recordings). In brackets: how many recordings lost the value (became unknown).",
        "",
        "| Measurement | " + " | ".join(CHANGES) + " |",
        "|---|" + "---:|" * len(CHANGES),
    ]
    for name in ALL:
        cells = []
        for change in CHANGES:
            got = summary[change][name]
            text = "–" if got["median"] is None else f"{got['median']:.2f}"
            cells.append(text + (f" [{got['became_unknown']}]" if got["became_unknown"] else ""))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
