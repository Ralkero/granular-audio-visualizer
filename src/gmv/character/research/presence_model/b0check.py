"""Which sample rates B0 actually supports: the L1-L4 checks on generated audio (readiness
proposal, revision 4.4, section 4). A rate is *supported* only when every level passes at
that exact rate; nothing is inferred from construction alone.

* L1 constructs ``FrameFeatures`` (20 bands, every band width > 0, 137 features).
* L2 runs ``FrameFeatures.step`` and ``Context.step`` on engine-style spectra of silence,
  noise and a tone, mono and stereo (identical and anti-phase channels): (137,) and (548,)
  finite float64 values; width features (indices 60-79) exactly 0 for mono and identical
  channels, at least 0.99 for anti-phase noise.
* L3 runs ``ActivityBank`` with generated weights: one finite score in [0, 1] and one bool;
  exactly 0 when quiet.
* L4 runs the engine pass (``gmvpass.run``) on 3 s of generated audio with the generated
  weights: one frame per ``frame_count``; every frame carries a finite ``vocals`` score (its
  absence is the silent fallback and fails); quiet frames score exactly 0.

Every level runs with warnings raised as errors and NumPy floating-point errors (divide,
overflow, invalid) raised. The fixtures are synthetic implementation choices, reviewed at
RG1: Gaussian noise with sigma 0.1; a 1 kHz tone at -20 dBFS peak (amplitude 0.1); random
weights of scale 0.01; the 3 s input is 1 s of noise, 0.5 s of digital silence, 1.5 s of tone.
"""

import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from . import frontend

SCAN_RATES = (8000, 16000, 22050, 23600, 23700, 23720, 23740, 23800, 24000, 31900, 31990, 31999,
              32000, 32001, 32050, 44100, 48000, 88200, 96000)
NOISE_SIGMA = 0.1
NOISE_SEED = 20261126
TONE_HZ = 1000.0
TONE_PEAK = 10 ** (-20 / 20)  # -20 dBFS peak
WEIGHT_SCALE = 0.01
WEIGHT_SEED = 7
L4_LAYOUT_S = (("noise", 1.0), ("silence", 0.5), ("tone", 1.5))
WIDTH = slice(60, 80)  # balance, rise, peak, then width: 20 bands each (activity.FrameFeatures.step)
FEATURES, CONTEXT_FEATURES = 137, 548


@contextmanager
def strict():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with np.errstate(divide="raise", over="raise", invalid="raise", under="ignore"):
            yield


def sizes(rate):
    return frontend.engine_window(rate), frontend.engine_hop(rate)


def signal(kind, samples, rate, seed=NOISE_SEED):
    if kind == "silence":
        return np.zeros(samples, dtype=np.float32)
    if kind == "noise":
        return (np.random.default_rng(seed).normal(0.0, NOISE_SIGMA, samples)).astype(np.float32)
    if kind == "tone":
        return (TONE_PEAK * np.sin(2 * np.pi * TONE_HZ * np.arange(samples) / rate)).astype(np.float32)
    raise ValueError("unknown fixture")


def channels(mono, layout):
    if layout == "mono":
        return mono[:, None]
    if layout == "identical":
        return np.stack((mono, mono), axis=1)
    if layout == "anti-phase":
        return np.stack((mono, -mono), axis=1)
    raise ValueError("unknown channel layout")


def spectra(audio, rate, frames=5):
    """Engine-style spectra: float32 windows, float32 Hann, ``rfft`` over samples."""
    size, hop = sizes(rate)
    window = np.hanning(size).astype(np.float32)
    return [np.fft.rfft(audio[i * hop : i * hop + size] * window[:, None], axis=0) for i in range(frames)]


def write_weights(path):
    from ....activity import FEATURE_VERSION

    rng = np.random.default_rng(WEIGHT_SEED)
    np.savez(path, feature_version=FEATURE_VERSION, name="a5.2-b0check", labels=np.array(["vocals"]),
             mean=np.zeros(CONTEXT_FEATURES), scale=np.ones(CONTEXT_FEATURES),
             w1=(WEIGHT_SCALE * rng.standard_normal((CONTEXT_FEATURES, 128))).astype(np.float32),
             b1=np.zeros(128, np.float32),
             w2=(WEIGHT_SCALE * rng.standard_normal((128, 1))).astype(np.float32),
             b2=np.zeros(1, np.float32), on=np.array([0.6]), off=np.array([0.4]))
    return Path(path)


def _result(ok, reason=""):
    return {"passed": bool(ok), "reason": reason}


def _failure(exc):
    return _result(False, f"{type(exc).__name__}")


def level1(rate):
    from ....activity import FrameFeatures

    size, hop = sizes(rate)
    try:
        with strict():
            features = FrameFeatures(rate, size, hop / rate)
    except Exception as exc:  # noqa: BLE001 - every failure is a result here
        return _failure(exc)
    if features.bands != 20 or not (features.widths > 0).all() or features.count != FEATURES:
        return _result(False, "unexpected band layout or feature count")
    return _result(True)


def level2(rate):
    from ....activity import Context, FrameFeatures

    size, hop = sizes(rate)
    samples = size + 4 * hop
    try:
        with strict():
            for kind in ("silence", "noise", "tone"):
                for layout in ("mono", "identical", "anti-phase"):
                    features, context = FrameFeatures(rate, size, hop / rate), Context(hop / rate)
                    for spectrum in spectra(channels(signal(kind, samples, rate), layout), rate):
                        out = features.step(spectrum)
                        if out.shape != (FEATURES,) or out.dtype != np.float64 or not np.isfinite(out).all():
                            return _result(False, "unexpected feature shape, dtype or value")
                        smoothed = context.step(out)
                        if smoothed.shape != (CONTEXT_FEATURES,) or not np.isfinite(smoothed).all():
                            return _result(False, "unexpected context shape or value")
                        if layout != "anti-phase" and not (out[WIDTH] == 0).all():
                            return _result(False, "width is not exactly 0 for identical channels")
                        if layout == "anti-phase" and kind == "noise" and not (out[WIDTH] >= 0.99).all():
                            return _result(False, "width below 0.99 for anti-phase noise")
    except Exception as exc:  # noqa: BLE001
        return _failure(exc)
    return _result(True)


def level3(rate, weights):
    from ....activity import ActivityBank, ActivityModel

    size, hop = sizes(rate)
    samples = size + 4 * hop
    try:
        with strict():
            for kind in ("silence", "noise", "tone"):
                for layout in ("mono", "identical", "anti-phase"):
                    bank = ActivityBank([ActivityModel(weights)], rate, size, hop / rate)
                    for i, spectrum in enumerate(spectra(channels(signal(kind, samples, rate), layout), rate)):
                        quiet = i % 2 == 1
                        scores, active = bank.step(spectrum, quiet)
                        if scores.shape != (1,) or not np.isfinite(scores).all() or not (0 <= scores[0] <= 1):
                            return _result(False, "unexpected activity score")
                        if active.shape != (1,) or active.dtype != bool:
                            return _result(False, "unexpected activity decision")
                        if quiet and scores[0] != 0:
                            return _result(False, "a quiet frame scored above 0")
    except Exception as exc:  # noqa: BLE001
        return _failure(exc)
    return _result(True)


def l4_audio(rate, layout):
    parts = [signal(kind, round(seconds * rate), rate) for kind, seconds in L4_LAYOUT_S]
    mono = np.concatenate(parts)
    return mono[:, None] if layout == "mono" else np.stack((mono, 0.7 * mono), axis=1)


def level4(rate, weights):
    from ....models import AnalysisConfig
    from ..gmvpass import run

    silence_dbfs = AnalysisConfig().silence_dbfs
    try:
        with strict():
            for layout in ("mono", "stereo"):
                audio = l4_audio(rate, layout)
                _messages, _rows, frames = run(audio, rate, weights)
                if len(frames) != frontend.frame_count(len(audio), rate):
                    return _result(False, "frame count differs from the engine rule")
                quiet_seen = False
                for frame in frames:
                    scores = dict(frame.instrument_scores)
                    if "vocals" not in scores:
                        return _result(False, "silent fallback: no activity score")
                    if not np.isfinite(scores["vocals"]):
                        return _result(False, "non-finite activity score")
                    if frame.loudness_dbfs < silence_dbfs:
                        quiet_seen = True
                        if scores["vocals"] != 0:
                            return _result(False, "a quiet frame scored above 0")
                if not quiet_seen:
                    return _result(False, "the fixture produced no quiet frame")
    except Exception as exc:  # noqa: BLE001
        return _failure(exc)
    return _result(True)


def classify(rate, weights):
    """All four levels at ``rate``; ``classification`` is one of not_constructible,
    not_computable, failed_other or supported."""
    levels = {"L1": level1(rate), "L2": level2(rate), "L3": level3(rate, weights), "L4": level4(rate, weights)}
    if not levels["L1"]["passed"]:
        kind = "not_constructible"
    elif not levels["L2"]["passed"]:
        kind = "not_computable"
    elif all(v["passed"] for v in levels.values()):
        kind = "supported"
    else:
        kind = "failed_other"
    return {"rate": rate, "classification": kind, "levels": levels}


class UnsupportedRate(RuntimeError):
    """A paired excerpt's rate is not supported by B0; fold generation stops."""


def require_supported(rates, weights):
    """Classify each distinct rate; raise ``UnsupportedRate`` (counts only) unless all pass."""
    results = {rate: classify(rate, weights) for rate in sorted(set(rates))}
    failing = sum(1 for r in results.values() if r["classification"] != "supported")
    if failing:
        raise UnsupportedRate(f"{failing} of {len(results)} sample rates are not supported by B0")
    return results
