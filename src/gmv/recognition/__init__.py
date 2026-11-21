"""Instrument recognition from a longer audio context, behind a replaceable backend.

A backend is a model that looks at the last few seconds of audio and returns one score per
label. It knows nothing about threads, queues or the visualizer. Around it:

``Recognizer``         keeps the recent audio, calls the backend every hop and turns scores
                       into one of three states per label. No threads; the live worker and
                       the offline evaluation both use it, so what is scored is what runs.
``RecognitionWorker``  (``worker.py``) runs a ``Recognizer`` on its own thread and
                       publishes ``InstrumentEstimate`` messages.

A model is described by a small JSON file next to its weights (``load_backend``). The file
names the backend kind, the labels, the decision thresholds chosen on validation data and
where the weights came from, so a result can be traced to the exact model.

States are deliberately coarse. ``detected`` means the score passed the threshold that was
chosen for a low false-alarm rate; ``candidate`` means it passed only a lower one. Scores are
not probabilities.
"""

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.signal import resample_poly

STATES = ("absent", "candidate", "detected")


class Backend(Protocol):
    name: str
    version: str
    labels: tuple[str, ...]
    sample_rate: int
    context_s: float  # audio the model is given
    hop_s: float  # how often it is asked
    thresholds: dict  # label -> {"detected": score, "candidate": score}

    def infer(self, context: np.ndarray) -> np.ndarray:
        """Scores (one per label) for mono audio of exactly ``context_s`` at ``sample_rate``.
        The newest sample is last; a recording shorter than the context starts with zeros."""


BACKENDS = {}


def register(kind, factory):
    """``factory(description: dict, folder: Path) -> Backend``."""
    BACKENDS[kind] = factory


def file_sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_backend(path):
    """Build the backend a model description file asks for. Raises ``ValueError`` with a
    readable reason when the file, its weights or a needed package is missing or wrong."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Recognition model file not found: {path}")
    try:
        description = json.loads(path.read_text(encoding="utf-8"))
        kind = description["kind"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Not a recognition model description: {path.name} ({exc})") from exc
    if kind == "onnx_mel_tagger" and kind not in BACKENDS:
        from . import onnx_backend  # noqa: F401  registers itself
    if kind not in BACKENDS:
        raise ValueError(f"Unknown recognition backend kind: {kind}")
    try:
        backend = BACKENDS[kind](description, path.parent)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Recognition model could not be loaded: {exc}") from exc
    missing = [label for label in backend.labels if label not in backend.thresholds]
    if missing:
        raise ValueError(f"No decision thresholds for: {', '.join(missing)}")
    return backend


@dataclass(frozen=True)
class Estimate:
    start: float  # audio time the newest hop began
    end: float  # audio time of the newest sample the model saw
    scores: tuple[tuple[str, float], ...]
    states: tuple[tuple[str, str], ...]


class Decision:
    """Three states from a stream of scores for one label.

    The score is first averaged over the last ``smooth`` hops. ``detected`` needs the average
    at or above the "detected" threshold; once detected it holds while the average stays at
    or above the "candidate" threshold, so a short dip does not make the lane flicker.
    """

    def __init__(self, detected, candidate, smooth=2):
        if candidate > detected:
            raise ValueError("The candidate threshold must not exceed the detected threshold")
        self.detected, self.candidate = detected, candidate
        self.recent = deque(maxlen=max(1, smooth))
        self.state = "absent"

    def step(self, score):
        self.recent.append(float(score))
        level = sum(self.recent) / len(self.recent)
        if level >= self.detected or (self.state == "detected" and level >= self.candidate):
            self.state = "detected"
        elif level >= self.candidate:
            self.state = "candidate"
        else:
            self.state = "absent"
        return self.state


def downmix(samples):
    """Mono from (frames, channels). More than two channels: front left and right only, as
    surround layouts put music there and the remaining channels are often silent."""
    return samples[:, : min(2, samples.shape[1])].mean(axis=1)


TARGET_RMS = 10 ** (-20 / 20)  # ordinary music level; the models were trained near it
MOST_GAIN = 10 ** (45 / 20)
SILENCE_RMS = 10 ** (-75 / 20)


def levelled(context):
    """Quiet audio raised to ordinary music level before the model sees it.

    Captured audio follows the player's volume and can be 30 to 40 dB below the level of
    the training material, where the models lose guitar they find at normal level. Audio
    at or above the target, and near-silence, is passed unchanged.
    """
    rms = float(np.sqrt(np.mean(np.square(context, dtype=np.float64))))
    if rms < SILENCE_RMS or rms >= TARGET_RMS:
        return context
    return (context * min(TARGET_RMS / rms, MOST_GAIN)).astype(np.float32)


class Recognizer:
    """Feed it consecutive audio; it returns an ``Estimate`` each time a hop has passed.

    Audio is kept at the rate it arrives in and resampled per inference, so there are no
    block-edge artefacts. ``reset`` forgets everything (a gap or a format change).
    """

    def __init__(self, backend, smooth=2):
        self.backend = backend
        self.smooth = smooth
        self.reset()

    def reset(self):
        self.rate = None
        self.buffer = None
        self.filled = 0  # samples received since the reset
        self.pending = 0  # samples since the last inference
        self.end = 0.0  # audio time just after the newest sample
        self.decisions = {
            label: Decision(
                self.backend.thresholds[label]["detected"],
                self.backend.thresholds[label]["candidate"],
                self.smooth,
            )
            for label in self.backend.labels
        }

    def push(self, mono, rate, start):
        """Add mono samples that begin at audio time ``start``. Returns whether a hop is due."""
        if rate != self.rate:
            self.reset()
            self.rate = rate
            self.buffer = np.zeros(round(self.backend.context_s * rate), np.float32)
        count = len(mono)
        if count >= len(self.buffer):
            self.buffer[:] = mono[-len(self.buffer) :]
        else:
            self.buffer[:-count] = self.buffer[count:]
            self.buffer[-count:] = mono
        self.filled += count
        self.pending += count
        self.end = start + count / rate
        return self.due

    @property
    def hops_waiting(self):
        if self.rate is None:
            return 0
        return self.pending // max(1, round(self.backend.hop_s * self.rate))

    @property
    def due(self):
        return self.hops_waiting >= 1

    def estimate(self):
        """Ask the backend about the audio held now. Call when ``due``."""
        backend = self.backend
        context = self.buffer
        if self.rate != backend.sample_rate:
            divisor = gcd(self.rate, backend.sample_rate)
            context = resample_poly(
                context, backend.sample_rate // divisor, self.rate // divisor
            ).astype(np.float32)
        size = round(backend.context_s * backend.sample_rate)
        context = (
            context[-size:] if len(context) >= size else np.pad(context, (size - len(context), 0))
        )
        scores = np.clip(np.asarray(backend.infer(levelled(context)), dtype=np.float64), 0.0, 1.0)
        if scores.shape != (len(backend.labels),) or not np.isfinite(scores).all():
            raise ValueError("The recognition backend returned unusable scores")
        # The answer is about the newest hop, however much audio went by unasked.
        span = min(self.pending, self.filled, round(backend.hop_s * self.rate)) / self.rate
        self.pending = 0
        return Estimate(
            start=max(0.0, self.end - span),
            end=self.end,
            scores=tuple(zip(backend.labels, (float(s) for s in scores), strict=True)),
            states=tuple(
                (label, self.decisions[label].step(score))
                for label, score in zip(backend.labels, scores, strict=True)
            ),
        )
