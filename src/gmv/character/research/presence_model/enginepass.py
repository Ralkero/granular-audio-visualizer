"""Per-frame quantities from the analysis engine on a canonical segment (application
environment: needs the engine and its dependencies).

* ``quiet_mask``: the engine's own level rule. A frame is quiet when
  ``FeatureFrame.loudness_dbfs`` is below ``AnalysisConfig().silence_dbfs`` (-65 dBFS): the
  unwindowed RMS over all channels of the native-rate float32 window. Not the 0.25 s
  ``FeatureFrame.silence`` state, and never C's mid channel.
* ``b1_features``: the 548 activity features B1 uses, computed exactly as the engine's
  ``ActivityBank`` computes them (float32 windows, float32 Hann, ``rfft`` over samples, then
  ``FrameFeatures`` and ``Context``).
"""

import numpy as np

from . import frontend


def _float32_2d(audio):
    audio = np.asarray(audio, dtype="<f4")
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2:
        raise ValueError("audio must be shaped (samples,) or (samples, channels)")
    return audio


def quiet_mask(audio, rate):
    from ....models import AnalysisConfig
    from ..gmvpass import run

    audio = _float32_2d(audio)
    _messages, _rows, frames = run(audio, rate)
    if len(frames) != frontend.frame_count(len(audio), rate):
        raise RuntimeError("the engine produced a different number of frames than the grid")
    threshold = AnalysisConfig().silence_dbfs
    return np.array([frame.loudness_dbfs < threshold for frame in frames], dtype=bool)


def b1_features(audio, rate):
    from ....activity import Context, FrameFeatures

    audio = _float32_2d(audio)
    size, hop = frontend.engine_window(rate), frontend.engine_hop(rate)
    window = np.hanning(size).astype(np.float32)
    features, context = FrameFeatures(rate, size, hop / rate), Context(hop / rate)
    rows = [context.step(features.step(np.fft.rfft(audio[i * hop : i * hop + size] * window[:, None], axis=0)))
            for i in range(frontend.frame_count(len(audio), rate))]
    return np.array(rows) if rows else np.zeros((0, 548))
