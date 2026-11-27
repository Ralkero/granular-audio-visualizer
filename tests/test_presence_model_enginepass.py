"""A5.2 Step 1, S1.3: the engine's quiet rule and B1's features on generated audio.

Application environment (the engine pass needs its dependencies).
"""

import numpy as np
import pytest

from gmv import activity
from gmv.character.research import gmvpass
from gmv.character.research.presence_model import b0check, enginepass, frontend


def generated(rate, seconds=4.0, seed=3):
    rng = np.random.default_rng(seed)
    n = round(seconds * rate)
    left = rng.normal(0, 0.05, n).astype(np.float32)
    right = (0.6 * left + rng.normal(0, 0.02, n)).astype(np.float32)
    audio = np.stack((left, right), axis=1)
    audio[round(1.5 * rate) : round(2.3 * rate)] = 0  # 0.8 s of digital silence
    audio[round(3.0 * rate) : round(3.4 * rate)] *= 10 ** (-70 / 20) / 0.05  # about -70 dBFS
    return audio


def independent_quiet(audio, rate):
    size, hop = frontend.engine_window(rate), frontend.engine_hop(rate)
    out = []
    for i in range(frontend.frame_count(len(audio), rate)):
        window = audio[i * hop : i * hop + size].astype("<f4")
        rms = float(np.sqrt(np.mean(np.square(window), dtype=np.float64)))
        out.append(max(-120.0, 20 * np.log10(max(rms, 1e-6))) < -65)
    return np.array(out)


@pytest.mark.parametrize("rate", (44100, 48000))
def test_the_quiet_mask_is_the_engine_level_rule(rate):
    audio = generated(rate)
    quiet = enginepass.quiet_mask(audio, rate)
    assert quiet.dtype == bool and len(quiet) == frontend.frame_count(len(audio), rate)
    assert np.array_equal(quiet, independent_quiet(audio, rate))
    assert quiet.sum() > 50 and not quiet.all()


def test_quiet_uses_every_channel_not_the_mid_channel():
    rate = 44100
    n = 2 * rate
    left = np.random.default_rng(1).normal(0, 0.05, n).astype(np.float32)
    anti = np.stack((left, -left), axis=1)  # the mid channel is silent; the engine is not
    assert not enginepass.quiet_mask(anti, rate).any()
    assert np.allclose(frontend.mid(anti), 0)


@pytest.mark.parametrize("rate", (44100, 48000))
def test_b1_features_equal_the_engine_activity_bank_input(rate, tmp_path, monkeypatch):
    audio = generated(rate)
    weights = b0check.write_weights(tmp_path / "w.npz")
    captured = []
    original = activity.Context.step

    def recording(self, frame):
        out = original(self, frame)
        captured.append(out.copy())
        return out

    monkeypatch.setattr(activity.Context, "step", recording)
    gmvpass.run(audio, rate, weights)
    monkeypatch.undo()
    ours = enginepass.b1_features(audio, rate)
    assert ours.shape == (frontend.frame_count(len(audio), rate), 548)
    assert np.array_equal(ours, np.array(captured))
