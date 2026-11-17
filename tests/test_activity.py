import numpy as np
import pytest

from gmv.activity import (
    FEATURE_VERSION,
    ActivityEstimator,
    ActivityModel,
    Context,
    FrameFeatures,
    add_context,
    intervals,
)
from gmv.evaluation.detect import run_detector
from gmv.models import AnalysisConfig

RATE, SIZE, HOP = 48000, 2229, 0.02


def spectra(count, gain=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.fft.rfft(gain * rng.standard_normal((count, SIZE, 2)), axis=1)


def weights(path, bias=(5.0, -5.0), version=FEATURE_VERSION):
    """A stand-in network: the first label always scores high, the second always low."""
    count = FrameFeatures(RATE, SIZE, HOP).count * 4
    np.savez(
        path,
        feature_version=version,
        name="test_model",
        labels=np.array(["drums", "bass"]),
        mean=np.zeros(count),
        scale=np.ones(count),
        w1=np.zeros((count, 3)),
        b1=np.zeros(3),
        w2=np.zeros((3, 2)),
        b2=np.array(bias),
        on=np.full(2, 0.6),
        off=np.full(2, 0.4),
    )
    return path


def test_streaming_context_matches_batch_context():
    features = FrameFeatures(RATE, SIZE, HOP)
    rows = np.array([features.step(s) for s in spectra(120)])
    context = Context(HOP)
    stepped = np.array([context.step(row) for row in rows])
    assert stepped == pytest.approx(add_context(rows, HOP), abs=1e-9)
    assert np.isfinite(stepped).all()


def test_features_do_not_depend_on_playback_level():
    quiet, loud = FrameFeatures(RATE, SIZE, HOP), FrameFeatures(RATE, SIZE, HOP)
    a = np.array([quiet.step(s) for s in spectra(40, 0.01)])
    b = np.array([loud.step(s) for s in spectra(40, 1.0)])
    assert a == pytest.approx(b, abs=1e-4)


def test_mono_input_has_zero_stereo_width():
    features = FrameFeatures(RATE, SIZE, HOP)
    row = features.step(spectra(1)[0][:, :1])
    assert not row[60:80].any()


def test_model_rejects_weights_for_other_features(tmp_path):
    with pytest.raises(ValueError):
        ActivityModel(weights(tmp_path / "old.npz", version="something_else"))


def test_estimator_applies_hysteresis_and_silence(tmp_path):
    model = ActivityModel(weights(tmp_path / "m.npz"))
    estimator = ActivityEstimator(model, RATE, SIZE, HOP)
    frame = spectra(1)[0]
    scores, active = estimator.step(frame, quiet=False)
    assert scores[0] > 0.9 and scores[1] < 0.1 and active.tolist() == [True, False]
    model.b2 = np.array([0.0, 0.0])  # 0.5: between the off and on thresholds
    assert estimator.step(frame, quiet=False)[1].tolist() == [True, False]
    assert estimator.step(frame, quiet=True)[1].tolist() == [False, False]


def test_intervals_from_frame_decisions():
    times = np.arange(10) * 0.02 + 0.01
    active = [False, True, True, False, False, True, True, True, False, True]
    spans = intervals(times, active, 0.02)
    assert np.asarray(spans) == pytest.approx(np.array([(0.02, 0.06), (0.10, 0.16), (0.18, 0.20)]))


def test_engine_reports_activity_only_with_a_model(tmp_path):
    audio = 0.1 * np.random.default_rng(2).standard_normal((RATE, 2)).astype(np.float32)
    audio[: RATE // 2] = 0
    frames, *_ = run_detector(audio, RATE, "candidate", AnalysisConfig())
    assert all(f.activity_method is None and not f.instruments_active for f in frames)
    config = AnalysisConfig(activity_model_path=str(weights(tmp_path / "m.npz")))
    frames, *_ = run_detector(audio, RATE, "candidate", config)
    assert all(f.activity_method == "test_model" for f in frames)
    assert frames[2].instruments_active == ()  # digital silence
    assert frames[-1].instruments_active == ("drums",)
    assert dict(frames[-1].instrument_scores)["bass"] < 0.1
