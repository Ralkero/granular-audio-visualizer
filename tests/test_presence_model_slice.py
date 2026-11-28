"""A5.2 Step 1, S1.3 (PyTorch part): model C on a slice of the canonical segment.

C trains on rows [s0, f1) of the segment's log-mel and predicts the whole segment. For every
scored frame (f0 ... f1-1) the two must agree: exactly in exact arithmetic, because C sees
only frames t-72 ... t; here within float32 rounding, and to 1e-12 in float64. Generated
audio only; the log-mel is not standardised (equivalence does not depend on it).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "presence"))

import cnn  # noqa: E402
import rng  # noqa: E402

from gmv.character.research.presence_model import frontend, grid  # noqa: E402

RATE = 44100
RECORDING = np.random.default_rng(5).normal(0, 0.05, (20 * RATE, 2)).astype(np.float32)
IDENTITY = "gmv-a5.2-gate:C:outer-0:conv:0"


@pytest.fixture(autouse=True)
def deterministic():
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


@pytest.mark.parametrize(("start", "s0"), ((3.013, 75), (0.5, 0), (7.0, 174)))
@pytest.mark.parametrize(("dtype", "tolerance"), ((torch.float32, 1e-5), (torch.float64, 1e-12)))
def test_scored_frames_agree_between_the_slice_and_the_full_segment(start, s0, dtype, tolerance):
    seg = grid.canonical(len(RECORDING), RATE, start, start + 10.0)
    mel = frontend.log_mel(RECORDING[seg.first : seg.end], RATE)
    assert mel.shape == (seg.frames, 64) and seg.context_start() == s0
    model = rng.build_for(IDENTITY, dtype).eval()
    full_input = torch.as_tensor(mel.T[None], dtype=dtype)
    rows = seg.c_rows()
    with torch.no_grad():
        full = model(full_input)[0, seg.f0 : seg.f1]
        sliced = model(full_input[..., rows])[0, seg.f0 - rows.start :]
    assert sliced.shape == full.shape == (seg.f1 - seg.f0,)
    assert float((full - sliced).abs().max()) <= tolerance
    assert cnn.RECEPTIVE_FIELD - 1 == grid.RECEPTIVE_HISTORY < grid.CONTEXT_FRAMES
