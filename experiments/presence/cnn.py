"""Model C: the causal log-mel CNN of protocol §5. Forward pass and streaming only.

Needs PyTorch (CPU) from the separate experiments environment; the application never
imports it. There is deliberately no training loop here (that is Stage 2).

Input: log-mel frames shaped (batch, 64 bands, frames). Output: one logit per frame,
shaped (batch, frames); ``torch.sigmoid`` gives the voice probability.

* Six Conv2d layers (kernel 3 frequency × 3 time, no bias), each followed by FrameNorm2d and
  ReLU; max-pooling over 2 bands after layers 2, 4 and 6 (64 → 32 → 16 → 8 bands). Padding:
  frequency (1, 1), time (2, 0): left only, so frame t sees frames t−2 … t.
* A Conv1d projection 512 → 64 (kernel 1, no bias), FrameNorm1d, ReLU: h⁽⁰⁾.
* Four residual Conv1d layers 64 → 64, kernel 3, dilations 2, 4, 8, 16, left padding 2d, no
  bias: h⁽ᵏ⁾ = h⁽ᵏ⁻¹⁾ + ReLU(FrameNorm1d(conv(h⁽ᵏ⁻¹⁾))).
* Dropout 0.2, then Conv1d 64 → 1 (kernel 1, with bias).

PyTorch convolutions are cross-correlations: with left padding p, output t combines inputs
t−p … t, tap 0 being the oldest. The protocol writes the same sums with tap 0 as the newest
(Σⱼ Wⱼ·x[t−j·d]); the two index the same weights in opposite order and compute the same
family of functions.

FrameNorm normalises each frame over its own channels (× frequency, in 2D) with a
per-channel scale and shift and no running statistics: training and inference compute the
same thing, and no frame depends on another frame or another batch item.

Dropout is ``SeededDropout``: the same inverted dropout as ``nn.Dropout`` (each element zeroed
with probability p, survivors scaled by 1/(1−p), identity in inference), drawing its mask
from its own ``torch.Generator`` so it never touches PyTorch's global random state. It has
no parameters or buffers. In training mode it refuses to run without a generator.
"""

import torch
from torch import nn
from torch.nn import functional

BANDS = 64
CHANNELS = (16, 16, 32, 32, 64, 64)
POOL_AFTER = (1, 3, 5)  # zero-based layers followed by frequency pooling
DILATIONS = (2, 4, 8, 16)
WIDTH = 64
DROPOUT = 0.2
EPS = 1e-5
RECEPTIVE_FIELD = 1 + 2 * len(CHANNELS) + 2 * sum(DILATIONS)  # 73 frames
PARAMETERS = 154_641


class FrameNorm2d(nn.Module):
    """Per frame: normalise over (channels, frequency); per-channel scale and shift."""

    def __init__(self, channels, eps=EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):  # (batch, channels, bands, frames)
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        y = (x - mean) / torch.sqrt(var + self.eps)
        return y * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class FrameNorm1d(nn.Module):
    """Per frame: normalise over channels; per-channel scale and shift."""

    def __init__(self, channels, eps=EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):  # (batch, channels, frames)
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        y = (x - mean) / torch.sqrt(var + self.eps)
        return y * self.weight[None, :, None] + self.bias[None, :, None]


class SeededDropout(nn.Module):
    """Inverted dropout with a dedicated generator (``generator`` must be set to train)."""

    def __init__(self, p):
        super().__init__()
        self.p = p
        self.generator = None

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        if self.generator is None:
            raise RuntimeError("SeededDropout needs its own generator in training mode")
        keep = torch.empty_like(x).bernoulli_(1 - self.p, generator=self.generator)
        return x * keep / (1 - self.p)


class PresenceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        inputs = (1, *CHANNELS[:-1])
        self.convs = nn.ModuleList(nn.Conv2d(i, o, (3, 3), bias=False) for i, o in zip(inputs, CHANNELS, strict=True))
        self.norms = nn.ModuleList(FrameNorm2d(o) for o in CHANNELS)
        self.pool = nn.MaxPool2d((2, 1), (2, 1))
        bands = BANDS // 2 ** len(POOL_AFTER)
        self.project = nn.Conv1d(CHANNELS[-1] * bands, WIDTH, 1, bias=False)
        self.project_norm = FrameNorm1d(WIDTH)
        self.temporal = nn.ModuleList(nn.Conv1d(WIDTH, WIDTH, 3, dilation=d, bias=False) for d in DILATIONS)
        self.temporal_norms = nn.ModuleList(FrameNorm1d(WIDTH) for _ in DILATIONS)
        self.dropout = SeededDropout(DROPOUT)
        self.out = nn.Conv1d(WIDTH, 1, 1, bias=True)

    def features(self, mel):
        """h⁽⁴⁾, shaped (batch, 64, frames): everything before dropout."""
        if mel.ndim != 3 or mel.shape[1] != BANDS:
            raise ValueError(f"expected log-mel input shaped (batch, {BANDS}, frames)")
        x = mel.unsqueeze(1)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            x = functional.relu(norm(conv(functional.pad(x, (2, 0, 1, 1)))))
            if i in POOL_AFTER:
                x = self.pool(x)
        h = functional.relu(self.project_norm(self.project(x.flatten(1, 2))))
        for conv, norm, d in zip(self.temporal, self.temporal_norms, DILATIONS, strict=True):
            h = h + functional.relu(norm(conv(functional.pad(h, (2 * d, 0)))))
        return h

    def forward(self, mel):
        return self.out(self.dropout(self.features(mel))).squeeze(1)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def build(seed, dtype=torch.float32, dropout_seed=None):
    """A freshly initialised model; the same seed always gives the same weights. PyTorch's
    global random state is left as it was. The dropout generator is seeded with
    ``dropout_seed`` (default: ``seed``); ``rng.build_for`` derives both from a seed plan."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = PresenceCNN().to(dtype)
    model.dropout.generator = torch.Generator().manual_seed(int(seed if dropout_seed is None else dropout_seed))
    return model


class Streamer:
    """Frame-by-frame inference with the model's own layers (inference mode, no dropout).
    State: the last two inputs of each 2D layer and the last 2d values entering each
    temporal layer, all zero at the start (as the batch path's zero padding)."""

    def __init__(self, model):
        self.model = model.eval()
        reference = next(model.parameters())
        self.dtype, self.device = reference.dtype, reference.device
        bands, previous = [BANDS], 1
        for i, _ in enumerate(CHANNELS):
            bands.append(bands[-1] // 2 if i in POOL_AFTER else bands[-1])
        self.history2d = [torch.zeros(1, c_in, bands[i], 2, dtype=self.dtype, device=self.device)
                          for i, c_in in enumerate((previous, *CHANNELS[:-1]))]
        self.history1d = [torch.zeros(1, WIDTH, 2 * d, dtype=self.dtype, device=self.device) for d in DILATIONS]

    @torch.no_grad()
    def step(self, frame):
        """One log-mel frame (64 values) in, one logit out."""
        frame = torch.as_tensor(frame, dtype=self.dtype, device=self.device).reshape(1, 1, BANDS, 1)
        model, x = self.model, frame
        for i, (conv, norm) in enumerate(zip(model.convs, model.norms, strict=True)):
            window = torch.cat((self.history2d[i], x), dim=3)  # frames t−2, t−1, t
            self.history2d[i] = window[..., 1:]
            x = functional.relu(norm(conv(functional.pad(window, (0, 0, 1, 1)))))
            if i in POOL_AFTER:
                x = model.pool(x)
        h = functional.relu(model.project_norm(model.project(x.flatten(1, 2))))
        for k, (conv, norm, d) in enumerate(zip(model.temporal, model.temporal_norms, DILATIONS, strict=True)):
            past = self.history1d[k]  # values for frames t−2d … t−1
            taps = torch.cat((past[..., 0:1], past[..., d : d + 1], h), dim=2)  # t−2d, t−d, t
            self.history1d[k] = torch.cat((past[..., 1:], h), dim=2)
            h = h + functional.relu(norm(functional.conv1d(taps, conv.weight)))
        return float(model.out(h).reshape(()))
