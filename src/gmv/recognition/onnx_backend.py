"""A mel-spectrogram tagger run with ONNX Runtime on the CPU.

The description file names an ONNX network that takes a log-mel spectrogram of the whole
context and returns, per label, one value per time column; the newest column is the answer
for "now". The mel filter bank is stored beside the network so that the spectrogram is
computed here exactly as it was during training (EfficientAT's front end: pre-emphasis,
25 ms Hann windows every 10 ms, 128 mel bands, log, fixed offset and scale).

ONNX Runtime is an optional dependency (``pip install granular-music-visualizer[recognition]``).
"""

from pathlib import Path

import numpy as np
from scipy.fft import rfft

from . import file_sha256, register


def log_mel(audio, mel_basis, n_fft, window, hop):
    """(mel bands, frames) for mono float32 audio; mirrors ``torch.stft(center=True)``."""
    emphasised = audio[1:] - 0.97 * audio[:-1]
    padded = np.pad(emphasised, n_fft // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
    spectrum = rfft(frames * window, axis=1)  # single precision in, single precision out
    power = spectrum.real**2 + spectrum.imag**2
    return ((np.log(power @ mel_basis.T + 1e-5) + 4.5) / 5.0).T.astype(np.float32)


class OnnxMelTagger:
    def __init__(self, description, folder):
        try:
            import onnxruntime
        except ImportError as exc:
            raise ValueError("ONNX Runtime is not installed (pip install onnxruntime)") from exc
        self.name = description["name"]
        self.version = str(description.get("version", ""))
        self.labels = tuple(description["labels"])
        self.sample_rate = int(description["sample_rate"])
        self.context_s = float(description["context_s"])
        self.hop_s = float(description["hop_s"])
        self.thresholds = description["thresholds"]
        model = Path(folder) / description["model"]
        frontend = Path(folder) / description["frontend"]
        for path, key in ((model, "model_sha256"), (frontend, "frontend_sha256")):
            if not path.is_file():
                raise ValueError(f"Recognition model file missing: {path.name}")
            if description.get(key) and file_sha256(path) != description[key]:
                raise ValueError(f"{path.name} does not match the checksum in the description")
        with np.load(frontend, allow_pickle=False) as data:
            self.mel_basis = data["mel_basis"].astype(np.float32)
            self.n_fft, self.hop = int(data["n_fft"]), int(data["hop"])
            window = data["window"].astype(np.float32)
        # torch.stft centres a window shorter than the transform inside it.
        left = (self.n_fft - len(window)) // 2
        self.window = np.pad(window, (left, self.n_fft - len(window) - left))
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = int(description.get("threads", 2))
        options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0].name

    def infer(self, context):
        mel = log_mel(context, self.mel_basis, self.n_fft, self.window, self.hop)
        logits = self.session.run(None, {self.input: mel[None, None]})[0]
        return 1.0 / (1.0 + np.exp(-logits[0, :, -1].astype(np.float64)))


register("onnx_mel_tagger", OnnxMelTagger)
