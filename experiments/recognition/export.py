"""Turn fine-tuned weights into the three files the application loads.

Runs in the experiments environment. Writes, next to each other:

``<name>.onnx``           the network: log-mel spectrogram in, one value per label and column out
``<name>.frontend.npz``   the mel filter bank and window used in training
``<name>.json``           the description ``gmv.recognition.load_backend`` reads: labels,
                          thresholds, checksums and where the weights came from

and checks that the application's own spectrogram code and ONNX Runtime give the same
scores as the training code on the same audio.

    python experiments/recognition/export.py --weights .model-cache/guitar_mn10_ft.pt \
        --source EfficientAT-src --name guitar_mn10_v1 --thresholds THRESHOLDS.json \
        --output .model-cache
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))
import embed  # noqa: E402
from finetune import CLIP, RATE, Net  # noqa: E402

N_FFT, WINDOW, HOP, MELS, FMAX = 1024, 800, 320, 128, 15000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", default="1")
    parser.add_argument("--thresholds", required=True, help="JSON: label -> detected/candidate")
    parser.add_argument("--hop", type=float, default=0.5)
    parser.add_argument("--notes", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    saved = torch.load(args.weights, map_location="cpu")
    body, mel = embed.load_model(Path(args.source).resolve(), saved["model"])
    net = Net(body.features)
    net.load_state_dict(saved["state"])
    net.eval()

    model_path = out / f"{args.name}.onnx"
    example = torch.zeros(1, 1, MELS, CLIP // HOP)
    torch.onnx.export(
        net,
        example,
        str(model_path),
        input_names=["mel"],
        output_names=["logits"],
        dynamic_axes={"mel": {3: "frames"}, "logits": {2: "columns"}},
        opset_version=17,
        dynamo=False,
    )
    basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
        MELS, N_FFT, RATE, 0.0, FMAX, vtln_low=100.0, vtln_high=-500.0, vtln_warp_factor=1.0
    )
    basis = torch.nn.functional.pad(basis, (0, 1)).numpy().astype(np.float32)
    frontend_path = out / f"{args.name}.frontend.npz"
    np.savez(
        frontend_path,
        mel_basis=basis,
        n_fft=N_FFT,
        hop=HOP,
        window=torch.hann_window(WINDOW, periodic=False).numpy(),
    )
    description = {
        "kind": "onnx_mel_tagger",
        "name": args.name,
        "version": args.version,
        "labels": list(saved["labels"]),
        "sample_rate": RATE,
        "context_s": CLIP / RATE,
        "hop_s": args.hop,
        "threads": 2,
        "thresholds": json.loads(Path(args.thresholds).read_text(encoding="utf-8")),
        "model": model_path.name,
        "model_sha256": sha256(model_path),
        "frontend": frontend_path.name,
        "frontend_sha256": sha256(frontend_path),
        "source": {
            "base_model": f"EfficientAT {saved['model']} (github.com/fschmid56/EfficientAT, MIT)",
            "base_weights": "release v0.0.1 of that repository; trained on AudioSet",
            "fine_tuned_on": "MoisesDB v0.1 (CC BY-NC-SA 4.0), mixes made from its stems",
            "redistribution": "not redistributed; derived from non-commercial data",
            "training_script": "experiments/recognition/finetune.py",
            "notes": args.notes,
        },
    }
    (out / f"{args.name}.json").write_text(json.dumps(description, indent=2), encoding="utf-8")

    # The application's path against the training code, on noise and on a tone burst.
    root = Path(__file__).parents[2] / "src"
    sys.path.insert(0, str(root))
    from gmv.recognition import load_backend

    backend = load_backend(out / f"{args.name}.json")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(3):
        audio = (0.1 * rng.standard_normal(CLIP)).astype(np.float32)
        audio[CLIP // 2 :] += 0.2 * np.sin(np.arange(CLIP // 2) * 2 * np.pi * 330 / RATE)
        with torch.inference_mode():
            wanted = torch.sigmoid(net(mel(torch.from_numpy(audio)[None]).unsqueeze(1)))[0, :, -1]
        worst = max(worst, float(np.abs(backend.infer(audio) - wanted.numpy()).max()))
    print(f"largest score difference, application against training code: {worst:.2e}")
    if worst > 1e-3:
        raise SystemExit("The exported model does not reproduce the trained one")
    print(
        f"wrote {args.name}.json, .onnx ({model_path.stat().st_size / 1e6:.1f} MB), .frontend.npz"
    )


if __name__ == "__main__":
    main()
