# Guitar recognition: audit and research (stage 1)

Written 2026-11-17, before any new model was scored. Results are in [evaluation/guitar_recognition_comparison.md](evaluation/guitar_recognition_comparison.md); the test protocol is in [evaluation/GUITAR_PROTOCOL.md](evaluation/GUITAR_PROTOCOL.md).

## Why the three earlier detectors failed

| Attempt | Held-out result | Why it is not usable |
|---|---|---|
| Band-energy network, `guitar_mlp_v1` ([report](evaluation/moisesdb_guitar_v1.md)) | recall 0.72, false-positive rate 0.17, F1 0.82 against 0.92 for "always on" | 20 coarse bands built for drums, bass and vocals cannot tell a guitar from keys or strings in the same register |
| Semitone spectra + small convolutional network ([report](evaluation/guitar_timbre_experiment.md)) | AUC 0.64 to 0.89 depending on which artists are held out | about 30 training artists are not enough to learn guitar timbre from nothing |
| Guitar output of six-source HTDemucs ([report](evaluation/moisesdb_guitar_separation.md)) | recall 0.92, false-positive rate 0.25, 2 to 4 s late | the separator puts other instruments into its guitar output at a low level |

Two causes sit under all three:

1. **No model had seen enough music.** Everything was trained from nothing on one dataset of 47 artists.
2. **The test could not show the failure that matters.** Guitar plays 82% to 85% of the time in MoisesDB and one held-out song has no guitar. Behaviour on synthesizers, piano, strings and bass without guitar was never measured, and "always on" beats or nearly matches every detector on F1.

## Approaches considered

### A. Pretrained audio taggers used as they are

Models trained on AudioSet output 527 classes, among them Guitar, Electric guitar, Bass guitar, Acoustic guitar, Synthesizer, Piano and Heavy metal. AudioSet labels a 10 s clip as a whole and its "strong" labels do not mark instruments inside music, so these outputs were never trained to say *when* a guitar plays. Scored as a zero-training baseline.

### B. Frozen pretrained model + small trained head (selected for prototyping)

The model's internal representation is kept fixed and a small classifier is trained on it with MoisesDB stems as the reference. Published frozen-probe results on OpenMIC-2018 (20 instruments, guitar is one class) favour AudioSet-trained models over self-supervised music models.

| Model | Frozen instrument evidence | Size | Measured / expected CPU cost | Code licence | Weights | Status |
|---|---|---|---|---|---|---|
| EfficientAT `mn10_as` (MobileNetV3 distilled from PaSST) | OpenMIC 83.8 fine-tuned; no published frozen figure | 4.9 M parameters, 20 MB | **measured here: 10 ms per 1 s window, 13.5 ms per 2 s window, one thread** | MIT | GitHub release; no licence separate from the repository (assumed MIT, unverified) | **first candidate** |
| EfficientAT `dymn10_as` | OpenMIC 84.4 fine-tuned | 10.6 M, 40 MB | similar | MIT | same | candidate if `mn10` is close |
| PaSST-S | OpenMIC 0.85 to 0.86 frozen on 1 s windows | about 86 M | tens of ms per window (estimate) | Apache-2.0 | GitHub release; no separate licence stated | accuracy ceiling |
| BEATs iter3+ AS2M | OpenMIC 86.8 frozen | about 90 M | heavier than PaSST (estimate) | MIT | OneDrive links only | only if the two above miss |

### C. Rejected before prototyping

| Option | Reason |
|---|---|
| MERT, MuQ | weights are CC BY-NC; MERT probes clearly worse on instruments (OpenMIC 78.6) |
| Essentia models (Discogs-EffNet + MTG-Jamendo instrument head) | library is AGPL with no Windows wheel; models CC BY-NC-SA; head trained on uploader tags (PR-AUC 0.20) |
| AST | fixed 10 s input |
| musicnn | pins Python 3.7-era dependencies, not installable |
| M2D, ATST | licence terms not verified |
| CLAP zero-shot | no instrument-timing evidence; kept as a possible weak extra signal only |
| Separation alternatives (SCNet, HS-TasNet, BS-RoFormer six-stem) | no guitar output, or no openly licensed weights; none removes the latency |
| Another small network from nothing | already shown not to generalize |

### D. Separation as a second opinion

HTDemucs six-source stays available as a slower source of evidence (its curves are cached for the MoisesDB validation and held-out songs). It is only combined with the selected model if that model is close to the targets and short of them.

### E. Fast tier + context tier

The chosen model needs 0.5 to 2 s of audio. A separate low-latency guess would have to come from the existing band-energy network, which has a 17% false-alarm rate. A fast tier is added only if the comparison shows a short window (0.5 s) holding the false-alarm target; otherwise there is one tier and its real delay is reported.

## Platform

- CPU inference is enough: at 10 ms per window the model can run several times per second on one core.
- ONNX Runtime (MIT) runs on Python 3.11 to 3.14 and is the intended runtime inside the application, as an optional dependency. Its DirectML provider supports the Radeon RX 9070 XT but is in maintenance mode; it is not needed for a model this small.
- PyTorch on AMD GPUs under Windows (ROCm 7.2) exists and is immature; not used. Training the head on cached model outputs takes minutes on the CPU.
- No NVIDIA or CUDA dependency anywhere.

## Known risk going in

Published work on instrument activity (Gururani et al., ISMIR 2018) reports that models confuse synthesizers with distorted guitars. No public model or benchmark targets distorted guitar in metal. The test material therefore has to contain synth-heavy and keyboard-heavy music without guitar; see the protocol.

## Data

No openly licensed dataset has real metal or djent recordings with separate stems. Sources used and their roles are in the protocol. Not used: Cambridge-MT "Mixing Secrets" and datasets derived from it (RawStems, the Mixing Secrets instrument dataset), whose terms are "educational purposes only" with research use unresolved.

## What was verified and what was not

- Verified on primary pages: code licences, weight download locations, ONNX Runtime package and Python support, dataset licences and access conditions.
- Measured on this machine: EfficientAT `mn10_as` inference time.
- Not verified: that the EfficientAT and PaSST weight files carry the repository licence; CPU cost of PaSST and BEATs; DirectML with these models.
