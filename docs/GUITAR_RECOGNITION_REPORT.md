# Guitar recognition milestone: report

2026-11-17. **The acceptance targets are not met, and the new model is not enabled by default.** It is a clear step over the earlier detectors: the same share of guitar found, at about a quarter of the false alarms on guitar-free music, and for the first time those false alarms are measured. It is selectable in Settings as an experimental model.

| Target | Result on the test recordings | Met |
|---|---|---|
| Recall where guitar is present ≥ 0.90 | 0.72 (rock songs: 0.89) | no |
| False-positive rate where guitar is absent ≤ 0.05 | 0.044 on guitar-removed songs, 0.039 on classical recordings; 0.082 in guitar-free passages of songs that do have guitar | partly |
| F1 ≥ 0.85 | 0.83 on real mixes (rock: 0.94); "always on" scores 0.91 there | no |
| Entrance delay ≤ 150 ms | median 1.0 s when detected; most entrances not detected within 5 s | no |
| No sustained false activation | 9 of 83 guitar-free recordings lit more than 20% of the time; longest false run 67 s | no |

## Status of each piece

| Piece | Status |
|---|---|
| Guitar-free test material (twins of MoisesDB songs, MusicNet), protocol | implemented and tested |
| Recognition backend, worker, message, lanes, note colouring, settings | implemented and tested (automated tests; not watched with sound) |
| Fine-tuned model `guitar_mn10_v1` | implemented, measured, **below target**; off by default |
| Distorted / clean sub-labels of that model | measured, not usable |
| Acoustic sub-label | measured, promising (see results), not validated further |
| Test on MedleyDB (untouched final test) | planned; waiting for the owner's access request |
| Check on the owner's own music (metal, electronic, orchestral) | planned; waiting for the two folders |
| Slakh2100 as extra training data | planned; download interrupted, being resumed (17 of 104 GB) |
| The Spheres orchestral recordings | downloaded; import rejected (instrument files differ in length), not used |
| PaSST, BEATs, hybrid with the separation model | investigated on paper / cost only, not scored |
| Note attribution beyond colouring notes by guitar state | not started, as agreed, until detection is reliable |
| Listening validation | not done |

## 1. Why the previous detectors failed

- They were trained from nothing on about 30 artists; none used a model that had already heard a large amount of music.
- Their inputs (20 band energies; later semitone spectra) do not separate a guitar from keyboards or strings in the same register.
- The test material was 82% to 85% guitar with one guitar-free song, so the failure that matters, reporting guitar where there is none, was never measured. "Always on" matched or beat every detector on F1.

Details: [GUITAR_RECOGNITION_RESEARCH.md](GUITAR_RECOGNITION_RESEARCH.md).

## 2. Approaches investigated

| Approach | Outcome |
|---|---|
| AudioSet "Guitar" output of a pretrained tagger, untrained | rejected: finds 11% of guitar at 5% false alarms |
| Frozen EfficientAT `mn10` on isolated 1 to 2 s windows + head | rejected: 46% to 52%; the model is unusable on clips shorter than about 5 s |
| Frozen `mn10`, 10 s context + head | 67%; superseded |
| **Fine-tuned `mn10`, 10 s context** | **selected**: 67% to 69%, one network |
| PaSST-S | not scored: 70 times the cost per estimate (2.5 s of one core) |
| BEATs, CLAP, MERT, Essentia models, musicnn, AST | not tried; licence, cost or installation reasons in the research document |
| Separation model (HTDemucs six-source) as second opinion | not combined; 2 to 4 s late and heavy, kept as a possible next step |
| Fast tier + context tier | not built: the only fast signal available has a 17% false-alarm rate |

Numbers and method: [evaluation/guitar_recognition_comparison.md](evaluation/guitar_recognition_comparison.md).

## 3. Selected model and architecture

- **Model:** EfficientAT `mn10_as` (MobileNetV3, 4.9 M parameters, pretrained on AudioSet), fine-tuned with a new output layer: guitar, distorted, clean, acoustic, one value per 0.31 s column of a 10 s clip.
- **Use:** every 0.5 s the last 10 s of audio are given to the model and the newest column is read. The score is averaged over two estimates and turned into **detected**, **candidate** or **absent** with two thresholds chosen on held-out artists (5% and 20% false alarms).
- **In the application:** `gmv.recognition` holds a backend interface, the streaming core (`Recognizer`) and a worker thread. The worker receives raw blocks from the analysis thread, never blocks it, and publishes `InstrumentEstimate` messages placed on the audio timeline by their interval. The offline scorer uses the same `Recognizer`, so the scored path is the shipped one.
- **Runtime:** ONNX Runtime on the CPU (optional dependency). No GPU is used or needed.

## 4. Data

| Source | Licence | Use | Split |
|---|---|---|---|
| MoisesDB v0.1 (owner's download) | CC BY-NC-SA 4.0 | training, selection, test | by artist: 147 train, 13 validation, 48 test (test read three times before this milestone) |
| MoisesDB twins: each song summed without its guitar files | same | guitar-free counterpart, same split as the song | 132 / 13 / 48 |
| MusicNet | CC BY 4.0 | guitar-free classical recordings | by work: 228 train, 68 validation, 34 test; only the 34 test recordings were used |
| The Spheres | CC BY-SA 4.0 | not used (import rejected) | – |
| Slakh2100 | CC BY 4.0 | not yet available | – |

No dataset has real metal or djent with separate stems. Cambridge-MT based sets were not used (terms unclear for research). Protocol: [evaluation/GUITAR_PROTOCOL.md](evaluation/GUITAR_PROTOCOL.md).

Label reliability: the reference is the activity of the dataset's own guitar files. About 30% of guitar time is more than 10 dB below the mix, and the model's recall there is 0.57 or lower; against an "audible" reference (guitar within 15 dB of the mix) selection-stage recall is 0.72 instead of 0.67. Twins can keep faint guitar where another file has bleed.

## 5. Test results (read once, through the application's path)

130 recordings: 48 held-out MoisesDB songs by 10 artists, their 48 guitar-free twins, 34 MusicNet recordings. Model, thresholds and smoothing were fixed beforehand. A decision is compared with the reference in the half second before it.

### Guitar, "detected"

| Pool | Recordings | Precision | Recall | F1 | False-positive rate | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Real mixes | 48 | 0.978 | 0.716 | 0.827 | 0.082 | 0.906 | 0.980 |
| Guitar-removed twins | 48 | – | – | – | 0.044 | – | – |
| MusicNet (classical) | 34 | – | – | – | 0.039 | – | – |
| All recordings | 130 | 0.847 | 0.716 | 0.776 | 0.044 | 0.902 | 0.845 |

- "Always on" on the real mixes: F1 0.909 (guitar plays 83% of the scored time). On all recordings it scores 0.407.
- False-negative rate 0.284.
- Guitar-free recordings lit more than 20% of the time: 6 of 48 twins, 3 of 34 MusicNet. Longest false run: 67 s (a twin), 40 s (MusicNet). Median guitar-free recording: never lit.
- MusicNet by ensemble: solo piano (9 recordings), winds (6) and solo violin (2) 0%; string quartet (7) 11%; solo cello (6) 14%. Bowed low strings are what it confuses with guitar there.
- State changes: 1.6 per minute on real mixes, 0.4 to 0.5 on guitar-free recordings.

### Guitar, "detected or candidate"

| Pool | Recall | False-positive rate | F1 |
|---|---:|---:|---:|
| Real mixes | 0.803 | 0.117 | 0.880 |
| Twins | – | 0.091 | – |
| MusicNet | – | 0.148 | – |

### By genre (real mixes, "detected")

| Genre | Songs | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Rock | 31 | 0.887 | 0.935 | 0.968 |
| Pop | 5 | 0.651 | 0.782 | 0.905 |
| Rap | 10 | 0.340 | 0.500 | 0.803 |

Rock is where the distorted guitars are, and it is the closest this material comes to the target genres. It is not metal.

### Entrances and exits

- 60 guitar entrances after at least a second without guitar: 24 were shown as detected within 5 s (median delay 1.0 s, 90th percentile 4.5 s); 36 were not.
- The model looks at 10 s, and its score rises and falls over several seconds around a change ([example](evaluation/plots/guitar_mn10_v1/r-f255d89a2c1f.png)). This is the main reason for the 8% false-alarm rate inside songs that have guitar: the lane stays lit for a few seconds after the guitar stops.

### Against the earlier detectors, same held-out songs

| Detector | Recall | False-positive rate, real mixes | Lit on twins | Lit on MusicNet | Delay |
|---|---:|---:|---:|---:|---|
| Band-energy network (`guitar_mlp_v1`) | 0.719 | 0.167 | 17.2% | 12.2% | about 50 ms |
| HTDemucs six-source guitar output (first 150 s of each song) | 0.92 | 0.25 | not measured | not measured | 2 to 4 s |
| **`guitar_mn10_v1`, detected** | 0.716 | 0.082 | 4.4% | 3.9% | 1 s or more |

## 6. Clean, distorted, acoustic

| Label ("detected") | Recall | Precision | False-positive rate, real mixes | Twins | MusicNet | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Distorted | 0.865 | 0.613 | 0.305 | 0.084 | 0.177 | not usable: fires on a third of the time other guitar plays |
| Clean | 0.632 | 0.667 | 0.204 | 0.099 | 0.205 | not usable |
| Acoustic | 0.781 | 0.901 | 0.023 | 0.010 | 0.000 | promising: F1 0.84 against 0.35 for "always on" |

The distorted and clean thresholds came from fold models and sit too low for the final model. In selection, songs whose only guitar is a clean electric one were found 24% of the time.

## 7. Performance (Ryzen 9 5950X, CPU only)

60 s of 48 kHz stereo fed at playback speed (`scripts/benchmark.py --realtime`):

| | Recognition off | Recognition on |
|---|---:|---:|
| Whole process, share of one core | 9.9% | 28.8% |
| Peak memory | 114 MiB | 192 MiB |
| Time per estimate (resample + spectrogram + network, 2 threads) | – | 25.5 ms median, 35.5 ms max |
| Estimates / skipped / dropped blocks | – | 120 / 0 / 0 |
| Capture blocks dropped | 0 | 0 |
| Frame analysis time, 95th percentile | 1.92 ms | 1.90 ms |
| Frame age at publication, 95th percentile | 43.2 ms | 43.1 ms |
| Estimate published after its newest audio | – | 44 ms |

The network alone takes 13 ms per estimate in isolation. The benchmark has no window; drawing cost with recognition on was not measured.

## 8. Examples

Timelines of test recordings (reference, score, what the application shows, errors):

- [Rock song, guitar throughout: found 99%](evaluation/plots/guitar_mn10_v1/r-6cb53f7ef737.png)
- [Rock song, guitar enters and leaves: found 94%, slow at each change](evaluation/plots/guitar_mn10_v1/r-f255d89a2c1f.png)
- [Rap song with clean guitar: missed almost entirely](evaluation/plots/guitar_mn10_v1/r-a25c36582140.png)
- [Rock song without its guitar files: lit 36% of the time](evaluation/plots/guitar_mn10_v1/r-8795aa19ec79.png)
- [Solo cello: lit 49% of the time](evaluation/plots/guitar_mn10_v1/r-f681f82fd4a5.png)

Known failure cases: clean electric rhythm guitar (rap, reggae, funk-like parts); guitar more than 10 dB below the mix; bowed low strings and, in pop arrangements, piano; several seconds of lag at entrances and exits. Not tested: metal, djent, the owner's music, anything with sound.

## 9. Changes to the application

- `src/gmv/recognition/`: backend interface and loader, three-state decision, streaming core, worker, ONNX backend.
- `InstrumentEstimate` message (schema regenerated), written to `advanced_analysis/recognition.jsonl` in session logs.
- Engine: starts the worker when a model is configured, offers it every block, reports its status, estimates, skipped hops and drops.
- Lanes: a lane row now carries detected (strong fill), candidate (weak fill) or nothing; a lane has one source; the guitar lane says "recognition unavailable" with the reason when the model cannot run.
- Notes: coloured by the guitar lane at drawing time, so a late estimate recolours notes already shown; pale for candidate.
- Settings: the Guitar field takes frame weights (`.npz`) or a recognition model (`.json`). Empty by default.
- A guitar weights file that cannot be read is now reported when Start is pressed instead of inside the analysis thread.
- Dataset adapters: `moisesdb_noguitar`, `musicnet`, `spheres`.

## 10. Checks

- `pytest -q`: 170 passed (13 new: decisions, timing, hop skipping, queue saturation, missing / broken / unknown model, inference failure, engine with and without recognition, lane and note states, twins adapter).
- `ruff check .`: clean. `python -m gmv.schema_export --check`: clean.
- `dotnet build native/ProcessLoopback -c Release`: 0 errors (no native code changed).
- Exported model against the training code on the same audio: largest score difference 9×10⁻⁷.

## 11. Weights and redistribution

| File | Origin | Terms |
|---|---|---|
| EfficientAT `mn10_as` | github.com/fschmid56/EfficientAT, release v0.0.1 | code MIT; the weight file carries no separate licence (assumed MIT, unverified); trained on AudioSet |
| `guitar_mn10_v1` (`.onnx`, `.frontend.npz`, `.json`) | fine-tuned here on MoisesDB | derived from CC BY-NC-SA data: kept on this machine, not in the repository, not for commercial use |
| ONNX Runtime | PyPI | MIT |

The description file records the checksums of the network and the front end, the thresholds and the training settings.

## 12. Limits and next steps

In order of expected value:

1. **Test on MedleyDB and on the owner's music.** Every figure above rests on one dataset family; rock is the nearest thing to metal in it.
2. **More training artists.** Frozen and fine-tuned models stop at the same ranking quality, and misses are whole songs, which points at variety of training material (about 160 songs) rather than at the network. Slakh2100 (2,100 synthesized songs with separate guitar, keyboards and strings) is the first thing to add; MedleyDB cannot be both training and final test.
3. **Faster reaction at changes.** Train with shorter contexts or with the newest column weighted, so the model commits within a second; or pair it with a fast cue for entrances only.
4. **Sub-label thresholds** set on the final model with held-out data, then a decision on showing the acoustic lane.
5. **Second opinion from the separation model** on stretches where this model says "candidate".

Not claimed: real-world validation, performance on metal, behaviour with sound, drawing cost with recognition on.

## Reproduce

```
gmv corpus --manifest corpus-data/corpus/manifest.json import-dataset moisesdb_noguitar C:\\Datasets\\moisesdb_v0.1
gmv corpus --manifest corpus-data/corpus/manifest.json import-dataset musicnet E:\\datasets\\musicnet
python experiments/recognition/prepare.py --corpus corpus-data/corpus/manifest.json --dataset moisesdb --dataset-root C:\\Datasets\\moisesdb_v0.1 --labels LABELS --jobs JOBS.json
python experiments/recognition/cache_stems.py --jobs JOBS.json --dataset-root C:\\Datasets\\moisesdb_v0.1 --output PARTS
python experiments/recognition/finetune.py --jobs JOBS.json --parts PARTS --labels LABELS --source EfficientAT-src --fold 0/4 --weights W.pt --scores SCORES
python experiments/recognition/summarise.py SCORES --labels LABELS
python experiments/recognition/finetune.py ... --fold=-1/4 --weights .model-cache/guitar_mn10_ft_all.pt
python experiments/recognition/export.py --weights .model-cache/guitar_mn10_ft_all.pt --source EfficientAT-src --name guitar_mn10_v1 --thresholds T.json --output .model-cache
python scripts/score_recognition.py run --corpus corpus-data/corpus/manifest.json --model .model-cache/guitar_mn10_v1.json --datasets moisesdb,moisesdb_noguitar,musicnet --split test --output OUT
python scripts/score_recognition.py report OUT
python experiments/recognition/plot_timeline.py OUT --model .model-cache/guitar_mn10_v1.json --output PLOTS
python scripts/benchmark.py --realtime --recognition .model-cache/guitar_mn10_v1.json
```

The first three experiment scripts run in the project environment; `finetune.py`, `summarise.py`, `export.py` and `plot_timeline.py` run in the experiments environment (torch, torchaudio, matplotlib; see `experiments/separation/README.md`).
