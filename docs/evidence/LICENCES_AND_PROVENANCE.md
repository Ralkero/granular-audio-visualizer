# Evidence enrichment: licences and provenance

Checked 2026-11-21. One layer of a source rarely shares the licence of another, so each is
listed on its own. Nothing in this table is redistributed by the project except its own
code. The per-recording counterpart of this page, for the owner's collection, is the local
file `research-local/evidence-v1/Enriched_Music_Inventory.csv`; for datasets it is
`External_Dataset_Manifest.csv` in the same folder.

## Tools

| Tool | Version | Source | Licence | Use | Integrity |
|---|---|---|---|---|---|
| MusicBrainz web service | ws/2 | https://musicbrainz.org/doc/MusicBrainz_API | data: core data CC0, supplementary data (tags, relationships annotations) CC BY-NC-SA 3.0 | recording identity and performer credits | every response cached with its URL and time |
| MusicBrainz Picard | 3.0.5 portable | https://picard.musicbrainz.org/downloads/ | GPL-2.0-or-later | available for looking recordings up by hand; never run in batch | Authenticode signature valid (MetaBrainz Foundation Inc.); SHA-256 `a761b4a9…506b86cb` |
| Chromaprint `fpcalc` | 1.6.2 | https://github.com/acoustid/chromaprint/releases | LGPL-2.1-or-later (binary bundles FFmpeg) | fingerprints, computed locally | SHA-256 of `fpcalc.exe` `1fff0e00…9027251e` |
| FFmpeg | 7.1.1 | already installed | LGPL/GPL build | decoding for the second model and for separation | system install |
| AcoustID web service | not used | https://acoustid.org/webservice | service terms; lookups free for non-commercial use | none until the owner approves sending fingerprints | |
| Discogs API | not used | https://www.discogs.com/developers | API terms of use; data partly CC0, partly restricted; needs an account token for search | none in this version | |

Picard is configured not to write tags, rename or move files (`Picard-readonly.ini`, passed
with `--config-file`). That configuration has not been exercised: Picard was not run in this
milestone, because the web service gave the same data without touching the collection.

## Models

| Model | Weights | Code | Trained on | Use | Restrictions |
|---|---|---|---|---|---|
| `guitar_mn10_v1` (this project) | local; SHA-256 `28384c43…373c0d1d` | this repository | EfficientAT `mn10_as` (AudioSet) fine-tuned on MoisesDB mixes | recogniser timelines | non-commercial, share-alike (CC BY-NC-SA 4.0) because of MoisesDB |
| `guitar_mn10_rv1` (this project) | local only | this repository | as above plus 223 excerpts of the owner's recordings | not used for evidence: it has seen the reviewed excerpts | local only |
| `guitar_mn10_rv2` (this project) | local only | this repository | as `guitar_mn10_v1` plus 385 human-confirmed excerpts of the owner's recordings | experimental alternative for listening; not used for evidence: it has seen every reviewed excerpt | local only |
| PANNs CNN14 `DecisionLevelMax` | Zenodo 3987831, `Cnn14_DecisionLevelMax_mAP=0.385.pth`, MD5 `70539c43…03a82c5a`, SHA-256 `dd3b4043…abdd142` | `panns-inference` 0.1.1 (MIT), `torchlibrosa` 0.1.0 | AudioSet | second audio model, frame-wise | weights CC BY 4.0 (Zenodo record): attribution to Kong et al. 2020 |
| HTDemucs six-source (`htdemucs_6s`, 5c90dfd2) | local; SHA-256 `34c22ccb…f9a02ddd` | `demucs` 4.1.0 (MIT) | MUSDB18-HQ plus 800 songs internal to its authors | guitar-stem level as supporting evidence | weights released with the MIT-licensed repository; the authors describe the guitar and piano outputs as experimental |

The two audio models share AudioSet as pretraining data. Their agreement is supporting
evidence, not two independent measurements. The separation model was trained on different
music and with a different objective, which is why it is treated as a third kind of source.

Environment for the second model and separation: `.separation-venv`, Python 3.12.9,
torch 2.9.1+cpu, librosa 1.0.1. None of it is imported by the application.

## Datasets

| Dataset | Audio | Annotations / metadata | Code | Weights trained on it | Redistribution by this project |
|---|---|---|---|---|---|
| OpenMIC-2018 v1.0.0 | each clip under its Free Music Archive track's Creative Commons licence; mostly NonCommercial, a large share NoDerivatives | CC BY 4.0 | MIT | not trained on. If it ever is: non-commercial, and NoDerivatives clips left out until that is settled | none |
| MoisesDB v0.1 | CC BY-NC-SA 4.0 | same | MIT (loader) | `guitar_mn10_v1`: non-commercial, share-alike, attribution | none |
| MedleyDB 1.0 / 2.0 | CC BY-NC-SA 4.0; maintainers ask that it is not republished | same | MIT | none planned (test use only) | none |
| MTG-Jamendo | per track, Creative Commons variants | CC BY-NC-SA 4.0 | Apache-2.0 | none | none; only the instrument tag file was fetched |
| MUSDB18-HQ, MusicNet, GuitarSet | see [DATASETS.md](../DATASETS.md) | | | | none |

Attribution:

- OpenMIC-2018: E. Humphrey, S. Durand, B. McFee, "OpenMIC-2018: An Open Dataset for
  Multiple Instrument Recognition", ISMIR 2018.
- PANNs: Q. Kong, Y. Cao, T. Iqbal, Y. Wang, W. Wang, M. Plumbley, "PANNs: Large-Scale
  Pretrained Audio Neural Networks for Audio Pattern Recognition", IEEE/ACM TASLP 2020.
- Demucs: S. Rouard, F. Massa, A. Défossez, "Hybrid Transformers for Music Source
  Separation", ICASSP 2023.
- MoisesDB: I. Pereira, F. Araújo, F. Korzeniowski, R. Vogl, ISMIR 2023.
- MedleyDB: R. Bittner et al., ISMIR 2014; MedleyDB 2.0, ISMIR 2016 late-breaking.
- MTG-Jamendo: D. Bogdanov et al., ICML 2019 ML4MD workshop.
- MusicBrainz: MetaBrainz Foundation and its contributors.

## The owner's collection

- Read only. Files are opened for reading, hashed and decoded in memory; nothing is written
  beside them, and no tag, name or location is changed. The fingerprint step and both model
  scans compare each file's SHA-256 with the inventory (or read it through a decoder that
  cannot write).
- What left this machine: MusicBrainz queries containing recording ids, ISRCs, or artist and
  title text from the files' tags, with a User-Agent naming this project. No audio, no
  fingerprint, no path, no listening history.
- What stays in ignored folders (`research-local/`, `review-local/`): identities, credits
  matched to recordings, timelines, label candidates, review queue, excerpts, labels.
- Weights trained on these recordings (`guitar_mn10_rv1`, `guitar_mn10_rv2`) are local only.

## Implications for a release

`guitar_mn10_v1` and `activity_v2` are in the 0.1.0 and 0.1.1 releases for non-commercial use. A model
trained with label candidates from this milestone would additionally depend on the owner's
recordings and, if OpenMIC audio is used, on NonCommercial and NoDerivatives clips. Such
weights should stay local until that is looked at again; the code can be released as it is.
