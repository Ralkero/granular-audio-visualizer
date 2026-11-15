# Real-music corpus specification (corpus schema 1.0)

The corpus measures the analysis engine against real recordings. **The repository never contains audio.** It contains a manifest of recording identities, the licensing basis for using each one locally, annotations, and the code that evaluates them. Schemas: [CorpusManifest.json](schemas/CorpusManifest.json), [AnnotationFile.json](schemas/AnnotationFile.json). Annotation workflow: [ANNOTATION.md](ANNOTATION.md).

## What may be registered

| `source_kind` | Meaning |
|---|---|
| `owned_local` | A recording you own a copy of (purchased download, your own CD rip), used only on your machine |
| `open_dataset` | A research dataset you downloaded yourself under its own licence (for example a multitrack set with stems) |
| `original` | Your own recording or composition |
| `public_domain` | Public-domain audio |
| `licensed` | Audio explicitly licensed for this use |

Not allowed: stream captures (Spotify or any other service), recordings without a permission basis, and uploading any audio to an external service. Every recording carries a `license` statement and a `permission_basis`; registration refuses to proceed without them. These are your statements and are not verified by the software.

## What is committed, and what is not

Committed (`corpus/`):

- `manifest.json` — per recording: `recording_id` (`r-` + first 12 hex digits of the file's SHA-256), full `sha256`, duration, sample rate, channel count, `source_kind`, `license`, `permission_basis`, `group_id`, `split`, descriptive `tags`, and SHA-256 hashes of any stems.
- `annotations/<recording_id>/<annotator_id>.json` — annotation files.

Never committed: audio, stems, file paths, file names, titles and artist names. The location of each file on your machine is stored in `corpus-local/paths.json`, which is git-ignored along with common audio extensions. Choose `group_id` and `tags` values that you are comfortable publishing (for example `album-03`, not the album title).

## Adding a recording

```powershell
gmv corpus register "D:\Music\track.flac" --source-kind owned_local `
  --license "Purchased download, personal copy" --permission-basis "Local analysis only; not redistributed" `
  --group album-03 --tag double_kick --tag harsh_vocals
gmv corpus register "D:\Datasets\song\mixture.wav" --source-kind open_dataset `
  --license "Dataset licence name" --permission-basis "Non-commercial research use" --group ds-song-017 `
  --stem drums="D:\Datasets\song\drums.wav" --stem bass="D:\Datasets\song\bass.wav" --stem vocals="D:\Datasets\song\vocals.wav"
gmv corpus assign-splits
gmv corpus status
gmv corpus validate
```

Registration hashes the file and reads its header; it does not copy or modify it. Supported formats are those of libsndfile (WAV, FLAC, OGG, MP3, AIFF). Convert other formats yourself. Registering the same file again on another machine only refreshes that machine's path map.

## Splits and leakage protection

- Splits are `train`, `validation` and `test`, assigned **per group**, never per segment. A group is the unit that must not straddle splits: an album, a recording session, one dataset song with its alternate versions.
- The manifest is invalid, and cannot be loaded, if any group appears in more than one split.
- `gmv corpus assign-splits` assigns only groups that are still `unassigned`, in a deterministic order derived from a seed, each to the split furthest below its target share (default 60/20/20). Existing assignments never move, so adding recordings later cannot shift a recording between splits. A recording added to an existing group inherits that group's split.
- `test` is the held-out set. The evaluation command refuses to score it without `--final`, and stamps the report when it does. Do not tune against it.

## Versioning and integrity

- `corpus_version` (`x.y.z`) is set with `gmv corpus set-version`. Bump it whenever recordings or reference annotations change in a way that affects results. Reports also record a SHA-256 over the manifest and every annotation file used, so two reports are comparable only when both match.
- Recording identity is the file hash. If a file changes (re-encode, retag that rewrites audio, different master) the hash no longer matches: `validate` and evaluation report "audio file changed" and skip it rather than score different audio against old annotations. A missing or unregistered file is reported as such. No other file is substituted.
- Each annotation stores the hash of the audio it describes; an annotation made against a different version is a validation error.
- `annotation_schema_version` and `corpus_schema_version` are fixed at `1.0`. A future incompatible change requires a new version and a documented migration.

## Tags

Tags describe musical conditions so results can be broken down by them. Suggested vocabulary (free-form, lower-case): `metal`, `metalcore`, `prog`, `double_kick`, `blast_beat`, `breakdown`, `tempo_change`, `layered_guitars`, `clean_vocals`, `harsh_vocals`, `atmospheric`, `quiet_passage`, `dense`, `sparse`.

## Current contents

The committed manifest is empty. Real-music evaluation is **pending** until recordings are registered, annotated and reviewed.
