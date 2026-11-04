import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from time import perf_counter
import numpy as np
import soundfile as sf
from gmv.dsp import Analyzer, WindowAssembler
from gmv.models import AnalysisConfig
from gmv.sources.base import AudioBlock
from .baseline_dsp import Analyzer as BaselineAnalyzer
from .corpus import CLASSES, CORPUS_VERSION, generate, Fixture
from .metrics import aggregate, onset_score, rhythm_score


def evaluate(fixture, detector="baseline", config=None):
    config = config or AnalysisConfig()
    analyzer = (BaselineAnalyzer if detector == "baseline" else Analyzer)(
        fixture.rate, fixture.audio.shape[1], config, "evaluation", fixture.name
    )
    assembler = WindowAssembler(config)
    frames, events, times = [], [], []
    step = round(fixture.rate * config.capture_ms / 1000)
    for pos in range(0, len(fixture.audio), step):
        audio = fixture.audio[pos : pos + step]
        block = AudioBlock(
            audio.astype("<f4").tobytes(),
            len(audio),
            fixture.rate,
            audio.shape[1],
            pos / fixture.rate,
            pos / fixture.rate,
            0,
        )
        for window, start, mono, _reset in assembler.push(block):
            began = perf_counter()
            frame, new_events = analyzer.process(window, start, mono, block, live=False)
            times.append((perf_counter() - began) * 1000)
            frames.append(frame)
            events.extend(new_events)
    predictions = [e.timestamp for e in events if e.event_type == "onset"]
    scores = {str(ms): onset_score(fixture.onsets, predictions, ms / 1000) for ms in (25, 50)}
    classes = {
        kind: onset_score(
            [t for t, c in fixture.attacks if c == kind],
            [e.timestamp for e in events if e.event_type == kind],
        )
        for kind in CLASSES
    }
    return {
        "fixture": fixture.manifest(),
        "onsets": scores,
        "classes": classes,
        "rhythm": rhythm_score(fixture, frames, events),
        "predicted_onsets": predictions,
        "performance": {
            "frames": len(frames),
            "dsp_p95_ms": float(np.percentile(times, 95)),
            "dsp_total_ms": sum(times),
        },
        "raw_rms_mean": float(np.mean([f.rms for f in frames])),
    }


def load_import(manifest):
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    required = (
        "audio",
        "sha256",
        "license",
        "provenance",
        "permission_verified",
        "onsets",
        "annotation_method",
        "uncertainty_seconds",
        "source_recording_id",
        "split",
    )
    if any(k not in data for k in required) or data["permission_verified"] is not True:
        raise ValueError(
            "Import requires verified permission, provenance, hash and annotation metadata"
        )
    path = Path(manifest).parent / data["audio"]
    if path.stat().st_size > 256 * 1024**2:
        raise ValueError("Import file exceeds 256 MiB")
    with path.open("rb") as source:
        file_hash = hashlib.file_digest(source, "sha256").hexdigest()
    if file_hash != data["sha256"]:
        raise ValueError("Audio hash mismatch")
    with sf.SoundFile(path) as file:
        if (
            file.frames / file.samplerate > 120
            or file.channels > 8
            or file.frames * file.channels * 4 > 128 * 1024**2
        ):
            raise ValueError("Imports are capped at 120 seconds / eight channels")
        audio = file.read(dtype="float32", always_2d=True)
        rate = file.samplerate
    fixture = Fixture(
        "local_import",
        "real_music",
        data["split"],
        audio,
        rate,
        [(t, "unknown") for t in data["onsets"]],
    )
    fixture.import_metadata = {
        "provenance": data["provenance"],
        "license": data["license"],
        "annotation": data["annotation_method"],
        "uncertainty_seconds": data["uncertainty_seconds"],
        "source_recording_id": data["source_recording_id"],
        "original_file_sha256": file_hash,
    }
    return fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["baseline", "candidate"], default="baseline")
    parser.add_argument("--split", choices=["development", "holdout"], default="development")
    parser.add_argument("--output", required=True)
    parser.add_argument("--import-manifest")
    parser.add_argument(
        "--export-fixtures", help="Explicitly generate original synthetic WAVs and annotations"
    )
    args = parser.parse_args()
    fixtures = [load_import(args.import_manifest)] if args.import_manifest else generate(args.split)
    results = []
    for fixture in fixtures:
        if args.export_fixtures:
            folder = Path(args.export_fixtures)
            folder.mkdir(parents=True, exist_ok=True)
            sf.write(folder / (fixture.name + ".wav"), fixture.audio, fixture.rate, subtype="FLOAT")
            (folder / (fixture.name + ".json")).write_text(
                json.dumps(fixture.manifest(), indent=2), encoding="utf-8"
            )
        results.append(evaluate(fixture, args.detector))
    report = {
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "detector": args.detector,
        "baseline_source_commit": "6d30da28242b4a34b56a221cfc6513fa9cd62bc7",
        "detector_source_sha256": hashlib.sha256(
            Path(__file__).with_name("baseline_dsp.py").read_bytes()
            if args.detector == "baseline"
            else Path(__file__).parents[1].joinpath("dsp.py").read_bytes()
        ).hexdigest(),
        "corpus_version": CORPUS_VERSION,
        "split": args.split,
        "configuration": AnalysisConfig().model_dump(),
        "matching": "mir_eval.util.match_events one-to-one; mix references deduplicated",
        "aggregate": {
            str(ms): aggregate([r["onsets"][str(ms)] for r in results]) for ms in (25, 50)
        },
        "fixtures": results,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(path), "aggregate": report["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
