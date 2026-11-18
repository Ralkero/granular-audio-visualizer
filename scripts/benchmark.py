"""Reproducible synthetic CPU benchmark; no audio is played or captured."""

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter
import numpy as np
import soundfile as sf
from gmv.engine import Engine
from gmv.models import AnalysisConfig
from gmv.sources.file import FileAudioSource


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--output", default="local-evidence/synthetic-benchmark.json")
    parser.add_argument("--recognition", help="recognition model description (.json) to run")
    parser.add_argument("--activity", help="drums / bass / vocals weights to run as well")
    parser.add_argument("--pitch", action="store_true", help="note candidates on")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="feed the file at playback speed: load, result age and queues as in live use",
    )
    args = parser.parse_args()
    if not 10 <= args.seconds <= 600:
        parser.error("Use a duration between 10 and 600 seconds")
    rate = 48000
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "synthetic.wav"
        # Streaming fixture generation keeps even benchmark input memory bounded.
        with sf.SoundFile(path, "w", samplerate=rate, channels=2, subtype="FLOAT") as file:
            for second in range(args.seconds):
                time = np.arange(rate) / rate
                tone = 0.12 * np.sin(2 * np.pi * 440 * time)
                for start in (0, rate // 2):
                    tone[start : start + 2400] += (
                        0.4 * np.sin(2 * np.pi * 90 * time[:2400]) * np.exp(-np.arange(2400) / 500)
                    )
                if second % 10 < 2:
                    tone *= 0
                file.write(np.column_stack((tone, -tone)))
        began = perf_counter()
        config = AnalysisConfig(
            recognition_model_path=args.recognition,
            activity_model_path=args.activity,
            pitch_tracking=args.pitch,
        )
        source = FileAudioSource(path, realtime=args.realtime)
        engine = Engine(source, config, logging=False)
        engine.start()
        if not engine.done.wait(120 + (args.seconds if args.realtime else 0)):
            engine.stop()
            raise RuntimeError("Benchmark timed out")
        elapsed = perf_counter() - began
        report = {
            "input": "synthetic tone, silence and 120 BPM bass pulses; opposite-phase stereo",
            "audio_seconds": args.seconds,
            "sample_rate": rate,
            "wall_seconds": elapsed,
            "realtime_factor": elapsed / args.seconds,
            "fed_at_playback_speed": args.realtime,
            "recognition_model": args.recognition,
            "performance": engine.metrics(),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        if engine.error:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
