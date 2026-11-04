"""Explicit, duration-capped debug recording. Never enabled by default."""

import json
from threading import Thread
import numpy as np
import soundfile as sf
from .buffer import BoundedQueue


class DebugRecorder:
    def __init__(self, path, seconds):
        if not 0 < seconds <= 60:
            raise ValueError("Debug recording must be explicitly capped to 1–60 seconds")
        self.path, self.remaining = path / "debug" / "audio", seconds
        self.path.mkdir(parents=True)
        self.queue = BoundedQueue(32)
        self.error = None
        self.finished = False
        self.saved_seconds = 0.0
        self.thread = Thread(target=self._run, name="debug-audio-writer", daemon=True)
        self.thread.start()

    def offer(self, block):
        if not self.finished:
            self.queue.put(block)

    def _run(self):
        file, fmt, expected, segment = None, None, None, 0
        try:
            with (self.path / "timeline.jsonl").open("w", encoding="utf-8") as timeline:
                while self.remaining > 1e-6:
                    block = self.queue.get()
                    if block is None:
                        if self.queue.closed:
                            break
                        continue
                    current = (block.sample_rate, block.channels, block.segment)
                    if (
                        current != fmt
                        or block.discontinuity
                        or expected is None
                        or abs(block.timestamp - expected) > 2 / block.sample_rate
                    ):
                        if file:
                            file.close()
                        segment += 1
                        file = sf.SoundFile(
                            self.path / f"segment_{segment:04}.wav",
                            "w",
                            samplerate=block.sample_rate,
                            channels=block.channels,
                            subtype="FLOAT",
                        )
                    frames = min(block.frames, round(self.remaining * block.sample_rate))
                    if frames <= 0:
                        break
                    audio = np.frombuffer(block.data, dtype="<f4").reshape(
                        block.frames, block.channels
                    )
                    timeline.write(
                        json.dumps(
                            {
                                "file": f"segment_{segment:04}.wav",
                                "file_frame": file.tell(),
                                "frames": frames,
                                "timestamp": block.timestamp,
                                "sample_rate": block.sample_rate,
                            }
                        )
                        + "\n"
                    )
                    file.write(audio[:frames])
                    self.remaining -= frames / block.sample_rate
                    self.saved_seconds += frames / block.sample_rate
                    expected = block.timestamp + frames / block.sample_rate
                    fmt = current
        except Exception as exc:
            self.error = str(exc)
        finally:
            if file:
                file.close()
            self.finished = True
            self.queue.close()

    def close(self):
        self.queue.close()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            self.error = "Debug recorder did not stop within five seconds"
