from time import perf_counter

import soundfile as sf

from .base import AudioBlock, AudioSource


class FileAudioSource(AudioSource):
    kind = "file_audio"

    def __init__(self, path, *, realtime=True, block_ms=10):
        self.path, self.live, self.block_ms = str(path), realtime, block_ms
        self.format_info = {}

    def describe(self):
        return {"kind": self.kind, "path": self.path, "realtime": self.live, **self.format_info}

    def run(self, output, stop, status):
        try:
            with sf.SoundFile(self.path) as file:
                rate, channels = file.samplerate, file.channels
                self.format_info = {
                    "sample_rate": rate,
                    "channels": channels,
                    "timestamp_quality": "file_sample_index",
                }
                if not 8000 <= rate <= 384000 or not 1 <= channels <= 32:
                    raise ValueError(f"Unsupported audio format: {rate} Hz / {channels} channels")
                status("capturing", f"File: {rate} Hz, {channels} channels")
                origin, position = perf_counter(), 0
                block_size = max(1, round(rate * self.block_ms / 1000))
                while not stop.is_set():
                    audio = file.read(block_size, dtype="float32", always_2d=True)
                    if not len(audio):
                        break
                    if self.live and stop.wait(
                        max(0, origin + (position + len(audio)) / rate - perf_counter())
                    ):
                        break
                    block = AudioBlock(
                        audio.tobytes(),
                        len(audio),
                        rate,
                        channels,
                        position / rate,
                        origin + position / rate,
                        perf_counter(),
                        timestamp_quality="file_sample_index",
                    )
                    if not output.put(block, wait=True, stop=stop):
                        break
                    position += len(audio)
                status("finished", "File analysis completed" if not stop.is_set() else "Stopped")
        finally:
            output.close()
