"""Excerpt playback through PortAudio (PyAudioWPatch, already a project dependency).

Playback only reads the excerpt file. Samples beyond full scale are limited to +-1 in the
copy sent to the sound device (nothing is written). If the device refuses the file's
sample rate, a resampled copy is made for playback only and the window says so.
"""

from threading import Lock

import numpy as np
import soundfile as sf


class PortAudioOutput:
    """Opens callback streams on the default output device."""

    def __init__(self):
        import pyaudiowpatch as pa

        self.pa_module = pa
        self.pa = pa.PyAudio()

    def open(self, rate, channels, callback):
        pa = self.pa_module
        return self.pa.open(
            format=pa.paFloat32,
            channels=channels,
            rate=rate,
            output=True,
            frames_per_buffer=1024,
            stream_callback=callback,
            start=False,
        )

    def complete(self):
        return self.pa_module.paComplete

    def proceed(self):
        return self.pa_module.paContinue

    def close(self):
        self.pa.terminate()


class Player:
    def __init__(self, output=None):
        self.output = output
        self.audio = None
        self.rate = 0
        self.pos = 0
        self.stream = None
        self.lock = Lock()
        self.note = ""
        self.error = None

    def _out(self):
        if self.output is None:
            self.output = PortAudioOutput()
        return self.output

    def load(self, path):
        self.stop()
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        self.note, self.error = "", None
        if audio.shape[1] > 2:
            audio = audio[:, :2]
            self.note = "playing the first two channels"
        self.audio, self.rate, self.pos = np.clip(audio, -1.0, 1.0), rate, 0
        self._open()

    def _open(self):
        out = self._out()
        try:
            self.stream = out.open(self.rate, self.audio.shape[1], self._callback)
        except Exception as first:
            # the device does not take this rate: resample a playback copy to 48 kHz
            try:
                from scipy.signal import resample_poly

                from math import gcd

                g = gcd(48000, self.rate)
                copy = resample_poly(self.audio, 48000 // g, self.rate // g, axis=0).astype(
                    np.float32
                )
                self.pos = round(self.pos * 48000 / self.rate)
                self.audio, self.rate = np.clip(copy, -1, 1), 48000
                self.note = "device needed 48 kHz: playing a resampled copy (file unchanged)"
                self.stream = out.open(self.rate, self.audio.shape[1], self._callback)
            except Exception as second:
                self.stream, self.error = None, f"Cannot open audio output: {first}; {second}"

    def _callback(self, in_data, frame_count, time_info, status):
        with self.lock:
            chunk = self.audio[self.pos : self.pos + frame_count]
            self.pos += len(chunk)
            done = len(chunk) < frame_count
        if done:
            chunk = np.vstack(
                [chunk, np.zeros((frame_count - len(chunk), self.audio.shape[1]), np.float32)]
            )
        flag = self.output.complete() if done else self.output.proceed()
        return chunk.astype(np.float32).tobytes(), flag

    @property
    def duration_s(self):
        return len(self.audio) / self.rate if self.audio is not None else 0.0

    @property
    def position_s(self):
        return self.pos / self.rate if self.rate else 0.0

    @property
    def playing(self):
        try:
            return bool(self.stream and self.stream.is_active())
        except Exception:
            return False

    def play(self):
        if self.stream is None or self.audio is None:
            return
        if self.pos >= len(self.audio):
            self.pos = 0
        if self.playing:
            return
        try:
            self.stream.stop_stream()  # a finished stream must be stopped before restarting
        except Exception:
            pass
        self.stream.start_stream()

    def pause(self):
        if self.stream is not None and self.playing:
            self.stream.stop_stream()

    def toggle(self):
        self.pause() if self.playing else self.play()

    def replay(self):
        self.pause()
        with self.lock:
            self.pos = 0
        self.play()

    def seek(self, seconds):
        with self.lock:
            self.pos = int(max(0, min(len(self.audio), round(seconds * self.rate))))

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def close(self):
        self.stop()
        if self.output is not None and hasattr(self.output, "close"):
            self.output.close()
