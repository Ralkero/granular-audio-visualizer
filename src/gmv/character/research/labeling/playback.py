"""Playing one excerpt: exactly its seconds, from the original file, at the file's own
sample rate and channels.

On Linux the samples go to the default output through ``pacat``, the PulseAudio client that
PipeWire systems carry (GMV's own capture already uses its sibling ``parec``). It opens an
ordinary playback stream: no volume, device or routing is changed. Pausing ends the stream
and playing again starts a new one at the paused position, so a pause is immediate.

On Windows the existing PortAudio player of ``gmv review`` is used (PyAudioWPatch, which
exists only on Windows). It has not been tried with this tool.

Nothing here writes audio anywhere but to the sound server.
"""

import shutil
import subprocess
import sys
import threading
import time

import numpy as np

BLOCK_S = 0.02
LATENCY_MS = 80


def prepared(audio):
    """The excerpt as playback will send it: float32, at most two channels, held to +-1."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    return np.clip(audio[:, :2], -1.0, 1.0)


class PacatPlayer:
    """Plays an excerpt through ``pacat``. ``program`` is the command to run instead (tests
    give a stand-in that only records what it receives)."""

    def __init__(self, program=None):
        found = shutil.which("pacat")
        self.program = list(program) if program else ([found] if found else None)
        self.audio, self.rate = None, 0
        self.pos = 0  # the frame playback starts from when it starts again
        self.process = None
        self.thread = None
        self.lock = threading.Lock()
        self.started = None  # monotonic time of the first write of the current stream
        self.first = 0  # frame the current stream started at
        self.written = 0  # frames handed to the current stream
        self.done = False  # the current stream played to the end
        self.error = None
        self.note = ""

    # -- loading --------------------------------------------------------------------------

    def load(self, audio, rate):
        self.stop()
        source = np.asarray(audio)
        self.note = "playing the first two channels" if source.ndim == 2 and source.shape[1] > 2 else ""
        self.audio, self.rate, self.pos, self.done, self.error = prepared(audio), int(rate), 0, False, None

    def unload(self):
        self.stop()
        self.audio, self.rate, self.pos, self.done = None, 0, 0, False

    # -- the stream -----------------------------------------------------------------------

    def _command(self):
        channels = self.audio.shape[1]
        return self.program + [
            "--playback",
            "--raw",
            "--format=float32le",
            f"--rate={self.rate}",
            f"--channels={channels}",
            f"--latency-msec={LATENCY_MS}",
            "--client-name=GMV vocal labelling",
            "--stream-name=excerpt",
        ]

    def _feed(self, process, first):
        block = max(1, int(self.rate * BLOCK_S))
        position = first
        try:
            while position < len(self.audio):
                chunk = self.audio[position : position + block]
                process.stdin.write(chunk.tobytes())
                with self.lock:
                    if self.started is None:
                        self.started = time.monotonic()
                    position += len(chunk)
                    self.written = position - first
            process.stdin.close()
            code = process.wait()
            with self.lock:
                if code == 0 and self.process is process:
                    self.done = True
                elif code not in (0, None) and self.process is process:
                    self.error = self._failure(process)
        except (BrokenPipeError, OSError, ValueError):
            # Stopped on purpose (the process was ended), or the sound server refused.
            code = process.poll()
            with self.lock:
                if self.process is process and code not in (None, 0, -9, -15):
                    self.error = self._failure(process)

    @staticmethod
    def _failure(process):
        try:
            text = process.stderr.read().decode(errors="replace").strip().splitlines()
        except Exception:
            text = []
        detail = text[-1] if text else f"exit code {process.returncode}"
        return f"The sound output did not play ({detail})."

    @property
    def playing(self):
        return self.process is not None and self.thread is not None and self.thread.is_alive()

    def play(self):
        if self.audio is None or self.playing:
            return
        if self.program is None:
            self.error = "No sound output: the program 'pacat' was not found."
            return
        if self.pos >= len(self.audio):
            self.pos = 0
        self.error, self.done = None, False
        try:
            process = subprocess.Popen(
                self._command(), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        except OSError as exc:
            self.error = f"No sound output ({type(exc).__name__})."
            return
        with self.lock:
            self.process, self.first, self.written, self.started = process, self.pos, 0, None
        self.thread = threading.Thread(target=self._feed, args=(process, self.pos), name="excerpt-playback", daemon=True)
        self.thread.start()

    def _heard(self):
        """Frames of the current stream heard so far, estimated from the time since it was
        first fed, never beyond what was handed over."""
        with self.lock:
            if self.done:
                return len(self.audio) - self.first
            if self.started is None:
                return 0
            elapsed = int((time.monotonic() - self.started) * self.rate)
            return max(0, min(self.written, elapsed))

    def stop(self):
        """End the current stream; the position stays where playback had got to."""
        process, thread = self.process, self.thread
        if process is None:
            return
        heard = self._heard() if self.audio is not None else 0
        with self.lock:
            self.process = None
        try:
            process.kill()
        except OSError:
            pass
        if thread is not None:
            thread.join(timeout=2)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        for stream in (process.stdin, process.stderr):
            try:
                stream.close()
            except Exception:
                pass
        if self.audio is not None:
            self.pos = len(self.audio) if self.done else min(len(self.audio), self.first + heard)
        self.thread = None

    def pause(self):
        if self.playing:
            self.stop()

    def toggle(self):
        self.pause() if self.playing else self.play()

    def replay(self):
        self.stop()
        self.pos, self.done = 0, False
        self.play()

    def seek(self, seconds):
        was = self.playing
        self.stop()
        if self.audio is not None:
            self.pos = int(max(0, min(len(self.audio) - 1, round(seconds * self.rate))))
        self.done = False
        if was:
            self.play()

    @property
    def finished(self):
        """The excerpt was played to its end and nothing is playing now."""
        if self.playing:
            return False
        if self.process is not None and self.done:
            self.stop()
        return self.audio is not None and self.pos >= len(self.audio)

    @property
    def duration_s(self):
        return len(self.audio) / self.rate if self.audio is not None and self.rate else 0.0

    @property
    def position_s(self):
        if self.audio is None or not self.rate:
            return 0.0
        if self.process is not None:
            return min(len(self.audio), self.first + self._heard()) / self.rate
        return self.pos / self.rate

    def close(self):
        self.stop()


def windows_player():
    """The PortAudio player of ``gmv review``, taught to play an array (Windows only)."""
    from ....review.player import Player

    class ArrayPlayer(Player):
        def load(self, audio, rate):
            self.stop()
            self.note, self.error = "", None
            self.audio, self.rate, self.pos = prepared(audio), int(rate), 0
            self._open()

        def unload(self):
            self.stop()
            self.audio, self.rate, self.pos = None, 0, 0

        @property
        def finished(self):
            return self.audio is not None and not self.playing and self.pos >= len(self.audio)

    return ArrayPlayer()


def default_player():
    return windows_player() if sys.platform == "win32" else PacatPlayer()
