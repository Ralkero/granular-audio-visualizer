"""Process-tree loopback via an isolated, bounded NAudio helper. No Spotify API."""

import struct
import subprocess
import sys
import os
from collections import deque
from pathlib import Path
from threading import Thread
from time import perf_counter
import psutil
from .base import AudioBlock, AudioSource


HEADER = struct.Struct("<4sIIIqqI")
HELPER = (
    Path(__file__).resolve().parents[3]
    / "native/ProcessLoopback/bin/Release/net9.0-windows/ProcessLoopback.exe"
)


def root_process(process):
    executable = process.exe()
    while True:
        try:
            parent = process.parent()
            same_executable = (
                parent is not None and parent.exe().casefold() == executable.casefold()
            )
        except (OSError, psutil.Error):
            return process
        if not same_executable:
            return process
        process = parent


def list_audio_applications():
    if sys.platform != "win32":
        return []
    import comtypes
    from pycaw.pycaw import AudioUtilities, IAudioSessionControl2
    from pycaw.utils import AudioSession

    comtypes.CoInitialize()
    result = {}
    try:
        for device in AudioUtilities.GetAllDevices(data_flow=0, device_state=1):
            try:
                sessions = device.AudioSessionManager.GetSessionEnumerator()
                for index in range(sessions.GetCount()):
                    session = AudioSession(
                        sessions.GetSession(index).QueryInterface(IAudioSessionControl2)
                    )
                    if session.ProcessId and session.ProcessId != os.getpid():
                        process = root_process(psutil.Process(session.ProcessId))
                        result[process.pid] = {
                            "pid": process.pid,
                            "name": process.name(),
                            "exe": process.exe(),
                            "created": process.create_time(),
                        }
            except (OSError, psutil.Error):
                continue  # Device/session disappeared during enumeration.
    finally:
        comtypes.CoUninitialize()
    return sorted(result.values(), key=lambda p: (p["name"].casefold(), p["pid"]))


def read_exact(stream, size):
    chunks = bytearray()
    while len(chunks) < size:
        part = stream.read(size - len(chunks))
        if not part:
            if not chunks:
                return None
            raise ValueError("Truncated process-capture packet")
        chunks.extend(part)
    return bytes(chunks)


def read_packet(stream):
    header = read_exact(stream, HEADER.size)
    if header is None:
        return None
    magic, size, rate, channels, sequence, qpc, flags = HEADER.unpack(header)
    if magic != b"GMV1" or rate != 48000 or channels != 2 or not 0 < size <= 1_048_576 or size % 8:
        raise ValueError("Invalid process-capture packet header")
    data = read_exact(stream, size)
    if data is None:
        raise ValueError("Missing process-capture packet payload")
    return data, rate, channels, sequence, qpc, flags


class ApplicationAudioSource(AudioSource):
    kind = "application_audio"

    def __init__(self, pid, name=None, exe=None, created=None):
        if sys.platform != "win32" or sys.getwindowsversion().build < 20348:
            raise RuntimeError(
                "Process loopback requires Windows build 20348+. Choose Default system output."
            )
        if not HELPER.is_file():
            raise RuntimeError(
                "Process helper is unavailable. Run Build Process Capture.ps1, "
                "or choose Default system output."
            )
        process = psutil.Process(pid)
        if created is not None and abs(process.create_time() - created) > 0.01:
            raise RuntimeError(
                "The selected process restarted. Refresh sources and select it again."
            )
        self.pid = pid
        self.name, self.exe = name or process.name(), exe or process.exe()
        self.created = process.create_time()
        self.native_drops = 0
        self.restarts = 0

    def describe(self):
        return {
            "kind": self.kind,
            "pid": self.pid,
            "process": self.name,
            "process_tree": True,
            "sample_rate": 48000,
            "channels": 2,
            "native_dropped_packets": self.native_drops,
            "restarts": self.restarts,
        }

    def _alive(self):
        try:
            return abs(psutil.Process(self.pid).create_time() - self.created) < 0.01
        except psutil.Error:
            return False

    def _find_restart(self):
        candidates = {}
        for process in psutil.process_iter(["exe"]):
            try:
                if process.info["exe"] and process.info["exe"].casefold() == self.exe.casefold():
                    root = root_process(process)
                    candidates[root.pid] = root
            except (OSError, psutil.Error):
                continue
        if len(candidates) == 1:
            process = next(iter(candidates.values()))
            self.pid, self.created = process.pid, process.create_time()
            self.restarts += 1
            return True
        return False

    def run(self, output, stop, status):
        origin, segment = getattr(self, "timeline_origin", perf_counter()), 0
        try:
            while not stop.is_set():
                if not self._alive():
                    status(
                        "waiting",
                        "Selected application closed. Waiting for an unambiguous restart of the same executable.",
                    )
                    while not stop.wait(1):
                        if self._find_restart():
                            break
                    if stop.is_set():
                        break
                segment += 1
                reader_errors, stderr = [], deque(maxlen=20)
                last_packet = [perf_counter()]
                helper = subprocess.Popen(
                    [str(HELPER), str(self.pid)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )

                def read_audio(
                    helper=helper,
                    segment=segment,
                    reader_errors=reader_errors,
                    last_packet=last_packet,
                ):
                    previous_sequence, next_timestamp = 0, None
                    try:
                        while not stop.is_set():
                            packet = read_packet(helper.stdout)
                            if packet is None:
                                break
                            data, rate, channels, sequence, qpc, flags = packet
                            now = perf_counter()
                            frames = len(data) // (4 * channels)
                            dropped = max(0, sequence - previous_sequence - 1)
                            self.native_drops += dropped
                            valid_qpc = qpc > 0 and not flags & 4 and abs(now - qpc / 1e7) < 10
                            mono = qpc / 1e7 if valid_qpc else now - frames / rate
                            timestamp = max(0, mono - origin)
                            gap = bool(
                                dropped
                                or flags & 1
                                or next_timestamp is None
                                or abs(timestamp - next_timestamp) > 2 / rate
                            )
                            if not gap:
                                timestamp = next_timestamp
                                mono = origin + timestamp
                            output.put(
                                AudioBlock(
                                    data,
                                    frames,
                                    rate,
                                    channels,
                                    timestamp,
                                    mono,
                                    now,
                                    segment,
                                    gap,
                                    "wasapi_qpc" if valid_qpc else "pipe_arrival_estimate",
                                )
                            )
                            previous_sequence = sequence
                            next_timestamp = timestamp + frames / rate
                            last_packet[0] = now
                    except Exception as exc:
                        reader_errors.append(str(exc))

                def read_status(helper=helper, stderr=stderr):
                    for line in iter(helper.stderr.readline, b""):
                        text = line.decode("utf-8", errors="replace").strip()
                        stderr.append(text)
                        if text.startswith("READY"):
                            status("capturing", f"{self.name} PID {self.pid}: {text}")

                reader = Thread(target=read_audio, name="process-pipe-reader", daemon=True)
                diagnostics = Thread(target=read_status, name="process-helper-status", daemon=True)
                reader.start()
                diagnostics.start()
                waiting = False
                try:
                    while not stop.wait(0.2):
                        if not self._alive():
                            break
                        if reader_errors:
                            raise RuntimeError(reader_errors[0])
                        if helper.poll() is not None:
                            diagnostics.join(timeout=1)
                            raise RuntimeError(
                                "Process capture stopped: "
                                + "; ".join(stderr)
                                + ". Choose Default system output as fallback."
                            )
                        quiet = perf_counter() - last_packet[0] > 1
                        if quiet != waiting:
                            status(
                                "waiting" if quiet else "capturing",
                                "No application audio packets; waiting for playback"
                                if quiet
                                else "Application audio resumed",
                            )
                            waiting = quiet
                finally:
                    try:
                        helper.stdin.write(b"stop\n")
                        helper.stdin.flush()
                    except (OSError, BrokenPipeError):
                        pass
                    try:
                        helper.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        helper.kill()  # Only our own child helper; never the selected application.
                        helper.wait(timeout=2)
                    reader.join(timeout=2)
                    diagnostics.join(timeout=2)
                    for pipe in (helper.stdin, helper.stdout, helper.stderr):
                        pipe.close()
        finally:
            output.close()
