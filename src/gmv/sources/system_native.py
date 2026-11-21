"""System-output loopback through the native WASAPI helper with per-packet QPC evidence.

Capture runs in the helper process, so a busy Python interpreter cannot starve the WASAPI
thread into silently discarding audio. Every packet carries the WASAPI QPC position; loss,
device discontinuities and clock disagreement become explicit timeline discontinuities
instead of unobserved drift. Read-only: no render stream, no routing or volume changes.

Endpoint loopback QPC values were observed about 21 ms ahead of packet delivery on the test
endpoint, independent of the capture buffer length. Whether that lead is the render
presentation time is unverified, so it is not claimed: the lead is measured once per device
open and subtracted, which keeps published evidence times at or before delivery (the
freshness contract rejects evidence from the future). The measured lead stays visible.
"""

import subprocess
from collections import deque
from statistics import median
from threading import Thread
from time import perf_counter

from .. import paths
from . import application, windows
from .base import AudioBlock, AudioSource


class QpcSampleClock:
    """Contiguous sample clock per run, checked against QPC on every packet.

    Packet QPC values jitter by about a millisecond, so they anchor a run rather than stamp
    each block. When QPC and the sample clock disagree by more than ``tolerance`` the timeline
    is re-anchored and the block is marked as a discontinuity. The signed disagreement stays
    visible in ``error_ms``; it is never clamped.
    """

    def __init__(self, origin, tolerance=0.010):
        self.origin, self.tolerance = origin, tolerance
        self.next_time = None
        self.error_ms = 0.0
        self.max_abs_error_ms = 0.0
        self.reanchors = 0

    def stamp(self, frames, rate, qpc, now, forced=False):
        valid = qpc is not None and qpc > 0 and abs(now - qpc) < 10
        observed = (qpc if valid else now - frames / rate) - self.origin
        gap = forced or self.next_time is None or abs(observed - self.next_time) > self.tolerance
        if gap:
            if self.next_time is not None:
                self.reanchors += 1
            # Never move the session timeline backwards over audio already published.
            timestamp = max(observed, self.next_time or 0.0, 0.0)
        else:
            timestamp = self.next_time
        self.error_ms = (observed - timestamp) * 1000
        self.max_abs_error_ms = max(self.max_abs_error_ms, abs(self.error_ms))
        self.next_time = timestamp + frames / rate
        quality = "wasapi_qpc_sample_clock" if valid else "pipe_arrival_estimate"
        return timestamp, self.origin + timestamp, gap, quality


class NativeSystemAudioSource(AudioSource):
    kind = "system_audio"
    settle_packets = 20  # measured, not published: about 200 ms once per device open
    lead_margin = 0.003  # covers the observed ~1 ms delivery/QPC jitter

    def __init__(self, device_name=None):
        self.device_name = device_name
        self.recoveries = 0
        self.native_drops = 0
        self.device_discontinuities = 0
        self.endpoint_id = None
        self.format = None
        self.clock = None
        self.qpc_lead = None
        self.settle_discarded_frames = 0

    def describe(self):
        clock = self.clock
        return {
            "kind": self.kind,
            "backend": "native_wasapi_loopback",
            "selection": self.device_name or "default",
            "format": self.format,
            "recoveries": self.recoveries,
            "native_dropped_packets": self.native_drops,
            "device_discontinuities": self.device_discontinuities,
            "clock_reanchors": clock.reanchors if clock else None,
            "qpc_clock_error_ms": clock.error_ms if clock else None,
            "qpc_clock_max_abs_error_ms": clock.max_abs_error_ms if clock else None,
            "qpc_lead_ms": None if self.qpc_lead is None else self.qpc_lead * 1000,
            "settle_discarded_frames": self.settle_discarded_frames,
        }

    def _read_audio(self, stream, segment, clock, output, stop, errors, last_packet):
        previous_sequence, settle, leads = 0, self.settle_packets, []
        self.qpc_lead = None
        try:
            while not stop.is_set():
                packet = application.read_packet(stream, fixed_format=False)
                if packet is None:
                    break
                data, rate, channels, sequence, qpc, flags = packet
                now = perf_counter()
                frames = len(data) // (4 * channels)
                qpc = None if flags & 4 else qpc / 1e7
                if settle > 0:
                    # Stream start delivers primed packets early; measure the lead, publish later.
                    settle -= 1
                    self.settle_discarded_frames += frames
                    if qpc is not None and abs(now - qpc) < 10:
                        leads.append(qpc + frames / rate + self.lead_margin - now)
                    last_packet[0] = now
                    continue
                if self.qpc_lead is None:
                    self.qpc_lead = max(0.0, median(leads)) if leads else 0.0
                dropped = max(0, sequence - previous_sequence - 1) if previous_sequence else 0
                self.native_drops += dropped
                if flags & 1 and previous_sequence:
                    self.device_discontinuities += 1
                timestamp, mono, gap, quality = clock.stamp(
                    frames,
                    rate,
                    None if qpc is None else qpc - self.qpc_lead,
                    now,
                    bool(dropped or flags & 1 or [rate, channels] != self.format),
                )
                self.format = [rate, channels]
                output.put(
                    AudioBlock(
                        data, frames, rate, channels, timestamp, mono, now, segment, gap, quality
                    )
                )
                previous_sequence = sequence
                last_packet[0] = now
        except Exception as exc:
            errors.append(str(exc))

    def run(self, output, stop, status):
        import comtypes

        comtypes.CoInitialize()
        clock = self.clock = QpcSampleClock(getattr(self, "timeline_origin", perf_counter()))
        segment = 0
        try:
            while not stop.is_set():
                try:
                    if self.device_name:
                        selected = windows.selected_endpoint_id(self.device_name)
                        if self.endpoint_id is not None and selected != self.endpoint_id:
                            raise RuntimeError(
                                "A different endpoint has the selected name; "
                                "explicitly select it to capture"
                            )
                        self.endpoint_id = selected
                    baseline = windows.endpoint_snapshot()
                    target = self.endpoint_id if self.device_name else baseline[0]
                    segment += 1
                    errors, stderr, last_packet = [], deque(maxlen=20), [perf_counter()]
                    helper = subprocess.Popen(
                        [str(application.HELPER), "system", target],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                        env=paths.helper_environment(),
                    )

                    def read_status(helper=helper, stderr=stderr):
                        for line in iter(helper.stderr.readline, b""):
                            text = line.decode("utf-8", errors="replace").strip()
                            stderr.append(text)
                            if text.startswith("READY"):
                                name = self.device_name or "Default output"
                                status("capturing", f"{name}: {text}")

                    workers = [
                        Thread(
                            target=self._read_audio,
                            args=(helper.stdout, segment, clock, output, stop, errors, last_packet),
                            name="system-pipe-reader",
                            daemon=True,
                        ),
                        Thread(target=read_status, name="system-helper-status", daemon=True),
                    ]
                    for worker in workers:
                        worker.start()
                    quiet_reported = False
                    try:
                        while not stop.wait(0.5):
                            if errors:
                                raise RuntimeError(errors[0])
                            if helper.poll() is not None:
                                workers[1].join(timeout=1)
                                raise RuntimeError(
                                    "System capture stopped: " + "; ".join(stderr)[:300]
                                )
                            current = windows.endpoint_snapshot()
                            changed = (
                                current != baseline
                                if self.device_name is None
                                else current[1] != baseline[1]
                            )
                            if changed:
                                raise RuntimeError("Output device configuration changed; reopening")
                            quiet = perf_counter() - last_packet[0] > 1
                            if quiet != quiet_reported:
                                status(
                                    "waiting" if quiet else "capturing",
                                    "No audio packets; waiting for playback"
                                    if quiet
                                    else "Audio packets resumed",
                                )
                                quiet_reported = quiet
                    finally:
                        try:
                            helper.stdin.write(b"stop\n")
                            helper.stdin.flush()
                        except (OSError, BrokenPipeError):
                            pass
                        try:
                            helper.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            helper.kill()  # Only our own child helper.
                            helper.wait(timeout=2)
                        for worker in workers:
                            worker.join(timeout=2)
                        for pipe in (helper.stdin, helper.stdout, helper.stderr):
                            try:
                                pipe.close()
                            except OSError:  # unsent "stop" to a helper that has gone
                                pass
                except Exception as exc:
                    if stop.is_set():
                        break
                    self.recoveries += 1
                    status("recovering", str(exc))
                    stop.wait(1)
        finally:
            comtypes.CoUninitialize()
            output.close()


def system_audio_source(device_name=None, backend="auto"):
    """Native QPC-checked capture when the helper is built; PortAudio fallback otherwise."""
    if backend not in ("auto", "native", "portaudio"):
        raise ValueError("backend must be auto, native or portaudio")
    if backend == "native" and not application.HELPER.is_file():
        raise RuntimeError("Native helper is unavailable. Run Build Process Capture.ps1.")
    if backend != "portaudio" and application.HELPER.is_file():
        return NativeSystemAudioSource(device_name)
    return windows.SystemAudioSource(device_name)
