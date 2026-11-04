"""Read-only endpoint discovery and shared-mode WASAPI capture; no render stream."""

import sys
from time import perf_counter

from .base import AudioBlock, AudioSource


def list_output_devices():
    if sys.platform != "win32":
        return []
    import pyaudiowpatch as pa

    with pa.PyAudio() as audio:
        return [dict(d) for d in audio.get_loopback_device_info_generator()]


def endpoint_snapshot():
    from pycaw.pycaw import AudioUtilities

    devices = AudioUtilities.GetAllDevices(data_flow=0, device_state=1)
    return AudioUtilities.GetSpeakers().id, tuple(sorted(d.id for d in devices))


def selected_endpoint_id(loopback_name):
    from pycaw.pycaw import AudioUtilities

    name = loopback_name.removesuffix(" [Loopback]")
    matches = [
        d.id
        for d in AudioUtilities.GetAllDevices(data_flow=0, device_state=1)
        if d.FriendlyName == name
    ]
    if len(matches) != 1:
        raise RuntimeError("Selected output endpoint is absent or ambiguous")
    return matches[0]


class CaptureClock:
    """Use a sample clock within each segment, flag ADC jumps and callback outages.

    PortAudio ADC time is used to estimate sample age when valid. It is not a
    hardware/device presentation-latency measurement. Discontinuities start new segments.
    """

    def __init__(self, origin):
        self.origin = origin
        self.next_time = None
        self.last_received = None
        self.next_adc = None
        self.segment = 0
        self.callback_clock_error_ms = 0.0

    def stamp(self, frames, rate, info, now, flagged=False):
        adc, current = info.get("input_buffer_adc_time", 0), info.get("current_time", 0)
        valid = adc > 0 and current >= adc and current - adc < 2
        gap = (
            self.next_time is None
            or flagged
            or (self.last_received is not None and now - self.last_received > 0.5)
            or (valid and self.next_adc is not None and abs(adc - self.next_adc) > 0.05)
        )
        age = current - adc if valid else frames / rate
        if gap:
            self.segment += 1
            self.next_time = max(self.next_time or 0, now - self.origin - age, 0)
        timestamp = self.next_time
        self.callback_clock_error_ms = (now - self.origin - age - timestamp) * 1000
        self.next_time += frames / rate
        self.next_adc = adc + frames / rate if valid else None
        self.last_received = now
        return (
            timestamp,
            self.origin + timestamp,
            self.segment,
            gap,
            ("portaudio_adc_anchored_sample_clock" if valid else "callback_anchored_sample_clock")
            + ("_drift_warning" if abs(self.callback_clock_error_ms) > 100 else ""),
        )


class SystemAudioSource(AudioSource):
    kind = "system_audio"

    def __init__(self, device_name=None, *, block_ms=10):
        self.device_name, self.block_ms = device_name, block_ms
        self.recoveries = 0
        self.callback_flags = 0
        self.device_info = {}
        self.endpoint_id = None
        self.clock = None

    def describe(self):
        return {
            "kind": self.kind,
            "selection": self.device_name or "default",
            "device": self.device_info,
            "recoveries": self.recoveries,
            "callback_flags": self.callback_flags,
            "callback_clock_error_ms": self.clock.callback_clock_error_ms if self.clock else None,
        }

    def run(self, output, stop, status):
        if sys.platform != "win32":
            output.close()
            raise RuntimeError("Windows WASAPI required. PipeWire adapter is planned.")
        import comtypes
        import pyaudiowpatch as pa

        comtypes.CoInitialize()
        clock = CaptureClock(getattr(self, "timeline_origin", perf_counter()))
        self.clock = clock
        try:
            while not stop.is_set():
                try:
                    with pa.PyAudio() as audio:
                        if self.device_name:
                            selected_id = selected_endpoint_id(self.device_name)
                            if self.endpoint_id is not None and selected_id != self.endpoint_id:
                                raise RuntimeError(
                                    "A different endpoint has the selected name; explicitly select it to capture"
                                )
                            self.endpoint_id = selected_id
                            choices = [
                                d
                                for d in audio.get_loopback_device_info_generator()
                                if d["name"] == self.device_name
                            ]
                            if len(choices) != 1:
                                raise RuntimeError(
                                    "Selected output is disconnected or ambiguous; "
                                    "reconnect it or select another output"
                                )
                            device = choices[0]
                        else:
                            device = audio.get_default_wasapi_loopback()
                        self.device_info = dict(device)
                        rate, channels = (
                            int(device["defaultSampleRate"]),
                            device["maxInputChannels"],
                        )
                        if not 8000 <= rate <= 384000 or not 1 <= channels <= 32:
                            raise RuntimeError(
                                f"Unsupported device format: {rate} Hz / {channels} ch"
                            )
                        baseline = endpoint_snapshot()
                        clock.next_adc = None
                        clock.last_received = None
                        clock.next_time = None
                        last_packet = [perf_counter()]

                        def callback(
                            data,
                            frames,
                            info,
                            flags,
                            rate=rate,
                            channels=channels,
                            last_packet=last_packet,
                        ):
                            now = perf_counter()
                            last_packet[0] = now
                            if flags:
                                self.callback_flags += 1
                            ts, mono, segment, gap, quality = clock.stamp(
                                frames, rate, info, now, bool(flags)
                            )
                            output.put(
                                AudioBlock(
                                    data,
                                    frames,
                                    rate,
                                    channels,
                                    ts,
                                    mono,
                                    now,
                                    segment,
                                    gap,
                                    quality,
                                )
                            )
                            return (None, pa.paComplete if stop.is_set() else pa.paContinue)

                        stream = audio.open(
                            format=pa.paFloat32,
                            channels=channels,
                            rate=rate,
                            input=True,
                            input_device_index=device["index"],
                            frames_per_buffer=round(rate * self.block_ms / 1000),
                            stream_callback=callback,
                        )
                        try:
                            status(
                                "capturing", f"{device['name']} | {rate} Hz | {channels} channels"
                            )
                            quiet_reported = False
                            while not stop.wait(0.5):
                                current = endpoint_snapshot()
                                changed = (
                                    current != baseline
                                    if self.device_name is None
                                    else (current[1] != baseline[1])
                                )
                                if changed:
                                    raise RuntimeError(
                                        "Output device configuration changed; reopening"
                                    )
                                if not stream.is_active():
                                    raise RuntimeError("Capture stream stopped; reconnecting")
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
                            stream.close()
                except Exception as exc:
                    if stop.is_set():
                        break
                    self.recoveries += 1
                    status("recovering", str(exc))
                    stop.wait(1)
        finally:
            comtypes.CoUninitialize()
            output.close()
