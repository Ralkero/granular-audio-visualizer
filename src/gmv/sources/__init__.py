"""Platform adapters implement AudioSource without importing the UI or DSP.

Which adapters a system has is ``gmv.host``'s answer. The functions below are how the
application reaches the capture adapters: it imports no capture module itself, so a system
without one never loads it, and asking for a source the system does not have fails with a
sentence for the user instead of a missing library. ``sources.file`` works everywhere and is
imported directly.

A new backend (PipeWire, for instance) is added by listing its sources in
``host.LIVE_CAPTURE`` and building it here. There is no stand-in backend for a system
without capture: nothing here ever pretends to capture.
"""

from .. import host


class Unavailable(ValueError):
    """An audio source this system does not have."""


def require(key):
    """Raise ``Unavailable`` unless this system has the audio source ``key``."""
    if key in host.audio_sources():
        return
    if not host.live_capture():
        raise Unavailable(
            f"{host.capture_note()} Choose an audio file, or replay a recorded session."
        )
    raise Unavailable(f"This audio source is not available on {host.name()}: {key}")


def system_audio_source(device_name=None, backend="auto"):
    """Everything the system plays, or what one output device plays."""
    require("device" if device_name else "system")
    from .system_native import system_audio_source as build

    return build(device_name, backend)


def application_audio_source(**selection):
    """One application and its child processes."""
    require("application")
    from .application import ApplicationAudioSource

    return ApplicationAudioSource(**selection)


def list_output_devices():
    if "device" not in host.audio_sources():
        return []
    from .windows import list_output_devices as found

    return found()


def list_audio_applications():
    if "application" not in host.audio_sources():
        return []
    from .application import list_audio_applications as found

    return found()
