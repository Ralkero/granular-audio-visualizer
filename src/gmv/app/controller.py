"""What the application does, without any widgets: build a source, run an engine, replay.

The window owns one ``SessionController`` and polls it. Capture and analysis stay in the
engine's own threads; nothing here blocks the GUI thread for longer than a join on close.
"""

import os
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

from ..activity import ActivityModel
from ..engine import Engine
from ..instruments import analyzers
from ..models import AnalysisConfig
from ..sources.file import FileAudioSource
from ..sources.system_native import system_audio_source
from ..sources.windows import list_output_devices

SOURCES = (
    ("system", "Default system output"),
    ("device", "Specific output device"),
    ("application", "Application / process tree"),
    ("file", "Audio file"),
)
DEFAULT_EDGES = AnalysisConfig().band_edges


@dataclass(frozen=True)
class Settings:
    source: str = "system"
    device: str | None = None
    application: dict | None = None  # {"pid": ..., "name": ..., ...} from discovery
    application_label: str = ""
    file: str = ""
    activity_model: str = ""
    guitar_model: str = ""
    beat_tracker: bool = False
    pitch_tracking: bool = True
    logging: bool = True
    # Capture and analysis in a process of their own, so that drawing and analysis do
    # not wait for each other (gmv.remote). Off: both in this process, as before.
    analysis_process: bool = True
    # How readily each fast instrument analyzer reports, 0..1. Not a display setting.
    instrument_sensitivity: tuple[tuple[str, float], ...] = ()
    # Advanced
    delayed_analysis: bool = False
    debug_audio_seconds: int = 0
    window_ms: float = 46.44
    hop_ms: float = 20.0
    band_edges: tuple[float, ...] = DEFAULT_EDGES

    def changed(self, **values):
        return replace(self, **values)

    def source_label(self):
        if self.source == "device":
            return self.device or "No output device selected"
        if self.source == "application":
            return self.application_label or "No application selected"
        if self.source == "file":
            return Path(self.file).name if self.file else "No file selected"
        return "Default system output"


def discover():
    """Output devices and audio applications; safe to call from a worker thread."""
    devices, applications, error = [], [], ""
    try:
        devices = list_output_devices()
        try:
            from ..sources.application import list_audio_applications

            applications = list_audio_applications()
        except ImportError:
            pass
    except Exception as exc:
        error = str(exc)
    return devices, applications, error


def guitar_paths(path):
    """(frame weights, recognition model): the Guitar setting takes either kind of file.
    A ``.json`` file describes a context recognition model (``gmv.recognition``)."""
    path = (path or "").strip()
    if path.lower().endswith(".json"):
        return None, path
    return path or None, None


def analysis_config(settings):
    for label, path in (("Activity", settings.activity_model), ("Guitar", settings.guitar_model)):
        if path and not Path(path).is_file():
            raise ValueError(f"{label} weights file not found: {path}")
    guitar_weights, recognition = guitar_paths(settings.guitar_model)
    for path in (settings.activity_model, guitar_weights):
        if path:
            ActivityModel(path)  # fail here with a readable message
    # A recognition model that cannot be loaded does not stop the session: its worker
    # reports "unavailable" and everything else runs.
    return AnalysisConfig(
        window_ms=settings.window_ms,
        hop_ms=settings.hop_ms,
        band_edges=settings.band_edges,
        advanced=settings.delayed_analysis,
        experimental_beat_phase=settings.beat_tracker,
        activity_model_path=settings.activity_model or None,
        guitar_model_path=guitar_weights,
        recognition_model_path=recognition,
        pitch_tracking=settings.pitch_tracking,
        # Every instrument that has a fast analyzer is analysed, whether or not its layer
        # is shown: hiding a layer is a display choice.
        instrument_analyzers=tuple(analyzers()),
        instrument_sensitivity=settings.instrument_sensitivity,
        debug_audio_seconds=settings.debug_audio_seconds,
    )


def audio_source(settings):
    if settings.source == "file":
        if not Path(settings.file).is_file():
            raise ValueError("Choose an existing audio file in Settings")
        return FileAudioSource(settings.file, realtime=True)
    if settings.source == "application":
        from ..sources.application import ApplicationAudioSource

        if not settings.application:
            raise ValueError("No audio application selected. Refresh sources in Settings.")
        return ApplicationAudioSource(**settings.application)
    if settings.source == "device" and not settings.device:
        raise ValueError("No output device selected")
    return system_audio_source(settings.device if settings.source == "device" else None)


class SessionController:
    def __init__(self, session_root):
        self.session_root = Path(session_root)
        self.engine = None
        self.settings = None  # the settings the current or last engine was started with
        self.replaying = None

    # -- live analysis -----------------------------------------------------------------------

    @property
    def running(self):
        return self.engine is not None and not self.engine.done.is_set()

    @property
    def session_path(self):
        return self.engine.writer.path if self.engine and self.engine.writer else None

    def prepare(self, settings):
        """Build the engine without starting it, so a subscriber can attach first."""
        if self.running:
            raise RuntimeError("Analysis is already running")
        config = analysis_config(settings)
        source = audio_source(settings)
        self.stop_replay()
        if settings.analysis_process and os.environ.get("GMV_ANALYSIS", "").lower() != "thread":
            from ..remote import RemoteEngine

            # The source built above only proved that the settings describe one; the worker
            # builds its own from the same settings.
            self.engine = RemoteEngine(
                audio_source,
                (settings,),
                None,
                config,
                session_root=self.session_root,
                logging=settings.logging,
                source=source,
            )
        else:
            self.engine = Engine(
                source, config, session_root=self.session_root, logging=settings.logging
            )
        self.settings = settings
        return self.engine

    def stop(self):
        if self.engine is not None:
            self.engine.stop()

    def describe(self):
        return {
            "source": self.settings.source_label(),
            "beat_tracker": self.settings.beat_tracker,
            "activity_model": self.settings.activity_model or None,
            "guitar_model": guitar_paths(self.settings.guitar_model)[0],
            "recognition_model": guitar_paths(self.settings.guitar_model)[1],
            "instruments": tuple(analyzers()),
            "band_edges": tuple(self.settings.band_edges),
        }

    def set_sensitivity(self, instrument, value):
        """Change an analyzer's sensitivity while it runs; it follows at its next frame."""
        if self.engine is not None:
            self.engine.instrument_sensitivity[instrument] = min(1.0, max(0.0, float(value)))

    # -- replay ------------------------------------------------------------------------------

    def replay(self, path, speed=1.0):
        from ..visualizer.replay import ReplaySession

        if self.running:
            raise RuntimeError("Stop the running analysis before replaying a session")
        self.stop_replay()
        self.replaying = ReplaySession(path, speed)
        return self.replaying

    def stop_replay(self):
        if self.replaying is not None:
            self.replaying.stop()
            self.replaying = None

    # -- housekeeping --------------------------------------------------------------------------

    def export(self, target):
        """Zip the last session's logs, leaving out any debug audio."""
        path = self.session_path
        if path is None or self.running:
            raise RuntimeError("Stop analysis with logging enabled before exporting")
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for file in path.rglob("*"):
                if (
                    file.is_file()
                    and file.resolve() != Path(target).resolve()
                    and "audio" not in file.relative_to(path).parts
                ):
                    archive.write(file, file.relative_to(path))

    def shutdown(self, timeout=10):
        """Stop everything this controller started; returns False if the engine is stuck."""
        self.stop_replay()
        if self.engine is None:
            return True
        self.engine.stop()
        return self.engine.done.wait(timeout)

    def diagnostics(self):
        """Readable lines for the Diagnostics section."""
        if self.engine is None:
            return {"Capture": "Not started", "Analysis": "Not started"}
        engine, metrics = self.engine, self.engine.metrics()
        latest, dsp = engine.latest, metrics["dsp_ms"]
        age = metrics["estimated_capture_center_to_publish_ms"]
        resources = metrics["owned_process_resources"]
        lines = {
            "Capture": f"{engine.state.capitalize()} — {engine.detail}",
            "Audio": (
                f"{latest.sample_rate} Hz, {latest.channels} channels, {latest.timestamp_quality}"
                if latest
                else "No audio analysed yet"
            ),
            "Analysis": (
                f"{metrics['frames']} frames, {metrics['events']} events; "
                f"DSP p95 {dsp['p95']:.1f} ms of {engine.config.hop_ms:.0f} ms"
                if dsp
                else "Waiting for audio"
            ),
            "Latency": (
                f"capture to publication p95 {age['p95']:.0f} ms (estimate, not sound-to-screen)"
                if age
                else "Not measured for this source"
            ),
            "Dropped": (
                f"capture {metrics['dropped_blocks']} · native {metrics['native_dropped_packets']}"
                f" · logger {metrics['logger_dropped_messages']}"
                f" · discontinuities {metrics['discontinuities']}"
            ),
            "Process": (
                f"CPU {resources.get('cpu_one_core_percent', 0):.0f}% of one core · "
                f"memory {resources.get('rss_mb', 0):.0f} MiB"
            ),
        }
        if engine.config.advanced:
            lines["Delayed analysis"] = metrics["advanced_status"]
        timing = metrics["instrument_analysis_ms"]
        if timing:
            lines["Instrument analysis"] = (
                f"{', '.join(engine.config.instrument_analyzers)}: "
                f"p95 {timing['p95']:.2f} ms per frame, inside the analysis pass"
            )
        if engine.config.recognition_model_path:
            lines["Guitar recognition"] = (
                f"{metrics['recognition_status']} · estimates {metrics['recognition_jobs']}"
                f" · skipped {metrics['recognition_skipped_hops']}"
                f" · dropped blocks {metrics['recognition_drops']}"
            )
        if getattr(engine, "process_id", None):
            lines["Analysis process"] = (
                f"separate process {engine.process_id}; messages lost on the way: "
                f"{engine.forwarder_dropped}"
            )
        if engine.error:
            lines["Error"] = engine.error
        return lines
