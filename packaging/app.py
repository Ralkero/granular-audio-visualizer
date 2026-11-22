"""Entry point of the packaged application (see ``Build App.ps1``).

``--self-check OUT.json`` runs a short session on a generated sound file without touching
saved settings and writes what happened, to confirm that a build is complete.
"""

import json
import os
import multiprocessing
import sys
import tempfile
from threading import Thread
from pathlib import Path
from time import perf_counter, sleep


def play_tone(seconds):
    """A quiet 440 Hz tone on the default output, so the capture check has something to hear."""
    import numpy as np
    import pyaudiowpatch as pa

    audio = pa.PyAudio()
    try:
        t = np.arange(int(48000 * seconds)) / 48000
        wave = (0.02 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        stream = audio.open(format=pa.paFloat32, channels=2, rate=48000, output=True)
        stream.write(np.repeat(wave[:, None], 2, axis=1).tobytes())
        stream.close()
    finally:
        audio.terminate()


def self_check(report):
    import numpy as np
    import soundfile as sf

    from gmv import paths
    from gmv.app.controller import discover
    from gmv.app.window import MainWindow, application

    app = application()
    result = {"packaged": paths.packaged(), "capture_helper": paths.capture_helper().is_file()}
    with tempfile.TemporaryDirectory() as folder:
        rate, seconds = 48000, 20
        t = np.arange(rate * seconds) / rate
        tone = 0.2 * np.sin(2 * np.pi * 196 * t) * (1 + np.sin(2 * np.pi * 2 * t)) / 2
        sf.write(Path(folder) / "tone.wav", np.stack([tone, tone], 1).astype(np.float32), rate)
        window = MainWindow(session_root=Path(folder), remember=False, discover_sources=False)
        drawer = window.drawer
        for name, path in paths.build_defaults().items():
            getattr(drawer, name).setText(path)
        result["guitar_model"] = Path(drawer.guitar_model.text()).name
        result["activity_model"] = Path(drawer.activity_model.text()).name
        drawer.source.setCurrentIndex(drawer.source.findData("file"))
        drawer.file.setText(str(Path(folder) / "tone.wav"))
        drawer.logging.setChecked(False)
        # GMV_SCREEN: part of the name of the display the check should open on.
        wanted = os.environ.get("GMV_SCREEN", "").lower()
        screen = next((s for s in app.screens() if wanted and wanted in s.name().lower()), None)
        if screen is not None:
            area = screen.geometry()
            window.setGeometry(
                area.x() + 40,
                area.y() + 40,
                min(1500, area.width() - 80),
                min(900, area.height() - 80),
            )
        window.show()
        window.start()
        result["problem_at_start"] = window.problem
        deadline, metrics = perf_counter() + 14, {}
        while perf_counter() < deadline and window.controller.engine:
            app.processEvents()
            metrics = window.controller.engine.metrics()
            sleep(0.01)
        result["recognition_status"] = metrics.get("recognition_status")
        result["recognition_estimates"] = metrics.get("recognition_jobs")
        result["recognition_ms"] = metrics.get("recognition_processing_ms")
        result["frames_drawn"] = len(window.view.canvas.paint_ms)
        result["engine_error"] = str(window.controller.engine.error or "")
        result["analysis_process"] = getattr(window.controller.engine, "process_id", None)
        # The unified view: both modes, every instrument off and on, every section.
        try:
            view = window.view
            for mode in ("radial", "bars"):
                view.set_mode(mode)
                for control in (*window.chips.values(), *window.section_buttons.values()):
                    control.click()
                    app.processEvents()
                    view.grab()
                    control.click()
                    app.processEvents()
            result["ui_exercise"] = "ok"
            result["layers_active"] = [layer.key for layer in view.canvas.active_layers()]
            timings = view.performance()
            result["paint_ms"] = timings.get("paint")
            result["frame_interval_ms"] = timings.get("frame interval")
        except Exception as exc:
            result["ui_exercise"] = f"failed: {exc}"
        # live capture of the default output while a quiet tone plays on it
        devices, applications, error = discover()
        result["output_devices"], result["audio_applications"] = len(devices), len(applications)
        result["discovery_error"] = error
        window.stop()
        deadline = perf_counter() + 10
        while window.controller.running and perf_counter() < deadline:
            app.processEvents()
            sleep(0.01)
        drawer.source.setCurrentIndex(drawer.source.findData("system"))
        window.start()
        result["live_problem_at_start"] = window.problem
        deadline, metrics, states = perf_counter() + 8, {}, []
        tone = Thread(target=play_tone, args=(6.0,), daemon=True)
        tone.start()
        while perf_counter() < deadline and window.controller.engine:
            app.processEvents()
            engine = window.controller.engine
            metrics = engine.metrics()
            state = f"{engine.state}: {engine.detail}"
            if not states or states[-1] != state:
                states.append(state)
            sleep(0.01)
        result["live_states"] = states[-6:]
        result["live_frames_analysed"] = metrics.get("frames")
        result["live_recognition_status"] = metrics.get("recognition_status")
        result["live_engine_error"] = str(window.controller.engine.error or "")
        window.close()
    Path(report).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) == 3 and sys.argv[1] == "--self-check":
        sys.exit(self_check(sys.argv[2]))
    from gmv.app.window import main

    sys.exit(main())
