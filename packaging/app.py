"""Entry point of the packaged application (see ``Build App.ps1``).

``--self-check OUT.json`` runs a short session on a generated sound file without touching
saved settings and writes what happened, to confirm that a build is complete.
"""

import json
import multiprocessing
import sys
import tempfile
from pathlib import Path
from time import perf_counter, sleep


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
        # live capture of the default output for a few seconds (silence is fine)
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
        deadline, metrics = perf_counter() + 6, {}
        while perf_counter() < deadline and window.controller.engine:
            app.processEvents()
            metrics = window.controller.engine.metrics()
            sleep(0.01)
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
