"""Catch missing Qt runtime DLLs that core-only tests cannot detect."""

import os
import subprocess
import sys


def test_qt_runtime_and_control_center_import():
    environment = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from gmv.ui import QApplication, ControlCenter; "
            "app=QApplication([]); assert ControlCenter; app.quit()",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
