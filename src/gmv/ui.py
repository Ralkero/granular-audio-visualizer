"""Former control center. The application is now one window: see ``gmv.app.window``.

Kept so that ``python -m gmv.ui``, the ``gmv-control`` command and older launchers open the
unified application.
"""

from PySide6.QtWidgets import QApplication  # noqa: F401  (imported by the desktop runtime check)

from .app.window import MainWindow, main

ControlCenter = MainWindow

if __name__ == "__main__":
    raise SystemExit(main())
