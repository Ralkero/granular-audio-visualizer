"""The effects layer as a widget: a ``QOpenGLWidget`` behind the canvas.

The canvas stays what it was (a CPU-painted widget). While this view is shown the canvas
leaves its background clear, and Qt composes the two: this view's picture below, the
canvas's geometry and text above. Nothing is read back from the card.

Qt owns the context. ``initializeGL`` is called whenever there is a new one (the first
time, and again if the window's context is replaced), and compiles everything there;
``paintGL`` draws into the framebuffer Qt hands out, never framebuffer zero. If setting up
fails the view says so once (``failed``) and from then on only clears itself: the
application carries on with the phase 1 drawing.
"""

from PySide6.QtCore import QTimer, Signal
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .renderer import GL_COLOR_BUFFER_BIT, PlasmaRenderer


class PlasmaView(QOpenGLWidget):
    failed = Signal(str)

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.source = source  # () -> the frame's values (``frame.values``) or None
        self.renderer = PlasmaRenderer()
        self.problem = ""

    def initializeGL(self):
        if not self.renderer.initialize():
            self.problem = self.renderer.error
            # Not from inside Qt's paint: the window rearranges itself in answer.
            QTimer.singleShot(0, lambda: self.failed.emit(self.problem))

    def paintGL(self):
        values = None
        if self.renderer.ready:
            try:
                values = self.source()
            except Exception as error:  # noqa: BLE001  never let a frame's numbers stop painting
                self.problem = f"{type(error).__name__}: {error}"
        if values is not None:
            ratio = self.devicePixelRatioF()
            width, height = round(self.width() * ratio), round(self.height() * ratio)
            try:
                if self.renderer.render(values, width, height, self.defaultFramebufferObject()):
                    return
            except Exception as error:  # noqa: BLE001  a driver failure mid-run: fall back
                self.problem = f"{type(error).__name__}: {error}"
                self.renderer.ready = False
                QTimer.singleShot(0, lambda: self.failed.emit(self.problem))
        gl = self.context().functions()
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(GL_COLOR_BUFFER_BIT)

    def shutdown(self):
        """Give back everything made in the context, with the context current, while both
        still exist. Called by the window when it closes: left to the order in which
        Python and Qt happen to destroy things at exit, the buffers can outlive the
        context they belong to."""
        if self.context() is not None and self.context().isValid():
            self.makeCurrent()
            self.renderer.release()
            self.doneCurrent()
        else:
            self.renderer.release()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
