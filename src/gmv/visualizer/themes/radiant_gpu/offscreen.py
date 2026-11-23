"""An OpenGL context without a window, for tests and still pictures.

    with Offscreen() as plasma:
        image = plasma.image(values, 1920, 1080)

Reading the picture back is what this is for; the application's view never reads back.
Qt's ``offscreen`` platform has no OpenGL on Windows, so this needs the ordinary platform
(a ``QGuiApplication`` made without ``QT_QPA_PLATFORM=offscreen``). No window is shown.
"""

from PySide6.QtCore import QSize
from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLFramebufferObject

from .renderer import PlasmaRenderer
from .resources import ShaderError


def available():
    """(True, description) if an OpenGL 3.3 context can be made here, else (False, why)."""
    try:
        with Offscreen() as plasma:
            return True, plasma.renderer.info
    except ShaderError as error:
        return False, str(error)


class Offscreen:
    def __init__(self, folder=None):
        form = QSurfaceFormat()
        form.setVersion(3, 3)
        form.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        self.surface = QOffscreenSurface()
        self.surface.setFormat(form)
        self.surface.create()
        self.context = QOpenGLContext()
        self.context.setFormat(form)
        self.target = None
        if not self.context.create() or not self.context.makeCurrent(self.surface):
            raise ShaderError("no OpenGL 3.3 context could be made on this platform")
        self.renderer = PlasmaRenderer(folder) if folder else PlasmaRenderer()
        if not self.renderer.initialize():
            error = self.renderer.error
            self.close()
            raise ShaderError(error)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.context is not None and self.context.isValid():
            self.context.makeCurrent(self.surface)
            self.target = None
            self.renderer.release() if hasattr(self, "renderer") else None
            self.context.doneCurrent()
        self.context = None

    def image(self, values, width, height, only=None):
        """The frame as a QImage of ``width`` by ``height`` pixels."""
        self.context.makeCurrent(self.surface)
        if self.target is None or self.target.size() != QSize(width, height):
            self.target = QOpenGLFramebufferObject(QSize(width, height))
        self.renderer.render(values, width, height, self.target.handle(), only)
        self.context.functions().glFinish()
        return self.target.toImage()
