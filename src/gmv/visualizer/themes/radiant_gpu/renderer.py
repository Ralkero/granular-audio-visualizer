"""The OpenGL side: shader programs, framebuffers and the passes of one frame.

It draws into whatever framebuffer it is handed, in whatever context is current, so the
view (``view.PlasmaView``) and the offscreen renders (``offscreen``) use the same code.

One frame:

1. the scene, at ``SCENE_SCALE`` of the target's size, into a floating-point buffer
   (brightness is not capped at 1): the plasma field, then rings, filaments and core
   added to it;
2. what is bright in it, into a buffer of half that size, blurred across and down: the
   tight bloom;
3. that again at half its size, blurred: the wide bloom;
4. the composite at the target's full size: background, scene and both blooms, brought
   down to what a display shows.

Programs are compiled once. Buffers are made once per size and reused; nothing is
allocated per frame and nothing is read back. Everything is released when the context
goes (``release``), and built again if a new one comes.
"""

from time import perf_counter

from PySide6.QtCore import QSize
from PySide6.QtGui import QOpenGLContext, QVector2D, QVector3D, QVector4D
from PySide6.QtOpenGL import (
    QOpenGLFramebufferObject,
    QOpenGLFramebufferObjectFormat,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLTimeMonitor,
    QOpenGLVertexArrayObject,
)

from . import POST, SHADERS, resources
from .resources import ShaderError

SCENE_SCALE = 0.7  # the scene's size as a share of the target's (measured: see the docs)
BLOOM_THRESHOLD, BLOOM_KNEE = 0.55, 0.9
# OpenGL names used here (the functions object of a context has no constants).
GL_TRIANGLES, GL_BLEND, GL_ONE = 0x0004, 0x0BE2, 1
GL_DEPTH_TEST, GL_SCISSOR_TEST, GL_STENCIL_TEST, GL_CULL_FACE = 0x0B71, 0x0C11, 0x0B90, 0x0B44
GL_FRAMEBUFFER, GL_COLOR_BUFFER_BIT = 0x8D40, 0x4000
GL_TEXTURE_2D, GL_TEXTURE0, GL_LINEAR, GL_CLAMP_TO_EDGE = 0x0DE1, 0x84C0, 0x2601, 0x812F
GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER = 0x2801, 0x2800
GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T = 0x2802, 0x2803
GL_RGBA16F, GL_RGBA8 = 0x881A, 0x8058
GL_VENDOR, GL_RENDERER, GL_VERSION, GL_SHADING_LANGUAGE_VERSION = 0x1F00, 0x1F01, 0x1F02, 0x8B8C


class PlasmaRenderer:
    def __init__(self, folder=resources.FOLDER):
        self.folder = folder
        self.ready = False
        self.error = ""  # why it is not ready, when it tried and failed
        self.info = {}  # what the context said about itself
        self.scale = SCENE_SCALE
        self.gl = self.context = None
        self.programs, self.locations = {}, {}
        self.buffers, self.size = {}, None
        self.vao = self.monitor = None
        self._timing = False
        self.cpu_ms = self.gpu_ms = None  # of the last frame measured
        self.frames = 0

    # -- resources -----------------------------------------------------------------------------

    def initialize(self):
        """Compile everything in the current context. Returns whether the renderer can be
        used; if not, ``error`` says why and nothing is left half-made."""
        self.release()
        self.error = ""
        try:
            context = QOpenGLContext.currentContext()
            if context is None:
                raise ShaderError("no OpenGL context is current")
            self.context, self.gl = context, context.functions()
            names = (("vendor", GL_VENDOR), ("renderer", GL_RENDERER), ("version", GL_VERSION),
                     ("glsl", GL_SHADING_LANGUAGE_VERSION))  # fmt: skip
            self.info = {name: str(self.gl.glGetString(code) or "") for name, code in names}
            vertex = resources.vertex(self.folder)
            for name in SHADERS + POST:
                self.programs[name] = self._program(name, vertex, resources.fragment(name, self.folder))
                self.locations[name] = {}
            self.vao = QOpenGLVertexArrayObject()
            self.vao.create()
            context.aboutToBeDestroyed.connect(self._context_going)
            self._watching = True
            self.ready = True
        except ShaderError as error:
            self.error = str(error)
            self.release()
        except Exception as error:  # noqa: BLE001  a driver or binding failure: fall back
            self.error = f"OpenGL set-up failed: {type(error).__name__}: {error}"
            self.release()
        return self.ready

    def _program(self, name, vertex, fragment):
        program = QOpenGLShaderProgram()
        for kind, text, what in (
            (QOpenGLShader.ShaderTypeBit.Vertex, vertex, "vertex shader"),
            (QOpenGLShader.ShaderTypeBit.Fragment, fragment, f"{name}.frag"),
        ):
            if not program.addShaderFromSourceCode(kind, text):
                raise ShaderError(f"{what} did not compile: {program.log().strip()}")
        if not program.link():
            raise ShaderError(f"{name} did not link: {program.log().strip()}")
        return program

    def _context_going(self):
        if self.context is not None and QOpenGLContext.currentContext() is not self.context:
            return  # cannot release without it current; Qt frees what belongs to the context
        self.release()

    def release(self):
        """Give back everything made in the context (which must be current, or gone)."""
        self.buffers, self.size = {}, None  # freed as they are dropped
        self.programs, self.locations = {}, {}
        if self.vao is not None:
            self.vao.destroy()
        self.vao = self.monitor = None
        self._timing = False
        if self.context is not None and getattr(self, "_watching", False):
            self._watching = False
            try:
                self.context.aboutToBeDestroyed.disconnect(self._context_going)
            except (RuntimeError, TypeError):
                pass
        self.context = self.gl = None
        self.ready = False

    def _buffer(self, name, width, height, internal):
        form = QOpenGLFramebufferObjectFormat()
        form.setInternalTextureFormat(internal)
        buffer = QOpenGLFramebufferObject(QSize(max(1, width), max(1, height)), form)
        if not buffer.isValid():
            raise ShaderError(f"framebuffer {name} ({width}x{height}) could not be made")
        gl = self.gl
        gl.glBindTexture(GL_TEXTURE_2D, buffer.texture())
        for parameter, value in (
            (GL_TEXTURE_MIN_FILTER, GL_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
            (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
        ):  # fmt: skip
            gl.glTexParameteri(GL_TEXTURE_2D, parameter, value)
        gl.glBindTexture(GL_TEXTURE_2D, 0)
        return buffer

    def resize(self, width, height):
        """Buffers for a target of this size in pixels; only remade when it changes."""
        key = (width, height, self.scale)
        if key == self.size:
            return
        self.buffers, self.size = {}, None  # the old ones are freed as they are dropped
        scene = (max(2, round(width * self.scale)), max(2, round(height * self.scale)))
        sizes = {
            "scene": scene,
            "tight a": (scene[0] // 2, scene[1] // 2), "tight b": (scene[0] // 2, scene[1] // 2),
            "wide a": (scene[0] // 4, scene[1] // 4), "wide b": (scene[0] // 4, scene[1] // 4),
        }  # fmt: skip
        self.buffers = {name: self._buffer(name, w, h, GL_RGBA16F) for name, (w, h) in sizes.items()}
        self.size = key

    def memory(self):
        """Bytes of video memory the buffers take (eight bytes a pixel)."""
        return sum(b.width() * b.height() * 8 for b in self.buffers.values())

    # -- drawing -------------------------------------------------------------------------------

    def _set(self, name, values):
        program, locations = self.programs[name], self.locations[name]

        def place(uniform):
            found = locations.get(uniform)
            if found is None:
                found = locations[uniform] = program.uniformLocation(uniform)
            return found

        def put(uniform, value):
            location = place(uniform)
            if location < 0:
                return
            if isinstance(value, bool | int):
                program.setUniformValue1i(location, int(value))
            elif isinstance(value, float):
                program.setUniformValue1f(location, value)
            elif len(value) == 2:
                program.setUniformValue(location, QVector2D(*value))
            elif len(value) == 3:
                program.setUniformValue(location, QVector3D(*value))
            else:
                program.setUniformValue(location, QVector4D(*value))

        for uniform, value in values.items():
            if isinstance(value, list):
                if place(uniform + "[0]") < 0:
                    continue
                for index, item in enumerate(value):
                    put(f"{uniform}[{index}]", item)
            else:
                put(uniform, value)

    def _pass(self, name, target, width, height, values, textures=(), add=False):
        gl = self.gl
        gl.glBindFramebuffer(GL_FRAMEBUFFER, target)
        gl.glViewport(0, 0, width, height)
        if add:
            gl.glEnable(GL_BLEND)
            gl.glBlendFunc(GL_ONE, GL_ONE)
        else:
            gl.glDisable(GL_BLEND)
        program = self.programs[name]
        program.bind()
        for unit, (uniform, texture) in enumerate(textures):
            gl.glActiveTexture(GL_TEXTURE0 + unit)
            gl.glBindTexture(GL_TEXTURE_2D, texture)
            values = values | {uniform: unit}
        self._set(name, values)
        gl.glDrawArrays(GL_TRIANGLES, 0, 3)
        program.release()

    def render(self, values, width, height, target, only=None):
        """Draw the frame described by ``values`` (``frame.values``) into framebuffer
        ``target``, ``width`` by ``height`` pixels. ``only``: scene passes to draw, for
        looking at one effect alone."""
        if not self.ready:
            return False
        began = perf_counter()
        gl = self.gl
        self.resize(width, height)
        self._time_begin()
        for state in (GL_DEPTH_TEST, GL_SCISSOR_TEST, GL_STENCIL_TEST, GL_CULL_FACE):
            gl.glDisable(state)
        self.vao.bind()
        buffers = self.buffers
        scene = buffers["scene"]
        gl.glBindFramebuffer(GL_FRAMEBUFFER, scene.handle())
        gl.glViewport(0, 0, scene.width(), scene.height())
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(GL_COLOR_BUFFER_BIT)
        for name in SHADERS:
            if only is None or name in only:
                self._pass(name, scene.handle(), scene.width(), scene.height(), values, add=True)
        source = scene
        for a, b in (("tight a", "tight b"), ("wide a", "wide b")):
            first, second = buffers[a], buffers[b]
            if a == "tight a":
                self._pass(
                    "bloom_extract", first.handle(), first.width(), first.height(),
                    {"u_threshold": BLOOM_THRESHOLD, "u_knee": BLOOM_KNEE}, (("u_source", source.texture()),),
                )  # fmt: skip
            else:  # the wide bloom starts from the tight one, at half its size
                self._pass(
                    "blur", first.handle(), first.width(), first.height(),
                    {"u_step": (0.0, 0.0)}, (("u_source", source.texture()),),
                )  # fmt: skip
            self._pass(
                "blur", second.handle(), second.width(), second.height(),
                {"u_step": (1.0 / first.width(), 0.0)}, (("u_source", first.texture()),),
            )  # fmt: skip
            self._pass(
                "blur", first.handle(), first.width(), first.height(),
                {"u_step": (0.0, 1.0 / second.height())}, (("u_source", second.texture()),),
            )  # fmt: skip
            source = first
        self._pass(
            "composite", target, width, height, values,
            (("u_scene", scene.texture()), ("u_bloom_tight", buffers["tight a"].texture()),
             ("u_bloom_wide", buffers["wide a"].texture())),
        )  # fmt: skip
        for unit in (2, 1, 0):
            gl.glActiveTexture(GL_TEXTURE0 + unit)
            gl.glBindTexture(GL_TEXTURE_2D, 0)
        self.vao.release()
        gl.glDisable(GL_BLEND)
        self._time_end()
        self.cpu_ms = (perf_counter() - began) * 1000.0
        self.frames += 1
        return True

    # -- how long the card takes ---------------------------------------------------------------

    def _time_begin(self):
        """Start timing this frame on the card, if the last timing has been collected. The
        result is asked for later and never waited for, so measuring does not hold a frame
        up; not every frame is measured."""
        try:
            if self.monitor is None:
                self.monitor = QOpenGLTimeMonitor()
                self.monitor.setSampleCount(2)
                if not self.monitor.create():
                    self.monitor = False
            if self.monitor is False:
                return
            if self._timing:
                if not self.monitor.isResultAvailable():
                    self._began = False
                    return
                intervals = self.monitor.waitForIntervals()
                if intervals:
                    self.gpu_ms = intervals[0] / 1e6
                self.monitor.reset()
                self._timing = False
            self.monitor.recordSample()
            self._began = True
        except Exception:  # noqa: BLE001  timing is optional: never let it stop a frame
            self.monitor, self._began = False, False

    def _time_end(self):
        if self.monitor and getattr(self, "_began", False):
            try:
                self.monitor.recordSample()
                self._timing = True
            except Exception:  # noqa: BLE001
                self.monitor = False
