"""Radiant's effects drawn by the graphics card: plasma, energy rings, filaments, a core
with depth, and bloom.

This is Radiant: a complete renderer of its own. While it is in use it draws everything in
the central view (atmosphere, formations, the spectrum, the four instruments, filaments,
the core, bloom) with GLSL shaders (``shaders/``), and the canvas draws none of it. It
shares the detection core's visual state with the other renderers and nothing else: no
Classic geometry, no QPainter instrument layer. The QPainter Radiant of phase 1 is only
the reduced picture shown when the graphics card cannot draw this one.

* ``scene``     the composition (where each structure is) and what each instrument is
                doing, from the live states. Numbers only.
* ``frame``     what one frame's shaders are told: a plain dictionary of numbers, worked
                out from the visual state, the layout, the palette and audio time. No Qt,
                no OpenGL; this is where "what the music does to the picture" is decided.
* ``resources`` the shader sources, read from files, with ``#include`` resolved.
* ``renderer``  OpenGL: programs, framebuffers, the passes. Works in whatever context is
                current, so the same code serves the view and the offscreen renders.
* ``view``      the ``QOpenGLWidget`` that sits behind the canvas.
* ``offscreen`` a context without a window, for tests and still pictures.

The shaders are given audio time and audio-derived numbers and nothing else: no clock of
their own, no frame counter, no random numbers. The same ``frame`` gives the same picture.
"""

SHADERS = ("plasma_field", "smoke", "formations", "instruments", "filaments", "core")  # the scene, in drawing order
POST = ("bloom_extract", "blur", "composite")
