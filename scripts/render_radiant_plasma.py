"""Render Radiant with its graphics-card effects to still pictures and a frame sequence.

    python scripts/render_radiant_plasma.py --out local-evidence/radiant-phase3
    python scripts/render_radiant_plasma.py --session PATH --at 40,95
    python scripts/render_radiant_plasma.py --sequence 48 --only none

Each picture is what the application shows: Radiant's shaders run on the graphics card in
a context without a window (``radiant_gpu.offscreen``) and draw the whole central view,
and the application's own shell (header text, sections) is laid over the result, as Qt
does on screen. No window opens. It needs a real OpenGL 3.3 context, so it does not run on Qt's
``offscreen`` platform.

Without ``--session`` the visual state is made up (see ``render_radiant.synthetic``) and
the file names say ``synthetic``; with it, a recorded session's messages are used (analysis
messages only, never audio). ``--sequence N`` also writes N consecutive frames, 1/24 s of
audio time apart, to show that the picture moves.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "windows" if sys.platform == "win32" else os.environ.get("QT_QPA_PLATFORM", "")
# No effects widget is made here (the window is never shown): the shaders are run in a
# context of this script's own and the canvas is told that they are.
os.environ["GMV_RADIANT_GPU"] = "0"
sys.path.insert(0, str(Path(__file__).parent))

import render_radiant as phase1  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QImage, QPainter, QRegion  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from gmv.visualizer.themes.radiant_palette import PRESETS  # noqa: E402

# The plasma and filament colours that go with phase 1's three test palettes.
CUSTOM = {
    "hot_pink_violet": phase1.CUSTOM["hot_pink"] | {
        "plasma_primary": "#ff1493", "plasma_secondary": "#7c3aed", "plasma_highlights": "#ffd1f0",
        "filament_core": "#ffffff", "filament_glow": "#c084fc", "atmosphere_secondary": "#6d28d9",
        "segmented_ring": "#a855f7",
    },
    "green_yellow_cyan": phase1.CUSTOM["green"] | {
        "plasma_primary": "#16a34a", "plasma_secondary": "#0891b2", "plasma_highlights": "#fef08a",
        "filament_core": "#fffbe0", "filament_glow": "#facc15", "primary_ring": "#fde047",
        "spectrum_body": "#06b6d4", "spectrum_highlights": "#cffafe", "core_halo": "#22d3ee",
    },
    "mixed": phase1.CUSTOM["mixed"] | {
        "plasma_primary": "#ff1493", "plasma_secondary": "#0ea5e9", "plasma_highlights": "#fde047",
        "filament_core": "#f0fff4", "filament_glow": "#39ff14",
    },
}  # fmt: skip


def composed(window, plasma, width, height, effects=True):
    """The window as it shows on screen: the effects layer, and the canvas over it."""
    canvas = window.canvas
    canvas.gpu = effects and canvas.themed() is not None
    window.grab()  # lays the window out, and paints the canvas once so the frame is known
    overlay = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    overlay.fill(Qt.GlobalColor.transparent)
    canvas.render(overlay, QPoint(), QRegion(), QWidget.RenderFlag.DrawChildren)
    if not canvas.gpu:
        return window.grab().toImage()
    picture = plasma.image(canvas.plasma_values(), width, height).convertToFormat(
        QImage.Format.Format_ARGB32_Premultiplied
    )
    painter = QPainter(picture)
    painter.drawImage(0, 0, overlay)
    painter.end()
    return picture


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="local-evidence/radiant-phase3")
    parser.add_argument("--sizes", default="1920x1080,2560x1440")
    parser.add_argument("--session", help="a recorded session to take the pictures from")
    parser.add_argument("--at", default="40", help="audio times in the session, comma list")
    parser.add_argument("--only", default="", help="comma list of picture names to make")
    parser.add_argument("--sequence", type=int, default=0, help="also this many consecutive frames")
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv[:1])
    from gmv.visualizer.themes.radiant_gpu.offscreen import Offscreen
    from gmv.visualizer.window import VisualizerWindow

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    window = VisualizerWindow()
    window.timer.stop()
    window.set_mode("radial")
    for name in ("timeline", "details", "notes"):
        window.set_section(name, shown=False)
    window.canvas.info["instruments"] = phase1.KEYS
    if args.session:
        label = "session"
        sources = {f"t{float(at):g}": (lambda at=float(at): phase1.recorded(args.session, at)) for at in args.at.split(",")}
    else:
        label = "synthetic"
        sources = {
            "full_mix_onset": lambda: phase1.synthetic(6.06, 1.0),  # ends 60 ms after a strong onset
            "quiet": lambda: phase1.synthetic(6.06, 0.3),
            "bass": lambda: phase1.synthetic(6.06, 0.9, ("bass",)),
            "guitar": lambda: phase1.synthetic(6.06, 0.9, ("guitar",)),
            "vocals": lambda: phase1.synthetic(6.06, 0.9, ("vocals",)),
            "drums": lambda: phase1.synthetic(6.06, 0.9, ("drums",)),
            "no_instruments": lambda: phase1.synthetic(6.06, 0.8, ()),
        }
    first = next(iter(sources))
    # name, theme, effects, palette, expressive, source
    pictures = [
        ("1_classic", "classic", "off", None, False, first),
        ("2_radiant_fallback_qpainter", "radiant", "off", "electric_blue", False, first),
        ("4_radiant_gpu", "radiant", "auto", "electric_blue", False, first),
        ("5_gpu_hot_pink_violet", "radiant", "auto", "hot_pink_violet", False, first),
        ("5_gpu_green_yellow_cyan", "radiant", "auto", "green_yellow_cyan", False, first),
        ("5_gpu_multicolour", "radiant", "auto", "mixed", False, first),
        ("6_gpu_expressive", "radiant", "auto", "electric_blue", True, first),
    ]
    pictures += [(f"5_gpu_preset_{name}", "radiant", "auto", name, False, first) for name in PRESETS if name != "electric_blue"]
    pictures += [(f"7_gpu_{name}", "radiant", "auto", "electric_blue", False, name) for name in list(sources)[1:]]
    wanted = {name for name in args.only.split(",") if name}
    made = []

    def prepare(theme, effects, colours, expressive):
        window.effects = effects
        window.set_radial_theme(theme)
        window.set_expressive(expressive)
        window.radiant_palette.restore()
        if colours in PRESETS:
            window.set_radiant_preset(colours)
        elif colours in CUSTOM:
            for role, colour in CUSTOM[colours].items():
                window.set_radiant_colour(role, colour)

    with Offscreen() as plasma:
        print("context:", plasma.renderer.info)
        for size in args.sizes.split(","):
            width, height = map(int, size.split("x"))
            window.resize(width, height)
            for name, theme, effects, colours, expressive, source in pictures:
                if wanted and name not in wanted:
                    continue
                prepare(theme, effects, colours, expressive)
                phase1.show(window, sources[source]())
                path = out / f"{label}_{name}_{source if source != first else first}_{size}.png"
                composed(window, plasma, width, height, effects == "auto").save(str(path))
                made.append(str(path))
        if args.sequence:
            width, height = 1920, 1080
            window.resize(width, height)
            prepare("radiant", "auto", "electric_blue", False)
            folder = out / f"sequence_{label}"
            folder.mkdir(exist_ok=True)
            messages = list(sources[first]()) if not args.session else None
            for index in range(args.sequence):
                if args.session:
                    stretch = list(phase1.recorded(args.session, float(args.at.split(",")[0]) + index / 24))
                else:  # the same stream, cut 1/24 s later each time
                    until = 5.0 + index / 24
                    stretch = [m for m in messages if m.timestamp <= until]
                phase1.show(window, stretch)
                composed(window, plasma, width, height).save(str(folder / f"frame_{index:03d}.png"))
            made.append(f"{folder} ({args.sequence} frames, 1/24 s of audio apart)")
    window.close()
    print("\n".join(made))
    app.quit()


if __name__ == "__main__":
    main()
