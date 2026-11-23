"""Render frame sequences of Radiant's smoke: drifting on its own, and disturbed by beats.

    python scripts/render_radiant_smoke.py --out local-evidence/radiant-smoke
    python scripts/render_radiant_smoke.py --only idle,one_beat --frames 36
    python scripts/render_radiant_smoke.py --session PATH --at 40 --frames 48

Each sequence is consecutive frames of the real GPU picture, 1/24 s of audio time apart,
rendered in a context without a window. The synthetic sequences are a made-up stream
(``render_radiant.synthetic``) with predicted-beat events placed at chosen audio times, so
that what a beat does can be looked at in isolation; their folders say ``synthetic``.
``--session`` renders a stretch of a recorded session instead (analysis messages only).
The session is read once; the minute before the stretch is fed first, so that the
whole-mix intensity has settled, and then each frame takes in the messages up to its own
audio time, as playback would. ``--step`` spaces the frames further apart in audio time
(to see a passage change over several seconds); ``--name`` names the sequence's folder.

A sequence folder also gets ``timing.json``: what the graphics card took per frame with
and without the smoke pass, measured with a forced finish (so higher than in the
application, but comparable between the two).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "windows" if sys.platform == "win32" else os.environ.get("QT_QPA_PLATFORM", "")
os.environ["GMV_RADIANT_GPU"] = "0"  # no effects widget: the shaders run in this script's own context
sys.path.insert(0, str(Path(__file__).parent))

import render_radiant as phase1  # noqa: E402
import render_radiant_plasma as plasma_tools  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gmv.models import MusicalEvent  # noqa: E402
from gmv.visualizer.themes.radiant_palette import PRESETS  # noqa: E402

START = 5.0  # audio time of the first frame of a synthetic sequence
PALETTES = {
    "hot_pink_violet": {
        "smoke_primary": "#ff1493", "smoke_highlight": "#ffd1f0", "smoke_shadow": "#5b21b6",
        "plasma_primary": "#ff1493", "plasma_secondary": "#7c3aed", "primary_ring": "#ff69b4",
        "background_base": "#14030f", "atmosphere_primary": "#8a0f5c", "atmosphere_secondary": "#6d28d9",
    },
    "green_cyan": {
        "smoke_primary": "#16a34a", "smoke_highlight": "#a5f3fc", "smoke_shadow": "#0e7490",
        "plasma_primary": "#16a34a", "plasma_secondary": "#0891b2", "primary_ring": "#4ade80",
        "background_base": "#020d04", "atmosphere_primary": "#0f6b1f", "atmosphere_secondary": "#0e7490",
    },
}  # fmt: skip
# name -> (beat times after START with strengths, loudness of the stream, expressive, palette)
SEQUENCES = {
    "idle": ((), 0.7, False, None),
    "one_beat": (((0.25, 0.9),), 0.7, False, None),
    "two_beats": (((0.25, 0.9), (0.65, 0.9)), 0.7, False, None),
    "regular_beats": (tuple((0.2 + 0.5 * n, 0.8) for n in range(4)), 0.7, False, None),
    "rapid_beats": (tuple((0.2 + 0.16 * n, 0.8) for n in range(8)), 0.7, False, None),
    "strong_beat_high_intensity": (((0.25, 1.0),), 1.0, False, None),
    "recovery": (tuple((-1.4 + 0.35 * n, 0.9) for n in range(4)), 0.7, False, None),  # the last beat is before the first frame
    "expressive": (((0.25, 0.9),), 0.7, True, None),
    "hot_pink_violet": (((0.25, 0.9),), 0.7, False, "hot_pink_violet"),
    "green_cyan": (((0.25, 0.9),), 0.7, False, "green_cyan"),
    "preset_plasma_magenta": (((0.25, 0.9),), 0.7, False, "plasma_magenta"),
    "preset_aurora_green": (((0.25, 0.9),), 0.7, False, "aurora_green"),
    "preset_multicolor_energy": (((0.25, 0.9),), 0.7, False, "multicolor_energy"),
}


def stream(beats, loud, until):
    """A made-up stream up to audio time ``until``, with predicted beats at the given
    times after ``START`` (each published a little ahead of its time, as the tracker does)."""
    messages = [m for m in phase1.synthetic(until + 0.5, loud, ()) if m.timestamp <= until]
    for index, (offset, strength) in enumerate(beats):
        when = START + offset
        if when - 0.3 <= until:
            messages.append(
                MusicalEvent(
                    session_id="synthetic", source="synthetic", detected_monotonic=0, timestamp=when,
                    event_id=f"beat-{index}", event_type="beat_prediction", intensity=strength,
                    confidence=strength, method="synthetic", evidence_kind="predicted",
                )  # fmt: skip
            )
    return sorted(messages, key=lambda m: (min(m.timestamp, until), m.timestamp))


def session_sequence(window, plasma, args, folder, width, height):
    """Frames of a recorded session from ``args.at``, fed incrementally into one state."""
    from time import perf_counter

    from gmv.models import FeatureFrame

    end = args.at + args.frames * args.step + 0.5
    messages = [m for m in phase1.recorded(args.session, end) if m.timestamp >= args.at - 60.0]
    messages.sort(key=lambda m: m.timestamp)
    window.state.status = "starting"
    window.state.reset()
    base = perf_counter() - end - 5.0  # local clock of audio time zero
    fed = 0
    log = []
    for index in range(args.frames):
        now = args.at + index * args.step
        while fed < len(messages) and messages[fed].timestamp <= now:
            message = messages[fed]
            arrival = base + message.timestamp + 0.03 if isinstance(message, FeatureFrame) else None
            window.state.consume(message.model_copy(update={"session_id": "render"}), arrival=arrival)
            fed += 1
        window.canvas.advance(args.step, base + now + 0.03)
        picture = plasma_tools.composed(window, plasma, width, height)
        picture.save(str(folder / f"frame_{index:03d}.png"))
        found = window.canvas.plasma_values() or {}
        log.append({
            "frame": index, "audio_time": round(now, 3), "beats": found.get("u_beat_count"),
            "musical_intensity": round(found.get("u_musical", 0.0), 3), "level": round(found.get("u_level", 0.0), 3),
        })  # fmt: skip
    (folder / "frames.json").write_text(
        json.dumps({"session": Path(args.session).name, "from": args.at, "step_s": args.step, "frames": log}, indent=1),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="local-evidence/radiant-smoke")
    parser.add_argument("--size", default="1280x720")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--only", default="", help="comma list of sequences")
    parser.add_argument("--session", help="a recorded session to render a stretch of")
    parser.add_argument("--at", type=float, default=40.0, help="audio time in the session to start at")
    parser.add_argument("--label", default="", help="put the sequences under this sub-folder")
    parser.add_argument("--step", type=float, default=1 / 24, help="audio seconds between frames")
    parser.add_argument("--name", default="", help="folder name of a session sequence")
    parser.add_argument("--expressive", action="store_true", help="session sequence in Expressive Mode")
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv[:1])
    from gmv.visualizer.themes.radiant_gpu.offscreen import Offscreen
    from gmv.visualizer.window import VisualizerWindow

    out = Path(args.out) / args.label if args.label else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    width, height = map(int, args.size.split("x"))
    window = VisualizerWindow()
    window.timer.stop()
    window.set_mode("radial")
    window.set_radial_theme("radiant")
    window.effects = "auto"
    window.set_beats(True)
    for name in ("timeline", "details", "notes"):
        window.set_section(name, shown=False)
    window.canvas.info["instruments"] = ()
    window.resize(width, height)
    wanted = {name for name in args.only.split(",") if name}
    made = []
    with Offscreen() as plasma:
        if args.session:
            jobs = {"session": None}
        else:
            jobs = {name: spec for name, spec in SEQUENCES.items() if not wanted or name in wanted}
        if args.session:
            folder = out / (args.name or f"session_t{args.at:g}")
            folder.mkdir(exist_ok=True)
            window.radiant_palette.restore()
            window.set_expressive(args.expressive)
            session_sequence(window, plasma, args, folder, width, height)
            made.append(f"{folder} ({args.frames} frames, {args.step:g} s apart)")
            jobs = {}
        for name, spec in jobs.items():
            folder = out / f"synthetic_{name}"
            folder.mkdir(exist_ok=True)
            window.radiant_palette.restore()
            window.set_expressive(bool(spec and spec[2]))
            if spec and spec[3] in PRESETS:
                window.set_radiant_preset(spec[3])
            elif spec and spec[3]:
                for role, colour in PALETTES[spec[3]].items():
                    if role in window.radiant_palette.colours:
                        window.set_radiant_colour(role, colour)
            card = {"with smoke": [], "without smoke": []}
            for index in range(args.frames):
                now = (args.at if spec is None else START) + index / 24
                messages = list(phase1.recorded(args.session, now)) if spec is None else stream(spec[0], spec[1], now)
                phase1.show(window, messages)
                picture = plasma_tools.composed(window, plasma, width, height)
                picture.save(str(folder / f"frame_{index:03d}.png"))
                if index % 6 == 0:  # the card's time for this frame, both ways
                    found = window.canvas.plasma_values()
                    passes = plasma.renderer.programs
                    for label, only in (("with smoke", None), ("without smoke", tuple(p for p in passes if p != "smoke"))):
                        began = time.perf_counter()
                        for _ in range(3):
                            plasma.image(found, width, height, only)
                        card[label].append((time.perf_counter() - began) / 3 * 1000)
            (folder / "timing.json").write_text(
                json.dumps(
                    {
                        "size": args.size, "context": plasma.renderer.info,
                        "ms per frame with a forced finish and read-back": {k: round(sum(v) / len(v), 2) for k, v in card.items() if v},
                        "buffers MiB": round(plasma.renderer.memory() / 2**20, 1),
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )  # fmt: skip
            made.append(f"{folder} ({args.frames} frames)")
    window.close()
    print("\n".join(made))
    app.quit()


if __name__ == "__main__":
    assert PRESETS  # the palettes above are changes to the factory preset
    main()
