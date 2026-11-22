"""How much of each instrument's radial picture changes from one display frame to the next.

    python scripts/measure_ring_motion.py SESSION_DIR [--from 20 --seconds 12 --fps 72]

A recorded session is fed to an offscreen visualizer on a simulated clock (no display, no
real time), one frame every 1/fps seconds, once per instrument with only that instrument's
ring in the picture (spectrum and timeline hidden), for the live rings and for the
six-second history rings. After every frame the picture is compared with the frame before:

    changed     share of the lit pixels (of either frame) that differ

For comparison the same is done for the whole-mix spectrum alone, which is redrawn from
the current sound every frame. A live animation changes a large part of its pixels in most
frames; a history changes only where the newest slice is written.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication


def pixels(picture):
    image = (picture.toImage() if hasattr(picture, "toImage") else picture).convertToFormat(
        QImage.Format.Format_ARGB32_Premultiplied
    )
    data = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine() // 4, 4)
    return data[:, : image.width()].astype(np.int16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("--from", dest="start", type=float, default=20.0)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=float, default=72.0)
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--rings", default="live,history")
    parser.add_argument("--out")
    args = parser.parse_args()
    from gmv.contract import parse_message
    from gmv.visualizer.window import VisualizerWindow

    app = QApplication.instance() or QApplication([])
    messages = []
    with open(Path(args.session) / "messages.jsonl", encoding="utf-8") as file:
        for line in file:
            message = parse_message(json.loads(line))
            if message.published_monotonic is not None:
                messages.append(message)
    first = next(m.published_monotonic - m.timestamp for m in messages if hasattr(m, "window_end"))
    report = {}
    passes = [(f"{kind}: {key}", kind, key) for kind in args.rings.split(",") for key in ("vocals", "guitar", "bass", "drums")]
    passes.append(("spectrum (whole mix) alone", "live", None))
    for label, kind, only in passes:
        view = VisualizerWindow()
        view.timer.stop()
        width, height = map(int, args.size.split("x"))
        view.resize(width, height)
        view.set_mode("radial")
        view.set_radial_rings(kind)
        view.set_section("timeline", shown=False)
        view.set_spectrum_place("beside" if only is None else "hidden")
        for key in view.canvas.layers:
            view.set_layer(key, enabled=key == only)
        view.canvas.info["instruments"] = tuple(view.canvas.layers)
        canvas, state = view.canvas, view.state
        clock = first + args.start - 8.0  # eight seconds of run-in, not measured
        end = first + args.start + args.seconds
        index, before, shares = 0, None, []
        while clock < end:
            while index < len(messages) and messages[index].published_monotonic <= clock:
                state.consume(messages[index], arrival=messages[index].published_monotonic)
                index += 1
            canvas.advance(1 / args.fps, clock)
            now = pixels(view.grab())[int(height * 0.06) :]  # without the header line
            if clock >= first + args.start:
                if before is not None:
                    lit = (now[:, :, :3].max(axis=2) > 70) | (before[:, :, :3].max(axis=2) > 70)
                    changed = (np.abs(now - before).max(axis=2) > 6) & lit
                    if lit.sum() > 50:
                        shares.append(float(changed.sum() / lit.sum()))
                before = now
            clock += 1 / args.fps
        report[label] = {
            "changed_share_of_lit_pixels_per_frame": {
                "p50": round(float(np.median(shares)), 4) if shares else None,
                "p95": round(float(np.percentile(shares, 95)), 4) if shares else None,
            },
            "frames_with_anything_lit": len(shares),
        }
        view.close()
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    del app


if __name__ == "__main__":
    sys.exit(main())
