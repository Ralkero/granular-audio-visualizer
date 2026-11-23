"""Render still pictures of the radial themes offscreen, for looking at.

    python scripts/render_radiant.py --out local-evidence/radiant
    python scripts/render_radiant.py --out local-evidence/radiant --session PATH --at 42,95

Without ``--session`` the pictures come from a made-up visual state: analysis messages
written here (a spectrum with moving humps, onsets on a steady pulse, activity for the
four instruments), fed to the real window exactly as a stream would be. It is not music,
and the file names say ``synthetic``. With ``--session`` the messages of a recorded
session are fed up to each of the ``--at`` audio times (seconds) instead; a session holds
analysis messages only, never audio.

Each picture is the real ``VisualizerWindow`` drawn by Qt's offscreen platform at the
sizes given, so no window opens. Pictures of the owner's own music belong in an ignored
folder (``local-evidence/``), which is the default here.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from gmv.contract import parse_message
from gmv.models import FeatureFrame, InstrumentActivity, InstrumentEstimate, MusicalEvent
from gmv.visualizer.themes.radiant_palette import PRESETS

HOP = 0.02
KEYS = ("vocals", "guitar", "bass", "drums")
# Palettes that are not presets: every role a hue of its own choosing, to show that the
# picture does not depend on one colour family.
CUSTOM = {
    "hot_pink": {
        "background_base": "#14030f", "atmosphere_primary": "#8a0f5c", "atmosphere_secondary": "#c2185b",
        "core_center": "#fff0f7", "core_halo": "#ff2d95", "core_rays": "#ffb3d9",
        "spectrum_body": "#ff1493", "spectrum_highlights": "#ffd1ea", "primary_ring": "#ff69b4",
        "secondary_ring": "#ff85c8", "segmented_ring": "#ff3da6", "energy_accents": "#fff0fa",
        "particle_accents": "#ff9fd0",
    },
    "green": {
        "background_base": "#020d04", "atmosphere_primary": "#0f6b1f", "atmosphere_secondary": "#3c8a0a",
        "core_center": "#f4fff0", "core_halo": "#39ff14", "core_rays": "#c8ffb8",
        "spectrum_body": "#22c55e", "spectrum_highlights": "#dcfce7", "primary_ring": "#4ade80",
        "secondary_ring": "#a3e635", "segmented_ring": "#16a34a", "energy_accents": "#f0fff0",
        "particle_accents": "#86efac",
    },
    "mixed": {
        "background_base": "#0a0418", "atmosphere_primary": "#0e7490", "atmosphere_secondary": "#7e22ce",
        "core_center": "#ff2d95", "core_halo": "#ff7a00", "core_rays": "#ffe600",
        "spectrum_body": "#2563eb", "spectrum_highlights": "#22d3ee", "primary_ring": "#22c55e",
        "secondary_ring": "#a855f7", "segmented_ring": "#facc15", "energy_accents": "#ff7a00",
        "particle_accents": "#f472b6",
        "vocals": "#22d3ee", "guitar": "#ff2d95", "bass": "#a3e635", "drums": "#c084fc",
    },
}  # fmt: skip


def spectrum(time, loud):
    """48 display bands in dB: a bass hump, a moving mid hump and a falling top, with a
    little grain that changes from frame to frame (seeded by the frame's time)."""
    place = np.linspace(0.0, 1.0, 48)
    grain = np.random.default_rng(int(round(time * 1000))).normal(0.0, 2.5, 48)
    bass = 16 * np.exp(-(((place - 0.12) / 0.09) ** 2)) * (0.6 + 0.4 * math.sin(time * 7.0) ** 2)
    mid = 12 * np.exp(-(((place - 0.45 - 0.08 * math.sin(time * 1.3)) / 0.14) ** 2))
    top = 7 * np.exp(-(((place - 0.8) / 0.1) ** 2)) * (0.5 + 0.5 * math.sin(time * 11.0) ** 2)
    return (-52 + 26 * loud - 14 * place + bass + mid + top + grain).tolist()


def synthetic(seconds, loud=1.0, playing=KEYS):
    """Messages of ``seconds`` of a made-up stream, in the order an engine publishes."""
    common = dict(session_id="synthetic", source="synthetic", detected_monotonic=0)
    messages = []
    for index in range(round(seconds / HOP)):
        time = 1.0 + index * HOP
        beat = index % 25  # two pulses a second
        hit = beat in (0, 12)
        onset = (0.85 if beat == 0 else 0.55) * loud if hit else 0.08 * loud
        level = -34 + 22 * loud + (4 if hit else 0)
        messages.append(
            FeatureFrame(
                **common, timestamp=time, window_end=time + 0.023, sample_rate=48000, channels=2,
                rms=0.1, peak=0.3, loudness_dbfs=level, short_term_loudness_dbfs=level,
                spectral_centroid_hz=1000, spectral_bandwidth_hz=1000, spectral_rolloff_hz=4000,
                spectral_contrast_db=(0,) * 7, zero_crossing_rate=0.1,
                band_energy=(0.05, 0.3, 0.15, 0.25, 0.15, 0.07, 0.03), spectral_flux=0.1,
                transient_strength=onset, onset_strength=onset, dynamic_intensity=0.25 + 0.6 * loud,
                silence=False, spectrum_db=tuple(spectrum(time, loud)), spectrum_low_hz=30,
                spectrum_high_hz=18000,
                instrument_scores=tuple((k, 0.9 if k in playing else 0.05) for k in ("drums", "bass", "vocals")),
                instruments_active=tuple(k for k in ("drums", "bass", "vocals") if k in playing),
                activity_method="synthetic",
            )  # fmt: skip
        )
        if hit:
            messages.append(
                MusicalEvent(
                    **common, timestamp=time, event_id=f"onset-{index}", event_type="onset",
                    intensity=onset, confidence=0.6, method="synthetic", evidence_kind="observed",
                )  # fmt: skip
            )

        def row(key, time=time, **values):
            base = dict(instrument=key, method="synthetic", level=0.5, flux=0.02, attack=0.0)
            messages.append(InstrumentActivity(**common, timestamp=time, **(base | values)))

        if "guitar" in playing:
            strum = 0.8 * loud * (index % 10 == 0)
            row("guitar", level=0.6 * loud, attack=strum, flux=strum, centre=0.45, spread=0.7)
        if "drums" in playing:
            strength = 0.9 * loud * hit
            row(
                "drums", attack=strength, flux=strength,
                components=(("low", 0.9 * loud * (beat == 0)), ("mid", 0.8 * loud * (beat == 12)),
                            ("high", 0.6 * loud * (index % 6 == 3))),
            )  # fmt: skip
        if "bass" in playing:
            note = 28 + 5 * (index // 50 % 3)
            row(
                "bass", level=0.75 * loud, energy=0.75 * loud, pitch_midi=float(note),
                pitch_confidence=0.9, attack=0.7 * (index % 50 == 0), pulse=0.8 * loud * (beat == 0),
            )  # fmt: skip
        if "vocals" in playing:
            phrase = index % 100
            row(
                "vocals", level=0.7 * loud * (phrase < 75),
                pitch_midi=62.0 + 5 * math.sin(index / 18) if phrase < 75 else None,
                pitch_confidence=0.9,
                boundary="start" if phrase == 0 else "end" if phrase == 80 else "",
            )  # fmt: skip
        if index % 25 == 24:  # the recognition model's word on the guitar, twice a second
            messages.append(
                InstrumentEstimate(
                    **common, timestamp=max(0.0, time - 0.5), interval_end=time, backend="synthetic",
                    scores=(("guitar", 0.9 if "guitar" in playing else 0.05),),
                    states=(("guitar", "detected" if "guitar" in playing else "absent"),),
                    context_s=0.5,
                )  # fmt: skip
            )
    return messages


def recorded(path, until):
    """A recorded session's messages up to audio time ``until``."""
    path = Path(path)
    path = path / "messages.jsonl" if path.is_dir() else path
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            message = parse_message(json.loads(line))
            if getattr(message, "timestamp", 0.0) > until + 1.0:
                break
            yield message


def show(window, messages):
    """Feed ``messages`` so that the newest has only just arrived, and step the display."""
    messages = list(messages)
    frames = [m for m in messages if isinstance(m, FeatureFrame)]
    if not frames:
        raise SystemExit("no analysis frames in that stretch")
    window.state.status = "starting"
    window.state.reset()
    base = perf_counter() - frames[-1].timestamp - 0.03
    for message in messages:
        arrival = base + message.timestamp + 0.03 if isinstance(message, FeatureFrame) else None
        window.state.consume(message.model_copy(update={"session_id": "render"}), arrival=arrival)
    clock = perf_counter()
    for _ in range(40):  # let the display's own easing (bar heights, glow) settle
        window.canvas.advance(1 / 72, clock)
    window.canvas.display.step(window.state.spectrum, 1 / 72, window.state.spectrum_range)
    window.canvas.advance(1 / 72, clock)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="local-evidence/radiant")
    parser.add_argument("--sizes", default="1920x1080,2560x1440")
    parser.add_argument("--session", help="a recorded session to take the pictures from")
    parser.add_argument("--at", default="30", help="audio times in the session, comma list")
    parser.add_argument("--only", default="", help="comma list of picture names to make")
    args = parser.parse_args()

    from PySide6.QtWidgets import QApplication

    from gmv.visualizer.window import VisualizerWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    window = VisualizerWindow()
    window.timer.stop()
    window.set_mode("radial")
    for name in ("timeline", "details", "notes"):
        window.set_section(name, shown=False)
    window.canvas.info["instruments"] = KEYS

    # name -> (theme, palette, expressive, source of messages)
    if args.session:
        label = "session"
        sources = {f"t{float(at):g}": (lambda at=float(at): recorded(args.session, at)) for at in args.at.split(",")}
    else:
        label = "synthetic"
        sources = {
            "intense": lambda: synthetic(6.06, 1.0),
            "quiet": lambda: synthetic(6.06, 0.3),
            "bass_only": lambda: synthetic(6.06, 0.8, ("bass",)),
            "vocals_and_guitar": lambda: synthetic(6.06, 0.8, ("vocals", "guitar")),
        }
    first = next(iter(sources))
    pictures = [("classic", "classic", None, False, first)]
    pictures += [(f"radiant_{name}", "radiant", name, False, first) for name in PRESETS]
    pictures += [(f"radiant_custom_{name}", "radiant", name, False, first) for name in CUSTOM]
    pictures += [("radiant_expressive", "radiant", "electric_blue", True, first)]
    pictures += [(f"radiant_{name}", "radiant", "electric_blue", False, name) for name in list(sources)[1:]]
    wanted = {name for name in args.only.split(",") if name}
    made = []
    for size in args.sizes.split(","):
        width, height = map(int, size.split("x"))
        window.resize(width, height)
        for name, theme, colours, expressive, source in pictures:
            if wanted and name not in wanted:
                continue
            window.set_radial_theme(theme)
            window.set_expressive(expressive)
            if colours in PRESETS:
                window.set_radiant_preset(colours)
            elif colours in CUSTOM:
                window.radiant_palette.restore()
                for role, colour in CUSTOM[colours].items():
                    window.set_radiant_colour(role, colour)
            show(window, sources[source]())
            path = out / f"{label}_{name}_{first if source == first else ''}{size}.png".replace("__", "_")
            window.grab().save(str(path))
            made.append(str(path))
    window.close()
    app.quit()
    print("\n".join(made))


if __name__ == "__main__":
    main()
