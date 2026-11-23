"""Run Radiant's shaders on this machine's graphics card and measure what they draw.

    python tests/plasma_gpu_probe.py OUT.json

Not a test module: ``test_radiant_plasma.py`` runs it in a process of its own, because the
tests use Qt's ``offscreen`` platform, which has no OpenGL on Windows, and this needs the
real one. No window is opened. It writes ``{"available": false, "reason": ...}`` when no
OpenGL 3.3 context can be made, and otherwise one number per question the tests ask.
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

WIDTH, HEIGHT = 640, 360
KEYS = ("vocals", "guitar", "bass", "drums")
# One instrument at work, as the shaders are told it (see radiant_gpu/scene.py).
PLAYING = {
    "vocals": {"u_vocals": (1.0, 0.8, 0.5, 0.0), "u_vocals_more": (12.0, 1.0, 0.0, 9.0)},
    "guitar": {"u_guitar": (1.0, 0.7, 0.9, 0.9), "u_guitar_more": (0.05, 0.5, 0.8, 0.6)},
    "bass": {"u_bass": (1.0, 0.8, 0.5, 0.0), "u_bass_more": (1.3, 1.0, 0.0, 0.0)},
    "drums": {"u_drums": (1.0, 0.5, 0.0, 0.0), "u_drum_hits": [(0.05, 0.37)] + [(9.0, 0.0)] * 7,
              "u_drum_regions": [(0.9, 0.8, 0.7)] + [(0.0, 0.0, 0.0)] * 7},
}  # fmt: skip


def view_of(lay, geometry, *, now=12.3, level=0.6, intensity=0.6, flash=0.0, pulse=0.0, onsets=(), attacks=None,
            spectrum=None, beats=()):  # fmt: skip
    spectrum = np.clip(0.7 - 0.45 * np.linspace(0, 1, 48), 0, 1) if spectrum is None else spectrum
    view = SimpleNamespace(
        now=now, running=True, level=level, intensity=intensity, musical=intensity, flash=flash, pulse=pulse,
        segments=geometry.segment_levels(spectrum), segment_ring=geometry.segment_ring(lay, pulse),
        onsets=list(onsets), streaks=geometry.streaks(lay, list(onsets), attacks or {}), ray_turn=0.035 * now,
        beats=[(age, strength, now - age) for age, strength in beats],
    )  # fmt: skip
    view.core, view.halo, view.rays = geometry.core(lay, level, intensity, flash, pulse)
    return view, spectrum


def main(out):
    from PySide6.QtGui import QGuiApplication, QImage

    app = QGuiApplication(sys.argv[:1])  # noqa: F841
    from gmv.visualizer.response import BASELINE, EXPRESSIVE
    from gmv.visualizer.themes import radiant_geometry as geometry
    from gmv.visualizer.themes.radiant_gpu import POST, SHADERS, frame, resources
    from gmv.visualizer.themes.radiant_gpu import scene as composition
    from gmv.visualizer.themes.radiant_gpu.offscreen import Offscreen
    from gmv.visualizer.themes.radiant_gpu.resources import ShaderError
    from gmv.visualizer.themes.radiant_palette import RadiantPalette

    result = {"available": False, "platform": QGuiApplication.platformName()}
    try:
        plasma = Offscreen()
    except ShaderError as error:
        result["reason"] = str(error)
        Path(out).write_text(json.dumps(result), encoding="utf-8")
        return
    result.update(available=True, context=plasma.renderer.info, programs=sorted(plasma.renderer.programs))
    lay = composition.layout(WIDTH / 2, HEIGHT / 2, HEIGHT / 2 - 40, KEYS)
    palette = RadiantPalette()

    def values(response=BASELINE, colours=None, spectrum=None, playing=(), more=None, **scene):
        view, base = view_of(lay, geometry, spectrum=spectrum, **scene)
        chosen = palette
        if colours:
            chosen = RadiantPalette()
            for role, colour in colours.items():
                chosen.set(role, colour)
        found = frame.values(lay, view, chosen, response, size=(WIDTH, HEIGHT), area=(0, HEIGHT),
                             spectrum=base, instrument=lambda key: chosen[key])  # fmt: skip
        for key in playing:
            found.update(PLAYING[key])
        found.update(more or {})
        return found

    def raw(found, only=None, show=1):
        image = plasma.image(found | {"u_show": show}, WIDTH, HEIGHT, only).convertToFormat(QImage.Format.Format_RGB32)
        data = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(HEIGHT, image.bytesPerLine() // 4, 4)
        return data[:, :WIDTH, 2::-1].astype(float)  # red, green, blue

    def picture(found, only=None, show=1):
        """What the passes add to the plain background colour."""
        return np.clip(raw(found, only, show) - raw(found, (), 1), 0.0, None)

    def light(image):
        return image.max(axis=2)

    def tint(image, mask=None):
        pixels = image if mask is None else image[mask]
        total = pixels.reshape(-1, 3).sum(axis=0)
        return (total / max(1.0, total.sum())).tolist()

    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]
    radius = np.hypot(xs - lay.cx, ys - lay.cy) / lay.reach
    angle = np.mod(np.arctan2(xs - lay.cx, -(ys - lay.cy)), 2 * np.pi)
    down = np.where(angle < np.pi, angle, 2 * np.pi - angle) / np.pi  # 0 top .. 1 bottom
    rim, middle_r = lay.rim / lay.reach, lay.frame / lay.reach
    floor, tips, segments = lay.base / lay.reach, lay.tips / lay.reach, lay.segments / lay.reach
    base = values()

    # -- every pass draws something, the same thing twice ---------------------------------------
    busy = values(playing=KEYS)
    for name in SHADERS:
        first, again = light(picture(busy, (name,))), light(picture(busy, (name,)))
        result[f"{name} spread"] = float(first.std())
        result[f"{name} lit share"] = float((first > 40).mean())
        result[f"{name} repeat difference"] = float(np.abs(first - again).max())
    result["whole picture repeat difference"] = float(np.abs(raw(busy, show=0) - raw(busy, show=0)).max())

    # -- the field: structure, movement, seeds ----------------------------------------------------
    field = light(picture(base, ("plasma_field",)))
    corona = (radius > rim + 0.02) & (radius < rim + 0.22)  # the plasma hugging the outer formation
    result["field dark share in corona"] = float((field[corona] < 40).mean())
    result["field bright share in corona"] = float((field[corona] > 120).mean())
    result["field over the middle"] = float(field[radius < middle_r * 0.9].mean())
    result["field in corona"] = float(field[corona].mean())

    def moved(seconds):
        other = light(picture(values(now=12.3 + seconds), ("plasma_field",)))
        return float(np.abs(other - field).mean()), float(np.corrcoef(other.ravel(), field.ravel())[0, 1])

    for seconds in (0.05, 0.5, 5.0):
        result[f"field change after {seconds} s"], result[f"field likeness after {seconds} s"] = moved(seconds)
    seeded = light(picture(base | {"u_seed": 23.0}, ("plasma_field",)))
    result["field likeness other seed"] = float(np.corrcoef(seeded.ravel(), field.ravel())[0, 1])
    result["field other seed repeat difference"] = float(
        np.abs(seeded - light(picture(base | {"u_seed": 23.0}, ("plasma_field",)))).max()
    )

    # -- formations: surfaces with thickness, each different, the outer one with tendrils --------
    lows, highs, none = np.zeros(48), np.zeros(48), np.zeros(48)
    lows[:12], highs[32:] = 1.0, 1.0
    forms = light(picture(base, ("formations",)))
    outer_band, middle_band = np.abs(radius - rim) < 0.05, np.abs(radius - middle_r) < 0.02
    result["outer formation mean"], result["outer formation spread"] = float(forms[outer_band].mean()), float(forms[outer_band].std())
    result["middle formation mean"], result["middle formation spread"] = float(forms[middle_band].mean()), float(forms[middle_band].std())
    # How thick the outer formation is: the share of a ring five times a hairline's width that is lit.
    result["outer formation lit share of a broad ring"] = float((forms[np.abs(radius - rim) < 0.04] > 80).mean())
    later = light(picture(values(now=13.3), ("formations",)))
    result["formations change after 1 s"] = float(np.abs(later - forms)[outer_band | middle_band].mean())
    heavy = light(picture(values(spectrum=lows), ("formations",)))
    quiet = light(picture(values(spectrum=none), ("formations",)))
    far = (radius > rim + 0.06) & (radius < rim + 0.3)
    result["tendrils with lows"], result["tendrils without"] = float(heavy[far].mean()), float(quiet[far].mean())
    result["tendril lit share"] = float((heavy[far] > 60).mean())
    result["tendrils beyond their reach"] = float(heavy[radius > rim + 0.6].mean())
    ring_band = np.abs(radius - segments) < 0.008
    place = np.abs(np.mod(angle / (2 * np.pi) * 40, 1.0) - 0.5)
    result["segment light in segments"] = float(forms[ring_band & (place < 0.25)].mean())
    result["segment light in gaps"] = float(forms[ring_band & (place > 0.45)].mean())

    # -- the spectrum is the surface of the inner formation ---------------------------------------
    inner_zone = (radius > floor + 0.045) & (radius < tips + 0.02)
    top, bottom = inner_zone & (down < 0.22), inner_zone & (down > 0.72)
    for label, spectrum in (("lows", lows), ("highs", highs), ("silence", none), ("full", np.ones(48))):
        image = light(picture(values(spectrum=spectrum), ("formations",)))
        result[f"spectrum formation top with {label}"] = float(image[top].mean())
        result[f"spectrum formation bottom with {label}"] = float(image[bottom].mean())
    for strength in (0.25, 0.5, 1.0):
        image = light(picture(values(spectrum=np.full(48, strength)), ("formations",)))
        lit = (image > 70) & (radius > floor) & (radius < tips + 0.012)
        result[f"spectrum surface radius at {strength}"] = float(np.percentile(radius[lit], 97)) if lit.any() else 0.0
        result[f"spectrum formation light at {strength}"] = float(image[inner_zone].mean())
    ramp = np.linspace(0.1, 1.0, 48)  # rising from lows to highs: the surface must rise top to bottom
    image = light(picture(values(spectrum=ramp), ("formations",)))
    result["rising spectrum light by quarter"] = [
        float(image[inner_zone & (down >= a) & (down < a + 0.25)].mean()) for a in (0.0, 0.25, 0.5, 0.75)
    ]
    one = np.zeros(48)
    one[24] = 1.0  # one band alone: light there and beside it, none far from it
    image = light(picture(values(spectrum=one), ("formations",)))
    result["one band light at its place"] = float(image[inner_zone & (np.abs(down - 24.5 / 48) < 0.02)].mean())
    result["one band light far from it"] = float(image[inner_zone & (np.abs(down - 24.5 / 48) > 0.12)].mean())
    # No radial spikes: along a circle through the formation the light is one continuous body.
    circle = np.abs(radius - (floor + 0.06)) < 0.006
    along = light(picture(values(spectrum=np.full(48, 0.8)), ("formations",)))[circle]
    result["spectrum formation lit share along a circle"] = float((along > 60).mean())

    # -- the four instruments -------------------------------------------------------------------------
    zones = {key: (radius >= low - 0.01) & (radius <= high + 0.01) for key, (low, high) in composition.BANDS.items()}
    for key in KEYS:
        image = picture(values(playing=(key,)), ("instruments",))
        lit = light(image)
        result[f"{key} light"] = float(lit.mean())
        result[f"{key} share inside its zone"] = float(lit[zones[key]].sum() / max(1.0, lit.sum()))
        result[f"{key} tint"] = tint(image)
        result[f"{key} tint custom green"] = tint(picture(values(playing=(key,), colours={key: "#00ff00"}), ("instruments",)))
        hidden = dict(PLAYING[key])
        first_name = next(iter(hidden))
        hidden[first_name] = (0.0,) + tuple(hidden[first_name][1:])
        result[f"{key} light when hidden"] = float(light(picture(values(more=hidden), ("instruments",))).max())
        others = light(picture(values(playing=tuple(k for k in KEYS if k != key)), ("instruments",)))
        all_four = light(picture(values(playing=KEYS), ("instruments",)))
        result[f"{key} adds to the others"] = float((all_four - others)[zones[key]].mean())
        without = tuple(k for k in KEYS if k != key)
        hidden_one = values(playing=without)
        shown_all = values(playing=KEYS)
        other_zones = np.zeros_like(zones[key])
        for other in without:
            other_zones |= zones[other]
        other_zones &= ~zones[key] & (radius < composition.BANDS[key][0] - 0.05)
        result[f"{key} hiding changes the others"] = float(
            np.abs(light(picture(shown_all, ("instruments",))) - light(picture(hidden_one, ("instruments",))))[other_zones].max()
        ) if other_zones.any() else 0.0
    result["nothing playing light"] = float(light(picture(base, ("instruments",))).max())

    def played(key, **changes):
        return float(light(picture(values(more=PLAYING[key] | changes), ("instruments",))).mean())

    result["vocals soft"], result["vocals loud"] = played("vocals", u_vocals=(1.0, 0.2, 0.5, 0.0)), played("vocals", u_vocals=(1.0, 1.0, 0.5, 0.0))
    result["vocals with a phrase start"] = played("vocals", u_vocals=(1.0, 0.8, 0.5, 1.0), u_vocals_more=(12.0, 1.0, 0.0, 0.2))
    for label, where in (("low", 0.0), ("high", 1.0)):
        lit = light(picture(values(more=PLAYING["vocals"] | {"u_vocals": (1.0, 0.8, where, 0.0)}), ("instruments",)))
        result[f"vocals radius with a {label} note"] = float((radius * lit).sum() / max(1.0, lit.sum()))
    result["vocals moves in 0.3 s"] = float(np.abs(
        light(picture(values(more=PLAYING["vocals"] | {"u_vocals_more": (12.3, 1.0, 0.0, 9.0)}), ("instruments",)))
        - light(picture(values(playing=("vocals",)), ("instruments",)))).mean())
    result["guitar idle"] = played("guitar", u_guitar=(1.0, 0.0, 0.0, 0.0), u_guitar_more=(9.0, 0.5, 0.0, 0.0))
    result["guitar just struck"] = played("guitar")
    result["guitar struck softly"] = played("guitar", u_guitar=(1.0, 0.7, 0.3, 0.3))
    result["guitar ringing later"] = played("guitar", u_guitar=(1.0, 0.7, 0.5, 0.9), u_guitar_more=(0.6, 0.5, 0.8, 0.6))
    result["guitar strumming steadily"] = played("guitar", u_guitar_more=(0.05, 0.5, 0.8, 1.0))
    result["guitar single stroke"] = played("guitar", u_guitar_more=(0.05, 0.5, 0.8, 0.0))
    result["bass weak"], result["bass strong"] = played("bass", u_bass=(1.0, 0.2, 0.5, 0.0)), played("bass", u_bass=(1.0, 1.0, 0.5, 0.0))
    for label, energy in (("weak", 0.2), ("strong", 1.0)):
        lit = light(picture(values(more=PLAYING["bass"] | {"u_bass": (1.0, energy, 0.5, 0.0)}), ("instruments",)))
        result[f"bass thickness {label}"] = float((lit > 60).mean())
    pulse = {"u_bass": (1.0, 0.8, 0.5, 0.9), "u_bass_pulses": [(0.15, 0.9)] + [(9.0, 0.0)] * 3}
    beyond = radius > composition.BANDS["bass"][1] + 0.03
    result["bass wave beyond its zone with a note onset"] = float(light(picture(values(more=PLAYING["bass"] | pulse), ("instruments",)))[beyond].mean())
    result["bass wave beyond its zone without"] = float(light(picture(values(playing=("bass",)), ("instruments",)))[beyond].mean())
    result["bass phase moves the ring"] = float(np.abs(
        light(picture(values(more=PLAYING["bass"] | {"u_bass_more": (2.4, 1.0, 0.0, 0.0)}), ("instruments",)))
        - light(picture(values(playing=("bass",)), ("instruments",)))).mean())

    def hit(age=0.05, low=0.0, mid=0.0, high=0.0):
        return {"u_drums": (1.0, 0.5, 0.0, 0.0), "u_drum_hits": [(age, 0.37)] + [(9.0, 0.0)] * 7,
                "u_drum_regions": [(low, mid, high)] + [(0.0, 0.0, 0.0)] * 7}  # fmt: skip

    def struck(**what):
        return float(light(picture(values(more=hit(**what)), ("instruments",))).mean())

    result["drums no hit"] = struck()
    for region in ("low", "mid", "high"):
        result[f"drums {region} hit"] = struck(**{region: 0.9})
        result[f"drums weak {region} hit"] = struck(**{region: 0.3})
        result[f"drums {region} hit lit share"] = float((light(picture(values(more=hit(**{region: 0.9})), ("instruments",))) > 50).mean())
    result["drums low hit old"] = struck(low=0.9, age=0.55)
    result["drums mid hit old"] = struck(mid=0.9, age=0.3)
    two = hit(low=0.9)
    two["u_drum_hits"][1], two["u_drum_regions"][1] = (0.02, 0.81), (0.0, 0.9, 0.0)
    result["drums two hits at once"] = float(light(picture(values(more=two), ("instruments",))).mean())

    # -- filaments: thin, not straight, more with highs; bolts only with events --------------------
    calm = light(picture(values(spectrum=none), ("filaments",)))
    live = light(picture(values(spectrum=highs), ("filaments",)))
    result["filaments light calm"], result["filaments light with highs"] = float(calm.mean()), float(live.mean())
    result["filaments lit share"] = float((live > 60).mean())
    lit = live > 60
    result["filaments angle spread per radius"] = float(
        np.mean([np.std(angle[lit & (np.abs(radius - r) < 0.02)]) for r in np.linspace(middle_r + 0.1, rim, 6)
                 if (lit & (np.abs(radius - r) < 0.02)).sum() > 5] or [0.0])
    )  # fmt: skip
    result["bolts without events"] = float(values()["u_bolt_count"])
    struck_values = values(onsets=[(0.05, 0.9, 12.25)], attacks={"guitar": (1, [(0.04, 0.9, 12.26)])}, flash=0.6)
    result["bolts with events"] = float(struck_values["u_bolt_count"])
    result["filaments light with bolts"] = float(light(picture(struck_values, ("filaments",))).mean())
    result["filaments light without bolts"] = float(light(picture(values(flash=0.6), ("filaments",))).mean())
    faded = values(onsets=[(0.40, 0.9, 11.9)], attacks={"guitar": (1, [(0.33, 0.9, 11.97)])}, flash=0.6)
    result["filaments light with old bolts"] = float(light(picture(faded, ("filaments",))).mean())

    # -- the core: several regions, rays, no flat white disc, its own colour -----------------------
    core = light(picture(base, ("core",)))
    c = base["u_core"]
    for name, low, high in (("centre", 0.0, 0.25), ("body", 0.5, 0.85), ("limb", 0.98, 1.08), ("corona", 1.5, 2.5), ("far", 4.0, 6.0)):
        result[f"core {name}"] = float(core[(radius >= low * c) & (radius < high * c)].mean())
    result["core ray spread"] = float(core[(radius > 2.4 * c) & (radius < 2.7 * c)].std())
    loud_values = values(level=1.0, intensity=1.0, flash=1.0)
    loud, soft = light(picture(loud_values, ("core",))), light(picture(values(level=0.1, intensity=0.1), ("core",)))
    result["core light loud"], result["core light soft"] = float(loud.mean()), float(soft.mean())
    result["core lit radius loud"] = float(radius[loud > 80].max() if (loud > 80).any() else 0.0)
    result["core lit radius soft"] = float(radius[soft > 80].max() if (soft > 80).any() else 0.0)
    disc = radius < loud_values["u_core"]
    whole_loud = raw(loud_values, show=0)
    result["core white share of its disc when loud"] = float((whole_loud[disc].min(axis=1) > 245).mean())
    result["core body spread when loud"] = float(whole_loud[disc].max(axis=1).std())
    pink = {"core_center": "#ff1493", "core_halo": "#ff1493", "core_rays": "#ff1493"}
    pink_loud = raw(values(colours=pink, level=1.0, intensity=1.0, flash=1.0), show=0)[disc]
    result["pink core tint when loud"] = tint(pink_loud)
    result["pink core white share when loud"] = float((pink_loud.min(axis=1) > 235).mean())

    # -- bloom: light round bright things, none where nothing is bright ---------------------------
    scene = light(picture(busy, show=1))
    bloom = light(picture(busy, show=2))
    bright = scene > 200
    corners = np.zeros_like(bright)
    for rows in (slice(0, 24), slice(-24, None)):
        for columns in (slice(0, 24), slice(-24, None)):
            corners[rows, columns] = True
    grown = np.zeros_like(bright)
    for dy, dx in ((0, 6), (0, -6), (6, 0), (-6, 0), (4, 4), (-4, -4), (4, -4), (-4, 4)):
        grown |= np.roll(np.roll(bright, dy, 0), dx, 1)
    halo = grown & ~bright
    result["bright share"] = float(bright.mean())
    result["bloom beside bright"] = float(bloom[halo].mean()) if halo.any() else 0.0
    result["bloom in the corners"] = float(bloom[corners].mean())
    whole, without = raw(busy, show=0), raw(busy | {"u_bloom": 0.0}, show=0)
    result["bloom adds beside bright"] = float((light(whole) - light(without))[halo].mean()) if halo.any() else 0.0
    result["bloom adds in the corners"] = float((light(whole) - light(without))[corners].mean())
    result["picture white share"] = float((whole.min(axis=2) > 250).mean())
    result["picture dark share"] = float((whole.max(axis=2) < 60).mean())

    # -- colours go where they belong, and stay themselves when bright ----------------------------
    reds = {"plasma_primary": "#ff0000", "plasma_secondary": "#ff0000", "plasma_highlights": "#ff0000"}
    greens = {"plasma_primary": "#00ff00", "plasma_secondary": "#00ff00", "plasma_highlights": "#00ff00"}
    hazeless = {"atmosphere_primary": "#000000", "atmosphere_secondary": "#000000", "particle_accents": "#000000"}
    result["field tint red plasma"] = tint(picture(values(colours=reds | hazeless), ("plasma_field",)), corona)
    result["field tint green plasma"] = tint(picture(values(colours=greens | hazeless), ("plasma_field",)), corona)
    result["filament tint green"] = tint(picture(values(colours={"filament_glow": "#00ff00", "filament_core": "#00ff00"}, spectrum=highs), ("filaments",)))
    result["core tint orange"] = tint(picture(values(colours={"core_center": "#ff7a00", "core_halo": "#ff7a00", "core_rays": "#ff7a00"}), ("core",)))
    spectrum_zone = (radius > floor) & (radius < tips)
    result["spectrum tint red"] = tint(picture(values(colours={"spectrum_body": "#ff0000", "spectrum_highlights": "#ff0000"}), ("formations",)), spectrum_zone)
    result["waveform tint green"] = tint(picture(values(colours={"secondary_ring": "#00ff00"}, more={"u_wave": [0.8] * 48}), ("formations",)),
                                         np.abs(radius - lay.contour / lay.reach - 0.02) < 0.012)  # fmt: skip
    result["segment tint yellow"] = tint(picture(values(colours={"segmented_ring": "#ffff00", "plasma_highlights": "#ffff00"}), ("formations",)), ring_band)
    result["outer formation tint green"] = tint(picture(values(colours={"primary_ring": "#00ff00", "plasma_highlights": "#00ff00"}), ("formations",)), np.abs(radius - rim) < 0.03)
    blank = {role: "#000000" for role in ("plasma_primary", "plasma_secondary", "plasma_highlights", "atmosphere_primary", "atmosphere_secondary")}
    specks = picture(values(colours=blank | {"particle_accents": "#ff0000"}), ("plasma_field",))
    result["particle tint red"], result["particle light"] = tint(specks), float(light(specks).max())
    changed = values(colours={"filament_glow": "#ff1493"}, playing=KEYS)
    for name in ("plasma_field", "formations", "instruments", "core"):
        result[f"{name} change when filament colour changes"] = float(np.abs(picture(changed, (name,)) - picture(busy, (name,))).max())
    # Saturated colours stay saturated where they are bright (the composite keeps the hue).
    hot = raw(values(colours={"plasma_primary": "#ff1493", "plasma_secondary": "#ff1493", "plasma_highlights": "#ff1493",
                              "primary_ring": "#ff1493"} | hazeless, level=1.0, intensity=1.0), ("plasma_field", "formations"), 0)  # fmt: skip
    brightest = (hot.max(axis=2) > 200) & (radius > middle_r + 0.02)
    saturation = (hot.max(axis=2) - hot.min(axis=2)) / np.maximum(hot.max(axis=2), 1.0)
    result["bright pink share"] = float(brightest.mean())
    result["bright pink saturation"] = float(saturation[brightest].mean()) if brightest.any() else 0.0
    red_smoke = {"smoke_primary": "#ff0000", "smoke_highlight": "#ff0000", "smoke_shadow": "#ff0000"}
    mixed = raw(values(colours=reds | red_smoke | hazeless | {"filament_glow": "#00ff00", "filament_core": "#00ff00", "core_center": "#0000ff",
                                                    "core_halo": "#0000ff", "core_rays": "#0000ff"}, spectrum=highs), show=0)  # fmt: skip
    background = raw(base, (), 1)
    result["mixed corona tint"] = tint(np.clip(mixed - background, 0, None), corona)
    result["mixed core tint"] = tint(np.clip(mixed - background, 0, None), radius < 0.9 * c)

    # -- the music moves it --------------------------------------------------------------------------
    for label, scene_values in (("quiet", dict(level=0.1, intensity=0.1)), ("loud", dict(level=0.9, intensity=0.95))):
        image = light(picture(values(**scene_values), ("plasma_field",)))
        result[f"field light {label}"] = float(image[corona].mean())
        result[f"field reach {label}"] = float(image[(radius > rim + 0.14) & (radius < rim + 0.34)].mean())
        result[f"whole picture light {label}"] = float(light(raw(values(**scene_values), show=0)).mean())
    reach_zone = (radius > rim + 0.14) & (radius < rim + 0.34)
    result["field reach with lows"] = float(light(picture(values(spectrum=lows), ("plasma_field",)))[reach_zone].mean())
    result["field reach without lows"] = float(light(picture(values(spectrum=none), ("plasma_field",)))[reach_zone].mean())
    result["field veins with highs"] = float((light(picture(values(spectrum=highs), ("plasma_field",)))[corona] > 150).mean())
    result["field veins without highs"] = float((light(picture(values(spectrum=none), ("plasma_field",)))[corona] > 150).mean())
    onset = light(picture(values(onsets=[(0.12, 1.0, 12.18)]), ("plasma_field",)))
    result["field change with an onset"] = float(np.abs(onset - field).mean())
    old = light(picture(values(onsets=[(0.95, 1.0, 11.35)]), ("plasma_field",)))
    result["field change with an old onset"] = float(np.abs(old - field).mean())
    result["scene light baseline"] = float(light(picture(busy, show=0)).mean())
    result["scene light expressive"] = float(light(picture(values(response=EXPRESSIVE, playing=KEYS), show=0)).mean())

    # -- smoke: alive on its own, and moved by beats --------------------------------------------------
    smoke_zone = (radius > rim + 0.08) & (radius < rim + 0.9)

    def smoke(**scene):
        return light(picture(values(**scene), ("smoke",)))

    def unwrapped(image, turns=180, rings=24):
        """The smoke zone as (angle, radius): to turn it round the centre, roll it."""
        angles = (np.arange(turns) + 0.5) / turns * 2 * np.pi
        radii = np.linspace(rim + 0.1, rim + 0.85, rings)
        x = np.clip((lay.cx + np.outer(np.sin(angles), radii) * lay.reach).round().astype(int), 0, WIDTH - 1)
        y = np.clip((lay.cy - np.outer(np.cos(angles), radii) * lay.reach).round().astype(int), 0, HEIGHT - 1)
        return image[y, x]

    def likeness(one, other):
        return float(np.corrcoef(one.ravel(), other.ravel())[0, 1])

    def roughness(image):  # how much fine structure: mean step between neighbouring pixels
        down_steps = np.abs(np.diff(image, axis=0))[smoke_zone[1:]]
        across_steps = np.abs(np.diff(image, axis=1))[smoke_zone[:, 1:]]
        return float(down_steps.mean() + across_steps.mean())

    def by_sector(image):
        return [float(image[smoke_zone & (angle >= k * np.pi / 6) & (angle < (k + 1) * np.pi / 6)].mean()) for k in range(12)]

    still = smoke()
    result["smoke light with no beats"] = float(still[smoke_zone].mean())
    result["smoke lit share"] = float((still[smoke_zone] > 30).mean())
    result["smoke dark share"] = float((still[smoke_zone] < 10).mean())
    result["smoke dense share"] = float((still[smoke_zone] > 110).mean())
    result["smoke brightest"] = float(still.max())
    result["smoke inside the formations"] = float(still[radius < rim - 0.1].max())
    result["smoke repeat difference"] = float(np.abs(still - smoke()).max())
    result["smoke light by sector"] = by_sector(still)
    # Movement with no beat at all.
    for seconds in (0.25, 1.0, 4.0, 12.0):
        later = smoke(now=12.3 + seconds)
        result[f"smoke change after {seconds} s"] = float(np.abs(later - still)[smoke_zone].mean())
        result[f"smoke likeness after {seconds} s"] = likeness(later[smoke_zone], still[smoke_zone])
    # Is it one picture being turned? Turn the later picture back by every angle and take the best match.
    first, later = unwrapped(still), unwrapped(smoke(now=16.3))
    result["smoke best likeness to a turned copy after 4 s"] = max(likeness(np.roll(later, shift, 0), first) for shift in range(-30, 31))
    result["smoke likeness of a turned picture turned back"] = max(likeness(np.roll(np.roll(first, 7, 0), shift, 0), first) for shift in range(-30, 31))
    result["smoke change by sector after 1 s"] = by_sector(np.abs(smoke(now=13.3) - still))
    # The seam: straight up from the centre is where an angle wraps. Is there an edge there?
    column = int(round(lay.cx))
    heights = np.arange(HEIGHT)
    rows = heights[((lay.cy - heights) / lay.reach > rim + 0.1)]
    busy_smoke = smoke(level=1.0, intensity=1.0)
    step = lambda k: float(np.abs(busy_smoke[rows, column + k] - busy_smoke[rows, column + k - 1]).mean())  # noqa: E731
    result["smoke step across the top seam"] = max(step(0), step(1))
    elsewhere = [step(k) for k in range(-60, 61, 5) if abs(k) > 6]
    result["smoke step elsewhere"] = float(np.mean(elsewhere))
    result["smoke largest step elsewhere"] = float(np.max(elsewhere))
    # One beat, followed through its life. The same audio time throughout: only the beat's age differs.
    result["smoke roughness at rest"] = roughness(still)
    for age in (0.0, 0.1, 0.25, 0.45, 0.9, 1.6, 2.5):
        struck_smoke = smoke(beats=[(age, 0.9)])
        change = np.abs(struck_smoke - still)
        result[f"smoke change {age} s after a beat"] = float(change[smoke_zone].mean())
        result[f"smoke roughness {age} s after a beat"] = roughness(struck_smoke)
        if age == 0.45:
            result["smoke change by sector after a beat"] = by_sector(change)
            # Could one stretch of the whole picture explain it? Stretch the still picture outwards
            # by every amount and see how much of the change the best stretch leaves unexplained.
            flat, hit_smoke = unwrapped(still), unwrapped(struck_smoke)
            best = min(float(np.abs(np.roll(flat, shift, 1) - hit_smoke)[:, 4:-4].mean()) for shift in range(-4, 5))
            result["smoke change a uniform stretch cannot explain"] = best / max(1e-6, float(np.abs(flat - hit_smoke)[:, 4:-4].mean()))
            result["smoke light with a beat passing"] = float(struck_smoke[smoke_zone].mean())
    result["smoke change before the wave arrives"] = float(np.abs(smoke(beats=[(0.0, 0.9)]) - still)[radius > rim + 0.3].mean())
    result["smoke change with a weak beat"] = float(np.abs(smoke(beats=[(0.45, 0.45)]) - still)[smoke_zone].mean())
    result["smoke change with a strong beat"] = float(np.abs(smoke(beats=[(0.45, 1.0)]) - still)[smoke_zone].mean())
    bold_still = smoke(response=EXPRESSIVE)
    result["smoke change with a beat, expressive"] = float(np.abs(smoke(beats=[(0.45, 0.9)], response=EXPRESSIVE) - bold_still)[smoke_zone].mean())
    result["smoke change with a beat, baseline"] = float(np.abs(smoke(beats=[(0.45, 0.9)]) - still)[smoke_zone].mean())
    one_new, one_old = smoke(beats=[(0.3, 0.9)]), smoke(beats=[(0.8, 0.9)])
    both = smoke(beats=[(0.3, 0.9), (0.8, 0.9)])
    result["smoke change with two beats"] = float(np.abs(both - still)[smoke_zone].mean())
    result["smoke change with the newer beat alone"] = float(np.abs(one_new - still)[smoke_zone].mean())
    result["smoke change with the older beat alone"] = float(np.abs(one_old - still)[smoke_zone].mean())
    result["smoke two beats differ from the newer alone"] = float(np.abs(both - one_new)[smoke_zone].mean())
    result["smoke two beats differ from the older alone"] = float(np.abs(both - one_old)[smoke_zone].mean())
    sixteen = [(0.05 + 0.16 * n, 1.0) for n in range(16)]
    rapid = smoke(beats=sixteen, level=1.0, intensity=1.0)
    result["smoke brightest with sixteen beats"] = float(rapid.max())
    result["smoke light with sixteen beats"] = float(rapid[smoke_zone].mean())
    result["smoke white share with sixteen beats"] = float(
        (raw(values(beats=sixteen, level=1.0, intensity=1.0), ("smoke",), 0).min(axis=2) > 245).mean()
    )
    result["smoke inside the formations with sixteen beats"] = float(rapid[radius < rim - 0.45].max())
    result["smoke quiet"] = float(smoke(level=0.1, intensity=0.1)[smoke_zone].mean())
    result["smoke loud"] = float(smoke(level=0.9, intensity=0.95)[smoke_zone].mean())
    # Its three colours.
    black = {"smoke_primary": "#000000", "smoke_highlight": "#000000", "smoke_shadow": "#000000"}
    for role, colour, name in (
        ("smoke_primary", "#ff0000", "red primary"), ("smoke_highlight", "#00ff00", "green highlight"),
        ("smoke_shadow", "#0000ff", "blue shadow"),
    ):  # fmt: skip
        image = picture(values(colours=black | {role: colour}, beats=[(0.4, 0.9)]), ("smoke",))
        result[f"smoke tint {name}"] = tint(image, smoke_zone)
        result[f"smoke light {name}"] = float(light(image)[smoke_zone].mean())
    three = picture(values(colours={"smoke_primary": "#ff0000", "smoke_highlight": "#00ff00", "smoke_shadow": "#0000ff"}), ("smoke",))
    result["smoke lit share of the smoke"] = float((three[..., 0] > three[..., 2] + 20)[smoke_zone].mean())
    result["smoke shaded share of the smoke"] = float((three[..., 2] > three[..., 0] + 20)[smoke_zone].mean())
    all_pink = {"smoke_primary": "#ff1493", "smoke_highlight": "#ff1493", "smoke_shadow": "#ff1493"}
    pink_smoke = raw(values(colours=all_pink | hazeless, level=1.0, intensity=1.0, beats=[(0.4, 1.0)]), ("smoke",), 0)
    bright_smoke = (pink_smoke.max(axis=2) > 150) & smoke_zone
    saturated = (pink_smoke.max(axis=2) - pink_smoke.min(axis=2)) / np.maximum(pink_smoke.max(axis=2), 1.0)
    result["smoke bright pink share"] = float(bright_smoke.mean())
    result["smoke bright pink saturation"] = float(saturated[bright_smoke].mean()) if bright_smoke.any() else 0.0
    recoloured, plain = values(colours={"smoke_primary": "#00ff00"}, beats=[(0.4, 0.9)]), values(beats=[(0.4, 0.9)])
    shape = lambda found: (light(picture(found, ("smoke",))) > 25).astype(float)  # noqa: E731
    result["smoke shape change when its colour changes"] = float(np.abs(shape(recoloured) - shape(plain)).mean())
    for name in ("plasma_field", "formations", "core"):
        result[f"{name} change when smoke colour changes"] = float(np.abs(picture(recoloured, (name,)) - picture(plain, (name,))).max())
    # -- smoke, continued: the checks of the smoke milestone -------------------------------------------
    # Intensity: more turbulent, not just brighter, and still clouds with gaps.
    quiet_smoke, loud_smoke = smoke(level=0.1, intensity=0.1), smoke(level=0.95, intensity=0.95)
    result["smoke roughness quiet"], result["smoke roughness loud"] = roughness(quiet_smoke), roughness(loud_smoke)
    result["smoke dark share loud"] = float((loud_smoke[smoke_zone] < 10).mean())
    result["smoke change after 1 s, quiet"] = float(np.abs(smoke(level=0.1, intensity=0.1, now=13.3) - quiet_smoke)[smoke_zone].mean())
    result["smoke quiet and loud likeness"] = likeness(quiet_smoke[smoke_zone], loud_smoke[smoke_zone])
    # It never covers the instruments, however hard it is pushed.
    sixteen_bold = smoke(beats=sixteen, level=1.0, intensity=1.0, response=EXPRESSIVE)
    instrument_zone = radius < rim - 0.025
    result["smoke over the instruments, sixteen beats, expressive"] = float(sixteen_bold[instrument_zone].max())
    result["smoke over the instruments, no beats"] = float(still[instrument_zone].max())
    result["smoke brightest, sixteen beats, expressive"] = float(sixteen_bold.max())
    result["smoke white share, sixteen beats, expressive"] = float(
        (raw(values(beats=sixteen, level=1.0, intensity=1.0, response=EXPRESSIVE), ("smoke",), 0).min(axis=2) > 245).mean()
    )
    # Nothing jumps: when a beat leaves the list, and when a new beat starts while an older wake is going.
    from gmv.visualizer.themes import radiant_geometry as geometry_module

    leaving = smoke(beats=[(geometry_module.BEAT_LIFE - 0.005, 1.0)])
    result["smoke step when a beat leaves the list"] = float(np.abs(leaving - still)[smoke_zone].max())
    older = smoke(beats=[(0.8, 1.0)])
    older_and_new = smoke(beats=[(0.0, 1.0), (0.8, 1.0)])
    result["smoke step when a new beat starts during a wake"] = float(np.abs(older_and_new - older)[smoke_zone].max())
    # Regular beats: each reaction like the others in size, not in shape. Four beats of a regular
    # sequence (half a second apart) seen at the same age over the same smoke: only what is their
    # own (their time, so their pattern) differs. Compared as change pictures, pixel by pixel,
    # and as the change in each twelfth of the circle relative to the smoke there.
    footprints, pictures, sizes = [], [], []
    base_light = np.maximum(np.array(by_sector(still)), 1.0)
    for n in range(4):
        beat_time = 12.3 - 0.45 + 0.5 * n
        own = [(0.45, 0.9, frame.own_turn(beat_time), 0.0)] + [(0.0, 0.0, 0.0, 0.0)] * 15
        hit_smoke = light(picture(values(more={"u_beats": own, "u_beat_count": 1}), ("smoke",)))
        change = hit_smoke - still
        pictures.append(change[smoke_zone])
        footprints.append(np.array(by_sector(np.abs(change))) / base_light)
        sizes.append(float(np.abs(change)[smoke_zone].mean()))
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    result["smoke regular beats change-picture likeness"] = [likeness(pictures[i], pictures[j]) for i, j in pairs]
    result["smoke regular beats footprint likeness"] = [likeness(footprints[i], footprints[j]) for i, j in pairs]
    result["smoke regular beats sizes"] = sizes

    # The same audio time at any frame rate, rendered: the stream is fed as it would arrive at
    # each rate and the frame at the same audio times is drawn.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import render_radiant as synthetic_stream
    from gmv.models import FeatureFrame, MusicalEvent
    from gmv.visualizer.state import VisualState
    from gmv.visualizer.themes.radiant import Radiant

    stream_messages = [m for m in synthetic_stream.synthetic(7.0, 0.85, ()) if isinstance(m, FeatureFrame | MusicalEvent)]
    timed = [(m.timestamp, m) for m in stream_messages]
    for index, beat_time in enumerate((4.6, 5.05, 5.5, 5.8, 6.1, 6.25, 6.4)):  # regular, then quicker
        prediction = MusicalEvent(
            session_id="synthetic", source="synthetic", detected_monotonic=0, timestamp=beat_time,
            event_id=f"probe-beat-{index}", event_type="beat_prediction", intensity=0.3 + 0.06 * index,
            confidence=0.5, method="probe", evidence_kind="predicted",
        )  # fmt: skip
        timed.append((beat_time - 0.25, prediction))  # published a quarter of a second ahead
    timed.sort(key=lambda item: item[0])

    def frame_at(state, now):
        spectrum = state.spectrum_at(now)
        view = Radiant().scene(state, now, lay, values=spectrum, peaks=None, response=BASELINE, running=True, keys=())
        return frame.values(lay, view, palette, BASELINE, size=(WIDTH, HEIGHT), area=(0, HEIGHT), spectrum=spectrum, state=state)

    def stepped(rate, targets):
        state, waiting, tick, found = VisualState(), list(timed), 0, {}
        while len(found) < len(targets):
            now = 1.0 + tick / rate
            while waiting and waiting[0][0] <= now + 1e-9:
                _, message = waiting.pop(0)
                state.consume(message, arrival=1000.0 + message.timestamp if isinstance(message, FeatureFrame) else None)
            for target in targets:
                if target not in found and now >= target - 1e-9:
                    found[target] = frame_at(state, now)
            tick += 1
        return found, state

    targets = (5.5, 6.0, 6.5)  # whole numbers of frames at all five rates
    reference, _ = stepped(60, targets)
    result["frame rates compared"] = [60, 72, 90, 120, 144]
    result["smoke beats at the compared times"] = [reference[target]["u_beat_count"] for target in targets]
    worst_smoke, worst_whole = 0.0, 0.0
    for rate in (72, 90, 120, 144):
        other, _ = stepped(rate, targets)
        for target in targets:
            worst_smoke = max(worst_smoke, float(np.abs(raw(other[target], ("smoke",)) - raw(reference[target], ("smoke",))).max()))
            worst_whole = max(worst_whole, float(np.abs(raw(other[target], show=0) - raw(reference[target], show=0)).max()))
    result["smoke largest pixel difference between frame rates"] = worst_smoke
    result["whole picture largest pixel difference between frame rates"] = worst_whole
    # Reconstructed: a state that is handed everything up to 6.5 s at once (as after opening a
    # recording there) draws the same smoke as one that played through to it.
    fresh = VisualState()
    for _, message in timed:
        if message.timestamp <= 6.5 + 1e-9 or isinstance(message, MusicalEvent):
            if isinstance(message, FeatureFrame) and message.timestamp > 6.5 + 1e-9:
                continue
            fresh.consume(message, arrival=1000.0 + message.timestamp if isinstance(message, FeatureFrame) else None)
    rebuilt = frame_at(fresh, 6.5)
    result["smoke pixel difference, reconstructed against played through"] = float(
        np.abs(raw(rebuilt, ("smoke",)) - raw(reference[6.5], ("smoke",))).max()
    )
    result["smoke beats, reconstructed and played through"] = [rebuilt["u_beat_count"], reference[6.5]["u_beat_count"]]

    # What it costs: the same frame with and without the pass, with a forced finish, for no
    # beat, one beat, and sixteen beats at once (the most the list holds), and in Expressive
    # Mode with the multicolour palette. Each figure is the best of three runs of twelve frames.
    multicolour = RadiantPalette()
    multicolour.apply("multicolor_energy")
    from PySide6.QtCore import QSize
    from PySide6.QtOpenGL import QOpenGLFramebufferObject

    def card_ms(found, size, only):
        """Milliseconds the card takes for a frame: frames drawn into a buffer of their own and
        finished, nothing read back; the best of five runs of twenty frames."""
        plasma.context.makeCurrent(plasma.surface)
        target = QOpenGLFramebufferObject(QSize(*size))
        gl = plasma.context.functions()
        plasma.renderer.render(found, size[0], size[1], target.handle(), only)
        gl.glFinish()
        runs = []
        for _ in range(5):
            began = time.perf_counter()
            for _ in range(20):
                plasma.renderer.render(found, size[0], size[1], target.handle(), only)
            gl.glFinish()
            runs.append((time.perf_counter() - began) / 20 * 1000)
        return min(runs)

    scenarios = (
        ("no beats", [], BASELINE, palette),
        ("one beat", [(0.4, 0.9)], BASELINE, palette),
        ("sixteen beats", [(0.05 + 0.15 * n, 0.9) for n in range(16)], BASELINE, palette),
        ("sixteen beats, expressive, multicolour", [(0.05 + 0.15 * n, 0.9) for n in range(16)], EXPRESSIVE, multicolour),
    )
    for size in ((1920, 1080), (2560, 1440)):
        big = composition.layout(size[0] / 2, size[1] / 2, size[1] / 2 - 50, KEYS)
        for name, beats, response, colours in scenarios:
            full = frame.values(big, view_of(big, geometry, beats=beats, level=0.9, intensity=0.9)[0], colours, response,
                                size=size, area=(0, size[1]))  # fmt: skip
            timed_ms = {}
            for label, only in (("with", None), ("without", tuple(p for p in SHADERS if p != "smoke"))):
                timed_ms[label] = card_ms(full, size, only)
            result[f"smoke pass ms {size[0]}x{size[1]}, {name}"] = timed_ms["with"] - timed_ms["without"]
            result[f"all passes ms with a forced finish {size[0]}x{size[1]}, {name}"] = timed_ms["with"]
        result[f"smoke pass ms {size[0]}x{size[1]}"] = result[f"smoke pass ms {size[0]}x{size[1]}, one beat"]
        result[f"all passes ms with a forced finish {size[0]}x{size[1]}"] = result[f"all passes ms with a forced finish {size[0]}x{size[1]}, one beat"]

    # -- a broken or missing shader is reported, not drawn -------------------------------------------
    folder = Path(tempfile.mkdtemp())
    try:
        broken = folder / "broken"
        shutil.copytree(resources.FOLDER, broken)
        (broken / "core.frag").write_text((broken / "core.frag").read_text(encoding="utf-8").replace("void main()", "void main() { this is not GLSL; }\nvoid other()"), encoding="utf-8")
        resources.source.cache_clear()
        try:
            Offscreen(broken)
            result["broken shader error"] = ""
        except ShaderError as error:
            result["broken shader error"] = str(error)[:300]
        missing = folder / "missing"
        shutil.copytree(resources.FOLDER, missing)
        (missing / "bass.glsl").unlink()
        resources.source.cache_clear()
        try:
            Offscreen(missing)
            result["missing shader error"] = ""
        except ShaderError as error:
            result["missing shader error"] = str(error)[:300]
    finally:
        resources.source.cache_clear()
        shutil.rmtree(folder, ignore_errors=True)

    # -- cost, sizes, and that nothing grows ---------------------------------------------------------
    plasma.context.makeCurrent(plasma.surface)
    result["programs expected"] = sorted(SHADERS + POST)
    for size in ((1920, 1080), (2560, 1440)):
        big = composition.layout(size[0] / 2, size[1] / 2, size[1] / 2 - 50, KEYS)
        full = frame.values(big, view_of(big, geometry)[0], palette, BASELINE, size=size, area=(0, size[1])) | {
            k: v for key in KEYS for k, v in PLAYING[key].items()
        }
        plasma.image(full, *size)
        began = time.perf_counter()
        for _ in range(10):
            plasma.image(full, *size)
        result[f"ms per frame with read-back {size[0]}x{size[1]}"] = (time.perf_counter() - began) * 100
        result[f"buffers MiB {size[0]}x{size[1]}"] = plasma.renderer.memory() / 2**20
    count = len(plasma.renderer.buffers)
    for size in ((800, 600), (1024, 640), (800, 600), (640, 360)):
        plasma.image(base, *size)
    result["buffers after resizing"] = len(plasma.renderer.buffers)
    result["buffers before resizing"] = count
    plasma.close()
    result["ready after release"] = bool(plasma.renderer.ready)
    result["left after release"] = len(plasma.renderer.buffers) + len(plasma.renderer.programs)
    # Set up and torn down over and over, each time with frames at two sizes in between.
    cycles = 0
    for _ in range(6):
        again = Offscreen()
        again.image(base, 320, 180)
        again.image(base, 480, 270)
        again.close()
        cycles += 1
    result["set-up and tear-down cycles"] = cycles
    Path(out).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
