"""The colours of the Radiant theme: one independent colour per visible role. No Qt here.

Every colour Radiant draws with comes from a role of this palette; the renderer holds no
colour literals of its own. Colours are plain ``#rrggbb`` strings, so the same palette can
be handed to another renderer (a shader, later) unchanged.

Roles are independent: nothing derives one role's colour from another's, so a pink core
can sit inside green rings round a blue spectrum. The four instrument roles start at the
canonical instrument colours (``visualizer.palette.INSTRUMENTS``) and are Radiant's own
copies: changing one changes neither the canonical colour nor any other mode.

A preset is a starting point, not a mode. ``preset`` names the preset the colours came
from and becomes ``"custom"`` the moment one colour is changed; the colours themselves are
what is saved and what counts.
"""

from .. import palette as canonical

GROUPS = (
    ("background", "Background and atmosphere"),
    ("core", "Central core"),
    ("spectrum", "Spectrum"),
    ("structures", "Concentric structures"),
    ("instruments", "Instruments"),
)
# (role, title, group, what it colours)
ROLES = (
    ("background_base", "Background base", "background", "The colour behind everything."),
    ("atmosphere_primary", "Atmosphere primary", "background",
     "The diffuse glow round the centre, and the background glow when no instrument leads."),
    ("atmosphere_secondary", "Atmosphere secondary", "background",
     "The outer haze and the faint arcs."),
    ("core_center", "Core centre", "core", "The concentrated point in the middle."),
    ("core_halo", "Core halo", "core", "The halo round the centre and the core's boundary."),
    ("core_rays", "Core rays", "core", "The fine rays going out from the centre."),
    ("spectrum_body", "Spectrum body", "spectrum", "The radial spikes and their baseline."),
    ("spectrum_highlights", "Spectrum highlights", "spectrum",
     "The tips of strong spikes and the peak marks."),
    ("primary_ring", "Primary ring", "structures",
     "The two luminous circles that frame the instruments."),
    ("secondary_ring", "Secondary ring", "structures", "The jagged waveform-like contour."),
    ("segmented_ring", "Segmented ring", "structures", "The ring of separate lit segments."),
    ("energy_accents", "Energy accents", "structures",
     "Radial streaks at onsets and attacks, and the beat glow."),
    ("particle_accents", "Particle accents", "structures", "The small drifting points of light."),
    ("vocals", "Vocals", "instruments", "The vocals' strands in Radiant."),
    ("guitar", "Guitar", "instruments", "The guitar's strings in Radiant."),
    ("bass", "Bass", "instruments", "The bass ring in Radiant."),
    ("drums", "Drums", "instruments", "The drums' impacts in Radiant."),
)  # fmt: skip
KEYS = tuple(role[0] for role in ROLES)
INSTRUMENTS = ("vocals", "guitar", "bass", "drums")
CUSTOM = "custom"


def _preset(**colours):
    return {**colours, **canonical.INSTRUMENTS}


# name -> (title, colours). The first is the factory palette.
PRESETS = {
    "electric_blue": (
        "Electric Blue",
        _preset(
            background_base="#030816",
            atmosphere_primary="#0b3a8f",
            atmosphere_secondary="#0a6fb8",
            core_center="#f4fbff",
            core_halo="#5cc8ff",
            core_rays="#bfe9ff",
            spectrum_body="#1e90ff",
            spectrum_highlights="#c8f4ff",
            primary_ring="#3fb4ff",
            secondary_ring="#7fd8ff",
            segmented_ring="#2f7cff",
            energy_accents="#e6fbff",
            particle_accents="#8fd0ff",
        ),
    ),
    "plasma_magenta": (
        "Plasma Magenta",
        _preset(
            background_base="#0c0314",
            atmosphere_primary="#5a0f8a",
            atmosphere_secondary="#b0128f",
            core_center="#fff2fd",
            core_halo="#ff5ce1",
            core_rays="#ffc4f4",
            spectrum_body="#d81bff",
            spectrum_highlights="#ffd6fb",
            primary_ring="#ff4fd8",
            secondary_ring="#b46bff",
            segmented_ring="#7a5cff",
            energy_accents="#ffe9fb",
            particle_accents="#f0a6ff",
        ),
    ),
    "aurora_green": (
        "Aurora Green",
        _preset(
            background_base="#02100c",
            atmosphere_primary="#0b6b4a",
            atmosphere_secondary="#0a7f8a",
            core_center="#f2fff8",
            core_halo="#4dffb0",
            core_rays="#c6ffe6",
            spectrum_body="#19d98a",
            spectrum_highlights="#d9ffe9",
            primary_ring="#3dffa2",
            secondary_ring="#5fe8d0",
            segmented_ring="#9be15d",
            energy_accents="#f0fff2",
            particle_accents="#9df5cf",
        ),
    ),
    "multicolor_energy": (
        "Multicolor Energy",
        _preset(
            background_base="#070512",
            atmosphere_primary="#3b1a8f",
            atmosphere_secondary="#0a6f8f",
            core_center="#fff6e0",
            core_halo="#ff9d3c",
            core_rays="#ffe08a",
            spectrum_body="#22d3ee",
            spectrum_highlights="#f0abfc",
            primary_ring="#a855f7",
            secondary_ring="#f43f8e",
            segmented_ring="#facc15",
            energy_accents="#fff4d6",
            particle_accents="#7dd3fc",
        ),
    ),
}
FACTORY = "electric_blue"


def valid(colour):
    """``colour`` as a lower-case ``#rrggbb`` string, or None if it is not a colour.
    ``#rgb`` and a missing ``#`` are accepted; names, alpha and anything else are not."""
    if not isinstance(colour, str):
        return None
    text = colour.strip().lstrip("#").lower()
    if len(text) == 3:
        text = "".join(character * 2 for character in text)
    if len(text) != 6 or any(character not in "0123456789abcdef" for character in text):
        return None
    return "#" + text


class RadiantPalette:
    """The colours in use, and the preset they started from. ``version`` grows with every
    change, so a renderer knows when what it cached in the old colours is stale."""

    def __init__(self):
        self.version = 0
        self.preset = FACTORY
        self.colours = dict(PRESETS[FACTORY][1])

    def __getitem__(self, role):
        return self.colours[role]

    def set(self, role, colour):
        """Give one role a colour. Nothing else changes. Returns whether it was taken."""
        colour = valid(colour)
        if role not in self.colours or colour is None:
            return False
        if colour != self.colours[role]:
            self.colours[role] = colour
            self.preset = CUSTOM
            self.version += 1
        return True

    def apply(self, preset):
        """Take every colour of a named preset; they can be changed one by one afterwards."""
        if preset not in PRESETS:
            return False
        self.colours = dict(PRESETS[preset][1])
        self.preset = preset
        self.version += 1
        return True

    def restore(self):
        """Back to the factory palette."""
        self.apply(FACTORY)

    def saved(self):
        """What to keep between runs: every colour, and where they came from."""
        return {"preset": self.preset, "colours": dict(self.colours)}

    def load(self, saved):
        """Take saved colours. Anything missing or not a colour gets its factory colour;
        nothing in ``saved`` can make this fail. The saved colours are authoritative: a
        preset's name is only kept when the colours still are that preset's."""
        factory = PRESETS[FACTORY][1]
        colours = saved.get("colours") if isinstance(saved, dict) else None
        colours = colours if isinstance(colours, dict) else {}
        self.colours = {role: valid(colours.get(role)) or factory[role] for role in KEYS}
        name = saved.get("preset") if isinstance(saved, dict) else None
        if not isinstance(name, str) or name not in PRESETS or PRESETS[name][1] != self.colours:
            name = next(
                (key for key, (_, preset) in PRESETS.items() if preset == self.colours), CUSTOM
            )
        self.preset = name
        self.version += 1
