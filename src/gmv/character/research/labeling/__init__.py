"""Phase A3: listening to short excerpts and labelling how the voice is delivered.

Research tooling, separate from the visualizer. ``dataset`` decides the excerpts and the
artist-disjoint split before anyone listens; ``store`` keeps every answer (append-only, so
nothing is lost and every change is kept); ``playback`` plays exactly the ten seconds of an
excerpt from the original file; ``window`` is the labelling window; ``cli`` adds
``gmv character label`` and ``gmv character prepare-labels``.

The data lives outside Git in ``local-evidence/sonic-character/vocal-labeling/``.
"""

CLASSES = {
    "clean": (
        "Clean singing",
        "Melodic, pitched singing with a relatively clean vocal tone.",
    ),
    "scream": (
        "Screaming",
        "Harsh vocal delivery: screams, growls, shrieks and similar aggressive techniques.",
    ),
    "shout": (
        "Shouting",
        "Forceful, yelled or speech-like delivery that is not mainly clean melodic singing "
        "and not a sustained scream.",
    ),
    "mixed": (
        "Mixed / ambiguous",
        "Several vocal techniques in the excerpt, overlapping delivery, or a passage that "
        "cannot confidently be given one of the other answers.",
    ),
    "none": (
        "No vocals",
        "An instrumental passage: no audible vocal performance.",
    ),
}
