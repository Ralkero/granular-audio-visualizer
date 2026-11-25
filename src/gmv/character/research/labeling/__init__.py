"""Phase A3: listening to short excerpts and labelling how the voice is delivered.

Research tooling, separate from the visualizer. ``dataset`` decides the excerpts and the
artist-disjoint split before anyone listens; ``store`` keeps every answer (append-only, so
nothing is lost and every change is kept); ``titles`` keeps each recording's song title
beside the dataset; ``playback`` plays exactly the ten seconds of an excerpt from the
original file; ``window`` is the labelling window; ``cli`` adds ``gmv character label``,
``gmv character prepare-labels`` and ``gmv character label-titles``.

The data lives outside Git in ``local-evidence/sonic-character/vocal-labeling/``.

**Answer schemes.** The set of answers has changed once. Every answer is saved with the
number of the scheme it belongs to; an answer from an earlier scheme is kept and shown as
such, never translated into the current answers.
"""

# The answers of the first version (2026-11-24). Kept only to read and show earlier answers.
SCHEME_1 = {
    "clean": ("Clean singing", "Melodic, pitched singing with a relatively clean vocal tone."),
    "scream": ("Screaming", "Harsh vocal delivery: screams, growls, shrieks and similar."),
    "shout": ("Shouting", "Forceful, yelled or speech-like delivery."),
    "mixed": ("Mixed / ambiguous", "Several techniques, or no single answer fits."),
    "none": ("No vocals", "An instrumental passage."),
}

# The current answers: (name, one-line summary). Exactly one per excerpt: the technique
# that dominates it.
SCHEME_2 = {
    "cleans": (
        "Cleans",
        "Conventional clean singing: a recognisable melodic performance without sustained "
        "harsh distortion.",
    ),
    "high_screams": (
        "High Screams",
        "Harsh delivery in a noticeably high register: sharp, piercing, shrill screaming.",
    ),
    "low_screams": (
        "Low Screams",
        "Harsh delivery in a lower or lower-middle register: thick, gritty screams or growls, "
        "not distinctly guttural.",
    ),
    "gutturals": (
        "Gutturals",
        "Extremely low, throat-heavy harsh vocals with a deep, wet, bubbling or cavernous sound.",
    ),
    "sing_screams": (
        "Sing-Screams",
        "Harsh or distorted singing that keeps a clearly recognisable melodic line.",
    ),
}

SCHEMES = {1: SCHEME_1, 2: SCHEME_2}
SCHEME = 2
CLASSES = SCHEMES[SCHEME]

# What the current answers mean, in full, as shown in the window's guide.
GUIDE = {
    "cleans": (
        "Conventional clean singing where the vocalist maintains a recognizable melodic "
        "performance without sustained harsh vocal distortion.",
        [
            "Clearly identifiable sung notes.",
            "Recognizable melodic phrases.",
            "Relatively natural, open, or resonant vocal tone.",
            "May include breathiness, emotional delivery, slight rasp, vocal grit, or belting.",
            "Can range from soft and atmospheric to powerful and forceful.",
        ],
        "A loud, forceful, or emotionally intense performance is still Cleans if the vocal "
        "technique remains predominantly clean.",
    ),
    "high_screams": (
        "Harsh vocal delivery in a noticeably high register, characterized by sharp, "
        "piercing, shrill, or upper-register screaming.",
        [
            "High-pitched screams and shrieks.",
            "Sharp, cutting vocal texture.",
            "Upper-register fry screams.",
            "Aggressive, piercing vocal tone.",
            "Usually lacks the sustained, clearly melodic character required for Sing-Screams.",
        ],
        "This category is about vocal register and texture, not simply loudness or aggression.",
    ),
    "low_screams": (
        "Harsh vocal delivery in a lower or lower-middle register, characterized by thick, "
        "gritty, forceful growls or screams that are not exceptionally deep or distinctly "
        "guttural.",
        [
            "Lower-register fry screams.",
            "Thick, coarse, gravelly screams.",
            "Chestier growls with a recognizable vocal tone.",
            "Lower-pitched harsh delivery without the especially deep, wet, bubbling quality "
            "of Gutturals.",
        ],
        "Low Screams are not simply quieter screams. They are differentiated by vocal "
        "register and texture.",
    ),
    "gutturals": (
        "Extremely low, throat-heavy harsh vocalizations with a deep, resonant, wet, "
        "bubbling, gurgling, or cavernous sound.",
        [
            "Exceptionally deep growls.",
            "Thick, wet, bubbling throat sounds.",
            "Deep rumbling or guttural resonance.",
            "Strongly distorted, throaty vocal texture.",
            "Deathcore-style guttural growls and similarly extreme low vocalizations.",
        ],
        "The defining characteristic is not pitch alone. It is the distinctive guttural "
        "texture and resonance. A relatively low scream should remain Low Screams if it "
        "sounds like a conventional lower-register scream or growl. Reserve Gutturals for "
        "distinctly deeper, more throat-dominated vocalizations.",
    ),
    "sing_screams": (
        "Harsh or distorted singing in which the vocalist retains a clearly recognizable "
        "melodic line or intentional pitched progression.",
        [
            "Sung melodies delivered with a harsh, raspy, distorted, or screamed vocal texture.",
            "Notes that retain recognizable musical pitch and melodic movement despite vocal "
            "distortion.",
            "Emotional scream-singing.",
            "Melodic passages between conventional clean singing and conventional screaming.",
            "Vocal phrases where harshness is part of the sung performance rather than an "
            "unpitched scream.",
        ],
        "The vocalist is still audibly singing a melody, even though the delivery is harsh. "
        "A scream that happens to have some pitch is not automatically a Sing-Scream. There "
        "must be a recognizable melodic or intentionally sustained pitched performance.",
    ),
}

RULE = (
    "Choose the one technique that dominates the excerpt. If it moves between several "
    "techniques with none dominating, or cannot reasonably be given one answer, skip it for "
    "review. If there is no voice at all, use “No vocals (skip)”: instrumental "
    "passages are not forced into a vocal answer."
)

# Ways to leave an excerpt without an answer. Neither is an answer.
SKIPS = {
    "later": "Skip / review later",
    "no_vocals": "No vocals (skip)",
}

assert set(GUIDE) == set(CLASSES)
