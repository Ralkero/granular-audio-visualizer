"""Experimental measurements for Sonic Character (phase A2). Research only.

Phase A's profile describes the texture of the mix. These prototypes ask what it misses:
how the lead line or voice behaves, what the harmony does, and how the rhythm feels. They
run offline on recordings named explicitly, beside the unchanged Phase A profile, and
nothing in the application imports them. Findings: docs/SONIC_CHARACTER_RESEARCH.md.

* ``frontend``  one decode, one spectrogram, harmonic/percussive split, shared by all
* ``tonal``     harmonic share, pitch classes, harmonic movement, key clarity
* ``lead``      the centre lead line (the existing vocal analyzer's rows), vocal activity
* ``rhythm``    pulse, tempo ambiguity, events per beat, off-beat weight, repetition
* ``profile``   every family over a recording, in blocks and as a whole
* ``evaluate``  group-disjoint corpus statistics; ``stems``: checks against stems and beats

Every value is a ``(value, confidence)`` pair. ``None`` means "could not be measured
here", which is never the same as zero: no reliable pitch is not "no melody", and no
reliable pulse is not "an irregular rhythm".
"""

RESEARCH_VERSION = "a2.1"
