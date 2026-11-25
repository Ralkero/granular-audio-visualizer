"""Sonic Character: a slow-changing, measurable description of how the music sounds.

It complements ``gmv.visualizer.intensity.MusicalIntensity`` (how intense the music is at
this moment, relative to the rest of the same recording) with what kind of sound it is:
see docs/SONIC_CHARACTER.md. Nothing here touches the capture, the analysis pass, the
message schema or any renderer; it reads the frames those already produce.

* ``measures``   one feature frame -> level-independent measurements
* ``aggregate``  running averages over several lengths of audio time
* ``profile``    measurements -> the seven dimensions, with confidence
* ``tracker``    ``SonicCharacter``: the stream in, profiles out; segments and silence
* ``diagnose``   offline analysis of files and recorded sessions, reports (``gmv character``)
"""

from .profile import DIMENSION_NAMES, CharacterProfile
from .tracker import WINDOWS, SonicCharacter

__all__ = ["DIMENSION_NAMES", "WINDOWS", "CharacterProfile", "SonicCharacter"]
