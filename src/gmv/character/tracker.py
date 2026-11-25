"""Follows the character of the music from the analysis stream.

``SonicCharacter`` takes the feature frames (and the onset events, when it is given them)
in the order they were published and keeps running averages of what they measure over
several lengths of time. It never looks at a clock: its only time is the audio time the
frames carry, so the same stream gives the same result live, from a file, in a replay at
any speed and whatever the display's frame rate. It holds no audio and nothing of the
renderer.

What makes it start again, and what does not:

* **A newer segment** (a gap or a format change in the capture) starts everything afresh:
  nothing of the earlier segment is kept. The newest segment seen only moves forward.
* **A message of a segment that is over** is refused, unless it lies well before the newest
  audio time seen: then the timeline has started again (a replay from the start) and
  everything starts afresh with it.
* **Time going backwards, or jumping ahead by more than ten seconds**, inside one segment
  is the timeline starting again or a seek: everything starts afresh.
* **Silence** (and a short hole in the timeline) holds every window as it is. No time
  passes for them: a pause does not age the profile, and silence is not averaged in as
  "very quiet music". When music returns after ``BREAK_S`` or more of silence, the windows
  keep their averages but only ``LOOSEN`` of their weight, because the silence may have been
  the gap between two recordings: the old character is the best guess until the new music
  says otherwise, and it gives way faster and with lower confidence.
"""

import math

from ..models import FeatureFrame, MusicalEvent
from .aggregate import LevelDistribution, Windows
from .measures import COUNT, INDEX, measure, number, silent
from .profile import build

# name -> time constant in seconds. See docs/SONIC_CHARACTER.md for how they were chosen.
WINDOWS = {"short": 4.0, "medium": 20.0, "long": 90.0, "whole": math.inf}
LEVEL_TAU_S = 0.4  # the level the dynamics are read from is averaged over this, in dB
STEP_MOST_S = 0.25  # frames further apart than this have a hole between them
JUMP_S = 10.0  # a jump ahead of more than this inside a segment is a seek
RESTART_S = 2.0  # as gmv.visualizer.state: how far back an old segment's frame is a restart
BREAK_S = 3.0  # silence of this length may be the gap between two recordings
LOOSEN = 0.25  # share of the weight kept across such a silence
INTERVAL_S = (0.03, 2.0)  # two onsets this far apart are an interval; further is a break
FIRST_STEP_S = 0.02  # what the first frame stands for, before a second one gives the step

_PUNCH, _INTERVAL, _INTERVAL_OCT = INDEX["punch_db"], INDEX["interval_s"], INDEX["interval_oct"]


class SonicCharacter:
    """Feed it the stream with ``consume`` (or ``frame`` and ``onset``); read ``profile``."""

    def __init__(self, windows=None):
        self.names = tuple(windows or WINDOWS)
        lengths = tuple((windows or WINDOWS).values())
        self._windows = Windows(COUNT, lengths)
        self._levels = LevelDistribution(lengths)
        self._finite = [index for index, t in enumerate(lengths) if math.isfinite(t)]
        self.segment = None  # newest segment seen; survives nothing but a restart
        self.generation = 0  # grows each time everything starts afresh
        self._clear()

    def _clear(self):
        self._windows.clear()
        self._levels.clear()
        self.time = None  # audio time of the newest accepted frame
        self.newest = None
        self.heard = 0.0  # seconds of music taken in since the last fresh start
        self.quiet = 0.0  # seconds of silence or hole since music was last heard
        self.level = None  # level of the last 0.4 s, dB
        self.active = False  # whether the newest frame was music
        self.frames = 0  # frames accepted, silence included
        self.music_frames = 0
        self.spectrum_missing = 0  # music frames that carried no display spectrum
        self._last_onset = None
        self._pending = []

    def reset(self):
        """Forget everything, the segment included."""
        self.segment = None
        self.generation += 1
        self._clear()

    def _afresh(self, segment):
        self.segment = segment
        self.generation += 1
        self._clear()

    # -- input ---------------------------------------------------------------------------

    def consume(self, message):
        if isinstance(message, FeatureFrame):
            return self.frame(message)
        if isinstance(message, MusicalEvent) and message.event_type == "onset":
            return self.onset(message)
        return False

    def onset(self, event):
        """Take an onset event. They are optional: without them the spacing of onsets is
        simply not known. Returns whether it was accepted."""
        segment = event.segment
        if segment is not None and self.segment is not None and segment < self.segment:
            return False  # of a segment that is over
        self._pending.append(number(event.timestamp))
        return True

    def frame(self, frame):
        """Take one feature frame. Returns whether it was accepted."""
        time, segment = number(frame.timestamp), frame.segment
        if not math.isfinite(time):
            return False
        if self.segment is None:
            self.segment = segment
        elif segment < self.segment:
            if self.newest is not None and time > self.newest - RESTART_S:
                return False  # a late frame of a segment that is over
            self._afresh(segment)  # the timeline started again
        elif segment > self.segment:
            self._afresh(segment)
        if self.time is None:
            elapsed = FIRST_STEP_S
        else:
            elapsed = time - self.time
            if elapsed == 0.0:
                return False  # the same frame twice
            if elapsed < 0.0 or elapsed > JUMP_S:
                self._afresh(segment)
                elapsed = FIRST_STEP_S
        self.time = time
        self.newest = time if self.newest is None else max(self.newest, time)
        self.frames += 1
        # A frame stands for the time since the one before; after a hole, for one step.
        step = elapsed if elapsed <= STEP_MOST_S else FIRST_STEP_S
        if silent(frame):
            self.quiet += elapsed
            self.active = False
            self._pending.clear()
            return True
        self.quiet += elapsed - step  # a hole before this frame is not music either
        if self.quiet >= BREAK_S and self.heard > 0.0:
            for window in self._finite:
                self._windows.weight[window] *= LOOSEN
                self._windows.total[window] *= LOOSEN
                self._windows.square[window] *= LOOSEN
                self._levels.bins[window] *= LOOSEN
        if self.quiet >= STEP_MOST_S:
            self.level = None  # the level before the silence says nothing about this frame
            self._last_onset = None
        self.quiet = 0.0
        self.active = True
        self.heard += step
        values = measure(frame)
        level = values[0]
        if self.level is None:
            self.level = level
        else:
            values[_PUNCH] = max(0.0, level - self.level)
            self.level = level + (self.level - level) * math.exp(-step / LEVEL_TAU_S)
        for onset in self._pending:
            if self._last_onset is not None:
                interval = onset - self._last_onset
                if INTERVAL_S[0] <= interval <= INTERVAL_S[1]:
                    values[_INTERVAL] = interval
                    values[_INTERVAL_OCT] = math.log2(interval)
            self._last_onset = onset
        self._pending.clear()
        self.music_frames += 1
        if not math.isfinite(values[INDEX["tilt_db_oct"]]):
            self.spectrum_missing += 1
        self._windows.add(values, step)
        self._levels.add(self.level, step)
        return True

    # -- output --------------------------------------------------------------------------

    def profile(self, window="long"):
        """The character over one of the windows, as a ``CharacterProfile``."""
        index = self.names.index(window)
        return build(
            window=window,
            window_s=self._windows.lengths[index],
            time=self.time,
            segment=self.segment,
            active=self.active,
            heard_s=self.heard,
            weight=self._windows.weight[index],
            mean=self._windows.mean(index),
            spread=self._windows.spread(index),
            level_quantiles=self._levels.quantiles(index, (0.1, 0.5, 0.95)),
            spectrum_missing=self.spectrum_missing / max(1, self.music_frames),
        )

    def profiles(self):
        return {name: self.profile(name) for name in self.names}
