"""The existing analysis pass, run once over decoded audio for the research measurements.

It is ``gmv.dsp.Analyzer`` exactly as the engine runs it, with the existing vocal analyzer
switched on (``instrument_analyzers=("vocals",)``) and, when a weights file is named, the
existing activity model. Neither changes any field the Phase A profile reads, so the
profile built from these frames is the Phase A profile.
"""

from ...dsp import Analyzer, WindowAssembler
from ...models import AnalysisConfig
from ...sources.base import AudioBlock


def run(audio, rate, activity_model=None):
    """Returns (messages for the Phase A profile, vocal analyzer rows, frames)."""
    config = AnalysisConfig(
        instrument_analyzers=("vocals",),
        activity_model_path=str(activity_model) if activity_model else None,
    )
    try:
        analyzer = Analyzer(rate, audio.shape[1], config, "research", "file_audio")
    except ValueError:
        if not activity_model:
            raise
        # The activity model cannot be built at every sample rate (its narrowest bands
        # need the analysis window of 44.1 kHz or more). Then there is no model, which the
        # measurements report as "unknown", as for a run without one.
        config = config.model_copy(update={"activity_model_path": None})
        analyzer = Analyzer(rate, audio.shape[1], config, "research", "file_audio")
    assembler = WindowAssembler(config)
    messages, rows, frames = [], [], []
    step = round(rate * config.capture_ms / 1000)
    for position in range(0, len(audio), step):
        chunk = audio[position : position + step]
        block = AudioBlock(
            chunk.astype("<f4").tobytes(), len(chunk), rate, chunk.shape[1],
            position / rate, position / rate, 0,
        )
        for window, start, mono, _reset in assembler.push(block):
            frame, events = analyzer.process(window, start, mono, block, live=False)
            frames.append(frame)
            messages.append(frame)
            messages.extend(e for e in events if e.event_type == "onset")
            rows.extend(analyzer.instrument_activity)
    return messages, rows, frames
