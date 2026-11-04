import numpy as np
from gmv.engine import Engine
from gmv.models import AnalysisConfig
from gmv.sources.base import AudioBlock, AudioSource


def test_overloaded_live_source_stays_bounded_and_counts_frames():
    class Burst(AudioSource):
        kind = "synthetic_burst"

        def run(self, output, stop, status):
            data = np.zeros((480, 2), dtype=np.float32).tobytes()
            for i in range(5000):
                output.put(AudioBlock(data, 480, 48000, 2, i / 100, 0, 0))
            output.close()

    engine = Engine(Burst(), AnalysisConfig(queue_blocks=8), logging=False)
    engine.start()
    assert engine.done.wait(5) and not engine.error
    assert engine.audio.dropped > 0
    assert engine.audio.high_water == 8
    assert engine.audio.dropped_frames == engine.audio.dropped * 480
    assert len(engine.dsp_times) <= 4096
