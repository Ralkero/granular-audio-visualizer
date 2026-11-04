import numpy as np
import soundfile as sf
from gmv.engine import Engine
from gmv.models import AnalysisConfig
from gmv.sources.file import FileAudioSource


def test_debug_recording_is_explicit_and_capped(tmp_path):
    audio = tmp_path / "test.wav"
    sf.write(audio, np.zeros(48000 * 3), 48000)
    engine = Engine(
        FileAudioSource(audio, realtime=False),
        AnalysisConfig(debug_audio_seconds=1),
        session_root=tmp_path,
    )
    engine.start()
    assert engine.done.wait(10)
    assert not engine.error
    files = list(engine.writer.path.rglob("*.wav"))
    assert files and sum(sf.info(p).duration for p in files) <= 1.0001
    assert engine.writer.manifest["raw_audio_saved"]
