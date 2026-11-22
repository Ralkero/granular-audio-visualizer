"""Run a detector offline over decoded audio, exactly as the live assembler would feed it."""

from time import perf_counter

from gmv.dsp import Analyzer, WindowAssembler
from gmv.models import AnalysisConfig
from gmv.sources.base import AudioBlock

from .baseline_dsp import Analyzer as BaselineAnalyzer


def run_detector(audio, rate, detector="candidate", config=None, name="evaluation"):
    """Return (frames, events, dsp_times_ms, emitted_at).

    ``emitted_at[i]`` is the end of the analysis window that produced ``events[i]``: the
    earliest audio time at which that event could have been reported.
    """
    config = config or AnalysisConfig()
    analyzer = (BaselineAnalyzer if detector == "baseline" else Analyzer)(
        rate, audio.shape[1], config, "evaluation", name
    )
    assembler = WindowAssembler(config)
    frames, events, times, emitted_at = [], [], [], []
    step = round(rate * config.capture_ms / 1000)
    for pos in range(0, len(audio), step):
        chunk = audio[pos : pos + step]
        block = AudioBlock(
            chunk.astype("<f4").tobytes(),
            len(chunk),
            rate,
            chunk.shape[1],
            pos / rate,
            pos / rate,
            0,
        )
        for window, start, mono, _reset in assembler.push(block):
            began = perf_counter()
            frame, new_events = analyzer.process(window, start, mono, block, live=False)
            times.append((perf_counter() - began) * 1000)
            frames.append(frame)
            events.extend(new_events)
            emitted_at.extend([start + len(window) / rate] * len(new_events))
    return frames, events, times, emitted_at


def run_recognition(audio, rate, backend, config=None):
    """Estimates of a context recognition backend over decoded audio, as the live worker
    would produce them: the same ``Recognizer``, fed the same blocks. Returns
    (estimates, inference_times_ms). The worker's skipping under load is not simulated:
    every hop is asked."""
    from gmv.recognition import Recognizer, downmix

    config = config or AnalysisConfig()
    recognizer = Recognizer(backend)
    estimates, times = [], []
    step = round(rate * config.capture_ms / 1000)
    for pos in range(0, len(audio), step):
        if recognizer.push(downmix(audio[pos : pos + step]), rate, pos / rate):
            began = perf_counter()
            estimates.append(recognizer.estimate())
            times.append((perf_counter() - began) * 1000)
    return estimates, times


def estimate_intervals(estimates, label, states=("detected",)):
    """Merged (start, end) spans in which ``label`` was in one of ``states``."""
    spans = []
    for estimate in estimates:
        if dict(estimate.states).get(label) in states:
            if spans and estimate.start <= spans[-1][1] + 1e-9:
                spans[-1] = (spans[-1][0], estimate.end)
            else:
                spans.append((estimate.start, estimate.end))
    return spans


def run_instruments(audio, rate, config=None, sensitivity=None):
    """``InstrumentActivity`` messages of the fast instrument analyzers over decoded audio
    (frames, channels), produced by the same ``Analyzer`` pass the engine runs. ``config``
    names the analyzers; guitar when omitted. Returns (messages, analysis_times_ms)."""
    config = config or AnalysisConfig(instrument_analyzers=("guitar",), spectrum_bands=0)
    analyzer = Analyzer(rate, audio.shape[1], config, "evaluation", "evaluation", sensitivity)
    assembler = WindowAssembler(config)
    found, times = [], []
    step = round(rate * config.capture_ms / 1000)
    for pos in range(0, len(audio), step):
        chunk = audio[pos : pos + step]
        block = AudioBlock(
            chunk.astype("<f4").tobytes(), len(chunk), rate, chunk.shape[1], pos / rate, pos / rate, 0
        )
        for window, start, mono, _reset in assembler.push(block):
            analyzer.process(window, start, mono, block, live=False)
            found.extend(analyzer.instrument_activity)
            times.append(analyzer.instrument_ms)
    return found, times
