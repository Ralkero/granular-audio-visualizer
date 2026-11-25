"""Every experimental family over one recording, in blocks and as a whole, beside the
unchanged Phase A profile.

Families do not share one window. The lead line is read every 6 s (a phrase or two), the
harmony and the rhythm every 12 s (several bars: a pulse cannot be established in less).
Blocks overlap by half. A recording's value is the mean of its blocks weighted by their
confidence; a value no block could state stays unknown.
"""

import numpy as np

from .. import diagnose
from . import RESEARCH_VERSION, gmvpass, lead, rhythm, tonal
from .frontend import Analysis, load

FAMILIES = {"tonal": tonal.DEFINITIONS, "lead": lead.DEFINITIONS, "rhythm": rhythm.DEFINITIONS}
BLOCK_S = {"tonal": 12.0, "lead": 6.0, "rhythm": 12.0}
MEDIANS = {"tempo_bpm", "phrase_s"}
NAMES = tuple(f"{family}.{name}" for family, names in FAMILIES.items() for name in names)


def blocks(duration, length):
    starts = np.arange(0.0, max(duration - length / 2, length / 2), length / 2)
    return [(float(s), float(min(s + length, duration))) for s in starts]


def summarise(values, middle=False):
    """Confidence-weighted mean of block results [(value, confidence)], with the mean
    confidence over all blocks (so blocks that could say nothing count against it). With
    ``middle`` the value is the confidence-weighted median instead."""
    known = [(v, c) for v, c in values if v is not None and c > 0]
    if not known:
        return {"value": None, "confidence": 0.0, "spread": None, "available": 0.0}
    v, c = np.array(known).T
    mean = float((v * c).sum() / c.sum())
    if middle:  # a tempo read at two metrical levels has no meaningful mean
        order = np.argsort(v)
        mean = float(v[order][np.searchsorted(np.cumsum(c[order]), c.sum() / 2)])
    return {
        "value": mean,
        "confidence": float(c.sum() / len(values)),
        "spread": float(np.sqrt(((v - mean) ** 2 * c).sum() / c.sum())),
        "available": len(known) / len(values),
    }


def measure_audio(audio, rate, activity_model=None, baseline=True):
    """The experimental profile of decoded audio: per family, per-block evolution and a
    whole-recording summary; and, with ``baseline``, the Phase A diagnostics."""
    analysis = Analysis(audio, rate)
    messages, vocal_rows, frames = gmvpass.run(audio, rate, activity_model)
    rows = lead.Rows(vocal_rows, frames)
    out = {"research_version": RESEARCH_VERSION, "families": {}, "evolution": {}}
    caches = {"tonal": {}, "rhythm": {}}
    for family in FAMILIES:
        spans = blocks(analysis.duration_s, BLOCK_S[family])
        if family == "lead":
            found = [lead.measure(rows, *span) for span in spans]
        else:
            module = tonal if family == "tonal" else rhythm
            found = [module.measure(analysis, *span, cache=caches[family]) for span in spans]
        out["families"][family] = {
            name: summarise([block[name] for block in found], name in MEDIANS)
            for name in FAMILIES[family]
        }
        out["evolution"][family] = {
            "start_s": [round(s, 2) for s, _ in spans],
            "end_s": [round(e, 2) for _, e in spans],
            **{
                name: [None if block[name][0] is None else round(block[name][0], 4) for block in found]
                for name in FAMILIES[family]
            },
        }
    warnings = []
    if not analysis.stereo:
        warnings.append("mono: no stereo image, so the 'centre' is the whole mix and the lead-line values describe the mix")
    if not rows.model:
        warnings.append("no_activity_model: nothing here knows when a voice sounds; vocal_* values are unknown")
    for family, names in out["families"].items():
        low = [name for name, found in names.items() if found["value"] is not None and found["confidence"] < 0.4]
        if low:
            warnings.append(f"low_confidence {family}: " + ", ".join(low))
    out["tuning_thirds_of_semitone"] = caches["tonal"].get("tuning")
    out["stereo"] = analysis.stereo
    out["warnings"] = warnings
    if baseline:
        source = {"kind": "audio_file", "sample_rate": int(rate), "channels": int(audio.shape[1]),
                  "duration_s": len(audio) / rate, "gain_db": 0.0, "path": ""}
        out["phase_a"] = diagnose.analyse(messages, source)
    return out


def measure_path(path, gain_db=0.0, activity_model=None, seconds=None):
    audio, rate = load(path, gain_db, seconds)
    out = measure_audio(audio, rate, activity_model)
    out["phase_a"]["source"].update(path=str(path), gain_db=gain_db)
    return out


def vector(found, names=NAMES):
    """The whole-recording values in the order of ``names``, NaN where unknown."""
    values = []
    for name in names:
        family, key = name.split(".", 1)
        value = found["families"][family][key]["value"]
        values.append(np.nan if value is None else value)
    return np.array(values)
