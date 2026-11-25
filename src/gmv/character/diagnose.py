"""Offline diagnostics of Sonic Character: what it makes of audio files and recorded sessions.

An audio file is decoded into memory and put through the same analysis pass the engine
runs (``gmv.dsp``), as fast as it will go; a recorded session's messages are read back in
the order they were published. Either way the stream goes through a ``SonicCharacter`` and
a ``MusicalIntensity`` side by side, and what comes out is written as JSON, a report in
plain text (Markdown) and one chart per recording (SVG).

Nothing here runs while music plays in the application, nothing is captured or recorded,
and no file given to it is changed, copied or sent anywhere. The reports name the files
they were made from, so they belong beside the other local evidence, not in the repository.
"""

import json
import math
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np

from ..models import AnalysisConfig, FeatureFrame, MusicalEvent
from ..visualizer.intensity import MusicalIntensity
from .measures import INDEX, MEASURES, NAMES, measure, silent
from .profile import DIMENSION_NAMES, DIMENSIONS
from .tracker import WINDOWS, SonicCharacter

SAMPLE_S = 1.0  # how often the evolution is sampled
SECTION_SPAN_S = 8.0  # each side of a possible section boundary
SECTION_LEAST_S = 12.0  # shortest section
SECTION_CHANGE = 0.12  # mean change over the dimensions that makes a boundary
THREAD_LIMITS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
DISTRIBUTION = (5, 25, 50, 75, 95)
SHOWN = (  # measurements whose distribution over the frames is reported
    "centroid_oct",
    "tilt_db_oct",
    "low_share",
    "flatness_db",
    "occupancy",
    "ripple_db",
    "contrast_db",
    "flux",
    "attack",
    "crest_db",
    "punch_db",
    "onset_rate",
    "pulse_clarity",
)


# -- reading a stream ---------------------------------------------------------------------


def audio_messages(path, gain_db=0.0, config=None, seconds=None):
    """Feature frames and onset events of an audio file, in publication order, with the
    time the analysis pass took per frame. ``gain_db`` scales the decoded samples in memory
    (to test that the level does not matter); the file is only read."""
    from ..corpus.registry import read_audio
    from ..dsp import Analyzer, WindowAssembler
    from ..sources.base import AudioBlock

    audio, rate = read_audio(path)
    if seconds:
        audio = audio[: round(seconds * rate)]
    if gain_db:
        audio = audio * np.float32(10.0 ** (gain_db / 20.0))
    config = config or AnalysisConfig()
    analyzer = Analyzer(rate, audio.shape[1], config, "character", "file_audio")
    assembler = WindowAssembler(config)
    messages, times = [], []
    step = round(rate * config.capture_ms / 1000)
    for position in range(0, len(audio), step):
        chunk = audio[position : position + step]
        block = AudioBlock(
            chunk.astype("<f4").tobytes(),
            len(chunk),
            rate,
            chunk.shape[1],
            position / rate,
            position / rate,
            0,
        )
        for window, start, mono, _reset in assembler.push(block):
            began = perf_counter()
            frame, events = analyzer.process(window, start, mono, block, live=False)
            times.append((perf_counter() - began) * 1000)
            messages.append(frame)
            messages.extend(e for e in events if e.event_type == "onset")
    source = {
        "kind": "audio_file",
        "path": str(path),
        "sample_rate": rate,
        "channels": int(audio.shape[1]),
        "duration_s": len(audio) / rate,
        "gain_db": gain_db,
    }
    return messages, source, times


def session_messages(path):
    """Feature frames and onset events of a recorded session, in the order they were
    published. Nothing is analysed again."""
    from ..contract import parse_message

    path = Path(path)
    if path.is_dir():
        path = path / "messages.jsonl"
    messages = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            message = parse_message(json.loads(line))
            if isinstance(message, FeatureFrame) or (
                isinstance(message, MusicalEvent) and message.event_type == "onset"
            ):
                messages.append(message)
    frames = [m for m in messages if isinstance(m, FeatureFrame)]
    source = {
        "kind": "recorded_session",
        "path": str(path),
        "sample_rate": frames[0].sample_rate if frames else None,
        "channels": frames[0].channels if frames else None,
        "duration_s": frames[-1].window_end - frames[0].timestamp if frames else 0.0,
        "gain_db": 0.0,
    }
    return messages, source, []


# -- one recording ------------------------------------------------------------------------


def rounded(value, digits=4):
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def brief(profile):
    """A profile as plain data: values and confidences by dimension."""
    return {
        "evidence_s": rounded(profile.evidence_s, 2),
        "values": {d.name: rounded(d.value) for d in profile.dimensions},
        "confidence": {d.name: rounded(d.confidence) for d in profile.dimensions},
        "warnings": list(profile.warnings),
    }


def whole(messages):
    """The profile of a stretch of messages taken as one thing."""
    tracker = SonicCharacter({"whole": math.inf})
    for message in messages:
        tracker.consume(message)
    return tracker.profile("whole")


def sections(times, short, least=SECTION_LEAST_S, span=SECTION_SPAN_S, change=SECTION_CHANGE):
    """Where the character changes: boundaries (audio times) at which the short-term
    profile of the seconds after differs most from that of the seconds before."""
    times, short = np.asarray(times), np.asarray(short, dtype=float)
    if len(times) < 2:
        return []
    step = float(np.median(np.diff(times)))
    reach = max(1, round(span / step))
    novelty = np.zeros(len(times))
    for index in range(reach, len(times) - reach):
        before = np.nanmean(short[index - reach : index], axis=0)
        after = np.nanmean(short[index : index + reach], axis=0)
        novelty[index] = float(np.nanmean(np.abs(after - before)))
    found = []
    for index in np.argsort(-novelty):
        if novelty[index] < change:
            break
        time = float(times[index])
        if all(abs(time - other) >= least for other in found):
            found.append(time)
    return sorted(t for t in found if t - times[0] >= least and times[-1] - t >= least)


def analyse(messages, source, dsp_ms=()):
    """Everything the diagnostics say about one stream of messages."""
    tracker, intensity = SonicCharacter(), MusicalIntensity()
    frames = [m for m in messages if isinstance(m, FeatureFrame)]
    evolution = {"time": [], "intensity": [], "active": []}
    for name in WINDOWS:
        if name != "whole":
            evolution[name] = []
            evolution[name + "_confidence"] = []
    rows, intensities, spent, due = [], [], 0.0, None
    for message in messages:
        began = perf_counter()
        tracker.consume(message)
        spent += perf_counter() - began
        if not isinstance(message, FeatureFrame):
            continue
        value = intensity.step(message)
        if not silent(message):
            rows.append(measure(message))
            intensities.append(value)
        if due is None:
            due = message.timestamp + SAMPLE_S
        if message.timestamp >= due:
            due += SAMPLE_S
            evolution["time"].append(rounded(message.timestamp, 3))
            evolution["intensity"].append(rounded(value))
            evolution["active"].append(tracker.active)
            for name in WINDOWS:
                if name != "whole":
                    profile = tracker.profile(name)
                    evolution[name].append([rounded(v) for v in profile.values()])
                    evolution[name + "_confidence"].append(
                        [rounded(v) for v in profile.confidences()]
                    )
    overall = tracker.profile("whole")
    rows = np.asarray(rows, dtype=float).reshape(-1, len(NAMES))
    distributions = {}
    for name in SHOWN:
        column = rows[:, INDEX[name]]
        column = column[np.isfinite(column)]
        if len(column):
            distributions[name] = {
                "unit": MEASURES[name][0],
                "percentiles": dict(
                    zip(
                        (f"p{p}" for p in DISTRIBUTION),
                        (rounded(v) for v in np.percentile(column, DISTRIBUTION)),
                        strict=True,
                    )
                ),
            }
    # Sections, each profiled on its own from the messages that fall inside it.
    short = [[math.nan if v is None else v for v in row] for row in evolution["short"]]
    edges = sections(evolution["time"], short) if short else []
    start = frames[0].timestamp if frames else 0.0
    end = frames[-1].timestamp if frames else 0.0
    bounds = [start, *edges, end + 1e-6]
    parts = []
    for begin, finish in zip(bounds, bounds[1:], strict=False):
        inside = [m for m in messages if begin <= m.timestamp < finish]
        part = whole(inside)
        own = MusicalIntensity()
        levels = [own.step(m) for m in inside if isinstance(m, FeatureFrame)]
        taken = [
            v
            for t, v in zip(evolution["time"], evolution["intensity"], strict=True)
            if begin <= t < finish
        ]
        parts.append(
            {
                "start_s": rounded(begin, 2),
                "end_s": rounded(min(finish, end), 2),
                "intensity_mean": rounded(np.mean(taken)) if taken else rounded(np.mean(levels)),
                **brief(part),
            }
        )
    music_s = tracker.heard
    # A file ends in silence as a rule: that the profile is "held" there says nothing.
    warnings = [w for w in overall.warnings if not w.startswith("held:")]
    if source.get("sample_rate") and source["sample_rate"] < 32000:
        warnings.append(
            "low_sample_rate: nothing above %d Hz was recorded; brightness reads lower"
            % (source["sample_rate"] // 2)
        )
    if music_s < 30.0:
        warnings.append(
            "short_recording: under thirty seconds of music; dynamics and the long window "
            "have little to go on"
        )
    low_confidence = [d.name for d in overall.dimensions if d.confidence < 0.5]
    if low_confidence:
        warnings.append("low_confidence: " + ", ".join(low_confidence))
    intensities = np.asarray(intensities)
    return {
        "source": source,
        "music_s": rounded(music_s, 2),
        "frames": len(frames),
        "overall": overall.model_dump(mode="json"),
        "windows_at_end": {
            name: brief(tracker.profile(name)) for name in WINDOWS if name != "whole"
        },
        "intensity": {
            "mean": rounded(intensities.mean()) if len(intensities) else None,
            "percentiles": dict(
                zip(
                    (f"p{p}" for p in DISTRIBUTION),
                    (rounded(v) for v in np.percentile(intensities, DISTRIBUTION)),
                    strict=True,
                )
            )
            if len(intensities)
            else {},
        },
        "distributions": distributions,
        "evolution": evolution,
        "sections": parts,
        "warnings": warnings,
        "cost": {
            "character_us_per_frame": rounded(spent / max(1, len(frames)) * 1e6, 2),
            "analysis_ms_per_frame": rounded(float(np.mean(dsp_ms)), 4) if len(dsp_ms) else None,
        },
    }


def analyse_path(path, gain_db=0.0, seconds=None):
    """Diagnostics of an audio file, or of a recorded session (a folder or its
    ``messages.jsonl``)."""
    path = Path(path)
    if path.is_dir() or path.suffix == ".jsonl":
        return analyse(*session_messages(path))
    return analyse(*audio_messages(path, gain_db, seconds=seconds))


# -- several recordings -------------------------------------------------------------------


def vector(result):
    values = result["overall"]["dimensions"]
    return np.array([math.nan if d["value"] is None else d["value"] for d in values])


def compare(results):
    """How the recordings differ from each other, and how that relates to intensity."""
    names = list(results)
    profiles = np.array([vector(results[n]) for n in names])
    intensity = np.array([results[n]["intensity"]["mean"] or math.nan for n in names])
    out = {
        "recordings": names,
        "spread": {
            d: {
                "least": rounded(np.nanmin(profiles[:, i])),
                "most": rounded(np.nanmax(profiles[:, i])),
                "standard_deviation": rounded(np.nanstd(profiles[:, i])),
            }
            for i, d in enumerate(DIMENSION_NAMES)
        },
        "intensity_spread": {
            "least": rounded(np.nanmin(intensity)),
            "most": rounded(np.nanmax(intensity)),
            "standard_deviation": rounded(np.nanstd(intensity)),
        },
    }
    if len(names) >= 4:
        table = np.column_stack((profiles, intensity))
        table = table[np.isfinite(table).all(axis=1)]
        if len(table) >= 4 and (table.std(axis=0) > 0).all():
            matrix = np.corrcoef(table.T)
            labels = (*DIMENSION_NAMES, "intensity")
            out["correlation"] = {
                a: {b: rounded(matrix[i, j], 2) for j, b in enumerate(labels)}
                for i, a in enumerate(labels)
            }
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            distance = float(np.nanmean(np.abs(profiles[i] - profiles[j])))
            pairs.append((names[i], names[j], distance, abs(float(intensity[i] - intensity[j]))))
    pairs = [p for p in pairs if math.isfinite(p[2]) and math.isfinite(p[3])]

    def described(pair):
        return {
            "a": pair[0],
            "b": pair[1],
            "character_distance": rounded(pair[2]),
            "intensity_difference": rounded(pair[3]),
        }

    if pairs:
        # Same intensity, different character: the largest character distance among the
        # pairs whose mean intensities are within 0.03 of each other.
        same = [p for p in pairs if p[3] <= 0.03]
        if same:
            out["similar_intensity_different_character"] = described(max(same, key=lambda p: p[2]))
        # Different intensity, similar character: the largest intensity difference among
        # the tenth of the pairs that are closest in character.
        closest = sorted(pairs, key=lambda p: p[2])[: max(1, len(pairs) // 10)]
        out["different_intensity_similar_character"] = described(max(closest, key=lambda p: p[3]))
    # Within one recording: intensity moving while the character stays, and the character
    # (roughness, density) really changing.
    steady, moving = None, None
    texture = [DIMENSION_NAMES.index("roughness"), DIMENSION_NAMES.index("density")]
    for name in names:
        evolution = results[name]["evolution"]
        active = np.array(evolution["active"], dtype=bool)
        if active.sum() < 60:
            continue
        values = np.array(
            [[math.nan if v is None else v for v in row] for row in evolution["medium"]]
        )[active][20:]
        levels = np.array(evolution["intensity"], dtype=float)[active][20:]
        if len(values) < 30:
            continue
        swing = float(np.percentile(levels, 95) - np.percentile(levels, 5))
        drift = float(np.nanmean(np.nanpercentile(values, 95, axis=0) - np.nanpercentile(values, 5, axis=0)))
        change = float(
            np.nanmax(
                np.nanpercentile(values[:, texture], 95, axis=0)
                - np.nanpercentile(values[:, texture], 5, axis=0)
            )
        )
        if steady is None or swing - 2 * drift > steady[1] - 2 * steady[2]:
            steady = (name, swing, drift)
        if moving is None or change > moving[1]:
            moving = (name, change, swing)
    if steady:
        out["intensity_changes_character_holds"] = {
            "recording": steady[0],
            "intensity_swing": rounded(steady[1]),
            "mean_dimension_swing_medium_window": rounded(steady[2]),
        }
    if moving:
        out["texture_or_density_changes"] = {
            "recording": moving[0],
            "largest_swing_of_roughness_or_density_medium_window": rounded(moving[1]),
            "intensity_swing": rounded(moving[2]),
        }
    return out


def level_check(low, high):
    """How far each dimension moved between two analyses of the same audio at two levels."""
    a, b = vector(low), vector(high)
    return {
        d: rounded(abs(float(a[i] - b[i])) if np.isfinite([a[i], b[i]]).all() else None)
        for i, d in enumerate(DIMENSION_NAMES)
    }


# -- writing ------------------------------------------------------------------------------

COLOURS = ("#c0392b", "#e6a100", "#2e86c1", "#1abc9c", "#8e44ad", "#27ae60", "#7f8c8d")


def chart(result, title, window="medium"):
    """The evolution of one recording as an SVG: the dimensions over the given window, and
    ``MusicalIntensity`` as a dashed black line, against audio time."""
    evolution = result["evolution"]
    times = evolution["time"]
    width, height, left, top, bottom = 960, 360, 46, 28, 30
    plot_w, plot_h = width - left - 150, height - top - bottom
    if len(times) < 2:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"/>'
    first, last = times[0], times[-1]

    def path(values):
        points, pen = [], "M"
        for time, value in zip(times, values, strict=True):
            if value is None:
                pen = "M"
                continue
            x = left + (time - first) / (last - first) * plot_w
            y = top + (1 - value) * plot_h
            points.append(f"{pen}{x:.1f},{y:.1f}")
            pen = "L"
        return " ".join(points)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        'font-family="sans-serif" font-size="11">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{left}" y="16" font-size="13">{escape(title)}: {window} window, '
        "and MusicalIntensity (dashed)</text>",
    ]
    for level in (0, 0.25, 0.5, 0.75, 1):
        y = top + (1 - level) * plot_h
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd"/>'
            f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">{level:g}</text>'
        )
    for section in result["sections"][1:]:
        x = left + (section["start_s"] - first) / (last - first) * plot_w
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#999" '
            'stroke-dasharray="2,3"/>'
        )
    ticks = max(1, round((last - first) / 8 / 10) * 10)
    for tick in range(0, int(last) + 1, ticks):
        if tick >= first:
            x = left + (tick - first) / (last - first) * plot_w
            parts.append(
                f'<text x="{x:.1f}" y="{height - 10}" text-anchor="middle">{tick} s</text>'
            )
    for index, name in enumerate(DIMENSION_NAMES):
        values = [row[index] for row in evolution[window]]
        parts.append(
            f'<path d="{path(values)}" fill="none" stroke="{COLOURS[index]}" stroke-width="1.6"/>'
        )
        y = top + 14 + index * 18
        parts.append(
            f'<line x1="{left + plot_w + 12}" y1="{y - 4}" x2="{left + plot_w + 30}" y2="{y - 4}" '
            f'stroke="{COLOURS[index]}" stroke-width="2"/>'
            f'<text x="{left + plot_w + 36}" y="{y}">{name}</text>'
        )
    parts.append(
        f'<path d="{path(evolution["intensity"])}" fill="none" stroke="black" '
        'stroke-width="1.2" stroke-dasharray="5,4"/>'
    )
    y = top + 14 + len(DIMENSION_NAMES) * 18
    parts.append(
        f'<line x1="{left + plot_w + 12}" y1="{y - 4}" x2="{left + plot_w + 30}" y2="{y - 4}" '
        'stroke="black" stroke-dasharray="5,4"/>'
        f'<text x="{left + plot_w + 36}" y="{y}">intensity</text></svg>'
    )
    return "\n".join(parts)


def escape(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def cell(value, digits=2):
    return "–" if value is None else f"{value:.{digits}f}"


def report(results, comparison=None, level_checks=None):
    """The human-readable report (Markdown)."""
    lines = [
        "# Sonic Character diagnostics",
        "",
        "Every value is between 0 and 1 unless a unit is given. `conf` is how far a value can "
        "be relied on, not how strong it is. `intensity` is the existing MusicalIntensity, "
        "shown for comparison; Sonic Character does not use it. These are measurements of "
        "the sound (see docs/SONIC_CHARACTER.md), not of mood, and they name no genre.",
        "",
        "## Overall profiles",
        "",
        "| Recording | " + " | ".join(DIMENSION_NAMES) + " | intensity (mean) | music (s) |",
        "|---|" + "---:|" * (len(DIMENSION_NAMES) + 2),
    ]
    for name, result in results.items():
        values = [d["value"] for d in result["overall"]["dimensions"]]
        lines.append(
            f"| {name} | "
            + " | ".join(cell(v) for v in values)
            + f" | {cell(result['intensity']['mean'])} | {result['music_s']:.0f} |"
        )
    if comparison:
        lines += ["", "## Across the recordings", ""]
        lines.append("| | least | most | standard deviation |")
        lines.append("|---|---:|---:|---:|")
        for name, spread in comparison["spread"].items():
            lines.append(
                f"| {name} | {cell(spread['least'])} | {cell(spread['most'])} | "
                f"{cell(spread['standard_deviation'])} |"
            )
        spread = comparison["intensity_spread"]
        lines.append(
            f"| intensity (mean) | {cell(spread['least'])} | {cell(spread['most'])} | "
            f"{cell(spread['standard_deviation'])} |"
        )
        if "correlation" in comparison:
            labels = list(comparison["correlation"])
            lines += [
                "",
                "Correlation between the overall values across the recordings (1 or -1: two "
                "columns say the same thing; 0: they are unrelated):",
                "",
                "| | " + " | ".join(labels) + " |",
                "|---|" + "---:|" * len(labels),
            ]
            for a in labels:
                lines.append(
                    f"| {a} | " + " | ".join(cell(comparison["correlation"][a][b]) for b in labels) + " |"
                )
        found = comparison.get("similar_intensity_different_character")
        if found:
            lines += [
                "",
                f"**Similar intensity, different character:** {found['a']} and {found['b']} "
                f"(intensity differs by {found['intensity_difference']:.2f}; the profiles "
                f"differ by {found['character_distance']:.2f} on average per dimension).",
            ]
        found = comparison.get("different_intensity_similar_character")
        if found:
            lines += [
                "",
                f"**Different intensity, similar character:** {found['a']} and {found['b']} "
                f"(intensity differs by {found['intensity_difference']:.2f}; the profiles "
                f"differ by {found['character_distance']:.2f}).",
            ]
        found = comparison.get("intensity_changes_character_holds")
        if found:
            lines += [
                "",
                f"**Intensity changes, character holds:** {found['recording']} (intensity "
                f"swings by {found['intensity_swing']:.2f} inside the recording; the "
                f"dimensions over the medium window swing by "
                f"{found['mean_dimension_swing_medium_window']:.2f} on average).",
            ]
        found = comparison.get("texture_or_density_changes")
        if found:
            lines += [
                "",
                f"**Texture or density really changes:** {found['recording']} (roughness or "
                f"density over the medium window swings by "
                f"{found['largest_swing_of_roughness_or_density_medium_window']:.2f}).",
            ]
    if level_checks:
        lines += [
            "",
            "## The same audio at another level",
            "",
            "Largest change of each dimension when the decoded audio was scaled before "
            "analysis (0 means the level made no difference):",
            "",
            "| Recording | gain (dB) | " + " | ".join(DIMENSION_NAMES) + " |",
            "|---|---:|" + "---:|" * len(DIMENSION_NAMES),
        ]
        for name, gain, moved in level_checks:
            lines.append(
                f"| {name} | {gain:+.0f} | " + " | ".join(cell(moved[d], 3) for d in DIMENSION_NAMES) + " |"
            )
    for name, result in results.items():
        overall = result["overall"]
        lines += ["", f"## {name}", ""]
        source = result["source"]
        lines.append(
            f"{source['kind'].replace('_', ' ')}, {source['duration_s']:.0f} s, "
            f"{result['music_s']:.0f} s of it music; {source['sample_rate']} Hz. "
            f"Tempo reported by the engine: "
            + (f"{overall['tempo_bpm']:.0f} BPM" if overall["tempo_bpm"] else "none")
            + "."
        )
        if result["warnings"]:
            lines += [""] + [f"- Warning: {w}" for w in result["warnings"]]
        lines += [
            "",
            "| Dimension | whole | conf | short (end) | medium (end) | long (end) | made from |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        at_end = result["windows_at_end"]
        for dimension in overall["dimensions"]:
            made = "; ".join(
                f"{c['measurement']} {cell(c['value'])} {c['unit']} → {cell(c['normalized'])}"
                for c in dimension["components"] + ([dimension["gate"]] if dimension["gate"] else [])
            )
            key = dimension["name"]
            lines.append(
                f"| {key} ({dimension['meaning']}) | {cell(dimension['value'])} | "
                f"{cell(dimension['confidence'])} | {cell(at_end['short']['values'][key])} | "
                f"{cell(at_end['medium']['values'][key])} | {cell(at_end['long']['values'][key])} | "
                f"{made} |"
            )
        intensity = result["intensity"]
        if intensity["percentiles"]:
            lines += [
                "",
                f"MusicalIntensity: mean {cell(intensity['mean'])}, from "
                f"{cell(intensity['percentiles']['p5'])} (5th percentile) to "
                f"{cell(intensity['percentiles']['p95'])} (95th).",
            ]
        if len(result["sections"]) > 1:
            lines += [
                "",
                "Sections (where the short-term profile changes most):",
                "",
                "| from (s) | to (s) | " + " | ".join(DIMENSION_NAMES) + " | intensity |",
                "|---:|---:|" + "---:|" * (len(DIMENSION_NAMES) + 1),
            ]
            for section in result["sections"]:
                lines.append(
                    f"| {section['start_s']:.0f} | {section['end_s']:.0f} | "
                    + " | ".join(cell(section["values"][d]) for d in DIMENSION_NAMES)
                    + f" | {cell(section['intensity_mean'])} |"
                )
        else:
            lines += ["", "No section boundary found: the short-term profile stays close."]
        lines += [
            "",
            "Distribution of the measurements over the music frames:",
            "",
            "| Measurement | unit | 5% | 25% | 50% | 75% | 95% |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for key, found in result["distributions"].items():
            lines.append(
                f"| {key} | {found['unit']} | "
                + " | ".join(cell(found["percentiles"][f"p{p}"], 3) for p in DISTRIBUTION)
                + " |"
            )
        cost = result["cost"]
        lines += [
            "",
            f"Cost: {cell(cost['character_us_per_frame'], 1)} µs a frame for Sonic Character"
            + (
                f", beside {cost['analysis_ms_per_frame'] * 1000:.0f} µs for the analysis pass."
                if cost["analysis_ms_per_frame"]
                else "."
            ),
        ]
    return "\n".join(lines) + "\n"


def dimension_table():
    """The definitions as plain data, for the JSON output."""
    return {
        name: {
            "meaning": meaning,
            "seconds_needed": needed,
            "components": [
                {"measurement": m, "gives_0": a, "gives_1": b, "weight": w} for m, a, b, w in parts
            ],
            "gate": {"measurement": gate[0], "gives_0": gate[1], "gives_1": gate[2]} if gate else None,
        }
        for name, (meaning, needed, parts, gate) in DIMENSIONS.items()
    }


def label(path, taken):
    """A short unique name for a recording in the outputs."""
    path = Path(path)
    base = path.parent.name if path.name == "messages.jsonl" else path.stem
    name, count = base, 2
    while name in taken:
        name, count = f"{base} ({count})", count + 1
    return name


def safe(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:80] or "recording"


def _job(job):
    """One recording, for ``run``: (result, level check or None), or the reason it failed."""
    path, gain_db = job
    try:
        result = analyse_path(path)
        check = None
        if gain_db and result["source"]["kind"] == "audio_file":
            check = level_check(result, analyse_path(path, gain_db))
        return result, check
    except Exception as exc:  # one unreadable file must not lose the others
        return f"{type(exc).__name__}: {exc}", None


def run(paths, out, gain_db=None, processes=1, progress=print):
    """Analyse every path, write the outputs into ``out`` and return the summary."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results, failed, level_checks = {}, {}, []
    jobs = [(path, gain_db) for path in paths]
    if processes > 1 and len(jobs) > 1:
        import os
        from concurrent.futures import ProcessPoolExecutor

        # One thread per process: several processes each using every core for their linear
        # algebra are slower than one. Only the processes started here are told.
        limits = {name: os.environ.get(name) for name in THREAD_LIMITS}
        os.environ.update(dict.fromkeys(THREAD_LIMITS, "1"))
        try:
            pool = ProcessPoolExecutor(min(processes, len(jobs)), mp_context=get_context("spawn"))
            list(pool.map(int, range(min(processes, len(jobs)))))  # start them now
        finally:
            for name, value in limits.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        done = pool.map(_job, jobs)
    else:
        pool, done = None, map(_job, jobs)
    for number, (path, (result, check)) in enumerate(zip(paths, done, strict=True), 1):
        name = label(path, results)
        if isinstance(result, str):
            failed[str(path)] = result
            progress(f"[{number}/{len(paths)}] {name}: not analysed: {result}")
            continue
        progress(f"[{number}/{len(paths)}] {name}")
        if check is not None:
            level_checks.append((name, gain_db, check))
        results[name] = result
        (out / (safe(name) + ".json")).write_text(
            json.dumps(result, indent=1, allow_nan=False), encoding="utf-8"
        )
        (out / (safe(name) + ".svg")).write_text(chart(result, name), encoding="utf-8")
    comparison = compare(results) if len(results) >= 2 else None
    summary = {
        "profile_version": results and next(iter(results.values()))["overall"]["profile_version"],
        "windows_s": {k: (None if math.isinf(v) else v) for k, v in WINDOWS.items()},
        "dimensions": dimension_table(),
        "recordings": {
            name: {
                "source": result["source"],
                "music_s": result["music_s"],
                "values": brief_values(result),
                "confidence": {
                    d["name"]: rounded(d["confidence"]) for d in result["overall"]["dimensions"]
                },
                "intensity_mean": result["intensity"]["mean"],
                "warnings": result["warnings"],
            }
            for name, result in results.items()
        },
        "comparison": comparison,
        "level_checks": [
            {"recording": name, "gain_db": gain, "largest_change": moved}
            for name, gain, moved in level_checks
        ],
        "not_analysed": failed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, allow_nan=False), encoding="utf-8")
    text = report(results, comparison, level_checks)
    if failed:
        text += "\n## Not analysed\n\n" + "\n".join(f"- {p}: {why}" for p, why in failed.items()) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    if pool is not None:
        pool.shutdown()
    return summary


def brief_values(result):
    return {d["name"]: rounded(d["value"]) for d in result["overall"]["dimensions"]}
