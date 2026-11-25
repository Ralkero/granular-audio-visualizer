"""Reports of the research measurements: Markdown for people, SVG for a look at how a
recording evolves. Reports name recordings by the id or file stem they were given."""

from ..diagnose import COLOURS, cell, escape
from ..profile import DIMENSION_NAMES
from .profile import BLOCK_S, FAMILIES

CHARTED = (  # measurements drawn in a recording's chart, with the range mapped to 0..1
    ("tonal", "harmonic_share", 0.0, 1.0),
    ("tonal", "harmonic_movement", 0.0, 1.5),
    ("lead", "pitched_share", 0.0, 1.0),
    ("lead", "centre_line_level", 0.0, 1.0),
    ("rhythm", "pulse_clarity", 0.0, 1.0),
    ("rhythm", "events_per_beat", 0.0, 5.0),
)


def one(name, found):
    """The section of a report about one recording."""
    phase_a = found["phase_a"]
    lines = [f"## {name}", ""]
    lines.append(
        "Phase A (unchanged): "
        + ", ".join(f"{d['name']} {cell(d['value'])}" for d in phase_a["overall"]["dimensions"])
        + f"; mean intensity {cell(phase_a['intensity']['mean'])}."
    )
    for warning in found["warnings"]:
        lines.append(f"- Warning: {warning}")
    for family, definitions in FAMILIES.items():
        lines += [
            "",
            f"**{family}** (read every {BLOCK_S[family]:g} s)",
            "",
            "| Measurement | value | unit | kind | confidence | stated in | spread over the recording |",
            "|---|---:|---|---|---:|---:|---:|",
        ]
        for key, (unit, kind, _what, _usual) in definitions.items():
            got = found["families"][family][key]
            lines.append(
                f"| {key} | {'unknown' if got['value'] is None else cell(got['value'], 3)} | {unit} | {kind} | "
                f"{cell(got['confidence'])} | {got['available'] * 100:.0f}% of blocks | {cell(got['spread'], 3)} |"
            )
    return lines


def recordings(results, title="Sonic Character research: experimental measurements"):
    lines = [
        f"# {title}",
        "",
        "Experimental, offline, not used by the visualizer. `unknown` means the measurement "
        "could not be made here; it does not mean zero. `inferred` values interpret the "
        "signal (a key, a chord change, a voice) and can be wrong where the interpretation "
        "does not apply. See docs/SONIC_CHARACTER_RESEARCH.md.",
        "",
        "| Recording | " + " | ".join(DIMENSION_NAMES) + " | harmonic share | harmonic movement | pitched share | pulse clarity | tempo (BPM, level ambiguous) | events per beat |",
        "|---|" + "---:|" * (len(DIMENSION_NAMES) + 6),
    ]
    for name, found in results.items():
        f = found["families"]
        lines.append(
            f"| {name} | "
            + " | ".join(cell(d["value"]) for d in found["phase_a"]["overall"]["dimensions"])
            + " | "
            + " | ".join(
                "unknown" if v is None else cell(v)
                for v in (
                    f["tonal"]["harmonic_share"]["value"], f["tonal"]["harmonic_movement"]["value"],
                    f["lead"]["pitched_share"]["value"], f["rhythm"]["pulse_clarity"]["value"],
                    f["rhythm"]["tempo_bpm"]["value"], f["rhythm"]["events_per_beat"]["value"],
                )
            )
            + " |"
        )
    for name, found in results.items():
        lines += ["", *one(name, found)]
    return "\n".join(lines) + "\n"


def chart(found, title):
    """How a few of the measurements move through one recording."""
    width, height, left, top, bottom = 960, 340, 46, 28, 30
    plot_w, plot_h = width - left - 190, height - top - bottom
    last = max(found["evolution"]["tonal"]["end_s"] or [1.0])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="11">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{left}" y="16" font-size="13">{escape(title)}: experimental measurements by block</text>',
    ]
    for level in (0, 0.5, 1):
        y = top + (1 - level) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
    for index, (family, key, low, high) in enumerate(CHARTED):
        evolution = found["evolution"][family]
        points, pen = [], "M"
        for start, end, value in zip(evolution["start_s"], evolution["end_s"], evolution[key], strict=True):
            if value is None:
                pen = "M"
                continue
            x = left + (start + end) / 2 / last * plot_w
            y = top + (1 - min(1.0, max(0.0, (value - low) / (high - low)))) * plot_h
            points.append(f"{pen}{x:.1f},{y:.1f}")
            pen = "L"
        parts.append(f'<path d="{" ".join(points)}" fill="none" stroke="{COLOURS[index]}" stroke-width="1.6"/>')
        y = top + 14 + index * 18
        parts.append(
            f'<line x1="{left + plot_w + 12}" y1="{y - 4}" x2="{left + plot_w + 30}" y2="{y - 4}" stroke="{COLOURS[index]}" stroke-width="2"/>'
            f'<text x="{left + plot_w + 36}" y="{y}">{key} ({low:g}..{high:g})</text>'
        )
    parts.append(f'<text x="{left + plot_w}" y="{height - 10}" text-anchor="end">{last:.0f} s</text></svg>')
    return "\n".join(parts)


def share(value):
    return "–" if value is None else f"{value * 100:.0f}%"


def corpus(evaluation):
    """The corpus evaluation as Markdown."""
    split = evaluation["split"]
    lines = [
        "# Sonic Character research: corpus evaluation",
        "",
        f"Calibration: {split['calibration_recordings']} recordings of {len(split['calibration_groups'])} groups. "
        f"Evaluation: {split['evaluation_recordings']} recordings of {len(split['evaluation_groups'])} other groups. "
        "Scales come from calibration only; every figure below is on evaluation.",
        "",
        "## Each measurement",
        "",
        "| Measurement | kind | available | confidence | already in Phase A (R²) | between recordings | between artists | most like | labels (AUC) |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for name, entry in evaluation["measurements"].items():
        partner = evaluation.get("strongest_correlation", {}).get(name)
        beyond = entry.get("label_auc_beyond_phase_a", {})
        labels = ", ".join(
            f"{k} {cell(v)}" + (f" ({cell(beyond[k])} beyond Phase A)" if beyond.get(k) is not None else "")
            for k, v in entry["label_auc"].items()
            if v is not None
        )
        lines.append(
            f"| {name} | {entry.get('kind', 'Phase A')} | {share(entry['available_share'])} | {cell(entry.get('mean_confidence'))} | "
            f"{cell(entry.get('predicted_by_phase_a_r2'))} | {cell(entry.get('between_recordings_share'))} | "
            f"{cell(entry['between_artists_share'])} | {partner['with'] + ' ' + cell(partner['r']) if partner else '–'} | {labels} |"
        )
    lines += ["", "## Nearest recording is by the same artist (evaluation groups)", "",
              "| Measurements used | hit rate | chance | recordings |", "|---|---:|---:|---:|"]
    for label, found in evaluation["retrieval_same_artist"].items():
        lines.append(f"| {label} | {share(found['hit_rate'])} | {share(found['chance'])} | {found['recordings']} |")
    for gain, found in evaluation["level"].items():
        lines += ["", f"## Level: {gain} dB ({found['recordings']} recordings), change in calibration standard deviations", "",
                  "| Measurement | median | 90th percentile | largest | became unknown |", "|---|---:|---:|---:|---:|"]
        for name, moved in found["measurements"].items():
            lines.append(f"| {name} | {cell(moved['median'], 3)} | {cell(moved['p90'], 3)} | {cell(moved['largest'], 3)} | {moved['became_unknown']} |")
    if evaluation.get("timescales_between_recordings_share"):
        lines += ["", "## Over time (evaluation groups)", "",
                  "Share of each measurement's variation that lies between recordings when its blocks are averaged 1, 2, 4 or 8 at a time, "
                  "and its mean correlation with MusicalIntensity from block to block inside a recording.", "",
                  "| Measurement | 1 block | 2 | 4 | 8 | with intensity (mean r) | (mean abs r) |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name, found in evaluation["timescales_between_recordings_share"].items():
            follows = evaluation["blocks_with_intensity"][name]
            lines.append(f"| {name} | " + " | ".join(cell(found[k]) for k in ("1", "2", "4", "8")) + f" | {cell(follows['mean_r'])} | {cell(follows['mean_abs_r'])} |")
    for label, found in (evaluation.get("contrasts") or {}).items():
        if found:
            lines += ["", f"**{label.replace('_', ' ')}:** {found['a']} and {found['b']} "
                      f"({found['alike_rms_sd']:.2f} s.d. apart on what is alike, {found['differ_rms_sd']:.2f} on what differs; of {found['pairs_considered']} pairs)."]
    found = evaluation.get("transitions")
    if found:
        lines += ["", "## Telling a change of track from a change inside one (Phase A short window)", "",
                  f"Largest change inside a recording: median {cell(found['largest_change_inside_a_recording_median'])}. "
                  f"Change across a boundary between two recordings: median {cell(found['change_across_a_boundary_median'])}.", "",
                  "| Recordings that would be reset by mistake | threshold | boundaries caught |", "|---:|---:|---:|"]
        lines += [f"| {share(t['recordings_with_a_false_reset'])} | {cell(t['threshold'])} | {share(t['boundaries_caught'])} |" for t in found["thresholds"]]
    for pair in evaluation["pairs"]:
        lines += ["", f"## {pair['a']} against {pair['b']}", "",
                  "| Measurements | distance (RMS, standard deviations) | further apart than |", "|---|---:|---:|"]
        for label, found in pair["distance"].items():
            lines.append(f"| {label} | {cell(found['rms_sd'])} | {share(found['more_apart_than_share_of_pairs'])} of evaluation pairs |")
        lines += ["", f"| Measurement | {pair['a']} | {pair['b']} | apart (s.d.) | further apart than |", "|---|---:|---:|---:|---:|"]
        for name, found in pair["measurements"].items():
            if found is None:
                lines.append(f"| {name} | unknown for one or both | | | |")
            else:
                lines.append(f"| {name} | {cell(found['a'], 3)} | {cell(found['b'], 3)} | {cell(found['apart_sd'])} | {share(found['more_apart_than_share_of_pairs'])} of pairs |")
    return "\n".join(lines) + "\n"


def stems(found):
    """The stem and beat checks as Markdown."""
    lines = ["# Sonic Character research: checks against stems and annotated beats", ""]
    if "musdb" in found:
        summary = found["musdb"]["summary"]
        lines += [f"## MUSDB18-HQ ({summary['songs']} songs, seconds {found['musdb']['excerpt_s'][0]:g} to {found['musdb']['excerpt_s'][1]:g} of each)", "",
                  "| The voice, against the vocal stem | median | quartiles | songs |", "|---|---:|---|---:|"]
        for key, got in summary["voice"].items():
            if got:
                lines.append(f"| {key} | {cell(got['median'])} | {cell(got['low_quartile'])} – {cell(got['high_quartile'])} | {got['songs']} |")
        lines += ["", "| Share of the voice that is pitched: mix against stem | rank correlation | songs |", "|---|---:|---:|"]
        for key, got in summary["delivery"].items():
            if got and "rank_correlation_with_stem" in got:
                lines.append(f"| {key} | {cell(got['rank_correlation_with_stem'])} | {got['songs']} |")
        lines += ["", "| Harmony: mix against the mix without drums | rank correlation |", "|---|---:|"]
        lines += [f"| {key} | {cell(got['rank_correlation_mix_vs_no_drums'])} |" for key, got in summary["tonal"].items()]
        lines += ["", "| Rhythm: mix against the drums alone | rank correlation |", "|---|---:|"]
        for key, got in summary["rhythm"].items():
            lines.append(f"| {key} | {cell(got['rank_correlation_mix_vs_drums'])} |" if isinstance(got, dict) else f"| {key} | {share(got)} |")
    if "mdb_drums" in found:
        summary = found["mdb_drums"]["summary"]
        lines += ["", f"## MDB Drums ({summary['mixes']} mixes with annotated beats)", "",
                  "Pulse found, against the annotated beat: "
                  + ", ".join(f"{name} {count}" for name, count in summary["tempo_relation_counts"].items()) + ".",
                  f"Median pulse clarity {cell(summary['median_pulse_clarity'])}; median tempo ambiguity {cell(summary['median_tempo_ambiguity'])}."]
    return "\n".join(lines) + "\n"
