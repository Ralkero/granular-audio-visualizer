"""Draw recognition timelines from the files ``scripts/score_recognition.py run`` writes.

One picture per recording: the reference (where the dataset's guitar files are active), the
model's score with its two thresholds, what the application would show (detected,
candidate), and where the two disagree (false alarms in red, misses in blue).

Runs in the experiments environment (matplotlib). No audio is read.

    python experiments/recognition/plot_timeline.py SCORES_DIR --model MODEL.json \
        --output OUT_DIR [--only r-0123456789ab ...] [--title "r-...=Song label"]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, GUITAR = "#1f2937", "#6b7280", "#d97706"
FALSE_ALARM, MISS = "#dc2626", "#2563eb"


def spans(mask, times, hop):
    """(start, width) for each run of True in ``mask``; a value covers the hop before it."""
    out, start = [], None
    for value, time in zip(mask, times, strict=True):
        if value and start is None:
            start = time - hop
        if not value and start is not None:
            out.append((start, time - hop - start))
            start = None
    if start is not None:
        out.append((start, times[-1] - start))
    return out


def draw(path, thresholds, title, target, column=0):
    with np.load(path, allow_pickle=False) as data:
        item = {key: data[key] for key in data.files}
    label = str(item["labels"][column])
    times, score = item["t"], item["score"][:, column]
    state, truth = item["state"][:, column], item["y"][:, column]
    certain = item["certain"][:, column]
    hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.5
    figure, (top, middle, bottom) = plt.subplots(
        3, 1, figsize=(12, 4.6), sharex=True, height_ratios=(1.0, 2.6, 1.4)
    )
    reference = item["frames"][:, column]
    grid = np.arange(len(reference)) * 0.02 + 0.02
    top.broken_barh(spans(reference, grid, 0.02), (0, 1), color=INK)
    top.broken_barh(spans(~certain, times, hop), (0, 1), color="#d1d5db")
    top.set_yticks([])
    top.set_ylabel("reference", rotation=0, ha="right", va="center", color=INK)
    top.set_title(title, loc="left", fontsize=11, color=INK)

    middle.plot(times, score, color=GUITAR, linewidth=1.2)
    for name, style in (("detected", "-"), ("candidate", ":")):
        middle.axhline(thresholds[name], color=MUTED, linewidth=0.8, linestyle=style)
        middle.text(times[-1], thresholds[name], f" {name}", va="center", fontsize=8, color=MUTED)
    middle.set_ylim(0, 1.02)
    middle.set_ylabel("score", rotation=0, ha="right", va="center", color=INK)

    bottom.broken_barh(spans(state == 2, times, hop), (1.1, 0.9), color=GUITAR)
    bottom.broken_barh(spans(state == 1, times, hop), (1.1, 0.9), color=GUITAR, alpha=0.35)
    bottom.broken_barh(
        spans((state == 2) & ~truth & certain, times, hop), (0, 0.9), color=FALSE_ALARM
    )
    bottom.broken_barh(spans((state < 2) & truth & certain, times, hop), (0, 0.9), color=MISS)
    bottom.set_yticks([0.45, 1.55], ["errors", "shown"])
    bottom.set_ylim(0, 2.05)
    bottom.set_xlabel("seconds")
    bottom.set_xlim(0, max(times[-1], grid[-1] if len(grid) else 0))
    for axis in (top, middle, bottom):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(colors=MUTED, labelsize=8)
    note = (
        f"{label}: dark = reference active, grey = reference uncertain (bleed) · orange = "
        "detected, pale orange = candidate · red = false alarm, blue = missed (against detected)"
    )
    figure.text(0.01, 0.005, note, fontsize=7.5, color=MUTED)
    figure.tight_layout(rect=(0, 0.02, 1, 1))
    figure.savefig(target, dpi=110)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("scores")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--title", nargs="*", default=[], help="id=text shown instead of the id")
    args = parser.parse_args()
    description = json.loads(Path(args.model).read_text(encoding="utf-8"))
    titles = dict(entry.split("=", 1) for entry in args.title)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(args.scores).glob("*.npz")):
        if args.only and path.stem not in args.only:
            continue
        label = description["labels"][0]
        draw(
            path,
            description["thresholds"][label],
            titles.get(path.stem, path.stem),
            out / f"{path.stem}.png",
        )
        print(out / f"{path.stem}.png")


if __name__ == "__main__":
    main()
