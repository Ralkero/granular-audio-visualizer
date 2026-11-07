"""Export only explicitly selected numerical soak metrics, excluding private identifiers."""

import argparse
import json
from pathlib import Path
import numpy as np


def summarize(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = [
        json.loads(line)
        for line in path.with_suffix(".samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    result = {
        k: data[k]
        for k in (
            "starting_commit",
            "ui",
            "logging",
            "advanced",
            "raw_audio_saved",
            "elapsed_seconds",
            "resources",
            "remaining_owned_child_processes",
        )
    }
    result["remaining_new_thread_count"] = len(data["remaining_new_threads"])
    result["sources"] = []
    for index, source in enumerate(data["sources"]):
        metrics = source["metrics"]
        safe = {
            "kind": source["source"]["kind"],
            "configuration": source["configuration"],
            "whole_session": source["whole_session"],
            "probe_subscriber_drops": source["probe_subscriber_drops"],
            "logger_drops": source["logger_drops"],
            "clean_shutdown": source["clean_shutdown"],
            "capture_error": bool(metrics["error"]),
        }
        safe["metrics"] = {
            k: metrics[k]
            for k in (
                "frames",
                "events",
                "dropped_blocks",
                "dropped_audio_frames",
                "queue_high_water",
                "discontinuities",
                "advanced_drops",
                "advanced_jobs",
                "advanced_processing_ms",
            )
        }
        safe["native_dropped_packets"] = source["source"].get("native_dropped_packets")
        points = [
            (r["elapsed"], r["monotonic"] - r["sources"][index]["audio_timestamp"])
            for r in samples
            if r["elapsed"] > 2 and r["sources"][index]["audio_timestamp"] is not None
        ]
        x, y = np.asarray(points).T
        safe["sampled_clock_age_slope_ms_per_minute"] = float(np.polyfit(x, y - y[0], 1)[0] * 60000)
        safe["clock_slope_scope"] = (
            "linear trend in per-second observation-minus-latest-audio time; includes poll jitter, not physical latency"
        )
        result["sources"].append(safe)
    return result, samples[0]["monotonic"] - samples[0]["elapsed"]


def model_contention(start, model_path, sessions):
    model = json.loads(model_path.read_text(encoding="utf-8"))["resources"]
    begin, end = model["started_monotonic"], model["ended_monotonic"]
    duration = end - begin
    results = []
    for folder in sessions.iterdir():
        path = folder / "messages.jsonl"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as stream:
            first = stream.readline()
            if not first:
                continue
            first = json.loads(first)
            if abs(first["detected_monotonic"] - start) > 2:
                continue
            values = {
                p: {"dsp_ms": [], "capture_to_publish_ms": []}
                for p in ("before", "during", "after")
            }
            for line in stream:
                message = json.loads(line)
                if message["type"] != "feature_frame":
                    continue
                time = message["published_monotonic"]
                period = (
                    "before"
                    if begin - duration <= time < begin
                    else "during"
                    if begin <= time <= end
                    else "after"
                    if end < time <= end + duration
                    else None
                )
                if period:
                    for metric in values[period]:
                        value = message.get(metric)
                        if value is not None:
                            values[period][metric].append(value)
            results.append(
                {
                    "kind": first["source"],
                    "windows_seconds": duration,
                    "scope": "same ongoing paired capture, UI off/logging and HPSS on; changing music and other developer load; observational, not causal isolation",
                    "periods": {
                        p: {
                            k: {"count": len(v), "p95": float(np.percentile(v, 95)), "max": max(v)}
                            if v
                            else None
                            for k, v in row.items()
                        }
                        for p, row in values.items()
                    },
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--sessions", type=Path, default=Path("sessions"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    baseline, _ = summarize(args.baseline)
    candidate, start = summarize(args.candidate)
    result = {
        "baseline_advanced_off": baseline,
        "candidate_advanced_on": candidate,
        "comparison_limit": "same source types/formats, UI off and logging on; audio content, manual transitions, DSP version and background workload differ",
    }
    if args.model:
        result["concurrent_model_observations"] = model_contention(start, args.model, args.sessions)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(str(args.output))


if __name__ == "__main__":
    main()
