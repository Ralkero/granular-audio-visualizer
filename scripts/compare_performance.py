"""Matched incremental synthetic DSP comparison; no hardware-latency claim."""

import argparse
import json
import subprocess
import numpy as np
from gmv.evaluation.corpus import generate
from gmv.evaluation.run import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for fixture in generate("development"):
        if fixture.category not in ("dense", "embedded_eight"):
            continue
        runs = []
        # Balanced order reduces the influence of startup and drifting background load.
        for detector in ("baseline", "candidate", "candidate", "baseline"):
            report = evaluate(fixture, detector)
            runs.append({"detector": detector, **report["performance"]})
        rows.append(
            {
                "fixture": fixture.manifest(),
                "runs": runs,
                "mean_run_p95_ms": {
                    d: float(np.mean([r["dsp_p95_ms"] for r in runs if r["detector"] == d]))
                    for d in ("baseline", "candidate")
                },
            }
        )
    result = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "scope": "same original audio and current schema; incremental 10ms packets; UI/logging/advanced/beat-phase off; baseline DSP algorithm frozen; not live capture latency",
        "results": rows,
    }
    from pathlib import Path

    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps([{"fixture": r["fixture"]["category"], **r["mean_run_p95_ms"]} for r in rows]))


if __name__ == "__main__":
    main()
