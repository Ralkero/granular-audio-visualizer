"""Run the experiment described by a configuration file: the baseline scoring, every arm and
fold, then the summary. The file is the whole description; nothing is typed by hand.

    python experiments/recognition/run_review_experiment.py --list          # the commands
    python experiments/recognition/run_review_experiment.py --dry-run       # check, train nothing
    python experiments/recognition/run_review_experiment.py --owner-approved

Runs that are already complete are skipped, so an interrupted experiment continues where it
stopped. Training starts only with ``--owner-approved``; without it the training commands
are refused by ``finetune_review.py`` itself.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import review_data as rd  # noqa: E402


def commands(config, approved=False, dry_run=False):
    """[(name, argv)] in the order they run. Both arms get the same settings."""
    paths = config["paths"]
    python = str(Path(config["python"]).resolve())  # a relative interpreter path is not found on Windows
    common = [python, str(HERE / "finetune_review.py")]
    for key in ("jobs", "parts", "labels", "source", "init", "review", "runs", "weights-dir"):
        common += [f"--{key}", paths[key]]
    for key, value in config["settings"].items():
        common += [f"--{key}", str(value)]
    extra = (["--dry-run"] if dry_run else []) + (["--owner-approved"] if approved else [])
    out = [] if dry_run else [("baseline", [*common, "--arm", "baseline"])]
    for fold in config["folds"]:  # the two arms of a fold run one after the other
        for arm in config["arms"]:
            out.append((f"{arm}/hold{fold}", [*common, "--arm", arm, "--hold", str(fold), *extra]))
    if not dry_run:
        summary = [python, str(HERE / "review_summary.py"), "--review", paths["review"], "--runs", paths["runs"]]
        summary += ["--model-config", paths["model-config"], "--report", str(Path(paths["runs"]) / "summary.json")]
        out.append(("summary", summary))
    return out


def complete(config, name):
    path = Path(config["paths"]["runs"]) / name / "run.json"
    return path.is_file() and json.loads(path.read_text(encoding="utf-8"))["status"] == "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=str(HERE / "review_experiment_v2.json"))
    parser.add_argument("--list", action="store_true", help="print the commands and stop")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--owner-approved", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    found = rd.sha256_of(config["paths"]["init"])
    if found != config["initial_weights_sha256"]:
        raise SystemExit("The starting weights are not the file this experiment was written for.")
    for name, command in commands(config, args.owner_approved, args.dry_run):
        if args.list:
            print(" ".join(command))
            continue
        if name != "summary" and not args.dry_run and complete(config, name):
            print(f"{name}: already complete", flush=True)
            continue
        print(f"--- {name}", flush=True)
        if subprocess.run(command).returncode:
            raise SystemExit(f"{name} failed; nothing after it was run")


if __name__ == "__main__":
    main()
