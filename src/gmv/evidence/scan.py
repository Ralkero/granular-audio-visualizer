"""The project's own recogniser over every recording of an inventory.

Each file is read once, its SHA-256 compared with the inventory, decoded without change and
fed through the same ``Recognizer`` path as the application (``review.timelines.compute``).
The result is model output and nothing else. One ``.npz`` per recording; finished recordings
are skipped, so the scan can be stopped and resumed.
"""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np

from . import read_inventory, recording_id

_STATE = {}


def _init(model_path):
    from ..recognition import load_backend

    _STATE["backend"] = load_backend(model_path)
    _STATE["description"] = json.loads(Path(model_path).read_text(encoding="utf-8"))


def _one(job):
    from ..review import audio as review_audio
    from ..review import timelines

    path, sha256, target = job
    target = Path(target)
    try:
        audio, rate, info = review_audio.decode(path)
        if info["sha256"] != sha256:
            return target.stem, 0.0, "the file's SHA-256 differs from the inventory"
        timeline = timelines.compute(audio, rate, _STATE["backend"])
        description = _STATE["description"]
        timeline.update(
            model=np.array(description["name"]),
            model_version=np.array(str(description["version"])),
            model_sha256=np.array(description["model_sha256"]),
            thresholds=np.array(json.dumps(description["thresholds"])),
            context_s=np.array(float(description["context_s"])),
            preprocessing=np.array(
                "front two channels averaged; quiet context raised to -20 dBFS RMS; "
                "resampled to the model rate; 10 s context ending at each 0.5 s hop; "
                "states smoothed over two estimates"
            ),
            source_sha256=np.array(sha256),
            level_dbfs=review_audio.level_profile(audio, rate)[: len(timeline["t_start"])],
        )
        temporary = target.with_suffix(".part.npz")
        np.savez_compressed(temporary, **timeline)
        temporary.replace(target)
        return target.stem, len(audio) / rate, ""
    except Exception as error:  # one unreadable file must not stop the scan
        return target.stem, 0.0, f"{type(error).__name__}: {error}"


def scan(inventory, workdir, model, processes=6, limit=None, log=print):
    out = Path(workdir) / "timelines" / Path(model).stem
    out.mkdir(parents=True, exist_ok=True)
    rows = read_inventory(inventory)[: limit or None]
    jobs = [
        (row["file_path"], row["sha256"], str(out / f"{recording_id(row['sha256'])}.npz"))
        for row in rows
    ]
    todo = [job for job in jobs if not Path(job[2]).is_file()]
    failures, seconds, started = {}, 0.0, perf_counter()
    if todo:
        with ProcessPoolExecutor(processes, initializer=_init, initargs=(str(model),)) as pool:
            futures = [pool.submit(_one, job) for job in todo]
            for count, future in enumerate(as_completed(futures), 1):
                name, length, problem = future.result()
                seconds += length
                if problem:
                    failures[name] = problem
                if count % 50 == 0 or count == len(todo):
                    speed = seconds / max(perf_counter() - started, 1e-9)
                    log(f"{count}/{len(todo)} recordings, {speed:.0f}x real time", flush=True)
    result = {
        "recordings": len(jobs),
        "already_done": len(jobs) - len(todo),
        "new": len(todo) - len(failures),
        "failed": failures,
        "hours_of_audio": round(seconds / 3600, 2),
        "minutes": round((perf_counter() - started) / 60, 1),
    }
    (out / "scan.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result
