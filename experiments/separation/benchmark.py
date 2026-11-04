"""Bounded offline HTDemucs CPU experiment on original synthetic references.

Parent uses the main environment; only the isolated child imports torch/Demucs.
No live audio is consumed, uploaded or saved. Only synthetic inputs and metrics persist.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter, sleep


def worker(args):
    import numpy as np
    import torch
    from scipy.signal import correlate, correlation_lags
    from demucs.hf import load_safetensors_model
    from demucs.apply import apply_model
    from setup_model import SHA256, REVISION, verify

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(129)
    weights = Path(".model-cache/955717e8.safetensors")
    verify(weights)
    began = perf_counter()
    model = load_safetensors_model(weights).cpu().eval()
    load_seconds = perf_counter() - began
    if model.samplerate != 44100 or model.audio_channels != 2:
        raise ValueError("Unexpected model format")
    rows, outputs = [], []
    with np.load(args.input, allow_pickle=False) as data:
        names = data["names"].tolist()
        for i, name in enumerate(names):
            mix = torch.from_numpy(data[f"mix_{i}"].T.copy())
            reference = data[f"ref_{i}"]
            mean, std = mix.mean(), mix.mean(0).std().clamp(min=1e-8)
            began = perf_counter()
            with torch.inference_mode():
                estimates = apply_model(
                    model,
                    ((mix - mean) / std)[None],
                    shifts=0,
                    split=True,
                    overlap=0.25,
                    device="cpu",
                    num_workers=0,
                    progress=False,
                )[0]
            seconds = perf_counter() - began
            estimates = (estimates * std + mean).numpy()
            outputs.append(estimates)
            total_power = float(np.mean(data[f"mix_{i}"] ** 2))
            scores = {}
            for index, source in enumerate(model.sources):
                estimate = estimates[index].T.astype(np.float64)
                truth = reference[index].astype(np.float64)
                ref_power = float(np.mean(truth**2))
                est_power = float(np.mean(estimate**2))
                score = {
                    "reference_rms": ref_power**0.5,
                    "estimated_rms": est_power**0.5,
                    "estimated_power_fraction_of_mix": est_power / max(total_power, 1e-20),
                    "absent_reference": ref_power < 1e-16,
                }
                if ref_power >= 1e-16:
                    a = truth.ravel() - truth.mean()
                    b = estimate.ravel() - estimate.mean()
                    target = a * (np.dot(a, b) / max(np.dot(a, a), 1e-20))
                    score["si_sdr_db"] = float(
                        10
                        * np.log10(
                            (np.dot(target, target) + 1e-20)
                            / (np.dot(b - target, b - target) + 1e-20)
                        )
                    )
                    corr = correlate(
                        estimate.mean(axis=1), truth.mean(axis=1), mode="full", method="fft"
                    )
                    lags = correlation_lags(len(estimate), len(truth))
                    valid = abs(lags) <= 2205
                    score["best_lag_within_50ms_ms"] = float(
                        lags[valid][np.argmax(corr[valid])] / 44.1
                    )
                leak = {}
                for j, other in enumerate(model.sources):
                    if j == index:
                        continue
                    ref_other = reference[j].ravel().astype(np.float64)
                    energy = float(np.dot(ref_other, ref_other))
                    if energy > 1e-12:
                        leak[other] = float(
                            np.dot(estimate.ravel(), ref_other) ** 2
                            / (energy * max(np.sum(estimate**2), 1e-20))
                        )
                score["other_reference_projection_fraction"] = leak
                scores[source] = score
            rows.append(
                {
                    "fixture": name,
                    "inference_seconds": seconds,
                    "input_seconds": 6,
                    "new_usable_seconds_in_1s_overlap_schedule": 5,
                    "offline_rtf": seconds / 6,
                    "overlapping_schedule_rtf": seconds / 5,
                    "earliest_oldest_sample_age_seconds": 6 + seconds,
                    "scores": scores,
                }
            )
    # First two six-second chunks overlap over the same original second [5,6).
    first, second = outputs[0][:, :, -44100:], outputs[1][:, :, :44100]
    boundary = {
        name: float(np.sqrt(np.mean((first[j] - second[j]) ** 2)))
        for j, name in enumerate(model.sources)
    }
    result = {
        "model": "htdemucs/955717e8",
        "revision": REVISION,
        "weight_sha256": SHA256,
        "device": "cpu",
        "torch_threads": 4,
        "inference_workers": 0,
        "load_seconds_excluding_imports": load_seconds,
        "model_segment_seconds": float(model.segment),
        "sample_rate": model.samplerate,
        "sources": model.sources,
        "jobs": rows,
        "shared_1s_context_boundary_rms_difference": boundary,
        "saved_estimated_audio": False,
        "alignment": "48k to 44.1k scipy resample_poly 147/160; zero-phase offline conversion; no timeline shift; full six-second future context required for earliest samples",
        "limitations": "Original synthetic signals, not professional music; SI-SDR scale invariant and projection leakage confounded by correlated references; no vocals reference present",
    }
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


def parent(args):
    import numpy as np
    import psutil
    from scipy.signal import resample_poly
    from gmv.evaluation.corpus import generate

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    synthetic = output.with_suffix(".synthetic.npz")
    fixtures = {
        f.category: f
        for f in generate("development")
        if f.category in ("dense", "quiet_loud", "sustain_negative")
    }
    jobs = [
        ("dense", 0),
        ("dense", 5),
        ("quiet_loud", 0),
        ("quiet_loud", 6),
        ("sustain_negative", 0),
    ]
    packed = {"names": np.array([f"{name}@{start}s" for name, start in jobs])}
    for i, (name, start) in enumerate(jobs):
        fixture = fixtures[name]
        region = slice(start * fixture.rate, (start + 6) * fixture.rate)
        packed[f"mix_{i}"] = resample_poly(fixture.audio[region], 147, 160, axis=0).astype(
            np.float32
        )
        packed[f"ref_{i}"] = np.stack(
            [
                resample_poly(fixture.stems[n][region], 147, 160, axis=0)
                for n in ("drums", "bass", "other", "vocals")
            ]
        ).astype(np.float32)
    np.savez_compressed(synthetic, **packed)
    command = [
        args.python,
        str(Path(__file__).resolve()),
        "--worker",
        "--input",
        str(synthetic),
        "--output",
        str(output),
    ]
    started = perf_counter()
    # Thread caps apply only to this explicitly owned worker.
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1",
    }
    peak, cpu, samples = 0.0, 0.0, 0
    failure = None
    with output.with_suffix(".worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=env)
        child = psutil.Process(process.pid)
        child.cpu_percent()
        monitored = {child.pid: child}

        def stop_owned_worker():
            try:
                owned = child.children(recursive=True)
            except psutil.NoSuchProcess:
                owned = []
            for owned_process in reversed(owned):
                try:
                    owned_process.kill()
                except psutil.NoSuchProcess:
                    pass
            if process.poll() is None:
                process.kill()

        try:
            while process.poll() is None:
                sleep(0.1)
                try:
                    memory, total_cpu = 0.0, 0.0
                    for member in [child] + child.children(recursive=True):
                        if member.pid not in monitored:
                            monitored[member.pid] = member
                            member.cpu_percent()
                        sample = monitored[member.pid]
                        try:
                            memory += sample.memory_info().rss / 1024**2
                            total_cpu += sample.cpu_percent()
                        except psutil.NoSuchProcess:
                            pass
                    peak = max(peak, memory)
                    cpu += total_cpu
                    samples += 1
                except psutil.NoSuchProcess:
                    continue
                if memory > 4096 or perf_counter() - started > 300:
                    failure = "Owned benchmark worker exceeded 4 GiB RSS or 300-second deadline"
                    stop_owned_worker()
                    break
        finally:
            if process.poll() is None:
                stop_owned_worker()
            process.wait(timeout=10)
    ended = perf_counter()
    result = (
        json.loads(output.read_text(encoding="utf-8"))
        if process.returncode == 0 and output.exists()
        else {"error": failure or f"Worker exit {process.returncode}; see local worker log"}
    )
    result["resources"] = {
        "peak_sampled_worker_rss_mib": peak,
        "scope": "isolated launcher plus all its owned descendants",
        "mean_sampled_worker_cpu_one_core_percent": cpu / max(samples, 1),
        "sample_interval_seconds": 0.1,
        "parent_inclusive_wall_seconds": ended - started,
        "started_monotonic": started,
        "ended_monotonic": ended,
        "remaining_worker": process.poll() is None,
    }
    result["fixture_hashes"] = {
        name: f.manifest()["sha256_float32_le"] for name, f in fixtures.items()
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"output": str(output), "error": result.get("error"), "resources": result["resources"]}
        )
    )
    if result.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output", default="local-evidence/phase1_1/separation.json")
    parser.add_argument("--python", default=".separation-venv/Scripts/python.exe")
    args = parser.parse_args()
    worker(args) if args.worker else parent(args)
