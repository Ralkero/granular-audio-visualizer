"""Check GMV's Linux foundation on a real Linux desktop and write a report.

    uv run python scripts/validate_linux.py            # about two minutes
    uv run python scripts/validate_linux.py --tests    # and the whole test suite (several more)

The application window opens and closes by itself a few times: once as the session gives
it, once more to see that settings came back, and once on each Qt platform the session has
(``GMV_QT_PLATFORM=xcb``, ``GMV_QT_PLATFORM=wayland``). Each time it analyses a generated
WAV and a FLAC, records the sessions, replays one and shows Radiant.

Nothing of yours is read or changed: the XDG folders are pointed at a scratch folder for the
runs, which is removed afterwards. The report is printed and saved under the cache folder
(``~/.cache/GranularMusicVisualizer/validation/``); it holds no audio and no file names of
yours. Live capture is not checked: Linux has none yet.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SMOKE = Path(__file__).with_name("platform_smoke.py")
SESSION_VARIABLES = (
    "XDG_SESSION_TYPE",
    "XDG_CURRENT_DESKTOP",
    "WAYLAND_DISPLAY",
    "DISPLAY",
    "GAMESCOPE_WAYLAND_DISPLAY",
    "QT_QPA_PLATFORM",
    "GMV_QT_PLATFORM",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
)


def facts():
    import PySide6
    from PySide6.QtCore import qVersion

    from gmv import host
    from gmv.app.window import platform_plugin_problem

    release = {}
    try:
        release = platform.freedesktop_os_release()
    except (OSError, AttributeError):
        pass
    return {
        "distribution": release.get("PRETTY_NAME") or platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "libc": " ".join(platform.libc_ver()),
        "python": sys.version.split()[0],
        "pyside6": PySide6.__version__,
        "qt": qVersion(),
        "session": {name: os.environ.get(name) for name in SESSION_VARIABLES},
        "gmv_system": host.system(),
        "inside_gamescope": host.gamescope(),
        "qt_platform_choice": host.qt_platform(),
        "plugin_problems": {name: platform_plugin_problem(name) for name in ("xcb", "wayland")},
        "perf_counter": str(time.get_clock_info("perf_counter")),
    }


def launch(arguments, environment, cwd, timeout=240):
    began = time.perf_counter()
    try:
        done = subprocess.run(
            [sys.executable, *arguments], env=environment, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        code, out, err = done.returncode, done.stdout, done.stderr
    except subprocess.TimeoutExpired as late:
        code, out, err = None, str(late.stdout or ""), f"timed out after {timeout} s\n{late.stderr or ''}"
    return {"exit": code, "seconds": round(time.perf_counter() - began, 1), "stdout": out[-2000:], "stderr": err[-2000:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tests", action="store_true", help="also run the whole test suite")
    parser.add_argument("--output", default="", help="where to save the report")
    parser.add_argument("--anyway", action="store_true", help="run on a system that is not Linux")
    args = parser.parse_args()

    from gmv import host

    linux = host.system() == "linux"
    if not linux and not args.anyway:
        raise SystemExit("This checks a Linux system. On Windows: python scripts/platform_smoke.py OUT.json")
    report = {"when": datetime.now().isoformat(timespec="seconds"), "facts": facts(), "runs": {}, "checks": []}
    checks = report["checks"]

    def check(name, passed, detail=""):
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})
        print(f"  {'ok  ' if passed else 'FAIL'}  {name}" + (f": {detail}" if detail else ""), flush=True)

    with tempfile.TemporaryDirectory(prefix="gmv-validation-") as scratch:
        scratch = Path(scratch)
        elsewhere = scratch / "an empty working directory"
        elsewhere.mkdir()
        environment = {key: value for key, value in os.environ.items() if key != "GMV_QT_PLATFORM"}
        for variable, folder in (("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"), ("XDG_CACHE_HOME", "cache")):
            environment[variable] = str(scratch / folder)
        sessions = scratch / "data" / "GranularMusicVisualizer" / "sessions"
        settings = scratch / "config" / "GranularMusicVisualizer" / "Granular Music Visualizer.conf"
        session = report["facts"]["session"]
        expected = "wayland" if session["WAYLAND_DISPLAY"] or session["XDG_SESSION_TYPE"] == "wayland" else "xcb"
        if session["QT_QPA_PLATFORM"]:
            expected = None  # set outside: whatever it says
        runs = [("as the session gives it", {}, expected), ("again, for the remembered settings", {}, expected)]
        if linux and session["DISPLAY"]:
            runs.append(("GMV_QT_PLATFORM=xcb", {"GMV_QT_PLATFORM": "xcb"}, "xcb"))
        if linux and session["WAYLAND_DISPLAY"]:
            runs.append(("GMV_QT_PLATFORM=wayland", {"GMV_QT_PLATFORM": "wayland"}, "wayland"))
        if not linux:
            runs = runs[:1]

        for number, (title, extra, platform_expected) in enumerate(runs):
            print(f"\n{title}", flush=True)
            out = scratch / f"run{number}.json"
            arguments = [str(SMOKE), str(out)] + (["--remember"] if linux else [])
            done = launch(arguments, environment | extra, elsewhere)
            found = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
            report["runs"][title] = done | {"report": found}
            check("the window opened, ran and closed", done["exit"] == 0 and found, f"exit {done['exit']} in {done['seconds']} s")
            if not found:
                print(done["stderr"][-800:])
                continue
            told = [line for line in done["stderr"].splitlines() if line.startswith("gmv: Qt platform")]
            if linux:
                check("it said which Qt platform and why", told, told[-1] if told else "no line on standard error")
            if linux and platform_expected:
                check(f"Qt platform is {platform_expected}", found["qt_platform"] == platform_expected, found["qt_platform"])
            check(
                "sources offered are the system's",
                found["audio_sources"] == list(host.audio_sources()) and len(found["source_choices"]) == len(host.audio_sources()),
                ", ".join(found["source_choices"]),
            )
            if not host.live_capture():
                check("it says live capture is not there yet", found["source_note"].startswith(host.capture_note()), found["source_note"][:70])
            for name, entry in found["files"].items():
                good = entry.get("finished") and (entry.get("frames_analysed") or 0) > 100 and not entry.get("engine_error")
                check(f"{name} analysed", good, f"{entry.get('frames_analysed')} frames, {entry.get('onsets_shown')} onsets shown, problem '{entry.get('problem_at_start')}'")
                check(f"{name} session recorded complete, without audio", entry.get("session_complete") and entry.get("raw_audio_saved") is False, entry.get("session", ""))
                if linux:
                    check(f"{name} session is in the data folder", Path(entry.get("session", "/")).parent == sessions)
            replay = found.get("replay") or {}
            check("replay shows the recorded times, without the audio", replay.get("same_times") and replay.get("marked_replayed") and not replay.get("error"), json.dumps(replay))
            check("nothing left running", not found.get("threads_left"), f"closed in {found.get('closed_in_s')} s")
            renderer = found.get("radiant_renderer") or {}
            print(f"  note  Radiant renderer: {renderer.get('backend')}: {renderer.get('detail')}")
            if linux:
                check("settings file is in the configuration folder", found["settings_store"] == str(settings) and settings.is_file(), found["settings_store"])
            if linux and number == 1:
                check("settings came back", found["restored"] == {"mode": "radial", "expressive": True}, json.dumps(found["restored"]))
        check("nothing was written in the working directory", list(elsewhere.iterdir()) == [])

        print("\nthe command line", flush=True)
        done = launch(["-m", "gmv.cli", "capture", "--seconds", "1"], environment, elsewhere, 60)
        report["runs"]["gmv capture"] = done
        if host.live_capture():
            check("capture runs", done["exit"] == 0)
        else:
            check("capture is refused in one sentence", done["exit"] == 1 and done["stderr"].strip() == f"{host.capture_note()} Choose an audio file, or replay a recorded session.", done["stderr"].strip()[:120])
        if args.tests:
            print("\nthe test suite (several minutes)", flush=True)
            done = launch(["-m", "pytest", "-q", "-p", "no:cacheprovider"], environment, REPOSITORY, 3600)
            lines = [line for line in done["stdout"].splitlines() if " passed" in line or " failed" in line or " error" in line]
            failed = [line for line in done["stdout"].splitlines() if line.startswith(("FAILED", "ERROR"))]
            report["runs"]["pytest"] = done | {"summary": lines[-1:], "failed": failed}
            check("the test suite passes", lines and " failed" not in lines[-1] and " error" not in lines[-1], lines[-1] if lines else done["stderr"][-200:])
            for line in failed:
                print("        " + line)

    failed = [entry for entry in checks if not entry["passed"]]
    report["passed"] = not failed
    target = Path(args.output) if args.output else host.cache_folder() / "validation" / f"{datetime.now():%Y-%m-%d_%H%M%S}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    facts_ = report["facts"]
    print(f"\n{facts_['distribution']}, kernel {facts_['kernel']}, Python {facts_['python']}, Qt {facts_['qt']}")
    print(f"session: {facts_['session']}")
    print(f"{len(checks) - len(failed)} of {len(checks)} checks passed. Report: {target}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
