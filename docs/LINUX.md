# Linux

Status: **phase 1a of the Linux port** (2026-11-23), **validated on a real CachyOS desktop
on 2026-11-23.** The application starts from a source checkout on Linux, analyses audio
files, records sessions and replays them, with the same engine and the same renderers as on
Windows. Every check of the application in `scripts/validate_linux.py` passed there. The
test suite passes there only with two test fixes that are not yet on `main` (see
[What has been checked](#what-has-been-checked)). It does **not** capture what the system
plays: there is no PipeWire or PulseAudio backend yet, and that is the next phase (1b).
CachyOS on a desktop is not SteamOS: nothing here is a claim about Flatpak, SteamOS, Steam's
Gaming Mode or the Steam Deck.

What was run on CachyOS, what was measured there, what was only tested on Windows, what is
still unchecked, and the commands that check it, are in
[What has been checked](#what-has-been-checked). The plan for the whole port, and the
research behind it, is [LINUX_PORT_AUDIT.md](LINUX_PORT_AUDIT.md).

## What is there, and what is not

"Yes" below means implemented, tested on Windows with the system set to Linux, and run on
one CachyOS desktop on 2026-11-23 (see [What has been checked](#what-has-been-checked)).

| | Linux, as implemented |
|---|---|
| Open the application window | Yes; on CachyOS under Wayland and under Xwayland |
| Analyse an audio file (WAV, FLAC, OGG, MP3 and whatever else libsndfile reads) | Yes; on CachyOS WAV and FLAC in the window, and WAV, FLAC, OGG and MP3 in the test suite |
| Record a session's analysis logs | Yes, in the user's data folder |
| Replay a recorded session, from Linux or from Windows | Yes |
| Classic, Spectrum bars and Radiant | Yes; on CachyOS all three were drawn, Radiant on the graphics card (AMD RX 9070 XT, Mesa 26.3.1). Other graphics cards and drivers not tried |
| Capture the system's output, one output device, or one application | **No.** Not offered; see below |
| Packaged build, desktop entry, Flatpak | No |

## Requirements

- Linux on x86-64 with glibc 2.34 or newer (the PySide6 6.12 wheel is `manylinux_2_34`).
  aarch64 wheels exist for every dependency but were not tried.
- Python 3.11 or newer. `uv` installs one if the system has none.
- [`uv`](https://docs.astral.sh/uv/) (Arch, CachyOS: `sudo pacman -S uv`).
- A desktop session, Wayland or X11. The Qt in the PySide6 wheel brings its own platform
  plugins, but uses the system's libraries for them:
  - X11 (`xcb`), which is also what Steam's Gaming Mode needs: `xcb-util-cursor`,
    `xcb-util-wm`, `xcb-util-keysyms`, `xcb-util-image`, `xcb-util-renderutil`,
    `libxkbcommon-x11` (Debian/Ubuntu: `libxcb-cursor0` and the `libxcb-*` and
    `libxkbcommon-x11-0` packages). A KDE or GNOME desktop usually has them all; the one
    most often missing is `xcb-util-cursor`.
  - Wayland: `wayland`, `libxkbcommon` (part of any Wayland desktop).
  - OpenGL through Mesa or the vendor driver (`libglvnd`), and `fontconfig`.
- No .NET, no PyAudioWPatch, no pycaw: those are for Windows capture and are neither
  installed nor imported on Linux (`sys_platform == 'win32'` markers in `pyproject.toml`).
- No root, and nothing installed outside the checkout's `.venv`.

## Setup

```sh
git clone https://github.com/Ralkero/granular-audio-visualizer.git
cd granular-audio-visualizer
uv sync --locked --extra dev
```

`uv sync` makes `.venv` in the checkout and installs the versions pinned in `uv.lock`. Add
`--extra recognition` for the optional context recognition runtime (ONNX Runtime).

## Launch

```sh
uv run gmv-visualizer                         # the application
uv run gmv visualizer --replay SESSION        # open it replaying a recorded session
uv run gmv file track.flac                    # analyse a file without the window
uv run gmv replay SESSION/messages.jsonl      # print a session's messages
```

`SESSION` is a session folder or its `messages.jsonl`. The commands work from any directory
when given the project: `uv run --project /path/to/checkout gmv-visualizer`. The application
never writes into the working directory or the checkout.

In the window: **Settings**, then **Audio source**, **Choose…** an audio file, then
**Start**. The file is analysed at playing speed and is not played aloud. **Sessions →
Replay a session…** opens a recorded one.

## Where things are kept

All of this follows the XDG Base Directory specification. A variable that is empty or not an
absolute path is ignored, as the specification says.

| What | Folder | Default |
|---|---|---|
| Settings and Radiant palettes | `$XDG_CONFIG_HOME/GranularMusicVisualizer/` | `~/.config/GranularMusicVisualizer/Granular Music Visualizer.conf` |
| Sessions (analysis logs; never audio unless debug recording is switched on) | `$XDG_DATA_HOME/GranularMusicVisualizer/sessions/` | `~/.local/share/GranularMusicVisualizer/sessions/` |
| Your own model files | `$XDG_DATA_HOME/GranularMusicVisualizer/models/` | `~/.local/share/GranularMusicVisualizer/models/` |
| Cache | `$XDG_CACHE_HOME/GranularMusicVisualizer/` | `~/.cache/GranularMusicVisualizer/` (nothing is kept there yet) |

- The settings file is written by Qt (`QSettings`), which follows `XDG_CONFIG_HOME` itself.
- Folders are made when they are first needed, not at start-up. If the sessions folder
  cannot be made or written, **Start** says which folder and why and nothing else stops;
  switching off *Save analysis logs* starts without it.
- The command line's `--sessions` default is the same sessions folder (on Windows it is
  `sessions` in the working directory, as before).
- Diagnostics (in Settings) shows the system, the Qt platform in use and why, the sessions
  folder and the settings file.

### Models

No model is part of the repository and none is downloaded. The application starts with the
first of these it finds, for the instrument weights (`activity_v2.npz`) and the guitar
weights (`guitar_v1.npz`) separately:

1. the file named by `GMV_ACTIVITY_MODEL` / `GMV_GUITAR_MODEL`;
2. `$XDG_DATA_HOME/GranularMusicVisualizer/models/`;
3. the checkout's `.model-cache/`.

A file chosen in Settings is remembered and always wins over these; nothing found in a
folder ever replaces it. Only those two file names are looked for: a recognition model
(`.json`) or any other file lying in a folder is never switched on by being there. Copying
weights into the models folder is the user's own act; nothing in the project copies or
redistributes weights (see the licensing constraint in the audit, F.7).

## Audio sources

Linux offers one source, **Audio file**, and replay of recorded sessions. The source list
shows only what the system can do; the note under it reads *"Live system audio capture is
not available on Linux yet."* A source remembered from Windows falls back to the file
source, and asking for capture any other way (`gmv capture`, an old setting) is answered
with that sentence, not with a missing-library error. There is no stand-in backend: nothing
returns silence or pretends to capture.

Capture through PipeWire's PulseAudio server is the next phase (1b); it will be another
`AudioSource` and nothing else in the program changes for it. None of it is written yet.

## Qt platform (Wayland, X11, Gamescope)

On a desktop the choice is Qt's: Wayland when the session has it, X11 otherwise. GMV does
not force X11 on Linux.

Inside Valve's Gamescope compositor, which is what Steam's Gaming Mode runs applications in,
GMV asks Qt for the X11 plugin (`xcb`), because Gamescope shows X11 windows through its own
Xwayland. It recognises Gamescope by all three of:

- `GAMESCOPE_WAYLAND_DISPLAY` is set (Gamescope's own socket; it sets this for what it runs),
- `DISPLAY` is set (Gamescope's Xwayland),
- `WAYLAND_DISPLAY` is not set (Gamescope removes it unless started with `--expose-wayland`).

A desktop session has no `GAMESCOPE_WAYLAND_DISPLAY`, and a Wayland desktop has a
`WAYLAND_DISPLAY`, so neither is taken for Gamescope. These variables come from Gamescope's
source (`src/main.cpp`); **the rule has not been tried in Steam's Gaming Mode on a Steam
Deck**, where the session script, not Gamescope, starts the programs. If it does not apply
there, use the override.

Order of precedence, first match wins:

| | Effect |
|---|---|
| `-platform NAME` on the command line | Qt's own; never overruled |
| `QT_QPA_PLATFORM=NAME` | Qt's own; never overruled |
| `GMV_QT_PLATFORM=xcb` (or `wayland`, or any plugin name) | That plugin |
| `GMV_QT_PLATFORM=auto` | Qt chooses; the Gamescope rule is off |
| Inside Gamescope | `xcb` |
| Otherwise | Qt chooses |

```sh
GMV_QT_PLATFORM=xcb uv run gmv-visualizer        # what Gaming Mode gets, on any desktop with X11 or Xwayland
GMV_QT_PLATFORM=wayland uv run gmv-visualizer
GMV_QT_PLATFORM=auto uv run gmv-visualizer
gamescope -- uv run gmv-visualizer               # a nested Gamescope, if it is installed
```

As a Steam launch option: `GMV_QT_PLATFORM=xcb %command%`.

At start-up one line on standard error says which plugin is in use and why, for example
`gmv: Qt platform xcb (inside Gamescope, which shows X11 windows (GAMESCOPE_WAYLAND_DISPLAY
is set))`; Diagnostics shows the same. When GMV itself asked for a plugin, it first checks
that the plugin can load, because Qt otherwise ends the process with a message that does
not name the cause. The check names the missing library (for `xcb`, typically
`libxcb-cursor.so.0`, package `xcb-util-cursor`) or says that `DISPLAY` is not set, and
exits with status 1.

## Tests

```sh
uv run pytest -q                                              # everything
uv run pytest -q tests/test_host.py tests/test_platform_app.py   # the platform boundary
uv run ruff check .
```

- The tests draw on Qt's `offscreen` platform and need no display.
- The platform tests run the same on every system: they tell `gmv.host` which system it is.
  Three are for a real Linux only (POSIX permissions, Qt's settings file, loading Qt's
  plugins) and are skipped on Windows; one is for Windows only (launching the capture
  helper) and is skipped on Linux.
- On Linux two more are skipped while their optional packages are not installed:
  `tests/test_recognition.py` needs ONNX Runtime (the `recognition` extra) and
  `tests/test_review_experiment.py` needs PyTorch. A Linux run set up with `--extra dev`
  therefore reports 3 skipped.
- The Radiant graphics-card tests start a real OpenGL context in a process of their own.
  Without a display, or without OpenGL 3.3, they are skipped with the reason. On CachyOS
  (AMD RX 9070 XT, Mesa 26.3.1, from a KDE Plasma Wayland session) they ran and passed.
- No test reads or writes the user's settings, sessions or models.

## What has been checked

Phase 1a was written and first tested on Windows, with `gmv.host` told it was on Linux,
because no Linux system was available. On 2026-11-23 it was run on a real Linux desktop.
The two are kept apart below: what was run on CachyOS, what was tested on Windows before
that, and what has not been checked anywhere.

### On CachyOS (2026-11-23)

The machine: CachyOS (kernel 7.3.5-2-cachyos, glibc 2.44, x86-64), KDE Plasma on Wayland,
an AMD Radeon RX 9070 XT with Mesa 26.3.1; Python 3.14.10, PySide6 and Qt 6.12.0; a clone
of `main` at `766a3f7` with its own `.venv` made by `uv`. The run:

```sh
uv run python scripts/validate_linux.py --tests
```

The script points the XDG folders at a scratch folder and starts the application from an
empty working directory, so none of the user's own settings or sessions were read or
written. It opened the real window on the desktop four times. **59 of its 60 checks
passed.** The one that failed was *the test suite passes* ([below](#the-test-suite-on-cachyos));
every check of the application itself passed.

| Window run | Qt platform, with the reason the application gave | Drawn during the file analyses |
|---|---|---|
| As the session gives it | `wayland` (Qt's own choice) | Spectrum bars |
| Again, to find the remembered settings | `wayland` (Qt's own choice) | Radial, Classic theme (the remembered mode) |
| `GMV_QT_PLATFORM=xcb` | `xcb`, through the Plasma session's Xwayland | Radial, Classic theme |
| `GMV_QT_PLATFORM=wayland` | `wayland` | Radial, Classic theme |

In each of the four runs:

- the window opened, ran and closed cleanly (exit 0 in about 17 s, no thread left running)
  and said on standard error which Qt platform it used and why;
- the source list offered *Audio file* only, with the note that live capture is not
  available on Linux yet;
- a generated four-second click track was analysed as a WAV (16-bit) and as a FLAC
  (24-bit): 198 frames, 25 events and 7 onsets shown for each, with no engine error;
- each analysis recorded a complete session, without audio, in
  `$XDG_DATA_HOME/GranularMusicVisualizer/sessions/`;
- the FLAC was deleted and its session replayed: the same 7 onsets at the recorded times,
  marked as a replay;
- Radiant was shown for two seconds, drawn on the graphics card;
- the settings file was in `$XDG_CONFIG_HOME/GranularMusicVisualizer/`, and the second run
  found the settings the first had left (Radial, Expressive).

Also: nothing was written in the working directory, and `gmv capture` was refused with one
sentence: *"Live system audio capture is not available on Linux yet. Choose an audio file,
or replay a recorded session."* The window runs used the scratch XDG folders only; the
default folders with the variables unset (`~/.config`, `~/.local/share`) are covered by the
test suite, which ran on the same machine. The report records that frames were drawn, not
how they looked.

**Measured in that run.** These are one run's figures for short generated files on this
machine, not what another machine or real music will do. The analysis runs in a process of
its own. The figures are what Diagnostics showed at the end of each window run, so they are
for the last file analysed (the FLAC), and the CPU and memory are the analysis process's.

| | Range over the four window runs |
|---|---|
| Analysis (DSP) time per 20 ms block, 95th percentile | 2.5–2.7 ms, against a budget of 20 ms |
| Instrument analysis, 95th percentile, inside the analysis pass | 0.99–1.02 ms per frame |
| Analysis process | CPU 17–18% of one core; memory 122 MiB |
| Losses | capture 0, native 0, logger 0, discontinuities 0; messages lost between the analysis process and the window 0 |
| Radiant's renderer | AMD Radeon RX 9070 XT (radeonsi, gfx1201), OpenGL 4.6 Compatibility Profile, Mesa 26.3.1, GLSL 4.60 |
| Radiant, the last frame measured | 1.38–1.39 ms on the graphics card; 0.66–0.78 ms to submit it |

The window's size was not recorded and the Radiant figure is a single frame, so it is not
comparable with the 1920×1080 measurements made on Windows.

#### The test suite on CachyOS

In the validation run: 517 passed, 3 skipped, **3 failed**. Two separate problems turned up
on Linux, both in tests, neither in the application:

- **The 3 failures**, all in `tests/test_remote.py` at `assert not children()`. On Linux a
  worker started with `spawn` leaves Python's multiprocessing resource tracker running for
  the life of the parent, and the test counted the tracker as a worker left behind; the
  worker itself had exited. Windows has no resource tracker. The fix is in the test only. It
  is **local and uncommitted**, on branch `fix/linux-resource-tracker-test` in the CachyOS
  checkout, and is not on GitHub.
- **An intermittent failure** of
  `tests/test_app.py::test_replay_runs_inside_the_same_window_with_original_times`, seen in
  later runs (1 of 6 runs on its own, and once in another full run; it passed in the
  validation run). No message was dropped. Since `e1d4a3a` the view takes in at most about
  6 ms of messages per frame and leaves the rest queued for the next, and the test read the
  markers one frame after a 100× replay had ended. The fix makes the replay tests wait until
  the view has taken in every message. It is in
  [pull request #1](https://github.com/Ralkero/granular-audio-visualizer/pull/1), **open and
  not merged**.

With both fixes applied, the whole suite on the same machine: **520 passed, 3 skipped,
0 failed.** That was a working tree with both fixes, not `main` as published. Until both are
on `main`, `main` still shows the 3 `test_remote.py` failures on Linux, and the replay test
can still fail now and then.

The 3 skipped on Linux: `tests/test_failures.py` (it launches the Windows capture helper),
and `tests/test_recognition.py` and `tests/test_review_experiment.py`, whose optional
packages (ONNX Runtime, PyTorch) are not installed with `--extra dev`. The three Linux-only
tests ran and passed, and so did the Radiant graphics-card tests
(`tests/test_radiant_plasma.py`, `tests/test_radiant_smoke.py`). No other test written on
Windows needed a Windows-only mark.

### On Windows, with the system set to Linux (before CachyOS)

When phase 1a was written no Linux machine was available: the development machine is
Windows 11, it has no WSL distribution to work in, and the container engine installed on it
did not start. No virtual machine, WSL distribution or other system was installed to get
one. What follows was checked on Windows 11 (Python 3.14.3, PySide6 6.12.0) before the
CachyOS run:

- The platform boundary, by telling `gmv.host` which system it is (58 tests in
  `tests/test_host.py` and `tests/test_platform_app.py`; 55 run on Windows): the XDG folders
  with the variables set, unset, empty and relative; folders made when needed and a folder
  that cannot be made explained; the model order; Linux offering the file source alone and
  refusing capture in words, without loading any Windows capture module; a WAV, a FLAC, an
  OGG and an MP3 analysed through the real window, each session recorded under
  `XDG_DATA_HOME` and replayed with its recorded times after its audio was deleted; a
  session written the way Windows writes it (CRLF line ends, Windows paths in the manifest)
  replayed as it is; settings saved and restored in a store of the test's own; the
  Gamescope rule, the override and Qt's own overrides; Qt started only after the choice; the
  command line recording in the data folder from another working directory.
- Windows unchanged: the whole suite (519 passed, 4 skipped; three of the skips are the
  Linux-only tests, which have since passed on CachyOS); the real window with its four sources, file analysis, replay,
  Radiant on the graphics card and live capture through the native helper; the command
  line's `devices`, `file` and `capture`.

### Not checked yet

- Steam's Gaming Mode, and a nested Gamescope (`gamescope -- uv run gmv-visualizer`): the
  Gamescope rule has not run inside any Gamescope.
- Anything on SteamOS or a Steam Deck: its audio setup, its graphics, its performance.
- A native X11 desktop session. `xcb` ran only through the Xwayland of a Plasma Wayland
  session.
- Whether KWin puts the window on a chosen screen under Wayland, where the compositor places
  windows itself. The validation windows were on `DP-2`, which is also the primary screen
  there, so this is not settled.
- A session with music of the user's own, watched as it plays. The generated click tracks
  show that the pipeline works; they do not replace looking at the visualizer with real
  music.
- The plugin check's messages against a real missing library (nothing was missing on the
  CachyOS machine).
- Other graphics cards and drivers; resizing, full screen and long sessions on Linux.
- Live capture: it is not implemented.

### Checking it on Linux (CachyOS)

```sh
sudo pacman -S --needed uv git xcb-util-cursor
git clone https://github.com/Ralkero/granular-audio-visualizer.git
cd granular-audio-visualizer
uv sync --locked --extra dev
uv run python scripts/validate_linux.py --tests
```

`scripts/validate_linux.py` opens the window a few times by itself (as the session gives it,
again to see the settings come back, then with `GMV_QT_PLATFORM=xcb` and
`GMV_QT_PLATFORM=wayland` where the session has them), analyses a generated WAV and FLAC
each time, replays a session, shows Radiant, checks where everything was written, and with
`--tests` runs the whole suite. It points the XDG folders at a scratch folder for this, so
your own settings and sessions are not touched. It prints each check as `ok` or `FAIL` and
saves a report under `~/.cache/GranularMusicVisualizer/validation/`. Then, by hand:

```sh
uv run gmv-visualizer                                  # choose a file of your own, Start, Stop
ls ~/.local/share/GranularMusicVisualizer/sessions     # the session it recorded
uv run gmv visualizer --replay ~/.local/share/GranularMusicVisualizer/sessions/SESSION
cat ~/.config/GranularMusicVisualizer/*.conf           # the settings it remembered
gamescope -- uv run gmv-visualizer                     # optional: a nested Gamescope
```

## Known limitations

- **No live capture.** Linux offers the audio file and the replay of recorded sessions only:
  it cannot capture the system's output, one output device or one application. A file is
  analysed at playing speed and is not played aloud. PipeWire capture is the next phase
  (1b); none of it is written yet.
- **No packaging.** No installed desktop entry, no installed icon, no Flatpak: the
  application is started from the checkout with `uv run gmv-visualizer`. On Wayland the
  window icon comes from a desktop entry, so the window is expected to have a generic icon
  there.
- Radiant on the graphics card has been run on one Linux driver only: Mesa 26.3.1 (radeonsi)
  on an AMD RX 9070 XT. If its OpenGL context cannot be made, the application falls back to
  the geometry-only Radiant and Diagnostics says so, as on Windows. No software renderer is
  selected by GMV.
- The interface names "Segoe UI" as its font, which a Linux system usually does not have;
  Qt then uses the system's default sans font.
- `Open folder` under Sessions asks the desktop to open the session folder
  (`QDesktopServices`); it needs a file manager and `xdg-open`.
- The Windows capture tools in `scripts/` (`clock_probe.py`, `validate_capture.py`,
  `lifecycle_probe.py`, the `--live` mode of `benchmark_visualizer.py`) and the research
  commands `gmv evidence` and `gmv review` were not adapted; some of their defaults are
  Windows paths.
- A PyInstaller build on Linux is not a supported configuration.

## Separate windows, to look at for Gaming Mode

Gamescope gives the focused window the screen; reports say that switching between several
windows of one application is unreliable there (audit E.4). Nothing was changed for this in
phase 1a. These parts of the interface open a window of their own and need a decision in
the Gaming Mode phase:

| Where | What opens |
|---|---|
| Settings → Audio source → Choose… | `QFileDialog` for the audio file |
| Settings → Analysis → Choose… (twice) | `QFileDialog` for weights |
| Settings → Sessions → Replay a session… | `QFileDialog` |
| Settings → Sessions → Export logs… | `QFileDialog` (save) |
| Settings → Sessions → Open folder | another application (the file manager) |
| Settings → Visualizer → Radiant colour roles | `QColorDialog` (Qt's own, not the desktop's) |
| Settings → Visualizer → Show | a popup `QMenu` |
| Every drop-down list | the combo box's popup |
| 27 controls with explanations | tooltips (no hover on a touch screen) |
| F11, Esc, D | keyboard only; F11 also has a button, Esc and D do not |
