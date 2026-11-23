# Linux

Status: **phase 1a of the Linux port** (2026-11-23), **implemented but not yet run on a real
Linux system.** The application is made to start from a source checkout on Linux, analyse
audio files, record sessions and replay them, with the same engine and the same renderers
as on Windows. That behaviour is tested on Windows by telling the program it is on Linux;
no Linux machine was available to run it on. It does **not** capture what the system plays:
there is no PipeWire or PulseAudio backend yet. Nothing here is a claim about Flatpak,
SteamOS or the Steam Deck.

What exactly was and was not checked, and the commands that check it on Linux, are in
[What has been checked](#what-has-been-checked). The plan for the whole port, and the
research behind it, is [LINUX_PORT_AUDIT.md](LINUX_PORT_AUDIT.md).

## What is there, and what is not

"Yes" below means implemented and covered by tests that run on Windows with the system set
to Linux, not observed on Linux.

| | Linux, as implemented |
|---|---|
| Open the application window | Yes |
| Analyse an audio file (WAV, FLAC, OGG, MP3 and whatever else libsndfile reads) | Yes |
| Record a session's analysis logs | Yes, in the user's data folder |
| Replay a recorded session, from Linux or from Windows | Yes |
| Classic, Spectrum bars and Radiant | The same code as on Windows; Radiant on the graphics card is not yet validated on Linux drivers |
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
`AudioSource` and nothing else in the program changes for it.

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
- The Radiant graphics-card tests start a real OpenGL context in a process of their own.
  Without a display, or without OpenGL 3.3, they are skipped with the reason. Their passing
  on Linux is part of the rendering phase, not this one.
- No test reads or writes the user's settings, sessions or models.

## What has been checked

**Nothing in this phase has been run on a real Linux system.** No Linux machine was available
when it was written: the development machine is Windows 11, it has no WSL distribution to
work in, and the container engine installed on it did not start. No virtual machine, WSL
distribution or other system was installed to get one. So, as of 2026-11-23:

Checked on Windows 11 (Python 3.14.3, PySide6 6.12.0):

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
  Linux-only tests below); the real window with its four sources, file analysis, replay,
  Radiant on the graphics card and live capture through the native helper; the command
  line's `devices`, `file` and `capture`.

Not checked anywhere yet, and to be checked on Linux:

- `uv sync` on Linux and the application window actually opening, on Wayland and on X11.
- The three Linux-only tests: a read-only folder refused (POSIX permissions), Qt's settings
  file landing in `XDG_CONFIG_HOME`, and Qt's own platform plugins found loadable.
- The rest of the existing test suite on Linux. It was written on Windows; one test that
  launches the Windows capture helper is marked Windows-only, and others may turn out to
  need the same (the research tools `gmv evidence` and `gmv review` are the likeliest).
- The plugin check's messages against a real missing library.
- The Gamescope rule inside a real Gamescope, nested or in Steam's Gaming Mode.
- Radiant's shaders on Mesa or any Linux driver (the rendering phase).

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

(The work of this phase is not on GitHub until it is committed and pushed; until then copy
the checkout to the Linux machine instead of cloning.)

## Known limitations

- No live capture (see above). This is the next phase.
- Radiant on the graphics card has not been validated on Linux drivers. If its OpenGL context
  cannot be made, the application falls back to the geometry-only Radiant and Diagnostics
  says so, as on Windows. No software renderer is selected by GMV.
- The interface names "Segoe UI" as its font; Linux is expected to use the system's default sans font.
- No desktop entry or installed icon. On Wayland the window icon comes from a desktop entry,
  so the window is expected to have a generic icon there.
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
