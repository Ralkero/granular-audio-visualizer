# Linux

Status: **phases 1a and 1b of the Linux port, validated on a real CachyOS desktop on
2026-11-23.** The application starts from a source checkout on Linux, analyses audio files,
records sessions and replays them (phase 1a), and **captures what the system plays on its
default output** through PipeWire's PulseAudio service (phase 1b,
[System audio](#system-audio)), with the same engine and the same renderers as on Windows.
Linux does not capture one output device, one application or a microphone. The test suite
passes there apart from the three known `tests/test_remote.py` failures and the intermittent
replay test, whose fixes are not yet on `main` (see
[What has been checked](#what-has-been-checked)). CachyOS on a desktop is not
SteamOS: nothing here is a claim about Flatpak, SteamOS, Steam's Gaming Mode or the Steam
Deck.

What was run on CachyOS, what was measured there, what was only tested on Windows or against
a stand-in, what is still unchecked, and the commands that check it, are in
[What has been checked](#what-has-been-checked). The plan for the whole port, and the
research behind it, is [LINUX_PORT_AUDIT.md](LINUX_PORT_AUDIT.md).

## What is there, and what is not

"Yes" below means implemented, tested (on Windows with the system set to Linux, or against a
stand-in sound server), and run on one CachyOS desktop on 2026-11-23 (see
[What has been checked](#what-has-been-checked)).

| | Linux, as implemented |
|---|---|
| Open the application window | Yes; on CachyOS under Wayland and under Xwayland |
| Analyse an audio file (WAV, FLAC, OGG, MP3 and whatever else libsndfile reads) | Yes; on CachyOS WAV and FLAC in the window, and WAV, FLAC, OGG and MP3 in the test suite |
| Record a session's analysis logs | Yes, in the user's data folder |
| Replay a recorded session, from Linux or from Windows | Yes |
| Classic, Spectrum bars and Radiant | Yes; on CachyOS all three were drawn, Radiant on the graphics card (AMD RX 9070 XT, Mesa 26.3.1). Other graphics cards and drivers not tried |
| Capture what the system plays on its default output (System audio) | Yes, from the default output's monitor through PipeWire's PulseAudio service; following a change of default output tested against a stand-in only |
| Capture one output device, one application, or a microphone | **No.** Not offered; see below |
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
- For **System audio** only: a PulseAudio-compatible sound server, which on CachyOS (and
  most current desktops) is PipeWire's `pipewire-pulse`, and its client programs `pactl`
  and `parec` (on CachyOS and Arch in the `libpulse` package, normally installed already).
  Audio files and replay need neither. See [System audio](#system-audio).
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
uv run gmv capture --seconds 30               # capture what the system plays, without the window
uv run gmv replay SESSION/messages.jsonl      # print a session's messages
```

`SESSION` is a session folder or its `messages.jsonl`. The commands work from any directory
when given the project: `uv run --project /path/to/checkout gmv-visualizer`. The application
never writes into the working directory or the checkout.

In the window: **Settings**, then **Audio source**. **System audio (PipeWire)**, the first
entry and the one a new installation starts with, visualizes what the system is playing:
**Start**, then play music in any application. **Audio file**: **Choose…** a file, then
**Start**; the file is analysed at playing speed and is not played aloud. **Sessions →
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

Linux offers **System audio (PipeWire)**, **Audio file**, and replay of recorded sessions.
It does not offer the Windows sources *Specific output device* and *Application / process
tree*, and it never captures a microphone. A source remembered from Windows that Linux does
not have falls back to System audio; asking for one any other way (`gmv capture --device`,
an old setting) is answered with a sentence, not with a missing-library error.

### System audio

System audio is everything the system plays on its **default output**: Spotify, games,
browsers, media players, any application that plays through it, mixed as the output plays
it (after any effects a program such as EasyEffects puts in front of the output). It is
recorded from that output's **monitor**, the source the sound server keeps for every output
carrying what it plays. That is not a microphone: a microphone is an input source, and GMV
records only a source the server confirms is the monitor of the default output.

How it works (`src/gmv/sources/system_pulse.py`):

- GMV asks the sound server, through `pactl`, which output is the default and which source
  is its monitor, and checks that the monitor exists. No device name is assumed; names
  are passed to the programs as single arguments, never through a shell.
- `parec` records the monitor as 32-bit float stereo at the monitor's own sample rate (the
  server mixes other layouts down) into a pipe that GMV reads in 10 ms blocks. The
  analysis, the visual state and the renderers are the same as on Windows.
- The recording stream asks PipeWire not to move it and not to fall back to another source
  (`node.dont-reconnect`, `node.dont-fallback`, `node.dont-move`): if its output goes
  away, the stream ends instead of being moved, for instance to a microphone. Every two
  seconds GMV also checks which source the server really records the stream from, and
  stops a stream found anywhere else.
- Every half second GMV asks which output is the default. When it changes (headphones
  plugged in, another output chosen in the system's sound settings), the stream is closed
  and one is opened on the new default's monitor, without restarting anything. The analysis
  starts a new segment there, as after any gap. If the stream ends or the server goes
  away, GMV says why in the status line and tries again, waiting 0.5 s, then 1, 2, 4 and
  at most 5 s between attempts, until it is stopped.
- Nothing is played, and no volume, default device or routing is changed. No module is
  loaded into the server, and no setting of the server is touched.
- As on Windows, sessions hold the analysis, never the audio.

Timing: `parec` gives no capture times, so GMV counts samples and anchors the count on when
blocks arrive. The first 100 ms of each stream are measured, not shown, to find the
earliest arrival; times are then not later than arrival by more than 10 ms. A sustained
change of more than 20 ms behind (lost audio, an output clock running slow) or 10 ms ahead
(an output clock running fast) moves the anchor and starts a new run of the clock, marked
as a gap; a block or two of audio is left out at most, so that time never goes back. A pause in
delivery of more than 0.25 s, or a suspend of the computer, starts a new run that is
measured again. How long the sound took to reach the monitor and the speakers is not known
and is not claimed.

The requirements, and how to check them:

```sh
pactl info                          # "Server Name: PulseAudio (on PipeWire …)" and the default output
pactl get-default-sink              # the default output
pactl list short sources            # its monitor is the source named after it, ending in .monitor
systemctl --user status pipewire-pulse.socket   # the server, on CachyOS
pacman -Qo /usr/bin/pactl /usr/bin/parec        # both in libpulse on CachyOS and Arch
```

When System audio cannot start, **Start** says why in one sentence and the window carries
on (audio files and replay still work):

| What it says | What to do |
|---|---|
| …needs the programs pactl and parec… not installed… | `sudo pacman -S libpulse` |
| No PulseAudio-compatible sound server answered… | Start PipeWire's PulseAudio service: `systemctl --user start pipewire-pulse.socket`, or log in again |
| …has no default output… | Choose an output in the system's sound settings (`pactl list short sinks` lists them) |
| …has no monitor source… / …is not among the sources… | `pactl list short sources` shows which monitors exist; choose an output that has one |

While capturing, the status line says *recovering* with the reason when the stream ends or
the default output changes, and *waiting for playback* when the monitor delivers nothing
for a second.

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
uv run pytest -q tests/test_system_pulse.py                   # Linux system audio
uv run ruff check .
```

- **No test records anything.** `tests/conftest.py` points `PULSE_SERVER` at a socket that
  does not exist, so the real `pactl` and `parec` cannot reach the machine's sound server
  from any test or any process a test starts. The System audio tests
  (`tests/test_system_pulse.py`) run against `tests/fake_pulse.py`, a stand-in for both
  programs that pretends to be a server with two outputs and a microphone and delivers a
  known signal in real time. They cover finding the default output and its monitor; every
  refusal (no server, no default output, no monitor, a source that is not the monitor,
  missing programs, an old `pactl`); the exact samples and the sample clock; the stream
  ending by itself, the server going away and coming back, the output disappearing, the
  default output changing, the stream being moved to the microphone; stopping; a slow
  reader; the clock's resyncs; and a captured session analysed, logged without audio and
  replayed.

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

### System audio on CachyOS (phase 1b, 2026-11-23)

The same machine, on the phase 1b branch. The sound server: PipeWire 1.6.11 with
`pipewire-pulse` (it calls itself "PulseAudio (on PipeWire 1.6.11)"), WirePlumber 0.5.19,
and `pactl` and `parec` from `libpulse` 17.0. Its default output was a Sound BlasterX G6
(USB, 24-bit stereo at 48 kHz); its monitor was found, confirmed and recorded as 32-bit float
stereo at 48 kHz. A second output (a game controller's) and an EasyEffects output were
there and not used. Whatever was playing at the time was captured, quietly (a median of
about −45 dBFS); no test signal was played, and nothing about the server, the default
output, the routing or the volume was changed. Seen once on that server: PipeWire applies
`node.dont-reconnect`, `node.dont-fallback` and `node.dont-move` to the stream's node, and
records it from the monitor named.

| Run | What it showed |
|---|---|
| `scripts/validate_linux.py --audio-capture --capture-seconds 30` | **70 of 70 checks passed.** System audio was offered in all four window runs (Wayland, Wayland again with the remembered settings, Xwayland, Wayland forced), and each window captured the default output for six seconds through the separate analysis process the application uses: 203 to 205 analysis frames each. Every phase 1a check passed again (files, sessions, replay, settings, Radiant on the graphics card). The 30-second capture: 1491 of about 1500 frames, every check passed (below) |
| `scripts/capture_probe_linux.py --seconds 600`, with the first version of the clock (below) | 29,990 of 30,000 frames in ten minutes: one segment, no block dropped, no discontinuity, no clock correction, no recovery. Stopped at once with nothing left running; the session was logged without audio and replayed after the capture had stopped, all 29,990 frames |
| `scripts/capture_probe_linux.py --seconds 1800`, with the clock as it is now | 89,988 of 90,000 frames in thirty minutes, one segment, no block dropped, no recovery. The clock corrected itself once, between the 23rd and 24th minute, as the drift below predicts: one discontinuity, 20 ms of audio left out, three analysis frames fewer. Stopped at once with nothing left running; the session replayed in full, all 89,988 frames |
| `gmv capture --seconds 4 --no-log` | 189 frames, 49 events, no block dropped, no discontinuity |

Measured in those runs, on this machine and with what was playing; not a guarantee for
another machine:

| | |
|---|---|
| First audio after Start | about 90 ms, then 100 ms measured for the clock and not analysed |
| Delivery | in whole PipeWire periods of 1024 frames (21.3 ms), read as 10 ms blocks |
| Capture to publication (the engine's estimate, from the clock's capture times to the analysis frame, half a window included) | 33 ms median, 37 ms 95th percentile (ten minutes); 28 and 35 ms (thirty minutes) |
| Waiting in the engine's queue | 0.03 ms median; the queue never held more than one block |
| Analysis (DSP) time per 20 ms block, 95th percentile | 1.3 ms |
| CPU | 7 to 9% of one core for capture and analysis in one process; about 1% more for `parec` |
| Memory | about 128 MiB, flat over three minutes of capture; `parec` 5.7 MiB. (The probe's own record of every frame grows by about 0.9 MiB a minute; that is the probe, not the capture) |
| Audio time per second of the computer's clock | 1.00002 over ten minutes, 1.000012 over thirty |
| The output's clock against the computer's | about 11 ppm fast: the clock's floor fell steadily, from +5.9 to −0.2 ms in ten minutes and from +5.1 to −9.3 ms in 23 minutes before its correction |

The clock's first version, which ran the ten-minute capture, allowed 5 ms ahead and
re-measured for 100 ms at each correction; at 11 ppm that would have cost 100 ms of audio
every eight minutes or so. It now allows 10 ms and corrects by moving its anchor, leaving
out a block or two (see [System audio](#system-audio)). Nothing in the ten-minute run
reached a correction, so both versions behaved the same there; the thirty-minute run, with
the clock as it is now, showed its one correction.

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
- System audio following a change of default output between two real devices, an output
  being unplugged while captured, the sound server restarting, and a suspend while
  capturing: all tested only against the stand-in server (`tests/test_system_pulse.py`).
  The validation machine has a second output, but changing the user's default output was
  out of bounds for the validation.
- System audio on a PulseAudio server (only PipeWire's was tried), at a rate other than
  48 kHz, and for longer than thirty minutes.
- What System audio looks like with music the user plays and watches, judged by eye (see
  above).

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
your own settings and sessions are not touched. It prints each check as `ok` or `FAIL`,
things about the machine (no sound server, say) as `env`, and saves a report under
`~/.cache/GranularMusicVisualizer/validation/`.

Without `--audio-capture` it never records: it only asks the sound server, read-only,
whether System audio could run. With it, each window run also captures the default output
for six seconds, and `scripts/capture_probe_linux.py` records it for `--capture-seconds`,
measures it, logs the session and replays it. Play something first; nothing is played for
you:

```sh
uv run python scripts/validate_linux.py --audio-capture --capture-seconds 30
uv run python scripts/capture_probe_linux.py --seconds 600 --every 30   # a longer run on its own
```

Then, by hand:

```sh
uv run gmv-visualizer                                  # System audio: Start, play music; or a file
ls ~/.local/share/GranularMusicVisualizer/sessions     # the sessions it recorded
uv run gmv visualizer --replay ~/.local/share/GranularMusicVisualizer/sessions/SESSION
cat ~/.config/GranularMusicVisualizer/*.conf           # the settings it remembered
gamescope -- uv run gmv-visualizer                     # optional: a nested Gamescope
```

To see System audio follow a change of output, play music, start capture, and choose
another output in the system's sound settings (or plug in headphones): the status line
says the default output changed, and the picture carries on from the new output. This was
not done on the validation machine.

## Known limitations

- **System audio is the default output only.** Linux cannot capture one chosen output
  device, one application or its child processes, or a microphone; those are not offered.
  What is captured is the whole mix the default output plays, after any effects in front of
  it, at whatever volume the applications play.
- System audio needs a PulseAudio-compatible server and `pactl` and `parec` (see
  [System audio](#system-audio)). Times are anchored on delivery, not taken from the
  server; the sound's way to the monitor and the speakers is not measured. Each stream's
  first 100 ms are measured and not analysed, and an output clock drifting against the
  computer's costs a block or two of audio and an analysis restart every quarter of an hour
  or so (11 ppm, as measured on the CachyOS machine).
- A file is analysed at playing speed and is not played aloud.
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
