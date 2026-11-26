# GMV on a Steam Deck: brief for a Claude Code session on the Deck

You are helping the owner run Granular Music Visualizer (GMV) on their Steam Deck to **demo
it**. This is not a development machine. Install, start, check and troubleshoot GMV; do not
develop it here. The owner's own guide is [STEAM_DECK_DEMO.md](STEAM_DECK_DEMO.md); read it
too, because it is what the owner sees. The installer is
[`packaging/steamdeck/gmv-deck.sh`](../packaging/steamdeck/gmv-deck.sh).

## Rules for this session

- **The owner is not a programmer.** Explain in plain language, give concrete steps (what to
  tap, what to play, what to look for), and never hand them a technical question to decide.
  Do what you can yourself in Konsole; ask them only for what needs their hands or eyes.
- **No development here.** Do not edit GMV's code, commit, push or open pull requests from
  the Deck. `AGENTS.md`'s milestone workflow is for the development PC. If something in the
  repository needs fixing, write down what failed, the exact messages and the Deck's details
  (`gmv-deck.sh check`) for the owner to take to the development session.
- **Leave the system alone.** Do not change the system volume, the default output, audio
  routing or other applications' behaviour, and do not load PipeWire modules. Do not make
  SteamOS writable (`steamos-readonly disable`), do not run `pacman` on the host, and do not
  set or ask for a sudo password. GMV needs none of that. The one exception is the Distrobox
  fallback below, and only after the owner agrees.
- **Raw audio recording stays off.** Do not switch on debug audio recording in GMV's
  Settings → Advanced. `gmv-deck.sh listen` and `smoke` save no audio.
- **Nothing private leaves the Deck.** Do not upload music, sessions, logs, settings or file
  names anywhere.
- **Weights come from two places only:** the owner's own copy, or the project's
  [v0.1.1 release](https://github.com/Ralkero/granular-audio-visualizer/releases/tag/v0.1.1).
  The weights are `activity_v2.npz` and the guitar model `guitar_mn10_v1`.
  - The release's Windows zip is 157 MB, SHA-256
    `cdb4f15341e3567c680ea9a32fc5bdce088c3251ca41be80f7e0f92e1c577155`. Its `models` folder
    holds the weights and `MODEL-LICENCES.txt`.
  - The licence is CC BY-NC-SA 4.0, for non-commercial use only.
  - Ask the owner before downloading. Never take weights from anywhere else, and never pass
    them on.
- **Do not set `GMV_SCREEN`.** It picks a monitor on the owner's desktop PC and means nothing
  here.
- **Say what you saw.** Keep "ran on this Deck" apart from "the docs say it worked on the
  CachyOS PC". Before telling the owner a hands-on run worked, look for its traces (below).

## What GMV is, in brief

A Python desktop application (Python 3.11 or newer, PySide6 / Qt 6, NumPy, SciPy, soundfile,
pydantic, psutil; GPL-3.0-or-later). It analyses audio live and draws it in three looks:
*Spectrum bars*, and *Radial* with the theme *Classic* or *Radiant*. Radiant draws with
OpenGL 3.3 shaders on the GPU and falls back to a geometry-only drawing when it cannot.

- **Capture on Linux** (`src/gmv/sources/system_pulse.py`): GMV asks `pactl --format=json`
  for the default output (sink) and its monitor, then reads the monitor with a `parec`
  subprocess as float32 stereo. That is what the system plays, never a microphone.
- It follows a change of default output by itself. Nothing is played, and no volume, device
  or routing is changed. Sessions keep the analysis only, never the audio.
- **Other sources:** an audio file (analysed at playing speed and not played aloud), and
  replay of a recorded session.
- **Linux validation so far:** one CachyOS desktop (KDE Plasma Wayland, PipeWire 1.6, AMD
  RX 9070 XT, Mesa). Details are in [LINUX.md](LINUX.md), *What has been checked*. **Nothing
  has been run on a Steam Deck or under Steam's Game Mode before you.**

## The install route, and why

**Recommended: a user-space install with `uv`, in one folder of the home directory**, done
by `packaging/steamdeck/gmv-deck.sh`.

| Route | Verdict |
|---|---|
| **`uv` in the home folder (this one)** | No root, nothing in the read-only system, survives SteamOS updates. It installs exactly the versions in `uv.lock`, the same way GMV was validated from a checkout on CachyOS (Python 3.14, PySide6 6.12 wheel with its own Qt). Capture runs on the host, so `pactl`/`parec` reach PipeWire directly. |
| `pacman` / the Arch package (`packaging/arch/`) | Not on SteamOS: the root is read-only and is replaced by every SteamOS update. |
| Flatpak | Does not exist yet (planned; audit F). Recording the monitor from inside the sandbox is also unproven. |
| AppImage | None is built. The audit (F.1) argued against one. |
| Distrobox (preinstalled since SteamOS 3.5) | Works in principle, but heavier (a whole Arch container), and launching from Game Mode through it is more fragile. **Fallback only**, see below. |

What `gmv-deck.sh install` does:

1. It downloads uv 0.13.10 and GMV at the pinned commit
   (`main` at `cdc75e8758bdfb62bf747cc9470e53602c9d2abc`) from GitHub, and checks both
   against SHA-256 sums written in the script.
2. It runs `uv sync --locked --python 3.14` with Python, the cache and the environment all
   inside the install folder (`UV_PYTHON_PREFERENCE=only-managed`, `UV_NO_CONFIG=1`).
3. It writes a launcher and a menu entry with icons, then checks that GMV imports and that
   Qt's `wayland` and `xcb` plugins load.

`install --ref BRANCH` installs another branch or commit instead; that download cannot be
checksum-checked, so say so. `install --recognition` adds ONNX Runtime for the guitar
recognition model. Run `install` again to repair or update; settings and sessions are not in
the install folder and are kept.

| Path | What |
|---|---|
| `~/granular-music-visualizer/app/` | GMV's source at the pinned commit (a checkout: `paths.checkout()` is true) |
| `~/granular-music-visualizer/venv/` | The Python environment (`venv/bin/gmv-visualizer`, `venv/bin/python`) |
| `~/granular-music-visualizer/python/`, `cache/`, `uv/` | uv's Python, its download cache, uv itself |
| `~/granular-music-visualizer/launch.sh` | The launcher used by the menu entry and by Steam |
| `~/granular-music-visualizer/last-launch.log` | Launched without a terminal: the environment it got, and everything GMV printed |
| `~/granular-music-visualizer/BUILD` | Which build is installed, when, with which Python |
| `~/.local/share/applications/granular-music-visualizer.desktop` | Menu entry (Multimedia). The file name matches the app's Wayland app id, so the taskbar shows its icon |
| `~/.local/share/icons/hicolor/*/apps/granular-music-visualizer.png` | Icons |
| `~/.config/GranularMusicVisualizer/Granular Music Visualizer.conf` | Settings (`QSettings`), kept on update and removal |
| `~/.local/share/GranularMusicVisualizer/sessions/` | Recorded sessions (analysis only), kept |
| `~/.local/share/GranularMusicVisualizer/models/` | The owner's weights files, kept |

About 625 MB on disk; the download is about 200 MB.

## Steps

In Desktop Mode, in Konsole, from the home folder:

```bash
curl -fLo gmv-deck.sh https://raw.githubusercontent.com/Ralkero/granular-audio-visualizer/main/packaging/steamdeck/gmv-deck.sh
bash gmv-deck.sh check
bash gmv-deck.sh install
bash ~/granular-music-visualizer/gmv-deck.sh smoke
```

Until the pull request that adds this brief is merged, use `docs/steam-deck-demo` in place
of `main` in the URL.

1. **`check` changes nothing.** It shows the OS, glibc (2.34 or newer needed), the display
   variables, the system libraries Qt uses, `pactl`/`parec`, whether the sound server
   answers and whether the default output has a monitor, and the install state.
   - Read it before installing: a missing `libxcb-cursor.so.0` or a missing `parec` changes
     the plan (see troubleshooting).
   - It deliberately prints no device names.
2. **`install`** (see above). Its last step prints the Python, PySide6 and Qt versions and
   whether each display plugin loads.
3. **`smoke`** runs `app/scripts/validate_linux.py` with the environment's Python.
   - The window opens and closes by itself four times in about two minutes (Wayland, Wayland
     again for remembered settings, then `xcb` and `wayland` forced). Each run analyses a
     generated WAV and FLAC, records and replays a session, and shows Radiant.
   - It uses scratch XDG folders and records nothing.
   - It ends with `N of N checks passed`, prints the Radiant renderer string (GPU and Mesa
     version), and saves a JSON report in `~/.cache/GranularMusicVisualizer/validation/`.
   - Tell the owner to leave the window alone while it runs.

Then, with the owner:

4. Ask the owner to play music on the Deck, then run
   `bash ~/granular-music-visualizer/gmv-deck.sh listen 20`.
   - This runs `app/scripts/capture_probe_linux.py`, which records the default output's
     monitor through GMV's own engine for 20 s, prints the level every 3 s and checks the
     result.
   - Expect *It hears the music*. A level that stays near −100 dBFS or lower means silence.
   - The probe exits with 2 when capture cannot run at all and prints why.
5. Ask the owner to start GMV from the menu, press **Start** with *System audio (PipeWire)*
   selected, play music, and say what they see.
   - Then **verify the traces** yourself: a new folder in
     `~/.local/share/GranularMusicVisualizer/sessions/`, its `manifest.json`, the settings
     file's modification time, and `~/granular-music-visualizer/last-launch.log` with a line
     `gmv: Qt platform wayland (Qt's own choice)`.
   - Settings → Diagnostics in the window shows the capture state, DSP times and the
     Radiant renderer; the owner can read it out, or you can ask them to.
6. Optional, only if the owner wants Game Mode: follow the owner's guide, step 6. Afterwards,
   `last-launch.log` shows what Game Mode gave the process.

## Facts that matter on the Deck

| Topic | What to know |
|---|---|
| Python, Qt | uv's own CPython 3.14 in `~/granular-music-visualizer/python`; the system Python is not used. PySide6-Essentials 6.12 wheel (`manylinux_2_34`) with its own Qt 6.12, which uses system libraries: `libxcb-cursor`, `libxkbcommon(-x11)`, `libwayland-client`, `libEGL`/`libGL` (Mesa), `fontconfig`, `dbus`. On Arch these come with `qt6-base`, which Plasma needs, so SteamOS is expected to have them (inference; `check` shows it). |
| Audio | SteamOS uses PipeWire with `pipewire-pulse`. `pactl` and `parec` come from `libpulse`, which `pipewire-pulse` and Plasma's audio applet depend on in Arch, so they are expected (inference; `check` shows it). GMV records the default output's monitor: speakers, headphones, HDMI or Bluetooth, whatever is the default, and follows a change by itself (tested against a stand-in only). |
| Desktop Mode | SteamOS 3.8 and newer run Plasma 6.4 on **Wayland** by default, so Qt takes `wayland`. `GMV_QT_PLATFORM=xcb` forces X11 through Xwayland. |
| Game Mode | Gamescope gives programs an X11 display and normally no Wayland socket. GMV detects Gamescope (`GAMESCOPE_WAYLAND_DISPLAY` and `DISPLAY` set, `WAYLAND_DISPLAY` not set; `src/gmv/host.py`) and asks Qt for `xcb`. This rule has **never run inside a real Gamescope**. Overrides: Steam launch option `GMV_QT_PLATFORM=xcb %command%` (or `wayland`, or `auto` to let Qt choose). The launcher logs these variables and `SteamGameId` to `last-launch.log`. |
| Steam's library path | Steam can start shortcuts with `LD_LIBRARY_PATH` pointing into its own runtime (old libraries; this broke Blender launched from Steam). `launch.sh` restores `SYSTEM_LD_LIBRARY_PATH` (or clears the path) when it sees `steam-runtime`/`ubuntu12_*` in it, and logs that it did. `LD_PRELOAD` (Steam's overlay) is left alone. Untested on a real Steam. |
| Radiant on the Deck GPU | Needs OpenGL 3.3 / GLSL 330. The Deck's RDNA 2 APU on Mesa radeonsi offers OpenGL 4.6 (audit D.1), so it should start. **Speed is unmeasured**: the Deck has 8 compute units, against the RX 9070 XT it was measured on. If it stutters: Settings → Visualizer → Radiant *Effects: Off* (geometry only), or *Classic* / *Spectrum bars*. The `D` key toggles an on-screen stats overlay; Diagnostics shows the renderer and frame cost. |
| Frame rate | *Automatic*: the display rate divided by the whole number that brings it nearest 66 fps. An LCD Deck (60 Hz) gets 60 fps and an OLED Deck (90 Hz) 90 fps, which costs battery. *Every refresh* is the other choice; there is no 45 fps setting. |
| Screen | 1280×800 fits: the visualizer needs at least 720×480; the settings drawer is 440 px wide. The window starts windowed; **Full screen (F11)** is a button in Settings → Visualizer. **Esc** and **D** need a keyboard (Steam + X). |
| Separate windows | File dialogs (**Choose…**, **Replay a session…**, **Export logs…**), the colour dialog, the *Show* menu and drop-down lists open windows of their own. Game Mode handles several windows of one program poorly (audit E.4), so do these in Desktop Mode. |
| Weights (optional) | Looked for in this order: `GMV_ACTIVITY_MODEL` / `GMV_GUITAR_MODEL`, then `~/.local/share/GranularMusicVisualizer/models/activity_v2.npz` and `guitar_v1.npz`, then `app/.model-cache/`. A file chosen in Settings wins. **Without them everything runs**: spectrum, level, onsets and all three looks; the drum, bass, vocal and guitar layers are drawn as *tentative* (pale, dashed), and Settings → Analysis says "no weights chosen". The guitar recognition model `guitar_mn10_v1` is a `.json` with an `.onnx` file beside it. It is never loaded just by lying in a folder: it is chosen in Settings → Analysis → Guitar. It needs `install --recognition`, and has never run on Linux. Whether the v0.1.1 release's files load in this newer build is untried; `from gmv.activity import ActivityModel; ActivityModel(path)` in `venv/bin/python` loads `activity_v2.npz` the way the app does, as a quick check. |
| Settings | Qt `QSettings`: `~/.config/GranularMusicVisualizer/Granular Music Visualizer.conf`, written on close. `smoke` never touches it (scratch XDG folders). |

## Troubleshooting

| Symptom | Look at | What to do |
|---|---|---|
| `install` stops at step 1 or 2 | the message | No network, or GitHub unreachable: run it again. A checksum message means the download is not the expected file; do not work around it, report it. |
| `install` stops at step 3 | uv's output above the message | Network, or disk space (`check` shows free space). Run again; uv continues from its cache. |
| Step 5 says `A library it needs is missing: libxcb-cursor.so.0` | `check` libraries | Desktop Mode (Wayland) does not need it: demo there. For Game Mode, use the Distrobox fallback. |
| The `wayland` plugin does not load | step 5, `last-launch.log` | Start with `GMV_QT_PLATFORM=xcb ~/granular-music-visualizer/launch.sh`. If that works, the menu entry can carry it: edit the `Exec=` line to `env GMV_QT_PLATFORM=xcb …/launch.sh`. |
| GMV does not appear in Game Mode, or a black screen or spinner | `last-launch.log`: `GAMESCOPE_WAYLAND_DISPLAY`, `DISPLAY`, `WAYLAND_DISPLAY`, the `gmv: Qt platform …` line, any traceback | If the platform is not `xcb`, set the launch option `GMV_QT_PLATFORM=xcb %command%`. `symbol lookup error` or `version … not found` with `library path reset: no` means Steam's runtime path took another form: try the launch option `LD_LIBRARY_PATH= %command%` and report it. `ERROR: ld.so: object '…gameoverlayrenderer.so' … cannot be preloaded` lines are harmless. |
| **Start** says it needs `pactl` and `parec` | `check` | SteamOS normally has them. Without them, System audio cannot run on the host: offer *Audio file* for the demo, or the Distrobox fallback. |
| *No PulseAudio-compatible sound server answered* | `pactl info` | `systemctl --user status pipewire-pulse.socket`. Do not restart or reconfigure the owner's audio without asking. |
| *…has no default output…* / *…has no monitor source…* | `pactl get-default-sink`, `pactl list short sources` | The owner picks an output in the Deck's sound settings. Do not change it yourself. |
| Status *waiting for playback* while music plays | `listen` | The monitor delivers nothing: is the music app playing on the Deck, unmuted? Bluetooth or HDMI as default output: check that its monitor exists. |
| Radiant slow or hot | Diagnostics, `D` overlay | *Effects: Off*, or another look; plug in the charger. Note the figures for the development PC. |
| The menu entry is missing | `~/.local/share/applications/` | `kbuildsycoca6`, or log out and in; `launch.sh` always works directly. |

### Fallback: Distrobox (only if the host route cannot work, and only with the owner's OK)

Untested anywhere. It uses the Arch package recipe that was validated on CachyOS, inside an
Arch container, so `pacman` can supply what SteamOS lacks:

```bash
distrobox create --name gmv --image docker.io/library/archlinux:latest
distrobox enter gmv
# inside the container:
sudo pacman -Syu --needed base-devel git libpulse xcb-util-cursor
git clone https://github.com/Ralkero/granular-audio-visualizer.git
cd granular-audio-visualizer/packaging/arch && makepkg --syncdeps --install
distrobox-export --app granular-music-visualizer
```

The container shares the home folder and the PipeWire socket, so settings, sessions and
capture behave as on the host. Game Mode through Distrobox is less likely to work than the
host route.

## Reporting back

Give the owner a short plain-language summary: what works, what does not, and what to do
next. For the development PC, collect:

- the `gmv-deck.sh check` output;
- the `smoke` report path and its last lines;
- the `listen` summary;
- `last-launch.log` from a menu start and, if tried, a Game Mode start;
- the Diagnostics text from the window.

Label all of it as run on the Deck, with the SteamOS version and Deck model (LCD or OLED).
Leave out song titles, file names and anything personal. Recording these results in the
repository (`docs/LINUX.md`, *What has been checked*) happens on the development PC, not
here.
