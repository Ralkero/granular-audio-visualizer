# Linux and Steam Deck port: audit and plan

Status when written (2026-11-22): audit only. No application code, configuration,
dependency or packaging file was changed to write this. The port had not started.

> Update, 2026-11-23: phase 1a has since been implemented (section M) and validated on a
> real CachyOS desktop the same day (section N), and phase 1b, capture of the system's
> default output through PipeWire's PulseAudio service, implemented and validated there too
> (section O). Sections A to L are the audit as it was written on 2026-11-22 and have not
> been changed. Setup, launch and the CachyOS results in full: [LINUX.md](LINUX.md).
>
> Update, 2026-11-24: phase 1c, a native Arch package built with `makepkg` and installed
> with Pacman on the CachyOS desktop (section P). It is not the Flatpak of F and phase 4.

Written 2026-11-22 against branch `main` at `084b443` ("Radiant radial theme with a palette
of its own (phase 1)"), level with `origin/main`, with uncommitted Radiant work in the tree
(see B.1). Research was done the same day; where a source could not be fetched, or a fact
can only be settled on a physical Steam Deck, this says so.

Labels used throughout:

- **Verified (repo)**: read in this repository's code.
- **Verified (source)**: stated by a fetched external source, listed in section J.
- **Inference**: engineering reasoning or general knowledge not confirmed by a fetched source.
- **Deck test**: can only be settled on Steam Deck hardware.

The target experience: GMV and Spotify both installed on a Steam Deck and added to Steam;
Spotify playing; GMV launched alongside it, capturing what SteamOS plays and drawing it with
the full Radiant GPU renderer; the user switching between GMV, Spotify and a game in Gaming
Mode.

---

## A. Executive assessment

**Overall portability: good.** The analysis engine, event model, visual state, all three
renderers (Classic, Spectrum bars, Radiant on the GPU), session recording and replay are
platform-neutral Python, NumPy, SciPy and Qt. None of them needs to change for Linux.
What is Windows-specific is concentrated in three places: system-audio capture
(`sources/windows.py`, `sources/system_native.py`, `sources/application.py` and the .NET
helper), a few paths in `paths.py`, and the PowerShell/PyInstaller build. The engine already
talks to every capture backend through one small contract (`AudioSource.run` emitting
`AudioBlock`s), and it already accepts any sample rate and channel count.

**Steam Deck feasibility: realistic, with three things that must be proven on the device.**

1. *System-output capture.* SteamOS uses PipeWire with its PulseAudio-compatible server
   (verified, source). Recording the default sink's monitor is a standard, documented
   operation (verified, source). Whether a Flatpak with only `--socket=pulseaudio` can record
   a monitor on SteamOS is likely (indirect evidence: Flatpak OBS desktop-audio capture) but
   was not confirmed: **Deck test**.
2. *Gaming Mode windowing.* Gamescope gives apps an X11 display (Xwayland) and, with stock
   flags, no Wayland socket (verified, gamescope source). Qt must therefore run on `xcb` in
   Gaming Mode and on `wayland` in Desktop Mode (SteamOS 3.8+). Running two non-Steam apps
   side by side and switching between them works in user reports, with reported focus bugs
   for multi-window and Flatpak apps (verified, user reports only): **Deck test**.
3. *Radiant performance on the Deck APU.* radeonsi supports OpenGL 4.6 / GLSL 4.60
   (verified, source), far above the GLSL 330 the shaders need. Whether the scene, smoke and
   bloom passes hold 60 fps at 1280×800 on 8 RDNA 2 compute units within a sensible power
   budget is unmeasured: **Deck test**. The desktop measures about 5 ms per frame on an RX
   9070 XT at 1920×1080; the Deck has a small fraction of that GPU, so a lower internal
   render scale will probably be needed (inference).

**Most significant blockers (in order of risk).**

1. No Linux capture backend exists. "Default system output" raises "Windows WASAPI
   required. PipeWire adapter is planned." on Linux (verified, repo).
2. Timestamp integrity under load. On Windows, Python-callback capture lost audio when the
   interpreter was busy, which is why capture moved to a separate native helper (verified,
   repo: `docs/PHASE1_1_REPORT.md`). A Linux backend must be designed so that a late reader
   is *delayed*, not *dropped*, and must report any loss.
3. Gaming Mode focus and window handling for a Qt app launched as a non-Steam game,
   especially from Flatpak.
4. Redistribution of model weights. `Build App.ps1` states the packaged models are
   "derived from non-commercial data" and the build "must not be redistributed while it
   holds" them (verified, repo). A Flathub release cannot include them.

**Recommended approach.**

- One shared application. Add a Linux capture backend behind the existing `AudioSource`
  contract, make paths and source discovery platform-aware, and leave the engine and
  renderers alone.
- Capture through the PulseAudio protocol served by PipeWire (`pipewire-pulse`), recording
  the default sink's monitor with a pull-model stream whose data the server buffers.
  Native PipeWire is reserved for later per-application capture.
- Keep OpenGL and the existing GLSL 330 shaders. Do not write a Vulkan backend.
- Distribute as a Flatpak on the KDE runtime with the PySide BaseApp, installed as a normal
  Python package, not through PyInstaller.
- Force Qt's `xcb` platform under Gamescope; keep a single top-level window in Gaming Mode.

---

## B. Current architecture

### B.1 Repository state at the time of the audit

- Branch `main`, HEAD `084b443`, level with `origin/main` after a fetch.
- Uncommitted work (preserved; not touched by this audit): Radiant phase 2 and 3 (the
  independent GPU renderer in `src/gmv/visualizer/themes/radiant_gpu/`, 15 shader files),
  the procedural smoke pass (`shaders/smoke.frag`), five plasma and three smoke palette
  roles (25 in all), audio-time spectrum smoothing in `VisualState`, the GPU lifecycle and
  shutdown fixes, their tests (`tests/test_radiant_plasma.py`, `tests/test_radiant_smoke.py`,
  `tests/plasma_gpu_probe.py`), render scripts and documentation. 17 tracked files modified,
  7 untracked paths.
- The only file this audit adds is this document.

### B.2 Data flow

```
AudioSource.run()  ──AudioBlock──►  BoundedQueue  ──►  WindowAssembler ─► Analyzer (dsp.py)
 (capture thread)                                       (analysis thread)
                                                              │
                     FeatureFrame / MusicalEvent / InstrumentActivity / InstrumentEstimate
                                                              ▼
                       EventBus ──► SessionWriter (messages.jsonl), RemoteEngine pipe
                                                              ▼
                       GUI process: VisualState ─► Canvas (Classic, Bars) / PlasmaView (Radiant GPU)
```

- `Engine` (`engine.py`) runs capture and analysis as two threads. With
  `analysis_process=True` (the default), `RemoteEngine` (`remote.py`) runs the whole engine
  in a `multiprocessing` "spawn" worker and streams messages to the GUI process over a pipe
  (verified, repo).
- `AudioBlock` (`sources/base.py`): interleaved little-endian float32 bytes, `frames`,
  `sample_rate`, `channels`, a session-timeline `timestamp`, `capture_monotonic`,
  `received_monotonic`, `segment`, `discontinuity`, `timestamp_quality` (verified, repo).
- The analyzer accepts any sample rate and channel count. A change of rate, channel count or
  segment restarts its window (`dsp.py`, `WindowAssembler`). The delayed-context worker and
  the recognition worker resample internally with `scipy.signal.resample_poly` (verified,
  repo). Nothing downstream assumes 48 kHz or stereo.

### B.3 Component portability

| Component | Location | Classification | Notes |
|---|---|---|---|
| Analysis (DSP, beat tracker, pitch, instrument analyzers, intensity, dominance) | `dsp.py`, `beat.py`, `pitch.py`, `instruments/`, `activity.py` | Portable | NumPy/SciPy only |
| Context recognition | `recognition/` | Portable | onnxruntime (optional) has Linux x86-64 wheels (inference: verify for the runtime's Python) |
| Message models, schema, bus, buffer | `models.py`, `contract.py`, `bus.py`, `buffer.py` | Portable | pydantic |
| Engine and worker process | `engine.py`, `remote.py` | Portable | `spawn` context works on Linux. `remote.py`'s docstring reasons from Windows' system-wide clock; Linux's `perf_counter` is `CLOCK_MONOTONIC`, also system-wide (inference; verify with `time.get_clock_info`). The offset is measured and corrected anyway (verified, repo) |
| Session recording and replay | `session.py`, `visualizer/replay.py` | Portable | Paths come from `paths.py` |
| File source | `sources/file.py` | Portable | soundfile wheels bundle libsndfile on Linux (inference) |
| System capture, PortAudio/WASAPI | `sources/windows.py` | Replacement needed | `pyaudiowpatch`, `pycaw`, `comtypes`; imports are lazy, so the module imports safely on Linux (verified, repo) |
| System capture, native helper | `sources/system_native.py` | Replacement needed | Spawns `ProcessLoopback.exe` with `CREATE_NO_WINDOW`; QPC timestamps |
| Application capture | `sources/application.py`, `native/ProcessLoopback/` | Replacement needed (later) / not ported | Windows process-loopback API, .NET 9 + NAudio. Not needed for system capture on Linux |
| Device and application discovery | `sources/windows.py: list_output_devices`, `application.list_audio_applications` | Replacement needed | Both return `[]` off Windows (verified, repo) |
| Source selection | `app/controller.py: audio_source`, `SOURCES` | Configuration | Hard-wires `system_audio_source` → Windows backends |
| Paths | `paths.py` | Configuration | Packaged sessions go to `%LOCALAPPDATA%` or, without it, `~/GranularMusicVisualizer/sessions`; helper paths are `.exe` |
| Settings | `QSettings` in `app/window.py` | Portable | On Linux an INI file under `$XDG_CONFIG_HOME/GranularMusicVisualizer/` (inference: Qt's documented Unix behaviour); in Flatpak that is `~/.var/app/<id>/config` |
| Classic and Spectrum bars | `visualizer/window.py`, `visualizer/layers/` | Portable | QPainter |
| Radiant GPU renderer | `visualizer/themes/radiant_gpu/` | Conditional | Needs a desktop OpenGL ≥ 3.3 context; see D |
| Radiant reduced QPainter fallback | `visualizer/themes/radiant.py` | Portable | |
| Window shell, settings drawer | `app/window.py` | Conditional | Works; handheld usability issues, see G |
| Fonts | `QFont("Segoe UI")` in `visualizer/window.py` | Configuration | Falls back to the system sans font on Linux; layout uses font metrics |
| Icons | `packaging/app.ico`, `icon.png` | Configuration | Linux desktop entries want PNG/SVG in the hicolor theme |
| Diagnostics, renderer report | `app/window.py`, `visualizer/window.py` | Portable | Already reports GPU backend and context |
| Packaged entry and self-check | `packaging/app.py` | Conditional | `play_tone` uses `pyaudiowpatch` |
| Build | `Build App.ps1`, `Build Process Capture.ps1` | Replacement needed | PowerShell, PyInstaller, .NET download |
| Research tooling (evidence, review, corpus) | `evidence/`, `review/`, `corpus/` | Not shipped | Some Windows defaults (e.g. `fpcalc.exe`); out of scope |
| Tests | `tests/` | Conditional | `test_sources.py`, `test_system_native.py`, `test_app.py`, `test_recovery.py` reference Windows capture; the GPU probe pins `QT_QPA_PLATFORM` only on Windows |

### B.4 Rendering architecture (as it bears on Linux)

- The canvas (`visualizer/window.py: Canvas`) is a raster `QWidget` painting the shell
  (header, timeline, details, notes) and Classic and Spectrum bars.
- Radiant on the GPU is a `QOpenGLWidget` (`radiant_gpu/view.py: PlasmaView`) under a
  transparent canvas in a `QStackedLayout` with `StackAll`. It is created only when Radiant
  is first shown in a visible window. It compiles 9 programs once per context, renders into
  16-bit float FBOs at 0.7 of the view's size, then a two-level bloom and a composite into
  `defaultFramebufferObject()`. It never reads back. Resources are released when the
  context is about to be destroyed and when the window closes (verified, repo).
- Shaders are `#version 330`, use `in`/`out`, a bound empty VAO and attribute-less
  fullscreen triangles from `gl_VertexID`. FBO format `GL_RGBA16F`; timing via
  `QOpenGLTimeMonitor` (optional; failure disables timing only) (verified, repo).
- No `QSurfaceFormat.setDefaultFormat` is called; the view takes Qt's default context.
  On Windows that is a 4.6 compatibility context. The offscreen helper asks for 3.3 core
  (verified, repo).
- Any failure (no OpenGL modules, offscreen platform, compile/link failure, FBO failure, a
  frame failing) hides the view and falls back to the QPainter Radiant, reported in
  Diagnostics (verified, repo).

### B.5 Windows packaging today

`Build App.ps1` runs PyInstaller (onedir) on `packaging/app.py`, adds the shaders as data,
copies the .NET helper and a private .NET 9 runtime (downloaded, SHA-512 checked), copies
local models and writes `models/defaults.json`, then runs `--self-check`. None of this is
reusable for Linux except the idea of a self-check.

---

## C. Audio capture recommendation

### C.1 What SteamOS provides

- SteamOS uses PipeWire and WirePlumber (verified, source: Collabora 2022). Exact versions
  are not stated in any patch note found.
- `pipewire-pulse` is "a drop-in replacement for the PulseAudio daemon" (verified, source).
  Each sink has a monitor source named `<sink>.monitor`; `pactl` defines `@DEFAULT_MONITOR@`
  (verified, source).
- Native PipeWire can capture a sink's output with a `Stream/Input/Audio` stream carrying
  `stream.capture.sink=true`, optionally aimed with `target.object` (verified, source:
  PipeWire 1.6 keys). WirePlumber re-links a stream whose target disappears unless
  `node.dont-reconnect` is set (verified, source).
- Neither API needs to change routing, volume or the graph to record: this is ordinary
  loopback of what the OS already plays.

### C.2 Options compared

| Option | Monitor capture | Timing information | Flatpak access | Maturity of Python access | Verdict |
|---|---|---|---|---|---|
| **PulseAudio protocol via pipewire-pulse** (libpulse) | Yes: record `<default sink>.monitor` | `pa_stream_get_latency`, `pa_timing_info` (monitor latency may be negative) | `--socket=pulseaudio`: the standard permission | libpulse via ctypes (no new dependency) or SoundCard/pasimple | **Recommended for the first port** |
| Native PipeWire (libpipewire) | Yes: `stream.capture.sink=true` | `pw_stream_get_time_n`: monotonic `now`, graph `ticks`, `delay` | Needs `--filesystem=xdg-run/pipewire-0` (Spotify and OBS on Flathub use it; Flathub prefers minimal permissions) | No mature binding: pipewire-gobject is alpha; would need a small C helper or CFFI | Later, for per-application capture |
| PortAudio (sounddevice) | Only if the PortAudio build has the PulseAudio host API (in master, not in the last release, 19.7.0 of 2021), or through ALSA/JACK shims | Callback ADC times; on Windows they were unusable | Depends on what the runtime ships | Mature | **Not recommended**: same callback model that lost audio on Windows, uncertain monitor visibility |
| `pw-record`/`parec` subprocess writing raw float32 | Yes | None per packet: arrival-anchored only | Whether the runtime ships the tools is unverified | Trivial | Fallback or prototyping only |

### C.3 Recommended backend

**Module:** `src/gmv/sources/system_pulse.py`, class `PulseMonitorSource(AudioSource)`,
`kind = "system_audio"`. The name says what it talks to (the PulseAudio protocol), which
works against both pipewire-pulse and a real PulseAudio.

**Binding:** a small ctypes layer over `libpulse.so.0` using the threaded main loop
(`pa_threaded_mainloop`) and a record stream. That gives server info (default sink),
source info (monitor name, sample spec, channel map), overflow notification
(`pa_stream_set_overflow_callback`) and timing (`pa_stream_get_latency`). libpulse is on
SteamOS and in the Flatpak runtimes (inference: pipewire-pulse serves libpulse clients). It
adds no Python dependency. SoundCard (BSD-3, PulseAudio via CFFI, supports
`include_loopback=True`) is a good fast prototype for Phase 2's first spike, but it is a
self-described hobby project with no response guarantee and adds `cffi`; pasimple has no
overflow callback.

**Behaviour:**

1. Ask the server for the default sink; record from `<that sink>.monitor` by name (not
   `@DEFAULT_MONITOR@`), so a change of default sink is detectable.
2. Sample spec: `float32le`, the monitor's own rate (no resampling; the Deck is expected to
   run at 48 kHz: Deck test), **two channels**. pipewire-pulse downmixes a 5.1/7.1 HDMI sink
   with its standard channel map. Report the sink's real channel count and map in
   `describe()` so nothing is hidden. (Reason: the analyzer's default weighting is
   `1/min(channels, 2)` per channel, which sums more than two channels at 0.5 each; that is a
   pre-existing behaviour, not something to change for the port. Stereo input keeps
   Linux identical to the tested Windows path.)
3. **Pull model with server-side buffering.** Read from a dedicated capture thread inside
   the engine worker. `fragsize` about 10 ms (matching `capture_ms`), `maxlength` about 2 s.
   If the Python side is late, audio waits in the server buffer; only a full buffer loses
   audio, and that is reported by the overflow callback. This is the decisive difference
   from the Windows PortAudio callback, whose callback had to be served on time. ctypes
   releases the GIL during the blocking calls.
4. Emit `AudioBlock`s exactly as the other backends do. No change to `dsp.py` or anything
   downstream.
5. Status messages reuse the existing vocabulary: `capturing`, `waiting` (no data, e.g.
   nothing playing; a monitor still delivers silence while the sink runs: Deck test),
   `recovering`.
6. Device changes: subscribe to server and sink events (`pa_context_subscribe`). On a new
   default sink, a removed sink, a stream failure or an overflow, close and reopen with a new
   `segment`. This matches how `NativeSystemAudioSource` treats an endpoint change.
7. Optional "Specific output device" on Linux: list sinks with their monitors and record a
   chosen monitor by name. Defer to after the default path works.

**Escalation path.** If sustained-capture measurements (`scripts/clock_probe.py`, already
built for exactly this) show losses under load, move the same libpulse loop into a separate
capture process that writes the existing `GMV1` packet format (`application.read_packet`)
to a pipe. The packet header has a 64-bit timestamp field that can carry `CLOCK_MONOTONIC`
nanoseconds instead of QPC. That reuses the Windows helper architecture without .NET.

### C.4 Timestamp strategy

- **Clock domain.** Use one monotonic clock for everything: Python's `perf_counter`, which
  on Linux should be `CLOCK_MONOTONIC` (inference; the Phase 1 test must assert
  `time.get_clock_info("perf_counter")`). The engine's `timeline_origin` is already a
  `perf_counter` value.
- **Per segment, a contiguous sample clock** (frames / rate), as `QpcSampleClock` and
  `CaptureClock` do. Anchor it at segment start to
  `arrival_monotonic − stream_latency − frames_in_this_read / rate`, where `stream_latency`
  comes from `pa_stream_get_latency` (for a monitor this may be negative; clamp the anchor
  so no published timestamp is in the future, as `NativeSystemAudioSource` does with its
  QPC lead).
- **Drift detection.** On every read, compute the same anchor estimate and compare it with
  the sample clock. The monitor runs on the sink's device clock; the CPU monotonic clock
  runs on another. Expect a slow ppm-level slope plus scheduling jitter. Follow the Windows
  policy: never smooth or clamp silently; when the disagreement exceeds a tolerance (start
  at 10 ms, as on Windows), re-anchor, mark a discontinuity, count it, and expose the
  signed error and its maximum in `describe()`.
- **Quality label:** `pulse_latency_anchored_sample_clock`, or
  `pulse_arrival_anchored_sample_clock` when latency is unavailable.
- **Suspend and resume.** `CLOCK_MONOTONIC` does not advance during suspend on Linux,
  `CLOCK_BOOTTIME` does (inference: Linux clock semantics). Compare the two on each read
  (`time.clock_gettime(time.CLOCK_BOOTTIME)`): a jump in their difference means the machine
  slept, and the backend starts a new segment. This needs no D-Bus permission. Also expect
  the stream to fail or the sink to be recreated after resume (SteamOS 3.8 fixed resume
  audio underruns, verified source), which the reopen path already handles.
- **Do not** use GUI-timer arrival times; the GUI never sees capture.

### C.5 Channels and formats

| Case | Handling |
|---|---|
| Stereo sink (built-in speakers, headphones, most Bluetooth) | Captured as-is |
| Mono sink | Request 2 channels; the server duplicates. Report the real map |
| 5.1/7.1 sink (some HDMI/dock setups) | Request 2 channels; the server downmixes with its map. Report the real map. No level change by GMV |
| Rate other than 48 kHz | Capture at the sink's rate; the analyzer and models already handle any rate |
| Sample format | Always request `float32le`; the server converts |

### C.6 Per-application capture (future scope)

- System capture records the **sum of everything** SteamOS plays: Spotify, the game, the
  browser, notification sounds. The UI must say so ("All system audio").
- Two possible routes, both unverified on the Deck:
  - PulseAudio `pa_stream_set_monitor_stream` records one sink input of a monitor
    (verified, source: PulseAudio API). Whether pipewire-pulse implements it is unverified.
  - Native PipeWire: find the application's output stream node in the registry by
    `application.process.binary` / `application.name` (e.g. Spotify), capture with
    `target.object` set to it, `node.passive=true`, and re-target when the app recreates
    its stream (verified, source: PipeWire keys, WirePlumber linking). In Flatpak this needs
    `xdg-run/pipewire-0`.
- Spotify runs in its own Flatpak sandbox, but its PipeWire stream is visible to the graph;
  only `pipewire.*` properties are security-trustworthy (verified, source).
- Not a prerequisite for the first port.

### C.7 The .NET helper on Linux

Not needed. On Windows it exists to (a) keep WASAPI capture out of a busy Python interpreter
and (b) do process loopback. On Linux, (a) is handled by server-side buffering of a pull
stream (C.3), with a native capture process as the escalation path; (b) is future scope with
a PipeWire-native design. `sources/system_native.py` and `sources/application.py` stay
Windows-only.

---

## D. GPU rendering compatibility

### D.1 OpenGL on the Deck

- radeonsi has supported OpenGL 4.6 since Mesa 20.0; 4.6 "is only available if requested at
  context creation", and compatibility contexts "may report a lower version" (verified,
  source: Mesa 20.0 notes). Mesamatrix lists radeonsi at 4.6 / GLSL 4.60 (verified, source).
  User glxinfo dumps from Decks show OpenGL 4.6 including a 4.6 compatibility profile
  (user reports).
- Everything the renderer uses is core in OpenGL 3.3 or earlier: GLSL 330, `RGBA16F`
  color-renderable (GL 3.0), timer queries (GL 3.3 / ARB_timer_query), VAOs
  (inference: GL specification knowledge; the Khronos wiki was not fetchable).
  `QOpenGLTimeMonitor` needs GL ≥ 3.3 or ARB_timer_query (verified, source: Qt docs).
- Conclusion: **GLSL 330 is a sound target for the Deck.** No shader change is expected.

### D.2 Context creation risk and the smallest fix

- The view takes Qt's default context. On Mesa, a default request is normally answered with
  a compatibility context at the highest version (4.6 on radeonsi per user dumps), in which
  `#version 330` compiles. If Qt or the environment picked OpenGL ES (for example
  `QT_OPENGL=es2`), the shaders would fail and the fallback would engage.
- Smallest change (Phase 3): before `QApplication` is created, set an explicit default
  surface format: desktop OpenGL, version 3.3, core profile, plus
  `Qt.AA_ShareOpenGLContexts`. Qt documents that setting the default format before
  `QApplication` is "mandatory on some platforms" (verified, source). Make it Linux-only at
  first if Windows validation is not repeated in the same change.
- Wayland: Qt renders through EGL; best performance needs `EXT_platform_wayland` (verified,
  source). Under Gamescope on `xcb`, Qt uses GLX or EGL-on-X11 depending on its build
  (inference).

### D.3 Composition

The transparent raster canvas over the `QOpenGLWidget` relies on Qt's widget composition
(the GL widget renders to an FBO that Qt composes with the backing store). This is the same
mechanism on all platforms (verified, source: Qt docs, "all rendering happens into an OpenGL
framebuffer object"), but it was only exercised on Windows. Validate on Linux under both
`wayland` and `xcb`.

### D.4 Vulkan

Recommendation: **keep OpenGL**.

- Every required feature is available through Mesa's OpenGL; the shaders are small
  fragment passes; Qt's OpenGL widget integration already works and is tested.
- A Vulkan backend would mean `QRhi` or a separate Vulkan widget, SPIR-V shader builds and a
  new resource lifecycle, for no feature the renderer lacks.
- Valve recommends Vulkan for games' battery life (verified, source: Steamworks
  recommendations). The relevant lever here is how much work each frame does (render scale,
  pass count, frame rate), not the API. Revisit only if Deck measurements show a driver-level
  problem with OpenGL under Gamescope; none is documented (searched, not found).

### D.5 Deck performance planning

- Deck GPU: 8 RDNA 2 CUs at 1.6 GHz (verified, source). Desktop measurements of the full GPU
  scene (about 5 ms/frame at 1920×1080 on an RX 9070 XT; smoke 0.6 ms) do not transfer.
- The native panel is 1280×800 (LCD 60 Hz, OLED up to 90 Hz) (verified, source). That is
  about half the pixels of 1080p, but on a much smaller GPU.
- Levers already in the code: `SCENE_SCALE` (0.7 today), bloom at half and quarter of the
  scene, early exit in the smoke pass. Expect to need a lower scene scale on the Deck (for
  example 0.5) and possibly a cheaper bloom: Deck test.
- **Frame rate.** `automatic_rate()` divides the refresh rate by a whole number nearest 66:
  60 Hz gives 60 fps; **an OLED Deck at 90 Hz gives 90 fps** (verified, repo). On a battery
  device, 45 (90/2) or 60 should be considered, behind a setting.
- **Hidden-window cost.** The timer keeps ticking while another app is in front in Gaming
  Mode (inference: Gamescope keeps the process running; no documented throttling). Add
  throttling when the window is not exposed or not focused, if Qt receives those events under
  Gamescope: Deck test.
- Memory: 16 GB unified, 1 GB dedicated to the GPU by default, more shared on demand
  (verified, source, 2021 FAQ). Renderer buffers are 12.6 MiB at 1080p; not a concern.

### D.6 Resource lifecycle

The existing lifecycle code (release on `aboutToBeDestroyed`, explicit release on window
close, rebuild in `initializeGL`, failure → fallback) is platform-neutral. On the Deck the
process is frozen during suspend, not killed (verified, source: kernel freezer), so the GL
context should survive; keep the defensive paths and test resume on the device.

---

## E. SteamOS and Gaming Mode

### E.1 Facts

- SteamOS is Arch-based with a read-only, image-updated root (A/B, RAUC); `/etc` and `/var`
  are writable (verified, source: Collabora 2024). Valve recommends Flatpak for extra apps;
  Flathub is the default remote in Discover (verified, source: install guides; Valve's
  Desktop FAQ was not fetchable).
- Desktop Mode uses Wayland by default since SteamOS 3.8 (stable 18 June 2026), KDE Plasma
  6.x (verified, source). Current stable is 3.9.5 (24 Oct 2026); 3.10 previews exist.
- Gaming Mode runs apps under Gamescope with a nested Xwayland. Gamescope unsets
  `WAYLAND_DISPLAY` and sets it for children only with `--expose-wayland`; it always sets
  `DISPLAY` and `GAMESCOPE_WAYLAND_DISPLAY` (verified, source: gamescope `main.cpp`, master).
  Whether Valve's session passes `--expose-wayland` today was not verified. A 2022 report
  shows a Flatpak app with Wayland enabled still running on X11 in Gaming Mode.
- Qt chooses `wayland` first if `WAYLAND_DISPLAY` is set **or** `XDG_SESSION_TYPE=wayland`,
  else `xcb` if `DISPLAY` is set; `QT_QPA_PLATFORM` overrides (verified, source: qtbase 6.8).

### E.2 Launching GMV

- **Desktop Mode:** normal Flatpak launch; Qt picks `wayland`.
- **Gaming Mode:** add GMV as a non-Steam game (Desktop Mode → Steam → Add a Game → Add a
  Non-Steam Game, or KDE menu → "Add to Steam") (verified, source: third-party guides). The
  shortcut runs `flatpak run <app-id>`.
- **Platform choice (Phase 1 code, tiny):** in `application()`, before `QApplication`, if
  `GAMESCOPE_WAYLAND_DISPLAY` is set, `WAYLAND_DISPLAY` is not, and `QT_QPA_PLATFORM` is
  unset, set `QT_QPA_PLATFORM=xcb`. Users can also put `QT_QPA_PLATFORM=xcb %command%` in the
  shortcut's launch options. The Flatpak must grant `fallback-x11` and `share=ipc`, and it
  must be checked that the sandbox actually receives X11 in Gaming Mode: Deck test.
- **Scaling:** Gamescope sessions apply no app scaling (one developer's statement, 2026);
  on `xcb` Qt will likely see 96 DPI at 1280×800, i.e. scale 1 (inference). Size the UI for
  that panel explicitly (G).
- **Fullscreen:** start fullscreen in Gaming Mode (Gamescope shows the focused window full
  screen anyway; inference). Do not rely on window positioning, which Wayland forbids.

### E.3 Spotify

- Spotify's own Linux client is officially "not a platform that we actively support"; its
  official channels are Snap and a Debian/Ubuntu repository (verified, source).
- Flathub `com.spotify.Client` is **unverified, community-maintained**: "not verified by,
  affiliated with, or supported by Spotify". It repackages the official Snap (verified,
  source). It is the usual Deck install, via Discover.
- The web player needs Widevine; it works in a browser Flatpak (verified, source: Spotify
  support).
- GMV needs no Spotify integration: it records what the system plays. No Spotify API, login,
  DRM interaction or process injection is involved.
- Optional later nicety: Spotify exposes MPRIS (`org.mpris.MediaPlayer2.spotify`) (verified,
  source: its manifest). That could label sessions with track metadata, at the cost of a
  `--talk-name` permission. Not part of the port.

### E.4 Running both, and switching

- Steam added a window switcher in April 2022 (tap the Steam button) (verified, source).
- Users report that several non-Steam apps can run at once and be switched from the Steam
  menu, and that Spotify keeps playing while a game runs (verified, user reports and guides,
  2022–2024).
- Reported problems (user reports; current behaviour unverified):
  - Switching between **windows of the same app** often does not work.
  - Gamescope focus depends on X properties the Steam client sets (`STEAM_GAME`); non-Steam
    windows lacked them in 2022 (closed as fixed 2023).
  - umu-launcher documents a Gamescope issue affecting **Flatpak apps in Steam mode** (window
    not shown, spinner only) and works around it by setting `STEAM_GAME` on its X windows
    from `SteamGameId`.
  - Several Chromium/Electron apps exiting together crashed (2022–23). GMV plus Spotify is
    only one Chromium app.
- Consequences for GMV (inference):
  - **One top-level window** in Gaming Mode. Today `QColorDialog` (Radiant palette), the
    file dialogs (replay, export, model weights) and popup menus open separate windows.
    In Gaming Mode, replace them with in-window controls or hide them.
  - If the Flatpak focus issue reproduces, set `STEAM_GAME` on GMV's X window from the
    `SteamGameId` environment variable, as umu does. Small, xcb-only code.
- Audio in the background: no documented muting of background apps; GMV keeps capturing
  whatever plays (inference; Deck test).

### E.5 Gaming Mode limitations to expect

| Limitation | Source | Mitigation |
|---|---|---|
| X11 only (no Wayland socket) | gamescope source | Force `xcb` |
| Multi-window focus | user reports | Single window |
| Flatpak window not shown | umu FAQ | `STEAM_GAME` property if needed |
| No app scaling | developer statement | Size UI for 1280×800 |
| Controller is mouse/keyboard emulation | guides | Big targets; see G |
| Desktop apps don't auto-update from Gaming Mode | guide | Document |

---

## F. Packaging architecture

### F.1 Recommendation: Flatpak, Python installed natively

| Approach | For | Against |
|---|---|---|
| **Flatpak on `org.kde.Platform` + `io.qt.PySide.BaseApp`, GMV pip-installed** | Qt and PySide built for the runtime; host Mesa through the GL extension; Flathub-conformant; small app layer; updates by `flatpak update` | PySide version follows the BaseApp branch (6.9–6.11 maintained; 6.8 EOL) |
| PyInstaller onedir inside a Flatpak | Mirrors the Windows build | Duplicates Qt from the wheel; wheel Qt needs libs (e.g. xcb-cursor) the runtime must provide; two layers of bundling |
| AppImage | No sandbox | Not the SteamOS-recommended path; no updates; libc and GL coupling |

**Recommended:** the first row. `pyproject.toml` already allows any PySide6 6.x ≥ 6.8.

### F.2 Manifest outline (Phase 4)

- `runtime: org.kde.Platform`, `runtime-version: '6.10'` or `'6.11'`;
  `base: io.qt.PySide.BaseApp`, matching `base-version` (verified, source: BaseApp README).
  Use the BaseApp cleanup script; QtWebEngine is not needed.
- Modules: NumPy and SciPy (from the BaseApp if provided there, otherwise wheels),
  pydantic, psutil, soundfile, optional onnxruntime, then GMV itself. Pip downloads are
  not allowed during the build: generate module sources with `flatpak-pip-generator`
  (inference: standard Flathub practice).
- Shaders: packaged as package data so `resources.FOLDER` (beside the module) resolves.
  Today they are added to the Windows build with `--add-data`; for a pip install they must be
  declared as package data in `pyproject.toml` (Phase 4 change).
- Desktop entry, AppStream metainfo, PNG/SVG icons.
- `command: gmv-visualizer` (the existing gui script).

### F.3 Permissions (`finish-args`)

| Permission | Why | Status |
|---|---|---|
| `--socket=wayland` | Desktop Mode | required |
| `--socket=fallback-x11` | Gaming Mode (Xwayland) and X11 desktops | required; Flathub requires it with `wayland` |
| `--share=ipc` | X11 shared memory | required with X11 sockets (Flathub linter) |
| `--device=dri` | OpenGL | required |
| `--socket=pulseaudio` | Record the sink monitor through pipewire-pulse | required; monitor recording under this permission alone is a **Deck test** |
| `--filesystem=xdg-run/pipewire-0:ro` | Native PipeWire (per-app capture) | **not** for the first port |
| File access for replay/export | Use the file-chooser portal (Qt uses it in Flatpak) | no static filesystem permission |
| `--share=network` | — | not needed |

No `--filesystem=home`/`host`. No D-Bus names for the first port.

### F.4 GPU drivers

The Flatpak uses the `org.freedesktop.Platform.GL` extension; `GL.default` is a Mesa build
matched to the runtime and installed automatically (verified, source: Flatpak docs). Nothing
is bundled or replaced. Regressions in GL.default Mesa have happened (verified, forum
report): include the Mesa version in Diagnostics (already shown in the renderer report's
OpenGL version string).

### F.5 Persistent storage

| Data | Today (Windows) | Linux plan |
|---|---|---|
| Settings and palettes | `QSettings` (registry) | `QSettings` → `$XDG_CONFIG_HOME/GranularMusicVisualizer/…` (Flatpak: `~/.var/app/<id>/config`). No code change |
| Sessions | `%LOCALAPPDATA%/GranularMusicVisualizer/sessions` | `$XDG_DATA_HOME/GranularMusicVisualizer/sessions` via `QStandardPaths.AppLocalDataLocation` or `XDG_DATA_HOME`. **Change `paths.sessions_folder()`**: today a packaged Linux build would write to `~/GranularMusicVisualizer/sessions` |
| Logs | inside session folders | unchanged |
| Cache | none | `$XDG_CACHE_HOME` if needed |
| Models | beside the executable | `/app/share/gmv/models` (read-only) for bundled ones; user-chosen files through the portal. Change `paths.models_folder()` |
| Shaders | `_internal/gmv/...` | package data under `/app/lib/python3.x/site-packages/gmv/...` |
| Exports | user-chosen path | file-chooser portal |

Flatpak keeps `~/.var/app/<id>` across updates (verified, source: Flatpak conventions). The
working directory must never be assumed writable.

### F.6 Application ID

`com.ralkero.GMV` is only valid if the project controls the domain `ralkero.com` (verified,
source: Flathub requirements). The repository is on GitHub (`Ralkero/granular-audio-visualizer`),
so the conforming ID is **`io.github.ralkero.granular-audio-visualizer`** (code-host prefix,
lowercase user, dash allowed only in the last component). Renaming the repository would
change it. Decide before Phase 4; the ID is hard to change after publication.

### F.7 Licensing and distribution

- **Model weights:** must not be redistributed (B.5). A public Flatpak ships without them
  (models off, as a checkout does today); a private local build may include them for the
  owner's own use.
- PySide6/Qt: LGPLv3 used as shared libraries from the runtime/BaseApp, unmodified (inference:
  standard compliance; confirm in Phase 4).
- Python dependencies: NumPy, SciPy (BSD), pydantic (MIT), psutil (BSD), soundfile (BSD) with
  libsndfile (LGPL), onnxruntime (MIT) (inference: verify against the lock file in Phase 4).
- Shaders are the project's own.
- Spotify is a separate, proprietary, community-packaged Flatpak; GMV neither depends on it
  nor redistributes it.

---

## G. UI and handheld compatibility

Panel: 7" (LCD) / 7.4" (OLED), 1280×800, about 216 / 205 px per inch (inference from
specs). Steam Input emulates mouse (trackpad/stick) and keyboard; touch arrives as mouse or
touch events.

### G.1 Findings in the current UI (verified, repo)

| Item | Problem on the Deck |
|---|---|
| Settings drawer fixed at 440 px | A third of the screen; fine, but its contents are small |
| Style sheet fonts 11–15 px | About 1.3–1.8 mm tall at 216 ppi: hard to read at arm's length |
| Buttons, combos ~24 px tall; slider handles 12 px | Touch targets of 2–3 mm; guidance is ~7–9 mm (inference) |
| `QColorDialog` for palette colours | A separate window (see E.4) |
| `QFileDialog` for replay, export, model files | Separate windows; under Flatpak, portal dialogs |
| Popup `QMenu` for lanes, combo popups | Separate popup windows under Gamescope: Deck test |
| Tooltips carry much of the explanation | No hover on touch |
| Keyboard-only actions: F11, Esc, D (stats) | No keyboard in Gaming Mode; F11 has a drawer button, D and Esc do not |
| Section heights resized by dragging a 6 px edge | Too fine for touch |
| Visualizer minimum 720×480; scale clamps 0.85–1.6 by height | Works at 800 px tall (scale ≈ 0.89) |
| Instrument chips and section buttons in the top bar | Usable but small |

### G.2 Prioritized changes

1. **Must (port is unusable without):** start fullscreen in Gaming Mode; a large on-screen
   button to open and close settings; Start/Stop reachable by touch; no separate windows in
   Gaming Mode (hide file-based features there, replace the colour dialog with in-drawer
   swatches and hex presets).
2. **Should:** a "handheld" density (larger fonts, ≥ 48 px controls, bigger slider handles)
   chosen automatically under Gamescope or at small screen sizes; in-drawer explanations
   instead of tooltips; the stats toggle as a button.
3. **Could:** a simplified Deck drawer (source, theme, palette preset, Expressive Mode,
   instruments); Steam Input template recommendation (trackpad as mouse, A = click,
   B = close drawer, Start = settings).
4. **Later:** native controller navigation. Not required for the first port.

---

## H. Phased implementation plan

```
Phase 1 ─► Phase 2 ─┐
   │                ├─► Phase 4 ─► Phase 5 ─► Phase 6
   └──► Phase 3 ────┘
```
Phase 2 and 3 can run in parallel after Phase 1. Phase 5 needs a Steam Deck.

### Phase 1: platform abstraction and native Linux startup
- **Scope:** run GMV from a checkout on a Linux desktop (CachyOS) with file input, replay and
  all three renderers; Windows unchanged.
- **Changes:** `paths.py` (XDG sessions and models when not Windows); a platform-aware
  system-source factory and discovery (`app/controller.py`, a small `sources/` registry)
  so Linux offers only sources it has; `application()` sets `QT_QPA_PLATFORM=xcb` under
  Gamescope; Windows-only tests marked to skip off Windows; `play_tone` in the self-check
  made platform-aware; `scripts/render_radiant_plasma.py`'s platform pin.
- **Dependencies:** none. `uv sync` on Linux already skips `pyaudiowpatch`/`pycaw` by marker.
- **Deliverables:** startable Linux checkout; test suite running on Linux.
- **Acceptance:** full suite passes on Linux except tests explicitly marked Windows-only; file
  playback and replay show Classic, Bars and Radiant GPU; Windows suite still passes;
  `time.get_clock_info("perf_counter")` is monotonic.
- **Risks:** Qt plugin/library gaps on the host (e.g. `libxcb-cursor`); GPU probe needs a
  real display.
- **Complexity:** Low.

### Phase 2: Linux system-audio backend
- **Scope:** `PulseMonitorSource` (C.3–C.5).
- **Changes:** new `sources/system_pulse.py` (+ a small ctypes libpulse module); source
  factory selects it on Linux; Diagnostics shows its `describe()`; tests with a fake
  libpulse layer for the clock, reopen and format logic.
- **Dependencies:** Phase 1.
- **Deliverables:** "All system audio" capture on Linux desktop.
- **Acceptance:** captures Spotify/browser playback; 10-minute `clock_probe` run with zero
  unreported losses and bounded drift (report the slope); default-sink switch and device
  unplug produce a new segment and resume; suspend/resume produces a new segment; no change
  to `dsp.py`.
- **Risks:** server buffering behaviour of pipewire-pulse for record streams; negative monitor
  latency; Python load; Bluetooth codec switches changing the rate.
- **Complexity:** Medium.

### Phase 3: full renderer compatibility
- **Scope:** Radiant GPU, Classic, Bars on Linux under `wayland` and `xcb`.
- **Changes:** explicit default `QSurfaceFormat` (D.2) if needed; GPU probe runnable on Linux;
  fixes found by testing.
- **Dependencies:** Phase 1.
- **Acceptance:** `tests/test_radiant_plasma.py` and `test_radiant_smoke.py` GPU tests pass on
  Linux AMD; repeated open/resize/close and theme switching clean; fullscreen; nested
  `gamescope -- gmv-visualizer` on CachyOS renders.
- **Risks:** composition of the transparent canvas over the GL widget; GLX vs EGL differences.
- **Complexity:** Low–Medium.

### Phase 4: Linux packaging
- **Scope:** Flatpak (F).
- **Changes:** manifest, metainfo, desktop file, icons, pip module sources; shaders as package
  data; build script for Linux (no PowerShell); self-check runnable in the Flatpak.
- **Dependencies:** Phases 2 and 3.
- **Acceptance:** installs and runs on CachyOS from a local repo; records the monitor with only
  the listed permissions; settings and sessions persist across reinstall; no models in the
  public build.
- **Risks:** BaseApp version vs PySide requirements; NumPy/SciPy/onnxruntime wheel availability
  for the runtime's Python; whether `--socket=pulseaudio` allows monitor recording.
- **Complexity:** Medium.

### Phase 5: Gaming Mode integration (Steam Deck required)
- **Scope:** the target experience.
- **Changes:** whatever device testing requires; likely `STEAM_GAME` window property, startup
  fullscreen, frame-rate cap, hidden-window throttling.
- **Dependencies:** Phase 4; a Steam Deck.
- **Acceptance:** E.4 scenario end to end; music continues; GMV receives audio while in the
  background and in front; switching works with a game running; suspend/resume; Bluetooth
  headphones; dock HDMI audio.
- **Risks:** Gamescope focus; Flatpak window display; power use.
- **Complexity:** Medium (unknowns dominate).

### Phase 6: handheld refinement
- **Scope:** G.2 items 2–3; Deck render scale and frame-rate defaults; power tuning; startup
  and shutdown time.
- **Dependencies:** Phase 5 measurements.
- **Acceptance:** usable with touch and trackpad only; 60 fps (or the chosen cap) sustained
  with Radiant at the chosen scale, measured on the Deck; battery draw recorded.
- **Complexity:** Medium.

---

## I. Testing matrix

| Environment | Purpose | Phases |
|---|---|---|
| Windows 11 desktop (RX 9070 XT) | No Windows regression: full suite, packaged self-check | all |
| Native Linux desktop: CachyOS (AMD, Arch-based, PipeWire, KDE) | Startup, capture, renderer, Flatpak; nested `gamescope` approximation of Gaming Mode | 1–4 |
| SteamOS Desktop Mode | Flatpak install, Wayland path, permissions | 4–5 |
| SteamOS Gaming Mode | Target | 5–6 |

CachyOS success does not prove SteamOS behaviour: different PipeWire/WirePlumber
configuration, Gamescope session, Steam client integration, hardware.

| Scenario | CachyOS | Deck Desktop | Deck Gaming |
|---|---|---|---|
| **Audio:** default monitor | ✓ | ✓ | ✓ |
| Spotify; browser; a game; Spotify + game | ✓ | ✓ | ✓ (game + Spotify only meaningful here) |
| Built-in speakers; Bluetooth headphones; dock/HDMI | partly | ✓ | ✓ |
| Default-output switch; unplug/replug | ✓ | ✓ | ✓ |
| 10-minute and 60-minute capture (`clock_probe`) | ✓ | ✓ | ✓ |
| Pause/resume playback | ✓ | ✓ | ✓ |
| Suspend/resume | ✓ (laptop/desktop sleep) | ✓ | ✓ |
| **Visualizer:** Classic, Bars, Radiant GPU, four instruments, Expressive, custom palette, beats/smoke, replay | ✓ | ✓ | ✓ |
| **Lifecycle:** start, clean exit, repeated start/exit, resize, fullscreen, device switch while running | ✓ | ✓ | ✓ |
| Switching between Gaming Mode apps; GPU context after switching | – | – | ✓ |
| **Performance:** capture and analysis CPU, GPU frame time, memory, frame pacing, drops | ✓ | ✓ | ✓ |
| Battery draw, thermals, OLED 90 Hz behaviour | – | – | ✓ |

**Physical-Deck-only checks:** monitor recording from the Flatpak; Gaming Mode platform
(`env` inside the shortcut); window focus and switching with Spotify and a game; Radiant frame
time on the Deck GPU; power; suspend/resume audio graph behaviour; Bluetooth codec changes;
OLED refresh rate; touch and Steam Input usability.

---

## J. External research sources

All fetched 2026-11-22 unless noted. "Third-party" and "user report" mark non-authoritative
sources.

| Title | URL | Date | Establishes | Caveat |
|---|---|---|---|---|
| Steam Deck Tech Specs (OLED) | https://www.steamdeck.com/en/tech | undated | Zen 2 4c/8t, 8 RDNA 2 CUs, 16 GB LPDDR5, 1280×800 up to 90 Hz, 4–15 W | |
| Steam Deck Tech Specs (LCD) | https://www.steamdeck.com/en/tech/deck | undated | LCD 1280×800 60 Hz | |
| Steam Deck developer FAQ | https://partner.steamgames.com/doc/steamdeck/faq | 2021 | Unified memory; 1 GB dedicated, up to 8 GB shared; suspend design | Pre-launch, old |
| Steam Deck recommendations | https://partner.steamgames.com/doc/steamdeck/recommendations | undated | Vulkan recommended for games | Recommendation |
| Collabora: SteamOS 3.6 atomic updates | https://collabora.com/news-and-blog/news-and-events/steamos-3-6-how-the-steam-deck-atomic-updates-are-improving.html | 2024-05-10 | A/B, RAUC, `/etc` `/var` writable | |
| GamingOnLinux: SteamOS 3.8 | https://gamingonlinux.com/2026/06/steamos-3-8-is-out-with-initial-steam-machine-support-desktop-mode-upgrades-new-graphics-drivers | 2026-06-18 | Desktop Mode Wayland by default; resume underrun fix | Quotes Valve's notes |
| GamingOnLinux: SteamOS 3.9.5 | https://www.gamingonlinux.com/2026/10/steamos-3-9-5-released-with-plasma-6-7-linux-7-2-security-fixes-better-steam-frame-wireless-adapter-support/ | 2026-10-24 | Current stable | |
| Steam Deck News | https://www.steamdeck.com/en/news | 2026-11 | 3.9.6 beta, 3.10 previews | |
| KDE: Steam Deck and Wayland | https://planet.kde.org/marco-martin-2026-07-01-steam-deck-and-wayland/ | 2026-07-01 | Desktop Mode on Wayland | |
| gamescope README | https://github.com/ValveSoftware/gamescope | master | Xwayland per app | |
| gamescope `main.cpp` | https://raw.githubusercontent.com/ValveSoftware/gamescope/master/src/main.cpp | master | `WAYLAND_DISPLAY` only with `--expose-wayland`; sets `DISPLAY`, `GAMESCOPE_WAYLAND_DISPLAY` | Master may differ from shipped |
| Mozilla bug 1802802 | https://bugzilla.mozilla.org/show_bug.cgi?id=1802802 | 2022-11 | Wayland-enabled Flatpak ran on X11 in Gaming Mode | Old |
| Flathub Dolphin PR #178 | https://github.com/flathub/org.DolphinEmu.dolphin-emu/pull/178 | 2024-01 | `xdg-run/gamescope-0` socket; name not fixed | |
| gamescope issue #1107 | https://github.com/ValveSoftware/gamescope/issues/1107 | 2024 | Embedded Wayland server gaps | Open |
| qtbase `qguiapplication.cpp` (6.8, dev) | https://raw.githubusercontent.com/qt/qtbase/6.8/src/gui/kernel/qguiapplication.cpp | branch | Platform plugin selection rules | |
| Qt: Wayland and Qt; QPA; QOpenGLWidget; High DPI; QOpenGLTimeMonitor | https://doc.qt.io/qt-6/wayland-and-qt.html, https://doc.qt.io/qt-6/qopenglwidget.html, https://doc.qt.io/qt-6/highdpi.html, https://doc.qt.io/qt-6/qopengltimemonitor.html | Qt 6.12 | EGL on Wayland; default format before QApplication; FBO rendering; timer queries need GL 3.3 | |
| KDE discuss: automatic scaling | https://discuss.kde.org/t/automatic-scaling-for-qtquick/46163 | 2026-04 | Gamescope session applies no app scaling | One developer |
| pimylifeup: add a non-Steam game | https://pimylifeup.com/steam-deck-add-non-steam-game/ | undated | Workflow | Third-party |
| RetroDECK: adding to Steam | https://retrodeck.readthedocs.io/en/latest/wiki_steam/add-to-steam/ | undated | `flatpak run` shortcuts | Third-party |
| GamingOnLinux: window switcher | https://www.gamingonlinux.com/2022/04/steam-deck-gets-a-lock-screen-window-switcher-and-more/ | 2022-04-26 | Steam-button window switcher | |
| Steam Community threads (window switching; app widgets; Electron exit) | https://steamcommunity.com/app/1675200/discussions/1/3269060419611357975/ · https://steamcommunity.com/app/1675200/discussions/2/3186865924604258371 · https://steamcommunity.com/app/1675200/discussions/1/3416557114747120335 | 2022–2024 | Multi-app use; focus and crash reports | User reports |
| steam-for-linux #8513 | https://github.com/ValveSoftware/steam-for-linux/issues/8513 | 2022–2023 | Gamescope focus and `STEAM_GAME` | Closed by author |
| umu-launcher FAQ | https://github.com/Open-Wine-Components/umu-launcher/wiki/Frequently-asked-questions-(FAQ) | undated | Flatpak-in-Steam-mode window issue; `STEAM_GAME` workaround | Third-party |
| XDA: Spotify on Steam Deck | https://www.xda-developers.com/how-install-spotify-steam-deck/ | undated | Discover install; Spotify with games | Third-party |
| Collabora: PipeWire Bluetooth | https://www.collabora.com/news-and-blog/news-and-events/pipewire-bluetooth-support-status-update.html | 2022-04-29 | Deck uses PipeWire + WirePlumber | |
| pipewire-pulse module and man page | https://docs.pipewire.org/page_module_protocol_pulse.html · https://docs.pipewire.org/page_man_pipewire-pulse_1.html | 1.6.11 | Drop-in server; default record fragment 2 s | |
| PipeWire keys; props; pw_time; pw_stream; access module; pw-link; pw-cat | https://docs.pipewire.org/group__pw__keys.html · https://docs.pipewire.org/page_man_pipewire-props_7.html · https://docs.pipewire.org/structpw__time.html · https://docs.pipewire.org/group__pw__stream.html · https://docs.pipewire.org/page_module_access.html · https://docs.pipewire.org/page_man_pw-link_1.html · https://docs.pipewire.org/page_man_pw-cat_1.html | 1.6.11 | `stream.capture.sink`, `target.object`, timing fields, Flatpak client access | |
| WirePlumber linking policy | https://pipewire.pages.freedesktop.org/wireplumber/policies/linking.html | 0.5.19 | Target re-linking; stream-to-stream targets | |
| PulseAudio `pa_timing_info`; `stream.h` | https://freedesktop.org/software/pulseaudio/doxygen/structpa__timing__info.html · https://freedesktop.org/software/pulseaudio/doxygen/stream_8h.html | current | Record latency terms; negative monitor latency; `pa_stream_set_monitor_stream` | |
| pactl(1) | https://man.archlinux.org/man/pactl.1.en | libpulse 17.0 | `@DEFAULT_MONITOR@` | |
| SoundCard; pasimple; pulsectl; sounddevice; PortAudio releases; pipewire-gobject; pipewire_python | https://github.com/bastibe/SoundCard · https://pypi.org/project/pasimple/ · https://pypi.org/project/pulsectl/ · https://pypi.org/project/sounddevice/ · https://github.com/PortAudio/portaudio/releases · https://pypi.org/project/pipewire-gobject/ · https://pypi.org/project/pipewire_python/ | 2021–2026 | Library capabilities, status, licences (C.2) | |
| Flatpak sandbox permissions; conventions; extensions; runtimes | https://docs.flatpak.org/en/latest/sandbox-permissions.html · https://docs.flatpak.org/en/latest/conventions.html · https://docs.flatpak.org/en/latest/extension.html · https://docs.flatpak.org/en/latest/available-runtimes.html | latest | Permission names; XDG dirs; GL extension | |
| Flathub linter; requirements; verification | https://docs.flathub.org/docs/for-app-authors/linter · https://docs.flathub.org/docs/for-app-authors/requirements · https://docs.flathub.org/docs/for-app-authors/verification | current | Socket rules; app ID rules | |
| obs-pipewire-audio-capture; Flathub OBS manifest | https://github.com/dimtpap/obs-pipewire-audio-capture · https://github.com/flathub/com.obsproject.Studio/blob/master/com.obsproject.Studio.json | current | Precedent for `xdg-run/pipewire-0` | |
| Flathub discourse: read-only PipeWire socket | https://discourse.flathub.org/t/what-is-the-difference-if-the-pipewire-socket-is-read-only/11951 | 2026-03 | `:ro` suffices | Community |
| io.qt.PySide.BaseApp | https://github.com/flathub/io.qt.PySide.BaseApp | branch 6.11 updated 2026-11-02 | BaseApp exists; maintained branches | |
| KDE EOL runtimes list | https://discuss.kde.org/t/end-of-life-kde-runtimes-list/42566 | 2026 | 6.8 runtime EOL | |
| Spotify for Linux | https://www.spotify.com/us/download/linux/ | © 2026 | Unsupported platform; Snap and apt | |
| Flathub `com.spotify.Client` and manifest | https://flathub.org/apps/com.spotify.Client · https://github.com/flathub/com.spotify.Client/blob/master/com.spotify.Client.json | 2026 | Unverified community package; finish-args incl. pipewire-0 and MPRIS | |
| Spotify web player support | https://support.spotify.com/us/article/webplayer/ | — | Browsers; Widevine | Search snippet only |
| Mesa 20.0.0 release notes | https://docs.mesa3d.org/relnotes/20.0.0.html | 2020-02-19 | GL 4.6 on radeonsi | |
| Mesamatrix | https://mesamatrix.net/ | 2026-11-21 | radeonsi GL 4.6 / GLSL 4.60 | |
| ARB_timer_query | https://registry.khronos.org/OpenGL/extensions/ARB/ARB_timer_query.txt | 2010 | Timer query semantics | |
| Kernel: freezing of tasks | https://docs.kernel.org/power/freezing-of-tasks.html | current | Processes frozen during suspend | Generic Linux |
| GamingOnLinux: SteamOS 3.7.25 / 3.8.5 beta; OLED resume audio fix | https://gamingonlinux.com/2026/05/steamos-3-7-25-and-3-8-5-beta-released-bug-fixes-and-better-dgpu-video-memory-management · https://gamingonlinux.com/2025/03/fixing-a-steam-deck-oled-bug-for-the-linux-kernel-6-14-release-was-a-major-challenge | 2026-05-21; 2025-03-28 | Resume-related audio fixes | |

**Could not fetch:** Valve's Steam Deck Desktop FAQ (JavaScript-rendered), the Arch Wiki, the
PipeWire/WirePlumber GitLab wikis and issues, Mesa `features.txt`, the Khronos wiki. Where
those would have been the best source, the claim above is marked inference or user report.

---

## K. Outstanding unknowns

1. Does a Flatpak with `--socket=pulseaudio` (and no PipeWire socket) see and record
   `<sink>.monitor` on SteamOS? (`flatpak run --command=pactl <id> list sources short`)
2. In Gaming Mode, what are `DISPLAY`, `WAYLAND_DISPLAY`, `XDG_SESSION_TYPE` and
   `GAMESCOPE_WAYLAND_DISPLAY` inside a non-Steam Flatpak shortcut? Does the sandbox get X11?
3. Does a Flatpak Qt window display and take focus in Gaming Mode, or does it need
   `STEAM_GAME`?
4. Do Qt popups (combo boxes, menus) work under Gamescope?
5. Can GMV, Spotify and a game all stay running and be switched between today (2026)?
6. Does Gamescope stop or throttle unfocused apps, and does Qt get expose/focus events?
7. Radiant GPU frame time on the Deck at 1280×800 for each scene scale; power draw.
8. The pipewire-pulse record-stream buffering limits in practice; monitor latency values;
   whether the monitor delivers silence or nothing when no app plays.
9. What happens to PipeWire nodes and the monitor stream across suspend/resume on the Deck.
10. Sample rate and channel map of the Deck's speakers, Bluetooth (by codec) and dock HDMI.
11. Whether pipewire-pulse supports `pa_stream_set_monitor_stream` (per-app capture via Pulse).
12. Mesa, PipeWire and WirePlumber versions in the shipping SteamOS image.
13. Whether NumPy, SciPy and onnxruntime wheels exist for the chosen runtime's Python.
14. Whether Flathub would accept `xdg-run/pipewire-0` for a future per-app capture feature.

---

## L. Recommended first implementation task

**"Phase 1a: GMV starts and runs on Linux from a checkout, with files and replay, and Windows
is unchanged."** No audio backend yet.

Concrete changes:

1. `src/gmv/paths.py`
   - `sessions_folder()`: when packaged and not on Windows, use `$XDG_DATA_HOME` (default
     `~/.local/share`) `/GranularMusicVisualizer/sessions`.
   - `models_folder()`: when packaged and not on Windows, a share directory (for a Flatpak,
     `/app/share/gmv/models`), keeping Windows as is.
   - Helper and .NET functions documented as Windows-only (no behaviour change).
2. `src/gmv/sources/__init__.py` (or a new `sources/registry.py`)
   - `system_audio_source(device_name=None)` dispatching by platform: Windows → today's
     `system_native.system_audio_source`; Linux → a clear "Linux system capture not yet
     implemented" error until Phase 2 replaces it.
   - `available_sources()` returning which of `system`, `device`, `application`, `file`
     the platform offers; `discover()` delegating to it.
3. `src/gmv/app/controller.py` and the drawer
   - Import the factory from the registry instead of `system_native`/`windows` directly.
   - Offer only available sources (on Linux: file now; "All system audio" after Phase 2).
4. `src/gmv/app/window.py: application()`
   - Under Gamescope (`GAMESCOPE_WAYLAND_DISPLAY` set, `WAYLAND_DISPLAY` unset,
     `QT_QPA_PLATFORM` unset), set `QT_QPA_PLATFORM=xcb` before `QApplication`.
5. `packaging/app.py`: `play_tone` skipped off Windows (the self-check reports it skipped).
6. Tests
   - Mark Windows-capture tests (`test_system_native.py`, Windows parts of
     `test_sources.py`, `test_app.py`, `test_recovery.py`) with
     `pytest.mark.skipif(sys.platform != "win32", ...)`.
   - New tests: path selection per platform; source registry per platform; the Gamescope
     platform rule (environment-driven, no display needed); `perf_counter` monotonic.
   - Let `tests/plasma_gpu_probe.py` run on Linux (it already drops `QT_QPA_PLATFORM`).
7. `docs/LINUX_PORT_AUDIT.md`: record what Phase 1a established.

Acceptance: the full suite passes on Windows exactly as before; on CachyOS, `uv sync` and the
suite pass apart from the skipped Windows tests; the app opens, plays a file and replays a
session in Classic, Spectrum bars and Radiant GPU.

Why first: it is low risk, touches no analysis code, makes every later phase testable on
Linux, and proves the renderer half of the Deck question on AMD Linux before the harder audio
work starts.

---

## M. Phase 1a: implementation status (added 2026-11-23)

This section was added after the audit. Sections A to L above are unchanged.

**What phase 1a is.** Section L's task: GMV starts from a checkout on Linux, with audio
files and replay, and Windows is unchanged. No capture backend, no packaging, no change to
the engine, the analysis or the renderers.

**State.** Implemented and tested on Windows, with the Linux behaviour exercised by telling
the platform module which system it is, and since validated on a real CachyOS desktop on
2026-11-23 (section N). When this section was first written no Linux system was available
(no WSL distribution to work in; the container engine on the development machine did not
start; nothing was installed to get one), so M.1 and M.2 describe what the code and its
tests establish, not something observed on Linux. M.3 sets that beside what CachyOS
showed. [LINUX.md](LINUX.md) has the commands that check it on Linux.

### M.1 What was done, against section L

| L | Done | Difference from the proposal |
|---|---|---|
| 1. `paths.py` | Sessions in `$XDG_DATA_HOME/GranularMusicVisualizer/sessions` on Linux; a user models folder beside it, looked in before the checkout's `.model-cache`; the icon as PNG on Linux; the helper functions documented as Windows-only | Sessions leave the repository on Linux for a **checkout** too, not only when packaged: a checkout may be read-only and the working directory may be anything. No `/app/share/gmv/models`: that is packaging (phase 3 below) |
| 2. Source registry | `gmv.sources` (`__init__`): `system_audio_source`, `application_audio_source`, `list_output_devices`, `list_audio_applications`, `require`, `Unavailable` | Built on a new small module, `gmv.host`, which is the only place that asks which system this is: system, XDG folders, audio sources, Qt platform |
| 3. Controller and drawer | `available_sources()`; `Settings.source` defaults to the first of them; the drawer lists only those and hides the device, application and refresh rows without live capture; the note says *"Live system audio capture is not available on Linux yet."* | The command line goes through the registry as well; its `--sessions` default is the data folder on Linux |
| 4. Qt platform | `app.window.start_qt()` applies `host.select_qt_platform()` before the `QApplication` is made | Rule and override below (M.2). A plugin GMV asked for is checked for loadability first; the choice and its reason are printed and shown in Diagnostics |
| 5. `packaging/app.py` | The self-check leaves the capture part out on a system without live capture and says so | |
| 6. Tests | `tests/test_host.py`, `tests/test_platform_app.py` (58 tests) | Windows capture tests were **not** skipped wholesale: they are pure logic and import cleanly. One test that launches the helper (`subprocess.CREATE_NO_WINDOW`) is marked Windows-only. Whether others need it is not known until the suite runs on Linux (since answered on CachyOS: none did; N.3) |
| 7. This document | Section M | |

Also done: `paths.writable_folder` (a sessions folder that cannot be made or written is one
sentence before a session starts); settings that cannot be read are kept beside the ones
that replace them; Diagnostics shows the system, the Qt platform and why, the sessions
folder and the settings file; `scripts/platform_smoke.py` and `scripts/validate_linux.py`;
`uv.lock` brought in line with `pyproject.toml` (the optional `recognition` extra had been
added without relocking, so `uv sync --locked` refused the lock; three packages added, no
pin changed).

Not changed: `engine.py`, `dsp.py`, the analyzers, `models.py`, `session.py`,
`visualizer/` (all three renderers, the Radiant GPU renderer and its shaders), the Windows
capture modules, `Build App.ps1`, the session format.

### M.2 The Gamescope rule, and what was learned about it

Read again on 2026-11-23 (Gamescope `src/main.cpp`, master; Qt `qguiapplication.cpp`, 6.10;
ChimeraOS `gamescope-session-plus`):

- Gamescope sets, in its own process, `GAMESCOPE_WAYLAND_DISPLAY` and `DISPLAY` always;
  removes `WAYLAND_DISPLAY` and puts it back only with `--expose-wayland`; sets
  `XDG_SESSION_TYPE` to `x11` (`wayland` with `--expose-wayland`) and `XDG_CURRENT_DESKTOP`
  to `gamescope`. This confirms E.1 and adds the last two.
- A program Gamescope itself runs (`gamescope -- program`) inherits those. In a session
  script the client may be started by the **script**, not by Gamescope: ChimeraOS's script
  does that, exports `DISPLAY` and `GAMESCOPE_WAYLAND_DISPLAY` from what Gamescope reports,
  and exports `QT_QPA_PLATFORM=xcb` itself. There the client does not inherit Gamescope's
  `XDG_SESSION_TYPE` or `XDG_CURRENT_DESKTOP`. Valve's own session script was not found
  (K.2 stays open).
- Qt 6.10 on Linux, with `QT_QPA_PLATFORM` unset: `DISPLAY` adds `xcb`; `WAYLAND_DISPLAY`
  set or `XDG_SESSION_TYPE=wayland` puts `wayland` first; the result is a list tried in
  order. `-platform` overrides `QT_QPA_PLATFORM`.

The rule implemented: on Linux, when `GAMESCOPE_WAYLAND_DISPLAY` and `DISPLAY` are set and
`WAYLAND_DISPLAY` is not, ask for `xcb`. That is E.2's proposal plus the `DISPLAY`
condition. `GAMESCOPE_WAYLAND_DISPLAY` is the one variable both paths above provide.
`XDG_CURRENT_DESKTOP=gamescope` is deliberately not used: a script-started client may not
have it. With `--expose-wayland` the choice is left to Qt, as whoever passed the flag asked.
Overrides: `GMV_QT_PLATFORM=<plugin>` or `auto`; Qt's own `QT_QPA_PLATFORM` and `-platform`
are never overruled. **Limitation:** not tried in any Gamescope. If Steam's Gaming Mode
does not give its programs `GAMESCOPE_WAYLAND_DISPLAY`, the rule does not fire there and
`GMV_QT_PLATFORM=xcb %command%` in the shortcut's launch options is the answer. (Qt's own
list would also fall through to `xcb` when Wayland is listed first and cannot be reached.)

### M.3 Acceptance, honestly

The middle column is what was established on Windows before any Linux system was available;
the right column is what the CachyOS run of 2026-11-23 showed (section N).

| Criterion | Tested on Windows, the system set to Linux | On CachyOS, 2026-11-23 |
|---|---|---|
| Starts from a Linux checkout | Nothing Windows-only is imported on the way (tested in a process told it is Linux) | Yes: four window runs from an empty working directory, each closed cleanly |
| XDG locations for settings and data | Tested with the variables set, unset, empty, relative. The settings file's place is Qt's doing and is asserted by a Linux-only test that could not run there | Yes: sessions and the settings file where expected (validation, with the XDG folders pointed at a scratch folder); the Linux-only settings-file test passed |
| Windows paths unchanged | Tested; existing tests pass unchanged | Not applicable |
| Linux offers only supported sources; no advertised capture | Tested | Yes: *Audio file* only, with the note; `gmv capture` refused in one sentence |
| File analysis, session logging, replay on Linux | Tested through the real window with the system set to Linux, on Windows' libsndfile and Qt | Yes: WAV and FLAC in the window (198 frames, 7 onsets each), sessions complete without audio, replay after the audio was deleted with the same onset times; WAV, FLAC, OGG, MP3 and a session in the form Windows writes it (CRLF, Windows paths) in the test suite |
| Normal Qt platform selection | The decision is tested for Wayland and X11 sessions (nothing forced) | Yes on KDE Plasma Wayland: Qt chose `wayland` itself; `GMV_QT_PLATFORM=xcb` (through Xwayland) and `GMV_QT_PLATFORM=wayland` were honoured, each with its reason. No native X11 session tried |
| Gaming Mode `xcb` mechanism | Tested as a decision; documented; overridable | Not tried: no Gamescope, nested or in Gaming Mode |
| Windows-only APIs do not block Linux start-up | Tested (no capture module or library loaded) | Yes |
| Renderers untouched | Yes; their tests pass unchanged | Spectrum bars, Radial (Classic) and Radiant on the graphics card were drawn (AMD RX 9070 XT, Mesa 26.3.1); the Radiant graphics-card tests passed |
| Full suite on Linux apart from Windows-only tests (H, phase 1) | Could not run | `main`: 517 passed, 3 skipped, 3 failed. With two test fixes not yet on `main`: 520 passed, 3 skipped (N.3) |
| Windows regressions | Whole suite 519 passed, 4 skipped; real window and command line checked | Not applicable |

### M.4 What remains

The phases are numbered here as they are now planned; the audit's own numbers (section H)
are in brackets.

| Phase | Work | Status (2026-11-23) |
|---|---|---|
| 1a [H: 1] | Native Linux start-up, file analysis, recording, replay, platform foundations | Implemented and validated on CachyOS; two test fixes still to reach `main` (N.3) |
| 1b [H: 2] | Live system-audio capture through PipeWire's PulseAudio server | Implemented and validated on CachyOS (section O); following a change of output between real devices not yet tried |
| 2 [H: 3] | Linux GPU and Qt rendering validation beyond the CachyOS configuration tried | Partly seen on CachyOS (N.1); the rest remains |
| 3 [H: 4] | Flatpak packaging | Not started |
| 4 [H: 5] | SteamOS Gaming Mode integration | Not started; needs a Steam Deck |
| 5 [H: 6] | Handheld refinement | Not started; needs measurements on a Steam Deck |

- **Phase 1a, to close:** the run of `scripts/validate_linux.py --tests` on CachyOS is done
  (section N) and needed no application change. Left: the two test fixes reaching `main`,
  the resource-tracker fix to `tests/test_remote.py` (local and uncommitted) and the replay
  test fix in pull request #1 (open).
- **Phase 1b [H: phase 2]: system-audio capture.** Done; what was built and how it differs
  from C.3 to C.5 is in section O. Left from H's acceptance: a default-sink change, an
  unplug and a suspend on real hardware (tested against a stand-in only), and a PulseAudio
  server.
- **Phase 2 [H: phase 3]: Linux GPU and Qt rendering validation.** Seen on CachyOS
  (section N): Radiant, Classic and Spectrum bars drawn on Mesa (radeonsi) under `wayland`
  and under `xcb` through Xwayland, with the default context (no D.2 surface format was
  needed), and the GPU test probe and Radiant's graphics-card tests passing on Linux. Still
  open: other drivers and graphics cards; a native X11 session; resizing, full screen and
  theme switching while running; composition of the canvas over the OpenGL view (D.3)
  judged by eye; a nested Gamescope.
- **Phase 3 [H: phase 4]: Flatpak.** Manifest on the KDE runtime with the PySide BaseApp;
  shaders as package data; desktop entry, metainfo, icons; the model-redistribution
  constraint (F.7); the application ID (F.6).
- **Phase 4 [H: phase 5]: SteamOS Gaming Mode.** On a Steam Deck: the environment a
  non-Steam shortcut really gets (K.2), focus and the single-window question (the list of
  separate windows is in LINUX.md), `STEAM_GAME` if needed, fullscreen start, background
  capture.
- **Phase 5 [H: phase 6]: handheld refinement.** Touch-sized controls, render scale and
  frame-rate defaults for the Deck's GPU, power, measured on the device.

---

## N. Phase 1a: real CachyOS validation (2026-11-23)

This section was added after the first run on a real Linux system. It summarizes what was
run and measured; the full list of checks is in
[LINUX.md](LINUX.md#on-cachyos-2026-11-23). CachyOS on a desktop is not SteamOS (section I):
nothing here shows how GMV behaves on a Steam Deck, in Gaming Mode or as a Flatpak.

### N.1 What was run

- **Machine:** CachyOS, kernel 7.3.5-2-cachyos, glibc 2.44, x86-64; KDE Plasma on Wayland;
  AMD Radeon RX 9070 XT with Mesa 26.3.1; Python 3.14.10, PySide6 and Qt 6.12.0; a clone of
  `main` at `766a3f7` with its own `.venv` made by `uv`. `perf_counter` there is
  `clock_gettime(CLOCK_MONOTONIC)` and monotonic, as H's phase 1 acceptance asks.
- **Run:** `uv run python scripts/validate_linux.py --tests`, which points the XDG folders at
  a scratch folder and starts the application from an empty working directory.
- **Result: 59 of 60 checks passed.** The failed check was the test suite (N.3). Every check
  of the application passed:
  - the real window opened, ran and closed cleanly four times: as the session gives it and
    again for the remembered settings (Qt chose `wayland` both times), with
    `GMV_QT_PLATFORM=xcb` (through Xwayland) and with `GMV_QT_PLATFORM=wayland`, each time
    saying which platform it used and why;
  - in each run a generated WAV and FLAC were analysed (198 frames and 7 onsets each), their
    sessions recorded complete and without audio in the XDG data folder, and the FLAC's
    session replayed after the file was deleted, with the same onset times;
  - Spectrum bars (first run) and Radial with the Classic theme (the other three) were drawn
    during the analyses, and Radiant on the graphics card in every run;
  - the settings file was in the XDG configuration folder and the settings came back; only
    *Audio file* was offered; `gmv capture` was refused in one sentence; nothing was written
    in the working directory.

### N.2 Measured

One run's figures, for short generated files, on this one machine: not a guarantee for
another machine or for music. They are what Diagnostics showed at the end of each window
run, so they are for the FLAC analysis; CPU and memory are the analysis process's.

| | Range over the four window runs |
|---|---|
| Analysis (DSP) time per 20 ms block, 95th percentile | 2.5–2.7 ms, against a budget of 20 ms |
| Analysis process | CPU 17–18% of one core; memory 122 MiB |
| Losses | None: capture, native, logger, discontinuities and messages between the processes all 0 |
| Radiant, the last frame measured | 1.38–1.39 ms on the graphics card (radeonsi, OpenGL 4.6, GLSL 4.60) |

The window's size was not recorded and the Radiant figure is a single frame, so it is not
comparable with the "about 5 ms per frame on an RX 9070 XT at 1920×1080" measured on Windows
(section A).

### N.3 The test suite, and the two test fixes

- **Validation run:** 517 passed, 3 skipped, 3 failed. The failures were in
  `tests/test_remote.py`. On Linux a `spawn` worker leaves Python's multiprocessing resource
  tracker running for the life of the parent, and the test counted the tracker as a worker
  left behind; the worker had exited, and Windows has no resource tracker. The fix is in the
  test only, and is **local and uncommitted** on branch `fix/linux-resource-tracker-test` in
  the CachyOS checkout.
- **An intermittent failure** in later runs:
  `tests/test_app.py::test_replay_runs_inside_the_same_window_with_original_times`. It was
  test timing, not dropped messages: since `e1d4a3a` the view takes in a bounded share of
  messages per frame, and the test read the markers one frame after a 100× replay had ended.
  The fix, in the tests only, is in
  [pull request #1](https://github.com/Ralkero/granular-audio-visualizer/pull/1), **open and
  not merged**.
- **With both fixes applied:** 520 passed, 3 skipped, 0 failed. That is not `main` as
  published.
- The 3 skips are the test that launches the Windows capture helper and two tests whose
  optional packages (ONNX Runtime, PyTorch) are not installed with `--extra dev`. The three
  Linux-only tests and the Radiant graphics-card tests ran and passed. No other test written
  on Windows needed a Windows-only mark (M.1, item 6).

### N.4 Still not shown

- Steam's Gaming Mode and a nested Gamescope; anything on SteamOS or a Steam Deck. Section K
  stays open.
- A native X11 desktop session: `xcb` ran only through Xwayland.
- Whether KWin honours a chosen screen under Wayland: the windows were on `DP-2`, which is
  also the primary screen.
- A session with the user's own music, watched as it plays.
- Other graphics cards and drivers; resizing, full screen and long sessions.
- Live capture: phase 1b, not started. (Since done: section O.)

---

## O. Phase 1b: system-audio capture (2026-11-23)

This section was added with phase 1b. Linux now captures what the system plays on its
default output, from that output's monitor, through the PulseAudio protocol PipeWire serves
(`pipewire-pulse`). The engine, the analysis, the event attribution, the visual state, the
renderers and the session format are the ones Windows uses; nothing in them changed. CachyOS
on a desktop is not SteamOS (section I): nothing here shows whether a Flatpak may record a
monitor (K.1) or how SteamOS's audio graph behaves.

### O.1 What was built, against C and H's phase 2

| Planned | Built | Why it differs |
|---|---|---|
| `PulseMonitorSource` over libpulse through ctypes, a pull-model record stream with the server's timing info (C.3, C.4) | `PulseSystemAudioSource` (`src/gmv/sources/system_pulse.py`): `parec`, libpulse's own recorder, as a child process, read from its pipe in 10 ms blocks by one thread; `pactl --format=json` for discovery | No binding to write or keep, no Python audio callback, and the server still buffers while Python is busy, which is what C.3 wanted from a pull-model stream. The cost: no `pa_timing_info`, so no server latency is known; the clock is anchored on delivery instead (O.2) |
| The default sink's monitor (C.3) | As planned. The default sink and its monitor are asked for, and the monitor is recorded only once the server's JSON names it the monitor of that sink (`monitor_source` of the source); never a microphone | |
| — | The stream is opened with `node.dont-reconnect`, `node.dont-fallback` and `node.dont-move` (seen applied to the stream's node on CachyOS), and the source it records from is checked every 2 s | A stream whose output goes away is destroyed instead of being moved, for instance to a microphone. PulseAudio itself ignores those properties; the check covers it |
| Formats and channels (C.5) | 32-bit float, little-endian, stereo, at the monitor's own rate; the server mixes other layouts down | The engine accepts any rate, so no resampling is asked for |
| Default-sink change, unplug, suspend producing new segments (H, phase 2) | The default output is asked for every 0.5 s; a change closes the stream and opens one on the new monitor as a new segment. A stream that ends or fails is reopened after 0.5, 1, 2, 4, then at most 5 s, with the reason in the status line. A suspend is seen by `CLOCK_BOOTTIME` moving away from `CLOCK_MONOTONIC` | |
| `host.LIVE_CAPTURE` and `gmv.sources` (M.4) | `LIVE_CAPTURE["linux"] = ("system",)`; `sources.system_audio_source` builds the PulseAudio backend on Linux. The drawer offers *System audio (PipeWire)* and *Audio file*; device and application capture are not offered and are refused in a sentence | |
| — | `scripts/validate_linux.py --audio-capture` and `scripts/capture_probe_linux.py`. The ordinary validation never records (`platform_smoke.py --no-live`); the packaged self-check checks capture on Windows only | Capture records what the user plays, so it is opt-in |

Not changed: `engine.py`, `dsp.py`, the analyzers, `models.py`, `session.py`, the renderers
and their shaders, the Windows capture modules. No Python dependency was added: `pactl` and
`parec` are system programs (on CachyOS and Arch in `libpulse`).

### O.2 The clock, and what it claims

`parec` reports no capture times. The clock counts the stream's samples and anchors the count
on delivery (`DeliveryClock`): a block delivered at `t` whose last frame is stream frame `n`
means frame 0 was captured no later than `t − n / rate`. The first 100 ms of each stream are
measured, not published, and the lowest such bound becomes the anchor, so published times are
not later than delivery by more than 10 ms (well inside half an analysis window, 23 ms, so
evidence is never from the future for `contract.eligible_for_live_impact`). Measured on
CachyOS: PipeWire delivers whole periods of 1024 frames (21.3 ms at 48 kHz), the first data
about 89 ms after `parec` starts, and the bound settles within a few milliseconds over the
first seconds. The lateness of each later block against the anchor is watched as a floor
over 2 s of delivery. A floor more than 20 ms behind (lost audio, a slow output clock) or
10 ms ahead (a fast output clock) moves the anchor to the floor and starts a new run, marked
as a discontinuity; at most a block or two is left out, so the timeline never goes back. A
pause in delivery of more than 0.25 s, or a suspend, starts a new run measured afresh. How
long the sound took from the application to the monitor, or from the monitor to the
speakers, is not known and not claimed.

The output's clock and the computer's are not the same clock. Over ten minutes on CachyOS the
floor fell steadily from +5.9 to −0.2 ms: the Sound BlasterX G6 ran about 11 ppm fast against
`CLOCK_MONOTONIC`. At that rate the clock corrects itself about every 15 minutes, leaving out
a block or two each time and restarting the analysis context there (a discontinuity, as for
any gap). A first version re-measured for 100 ms at each correction and allowed 5 ms ahead;
this measurement is why it was changed. In thirty minutes with the clock as it is now, the
one correction came between the 23rd and 24th minute (the floor had fallen from +5.1 to
−9.3 ms) and left out 20 ms of audio.

### O.3 Tested, without a sound server

`tests/test_system_pulse.py` (28 tests) runs against `tests/fake_pulse.py`, a stand-in for
`pactl` and `parec` that pretends to be a server with two outputs (one with an awkward name,
one at 44.1 kHz) and a microphone, and delivers a known signal in real time. `conftest.py`
points the real client programs at a server that does not exist, so no test records
anything. Covered: the default output and its monitor; every refusal (no server, no default
output, a default that is not listed, no monitor, a monitor not listed, a source that is not
that output's monitor, missing `pactl`, missing `parec`, an old `pactl`); the exact samples,
channel order and gain; contiguous sample-clock times never later than delivery; the
arguments `parec` gets; stopping, six times in a row, with no process or thread left; a stream
ending by itself; the server going away and coming back; the output disappearing before a new
default exists; a new default output followed without a restart and never two streams at
once; a stream moved to the microphone being stopped; a slow reader losing the oldest blocks
while capture goes on; the clock's anchoring, lost audio, a fast output clock, a pause in
delivery and a suspend; and a captured session analysed by the engine, logged without audio
and replayed. The platform tests now expect Linux to offer System audio and refuse device and
application capture in a sentence.

The whole suite on the phase 1b branch, on CachyOS: 546 passed, 3 skipped, 3 failed. The 3
failures are the known `tests/test_remote.py` resource-tracker failures (N.3), whose fix is
not on this branch; on the same code without phase 1b the suite gave 517 passed, 3 skipped,
3 failed. The 29 more that pass are the new tests; no test that passed before fails. A
second run, after the last changes, gave the same plus one failure of the intermittent replay
test of N.3 (5 onsets live, 4 replayed: its known symptom, in code phase 1b does not touch);
its fix is in pull request #1, not on this branch.

### O.4 Validated on CachyOS

The machine of section N; PipeWire 1.6.11 with `pipewire-pulse`, WirePlumber 0.5.19,
`libpulse` 17.0; default output a Sound BlasterX G6 (USB, 24-bit stereo, 48 kHz). What was
playing was captured, quietly; no test signal was played and nothing about the server, the
default output, the routing or the volume was changed. The full figures are in
[LINUX.md](LINUX.md#system-audio-on-cachyos-phase-1b-2026-11-23).

- **Discovery:** the default output and its monitor were found through `pactl`; the monitor
  was confirmed as that output's; the stream's node carried the three `node.dont-*`
  properties and recorded the monitor named.
- **`scripts/validate_linux.py --audio-capture`: 70 of 70 checks passed.** In each of the
  four window runs (Wayland, Xwayland, Wayland forced) the window offered System audio and
  captured the default output for six seconds through the separate analysis process: 203 to
  205 frames. Every phase 1a check passed again.
- **Ten minutes** (`scripts/capture_probe_linux.py --seconds 600`): 29,990 of 30,000 frames,
  one segment, no dropped block, no discontinuity, no correction of the clock, no recovery;
  capture to publication 33 ms median and 37 ms at the 95th percentile; the capturing
  process at 7 to 9% of one core; a clean stop; the logged session (no audio in it) replayed
  in full after capture had stopped. The output's clock ran about 11 ppm fast against
  `CLOCK_MONOTONIC` (O.2). That run used the clock's first version; no correction was due in
  it.
- **Thirty minutes**, with the clock as it is now: 89,988 of 90,000 frames, no dropped block,
  no recovery, exactly one correction (O.2) and so one discontinuity; capture to publication
  28 ms median and 35 ms at the 95th percentile; 8.9% of one core with `parec` included;
  a clean stop and a full replay. Memory stayed at about 128 MiB over a separate three-minute
  capture that kept no records (`parec` 5.7 MiB).

### O.5 Implemented, but not tried on real hardware

Each of these is covered by the stand-in tests (O.3), not by a real device or server:

- following a change of default output between two real devices (the machine has a second
  output, a game controller's, but changing the user's default output was out of bounds);
- an output unplugged while captured, and a monitor that disappears;
- the sound server stopping and coming back;
- a suspend and resume while capturing;
- a stream moved off the monitor (PipeWire refuses to move it; PulseAudio itself would not,
  and the 2 s check covers that case);
- a PulseAudio server (only PipeWire's was tried), and a monitor at a rate other than 48 kHz;
- capture for longer than thirty minutes.

### O.6 Still outstanding

- A session with music the user plays and watches, judged by eye: not done.
- Capture of one output device, of one application or its process tree, and of a microphone:
  not offered on Linux and not planned for phase 1b; per-application capture through native
  PipeWire stays future work (C.6).
- Flatpak packaging and whether a Flatpak may record a monitor (K.1); SteamOS's audio graph,
  Gaming Mode, Gamescope, the Steam Deck's performance and handheld controls: phases 3 to 5.
  Nothing in this section is evidence about any of them.

## P. Phase 1c: native Arch package (2026-11-24)

This section was added with phase 1c. GMV can now be built on CachyOS or Arch Linux as an
ordinary Pacman package and installed beside the system's own Python and Qt: a menu entry
with its icon, the commands in `/usr/bin`, the user's files in the XDG folders, upgrades and
removal through Pacman. This is not phase 4 of H: F's recommendation of a Flatpak for SteamOS
stands, and nothing here is evidence about a Flatpak, its permissions (F.3, K.1), SteamOS or
the Steam Deck. The package is built locally from the recipe in `packaging/arch/`; it is not
on the AUR or in any repository. How to build, install, update and remove it:
[LINUX.md](LINUX.md#the-arch-package-cachyos-arch-linux) and
[packaging/arch/README.md](../packaging/arch/README.md).

### P.1 What was built

| Needed | Built |
|---|---|
| Know that it runs as an installed package (F.5) | `paths.installed()`: neither a PyInstaller build nor a checkout (a checkout has `pyproject.toml` and `src/gmv` two folders above `gmv`). The three ways GMV runs are now named in `gmv.paths` |
| The shaders installed with the code (F.2, F.5) | Radiant's GLSL files are package data in `pyproject.toml` (`[tool.setuptools.package-data]`), so the wheel and the sdist carry all 15. `resources.FOLDER`, beside the module, finds them unchanged |
| No writes beside the program (F.5) | An installed copy has no models folder of its own: models are looked for in `$XDG_DATA_HOME/GranularMusicVisualizer/models/` only, and Settings → **Choose…** opens that folder, made the first time (the home folder if it cannot be made). Sessions, settings and the cache were already in the XDG folders on Linux (M) |
| Menu entry and icon | `packaging/arch/granular-music-visualizer.desktop` (`desktop-file-validate` passes), and the icon at 16 to 512 pixels in the `hicolor` theme. The window sets its desktop file name, so KDE's Wayland taskbar matches it to the entry and its icon; an installed copy takes its window icon from the theme |
| A Pacman package | `packaging/arch/PKGBUILD`: one pinned, pushed commit downloaded from GitHub and checked by SHA-256; `python -m build --wheel --no-isolation` and `python -m installer`, the usual way for a PEP 517 project on Arch; every dependency an Arch package, nothing from PyPI and no `uv`; `arch=('any')`. `check()` installs the wheel in a scratch root and runs that copy away from the source tree: it must take itself for an installed package and read every shader with its includes resolved |

Not changed: the engine, the analysis, capture (Windows and Linux), the renderers and their
shaders, the session format, `uv.lock`, the Windows build (`Build App.ps1`) and how a
checkout or a PyInstaller build finds its models and its icon. No dependency was added to
`pyproject.toml`.

The recipe cannot name the commit that contains it, so it always builds an earlier commit:
the menu entry and icons it installs come from that commit.

### P.2 Automated tests

`tests/test_install_layout.py` (15 tests) builds the installed layout in a scratch folder and
checks: a checkout, an installed copy and a PyInstaller build are told apart; an installed
copy looks for models in the user's folder only, writes nothing beside itself (on Linux and
on Windows), and has no icon file; a checkout is unchanged; every non-Python file under
`src/gmv` is declared as package data; a copy installed outside the source tree, run in its
own interpreter, reads every shader and writes nothing; the three commands are declared; the
desktop entry's fields; the icon sizes; the window's desktop file name and its icon from a
theme; **Choose…** making the models folder and opening there. The `home`, `linux` and
`windows` fixtures moved from `tests/test_host.py` to `tests/conftest.py`, so both files use
them.

The whole suite on the phase 1c branch, on CachyOS, run in a terminal the owner could
watch: **562 passed, 3 skipped, 3 failed** in 3 minutes 7 seconds. The 15 new tests passed.
The 3 failures are the known `tests/test_remote.py` resource-tracker failures (N.3): each
found the same leftover process, Python's multiprocessing resource tracker; their fix is not
on `main`. The 3 skipped are the ones of N.3 (the Windows capture helper, and ONNX Runtime
and PyTorch not installed). The intermittent replay test passed in this run; its fix
(pull request #1) is still not on `main`.

### P.3 Built and installed on CachyOS

The machine of section N (CachyOS, kernel 7.3.5-2-cachyos, Python 3.14.10, Qt and PySide6
6.12.0, Mesa 26.3.1 on an AMD RX 9070 XT, PipeWire 1.6.11), Pacman and `makepkg` 7.1.0.

- **Build:** `makepkg --syncdeps --cleanbuild --log` as the user, of commit `5271e33`
  (`0.2.0.r7.g5271e33-1`). `--syncdeps` installed, with the user's password, what the
  machine lacked from the CachyOS repositories: `pyside6` 6.12.0 (the checkout had used the
  PyPI wheel in its `.venv`), `python-numpy` 2.5.5, `python-scipy` 1.18.1,
  `python-soundfile` 0.14.0 and their dependencies, then `python-build` 1.6.1,
  `python-installer` 1.1.0 and `python-setuptools` 84.2.1. The build, `check()` and
  `package()` logs have no warning. `check()`: "installed copy: 9 fragment shaders read,
  commands import".
- **The package:** 1.6 MB compressed, 5.0 MiB installed, 449 files (Pacman's count, folders
  included), all owned by root. 118 Python modules with their compiled `.pyc` files, the 15
  shaders, the three commands, the menu entry, nine icons and three documents.
- **Install:** `sudo pacman -U`, the user typing the password. `pacman -Qkk` then found 0
  altered files of 449.
- **The installed copy, run:** `scripts/validate_linux.py`, started with `/usr/bin/python -I`
  from an empty folder in `/tmp` so that only the installed `gmv` could be imported (the
  report's model folders confirm it: the user's folder alone, which only an installed copy
  gives). **59 of 59 checks passed**: four window runs (Wayland, Wayland again with the
  remembered settings, Xwayland, Wayland forced), each offering System audio and Audio file,
  analysing a WAV and a FLAC (198 frames, 7 onsets each), recording complete sessions without
  audio in the data folder, replaying with the same times, closing with nothing left
  running; settings remembered between runs; Radiant on the graphics card (AMD RX 9070 XT,
  radeonsi, OpenGL 4.6, Mesa 26.3.1; 1.37 ms on the graphics card, 0.65 ms to submit, for the
  frame measured); nothing written in the working directory; the sound server asked,
  read-only, without an error ("Sound BlasterX G6 Analog Stereo through PulseAudio (on
  PipeWire 1.6.11), 48000 Hz"). The validation points the XDG folders at a scratch folder;
  it did not record from the sound server (no `--audio-capture`).
- **Afterwards:** `pacman -Qkk` still 0 altered; no file under the package's folders that
  Pacman does not own; no GMV or `parec` process left; the user's own GMV folders the same as
  before (23 files compared).

### P.4 Upgraded with Pacman

The recipe was moved to commit `fe6d6dd`, the one that records P.1 to P.3, as its README
says (`_commit`, `pkgver` `0.2.0.r9.gfe6d6dd`, the new SHA-256; GitHub's archive of that
commit was compared with `git archive` of it, file for file, and matched). `makepkg
--syncdeps --cleanbuild --log` built it without a warning and without asking for anything,
every dependency being installed already; `check()` passed again. The two packages hold the
same 449 files, and the same bytes in all but one: the Linux guide in
`/usr/share/doc/granular-music-visualizer/`, the only file of the package that changed
between the two commits. `sudo pacman -U`, the user typing the password, upgraded
`0.2.0.r7.g5271e33-1` to `0.2.0.r9.gfe6d6dd-1`. Afterwards `pacman -Qkk` found 0 altered
files of 449, no file under the package's folders was left that Pacman does not own, and the
user's 23 GMV files (settings and two sessions) had the same SHA-256 as before the upgrade.
The program being byte for byte the one P.3 validated, the validation was not repeated.

After pull requests #4 (the GPL-3.0-or-later licence) and #5 (this phase) were merged, the
recipe was moved again, to `main` at `8520190` (`0.2.0.r15.g8520190`; GitHub's archive
again matched `git archive` file for file). Built without a warning, `check()` passing, and
installed over `0.2.0.r9.gfe6d6dd-1` with `sudo pacman -U`. Against the package before it,
the program's files are the same bytes; what changed is the licence, now in the wheel's
metadata (`License-Expression: GPL-3.0-or-later`) with its text in the `dist-info`
`licenses/` folder (451 files), and the README and Linux guide. `pacman -Qkk` found 0
altered files, nothing was left running, and the user's 34 GMV files (settings and three
sessions) were byte for byte as before.

### P.5 Used by the owner

The upgraded package (`0.2.0.r9.gfe6d6dd-1`) was started through its menu entry
(`kstart --application granular-music-visualizer`, which runs the entry's `Exec` the way
the menu does: systemd started it as "Granular Music Visualizer - Music Visualizer", in a
unit of its own named after the entry) on the owner's primary screen, with the owner's own
settings and data. The owner then used it and reported that everything worked: the menu
entry and its icon, the window's and the taskbar's icon, the visuals following the music,
and Classic, Spectrum Bars and Radiant. Those are the owner's own judgements by eye.

What the run left on the machine, checked afterwards:

- A complete System audio session, 78 s, in `~/.local/share/GranularMusicVisualizer/sessions/`:
  the default output's monitor through PipeWire 1.6.11, one stream, 3,893 analysis frames
  and 1,395 events; no block dropped, no discontinuity, no recovery, no logger loss; no audio
  saved. Music was playing, quietly (median −45 dBFS, 10% of frames below the silence
  threshold). Capture to publication 37 ms median and 39 ms at the 95th percentile; analysis
  2.8 ms per 20 ms block at the 95th percentile; the analysis process at 17% of one core and
  187 MiB.
- The window closed normally (its settings were saved), with no GMV or `parec` process left;
  the whole application peaked at 454 MiB.
- `pacman -Qkk` still 0 altered files; the user's earlier files unchanged.

Two steps of the test left no trace and are not counted as done by the owner. The models
folder had not been made, so **Choose…** beside the weights was not opened; the installed
copy's own buttons were then clicked automatically, offscreen with scratch XDG folders: both
opened the dialog in the user's models folder and made it, writing nothing in the working
directory. And the window was closed two seconds after capture stopped, too soon for a
replay; replaying a session with the installed copy is shown by P.3's validation, not by
the owner.

### P.6 Still outstanding

- **Choose…** for the weights, and replaying a session, done by the owner in the installed
  package (P.5); both are covered by automated checks of the installed copy.
- Installing on Arch Linux itself, or on any machine but this one: not done. The recipe uses
  only Arch's own packages, but only CachyOS's builds of them have been tried.
- `namcap` was not run (it is not installed on the machine, and installing more tools was out
  of bounds for this phase).
- The AUR, a Flatpak (phase 4), packages for other distributions, and signed releases: not
  started.
- Everything outstanding in M.4, N.4 and O.5 still is: the package changes how GMV is
  installed, not what it does.
