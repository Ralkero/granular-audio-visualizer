# Dependency decisions (2026-11-03)

| Dependency | Decision | Installation / licensing considerations |
|---|---|---|
| PyAudioWPatch | Windows shared-mode system loopback | Windows binary wheels; MIT PyAudio/PortAudio lineage. Small project, isolate behind adapter. Does not provide process filtering. |
| NumPy / SciPy | FFT, numerical features, optional background HPSS | CPU wheels; BSD licenses. No CUDA required. |
| SoundFile / libsndfile | Bounded file decoding | BSD wrapper / LGPL library; format support depends on bundled libsndfile. No automatic conversion. |
| Pydantic | Versioned validated messages/configuration | MIT; serialize on log thread. |
| PySide6 Essentials | Desktop controls and diagnostic meters | QtCore/Gui/Widgets only; LGPL/GPL/commercial options. Preserve licensing notices and relinking rights in future distribution. |
| psutil / pycaw | Resource metrics / read-only Windows endpoint/session discovery | BSD / MIT; pycaw depends on comtypes. Never call setters for volume or routing. |
| NAudio.Wasapi 3.3.0 / .NET 9 | Optional native API bridge for application capture | MIT; pinned NuGet lock. Small C# console helper, bounded 16-packet transfer queue, binary stdout. .NET runtime required only for this source. |
| mir_eval 0.8.2 | Evaluation only (`evaluation`/`dev` extras): one-to-one event matching and beat continuity | MIT; pinned; never imported by capture or the live pipeline. |
| Audacity | External annotation editor (not a dependency) | GPL application used by the annotator; the project only reads and writes its label text files. |
| Librosa | Considered, not a Tier A dependency | ISC, useful offline algorithms; extra runtime/JIT dependencies unnecessary for basic features. |
| Demucs | Investigate after immediate pipeline validation | MIT code; original Meta repository archived, successor limited maintenance. Torch/model downloads and CPU cost unsuitable as a mandatory dependency. Check each weight license before shipping. |
| Essentia | Not selected | AGPL/commercial and model-specific licensing; Windows installation and redistribution need separate evaluation. |
| FFmpeg | Optional future unsupported-format decoder | Build-dependent LGPL/GPL licensing; avoid bundling until required. |

Windows process capture is a separate backend, not a PyAudioWPatch feature. Microsoft's official sample uses ActivateAudioInterfaceAsync, a process-tree inclusion mode, and Windows build >=20348. It is endpoint-independent. Ordinary Windows 11 satisfies the OS requirement, but activation, child-process identity and protected playback still need runtime tests. Unavailable application capture must explain why and offer an explicit system-capture selection; never silently substitute a broader source.

Primary references:
- https://learn.microsoft.com/en-us/samples/microsoft/windows-classic-samples/applicationloopbackaudio-sample/
- https://learn.microsoft.com/en-us/windows/win32/coreaudio/loopback-recording
- https://github.com/s0d3s/PyAudioWPatch
- https://github.com/AndreMiras/pycaw
- https://python-soundfile.readthedocs.io/
- https://doc.qt.io/qtforpython-6/licenses.html
- https://github.com/facebookresearch/demucs
- https://essentia.upf.edu/licensing_information.html
- https://naudio.github.io/NAudio/docs/WasapiRecorder.html
- https://docs.pipewire.org/page_man_pw-cat_1.html

NAudio documentation gives a lower OS threshold than Microsoft's sample. The helper conservatively requires build 20348 and still handles activation failure at runtime. Tested here on build 26300. The original Microsoft sample writes WAV output; this helper instead streams bounded packets with QPC timestamps to Python and writes no audio file.

PipeWire investigation: `pw-record`/`pw-cat` support raw stdout, target nodes and configurable latency. A future source should select a sink monitor using explicit stream properties and expose PipeWire graph timestamps; a subprocess arrival clock would be a less precise fallback. This is an architectural plan, not a shipped Linux backend.

Advanced alternatives: median spectral HPSS is computationally inexpensive but cannot distinguish vocal/bass/drum identities. It was selected as the delayed CPU prototype. Demucs provides the desired four broad stems, but chunk buffering, model latency, Torch compatibility and maintenance warrant an isolated benchmark environment. Essentia and Librosa offer richer rhythm/harmonic algorithms; neither is required for the immediate pipeline. AMD acceleration remains uninvestigated; no GPU is used.

The lock file records the actual resolved Windows environment. OS adapters are optional on other platforms. This is a source checkout, not a signed packaged installer.

## Linux (port phase 1a)

No dependency was added, removed or made platform-specific for Linux. `uv.lock` is a universal lock: it already holds the Linux wheels (x86-64 and aarch64) of every dependency.

| Dependency | On Linux | Notes |
|---|---|---|
| PyAudioWPatch, pycaw (and comtypes through it) | Not installed, not imported | `sys_platform == 'win32'` markers in `pyproject.toml`, as before. The application reaches capture only through `gmv.sources`, which loads these modules on Windows alone. |
| .NET capture helper (NAudio) | Not needed, not looked for | Windows only. File analysis and replay never used it. |
| PySide6 Essentials | Wheel `manylinux_2_34` (glibc 2.34+) | Brings Qt and its platform plugins (`xcb`, `wayland`, `offscreen`). The plugins use system libraries: for `xcb`, `xcb-util-cursor` and the other `xcb-util-*` libraries and `libxkbcommon-x11`; for `wayland`, `wayland` and `libxkbcommon`; OpenGL through the system's `libGL`/`libEGL`. None is a Python dependency; [LINUX.md](LINUX.md) lists the packages. |
| NumPy, SciPy, SoundFile, Pydantic, psutil | Linux wheels | The SoundFile wheel carries its own libsndfile, so file formats do not depend on the system's. |
| ONNX Runtime (optional `recognition` extra) | Linux wheels for Python 3.11 to 3.14 | |
| PipeWire (`pipewire-pulse`) or PulseAudio, with `pactl` and `parec` | System programs, not Python dependencies | System audio (phase 1b) runs `pactl` and `parec` as subprocesses (`gmv.sources.system_pulse`); no Python package was added for it. On CachyOS and Arch both programs are in `libpulse`. Needed only for System audio. |

`uv.lock` was brought in line with `pyproject.toml` in this phase: the optional `recognition` extra (`onnxruntime` with `flatbuffers` and `protobuf`) had been added to `pyproject.toml` without relocking, so `uv sync --locked`, which `Setup.ps1` runs and Linux setup uses, would refuse the lock. `uv lock` added those three packages and changed no existing pin. The supported Python versions are unchanged (3.11 and newer).

## Arch package (port phase 1c)

The Arch package (`packaging/arch/PKGBUILD`) bundles no dependency and no Python
environment: the wheel is installed into the system's Python, and every dependency is an
Arch package that Pacman installs and updates. `uv.lock` plays no part in it; the versions
are whatever the Arch repositories have, which must satisfy the ranges in `pyproject.toml`.

| Python project dependency (`pyproject.toml`) | Arch runtime package | Notes |
|---|---|---|
| Python 3.11 or newer | `python` | The system's Python (3.14 on CachyOS, November 2026). |
| `PySide6-Essentials` 6.8 to 6 | `pyside6` | Arch's build against the system Qt 6 (`qt6-base`); has `QtOpenGL` and `QtOpenGLWidgets`, which Radiant uses. Not the PyPI wheel. |
| `numpy` 2.2 to 2 | `python-numpy` | |
| `scipy` 1.15 to 1 | `python-scipy` | |
| `soundfile` 0.13 to 0 | `python-soundfile` | Uses the system's `libsndfile` (the PyPI wheel brings its own). |
| `pydantic` 2.10 to 2 | `python-pydantic` | |
| `psutil` 7 | `python-psutil` | |
| `PyAudioWPatch`, `pycaw` | none | Windows only (`sys_platform == 'win32'` markers). |
| (not Python) `pactl`, `parec` for System audio | `libpulse` | Client programs only; not a sound server. |
| (not Python) the icon theme the icons go in | `hicolor-icon-theme` | |

| Optional | Arch package | Notes |
|---|---|---|
| A PulseAudio-compatible sound server, for System audio | `pipewire-pulse` (CachyOS's default) or `pulseaudio` | Listed as optional and never pulled in: the package does not install, replace or configure a sound server. Without one, files and replay still work. |
| `onnxruntime` (`recognition` extra) | `python-onnxruntime-cpu` | For context recognition models only. The CPU build is the one named; the CUDA and ROCm builds would also work but are not needed. No weights are packaged. |

| Build only (`makedepends`) | Notes |
|---|---|
| `python-build`, `python-installer`, `python-setuptools` | The Arch way for a PEP 517 project: `python -m build --wheel --no-isolation`, then `python -m installer --destdir="$pkgdir"`. setuptools 70.1 and newer make wheels without `python-wheel`, so it is not listed. No pip, no `uv`, nothing downloaded from PyPI. |

## Context recognition (optional)

| Dependency | Decision | Notes |
|---|---|---|
| ONNX Runtime (CPU) | Optional extra `recognition`; needed only when a recognition model is chosen | MIT. Wheels for Python 3.11 to 3.14 on Windows. The DirectML provider (AMD GPUs) is not used: the model takes 13 ms per estimate on the CPU. |
| EfficientAT `mn10_as` weights | Base of the experimental guitar model; local only | Code MIT (github.com/fschmid56/EfficientAT). The weight file has no separate licence statement; trained on AudioSet. |
| torch, torchaudio, torchvision, matplotlib | Experiments environment only (training, export, plots) | CPU builds. Never imported by the application. |
| PaSST (`hear21passt`) | Installed in the experiments environment to measure cost; not used | Apache-2.0. 2.5 s of one core per 10 s clip. |

## Evidence enrichment (`gmv evidence`)

No new dependency in the application environment: MusicBrainz and AcoustID are reached with the standard library, or with the system `curl` when Python has no TLS. External tools, all local files outside Git: Chromaprint `fpcalc` 1.6.2, FFmpeg, MusicBrainz Picard 3.0.5 portable (available, not used by any command). Experiments environment only: `panns-inference` 0.1.1 with `torchlibrosa` (PANNs CNN14, CPU), `demucs` 4.1.0. Licences: [evidence/LICENCES_AND_PROVENANCE.md](evidence/LICENCES_AND_PROVENANCE.md).

## Packaged build

PyInstaller (GPL-2.0 with an exception that allows bundling any program; 6.22) is a build tool only, installed by `Build App.ps1` when missing and not a runtime dependency. The build is a folder, not a single file, so start-up does not unpack anything. `Build App.ps1` also places Microsoft's .NET 9 runtime (MIT, pinned version and SHA-512) beside the capture helper; the helper uses it through `DOTNET_ROOT`, so capture does not depend on the machine's .NET installation. .NET 9 is out of support; moving the helper to .NET 10 is pending.

## Excerpt review (`gmv review`)

No new dependencies. Decoding uses soundfile (libsndfile with mpg123, reads MP3 and MPEG Layer II); measurements use numpy and scipy; the window uses PySide6 Essentials; playback uses PyAudioWPatch (PortAudio), already used for capture. Qt Multimedia was not added: it is in the PySide6 Addons wheel (about 200 MB). Model timelines need the optional `recognition` extra (ONNX Runtime, CPU).
