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
