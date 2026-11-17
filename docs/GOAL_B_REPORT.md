# Goal B report: one application, note tracking, honest scoring

Asked for: guitar tracking, the visualizer as the single application window, and fixes for the main Phase 2A issues. This report says what was built, what was measured, and what is still missing. The largest gap is stated first.

## What is not done

- **There is no usable guitar activity detector.** A first one was trained on MoisesDB after this report was first written (see the update at the end) and scored on held-out artists: it is too unreliable to show under the instrument's name, so it is not loaded by default and the guitar lane still says "no detector output".
- **Clean versus distorted guitar** is not available for the same reason.
- **No person has watched the application with sound.** Part 5 of the request (listening tests on varied music) was not carried out: it needs a listener. Everything below is measurement or inspection of saved frames. A checklist for that session is at the end.
- **Guitar inside a full mix is unmeasured.** Pitch tracking was scored on solo acoustic guitar only.

## 1. The unified application

One window (`gmv.app.window.MainWindow`). The visualizer fills it; a top bar has the status, **Start/Stop** and **Settings**. Settings is a drawer that slides over the right 440 pixels and never stops analysis. The separate control-center window and its launcher are removed; `Launch Visualizer.cmd`, `gmv visualizer`, `gmv ui` and `gmv-control` all open this window, and `gmv visualizer --replay SESSION` opens it replaying.

Drawer sections: Audio source, Analysis, Visualizer, Sessions, Diagnostics, Advanced (see the README). Behaviour:

- Opens idle with "Press Start"; nothing is captured until Start.
- Display settings apply at once. Source and analysis settings changed while running offer **Apply changes**, which restarts the analysis; nothing needs the application restarted.
- Start-up problems (missing file, missing weights, no application selected) appear in the top bar and leave the application usable.
- Replay runs in the same window and is labelled as replay.
- Closing the window stops the engine, the replay and the timers (tested: engine threads have exited, the subscription is released).
- Choices are remembered between runs (not process ids, not advanced settings).

Headless tools (`gmv capture`, `gmv file`, `gmv replay`, evaluation) are unchanged.

Not exercised here: specific-device and per-application capture through the new window (they use the same source code as before), and a device disappearing during capture.

## 2 and 3. Guitar: what exists

| Capability | State |
|---|---|
| Guitar activity (is a guitar playing) | Trained and measured, not good enough: held-out AUC 0.85, misses 28% of guitar time. Experimental weights only, not loaded by default |
| Clean / distorted / acoustic | Trained and measured, not good enough: each fires 18% to 25% of the time in songs without that kind of guitar |
| Note candidates, several at once | Implemented (`gmv.pitch`), opt-in, on by default in the application |
| Attack versus sustain | Implemented per note: a note is reported after it persists three frames, its attack is back-dated to when it began, and a later attack is reported only when the note's own harmonics get clearly louder again |
| Register | E2 to E6, shown as a piano-roll lane |
| Confidence | An uncalibrated score per note, drawn as brightness |
| Attribution to guitar | Only while a guitar detector reports guitar; today never |

How note tracking works: every 20 ms the last 85 ms are transformed, spectral peaks are kept relative to their neighbourhood, each semitone is scored by the peaks on its harmonic series, the best note is taken and its harmonics are attenuated before the next, up to four notes. A candidate an octave, a twelfth or two octaves above another accepted note is dropped unless it is clearly stronger than the lower note explains. Cost: about 0.2 ms per frame.

Known limits of the method: at most four notes; chord notes are often missed; a real octave above a played note is usually suppressed with the overtone rule; bends and slides appear as the nearest semitone; distorted chords create real spectral peaks that are not played notes; on a full mix it reports bass lines, vocals and keyboards in that register as readily as guitar.

## 4. Datasets and splits

| Dataset | Use | Split |
|---|---|---|
| GuitarSet (downloaded with approval, CC BY 4.0, 0.66 GB) | Pitch and note tracking reference | Players 00–03 development, 04–05 held out |
| MDB Drums | Corrected attack recall and beat scoring | Development only (all one artist group) |
| MUSDB18-HQ | Unchanged; no guitar stem | — |
| MoisesDB (obtained by the owner, CC BY-NC-SA 4.0) | Guitar activity training and scoring | By artist: 147 train, 13 validation, 48 held-out songs; 32 songs left out |

No model output was used as reference anywhere.

## 5. Results

### Pitch tracking on GuitarSet (real solo acoustic guitar)

| Held-out players | Frame precision | Frame recall | Frame F1 | Note onset F1 (±50 ms) |
|---|---:|---:|---:|---:|
| Single-note playing (30.4 min) | 0.759 | 0.836 | **0.795** | **0.771** |
| Chords (30.4 min) | 0.782 | 0.422 | **0.548** | 0.419 |

Notes are also reported in 7% to 13% of frames where the reference has none; part of that is strings still ringing after the annotated end, and it was not separated out. Full table and caveats: [evaluation/guitarset_pitch_v1.md](evaluation/guitarset_pitch_v1.md). The held-out players were scored twice (the method changed between, prompted by synthetic tests).

### False-positive behaviour

- On solo guitar: about one reported note in four is wrong on single notes (octave and neighbouring errors, ringing).
- On a full mix: not measured. In the one saved frame inspected, a dense rock mix produced many short scattered notes; a solo guitar file produced a clear line. The lane is labelled as not attributed to guitar and can be hidden.

### Corrected scoring on MDB Drums (23 development tracks)

Attack recall of the full-mix onset detector, one detected onset per attack, hits within 20 ms counted once: kick 0.816, cymbals 0.696, hi-hat 0.662, toms 0.644, snare 0.638, whole kit **0.632** over 6,125 distinct attacks. The earlier whole-kit figure of 0.69 let one onset answer several attacks.

Beat tracker, per recording: locked on the beat in 20, **on the off-beat in 3**, tempo within 4% in 18, half or double time in 3; aggregate F1 unchanged at 0.689. It stays experimental. Table: [evaluation/mdbdrums_goalb.md](evaluation/mdbdrums_goalb.md).

## 6. Performance

Real on-screen window, beat tracker and activity model on, development machine. "Before" is the Phase 2A report (file source).

| Window, layout, source | Drawing p50 / p95 | DSP p50 / p95 / max (budget 20 ms) | FPS | Drops | CPU (one core) | Memory |
|---|---:|---:|---:|---:|---:|---:|
| 1440×900 bars, file, before | 6.0 / 8.4 ms | 2.4 / 10.0 / 18.2 | 61.3 | 0 | 64% | 238 MiB |
| 1440×900 bars, file, now | 3.3 / 4.4 ms | 5.2 / 6.5 / 10.5 | 61.3 | 0 | 50% | 220 MiB |
| 2560×1440 bars, file, before | 10.6 / 13.3 ms | 5.5 / 14.1 / 18.4 | 61.2 | 0 | 127% | 276 MiB |
| 2560×1440 bars, file, now | 5.6 / 7.1 ms | 8.1 / 9.5 / 15.7 | 61.2 | 0 | 78% | 272 MiB |
| 1440×900 bars, live capture, now | 2.5 / 3.7 ms | 2.7 / 6.6 / 11.4 | 57.4 | 0 | 54% | 224 MiB |
| 2560×1440 bars, live capture, now | 4.4 / 5.7 ms | 3.0 / 8.6 / 14.0 | 60.1 | 0 | 74% | 277 MiB |
| 2560×1440 radial, live capture, now | 8.2 / 12.1 ms | 4.9 / 13.3 / **22.6** | 59.4 | 0 | 92% | 267 MiB |
| No window, file | — | 1.6 / 2.1 / 2.7 | — | 0 | 11% | 172 MiB |

- The "now" rows include note tracking, which the "before" rows did not have.
- Drawing time roughly halved and DSP p95 fell by about a third. DSP median with a file source is higher than before; a file source wakes on the same Windows timer tick as the display, so analysis frames and repaints start together nearly every time. Live capture is driven by the audio device and does not show this.
- The live 1440×900 run averaged 57.4 FPS over 12 s including start-up; the other runs held 59 to 61.
- **Remaining bottleneck:** the radial layout at full screen. Its worst DSP frame (22.6 ms) exceeded the 20 ms hop once in 12 s; no audio was dropped, but there is no headroom. Bars at full screen leave about 11 ms at p95.
- GPU drawing was not pursued: the earlier trial could not be verified. Painting still shares the interpreter lock with DSP; moving the renderer or the DSP to its own process is the structural fix and is not done.

## 7. Phase 2A issues

| Issue | State |
|---|---|
| Drawing slows analysis | Improved (table above); radial at full screen still tight |
| Two windows | Fixed: one application |
| One onset credited for several attacks | Fixed; recall recomputed |
| Uncertain beats scored as certain | Fixed: estimates at uncertain beats are set aside; off-beat lock, tempo, phase and tracked share reported |
| Tempo coverage across gaps | Fixed: gaps, untracked and confident time reported, accuracy over the whole span |
| Region labels accepted in point layers | Fixed: rejected with a clear message |
| Not judged by a person | Open |
| Only default output and file sources run | Open |
| Replay tested on a short synthetic session | Open (now also exercised inside the unified window, same kind of session) |
| Sound-to-screen latency unmeasured | Open |
| Display scaling above 100%, second monitor | Open |

Timing rules kept: marks and notes are placed by audio timestamp; note candidates carry their own earlier time; delayed analysis keeps its dotted style; predictions stay amber; gaps are shown as holes; a new session or restarted timeline clears state; replay keeps original times (tested in the unified window).

## 8. Checks

| Check | Result |
|---|---|
| `pytest -q` | 157 passed (106 before Phase 2A, 125 after) |
| `ruff check .` | clean |
| Schema files | regenerated, match the models |
| Synthetic onset regression | unchanged: development 565 found / 0 false / 0 missed; holdout 577 / 0 / 5 |
| `dotnet build native/ProcessLoopback -c Release` | 0 warnings, 0 errors (helper not changed in this work) |

## 9. Manual setup

- **Drums / bass / vocals weights**: `.model-cache/activity_v2.npz` on this machine is picked up automatically; elsewhere choose a file in Settings or train one with `scripts/train_activity.py`.
- **Guitar weights**: `.model-cache/guitar_experimental_v1.npz` on this machine, experimental and not loaded by default. It can be chosen under Settings, Analysis, Guitar to look at its behaviour; the commands that produced it are in [evaluation/moisesdb_guitar_v1.md](evaluation/moisesdb_guitar_v1.md).
- **GuitarSet** for re-running the pitch evaluation: Zenodo record 3371780, `annotation.zip` and `audio_mono-mic.zip`.

## Listening checklist (for the first session with sound)

Play each kind of material and note whether the picture matches what is heard: distorted metal riffs, clean guitar, palm mutes, held chords, fast picking, bass-heavy passages, dense drums, clean and harsh vocals, sparse passages, silence.

1. Do onset marks and bar flashes land on audible attacks, or visibly late?
2. Does the notes lane follow a guitar melody when one is exposed? Does it fill with unrelated notes when vocals or bass carry the tune?
3. Do the drums, bass and vocals lanes switch off when the instrument stops?
4. Does the beat pulse sit on the beat, or between beats?
5. Is the default smoothing sluggish or jittery?
6. Is Settings easy to reach and use while music plays?

## Update, 2026-11-17: guitar detector trained on MoisesDB

The owner obtained MoisesDB. Running the prepared path on the real files needed four fixes: a held-out artist split (the dataset ships none), rendered mixes that counted some files twice, remixes for instruments spread over several files, and leaving out time where the only active guitar file is flagged as containing bleed.

Result on 48 held-out songs by 10 artists the model never saw ([full table](evaluation/moisesdb_guitar_v1.md)):

| Label | Precision | Recall | False-positive rate | AUC | Lane lit in songs without it |
|---|---:|---:|---:|---:|---:|
| guitar | 0.960 | 0.719 | 0.167 | 0.854 | 37% (1 song only) |
| distorted | 0.648 | 0.773 | 0.241 | 0.837 | 25% |
| clean | 0.644 | 0.590 | 0.216 | 0.758 | 24% |
| acoustic | 0.511 | 0.740 | 0.195 | 0.834 | 18% |

This does not meet the bar for a lane named after an instrument, so nothing changes in what the application shows by default. The model fits songs it trained on far better than new artists, and changing its size did not help: the features, built for drums, bass and vocals, do not tell a guitar from other instruments in the same register. A better guitar detector needs different features (finer spectral detail over a longer window, or a small learned front end), which is a separate piece of work.

Not done: nobody has watched the experimental weights in the application with sound; 32 songs with unequal file lengths are unused; no metal-specific test.
