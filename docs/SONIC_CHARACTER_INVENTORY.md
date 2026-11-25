# Feature inventory for Sonic Character

What GMV already measures, read from the code at `main` (cdc75e8) before Sonic Character was
designed, and what each measurement is worth for describing the character of a recording.
Nothing in this list was added or changed by Phase A. The profile built on it is described
in [SONIC_CHARACTER.md](SONIC_CHARACTER.md).

## Where the measurements come from

One analysis pass (`gmv.dsp.Analyzer.process`) runs on every window of audio: a 46.4 ms Hann
window (2048 samples at 44.1 kHz) every 20 ms, so 50 frames a second. It publishes one
`FeatureFrame` per window and, when it finds them, `MusicalEvent`s. The channels are
averaged in power before anything is measured. All times are audio time: the centre of the
window on the session's timeline.

"Level-free" below means the value does not change when the audio is made louder or
quieter. It was checked on real recordings by analysing each one a second time 10 or 20 dB
down (results in SONIC_CHARACTER.md).

## On every feature frame

| Field | What it is | Unit and range | Level-free | Every | Worth for character | Weaknesses | Used by |
|---|---|---|---|---|---|---|---|
| `rms`, `peak` | RMS and largest sample of the window | linear, 0..1 | no | frame | Their ratio (crest factor, dB) is level-free: low when the waveform is saturated or a plain tone, high for isolated hits | Over 46 ms only; a pure tone also has a low crest factor (3 dB) | session log; Sonic Character (ratio only) |
| `loudness_dbfs` | `20·log10(rms)` | dBFS, -120..0 | no | frame | Only differences are usable: how far the level moves inside a recording | Not perceptual loudness (no frequency weighting); depends on playback volume | `MusicalIntensity`, level meter, Sonic Character (differences only) |
| `short_term_loudness_dbfs` | Mean power of the last 3 s, in dB | dBFS | no | frame, 3 s memory | Redundant with a running average of `loudness_dbfs`; not used | The same as above | session log |
| `dynamic_intensity` | 65% absolute level (-65..0 dBFS) + 35% level against a 20 s reference | 0..1 | no | frame | Not usable: mostly absolute level | Changes with the playback volume | `intensity_change` events, timeline history |
| `silence` | Level under `silence_dbfs` (-65) for 0.25 s | flag | no (absolute threshold) | frame | Decides what counts as music at all | A quiet capture loses its quiet passages to it | everything |
| `spectral_centroid_hz` | Power-weighted mean frequency | Hz, 0..Nyquist | yes | frame | The standard brightness measure; on real music 110 Hz to 1.1 kHz as a recording average | Power-weighted, so one strong bass note pulls it far down; follows the arrangement as much as the timbre | `spectral_change` events, transient labels, Sonic Character |
| `spectral_bandwidth_hz` | Power-weighted spread round the centroid | Hz | yes | frame | Moves with the centroid; adds little | As the centroid | session log |
| `spectral_rolloff_hz` | Frequency under which 85% of the power lies | Hz | yes | frame | The same information as the centroid (r = 0.99 across recordings); not used | Quantised to FFT bins; capped by a lossy file's cut-off | session log |
| `spectral_contrast_db` | Per band: 90th over 10th percentile of the bin powers | dB per band (7) | yes | frame | High when a band holds separate partials (tonal), low when it is filled evenly (noise-like, distorted, dense). The best harmonic-versus-noisy proxy available | The band above 6 kHz reads 40 dB and more on lossy files with a cut-off, whatever the music; that band has to be left out. The lowest bands hold only 2 to 9 bins | session log; Sonic Character (six bands under 6 kHz) |
| `zero_crossing_rate` | Share of adjacent sample pairs that change sign | 0..1 | yes | frame | Follows brightness (r = 0.87 with the centroid across recordings); not used | Adds nothing over the centroid here | session log |
| `band_energy` | Mean-square energy in seven bands: under 60, 60–250, 250–500, 500–2k, 2–4k, 4–6k, over 6 kHz | linear power | no; their shares are | frame | Shares of the total describe the balance: low-end weight, body, upper mids, air | The edges are a setting (`band_edges`); a recording made with other edges cannot be compared | `MusicalIntensity` (share under 250 Hz), `Dominance`, onset detection, Sonic Character (shares) |
| `spectrum_db` | 48 log-spaced bands, 30 Hz to 18 kHz, in dB | dB per band | no; its shape is | frame | The richest description of the spectrum's shape: slope (dB/octave), how much of it is filled, how peaky it is | Absent from sessions recorded before it existed and when `spectrum_bands` is 0; the lowest bands are narrower than an FFT bin and are interpolated; rounded to 0.1 dB | every spectrum display, `Dominance`, Sonic Character |
| `spectral_flux` | Share of the magnitude spectrum that is newly risen since the frame before | 0..1 | yes | frame | How much the sound is renewed from frame to frame: high for drums, distortion and dense playing, low for held notes | Depends on the hop (20 ms); a steady noise also has flux (its bins fluctuate) | onset detection, tempo, Sonic Character |
| `band_flux` | The same per band | 0..1 per band | yes | frame | Not needed beyond `spectral_flux` for a whole-mix character | Noisy in the narrow low bands | onset detection |
| `transient_strength`, `attack_energy_ratio` | Share of the frame's RMS that is a rise since the frame before (both fields hold the same number) | 0..1 | yes | frame | How percussive the level is: large for isolated hits, near zero in a wall of sound | Heavy-tailed; zero most of the time | session log; Sonic Character |
| `onset_strength` | `spectral_flux` held to 0..1, zero in silence | 0..1 | yes | frame | The same as flux | – | timeline history, tempo |
| `rhythmic_density` | Onsets in the last four seconds, divided by four | onsets/s, 0 to about 6 in practice | nearly (onsets are judged against the recent flux and level; only the silence threshold is absolute) | frame, 4 s memory | How busy the rhythm is. Not the tempo: a fast tempo with few articulated attacks reads low | Counts whole-mix onsets: in a dense distorted mix many attacks are masked and the count saturates near 3 to 4 a second; steps in quarters | `MusicalIntensity` (10%), Sonic Character |
| `tempo_bpm`, `tempo_confidence` | Strongest autocorrelation of the last 12 s of flux between 60 and 200 BPM, and its strength; refreshed once a second, tempo withheld under 0.2 | BPM; 0..1 | yes | 1 s, 12 s memory | The strength is a usable "is there a steady pulse" measure. The tempo figure itself is not reliable enough to build on | Octave errors (half or double time) are common; no preference between them; needs 4 s before it says anything | status display; Sonic Character (strength; the figure only as context) |
| `beat_locked`, `beat_phase` | A causal beat tracker's lock and phase | flag; 0..1 | yes | frame | Would give beat consistency | Off by default (`experimental_beat_phase`); not in most recorded sessions; not used | beat markers when turned on |
| `instrument_scores`, `instruments_active` | A frame-wise activity model's per-instrument judgement | scores; names | – | frame | Would say which instruments are present | Only with a weights file that is not shipped (derived from non-commercial data); empty otherwise; not used | instrument lanes |
| `pitch_candidates` | Multi-pitch note candidates | MIDI note, score | yes | frame | Not a character measure | Optional; guitar register only | note lane |
| `queue_latency_ms`, `dsp_ms`, `capture_to_publish_ms` | Timing of the analysis itself | ms | – | frame | None | – | diagnostics |

## Events

| Event | What it is | Worth for character | Weaknesses | Used by |
|---|---|---|---|---|
| `onset` | A new attack: flux above an adaptive threshold (median + 3 MAD of the last second), a band rising, at least 45 ms after the last unless two bands apart | Their spacing: the time from one to the next, and how even that is | Whole-mix onsets mix every instrument's attacks: even tightly played music has a wide spread of intervals (0.7 to 1.3 octaves standard deviation on real recordings). Timing is quantised to the 20 ms hop | timeline, `rhythmic_density`, Sonic Character (optional) |
| `bass_transient_candidate`, `snare_like_candidate`, `high_frequency_percussion_candidate` | Region labels on an onset from the band shares | None beyond the band shares themselves | Labels of a spectral region, not of an instrument | timeline |
| `energy_rise`, `intensity_change`, `spectral_change` | Frame-to-frame jumps of level, `dynamic_intensity` and centroid | These are Musical Expression (what happens now), not identity | Level-dependent thresholds | timeline |
| `silence_started`, `audio_resumed` | The silence flag changing | Marks where music stops and starts | – | state |
| `beat_prediction`, `beat_observed` | From the optional beat tracker | As `beat_locked` | Off by default | beat rings |
| `harmonic_percussive_balance` | A delayed median-filter separation: share of power that is harmonic versus percussive, over two seconds | The most direct harmonic-versus-percussive measure GMV has | Off by default (`advanced`), arrives late, costs a second FFT pass in its own thread; absent from ordinary sessions. Not used in Phase A; the first thing to try if roughness needs strengthening | context markers |

## Per-instrument activity

`InstrumentActivity` (guitar, drums, bass, vocals analyzers in `gmv.instruments`) carries a
level, flux, attacks, a repetition rate and its regularity, per instrument. `InstrumentEstimate`
is a recognition model's judgement of presence. `VisualState.attribution` joins the two, and
`Dominance` scores which instrument is most prominent.

They are not used for Sonic Character, deliberately:

- measured on multitrack songs (docs/INSTRUMENT_LAYERS.md), the fast analyzers are not
  specific to their instrument on their own: with the instrument removed from the mix they
  still report much of the same activity. They describe energy in a region, which the band
  shares already do;
- presence needs model weights that the installed application does not ship;
- which analyzers run, and how sensitive they are, is a setting the owner changes. A
  character that changed with the settings of the instrument layers would not be a property
  of the recording.

## What already uses these, and how Sonic Character differs

`MusicalIntensity` (`gmv.visualizer.intensity`, verified in the code, unchanged): 65% the
level of the last 0.4 s placed between a quiet and a loud reference that both adapt to the
recording being heard, 25% the share of energy under 250 Hz, 10% onsets a second; eased over
0.7 s rising and 2.5 s falling. Because its references adapt, it says how intense this
moment is *for this recording*. It deliberately does not say what kind of recording it is:
on 67 real recordings its average was between 0.49 and 0.84 for every one of them, a quiet
piano piece and a metalcore track alike (figures in SONIC_CHARACTER.md).

## Gaps

No new measurement was added to the analysis pass. Three things GMV does not measure were
considered and left for later, because Phase A could be built without them:

- **Perceptual loudness.** `loudness_dbfs` has no frequency weighting. Sonic Character uses
  only differences of level, where the weighting matters least.
- **Harmonic versus noise, measured directly.** `spectral_contrast_db` is a proxy. The
  optional delayed separation above measures it directly but is off by default.
- **Beat consistency.** The default tempo estimate has octave errors and the beat tracker is
  opt-in, so "how steady is the beat" rests on the strength of the periodicity and on the
  spread of onset intervals, both weak in dense mixes (see the limitations in
  SONIC_CHARACTER.md).
