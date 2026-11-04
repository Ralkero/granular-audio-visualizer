"""Original deterministic synthetic material, released CC0-1.0. No recorded audio."""

from dataclasses import dataclass, field
import hashlib
import numpy as np
from scipy.signal import butter, sosfilt

CORPUS_VERSION = "synthetic-v1"
CLASSES = (
    "bass_transient_candidate",
    "snare_like_candidate",
    "high_frequency_percussion_candidate",
)


@dataclass
class Fixture:
    name: str
    category: str
    split: str
    audio: np.ndarray
    rate: int
    attacks: list[tuple[float, str]]
    tempo_intervals: list[tuple[float, float, float]] = field(default_factory=list)
    beats: list[float] = field(default_factory=list)
    stems: dict[str, np.ndarray] = field(default_factory=dict)
    import_metadata: dict | None = None

    @property
    def onsets(self):
        return sorted({round(t, 6) for t, _ in self.attacks})

    def manifest(self):
        result = {
            "id": self.name,
            "category": self.category,
            "split": self.split,
            "rate": self.rate,
            "channels": self.audio.shape[1],
            "duration": len(self.audio) / self.rate,
            "sha256_float32_le": hashlib.sha256(self.audio.astype("<f4").tobytes()).hexdigest(),
            "mixed_onsets": self.onsets,
            "class_events": self.attacks,
            "tempo_intervals": self.tempo_intervals,
            "beats": self.beats,
            "provenance": "Original procedural waveforms; CC0-1.0; no third-party samples",
            "annotation": "Exact synthesis start samples; mixed coincident attacks deduplicated",
            "uncertainty_seconds": 1 / self.rate,
            "corpus_version": CORPUS_VERSION,
        }
        if self.import_metadata:
            result.update(self.import_metadata)
        return result


def generate(split="development", rate=48000):
    holdout = split == "holdout"
    rng = np.random.default_rng(91073 if holdout else 1279)
    duration = 16
    categories = (
        "isolated_bass",
        "snare_noise",
        "high_percussion",
        "simultaneous",
        "dense",
        "irregular_fills",
        "syncopation",
        "tempo_change",
        "quiet_loud",
        "sustain_negative",
        "silence_negative",
        "opposite_phase",
        "embedded_eight",
    )
    for category in categories:
        total = 24 if category == "tempo_change" else duration
        stems = {
            name: np.zeros((rate * total, 2), dtype=np.float32)
            for name in ("drums", "bass", "vocals", "other")
        }
        attacks, tempos, beats = [], [], []

        def add(
            time, kind, amplitude=0.3, sustained=False, total=total, stems=stems, attacks=attacks
        ):
            position = round(time * rate)
            length = round((0.6 if sustained else 0.055) * rate)
            length = min(length, rate * total - position)
            t = np.arange(length) / rate
            if kind == "bass":
                x = np.sin(2 * np.pi * (73 if holdout else 91) * t) * np.exp(
                    -t / (0.3 if sustained else 0.018)
                )
                stem, cls = "bass", CLASSES[0]
            elif kind == "snare":
                x = sosfilt(
                    butter(3, [650, 3200], btype="bandpass", fs=rate, output="sos"),
                    rng.normal(size=length),
                )
                x *= np.exp(-t / 0.012)
                stem, cls = "drums", CLASSES[1]
            elif kind == "high":
                x = sosfilt(
                    butter(3, 6500, btype="highpass", fs=rate, output="sos"),
                    rng.normal(size=length),
                )
                x *= np.exp(-t / 0.009)
                stem, cls = "drums", CLASSES[2]
            else:
                x = sum(np.sin(2 * np.pi * f * t) for f in (220, 330, 440)) / 3
                x *= np.minimum(1, t / 0.01) * np.minimum(1, (t[-1] - t) / 0.03)
                stem, cls = "other", "unknown"
            x *= amplitude / max(np.max(np.abs(x)), 1e-9)
            stems[stem][position : position + length, 0] += x
            stems[stem][position : position + length, 1] += x * (0.8 if holdout else 1)
            attacks.append((position / rate, cls))

        if category in ("isolated_bass", "snare_noise", "high_percussion"):
            kind = {"isolated_bass": "bass", "snare_noise": "snare", "high_percussion": "high"}[
                category
            ]
            for time in np.arange(0.75 if holdout else 0.5, 15, 0.71 if holdout else 0.5):
                add(time, kind, sustained=category == "isolated_bass" and int(time * 2) % 3 == 0)
        elif category == "simultaneous":
            for time in np.arange(0.63 if holdout else 0.5, 15, 0.43 if holdout else 0.5):
                for kind in ("bass", "snare", "high"):
                    add(time, kind)
        elif category == "dense":
            step = 60 / (196 if holdout else 180) / 4
            for index, time in enumerate(np.arange(0.5, 15, step)):
                if holdout and index % 17 in (8, 9):
                    continue
                add(time, ("bass", "snare", "high")[index % 3])
            beats = list(np.arange(0.5, 15, step * 4))
            tempos = [(0.5, 15, 196 if holdout else 180)]
        elif category == "irregular_fills":
            time, index = 0.5, 0
            pattern = [0.07, 0.11, 0.055, 0.24, 0.08] if holdout else [0.08, 0.06, 0.14, 0.33]
            while time < 15:
                add(time, ("high", "bass", "snare")[index % 3])
                time += pattern[index % len(pattern)]
                index += 1
        elif category == "syncopation":
            period = 60 / (132 if holdout else 120)
            beats = list(np.arange(0.5, 15, period))
            for i, time in enumerate(beats):
                if i % (5 if holdout else 4) != 2:
                    add(time, "bass")
                if i % 2:
                    add(time + period * (0.75 if holdout else 0.5), "high")
            tempos = [(0.5, 15, 60 / period)]
        elif category == "tempo_change":
            for begin, end, bpm in [
                (0.5, 12, 100 if holdout else 120),
                (12.5, 23, 150 if holdout else 180),
            ]:
                beat = list(np.arange(begin, end, 60 / bpm))
                for time in beat:
                    add(time, "bass")
                beats.extend(beat)
                tempos.append((begin, end, bpm))
        elif category in ("quiet_loud", "opposite_phase", "embedded_eight"):
            for i, time in enumerate(np.arange(0.5, 15, 0.46 if holdout else 0.5)):
                add(time, ("bass", "snare", "high")[i % 3], 0.015 if time < 8 else 0.5)
        elif category == "sustain_negative":
            # One annotated harmonic entrance; no repeated attacks in sustained chord.
            time = np.arange(rate * 14) / rate
            chord = (
                sum(
                    np.sin(2 * np.pi * f * time)
                    for f in ((196, 294, 392) if holdout else (220, 330, 440))
                )
                * 0.08
            )
            envelope = np.minimum(1, time / 0.02) * np.minimum(1, (time[-1] - time) / 0.05)
            stems["other"][rate : rate * 15] = (chord * envelope)[:, None]
            attacks.append((1, "unknown"))
        audio = sum(stems.values())
        if category == "opposite_phase":
            audio[:, 1] = -audio[:, 0]
        if category == "embedded_eight":
            audio = np.pad(audio, ((0, 0), (0, 6)))
        yield Fixture(
            f"{split}_{category}",
            category,
            split,
            audio.astype(np.float32),
            rate,
            attacks,
            tempos,
            beats,
            stems,
        )
