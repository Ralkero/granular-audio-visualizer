"""Versioned wire contract. Times are seconds on the session's audio timeline."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class AnalysisConfig(FrozenModel):
    window_ms: float = Field(default=46.44, ge=20, le=200)
    hop_ms: float = Field(default=20, ge=5, le=50)
    capture_ms: float = Field(default=10, ge=5, le=50)
    queue_blocks: int = Field(default=16, ge=2, le=128)
    silence_dbfs: float = Field(default=-65, ge=-120, le=-20)
    # Adjacent edges: sub-bass, bass, low mids, mids, upper mids, presence, air.
    band_edges: tuple[float, ...] = (20, 60, 250, 500, 2000, 4000, 6000, 24000)
    advanced: bool = False

    @model_validator(mode="after")
    def check(self):
        if self.hop_ms > self.window_ms:
            raise ValueError("hop_ms must not exceed window_ms")
        if len(self.band_edges) != 8 or self.band_edges[0] < 0:
            raise ValueError("Provide eight nonnegative frequency edges")
        if any(a >= b for a, b in zip(self.band_edges, self.band_edges[1:], strict=False)):
            raise ValueError("Frequency edges must strictly increase")
        return self


class Message(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    session_id: str
    source: str
    timestamp: float = Field(ge=0)
    detected_monotonic: float
    published_monotonic: float = 0
    sequence: int = Field(default=0, ge=0)


class FeatureFrame(Message):
    type: Literal["feature_frame"] = "feature_frame"
    window_end: float
    sample_rate: int
    channels: int
    segment: int = 0
    rms: float
    peak: float
    loudness_dbfs: float
    short_term_loudness_dbfs: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    spectral_rolloff_hz: float
    spectral_contrast_db: tuple[float, ...]
    zero_crossing_rate: float
    band_energy: tuple[float, ...]
    spectral_flux: float
    transient_strength: float
    onset_strength: float
    dynamic_intensity: float
    silence: bool
    tempo_bpm: float | None = None
    tempo_confidence: float = Field(default=0, ge=0, le=1)
    rhythmic_density: float = 0
    queue_latency_ms: float = 0
    dsp_ms: float = 0
    capture_to_publish_ms: float | None = None
    timestamp_quality: str = "sample_clock"


class MusicalEvent(Message):
    type: Literal["musical_event"] = "musical_event"
    event_id: str
    event_type: str
    intensity: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    analysis_tier: Literal["realtime", "advanced"] = "realtime"
    related_event_ids: tuple[str, ...] = ()
    group_id: str | None = None
    method: str
    interval_end: float | None = None


class Status(Message):
    type: Literal["status"] = "status"
    state: str
    detail: str


BAND_NAMES = ("sub_bass", "bass", "low_mids", "mids", "upper_mids", "presence", "air")
