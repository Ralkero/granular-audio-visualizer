from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Event


@dataclass(frozen=True, slots=True)
class AudioBlock:
    data: bytes  # interleaved little-endian float32; owned immutable buffer
    frames: int
    sample_rate: int
    channels: int
    timestamp: float  # first sample on session audio timeline
    capture_monotonic: float  # estimated first sample monotonic time
    received_monotonic: float
    segment: int = 0
    discontinuity: bool = False
    timestamp_quality: str = "sample_clock"


class AudioSource(ABC):
    kind = "unknown"
    live = True

    @abstractmethod
    def run(self, output, stop: Event, status):
        """Own resources until stopped; emit blocks without DSP; close output in finally."""

    def describe(self):
        return {"kind": self.kind}
