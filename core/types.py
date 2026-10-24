"""MediaPipe-independent data types used by every mode."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Hand:
    landmarks: tuple[Landmark, ...]
    gesture: str = "None"
    score: float = 0.0
    handedness: str = "Unknown"


@dataclass(frozen=True)
class Observation:
    hands: tuple[Hand, ...] = ()

    @property
    def primary(self) -> Hand | None:
        return self.hands[0] if self.hands else None
