"\"\"\"Helpers for managing speaker layouts and coordinate transforms.\"\"\""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import numpy as np


DIRECTION_LABELS = {
    "x": ("Left", "Right", "Center"),
    "y": ("Front", "Back", "Center"),
    "z": ("Above", "Below", "Level"),
}


@dataclass
class SpeakerLayout:
    """In-memory representation of the speaker positions in 3D space."""

    positions: List[np.ndarray] = field(default_factory=list)
    active_indices: List[int] = field(default_factory=list)

    def set_positions(self, vectors: Iterable[Sequence[float]]):
        self.positions = [self._vector3(vec) for vec in vectors]
        if not self.active_indices or len(self.active_indices) != len(self.positions):
            self.active_indices = list(range(len(self.positions)))

    def active_speakers(self) -> List[np.ndarray]:
        return [self.positions[idx] for idx in self.active_indices if idx < len(self.positions)]

    def ensure_vector(self, vector: Sequence[float]) -> np.ndarray:
        return self._vector3(vector)

    @staticmethod
    def _vector3(vector: Sequence[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=float)
        if arr.shape[0] == 2:
            arr = np.append(arr, 0.0)
        elif arr.shape[0] != 3:
            raise ValueError("Speaker positions must have 2 or 3 components.")
        return arr


def coordinate_to_distance_direction(value: float, axis: str) -> Tuple[float, str]:
    """Convert a signed coordinate into a magnitude and direction label."""
    distance = abs(value)
    if axis == "x":
        if value < 0:
            return distance, "Left"
        if value > 0:
            return distance, "Right"
        return distance, "Center"
    if axis == "y":
        if value < 0:
            return distance, "Back"
        if value > 0:
            return distance, "Front"
        return distance, "Center"
    if axis == "z":
        if value < 0:
            return distance, "Below"
        if value > 0:
            return distance, "Above"
        return distance, "Level"
    raise ValueError(f"Unsupported axis: {axis}")


def distance_with_direction(distance: float, direction: str, axis: str) -> float:
    """Translate a positive distance and direction label into a signed coordinate."""
    key = direction.lower()
    if axis == "x":
        if key == "left":
            return -distance
        if key == "right":
            return distance
        return 0.0
    if axis == "y":
        if key == "back":
            return -distance
        if key == "front":
            return distance
        return 0.0
    if axis == "z":
        if key == "below":
            return -distance
        if key == "above":
            return distance
        return 0.0
    raise ValueError(f"Unsupported axis: {axis}")

