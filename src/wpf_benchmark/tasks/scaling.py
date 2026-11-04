"""Training-only min/max scales, the same rule as the legacy SDWPF code."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .windows import SplitData


@dataclass(frozen=True)
class Scaler:
    minimum: Dict[str, float]
    maximum: Dict[str, float]
    target_name: str               # variable whose scale the target uses

    @classmethod
    def fit(cls, train: SplitData, target_input: Optional[str]) -> "Scaler":
        lo, hi = {}, {}
        for j, name in enumerate(train.input_names):
            column = train.inputs[:, :, j]
            if not np.isfinite(column).any():
                raise ValueError(f"Training data has no finite value of {name}")
            lo[name], hi[name] = float(np.nanmin(column)), float(np.nanmax(column))
        if target_input is None:
            lo["__target__"] = float(np.nanmin(train.target))
            hi["__target__"] = float(np.nanmax(train.target))
        return cls(lo, hi, target_input or "__target__")

    def span(self, name: str) -> np.float32:
        return np.float32(max(np.float32(self.maximum[name]) - np.float32(self.minimum[name]), 1e-6))

    def scale(self, x: np.ndarray, name: str) -> np.ndarray:
        return ((x - np.float32(self.minimum[name])) / self.span(name)).astype(np.float32)

    def unscale(self, x: np.ndarray, name: str) -> np.ndarray:
        return (x * self.span(name) + np.float32(self.minimum[name])).astype(np.float32)

    def scale_inputs(self, x: np.ndarray, names: List[str]) -> np.ndarray:
        """Scale [..., F] inputs, keeping NaN."""
        lo = np.asarray([self.minimum[n] for n in names], dtype=np.float32)
        span = np.asarray([self.span(n) for n in names], dtype=np.float32)
        return ((x - lo) / span).astype(np.float32)

    def scale_target(self, y: np.ndarray) -> np.ndarray:
        return self.scale(y, self.target_name)

    def unscale_target(self, y: np.ndarray) -> np.ndarray:
        return self.unscale(y, self.target_name)

    def to_dict(self) -> dict:
        return {"minimum": self.minimum, "maximum": self.maximum, "target_name": self.target_name}

    @classmethod
    def from_dict(cls, data: dict) -> "Scaler":
        return cls(dict(data["minimum"]), dict(data["maximum"]), data["target_name"])
