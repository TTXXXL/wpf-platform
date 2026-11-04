"""A complete regular time grid of turbine variables held as [T, N] arrays."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class Panel:
    meta: Dict[str, Any]
    times: np.ndarray                      # datetime64[us], [T], interval starts
    entities: List[str]                    # [N]
    values: Dict[str, np.ndarray] = field(default_factory=dict)   # float32 [T, N]
    masks: Dict[str, np.ndarray] = field(default_factory=dict)    # bool [T, N]
    fit_arrays: Dict[str, np.ndarray] = field(default_factory=dict)  # optional, saved as fit.npz

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype="datetime64[us]")
        shape = (len(self.times), len(self.entities))
        for kind, store in (("variable", self.values), ("mask", self.masks)):
            for name, array in store.items():
                if array.shape != shape:
                    raise ValueError(f"{kind} {name} has shape {array.shape}, expected {shape}")
        if len(self.times) > 1:
            step = np.diff(self.times)
            if not (step == step[0]).all() or step[0] <= np.timedelta64(0, "us"):
                raise ValueError("Panel times must form a strictly increasing regular grid")

    @property
    def resolution(self) -> np.timedelta64:
        return np.timedelta64(int(self.meta["resolution_minutes"]), "m").astype("timedelta64[us]")

    @property
    def rated_kw(self) -> np.ndarray:
        return np.asarray([float(e["rated_kw"]) for e in self.meta["entities"]], dtype=np.float32)

    def index_of(self, time) -> int:
        """Row index of an exact grid time, or of the first row at or after it."""
        return int(np.searchsorted(self.times, np.datetime64(time, "us"), side="left"))

    def variable(self, name: str) -> np.ndarray:
        if name not in self.values:
            raise KeyError(f"Variable {name!r} is not loaded; available: {sorted(self.values)}")
        return self.values[name]

    def mask(self, name: str) -> np.ndarray:
        if name not in self.masks:
            raise KeyError(f"Mask {name!r} is not loaded; available: {sorted(self.masks)}")
        return self.masks[name]
