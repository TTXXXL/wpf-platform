"""Task definitions, chronological windows and training-only scaling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import ResolvedTask, TaskConfig, resolve_task
from .scaling import Scaler
from .windows import ForecastBatch, SplitData, build_split, last_finite, panel_columns


@dataclass(frozen=True)
class TaskContext:
    """What a model may know about a task when it is fitted."""
    task: ResolvedTask
    input_names: List[str]
    target: str
    target_input: Optional[str]
    L: int
    H: int
    entities: List[str]
    rated_kw: np.ndarray
    scaler: Scaler

    @property
    def target_input_index(self) -> Optional[int]:
        return None if self.target_input is None else self.input_names.index(self.target_input)

    @property
    def resolution_minutes(self) -> int:
        return self.task.resolution_minutes

    @property
    def steps_per_day(self) -> int:
        return self.task.steps_per_day

    def input_index(self, name: str) -> int:
        try:
            return self.input_names.index(name)
        except ValueError as exc:
            raise KeyError(f"Task input {name!r} is not available; inputs: {self.input_names}") from exc


__all__ = ["ForecastBatch", "ResolvedTask", "Scaler", "SplitData", "TaskConfig", "TaskContext",
           "build_split", "last_finite", "panel_columns", "resolve_task"]
