"""Chronological splits, issue times and model input batches."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..data.panel import Panel
from .config import ResolvedTask


@dataclass
class ForecastBatch:
    x_past: np.ndarray                      # float32 [B, L, N, F], physical units, NaN = missing
    issue_time: np.ndarray                  # datetime64[us] [B]
    entities: List[str]
    x_future: Optional[np.ndarray] = None   # float32 [B, H, N, K], reserved for NWP
    y: Optional[np.ndarray] = None          # float32 [B, H, N]
    y_mask: Optional[np.ndarray] = None     # bool [B, H, N]

    @property
    def past_observed(self) -> np.ndarray:
        return np.isfinite(self.x_past)


@dataclass
class SplitData:
    name: str
    times: np.ndarray                       # datetime64[us] [T'], includes borrowed test history
    entities: List[str]
    input_names: List[str]
    inputs: np.ndarray                      # float32 [T', N, F]
    target: np.ndarray                      # float32 [T', N]
    target_mask: np.ndarray                 # bool [T', N]
    issues: np.ndarray                      # int64 [I], positions in times
    L: int
    H: int
    scoring: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)  # test only
    aux: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def issue_times(self) -> np.ndarray:
        return self.times[self.issues]

    def _future(self, positions: np.ndarray) -> np.ndarray:
        return positions[:, None] + np.arange(1, self.H + 1)[None, :]

    def windows(self, positions: np.ndarray):
        """Return x [B,L,N,F], y [B,H,N] and the training mask [B,H,N] for issue positions."""
        positions = np.asarray(positions, dtype=np.int64)
        past = positions[:, None] + np.arange(-self.L + 1, 1)[None, :]
        future = self._future(positions)
        y = self.target[future]
        return self.inputs[past], y, self.target_mask[future] & np.isfinite(y)

    def batch(self, positions: np.ndarray, with_targets: bool = False) -> ForecastBatch:
        positions = np.asarray(positions, dtype=np.int64)
        x, y, mask = self.windows(positions)
        return ForecastBatch(x, self.times[positions], self.entities,
                             y=y if with_targets else None, y_mask=mask if with_targets else None)

    def future_values(self, values: np.ndarray, positions: Optional[np.ndarray] = None) -> np.ndarray:
        """Gather a [T', N] array at target times, giving [I, H, N]."""
        positions = self.issues if positions is None else positions
        return values[self._future(np.asarray(positions, dtype=np.int64))]


def last_finite(values: np.ndarray, positions: np.ndarray, window: int, fallback: float = 0.0) -> np.ndarray:
    """Latest finite value of a [T', N] series within each history window, [I, N].

    This is the persistence forecast. A window with no finite value gives ``fallback``.
    """
    positions = np.asarray(positions, dtype=np.int64)
    out = values[positions].astype(np.float32, copy=True)
    for lag in range(1, window):
        missing = ~np.isfinite(out)
        if not missing.any():
            break
        earlier = values[positions - lag]
        out[missing] = earlier[missing]
    out[~np.isfinite(out)] = fallback
    return out


def panel_columns(task: ResolvedTask) -> tuple:
    """Variables and masks a run needs from the dataset store."""
    cfg = task.config
    variables = list(cfg.inputs) + [cfg.target] + [t for _, t in task.scorings]
    if cfg.wind_variable and cfg.wind_variable in task.meta["variables"]:
        variables.append(cfg.wind_variable)
    masks = [cfg.train_mask, cfg.valid_mask] + [m for m, _ in task.scorings]
    return list(dict.fromkeys(variables)), list(dict.fromkeys(masks))


def _issue_positions(first: int, last_target: int, H: int, stride: int) -> np.ndarray:
    return np.arange(first, last_target - H + 1, stride, dtype=np.int64)


def build_split(task: ResolvedTask, panel: Panel, split: str) -> SplitData:
    """Slice one split from a panel that covers it (and, for test, L-1 rows before it)."""
    cfg = task.config
    start, end = task.bounds[split]
    a, b = panel.index_of(start), panel.index_of(end)
    if a >= len(panel.times) or panel.times[a] != start:
        raise ValueError(f"Split {split} start {start} is not on the dataset time grid")
    if split == "test":
        lo, first, stride = a - (task.L - 1), task.L - 1, task.eval_stride
        if lo < 0:
            raise ValueError("Not enough history before the test split")
    else:
        lo, first, stride = a, task.L - 1, task.train_stride
    issues = _issue_positions(first, b - 1 - lo, task.H, stride)
    if not len(issues):
        raise ValueError(f"Split {split} is too short for L={task.L}, H={task.H}")
    rows = slice(lo, b)
    mask_name = {"train": cfg.train_mask, "valid": cfg.valid_mask, "test": cfg.eval_mask}[split]
    data = SplitData(split, panel.times[rows], panel.entities, list(cfg.inputs),
                     np.stack([panel.variable(v)[rows] for v in cfg.inputs], axis=-1),
                     panel.variable(cfg.target)[rows], panel.mask(mask_name)[rows], issues,
                     task.L, task.H)
    if split == "test":
        data.scoring = {m: {"target": panel.variable(t)[rows], "mask": panel.mask(m)[rows]}
                        for m, t in task.scorings}
        if cfg.wind_variable and cfg.wind_variable in panel.values:
            data.aux["wind"] = panel.variable(cfg.wind_variable)[rows]
    return data
