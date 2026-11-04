"""Forecast task definitions loaded from configs/tasks/*.json."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

SPLITS = ("train", "valid", "test")
DEFAULT_WIND_BINS = ((0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 40))


@dataclass(frozen=True)
class TaskConfig:
    name: str
    dataset: str
    target: str
    inputs: Tuple[str, ...]
    history_minutes: int
    horizon_minutes: int
    splits: Dict[str, Tuple[str, str]]
    train_mask: str
    eval_mask: str
    valid_mask: Optional[str] = None
    target_input: Optional[str] = None
    train_stride_minutes: int = 60
    eval_stride_minutes: Optional[int] = None
    extra_eval_masks: Tuple[Tuple[str, str], ...] = ()
    report_horizons_minutes: Tuple[int, ...] = ()
    future_inputs: Tuple[str, ...] = ()
    rated_kw: Optional[float] = None
    farm_min_valid_frac: float = 0.85
    wind_variable: Optional[str] = "wind_speed"
    wind_bins: Tuple[Tuple[float, float], ...] = DEFAULT_WIND_BINS
    ramp_threshold_frac: float = 0.10
    stability_block_days: int = 7
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TaskConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"Task {raw.get('name')!r} has unknown fields: {', '.join(unknown)}")
        data = dict(raw)
        data["inputs"] = tuple(data["inputs"])
        data["splits"] = {k: tuple(v) for k, v in data["splits"].items()}
        data["extra_eval_masks"] = tuple((e["mask"], e.get("target")) if isinstance(e, dict) else tuple(e)
                                         for e in data.get("extra_eval_masks", ()))
        for key in ("report_horizons_minutes", "future_inputs"):
            data[key] = tuple(data.get(key, ()))
        data["wind_bins"] = tuple(tuple(b) for b in data.get("wind_bins", DEFAULT_WIND_BINS))
        if data.get("valid_mask") is None:
            data["valid_mask"] = data["train_mask"]
        if data.get("target_input") is None and data["target"] in data["inputs"]:
            data["target_input"] = data["target"]
        return cls(**data)

    @classmethod
    def load(cls, path: Path) -> "TaskConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["extra_eval_masks"] = [{"mask": m, "target": t} for m, t in self.extra_eval_masks]
        return data


def _steps(minutes: int, resolution: int, label: str) -> int:
    if minutes <= 0 or minutes % resolution:
        raise ValueError(f"{label}={minutes} min is not a positive multiple of {resolution} min")
    return minutes // resolution


@dataclass(frozen=True)
class ResolvedTask:
    """A task checked against one dataset's metadata, with lengths in steps."""
    config: TaskConfig
    meta: Dict[str, Any]
    L: int
    H: int
    train_stride: int
    eval_stride: int
    bounds: Dict[str, Tuple[np.datetime64, np.datetime64]]
    rated_kw: np.ndarray
    scorings: Tuple[Tuple[str, str], ...]     # (mask, target); the first is eval_mask

    @property
    def resolution_minutes(self) -> int:
        return int(self.meta["resolution_minutes"])

    @property
    def steps_per_day(self) -> int:
        return 1440 // self.resolution_minutes

    @property
    def report_horizons(self) -> Tuple[int, ...]:
        return tuple(m // self.resolution_minutes for m in self.config.report_horizons_minutes)


def _day(value: str) -> np.datetime64:
    return np.datetime64(value, "us")


def resolve_task(cfg: TaskConfig, meta: Dict[str, Any]) -> ResolvedTask:
    """Validate a task against dataset metadata and convert minutes to grid steps."""
    if meta["name"] != cfg.dataset:
        raise ValueError(f"Task {cfg.name} uses dataset {cfg.dataset}, metadata is {meta['name']}")
    res = int(meta["resolution_minutes"])
    variables, masks = meta["variables"], meta["masks"]
    needed = list(cfg.inputs) + [cfg.target] + ([cfg.target_input] if cfg.target_input else [])
    absent = [v for v in needed if v not in variables]
    if absent:
        raise ValueError(f"Task {cfg.name}: dataset {cfg.dataset} lacks variables {absent}")
    if cfg.target_input is not None and cfg.target_input not in cfg.inputs:
        raise ValueError("target_input must be one of the task inputs")
    for role in ("train_mask", "valid_mask", "eval_mask"):
        name = getattr(cfg, role)
        if name not in masks:
            raise ValueError(f"Task {cfg.name}: unknown {role} {name!r}")
        if masks[name]["target"] != cfg.target:
            raise ValueError(f"{role} {name} scores {masks[name]['target']}, task target is {cfg.target}")
    scorings = [(cfg.eval_mask, cfg.target)]
    for mask, target in cfg.extra_eval_masks:
        if mask not in masks:
            raise ValueError(f"Task {cfg.name}: unknown extra eval mask {mask!r}")
        target = target or masks[mask]["target"]
        if target != masks[mask]["target"] or target not in variables:
            raise ValueError(f"Extra eval mask {mask} must use its registered target {masks[mask]['target']}")
        scorings.append((mask, target))
    if sorted(cfg.splits) != sorted(SPLITS):
        raise ValueError(f"Task {cfg.name} must define exactly the splits {SPLITS}")
    bounds = {k: (_day(a), _day(b)) for k, (a, b) in cfg.splits.items()}
    lo, hi = _day(meta["time_range"][0]), _day(meta["time_range"][1])
    previous = None
    for split in SPLITS:
        a, b = bounds[split]
        if not a < b or a < lo or b > hi:
            raise ValueError(f"Split {split} {cfg.splits[split]} is empty or outside {meta['time_range']}")
        if previous is not None and a < previous:
            raise ValueError("Splits must be chronological and non-overlapping")
        previous = b
    fit = meta.get("fit_window")
    if fit and bounds["valid"][0] < _day(fit[1]):
        raise ValueError(f"Validation and test must start after the cleaning fit window {fit}")
    if cfg.future_inputs:
        raise NotImplementedError("future_inputs are reserved; no dataset provides forecasts yet")
    entity_rated = np.asarray([float(e["rated_kw"]) for e in meta["entities"]], dtype=np.float32)
    rated = np.full_like(entity_rated, cfg.rated_kw) if cfg.rated_kw else entity_rated
    eval_stride = cfg.eval_stride_minutes or res
    for minutes in cfg.report_horizons_minutes:
        _steps(minutes, res, "report_horizons_minutes")
        if minutes > cfg.horizon_minutes:
            raise ValueError(f"Task {cfg.name}: report horizon {minutes} min exceeds horizon_minutes="
                             f"{cfg.horizon_minutes}")
    return ResolvedTask(cfg, meta, _steps(cfg.history_minutes, res, "history_minutes"),
                        _steps(cfg.horizon_minutes, res, "horizon_minutes"),
                        _steps(cfg.train_stride_minutes, res, "train_stride_minutes"),
                        _steps(eval_stride, res, "eval_stride_minutes"), bounds, rated, tuple(scorings))
