"""Forecaster contract shared by every model and the registry."""
from __future__ import annotations

from typing import Any, Dict, Optional, Type

import numpy as np

from ..tasks import ForecastBatch, SplitData, TaskContext

MODEL_REGISTRY: Dict[str, Type["Forecaster"]] = {}


def register_model(name: str):
    def decorator(cls):
        if not issubclass(cls, Forecaster):
            raise TypeError("Registered models must inherit Forecaster")
        if name in MODEL_REGISTRY:
            raise ValueError("Model already registered: " + name)
        cls.name = name
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str) -> Type["Forecaster"]:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model {name!r}; available: {', '.join(sorted(MODEL_REGISTRY))}") from exc


class Forecaster:
    """Fit on a training split, then forecast [B, H, N] in the target's physical units."""

    name = "base"
    default_params: Dict[str, Any] = {}
    supports_future_inputs = False

    def __init__(self, **params: Any):
        unknown = sorted(set(params) - set(self.default_params))
        if unknown:
            raise ValueError(f"{self.name} got unknown parameters: {', '.join(unknown)}")
        self._params = {**self.default_params, **params}
        self.n_params = 0
        self.validation_metrics: Optional[dict] = None
        self.log_lines: list = []
        self.ctx: Optional[TaskContext] = None

    @property
    def params(self) -> Dict[str, Any]:
        return dict(self._params)

    @property
    def deterministic(self) -> bool:
        return True

    def fit(self, ctx: TaskContext, train: SplitData, valid: SplitData) -> None:
        self.ctx = ctx

    def predict(self, batch: ForecastBatch) -> np.ndarray:
        raise NotImplementedError

    def log(self, line: str) -> None:
        self.log_lines.append(line)
