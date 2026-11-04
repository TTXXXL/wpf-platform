"""Rule baselines in physical units: persistence, seasonal persistence, climatology."""
from __future__ import annotations

import numpy as np

from ..tasks import ForecastBatch, SplitData, TaskContext
from .base import Forecaster, register_model


def _require_target_input(ctx: TaskContext, model: str) -> int:
    if ctx.target_input_index is None:
        raise ValueError(f"{model} needs a task target_input among the inputs")
    return ctx.target_input_index


def last_finite_window(series: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    """Latest finite value along axis 1 of [B, L, N], giving [B, N]."""
    finite = np.isfinite(series)
    last = series.shape[1] - 1 - np.argmax(finite[:, ::-1], axis=1)
    value = np.take_along_axis(series, last[:, None, :], axis=1)[:, 0]
    return np.where(finite.any(axis=1), value, np.float32(fallback)).astype(np.float32)


def time_of_day(times: np.ndarray, steps_per_day: int) -> np.ndarray:
    minutes = (times.astype("datetime64[m]") - times.astype("datetime64[D]")).astype(np.int64)
    return (minutes // (1440 // steps_per_day)).astype(np.int64)


@register_model("persistence")
class Persistence(Forecaster):
    """Every horizon repeats the latest finite target_input value (0 kW if none)."""

    def fit(self, ctx, train, valid):
        super().fit(ctx, train, valid)
        self.j = _require_target_input(ctx, self.name)

    def predict(self, batch: ForecastBatch) -> np.ndarray:
        last = last_finite_window(batch.x_past[:, :, :, self.j])
        return np.repeat(last[:, None, :], self.ctx.H, axis=1)


@register_model("seasonal_persistence")
class SeasonalPersistence(Forecaster):
    """Value at the same time one day before each target; persistence where it is missing."""

    def fit(self, ctx, train, valid):
        super().fit(ctx, train, valid)
        self.j = _require_target_input(ctx, self.name)
        if ctx.L < ctx.steps_per_day:
            raise ValueError("seasonal_persistence needs at least one day of history")
        self.offsets = np.arange(1, ctx.H + 1) - ctx.steps_per_day + ctx.L - 1
        if (self.offsets < 0).any() or (self.offsets >= ctx.L).any():
            raise ValueError("seasonal_persistence needs history >= 1 day and horizon <= 1 day")

    def predict(self, batch: ForecastBatch) -> np.ndarray:
        series = batch.x_past[:, :, :, self.j]
        out = series[:, self.offsets, :].astype(np.float32, copy=True)
        fallback = np.broadcast_to(last_finite_window(series)[:, None, :], out.shape)
        return np.where(np.isfinite(out), out, fallback).astype(np.float32)


@register_model("climatology")
class Climatology(Forecaster):
    """Training mean of the target for each turbine and time of day of the target time."""

    def fit(self, ctx, train: SplitData, valid):
        super().fit(ctx, train, valid)
        day = ctx.steps_per_day
        phase = time_of_day(train.times, day)
        good = train.target_mask & np.isfinite(train.target)
        if not good.any():
            raise ValueError("No valid training targets for climatology")
        total = np.zeros((day, len(ctx.entities)), np.float64)
        count = np.zeros_like(total)
        np.add.at(total, phase, np.where(good, train.target, 0.0))
        np.add.at(count, phase, good)
        overall = float(train.target[good].mean(dtype=np.float64))
        self.table = np.where(count > 0, total / np.maximum(count, 1), overall).astype(np.float32)
        self.n_params = int(self.table.size)

    def predict(self, batch: ForecastBatch) -> np.ndarray:
        step = np.timedelta64(self.ctx.resolution_minutes, "m").astype("timedelta64[us]")
        targets = batch.issue_time[:, None] + step * np.arange(1, self.ctx.H + 1)[None, :]
        phase = time_of_day(targets.ravel(), self.ctx.steps_per_day).reshape(targets.shape)
        return self.table[phase]
