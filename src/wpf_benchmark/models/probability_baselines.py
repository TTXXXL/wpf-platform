"""Training-only empirical persistence quantiles with auditable conditional fallback."""
from __future__ import annotations

import numpy as np

from .base import Forecaster, register_model
from ..probability import LEVELS, MEDIAN, postprocess
from .naive import last_finite_window

LEVEL_EDGES = np.asarray([0.02, 0.2, 0.5, 0.8, 0.97])


def history_conditions(x, target_index, rated, steps_per_day):
    power = np.asarray(x[:, -steps_per_day:, :, target_index], dtype=np.float64)
    finite = np.isfinite(power)
    count = finite.sum(axis=1)
    safe = np.where(finite, power, 0)
    mean = safe.sum(axis=1) / np.maximum(count, 1)
    var = (np.where(finite, (safe - mean[:, None]) ** 2, 0).sum(axis=1) / np.maximum(count, 1))
    vol = np.sqrt(var) / rated
    vol[count == 0] = np.nan
    anchor = last_finite_window(np.asarray(x[..., target_index]), fallback=0.0)
    return anchor, vol


@register_model("pers_q")
class PersistenceQuantiles(Forecaster):
    default_params = {"conditioning": "none", "min_samples": 200}
    monotone_quantiles = True

    def __init__(self, **params):
        super().__init__(**params)
        if self._params["conditioning"] not in ("none", "level", "level_volatility"):
            raise ValueError("conditioning must be none, level or level_volatility")
        if int(self._params["min_samples"]) <= 0:
            raise ValueError("min_samples must be positive")

    def validate_context(self, ctx):
        if ctx.target_input_index is None:
            raise ValueError("Persistence quantiles require target_input")

    def fit(self, ctx, train, valid):
        super().fit(ctx, train, valid)
        residuals, levels, vols, masks = [], [], [], []
        chunk = max(1, 2_000_000 // (ctx.L * len(ctx.entities) * len(ctx.input_names)))
        for first in range(0, len(train.issues), chunk):
            batch = train.batch(train.issues[first:first+chunk], with_targets=True)
            anchor, vol = history_conditions(batch.x_past, ctx.target_input_index, ctx.rated_kw, ctx.steps_per_day)
            residuals.append((batch.y - anchor[:, None]).transpose(0, 2, 1).reshape(-1, ctx.H))
            masks.append(batch.y_mask.transpose(0, 2, 1).reshape(-1, ctx.H))
            levels.append(np.digitize(anchor / ctx.rated_kw, LEVEL_EDGES).ravel())
            vols.append(vol.ravel())
        if not residuals:
            raise ValueError("No training windows")
        residual, mask = np.concatenate(residuals), np.concatenate(masks)
        level, vol = np.concatenate(levels), np.concatenate(vols)
        if not np.isfinite(vol).any():
            raise ValueError("No finite training volatility")
        self.volatility_edges = np.quantile(vol[np.isfinite(vol)], [1/3, 2/3], method="linear")
        vbin = np.digitize(vol, self.volatility_edges)
        minimum = int(self._params["min_samples"])
        self.pooled = np.empty((ctx.H, len(LEVELS)))
        self.level = np.empty((6, ctx.H, len(LEVELS)))
        self.joint = np.empty((6, 3, ctx.H, len(LEVELS)))
        self.fallback = []
        level_fallback = np.zeros((6, ctx.H), dtype=bool)
        joint_fallback = np.zeros((6, 3, ctx.H), dtype=bool)
        for h in range(ctx.H):
            good = mask[:, h] & np.isfinite(residual[:, h])
            if not good.any():
                raise ValueError(f"No training residuals for horizon {h+1}")
            self.pooled[h] = np.quantile(residual[good, h], LEVELS, method="linear")
            for l in range(6):
                chosen = good & (level == l)
                enough = chosen.sum() >= minimum
                level_fallback[l, h] = not enough
                self.level[l, h] = np.quantile(residual[chosen, h], LEVELS) if enough else self.pooled[h]
                self.fallback.append({"level": l, "volatility": None, "horizon": h+1,
                                      "n": int(chosen.sum()), "fallback": None if enough else "pers_q"})
                for v in range(3):
                    cell = chosen & (vbin == v) & np.isfinite(vol)
                    enough = cell.sum() >= minimum
                    joint_fallback[l, v, h] = not enough
                    self.joint[l, v, h] = np.quantile(residual[cell, h], LEVELS) if enough else self.level[l, h]
                    self.fallback.append({"level": l, "volatility": v, "horizon": h+1,
                                          "n": int(cell.sum()), "fallback": None if enough else "pers_cq"})
        self.level_fallback, self.joint_fallback = level_fallback, joint_fallback
        self._predicted_cells = self._fallback_cells = 0
        self.probability_metadata = {"volatility_edges": self.volatility_edges.tolist(),
                                     "fallback_cells": self.fallback, "residual_pool": "all training turbines per horizon"}

    def predict_quantiles(self, batch):
        ctx = self.ctx
        anchor, vol = history_conditions(batch.x_past, ctx.target_input_index, ctx.rated_kw, ctx.steps_per_day)
        level = np.digitize(anchor / ctx.rated_kw, LEVEL_EDGES)
        mode = self._params["conditioning"]
        if mode == "none":
            residual = np.broadcast_to(self.pooled, anchor.shape + self.pooled.shape)
        elif mode == "level":
            residual = self.level[level]
        else:
            residual = self.joint[level, np.digitize(vol, self.volatility_edges)]
            residual = np.where(np.isfinite(vol)[..., None, None], residual, self.level[level])
        if mode == "none":
            fallback = np.zeros(anchor.shape + (ctx.H,), dtype=bool)
        elif mode == "level":
            fallback = self.level_fallback[level]
        else:
            fallback = self.joint_fallback[level, np.digitize(vol, self.volatility_edges)] | ~np.isfinite(vol)[..., None]
        self._predicted_cells += int(fallback.size)
        self._fallback_cells += int(fallback.sum())
        self.probability_metadata["prediction_fallback_cells"] = self._fallback_cells
        self.probability_metadata["prediction_total_cells"] = self._predicted_cells
        self.probability_metadata["prediction_fallback_fraction"] = self._fallback_cells / self._predicted_cells
        return (anchor[..., None, None] + residual).transpose(0, 2, 1, 3).astype(np.float32)

    def predict(self, batch):
        return postprocess(self.predict_quantiles(batch), monotone=True)[..., MEDIAN]
