"""Per-turbine multi-output ridge regression on min/max-scaled history windows.

With the default inputs (target_input only) this is the legacy SDWPF ``linear`` model.
"""
from __future__ import annotations

import numpy as np

from ..tasks import ForecastBatch, SplitData, TaskContext
from .base import Forecaster, register_model


@register_model("ridge")
class Ridge(Forecaster):
    default_params = {"alpha": 1.0, "inputs": None}

    def fit(self, ctx: TaskContext, train: SplitData, valid: SplitData) -> None:
        super().fit(ctx, train, valid)
        alpha = float(self._params["alpha"])
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        names = self._params["inputs"] or [ctx.target_input]
        if None in names:
            raise ValueError("ridge needs a task target_input or explicit params.inputs")
        self.sel = [ctx.input_index(n) for n in names]
        self.names = [ctx.input_names[j] for j in self.sel]
        L, H, S = ctx.L, ctx.H, len(self.sel)
        issues = train.issues
        past = issues[:, None] + np.arange(-L + 1, 1)[None, :]
        future = issues[:, None] + np.arange(1, H + 1)[None, :]
        scaled = ctx.scaler.scale_inputs(train.inputs[:, :, self.sel], self.names)   # [T, N, S]
        target = ctx.scaler.scale_target(train.target)
        self.weights = np.empty((len(ctx.entities), L * S + 1, H), np.float32)
        self.fallback = []
        penalty = np.eye(L * S + 1, dtype=np.float64) * alpha
        penalty[-1, -1] = 0
        last = (L - 1) * S + (self.names.index(ctx.target_input) if ctx.target_input in self.names else 0)
        for j in range(len(ctx.entities)):
            y = target[future, j]
            good = train.target_mask[future, j].all(axis=1) & np.isfinite(y).all(axis=1)
            if not good.any():
                self.weights[j] = 0
                self.weights[j, last, :] = 1      # scaled persistence when no window is valid
                self.fallback.append(ctx.entities[j])
                continue
            x = np.nan_to_num(scaled[:, j][past[good]].reshape(int(good.sum()), L * S)).astype(np.float64)
            x = np.column_stack((x, np.ones(len(x))))
            yy = y[good].astype(np.float64)
            self.weights[j] = np.linalg.solve(x.T @ x + penalty, x.T @ yy).astype(np.float32)
        self.n_params = int((len(ctx.entities) - len(self.fallback)) * (L * S + 1) * H)
        self.log(f"ridge alpha={alpha} inputs={self.names} windows={len(issues)} fallback={self.fallback}")

    def predict(self, batch: ForecastBatch) -> np.ndarray:
        ctx = self.ctx
        x = ctx.scaler.scale_inputs(batch.x_past[:, :, :, self.sel], self.names)        # [B, L, N, S]
        B, L, N, S = x.shape
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).transpose(0, 2, 1, 3).reshape(B, N, L * S)
        out = np.einsum("bnk,nkh->bnh", x, self.weights[:, :-1, :]) + self.weights[None, :, -1, :]
        out = ctx.scaler.unscale_target(out)
        return np.clip(out, 0, ctx.rated_kw[None, :, None]).transpose(0, 2, 1).astype(np.float32)
