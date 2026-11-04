"""LightGBM direct multi-step forecaster, ported from the legacy SDWPF ``gbdt`` model.

One regressor per forecast step, shared by all turbines, trained on kW targets. Features are
computed in physical units with missing values set to 0, as in the legacy code: for every task
input the last, mean, std, min, max and least-squares slope over the history window; the
newest ``lags`` values of the task ``target_input`` (new here, the legacy model had none); the
issue time's phase within the day (0-1); and the turbine index.

The legacy model fell back to xgboost when LightGBM was missing. This one does not.
"""
from __future__ import annotations

import numpy as np

from ..tasks import ForecastBatch, SplitData, TaskContext
from .base import Forecaster, register_model

_STATS = 6                                  # last, mean, std, min, max, slope
_CHUNK_CELLS = 5_000_000                    # float32 cells of raw history per feature chunk


@register_model("lightgbm")
class LightGBM(Forecaster):
    default_params = {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31,
                      "n_jobs": 4, "lags": 6, "objective": "regression", "seed": 0,
                      "deterministic": True, "force_row_wise": True, "verbosity": -1}

    def __init__(self, **params):
        super().__init__(**params)
        p = self._params
        if min(p["n_estimators"], p["max_depth"], p["num_leaves"], p["n_jobs"]) <= 0 or p["learning_rate"] <= 0:
            raise ValueError("n_estimators, max_depth, num_leaves, n_jobs and learning_rate must be positive")
        if p["lags"] < 0:
            raise ValueError("lags must be nonnegative")
        if not isinstance(p["objective"], str):
            raise ValueError("objective must be a string")

    @property
    def deterministic(self) -> bool:
        # The seed comes from params, not from the run seed, so extra run seeds would repeat the same model.
        return True

    # ----- features -----
    def _phase(self, issue_times: np.ndarray) -> np.ndarray:
        """Position of each issue time within the UTC day, in [0, 1)."""
        micros = np.asarray(issue_times).astype("datetime64[us]").astype(np.int64)
        minutes = (micros % 86_400_000_000) // 60_000_000
        return ((minutes // self.ctx.resolution_minutes) / self.ctx.steps_per_day).astype(np.float32)

    def _features(self, x: np.ndarray, issue_times: np.ndarray) -> np.ndarray:
        """Features of [B, L, N, F] physical-unit history, shape [B*N, 6F + lags + 2]."""
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        B, L, N, F = x.shape
        positions = np.arange(L, dtype=np.float32)
        centered = positions - positions.mean()
        slope = np.einsum("blnf,l->bnf", x, centered) / np.sum(centered ** 2)
        parts = [x[:, -1], x.mean(axis=1), x.std(axis=1), x.min(axis=1), x.max(axis=1), slope]
        lags = int(self._params["lags"])
        if lags:
            parts.append(x[:, ::-1][:, :lags, :, self.target_index].transpose(0, 2, 1))     # newest first
        parts.append(np.broadcast_to(self._phase(issue_times)[:, None, None], (B, N, 1)))
        parts.append(np.broadcast_to(np.arange(N, dtype=np.float32)[None, :, None], (B, N, 1)))
        return np.concatenate(parts, axis=-1).reshape(B * N, -1).astype(np.float32)

    def _split_samples(self, split: SplitData):
        """Features [I*N, D], kW targets [I*N, H] and validity [I*N, H] for every issue of a split."""
        ctx = self.ctx
        L, H, N, F = ctx.L, ctx.H, len(ctx.entities), len(ctx.input_names)
        step = max(1, _CHUNK_CELLS // (L * N * F))
        past, future = np.arange(-L + 1, 1), np.arange(1, H + 1)
        x_parts, y_parts, good_parts = [], [], []
        for first in range(0, len(split.issues), step):
            take = split.issues[first:first + step]
            x_parts.append(self._features(split.inputs[take[:, None] + past[None, :]], split.times[take]))
            y = split.target[take[:, None] + future[None, :]]                               # [b, H, N]
            good = split.target_mask[take[:, None] + future[None, :]] & np.isfinite(y)
            y_parts.append(np.nan_to_num(y).transpose(0, 2, 1).reshape(-1, H))
            good_parts.append(good.transpose(0, 2, 1).reshape(-1, H))
        return np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(good_parts)

    # ----- fit / predict -----
    def fit(self, ctx: TaskContext, train: SplitData, valid: SplitData) -> None:
        try:
            import lightgbm
        except ImportError as exc:
            raise ImportError("The lightgbm model needs the lightgbm package (pip install lightgbm); "
                              "there is no fallback backend") from exc
        super().fit(ctx, train, valid)
        p = self._params
        lags = int(p["lags"])
        self.target_index = ctx.target_input_index
        if lags and self.target_index is None:
            raise ValueError("lightgbm needs a task target_input for its lag features (or set lags to 0)")
        if lags > ctx.L:
            raise ValueError(f"lags={lags} exceeds the history length L={ctx.L}")
        if not len(train.issues):
            raise ValueError("The training split has no windows")
        X, Y, good = self._split_samples(train)
        self.n_features = int(X.shape[1])
        booster_params = {"objective": p["objective"], "learning_rate": float(p["learning_rate"]),
                          "max_depth": int(p["max_depth"]), "num_leaves": int(p["num_leaves"]),
                          "num_threads": int(p["n_jobs"]), "seed": int(p["seed"]),
                          "deterministic": bool(p["deterministic"]), "force_row_wise": bool(p["force_row_wise"]),
                          "verbosity": int(p["verbosity"])}
        self.estimators, self.n_params = [], 0
        for h in range(ctx.H):
            if not good[:, h].any():
                raise ValueError(f"No valid training target for horizon step {h + 1}")
            dataset = lightgbm.Dataset(X[good[:, h]], label=Y[good[:, h], h])
            model = lightgbm.train(booster_params, dataset, num_boost_round=int(p["n_estimators"]))
            self.estimators.append(model)
            self.n_params += sum(tree["num_leaves"] for tree in model.dump_model()["tree_info"])
        self.log(f"model={self.name} features={self.n_features} lags={lags} train_windows={len(train.issues)} "
                 f"samples={len(X)} horizon_models={ctx.H} leaves={self.n_params}")
        if len(valid.issues):
            Xv, Yv, good_v = self._split_samples(valid)
            pred = np.column_stack([m.predict(Xv) for m in self.estimators])
            pred = np.clip(pred, 0, np.tile(ctx.rated_kw, len(Xv) // len(ctx.rated_kw))[:, None])
            if good_v.any():
                self.validation_metrics = {"MAE_kW": float(np.abs(pred - Yv)[good_v].mean())}
                self.log(f"valid_MAE_kW={self.validation_metrics['MAE_kW']:.4f}")

    def predict(self, batch: ForecastBatch) -> np.ndarray:
        ctx = self.ctx
        B, _, N, _ = batch.x_past.shape
        X = self._features(batch.x_past, batch.issue_time)
        out = np.column_stack([m.predict(X) for m in self.estimators])                     # [B*N, H]
        out = out.reshape(B, N, ctx.H).transpose(0, 2, 1)
        out = np.clip(out, 0, ctx.rated_kw[None, None, :]).astype(np.float32)
        if not np.isfinite(out).all():
            raise ValueError("lightgbm produced nonfinite predictions")
        return out
