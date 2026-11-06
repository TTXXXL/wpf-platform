"""I0-only direct quantile boosters; no time-of-day or turbine-ID features."""
from __future__ import annotations

import time
import numpy as np

from .base import Forecaster, register_model
from .lightgbm_model import LightGBM
from .probability_baselines import history_conditions
from ..probability import LEVELS, MEDIAN, postprocess


@register_model("lightgbm_q")
class LightGBMQuantiles(LightGBM):
    default_params = {k: v for k, v in LightGBM.default_params.items() if k != "objective"}

    def __init__(self, **params):
        Forecaster.__init__(self, **params)
        p = self._params
        if min(p["n_estimators"], p["max_depth"], p["num_leaves"], p["n_jobs"]) <= 0 or p["learning_rate"] <= 0:
            raise ValueError("Invalid LightGBM-Q parameters")
        if p["lags"] < 0:
            raise ValueError("lags must be nonnegative")

    def validate_context(self, ctx):
        super().validate_context(ctx)
        if ctx.target_input_index is None or ctx.L != ctx.steps_per_day:
            raise ValueError("LightGBM-Q I0 requires a target_input and exactly 24 h history")

    def _features(self, x, issue_times):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        B, L, N, F = x.shape
        ticks = np.arange(L, dtype=np.float32)
        ticks -= ticks.mean()
        slope = np.einsum("blnf,l->bnf", x, ticks) / np.square(ticks).sum()
        parts = [x[:, -1], x.mean(1), x.std(1), x.min(1), x.max(1), slope]
        if self._params["lags"]:
            parts.append(x[:, ::-1][:, :self._params["lags"], :, self.target_index].transpose(0, 2, 1))
        return np.concatenate(parts, axis=-1).reshape(B * N, -1).astype(np.float32)

    def _anchors(self, split):
        chunks = []
        for first in range(0, len(split.issues), 128):
            batch = split.batch(split.issues[first:first+128])
            anchor, _ = history_conditions(batch.x_past, self.target_index, self.ctx.rated_kw, self.ctx.steps_per_day)
            chunks.append(anchor.ravel())
        return np.concatenate(chunks)

    def fit(self, ctx, train, valid):
        import lightgbm
        Forecaster.fit(self, ctx, train, valid)
        self.target_index = ctx.target_input_index
        X, Y, mask = self._split_samples(train)
        anchor = self._anchors(train)
        capacities = np.tile(ctx.rated_kw, len(train.issues))
        if not np.all(capacities == capacities[0]):
            raise ValueError("Pooled LightGBM-Q currently requires equal rated capacity")
        Y = (Y - anchor[:, None]) / capacities[:, None]
        p = self._params
        params = {"objective": "quantile", "metric": "quantile", "learning_rate": p["learning_rate"],
                  "max_depth": p["max_depth"], "num_leaves": p["num_leaves"], "num_threads": p["n_jobs"],
                  "seed": p["seed"], "deterministic": p["deterministic"],
                  "force_row_wise": p["force_row_wise"], "verbosity": p["verbosity"]}
        self.estimators, self.n_params = [], 0
        started = time.perf_counter()
        for h in range(ctx.H):
            good = mask[:, h] & np.isfinite(Y[:, h])
            if not good.any():
                raise ValueError(f"No valid training labels for horizon {h+1}")
            models = []
            for tau in LEVELS:
                dataset = lightgbm.Dataset(X[good], label=Y[good, h], free_raw_data=True)
                booster = lightgbm.train({**params, "alpha": float(tau)}, dataset,
                                         num_boost_round=int(p["n_estimators"]))
                models.append(booster)
                self.n_params += sum(t["num_leaves"] for t in booster.dump_model()["tree_info"])
            self.estimators.append(models)
        self.probability_metadata = {"information_set": "I0", "calendar_features": False,
                                     "entity_id_feature": False, "objective": "quantile",
                                     "booster_count": ctx.H * len(LEVELS),
                                     "fit_seconds": time.perf_counter() - started,
                                     "target_space": "(power - last_finite_power) / rated_kw"}
        self.log(f"model=lightgbm_q boosters={ctx.H * len(LEVELS)} features={X.shape[1]}")
        from ..evaluation.probabilistic import probability_scores
        totals = absolute = count = 0.0
        for first in range(0, len(valid.issues), 128):
            batch = valid.batch(valid.issues[first:first+128], with_targets=True)
            metrics = probability_scores(self.predict_quantiles(batch), batch.y, batch.y_mask, ctx.rated_kw)
            if metrics["n_valid"]:
                n = metrics["n_valid"]
                totals += metrics["normalized_mean_pinball"] * n
                absolute += metrics["q50_MAE_kW"] * n
                count += n
        if not count:
            raise ValueError("No valid targets in the validation split")
        self.validation_metrics = {"normalized_mean_pinball": totals / count, "q50_MAE_kW": absolute / count}

    def predict_quantiles(self, batch):
        started = time.perf_counter()
        ctx = self.ctx
        B, _, N, _ = batch.x_past.shape
        X = self._features(batch.x_past, batch.issue_time)
        anchor, _ = history_conditions(batch.x_past, self.target_index, ctx.rated_kw, ctx.steps_per_day)
        raw = np.stack([np.column_stack([m.predict(X) for m in models]) for models in self.estimators], axis=1)
        raw = raw.reshape(B, N, ctx.H, len(LEVELS)).transpose(0, 2, 1, 3)
        self.probability_metadata["last_predict_seconds"] = time.perf_counter() - started
        return (anchor[:, None, :, None] + raw * ctx.rated_kw[None, None, :, None]).astype(np.float32)

    def predict(self, batch):
        return postprocess(self.predict_quantiles(batch))[..., MEDIAN]
