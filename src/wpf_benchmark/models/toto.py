"""Toto 1.0 inference adapter for the platform's model contract.

Toto is an optional, pretrained time-series foundation model. This adapter keeps the
platform's entity and feature dimensions by packing every turbine's input features
into one Toto variate group, then selecting the target variate from the forecast.
It intentionally supports inference only; target-domain fine-tuning belongs to a
separate adapter because Toto's training data objects do not match SplitData.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..base import Forecaster, register_model
from ...tasks import ForecastBatch, TaskContext


@register_model("toto")
class TotoForecasterModel(Forecaster):
    """Run a pretrained Toto model through the platform's ``Forecaster`` API."""

    default_params = {
        "model_id": "Datadog/Toto-Open-Base-1.0",
        "device": "auto",
        "num_samples": None,
        "samples_per_batch": 10,
        "use_kv_cache": True,
    }

    def __init__(self, **params: Any):
        super().__init__(**params)
        p = self._params
        if not isinstance(p["model_id"], str) or not p["model_id"]:
            raise ValueError("model_id must be a non-empty string")
        if p["device"] not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be 'auto', 'cpu' or 'cuda'")
        if p["num_samples"] is not None and int(p["num_samples"]) < 0:
            raise ValueError("num_samples must be nonnegative or null")
        if int(p["samples_per_batch"]) <= 0:
            raise ValueError("samples_per_batch must be positive")
        if p["num_samples"] is not None and int(p["num_samples"]) > 0:
            if int(p["num_samples"]) % int(p["samples_per_batch"]):
                raise ValueError("num_samples must be divisible by samples_per_batch")

        self._toto = None
        self._forecaster = None
        self._device = None
        self._target_index = None
        self._n_features = None
        self._n_entities = None

    @property
    def deterministic(self) -> bool:
        # The default point mode is deterministic. Sampling mode still uses one
        # deterministic study seed, but is intentionally not expanded by study.py.
        return True

    def validate_context(self, ctx: TaskContext) -> None:
        super().validate_context(ctx)
        if ctx.target_input_index is None:
            raise ValueError("toto needs a target_input to select the forecast variate")
        if ctx.L <= 0 or ctx.H <= 0:
            raise ValueError("toto needs positive history and horizon lengths")
        if int(self._params["num_samples"] or 0) > 0:
            samples = int(self._params["num_samples"])
            batch = int(self._params["samples_per_batch"])
            if samples % batch:
                raise ValueError("num_samples must be divisible by samples_per_batch")

    def _resolve_device(self):
        import torch

        requested = self._params["device"]
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("toto requested device='cuda', but CUDA is unavailable")
        return torch.device(requested)

    def fit(self, ctx: TaskContext, train, valid) -> None:
        """Load pretrained Toto; no platform SplitData fitting is performed."""
        from toto.model.toto import Toto
        from toto.inference.forecaster import TotoForecaster

        self._device = self._resolve_device()
        # Load on CPU first so Hugging Face/local checkpoints do not allocate a
        # second full model on the requested accelerator during deserialization.
        self._toto = Toto.from_pretrained(self._params["model_id"], map_location="cpu")
        self._toto.to(self._device)
        self._toto.eval()
        self._forecaster = TotoForecaster(self._toto.model)
        self._target_index = int(ctx.target_input_index)
        self._n_features = len(ctx.input_names)
        self._n_entities = len(ctx.entities)
        self.n_params = int(sum(parameter.numel() for parameter in self._toto.parameters()))
        self.log(
            f"model=toto model_id={self._params['model_id']} device={self._device} "
            f"variates_per_entity={self._n_features} params={self.n_params}"
        )

    def _inputs(self, batch: ForecastBatch):
        """Convert platform ``[B,L,N,F]`` arrays to Toto ``[B,V,L]`` inputs."""
        import torch
        from toto.data.util.dataset import MaskedTimeseries

        x = np.asarray(batch.x_past, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError(f"toto expects x_past with 4 dimensions, got {x.shape}")
        b, length, n_entities, n_features = x.shape
        if length != self.ctx.L or n_features != self._n_features:
            raise ValueError(
                f"toto input shape {x.shape} does not match L={self.ctx.L}, F={self._n_features}"
            )
        if n_entities != self._n_entities:
            raise ValueError(
                f"toto input entity count {n_entities} does not match fitted count {self._n_entities}"
            )

        observed = np.isfinite(x)
        values = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # Group each turbine's feature channels together. Toto can then perform
        # variate attention within a turbine while accepting different N per task.
        series = torch.from_numpy(np.ascontiguousarray(values.transpose(0, 2, 3, 1).reshape(b, n_entities * n_features, length)))
        padding_mask = torch.from_numpy(
            np.ascontiguousarray(observed.transpose(0, 2, 3, 1).reshape(b, n_entities * n_features, length))
        ).bool()
        entity_ids = torch.arange(n_entities, dtype=torch.int32).repeat_interleave(n_features)
        id_mask = entity_ids.view(1, -1, 1).expand(b, -1, -1)

        step_seconds = int(self.ctx.resolution_minutes) * 60
        issue_seconds = batch.issue_time.astype("datetime64[s]").astype(np.int64)
        offsets = np.arange(-length + 1, 1, dtype=np.int64) * step_seconds
        timestamps = issue_seconds[:, None] + offsets[None, :]
        timestamps = np.broadcast_to(timestamps[:, None, :], (b, n_entities * n_features, length))
        timestamp_seconds = torch.from_numpy(np.ascontiguousarray(timestamps)).to(dtype=torch.int64)
        intervals = torch.full((b, n_entities * n_features), step_seconds, dtype=torch.int64)

        return MaskedTimeseries(
            series=series.to(self._device),
            padding_mask=padding_mask.to(self._device),
            id_mask=id_mask.to(self._device),
            timestamp_seconds=timestamp_seconds.to(self._device),
            time_interval_seconds=intervals.to(self._device),
            num_exogenous_variables=0,
        )

    def predict(self, batch: ForecastBatch) -> np.ndarray:
        if self._forecaster is None:
            raise RuntimeError("toto must be fitted or loaded before predict")

        inputs = self._inputs(batch)
        sample_count = self._params["num_samples"]
        sample_count = None if sample_count in (None, 0) else int(sample_count)
        result = self._forecaster.forecast(
            inputs,
            prediction_length=self.ctx.H,
            num_samples=sample_count,
            samples_per_batch=int(self._params["samples_per_batch"]),
            use_kv_cache=bool(self._params["use_kv_cache"]),
        )
        point = result.mean if sample_count is None else result.median

        b, _, n_entities, _ = batch.x_past.shape
        if n_entities != self._n_entities:
            raise ValueError(
                f"toto batch entity count {n_entities} does not match fitted count {self._n_entities}"
            )
        target_variates = torch.arange(
            0, n_entities * self._n_features, self._n_features, device=point.device
        ) + int(self._target_index)
        selected = point[:, target_variates, :].permute(0, 2, 1).detach().cpu().numpy()
        selected = np.clip(selected, 0, self.ctx.rated_kw[None, None, :]).astype(np.float32)
        if selected.shape != (b, self.ctx.H, n_entities) or not np.isfinite(selected).all():
            raise ValueError(f"toto returned invalid target forecast with shape {selected.shape}")
        return selected
