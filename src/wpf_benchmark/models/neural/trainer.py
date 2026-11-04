"""Shared PyTorch training loop, ported from the legacy SDWPF NeuralForecaster.

Adam; masked MSE or MAE on min/max-scaled targets; early stopping on validation loss
with the best epoch restored; predictions unscaled and clipped to [0, rated_kw].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ...tasks import ForecastBatch, SplitData, TaskContext
from ..base import Forecaster

TRAIN_DEFAULTS: Dict[str, Any] = {"lr": 1e-3, "weight_decay": 0.0, "batch": 256, "epochs": 30,
                                  "patience": 5, "loss": "mse"}


@dataclass(frozen=True)
class NetworkSpec:
    L: int
    H: int
    F: int
    N: int
    input_names: Tuple[str, ...]
    target_input_index: Optional[int]
    target_zero: float                       # 0 kW on the scaled target axis
    params: Dict[str, Any] = field(default_factory=dict)


class NeuralForecaster(Forecaster):
    layout = "entity"                        # "entity": [B', L, F] -> [B', H]; "graph": [B, L, N, F] -> [B, H, N]
    raw_inputs = False                       # True: network receives NaN for missing inputs
    network_defaults: Dict[str, Any] = {}
    default_params = dict(TRAIN_DEFAULTS)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.default_params = {**TRAIN_DEFAULTS, **cls.network_defaults}

    def __init__(self, **params: Any):
        super().__init__(**params)
        p = self._params
        if p["loss"] not in ("mse", "mae"):
            raise ValueError("loss must be 'mse' or 'mae'")
        if min(p["batch"], p["epochs"], p["patience"]) <= 0 or p["lr"] <= 0 or p["weight_decay"] < 0:
            raise ValueError("batch, epochs, patience and lr must be positive")
        if self.layout not in ("entity", "graph"):
            raise ValueError("layout must be 'entity' or 'graph'")

    @property
    def deterministic(self) -> bool:
        return False

    def build_network(self, spec: NetworkSpec):
        raise NotImplementedError

    # ----- data -----
    def _prepare(self, split: SplitData):
        x = self.ctx.scaler.scale_inputs(split.inputs, self.ctx.input_names)
        if not self.raw_inputs:
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = self.ctx.scaler.scale_target(split.target)
        mask = split.target_mask & np.isfinite(y)
        return x, np.nan_to_num(y), mask

    def _batches(self, x, y, mask, issues, shuffle: bool):
        """Yield (inputs, targets, mask) numpy arrays in the network layout."""
        L, H, batch = self.ctx.L, self.ctx.H, int(self._params["batch"])
        past = np.arange(-L + 1, 1)
        future = np.arange(1, H + 1)
        if self.layout == "graph":
            order = np.random.permutation(issues) if shuffle else issues
            for first in range(0, len(order), batch):
                take = order[first:first + batch]
                yield (x[take[:, None] + past[None, :]],              # [B, L, N, F]
                       y[take[:, None] + future[None, :]],            # [B, H, N]
                       mask[take[:, None] + future[None, :]])
            return
        n = x.shape[1]
        pairs = np.arange(len(issues) * n)
        if shuffle:
            pairs = np.random.permutation(pairs)
        for first in range(0, len(pairs), batch):
            chosen = pairs[first:first + batch]
            take, entity = issues[chosen // n], chosen % n
            yield (x[take[:, None] + past[None, :], entity[:, None]],   # [B', L, F]
                   y[take[:, None] + future[None, :], entity[:, None]],  # [B', H]
                   mask[take[:, None] + future[None, :], entity[:, None]])

    def _spec(self) -> NetworkSpec:
        ctx = self.ctx
        zero = float(ctx.scaler.scale_target(np.zeros(1, np.float32))[0])
        network = {k: v for k, v in self._params.items() if k not in TRAIN_DEFAULTS}
        return NetworkSpec(ctx.L, ctx.H, len(ctx.input_names), len(ctx.entities), tuple(ctx.input_names),
                           ctx.target_input_index, zero, network)

    # ----- training -----
    def fit(self, ctx: TaskContext, train: SplitData, valid: SplitData) -> None:
        import torch
        super().fit(ctx, train, valid)
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        p = self._params
        self.network = self.build_network(self._spec()).to(self.device)
        self.n_params = int(sum(t.numel() for t in self.network.parameters() if t.requires_grad))
        optimizer = torch.optim.Adam(self.network.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        data = {"train": (*self._prepare(train), train.issues), "valid": (*self._prepare(valid), valid.issues)}
        span = float(ctx.scaler.span(ctx.scaler.target_name))
        self.log(f"model={self.name} layout={self.layout} device={self.device} params={self.n_params} "
                 f"train_windows={len(train.issues)} valid_windows={len(valid.issues)} loss={p['loss']}")
        best, best_state, best_epoch, best_metrics, stale = float("inf"), None, 0, None, 0
        for epoch in range(1, p["epochs"] + 1):
            scores = {}
            for split, training in (("train", True), ("valid", False)):
                self.network.train(training)
                total = abs_total = count = 0.0
                for xb, yb, mb in self._batches(*data[split], shuffle=training):
                    if not mb.any():
                        continue
                    xt = torch.from_numpy(np.ascontiguousarray(xb, dtype=np.float32)).to(self.device)
                    yt = torch.from_numpy(np.ascontiguousarray(yb, dtype=np.float32)).to(self.device)
                    mt = torch.from_numpy(np.ascontiguousarray(mb)).to(self.device)
                    with torch.set_grad_enabled(training):
                        error = (self.network(xt) - yt)[mt]
                        loss_sum = error.abs().sum() if p["loss"] == "mae" else error.square().sum()
                        if training:
                            optimizer.zero_grad()
                            (loss_sum / mt.sum().clamp(min=1)).backward()
                            optimizer.step()
                    total += float(loss_sum.detach())
                    abs_total += float(error.detach().abs().sum())
                    count += float(mt.sum())
                if count == 0:
                    raise ValueError(f"No valid targets in the {split} split")
                scores[split] = (total / count, abs_total / count * span)
            self.log(f"epoch={epoch} train_{p['loss']}={scores['train'][0]:.8f} valid_{p['loss']}="
                     f"{scores['valid'][0]:.8f} valid_MAE_kW={scores['valid'][1]:.4f}")
            if scores["valid"][0] < best - 1e-8:
                best, best_epoch, stale = scores["valid"][0], epoch, 0
                best_metrics = {p["loss"]: scores["valid"][0], "MAE_kW": scores["valid"][1]}
                best_state = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
            else:
                stale += 1
                if stale >= p["patience"]:
                    break
        self.network.load_state_dict(best_state)
        self.network.eval()
        self.validation_metrics = dict(best_metrics, best_epoch=best_epoch, stopped_epoch=epoch)
        self.log(f"best_epoch={best_epoch} best_valid_{p['loss']}={best:.8f} stopped_epoch={epoch}")

    # ----- inference -----
    def predict(self, batch: ForecastBatch) -> np.ndarray:
        torch, ctx = self.torch, self.ctx
        x = ctx.scaler.scale_inputs(batch.x_past, ctx.input_names)           # [B, L, N, F]
        if not self.raw_inputs:
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        B, L, N, F = x.shape
        if self.layout == "entity":
            parts, size = x.transpose(0, 2, 1, 3).reshape(B * N, L, F), 4096
        else:
            parts, size = x, int(self._params["batch"])
        outputs = []
        self.network.eval()
        with torch.no_grad():
            for first in range(0, len(parts), size):
                chunk = torch.from_numpy(np.ascontiguousarray(parts[first:first + size])).to(self.device)
                outputs.append(self.network(chunk).float().cpu().numpy())
        out = np.concatenate(outputs)
        out = out.reshape(B, N, ctx.H).transpose(0, 2, 1) if self.layout == "entity" else out
        out = ctx.scaler.unscale_target(out)
        out = np.clip(out, 0, ctx.rated_kw[None, None, :]).astype(np.float32)
        if not np.isfinite(out).all():
            raise ValueError(f"{self.name} produced nonfinite predictions")
        return out

    # ----- checkpoints -----
    def save_checkpoint(self, path, metadata: Optional[dict] = None) -> None:
        import os
        torch = self.torch
        state = {"format_version": CHECKPOINT_VERSION, "model": self.name, "params": self.params,
                 "task": self.ctx.task.config.to_dict(), "input_names": list(self.ctx.input_names),
                 "entities": list(self.ctx.entities), "scaler": self.ctx.scaler.to_dict(),
                 "validation_metrics": self.validation_metrics, "metadata": metadata or {},
                 "torch_version": str(torch.__version__),
                 "state_dict": {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}}
        temporary = str(path) + ".tmp"
        torch.save(state, temporary)
        os.replace(temporary, str(path))


CHECKPOINT_VERSION = 2


def load_checkpoint(path, task, device: Optional[str] = None) -> NeuralForecaster:
    """Rebuild a fitted network for ``task`` (a ResolvedTask) without training.

    ``device`` defaults to the one ``fit`` uses (CUDA when available), so a reloaded model
    reproduces the original predictions; CPU and GPU kernels differ in the last digits.
    """
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    from ...tasks import Scaler
    from ..base import get_model
    saved = torch.load(str(path), map_location="cpu", weights_only=True)
    if saved.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint format")
    model = get_model(saved["model"])(**saved["params"])
    cfg = task.config
    if list(cfg.inputs) != saved["input_names"] or [str(e["id"]) for e in task.meta["entities"]] != saved["entities"]:
        raise ValueError("Checkpoint inputs or entity order differ from the task")
    model.ctx = TaskContext(task, list(cfg.inputs), cfg.target, cfg.target_input, task.L, task.H,
                            saved["entities"], task.rated_kw, Scaler.from_dict(saved["scaler"]))
    model.torch, model.device = torch, torch.device(device)
    model.network = model.build_network(model._spec()).to(model.device)
    model.network.load_state_dict(saved["state_dict"], strict=True)
    model.network.eval()
    model.validation_metrics = saved["validation_metrics"]
    model.n_params = int(sum(t.numel() for t in model.network.parameters() if t.requires_grad))
    return model
