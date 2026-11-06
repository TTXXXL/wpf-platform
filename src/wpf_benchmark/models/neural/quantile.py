"""Nine-quantile variants of existing backbones; legacy trainers remain unchanged."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..base import Forecaster, get_model, register_model
from .trainer import NeuralForecaster, NetworkSpec
from ...probability import LEVELS, MEDIAN, ordered_head, postprocess, torch_pinball

Q_DEFAULTS = {"lr": 1e-3, "weight_decay": 0.0, "batch": 256, "epochs": 30,
              "patience": 5, "loss": "pinball"}


class QuantileNeuralForecaster(Forecaster):
    backbone_name = ""
    layout = "entity"
    monotone_quantiles = True
    _batches = NeuralForecaster._batches
    save_checkpoint = NeuralForecaster.save_checkpoint

    def __init__(self, **params):
        super().__init__(**params)
        p = self._params
        if p["loss"] != "pinball":
            raise ValueError("Q models require loss='pinball'")
        if min(p["batch"], p["epochs"], p["patience"]) <= 0 or p["lr"] <= 0 or p["weight_decay"] < 0:
            raise ValueError("Invalid quantile training parameters")
        self._backbone = get_model(self.backbone_name)(**{k: v for k, v in p.items() if k not in Q_DEFAULTS})

    @property
    def deterministic(self):
        return False

    def validate_context(self, ctx):
        self._backbone.validate_context(ctx)
        if self.layout == "entity" and not np.all(ctx.rated_kw == ctx.rated_kw[0]):
            raise ValueError("Entity Q trainer currently requires equal rated capacities")

    def _spec(self):
        ctx = self.ctx
        return NetworkSpec(ctx.L, ctx.H, len(ctx.input_names), len(ctx.entities), tuple(ctx.input_names),
                           ctx.target_input_index, float(ctx.scaler.scale_target(np.zeros(1))[0]),
                           {k: v for k, v in self._params.items() if k not in Q_DEFAULTS})

    def build_network(self, spec):
        from torch import nn
        backbone = self._backbone.build_network(replace(spec, H=spec.H * len(LEVELS)))
        graph, H = self.layout == "graph", spec.H

        class QuantileNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone

            def forward(self, x):
                out = self.backbone(x)
                if graph:
                    out = out.transpose(1, 2).reshape(out.shape[0], spec.N, H, len(LEVELS)).transpose(1, 2)
                else:
                    out = out.reshape(out.shape[0], H, len(LEVELS))
                return ordered_head(out)

        return QuantileNetwork()

    def _prepare(self, split):
        x = self.ctx.scaler.scale_inputs(split.inputs, self.ctx.input_names)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = split.target / self.ctx.rated_kw
        mask = split.target_mask & np.isfinite(y)
        return x, y, mask

    def fit(self, ctx, train, valid):
        import torch
        super().fit(ctx, train, valid)
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = self.build_network(self._spec()).to(self.device)
        self.n_params = sum(p.numel() for p in self.network.parameters())
        optimizer = torch.optim.Adam((p for p in self.network.parameters() if p.requires_grad),
                                     lr=self._params["lr"], weight_decay=self._params["weight_decay"])
        data = {"train": (*self._prepare(train), train.issues), "valid": (*self._prepare(valid), valid.issues)}
        import time
        best, best_state, stale = float("inf"), None, 0
        epoch_seconds = []
        rated = ctx.rated_kw
        # Entity batches use the common K/P capacity; graph batches retain the turbine axis.
        for epoch in range(1, int(self._params["epochs"]) + 1):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            started_epoch = time.perf_counter()
            scores = {}
            for name, training in (("train", True), ("valid", False)):
                self.network.train(training)
                total = absolute = count = 0
                for xb, yb, mb in self._batches(*data[name], shuffle=training):
                    if not mb.any():
                        continue
                    xt = torch.as_tensor(np.ascontiguousarray(xb), dtype=torch.float32, device=self.device)
                    yt = torch.as_tensor(np.ascontiguousarray(yb), dtype=torch.float32, device=self.device)
                    mt = torch.as_tensor(np.ascontiguousarray(mb), dtype=torch.bool, device=self.device)
                    with torch.set_grad_enabled(training):
                        pred = self.network(xt)
                        evaluated = pred if training else pred.clamp_min(0)
                        loss = torch_pinball(evaluated, yt, mt)
                        if training:
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    n = int(mt.sum())
                    total += float(loss.detach()) * n
                    scale = torch.as_tensor(rated if self.layout == "graph" else rated[0], device=self.device)
                    absolute += float(((evaluated[..., MEDIAN] - yt)[mt].abs() *
                                       (scale.expand_as(yt)[mt])).sum().detach())
                    count += n
                if not count:
                    raise ValueError(f"No valid targets in the {name} split")
                scores[name] = (total / count, absolute / count)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            epoch_seconds.append(time.perf_counter() - started_epoch)
            self.log(f"epoch={epoch} train_normalized_mean_pinball={scores['train'][0]:.8f} "
                     f"valid_normalized_mean_pinball={scores['valid'][0]:.8f} valid_q50_MAE_kW={scores['valid'][1]:.4f}")
            if scores["valid"][0] < best - 1e-4:
                best, best_epoch, stale = scores["valid"][0], epoch, 0
                best_metrics = {"normalized_mean_pinball": best, "q50_MAE_kW": scores["valid"][1]}
                best_state = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
            else:
                stale += 1
                if stale >= self._params["patience"]:
                    break
        self.network.load_state_dict(best_state)
        self.network.eval()
        self.validation_metrics = dict(best_metrics, best_epoch=best_epoch, stopped_epoch=epoch)
        self.probability_metadata = {"missing_input_encoding": "legacy_zero_fill",
                                     "epoch_seconds": epoch_seconds,
                                     "validation_postprocessing": "lower_clip_0_no_upper_clip",
                                     "quantile_head_output_space": "capacity_normalized",
                                     "trainable_parameters": sum(p.numel() for p in self.network.parameters() if p.requires_grad)}

    def predict_quantiles(self, batch):
        torch, ctx = self.torch, self.ctx
        x = ctx.scaler.scale_inputs(batch.x_past, ctx.input_names)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        B, L, N, F = x.shape
        parts = x.transpose(0, 2, 1, 3).reshape(B * N, L, F) if self.layout == "entity" else x
        output = []
        self.network.eval()
        with torch.no_grad():
            for first in range(0, len(parts), int(self._params["batch"])):
                xt = torch.as_tensor(np.ascontiguousarray(parts[first:first+self._params["batch"]]),
                                     dtype=torch.float32, device=self.device)
                output.append(self.network(xt).cpu().numpy())
        out = np.concatenate(output)
        if self.layout == "entity":
            out = out.reshape(B, N, ctx.H, len(LEVELS)).transpose(0, 2, 1, 3)
        return (out * ctx.rated_kw[None, None, :, None]).astype(np.float32)

    def predict(self, batch):
        return postprocess(self.predict_quantiles(batch), monotone=True)[..., MEDIAN]


def _register_q(name):
    base = get_model(name)
    cls = type(base.__name__ + "Quantiles", (QuantileNeuralForecaster,), {
        "backbone_name": name, "layout": base.layout,
        "default_params": {**base.default_params, "loss": "pinball"}, "__module__": __name__})
    register_model(name + "_q")(cls)
    globals()[cls.__name__] = cls


for _name in ("dlinear", "lstm", "tcn", "patchtst", "itransformer", "agcrn"):
    _register_q(_name)
