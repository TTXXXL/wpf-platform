"""Toto 1.0 native marginals and supervised capacity-normalized quantile heads.

The pretrained transformer layout, turbine grouping and scaling path are preserved.
Only a single future patch is supported; no joint trajectory distribution is claimed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import Forecaster, register_model
from .neural.quantile import QuantileNeuralForecaster, Q_DEFAULTS
from .neural.trainer import NetworkSpec
from ..probability import LEVELS, MEDIAN, ordered_head, postprocess
from ..provenance import canonical_sha256, sha256_file


def pack_toto(x, stride):
    """Physical [B,L,N,F] -> padded [B,N*F,T], observed mask and turbine IDs."""
    import torch
    import torch.nn.functional as functional
    B, L, N, F = x.shape
    series = x.permute(0, 2, 3, 1).reshape(B, N * F, L)
    observed = torch.isfinite(series)
    series = torch.where(observed, series, torch.zeros_like(series))
    padding = (-L) % stride
    series = functional.pad(series, (padding, 0))
    observed = functional.pad(observed, (padding, 0), value=False)
    ids = torch.arange(N, device=x.device, dtype=torch.int32).repeat_interleave(F)
    ids = ids.view(1, N * F, 1).expand(B, N * F, series.shape[-1])
    return series, observed, ids


def toto_runtime_identity():
    import importlib.metadata
    import toto.model.backbone
    import toto.model.toto
    import toto.model.scaler
    modules = (toto.model.toto, toto.model.backbone, toto.model.scaler)
    try:
        version = importlib.metadata.version("toto-ts")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"toto_ts_version": version,
            "source_kind": "installed_distribution" if version else "source_tree",
            "imported_sources": {module.__name__: {"path": str(Path(module.__file__).resolve()),
                                                   "sha256": sha256_file(module.__file__)} for module in modules}}


def load_toto(directory):
    """Require explicit local weights and a recorded config, never trigger a download."""
    from toto.model.toto import Toto
    directory = Path(directory)
    config_path, weights = directory / "config.json", directory / "model.safetensors"
    if not config_path.is_file() or not weights.is_file():
        raise ValueError("Toto probability models require a local directory with config.json and model.safetensors")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = Toto.load_from_checkpoint(str(directory), map_location="cpu", strict=True, **config)
    if model.model.patch_embed.patch_size != model.model.patch_embed.stride:
        raise ValueError("Toto probability path requires non-overlapping patches")
    return model, {"checkpoint_sha256": sha256_file(weights), "config_sha256": sha256_file(config_path),
                   "config": config, "target_training": "none", "runtime": toto_runtime_identity()}


@register_model("toto1_zs_q")
class TotoNativeQuantiles(Forecaster):
    default_params = {"model_dir": "", "batch": 8, "device": "auto"}
    monotone_quantiles = True

    def validate_context(self, ctx):
        if ctx.target_input_index is None or ctx.H <= 0 or ctx.H > 64:
            raise ValueError("Toto native quantiles require target_input and 1..64 future steps")
        if int(self._params["batch"]) <= 0 or self._params["device"] not in ("auto", "cpu", "cuda"):
            raise ValueError("Invalid Toto probability batch or device")

    def fit(self, ctx, train, valid):
        import torch
        super().fit(ctx, train, valid)
        self.toto, identity = load_toto(self._params["model_dir"])
        requested = self._params["device"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
        self.toto.to(self.device).eval()
        if ctx.H > self.toto.model.patch_embed.stride:
            raise ValueError("H exceeds the native single-patch protocol")
        self.n_params = sum(p.numel() for p in self.toto.parameters())
        from .mixture_quantiles import SOLVER
        runtime = identity["runtime"]
        self.probability_metadata = {**identity, "cdf_solver": SOLVER, "joint_distribution": False,
                                     "imported_toto_source_kind": runtime["source_kind"],
                                     "turbine_grouping": "independent", "local_target_labels_used": False}

    def _distribution_parameters(self, x):
        import torch
        model = self.toto.model
        series, observed, ids = pack_toto(x, model.patch_embed.stride)
        output = model(series, observed, ids, scaling_prefix_length=series.shape[-1], num_exogenous_variables=0)
        base, loc, scale = output
        components = base.component_distribution
        if base.mixture_distribution.probs.shape[-1] != 24:
            raise ValueError("Toto1 native protocol requires 24 Student-t mixture components")
        patch = model.patch_embed.stride
        target = torch.arange(len(self.ctx.entities), device=x.device) * len(self.ctx.input_names) + self.ctx.target_input_index
        # The first H positions of the final output patch forecast issue+1..issue+H.
        ix = slice(-patch, -patch + self.ctx.H if self.ctx.H < patch else None)
        weights = base.mixture_distribution.probs[:, target, ix]
        df = components.df[:, target, ix]
        local_loc = loc[:, target, -patch:] if loc.shape[-1] > 1 else loc[:, target]
        local_scale = scale[:, target, -patch:] if scale.shape[-1] > 1 else scale[:, target]
        if local_loc.shape[-1] > 1:
            local_loc, local_scale = local_loc[..., :self.ctx.H], local_scale[..., :self.ctx.H]
        center = components.loc[:, target, ix] * local_scale[..., None] + local_loc[..., None]
        spread = components.scale[:, target, ix] * local_scale[..., None]
        return [a.detach().cpu().numpy().transpose(0, 2, 1, 3) for a in (weights, df, center, spread)]

    def predict_quantiles(self, batch):
        import torch
        from .mixture_quantiles import mixture_quantiles
        outputs, means = [], []
        self.toto.eval()
        with torch.no_grad():
            for first in range(0, len(batch.x_past), int(self._params["batch"])):
                x = torch.as_tensor(batch.x_past[first:first+self._params["batch"]], dtype=torch.float32, device=self.device)
                weights, df, center, scale = self._distribution_parameters(x)
                outputs.append(mixture_quantiles(weights, df, center, scale))
                if np.any(df <= 1):
                    raise ValueError("Native mixture mean is undefined for df <= 1")
                means.append((weights * center).sum(-1))
        self.native_mean = np.concatenate(means).astype(np.float32)
        out = np.concatenate(outputs).astype(np.float32)
        self.native_median = out[..., MEDIAN].copy()
        return out

    def predict(self, batch):
        return postprocess(self.predict_quantiles(batch), monotone=True)[..., MEDIAN]


def build_toto_q_network(backbone, spec, adaptation):
    from torch import nn
    if adaptation not in ("head", "last4", "full"):
        raise ValueError("adaptation must be head, last4 or full")
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    if adaptation == "full":
        for parameter in backbone.parameters():
            parameter.requires_grad_(True)
    elif adaptation == "last4":
        for layer in list(backbone.transformer.layers)[-4:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
    # Original distribution projection is unused; keep it out of the optimizer in every mode.
    for parameter in backbone.output_distribution.parameters():
        parameter.requires_grad_(False)
    patch = backbone.patch_embed.stride
    if not 0 < spec.H <= patch or backbone.patch_embed.patch_size != patch:
        raise ValueError("Toto-Q requires a non-overlapping single future patch")

    class TotoQuantileNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.quantile_head = nn.Linear(backbone.embed_dim, len(LEVELS))

        def train(self, mode=True):
            super().train(mode)
            # Frozen pretrained layers remain deterministic, including their dropout.
            if mode and adaptation != "full":
                self.backbone.eval()
                if adaptation == "last4":
                    for layer in list(self.backbone.transformer.layers)[-4:]:
                        layer.train(True)
            return self

        def forward(self, x):
            import torch
            series, observed, ids = pack_toto(x, patch)
            hidden, _, _ = self.backbone.backbone(series, observed, ids,
                                                 scaling_prefix_length=series.shape[-1], num_exogenous_variables=0)
            target = torch.arange(spec.N, device=x.device) * spec.F + spec.target_input_index
            hidden = hidden[:, target, -patch:][:, :, :spec.H]
            # Capacity-normalized physical power, independent of native output loc/scale.
            return ordered_head(self.quantile_head(hidden)).permute(0, 2, 1, 3)

    return TotoQuantileNetwork()


@register_model("toto1_wpf_q")
class TotoWindQuantiles(QuantileNeuralForecaster):
    default_params = {**Q_DEFAULTS, "batch": 8, "lr": 1e-4, "model_dir": "", "adaptation": "head"}
    layout = "graph"

    def __init__(self, **params):
        Forecaster.__init__(self, **params)
        p = self._params
        if p["loss"] != "pinball" or p["adaptation"] not in ("head", "last4", "full"):
            raise ValueError("Toto1-WPF-Q requires pinball loss and head/last4/full adaptation")
        if min(p["batch"], p["epochs"], p["patience"]) <= 0 or p["lr"] <= 0 or p["weight_decay"] < 0:
            raise ValueError("Invalid Toto-Q training parameters")

    def validate_context(self, ctx):
        if ctx.target_input_index is None or not 0 < ctx.H <= 64:
            raise ValueError("Toto1-WPF-Q requires target_input and 1..64 future steps")

    def build_network(self, spec):
        toto, identity = load_toto(self._params["model_dir"])
        self._weight_identity = identity
        return build_toto_q_network(toto.model, spec, self._params["adaptation"])

    def _prepare(self, split):
        y = split.target / self.ctx.rated_kw
        return split.inputs, y, split.target_mask & np.isfinite(y)

    def fit(self, ctx, train, valid):
        super().fit(ctx, train, valid)
        trainable = [name for name, p in self.network.named_parameters() if p.requires_grad]
        self.probability_metadata.update({**self._weight_identity, "target_training": "supervised",
                                          "adaptation": self._params["adaptation"],
                                          "missing_input_encoding": "native_scaler_mask_legacy_embedding",
                                          "trainable_parameter_names": trainable,
                                          "trainable_names_sha256": canonical_sha256(trainable)})

    def save_checkpoint(self, path, metadata=None):
        super().save_checkpoint(path, {**(metadata or {}), "pretrained_identity": self._weight_identity})

    def predict_quantiles(self, batch):
        import torch
        self.network.eval()
        output = []
        with torch.no_grad():
            for first in range(0, len(batch.x_past), int(self._params["batch"])):
                x = torch.as_tensor(batch.x_past[first:first+self._params["batch"]], dtype=torch.float32, device=self.device)
                output.append(self.network(x).cpu().numpy())
        return (np.concatenate(output) * self.ctx.rated_kw[None, None, :, None]).astype(np.float32)
