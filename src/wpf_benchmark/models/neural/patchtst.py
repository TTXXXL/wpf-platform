"""PatchTST forecaster (entity layout).

Follows Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with
Patches" (ICLR 2023). Written from the paper and the structure of the Time-Series-Library
implementation (https://github.com/thuml/Time-Series-Library, MIT License,
models/PatchTST.py); no source code was copied.

Differences from the original paper:
  * The original predicts every channel. Here the network outputs only the target channel
    (the turbine power) for one turbine, as the platform requires.
  * head="channel" (default, as in the paper): channels are independent and share all
    weights, so only the target channel's patches influence the output. The network
    therefore encodes just the target channel; the other input features are unused.
    This is numerically identical to encoding all channels and selecting the target one.
  * head="joint": all channels are encoded and flattened into one linear head. This is
    the variant of the legacy SDWPF code, kept for comparison.
  * RevIN has no learnable affine parameters unless revin_affine=True.
  * The nonlinearity in the feed-forward block is ReLU (torch default) instead of GELU,
    and there is no separate head dropout.
"""
from __future__ import annotations

from ..base import register_model
from .trainer import NetworkSpec, NeuralForecaster

_EPS = 1e-5


@register_model("patchtst")
class PatchTSTForecaster(NeuralForecaster):
    layout = "entity"
    network_defaults = {"d_model": 128, "heads": 4, "layers": 2, "ff": 256, "patch": 12,
                        "patch_stride": 6, "dropout": 0.1, "head": "channel",
                        "revin_affine": False}

    def __init__(self, **params):
        super().__init__(**params)
        p = self._params
        if min(p["d_model"], p["heads"], p["layers"], p["ff"], p["patch"], p["patch_stride"]) <= 0:
            raise ValueError("d_model, heads, layers, ff, patch and patch_stride must be positive")
        if p["d_model"] % p["heads"]:
            raise ValueError("d_model must be divisible by heads")
        if p["head"] not in ("channel", "joint"):
            raise ValueError("head must be 'channel' or 'joint'")
        if not 0 <= p["dropout"] < 1:
            raise ValueError("dropout must be in [0, 1)")

    def build_network(self, spec: NetworkSpec):
        import torch
        from torch import nn

        q = spec.params
        if spec.target_input_index is None:
            raise ValueError("patchtst needs a target_input (the power input channel)")
        if q["patch"] > spec.L:
            raise ValueError("patch must not exceed the history length L")
        n_patches = (spec.L - q["patch"]) // q["patch_stride"] + 2
        target, joint = int(spec.target_input_index), q["head"] == "joint"
        channels = spec.F if joint else 1            # channels that go through the encoder
        local_target = target if joint else 0
        patch, stride, d_model, horizon = q["patch"], q["patch_stride"], q["d_model"], spec.H
        affine = bool(q["revin_affine"])

        class PatchTST(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(patch, d_model)
                self.position = nn.Parameter(torch.empty(1, n_patches, d_model).uniform_(-0.02, 0.02))
                self.drop = nn.Dropout(q["dropout"])
                layer = nn.TransformerEncoderLayer(d_model, q["heads"], dim_feedforward=q["ff"],
                                                   dropout=q["dropout"], batch_first=True,
                                                   norm_first=False)
                self.encoder = nn.TransformerEncoder(layer, q["layers"], enable_nested_tensor=False)
                self.head = nn.Linear(channels * n_patches * d_model, horizon)
                if affine:
                    self.affine_weight = nn.Parameter(torch.ones(1, channels, 1))
                    self.affine_bias = nn.Parameter(torch.zeros(1, channels, 1))

            def forward(self, x):                                   # [B, L, F] -> [B, H]
                x = x.transpose(1, 2)                               # [B, F, L]
                mean = x.mean(-1, keepdim=True)
                std = (x.var(-1, unbiased=False, keepdim=True) + _EPS).sqrt()
                z = (x - mean) / std
                if not joint:
                    z = z[:, target:target + 1]
                if affine:
                    z = z * self.affine_weight + self.affine_bias
                B = z.shape[0]
                z = torch.cat([z, z[..., -1:].expand(-1, -1, stride)], dim=-1)   # end padding
                z = z.unfold(-1, patch, stride).reshape(B * channels, n_patches, patch)
                z = self.encoder(self.drop(self.embed(z) + self.position))
                out = self.head(z.reshape(B, channels * n_patches * d_model))      # [B, H]
                if affine:
                    out = (out - self.affine_bias[0, local_target, 0]) / (
                        self.affine_weight[0, local_target, 0] + _EPS * _EPS)
                return out * std[:, target] + mean[:, target]

        return PatchTST()
