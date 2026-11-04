"""iTransformer forecaster (entity layout).

Follows Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series
Forecasting" (ICLR 2024). Written from the paper and the structure of the
Time-Series-Library implementation (https://github.com/thuml/Time-Series-Library, MIT
License, models/iTransformer.py, layers/Embed.py); no source code was copied.

Differences from the original paper:
  * Tokens are the input features of ONE turbine (wind speed, direction, power, ...),
    not the variables of a multivariate series across turbines. Turbines never interact.
  * Only the output of the target feature's token is used as the forecast.
  * No time-mark features are used.
  * The attention block is torch's nn.TransformerEncoder (post-norm, GELU) rather than
    the library's own AttentionLayer; the final LayerNorm is kept.
Normalization follows the original use_norm=True: per sample and per variable, subtract
the mean and divide by the standard deviation, then restore with the target's statistics.
"""
from __future__ import annotations

from ..base import register_model
from .trainer import NetworkSpec, NeuralForecaster

_EPS = 1e-5


@register_model("itransformer")
class ITransformerForecaster(NeuralForecaster):
    layout = "entity"
    network_defaults = {"d_model": 128, "heads": 4, "layers": 2, "ff": 256, "dropout": 0.1}

    def __init__(self, **params):
        super().__init__(**params)
        p = self._params
        if min(p["d_model"], p["heads"], p["layers"], p["ff"]) <= 0:
            raise ValueError("d_model, heads, layers and ff must be positive")
        if p["d_model"] % p["heads"]:
            raise ValueError("d_model must be divisible by heads")
        if not 0 <= p["dropout"] < 1:
            raise ValueError("dropout must be in [0, 1)")

    def build_network(self, spec: NetworkSpec):
        import torch
        from torch import nn

        q = spec.params
        if spec.target_input_index is None:
            raise ValueError("itransformer needs a target_input (the power input channel)")
        target, L, horizon = int(spec.target_input_index), spec.L, spec.H

        class ITransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(L, q["d_model"])
                self.drop = nn.Dropout(q["dropout"])
                layer = nn.TransformerEncoderLayer(q["d_model"], q["heads"], dim_feedforward=q["ff"],
                                                   dropout=q["dropout"], activation="gelu",
                                                   batch_first=True, norm_first=False)
                self.encoder = nn.TransformerEncoder(layer, q["layers"], norm=nn.LayerNorm(q["d_model"]),
                                                     enable_nested_tensor=False)
                self.project = nn.Linear(q["d_model"], horizon)

            def forward(self, x):                                   # [B, L, F] -> [B, H]
                x = x.transpose(1, 2)                               # [B, F, L]: one token per feature
                mean = x.mean(-1, keepdim=True)
                std = (x.var(-1, unbiased=False, keepdim=True) + _EPS).sqrt()
                tokens = self.encoder(self.drop(self.embed((x - mean) / std)))    # [B, F, d_model]
                out = self.project(tokens[:, target])                              # [B, H]
                return out * std[:, target] + mean[:, target]

        return ITransformer()
