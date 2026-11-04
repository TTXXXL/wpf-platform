"""DLinear (entity layout).

Zeng, Chen, Zhang, Xu (2023), "Are Transformers Effective for Time Series Forecasting?".
A moving-average decomposition splits the history into trend (sliding mean, window
``moving_avg``, both ends padded by repeating the first and last value) and seasonal
(history minus trend). One ``Linear(L, H)`` maps each part to the forecast and the two
outputs are added. Linear weights start at 1/L as in the reference code.

Channel handling:

* ``channels = "target"`` (default): only the ``target_input`` channel (``spec.target_input_index``)
  is decomposed and mapped; other input channels are ignored. Raises ValueError when the task
  has no ``target_input``. ``individual`` must be False here (there is a single channel).
* ``channels = "all"``: every input channel is decomposed and mapped to H steps, then
  ``Linear(F, 1)`` combines the F forecasts into the target. This is an extension of the
  original paper, which forecasts each channel's own future. ``individual = False`` shares the
  two ``Linear(L, H)`` across channels (as the reference code does); ``individual = True``
  gives each channel its own pair.

``moving_avg`` must be odd so that the pooled series keeps length L.

Code source: written from the paper, following the structure of Time-Series-Library
(https://github.com/thuml/Time-Series-Library, MIT license), ``models/DLinear.py`` and
``series_decomp`` in ``layers/Autoformer_EncDec.py``. The code is rewritten, not copied.
"""
from __future__ import annotations

from ..base import register_model
from .trainer import NetworkSpec, NeuralForecaster


@register_model("dlinear")
class DLinearForecaster(NeuralForecaster):
    layout = "entity"
    network_defaults = {"moving_avg": 25, "channels": "target", "individual": False}

    def __init__(self, **params):
        super().__init__(**params)
        p = self._params
        if int(p["moving_avg"]) < 1 or int(p["moving_avg"]) % 2 == 0:
            raise ValueError("moving_avg must be a positive odd integer")
        if p["channels"] not in ("target", "all"):
            raise ValueError("channels must be 'target' or 'all'")
        if p["individual"] and p["channels"] == "target":
            raise ValueError("individual=True needs channels='all'")

    def build_network(self, spec: NetworkSpec):
        import torch
        import torch.nn as nn

        p = {**self.network_defaults, **spec.params}
        window, mode, individual = int(p["moving_avg"]), p["channels"], bool(p["individual"])
        L, H, F = spec.L, spec.H, spec.F
        if mode == "target" and spec.target_input_index is None:
            raise ValueError("dlinear with channels='target' needs the task to define target_input")
        target_index = spec.target_input_index

        class SeriesDecomp(nn.Module):
            """Moving-average decomposition of [B, C, L] into (seasonal, trend)."""

            def __init__(self):
                super().__init__()
                self.pad = (window - 1) // 2
                self.avg = nn.AvgPool1d(kernel_size=window, stride=1, padding=0)

            def forward(self, x):
                if self.pad:
                    x_pad = torch.cat([x[:, :, :1].expand(-1, -1, self.pad), x,
                                       x[:, :, -1:].expand(-1, -1, self.pad)], dim=2)
                else:
                    x_pad = x
                trend = self.avg(x_pad)
                return x - trend, trend

        def linear():
            layer = nn.Linear(L, H)
            nn.init.constant_(layer.weight, 1.0 / L)
            return layer

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.decomp = SeriesDecomp()
                channels = 1 if mode == "target" else F
                if individual:
                    self.linear_seasonal = nn.ModuleList([linear() for _ in range(channels)])
                    self.linear_trend = nn.ModuleList([linear() for _ in range(channels)])
                else:
                    self.linear_seasonal, self.linear_trend = linear(), linear()
                self.combine = nn.Linear(F, 1) if mode == "all" else None

            def _map(self, layers, x):                 # x [B', C, L] -> [B', C, H]
                if individual:
                    return torch.stack([layer(x[:, c]) for c, layer in enumerate(layers)], dim=1)
                return layers(x)

            def forward(self, x):                      # [B', L, F] -> [B', H]
                x = x.transpose(1, 2)                  # [B', F, L]
                if mode == "target":
                    x = x[:, target_index:target_index + 1]
                seasonal, trend = self.decomp(x)
                y = self._map(self.linear_seasonal, seasonal) + self._map(self.linear_trend, trend)
                if mode == "target":
                    return y[:, 0]
                return self.combine(y.transpose(1, 2)).squeeze(-1)   # [B', H, F] -> [B', H]

        return Network()
