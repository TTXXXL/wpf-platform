"""LSTM encoder with an MLP multi-step head (entity layout).

Port of the legacy ``lstm_seq2seq`` model. The history window [B', L, F] goes through a
stacked ``nn.LSTM``; the hidden state of the last layer at the last time step is mapped to
the H-step forecast by ``Linear(hidden, hidden) -> ReLU -> Linear(hidden, H)``.

Differences from the legacy code: the input channel count is ``spec.F`` instead of a fixed 4,
and the input is already laid out as [B', L, F], so no transpose is needed. Submodule names
(``encoder``, ``decoder``) are kept so that legacy weights can be compared later.

Code source: written for this platform by porting the legacy file; not taken from an
external repository.
"""
from __future__ import annotations

from ..base import register_model
from .trainer import NetworkSpec, NeuralForecaster


@register_model("lstm")
class LSTMForecaster(NeuralForecaster):
    layout = "entity"
    network_defaults = {"hidden": 128, "layers": 2, "dropout": 0.1}

    def __init__(self, **params):
        super().__init__(**params)
        p = self._params
        if int(p["hidden"]) <= 0 or int(p["layers"]) <= 0:
            raise ValueError("hidden and layers must be positive")
        if not 0 <= float(p["dropout"]) < 1:
            raise ValueError("dropout must be in [0, 1)")

    def build_network(self, spec: NetworkSpec):
        import torch.nn as nn
        p = {**self.network_defaults, **spec.params}
        hidden, layers, dropout = int(p["hidden"]), int(p["layers"]), float(p["dropout"])
        horizon, features = spec.H, spec.F

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.LSTM(features, hidden, num_layers=layers,
                                       dropout=dropout if layers > 1 else 0, batch_first=True)
                self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                             nn.Linear(hidden, horizon))

            def forward(self, x):                      # [B', L, F] -> [B', H]
                _, (state, _) = self.encoder(x)
                return self.decoder(state[-1])

        return Network()
