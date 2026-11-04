"""Temporal convolutional network (entity layout).

Follows Bai, Kolter, Koltun (2018), "An Empirical Evaluation of Generic Convolutional and
Recurrent Networks for Sequence Modeling". Each ``TemporalBlock`` has two causal dilated
convolutions (weight-normalised, each followed by Chomp, ReLU and Dropout) and a residual
connection (1x1 convolution when the channel counts differ), followed by ReLU. Dilations are
1, 2, 4, ... one per block. The channel vector at the last time step feeds ``Linear(channels, H)``.

Receptive field. A block with kernel size k and dilation d has two convolutions, so it adds
2 (k - 1) d steps. With dilations 2^0 ... 2^(n-1) over n levels:

    receptive_field = 1 + 2 (k - 1) (2^n - 1)

When ``levels`` is None, n is the smallest integer with ``receptive_field >= L``.

Initialisation: convolution weights are drawn from N(0, 0.01) as in the authors' code. The
draw is made before weight normalisation, so the weight-norm gain ``g`` starts at the norm of
that draw and the effective weight is the N(0, 0.01) sample. (In the authors' code the draw
comes after ``weight_norm`` and is overwritten by the recomputed weight, so it has no effect
there; here it takes effect as documented.)

Code source: written from the paper and the structure of the authors' reference code
(https://github.com/locuslab/TCN, MIT license); layer names follow it, the code is not copied.
"""
from __future__ import annotations

from ..base import register_model
from .trainer import NetworkSpec, NeuralForecaster


def receptive_field(kernel: int, levels: int) -> int:
    return 1 + 2 * (kernel - 1) * (2 ** levels - 1)


def dilated_causal_conv(x, weight, bias, dilation: int):
    """Causal convolution with ``dilation``, computed as an ordinary (dilation 1) convolution.

    x is [B, C, L]. Zero-pad ``(k - 1) * dilation`` on the left, split the padded series into its
    ``dilation`` residue classes (positions j, j + d, j + 2d, ...), move those classes into the batch
    axis, and convolve each with the same kernel. This gives the same values as
    ``Conv1d(dilation=d)`` followed by Chomp, but cuDNN's deterministic mode has no fast algorithm
    for dilated convolutions (about 140 ms instead of 0.7 ms per layer for B=256, C=64, L=144 on an
    RTX 3090), while the folded form runs at the dilation-1 speed.
    """
    import torch.nn.functional as F

    k = weight.shape[-1]
    batch, channels, length = x.shape
    x = F.pad(x, ((k - 1) * dilation, 0))
    x = F.pad(x, (0, (-x.shape[-1]) % dilation))
    steps = x.shape[-1] // dilation
    folded = x.view(batch, channels, steps, dilation).permute(0, 3, 1, 2).reshape(batch * dilation, channels, steps)
    out = F.conv1d(folded, weight, bias)
    out = out.view(batch, dilation, out.shape[1], out.shape[2]).permute(0, 2, 3, 1)
    return out.reshape(batch, out.shape[1], -1)[:, :, :length]


def levels_for(length: int, kernel: int) -> int:
    """Smallest number of levels whose receptive field covers ``length`` steps."""
    levels = 1
    while receptive_field(kernel, levels) < length:
        levels += 1
    return levels


@register_model("tcn")
class TCNForecaster(NeuralForecaster):
    layout = "entity"
    network_defaults = {"channels": 64, "levels": None, "kernel": 3, "dropout": 0.1}

    def __init__(self, **params):
        super().__init__(**params)
        p = self._params
        if int(p["channels"]) <= 0:
            raise ValueError("channels must be positive")
        if int(p["kernel"]) < 2:
            raise ValueError("kernel must be at least 2")
        if p["levels"] is not None and int(p["levels"]) <= 0:
            raise ValueError("levels must be positive or None")
        if not 0 <= float(p["dropout"]) < 1:
            raise ValueError("dropout must be in [0, 1)")

    def build_network(self, spec: NetworkSpec):
        import torch.nn as nn
        from torch.nn.utils.parametrizations import weight_norm

        p = {**self.network_defaults, **spec.params}
        channels, kernel, dropout = int(p["channels"]), int(p["kernel"]), float(p["dropout"])
        levels = levels_for(spec.L, kernel) if p["levels"] is None else int(p["levels"])
        horizon, features = spec.H, spec.F

        class Chomp1d(nn.Module):
            def __init__(self, size):
                super().__init__()
                self.size = size

            def forward(self, x):
                return x[:, :, :-self.size].contiguous()

        def causal_conv(n_in, n_out, dilation):
            conv = nn.Conv1d(n_in, n_out, kernel, padding=(kernel - 1) * dilation, dilation=dilation)
            nn.init.normal_(conv.weight, 0.0, 0.01)       # before weight_norm, see module docstring
            return weight_norm(conv)

        class TemporalBlock(nn.Module):
            def __init__(self, n_in, n_out, dilation):
                super().__init__()
                pad = (kernel - 1) * dilation
                self.conv1, self.chomp1 = causal_conv(n_in, n_out, dilation), Chomp1d(pad)
                self.relu1, self.dropout1 = nn.ReLU(), nn.Dropout(dropout)
                self.conv2, self.chomp2 = causal_conv(n_out, n_out, dilation), Chomp1d(pad)
                self.relu2, self.dropout2 = nn.ReLU(), nn.Dropout(dropout)
                self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                         self.conv2, self.chomp2, self.relu2, self.dropout2)
                self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
                if self.downsample is not None:
                    nn.init.normal_(self.downsample.weight, 0.0, 0.01)
                self.relu = nn.ReLU()
                self.dilation = dilation

            def reference(self, x):
                """The layer stack as the authors wrote it (dilated Conv1d + Chomp); kept for the equivalence test."""
                res = x if self.downsample is None else self.downsample(x)
                return self.relu(self.net(x) + res)

            def forward(self, x):
                res = x if self.downsample is None else self.downsample(x)
                y = self.dropout1(self.relu1(dilated_causal_conv(x, self.conv1.weight, self.conv1.bias, self.dilation)))
                y = self.dropout2(self.relu2(dilated_causal_conv(y, self.conv2.weight, self.conv2.bias, self.dilation)))
                return self.relu(y + res)

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.receptive_field = receptive_field(kernel, levels)
                self.tcn = nn.Sequential(*[TemporalBlock(features if i == 0 else channels, channels, 2 ** i)
                                           for i in range(levels)])
                self.head = nn.Linear(channels, horizon)

            def forward(self, x):                      # [B', L, F] -> [B', H]
                y = self.tcn(x.transpose(1, 2))        # [B', channels, L]
                return self.head(y[:, :, -1])

        return Network()
