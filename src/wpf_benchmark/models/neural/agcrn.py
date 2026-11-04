"""AGCRN (Bai et al., NeurIPS 2020) on the platform's graph layout.

The network follows the authors' MIT-licensed implementation
(https://github.com/LeiBAI/AGCRN/tree/master/model) as ported in the legacy SDWPF code
(``agcrn_original.py``). Graph supports and node-specific kernels are computed once per forward
pass, which does not change the graph-convolution equations.

Platform changes: the network takes ``[B, L, N, F]`` and returns ``[B, H, N]``. ``forward``
transposes to the legacy ``(B, N, F, W)`` layout on entry and back to ``[B, H, N]`` on exit; the
computation in between is unchanged, and so are submodule and parameter names, so a legacy
``state_dict`` of the same shapes loads directly.

``current_power_skip=True`` adds the latest finite scaled target-input value to the head output
(a zero-initialized head then starts at persistence). The anchor must see missing values, so
this setting makes the model request ``raw_inputs`` and the network zero-fills NaN itself.

Nothing here imports torch at module level; the network classes are built on first use.
"""
from __future__ import annotations

from functools import lru_cache

from ..base import register_model
from .trainer import NetworkSpec, NeuralForecaster


def latest_history_power(histories, power_index: int, fallback: float = 0.0):
    """Latest finite scaled target-input value of each turbine, shape (B, N, 1).

    ``histories`` is (B, N, F, W) with NaN for missing cells. A turbine with no finite value in
    the window gets ``fallback`` (the scaled value of 0 kW). The anchor is chosen before NaN is
    zero-filled, so no zero-filled cell and no future value can become the anchor.
    """
    import torch
    power = histories[:, :, power_index, :]
    finite = torch.isfinite(power)
    ticks = torch.arange(power.shape[-1], device=power.device)
    last = torch.where(finite, ticks, ticks.new_full((), -1)).max(dim=-1).values
    safe_power = torch.where(finite, power, torch.zeros_like(power))
    selected = safe_power.gather(-1, last.clamp(min=0).unsqueeze(-1))
    return torch.where(last.unsqueeze(-1) >= 0, selected, torch.full_like(selected, fallback))


@lru_cache(maxsize=None)
def _network_classes():
    """Define the torch modules on first use (keeps ``import torch`` out of module level)."""
    import torch
    from torch import nn

    class AdaptiveGraphConvolution(nn.Module):
        """Node-adaptive parameter learning over Chebyshev graph supports."""

        def __init__(self, input_size: int, output_size: int, embedding_size: int, cheb_k: int):
            super().__init__()
            self.weight_pool = nn.Parameter(torch.empty(embedding_size, cheb_k, input_size, output_size))
            self.bias_pool = nn.Parameter(torch.empty(embedding_size, output_size))

        def project(self, embeddings):
            weights = torch.einsum("nd,dkio->nkio", embeddings, self.weight_pool)
            bias = embeddings @ self.bias_pool
            return weights, bias

        def forward(self, values, supports, projected):
            weights, bias = projected
            neighbors = torch.einsum("knm,bmc->bknc", supports, values)
            return torch.einsum("bknc,nkco->bno", neighbors, weights) + bias

    class AdaptiveGraphCell(nn.Module):
        """GRU gates and candidate built from node-adaptive graph convolutions."""

        def __init__(self, input_size: int, hidden: int, embedding_size: int, cheb_k: int):
            super().__init__()
            self.hidden = hidden
            self.gate = AdaptiveGraphConvolution(input_size + hidden, 2 * hidden, embedding_size, cheb_k)
            self.candidate = AdaptiveGraphConvolution(input_size + hidden, hidden, embedding_size, cheb_k)

        def forward(self, value, state, supports, gate_params, candidate_params):
            gates = self.gate(torch.cat((value, state), dim=-1), supports, gate_params).sigmoid()
            reset, update = gates.chunk(2, dim=-1)
            proposal = self.candidate(torch.cat((value, reset * state), dim=-1),
                                      supports, candidate_params).tanh()
            return update * state + (1 - update) * proposal

    class PaperAGCRNNetwork(nn.Module):
        """Full-resolution AGCRN encoder with the paper's direct forecast head."""

        def __init__(self, n_turbines: int, input_size: int, hidden: int, layers: int,
                     embedding_size: int, cheb_k: int, horizon: int,
                     current_power_skip: bool = False, power_index: int = 3,
                     power_fallback: float = 0.0):
            super().__init__()
            if min(n_turbines, input_size, hidden, layers, embedding_size, horizon) <= 0:
                raise ValueError("AGCRN dimensions must be positive")
            if cheb_k < 2:
                raise ValueError("cheb_k must be at least 2")
            if current_power_skip and not 0 <= power_index < input_size:
                raise ValueError("power_index must identify an input feature")
            self.n_turbines = n_turbines
            self.input_size = input_size
            self.hidden = hidden
            self.cheb_k = cheb_k
            self.current_power_skip = current_power_skip
            self.power_index = power_index
            self.power_fallback = power_fallback
            self.node_embeddings = nn.Parameter(torch.empty(n_turbines, embedding_size))
            self.cells = nn.ModuleList([
                AdaptiveGraphCell(input_size if layer == 0 else hidden, hidden, embedding_size, cheb_k)
                for layer in range(layers)
            ])
            self.end_conv = nn.Conv2d(1, horizon, kernel_size=(1, hidden))
            self.reset_parameters()
            if self.current_power_skip:
                # Start at persistence; the signed head learns departures from it.
                nn.init.zeros_(self.end_conv.weight)
                nn.init.zeros_(self.end_conv.bias)

        def reset_parameters(self):
            # Match the initialization pass in the authors' Run.py.
            for parameter in self.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
                else:
                    nn.init.uniform_(parameter)

        def adaptive_adjacency(self):
            scores = self.node_embeddings @ self.node_embeddings.transpose(0, 1)
            return torch.softmax(torch.relu(scores), dim=-1)

        def graph_supports(self):
            adjacency = self.adaptive_adjacency()
            identity = torch.eye(self.n_turbines, dtype=adjacency.dtype, device=adjacency.device)
            supports = [identity, adjacency]
            for _ in range(2, self.cheb_k):
                supports.append(2 * adjacency @ supports[-1] - supports[-2])
            return torch.stack(supports, dim=0)

        def forward(self, x):
            # Platform layout [B, L, N, F] -> legacy (B, N, F, W).
            if x.ndim != 4 or x.shape[2] != self.n_turbines or x.shape[3] != self.input_size:
                raise ValueError("Expected AGCRN input with shape [B, L, N, F]")
            x = x.permute(0, 2, 3, 1)
            if self.current_power_skip:
                anchor = latest_history_power(x, self.power_index, self.power_fallback)
                x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
            sequence = x.permute(0, 3, 1, 2)
            supports = self.graph_supports()
            for cell in self.cells:
                gate_params = cell.gate.project(self.node_embeddings)
                candidate_params = cell.candidate.project(self.node_embeddings)
                state = sequence.new_zeros(x.shape[0], self.n_turbines, self.hidden)
                outputs = []
                for value in sequence.unbind(dim=1):
                    state = cell(value, state, supports, gate_params, candidate_params)
                    outputs.append(state)
                sequence = torch.stack(outputs, dim=1)
            # The authors' 1 x hidden Conv2d emits every horizon directly.
            power = self.end_conv(sequence[:, -1].unsqueeze(1)).squeeze(-1).transpose(1, 2)   # (B, N, H)
            if self.current_power_skip:
                power = power + anchor
            return power.transpose(1, 2)                                                       # [B, H, N]

    return AdaptiveGraphConvolution, AdaptiveGraphCell, PaperAGCRNNetwork


@register_model("agcrn")
class AGCRNForecaster(NeuralForecaster):
    """NAPL + DAGG graph recurrence and a direct multi-horizon forecast head."""

    layout = "graph"
    network_defaults = {"hidden": 64, "layers": 2, "emb": 10, "cheb_k": 2, "current_power_skip": False,
                        "loss": "mae", "batch": 16}

    def __init__(self, **params):
        super().__init__(**params)
        p = self._params
        if not isinstance(p["current_power_skip"], bool):
            raise ValueError("current_power_skip must be a boolean")
        if min(p["hidden"], p["layers"], p["emb"]) <= 0 or p["cheb_k"] < 2:
            raise ValueError("hidden, layers and emb must be positive and cheb_k at least 2")
        # The latest-valid-power anchor needs to see missing values.
        self.raw_inputs = p["current_power_skip"]

    def build_network(self, spec: NetworkSpec):
        q = spec.params
        if q["current_power_skip"] and spec.target_input_index is None:
            raise ValueError("agcrn current_power_skip needs a task target_input")
        network_class = _network_classes()[2]
        return network_class(spec.N, spec.F, int(q["hidden"]), int(q["layers"]), int(q["emb"]),
                             int(q["cheb_k"]), spec.H, current_power_skip=q["current_power_skip"],
                             power_index=spec.target_input_index if spec.target_input_index is not None else 0,
                             power_fallback=spec.target_zero)
