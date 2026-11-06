"""AGCRN fast forward: same outputs and gradients as the legacy per-step loop, same state_dict layout.

``Reference*`` below is the network as it was before the forward was rearranged (one concatenation,
two einsum graph convolutions and four elementwise steps per time step). The platform network must
agree with it on outputs and on the gradient of every parameter, with and without the
current-power skip, and a checkpoint written by either must load into the other.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from wpf_benchmark.models.neural.agcrn import _network_classes, latest_history_power  # noqa: E402


class ReferenceConvolution(nn.Module):
    def __init__(self, input_size, output_size, embedding_size, cheb_k):
        super().__init__()
        self.weight_pool = nn.Parameter(torch.empty(embedding_size, cheb_k, input_size, output_size))
        self.bias_pool = nn.Parameter(torch.empty(embedding_size, output_size))

    def project(self, embeddings):
        return torch.einsum("nd,dkio->nkio", embeddings, self.weight_pool), embeddings @ self.bias_pool

    def forward(self, values, supports, projected):
        weights, bias = projected
        neighbors = torch.einsum("knm,bmc->bknc", supports, values)
        return torch.einsum("bknc,nkco->bno", neighbors, weights) + bias


class ReferenceCell(nn.Module):
    def __init__(self, input_size, hidden, embedding_size, cheb_k):
        super().__init__()
        self.hidden = hidden
        self.gate = ReferenceConvolution(input_size + hidden, 2 * hidden, embedding_size, cheb_k)
        self.candidate = ReferenceConvolution(input_size + hidden, hidden, embedding_size, cheb_k)

    def forward(self, value, state, supports, gate_params, candidate_params):
        gates = self.gate(torch.cat((value, state), dim=-1), supports, gate_params).sigmoid()
        reset, update = gates.chunk(2, dim=-1)
        proposal = self.candidate(torch.cat((value, reset * state), dim=-1), supports, candidate_params).tanh()
        return update * state + (1 - update) * proposal


class ReferenceNetwork(nn.Module):
    def __init__(self, n_turbines, input_size, hidden, layers, embedding_size, cheb_k, horizon,
                 current_power_skip=False, power_index=3, power_fallback=0.0):
        super().__init__()
        self.n_turbines, self.hidden, self.cheb_k = n_turbines, hidden, cheb_k
        self.current_power_skip, self.power_index, self.power_fallback = current_power_skip, power_index, power_fallback
        self.node_embeddings = nn.Parameter(torch.empty(n_turbines, embedding_size))
        self.cells = nn.ModuleList([ReferenceCell(input_size if i == 0 else hidden, hidden, embedding_size, cheb_k)
                                    for i in range(layers)])
        self.end_conv = nn.Conv2d(1, horizon, kernel_size=(1, hidden))

    def graph_supports(self):
        adjacency = torch.softmax(torch.relu(self.node_embeddings @ self.node_embeddings.transpose(0, 1)), dim=-1)
        identity = torch.eye(self.n_turbines, dtype=adjacency.dtype, device=adjacency.device)
        supports = [identity, adjacency]
        for _ in range(2, self.cheb_k):
            supports.append(2 * adjacency @ supports[-1] - supports[-2])
        return torch.stack(supports, dim=0)

    def forward(self, x):
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
        power = self.end_conv(sequence[:, -1].unsqueeze(1)).squeeze(-1).transpose(1, 2)
        if self.current_power_skip:
            power = power + anchor
        return power.transpose(1, 2)


def _pair(layers, cheb_k, skip, dtype, N=5, F=4, hidden=6, emb=3, H=4, seed=0):
    """A platform network and a reference network holding identical weights (random head included)."""
    torch.manual_seed(seed)
    fast = _network_classes()[2](N, F, hidden, layers, emb, cheb_k, H, current_power_skip=skip, power_index=3,
                                 power_fallback=0.07)
    with torch.no_grad():                                     # a zero head would hide errors in the cells
        nn.init.normal_(fast.end_conv.weight, std=0.3)
        nn.init.normal_(fast.end_conv.bias, std=0.1)
    reference = ReferenceNetwork(N, F, hidden, layers, emb, cheb_k, H, current_power_skip=skip, power_index=3,
                                 power_fallback=0.07)
    reference.load_state_dict(fast.state_dict(), strict=True)
    return fast.to(dtype), reference.to(dtype)


def _inputs(B, L, N, F, dtype, skip, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((B, L, N, F))
    if skip:                                                  # missing cells, including a turbine with no power at all
        x[rng.random(x.shape) < 0.15] = np.nan
        x[0, :, 1, 3] = np.nan
        x[1, -3:, 2, 3] = np.nan
    return torch.tensor(x, dtype=dtype)


def _gradients(network, x):
    network.zero_grad()
    out = network(x)
    weight = torch.linspace(0.5, 1.5, out.numel(), dtype=out.dtype).view_as(out)
    (out * weight).sum().backward()
    return out.detach(), {name: p.grad.clone() for name, p in network.named_parameters()}


@pytest.mark.parametrize("layers", [1, 2, 3])
@pytest.mark.parametrize("cheb_k", [2, 3])
@pytest.mark.parametrize("skip", [False, True])
def test_fast_forward_matches_reference_float64(layers, cheb_k, skip):
    fast, reference = _pair(layers, cheb_k, skip, torch.float64)
    x = _inputs(3, 9, 5, 4, torch.float64, skip)
    out_fast, grad_fast = _gradients(fast, x)
    out_ref, grad_ref = _gradients(reference, x)
    assert out_fast.shape == (3, 4, 5)
    np.testing.assert_allclose(out_fast.numpy(), out_ref.numpy(), rtol=1e-10, atol=1e-12)
    assert grad_fast.keys() == grad_ref.keys()
    for name in grad_ref:
        assert grad_ref[name].abs().max() > 0, f"{name} has no gradient, the test would prove nothing"
        np.testing.assert_allclose(grad_fast[name].numpy(), grad_ref[name].numpy(), rtol=1e-8, atol=1e-11,
                                   err_msg=name)


@pytest.mark.parametrize("skip", [False, True])
def test_fast_forward_matches_reference_float32(skip):
    fast, reference = _pair(2, 2, skip, torch.float32, N=7, hidden=16, emb=5, H=6)
    x = _inputs(4, 30, 7, 4, torch.float32, skip)
    out_fast, grad_fast = _gradients(fast, x)
    out_ref, grad_ref = _gradients(reference, x)
    np.testing.assert_allclose(out_fast.numpy(), out_ref.numpy(), rtol=1e-4, atol=2e-5)
    for name in grad_ref:
        scale = float(grad_ref[name].abs().max())
        np.testing.assert_allclose(grad_fast[name].numpy(), grad_ref[name].numpy(), rtol=1e-3, atol=1e-4 * scale,
                                   err_msg=name)


def test_fast_forward_is_repeatable_and_batch_consistent():
    fast, _ = _pair(2, 2, False, torch.float32)
    x = _inputs(4, 12, 5, 4, torch.float32, False)
    with torch.no_grad():
        first, second = fast(x), fast(x)
        single = torch.cat([fast(x[i:i + 1]) for i in range(4)])
    assert torch.equal(first, second)
    np.testing.assert_allclose(first.numpy(), single.numpy(), rtol=1e-5, atol=1e-6)


def test_state_dict_layout_is_unchanged():
    fast, reference = _pair(2, 2, False, torch.float32)
    assert list(fast.state_dict()) == list(reference.state_dict())
    for key, value in reference.state_dict().items():
        assert fast.state_dict()[key].shape == value.shape
    reference.load_state_dict(fast.state_dict(), strict=True)       # and the other way round
    fast.load_state_dict(reference.state_dict(), strict=True)


def test_skip_without_training_signal_is_persistence():
    """A zero-initialized head starts at the latest valid power, whatever the cells compute."""
    torch.manual_seed(0)
    network = _network_classes()[2](3, 4, 8, 2, 4, 2, 5, current_power_skip=True, power_index=3, power_fallback=0.07)
    x = _inputs(2, 12, 3, 4, torch.float32, True)
    with torch.no_grad():
        out = network(x)
    anchor = latest_history_power(x.permute(0, 2, 3, 1), 3, 0.07)
    np.testing.assert_allclose(out.numpy(), anchor.squeeze(-1)[:, None, :].expand(2, 5, 3).numpy(), atol=1e-6)
