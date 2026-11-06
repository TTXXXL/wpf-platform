"""The folded dilated convolution must equal the authors' Conv1d + Chomp form, outputs and gradients."""
from __future__ import annotations

import pytest
import torch

from wpf_benchmark.models.neural.tcn import TCNForecaster, dilated_causal_conv
from wpf_benchmark.models.neural.trainer import NetworkSpec


@pytest.mark.parametrize("dilation", [1, 2, 3, 4, 8, 16])
@pytest.mark.parametrize("length", [37, 144])
def test_fold_equals_dilated_conv(dilation, length):
    torch.manual_seed(0)
    conv = torch.nn.Conv1d(5, 7, 3, padding=2 * dilation, dilation=dilation)
    x = torch.randn(4, 5, length, requires_grad=True)
    expected = conv(x)[:, :, :length]
    got = dilated_causal_conv(x, conv.weight, conv.bias, dilation)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-6)
    g_expected = torch.autograd.grad(expected.square().sum(), [x, conv.weight], retain_graph=True)
    g_got = torch.autograd.grad(got.square().sum(), [x, conv.weight])
    for a, b in zip(g_expected, g_got):
        assert torch.allclose(a, b, atol=1e-4)


def _network(L=100, kernel=3):
    spec = NetworkSpec(L, 6, 4, 3, ("a", "b", "c", "d"), 0, 0.0, {})
    return TCNForecaster(channels=8, kernel=kernel, dropout=0.3).build_network(spec)


@pytest.mark.parametrize("training", [False, True])
def test_block_forward_equals_reference(training):
    torch.manual_seed(1)
    net = _network().train(training)
    x = torch.randn(6, 4, 100)
    for block in net.tcn:
        torch.manual_seed(5)
        want = block.reference(x)
        torch.manual_seed(5)                         # same dropout draws: same shapes, same order
        got = block(x)
        assert torch.allclose(got, want, atol=1e-6)
        x = want.detach()


def test_whole_network_gradients_match_reference():
    torch.manual_seed(2)
    net = _network(L=60, kernel=4).eval()
    x = torch.randn(5, 60, 4)
    out = net(x)
    grads = torch.autograd.grad(out.square().sum(), list(net.parameters()))
    h = x.transpose(1, 2)
    for block in net.tcn:
        h = block.reference(h)
    ref = net.head(h[:, :, -1])
    ref_grads = torch.autograd.grad(ref.square().sum(), list(net.parameters()))
    assert torch.allclose(out, ref, atol=1e-6)
    for a, b in zip(grads, ref_grads):
        assert torch.allclose(a, b, atol=1e-4)


def test_state_dict_keys_are_unchanged():
    keys = list(_network().state_dict())
    assert any(k.startswith("tcn.0.conv1.parametrizations.weight") for k in keys)
    assert all("dilation" not in k for k in keys)
