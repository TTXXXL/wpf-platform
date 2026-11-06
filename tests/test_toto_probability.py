"""Toto adapter contracts exercised without optional packages or pretrained weights."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from wpf_benchmark.models.toto_quantile import build_toto_q_network, pack_toto
from wpf_benchmark.models.neural.trainer import NetworkSpec
from wpf_benchmark.probability import torch_pinball


class TinyBackbone(nn.Module):
    """Test double for output and freezing contracts; not a pretrained-model validation."""
    def __init__(self):
        super().__init__()
        self.patch_embed = SimpleNamespace(stride=4, patch_size=4)
        self.embed_dim = 8
        self.projection = nn.Linear(4, 8)
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(6)])
        self.output_distribution = nn.Linear(8, 4)

    def backbone(self, series, observed, ids, **kwargs):
        out = self.projection(series.unfold(-1, 4, 4))
        for layer in self.transformer.layers:
            out = layer(out).tanh()
        return out.repeat_interleave(4, dim=2), None, None


def test_pack_padding_group_and_nan():
    x = torch.arange(1*6*2*4, dtype=torch.float32).reshape(1,6,2,4)
    x[0,-1,0,0] = float("nan")
    series, observed, ids = pack_toto(x, 4)
    assert series.shape == observed.shape == ids.shape == (1,8,8)
    assert not observed[...,:2].any()
    assert series[0,0,-1] == 0 and not observed[0,0,-1]
    assert ids[0,:,0].tolist() == [0]*4+[1]*4
    assert torch.all(ids == ids[...,:1])


@pytest.mark.parametrize("mode", ["head", "last4", "full"])
def test_toto_q_freezing_forward_backward_and_state(mode):
    spec = NetworkSpec(8,3,4,2,("ws","sin","cos","power"),3,0,{})
    network = build_toto_q_network(TinyBackbone(),spec,mode)
    trainable = {n for n,p in network.named_parameters() if p.requires_grad}
    expected = {"quantile_head.weight","quantile_head.bias"}
    if mode == "last4":
        expected |= {f"backbone.transformer.layers.{k}.{p}" for k in range(2,6) for p in ("weight","bias")}
    if mode == "full":
        expected |= {n for n,_ in network.named_parameters() if n.startswith("backbone.") and "output_distribution" not in n}
    assert trainable == expected
    x = torch.rand(2,8,2,4)
    x[0,-1,0,0] = float("nan")
    output = network(x)
    assert output.shape == (2,3,2,9)
    assert torch.isfinite(output).all() and torch.all(output.diff(dim=-1) >= 0)
    loss = torch_pinball(output,torch.rand(2,3,2),torch.ones((2,3,2),dtype=torch.bool))
    loss.backward()
    assert all(p.grad is None for n,p in network.named_parameters() if n not in trainable)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in network.named_parameters() if n in trainable)
    second = build_toto_q_network(TinyBackbone(),spec,mode)
    second.load_state_dict(network.state_dict())
    network.eval()
    second.eval()
    torch.testing.assert_close(network(x),second(x),rtol=0,atol=0)
