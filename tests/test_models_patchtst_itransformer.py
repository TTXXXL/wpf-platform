"""PatchTST and iTransformer: shapes, a short training run, checkpoints, RevIN shift behaviour."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wpf_benchmark.models import get_model  # noqa: E402
from wpf_benchmark.models.neural.trainer import NetworkSpec, load_checkpoint  # noqa: E402
from wpf_benchmark.runner import (forecast, load_task_config, load_task_data,  # noqa: E402
                                  make_context, seed_all)

NAMES = ["wind_speed", "wind_dir_sin", "wind_dir_cos", "power"]
SMALL = {"d_model": 16, "heads": 2, "layers": 1, "ff": 32}
VARIANTS = [("patchtst", {"head": "channel"}), ("patchtst", {"head": "joint"}),
            ("patchtst", {"head": "channel", "revin_affine": True}), ("itransformer", {})]
IDS = ["patchtst-channel", "patchtst-joint", "patchtst-affine", "itransformer"]


def net_params(model):
    return {k: v for k, v in model.params.items() if k in type(model).network_defaults}


def make_spec(model, L=48, H=6, target=3):
    return NetworkSpec(L, H, len(NAMES), 1, tuple(NAMES), target, 0.0, net_params(model))


def make_network(name, extra, L=48, H=6):
    model = get_model(name)(**SMALL, **extra)
    return model.build_network(make_spec(model, L, H)).eval(), L, H


@pytest.mark.parametrize("name,extra", VARIANTS, ids=IDS)
def test_output_shape(name, extra):
    net, L, H = make_network(name, extra)
    out = net(torch.randn(5, L, len(NAMES)))
    assert out.shape == (5, H) and torch.isfinite(out).all()


@pytest.mark.parametrize("name,extra", VARIANTS, ids=IDS)
def test_shift_invariance(name, extra):
    """Shifting every input by a constant shifts the target forecast by the same constant."""
    net, L, _ = make_network(name, extra)
    x = torch.randn(4, L, len(NAMES))
    shift = 3.5
    with torch.no_grad():
        base, shifted = net(x), net(x + shift)
    assert torch.allclose(shifted - base, torch.full_like(base, shift), atol=1e-4)


def test_invalid_configuration():
    with pytest.raises(ValueError):
        get_model("patchtst")(d_model=30, heads=4)
    with pytest.raises(ValueError):
        get_model("itransformer")(d_model=30, heads=4)
    with pytest.raises(ValueError):
        get_model("patchtst")(head="other")
    with pytest.raises(ValueError):
        get_model("patchtst")(unknown=1)
    long_patch = get_model("patchtst")(**SMALL, patch=64)
    with pytest.raises(ValueError):
        long_patch.build_network(make_spec(long_patch))
    model = get_model("itransformer")(**SMALL)
    with pytest.raises(ValueError):
        model.build_network(make_spec(model, target=None))


@pytest.mark.parametrize("name,extra", VARIANTS, ids=IDS)
def test_train_predict_checkpoint(name, extra, synthetic_root, tmp_path):
    cfg = load_task_config(synthetic_root, "synthetic_6h")
    data = load_task_data(synthetic_root, cfg)
    ctx = make_context(data)
    seed_all(0)
    model = get_model(name)(**SMALL, **extra, epochs=2, batch=512)
    model.fit(ctx, data.train, data.valid)
    pred = forecast(model, data)
    rated = data.task.rated_kw[None, None, :]
    assert pred.shape == (len(data.test.issues), data.task.H, len(data.test.entities))
    assert np.isfinite(pred).all() and (pred >= 0).all() and (pred <= rated + 1e-3).all()
    assert model.n_params > 0 and model.validation_metrics["MAE_kW"] > 0
    path = tmp_path / "checkpoint.pt"
    model.save_checkpoint(path)
    loaded = load_checkpoint(path, data.task)
    assert type(loaded) is type(model) and loaded.params == model.params
    np.testing.assert_allclose(forecast(loaded, data), pred, atol=1e-3)
