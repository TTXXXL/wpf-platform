"""Tests for the lstm, tcn and dlinear forecasters (entity layout)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wpf_benchmark.models import get_model  # noqa: E402
from wpf_benchmark.models.neural.tcn import levels_for, receptive_field  # noqa: E402
from wpf_benchmark.models.neural.trainer import TRAIN_DEFAULTS, NetworkSpec, load_checkpoint  # noqa: E402
from wpf_benchmark.runner import load_task_config, load_task_data, make_context, seed_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("wind_speed", "wind_dir_sin", "wind_dir_cos", "power")

# (id, model name, small parameters used to keep the tests fast)
CASES = [
    ("lstm", "lstm", {"hidden": 16, "layers": 2}),
    ("lstm_one_layer", "lstm", {"hidden": 16, "layers": 1}),
    ("tcn", "tcn", {"channels": 8}),
    ("tcn_fixed_levels", "tcn", {"channels": 8, "levels": 3, "kernel": 2}),
    ("dlinear", "dlinear", {}),
    ("dlinear_all", "dlinear", {"channels": "all"}),
    ("dlinear_all_individual", "dlinear", {"channels": "all", "individual": True}),
]
CASE_IDS = [c[0] for c in CASES]


def make_spec(model, L=48, H=6, F=4, target_index=3):
    network = {k: v for k, v in model.params.items() if k not in TRAIN_DEFAULTS}
    return NetworkSpec(L, H, F, 1, NAMES[:F], target_index, 0.0, network)


@pytest.fixture(scope="module")
def data(synthetic_root):
    cfg = load_task_config(synthetic_root, "synthetic_6h")
    return load_task_data(synthetic_root, cfg)


# ----- 1. output shapes -----
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_network_output_shape(case):
    _, name, params = case
    model = get_model(name)(**params)
    assert model.layout == "entity"
    net = model.build_network(make_spec(model, L=48, H=6, F=4)).eval()
    x = torch.randn(5, 48, 4)
    with torch.no_grad():
        out = net(x)
    assert out.shape == (5, 6)
    assert torch.isfinite(out).all()


def test_network_uses_spec_feature_count():
    for F, target in ((2, 1), (6, 5)):
        for name, params in (("lstm", {"hidden": 8}), ("tcn", {"channels": 8}), ("dlinear", {})):
            model = get_model(name)(**params)
            net = model.build_network(make_spec(model, L=30, H=4, F=F, target_index=target)).eval()
            assert net(torch.randn(3, 30, F)).shape == (3, 4)


# ----- 2. training on the synthetic task -----
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_train_and_predict(case, data):
    _, name, params = case
    ctx = make_context(data)
    seed_all(0)
    model = get_model(name)(epochs=2, **params)
    model.fit(ctx, data.train, data.valid)
    assert model.n_params > 0
    assert model.validation_metrics["best_epoch"] in (1, 2)
    batch = data.test.batch(data.test.issues[:20])
    pred = model.predict(batch)
    assert pred.shape == (20, data.task.H, len(data.test.entities))
    assert pred.dtype == np.float32
    assert np.isfinite(pred).all()
    assert pred.min() >= 0
    assert (pred <= data.task.rated_kw[None, None, :]).all()


# ----- 3. checkpoint round trip -----
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_checkpoint_round_trip(case, data, tmp_path):
    _, name, params = case
    ctx = make_context(data)
    seed_all(1)
    model = get_model(name)(epochs=2, **params)
    model.fit(ctx, data.train, data.valid)
    path = tmp_path / "checkpoint.pt"
    model.save_checkpoint(path)
    loaded = load_checkpoint(path, data.task)
    assert loaded.name == name
    assert loaded.params == model.params
    assert loaded.n_params == model.n_params
    batch = data.test.batch(data.test.issues[:20])
    np.testing.assert_allclose(loaded.predict(batch), model.predict(batch), rtol=1e-5, atol=1e-3)


# ----- model specifics -----
def test_tcn_receptive_field_and_levels():
    assert receptive_field(3, 6) == 253 and receptive_field(3, 5) == 125
    assert levels_for(144, 3) == 6                       # 125 < 144 <= 253
    assert levels_for(125, 3) == 5 and levels_for(126, 3) == 6
    assert levels_for(1, 3) == 1
    model = get_model("tcn")()
    net = model.build_network(make_spec(model, L=144, H=36))
    assert net.receptive_field == 253
    assert len(net.tcn) == 6
    explicit = get_model("tcn")(levels=2)
    assert explicit.build_network(make_spec(explicit, L=144, H=36)).receptive_field == 1 + 2 * 2 * 3


def test_tcn_is_causal():
    model = get_model("tcn")(channels=8)
    net = model.build_network(make_spec(model, L=40, H=3)).eval()
    x = torch.randn(2, 4, 40)
    changed = x.clone()
    changed[:, :, 25:] += torch.randn(2, 4, 15)
    with torch.no_grad():
        a, b = net.tcn(x), net.tcn(changed)
    assert torch.allclose(a[:, :, :25], b[:, :, :25], atol=1e-6)
    assert not torch.allclose(a[:, :, 25:], b[:, :, 25:])


def test_dlinear_decomposition():
    model = get_model("dlinear")(moving_avg=5)
    net = model.build_network(make_spec(model, L=20, H=3)).eval()
    x = torch.randn(4, 1, 20)
    seasonal, trend = net.decomp(x)
    assert seasonal.shape == trend.shape == x.shape
    assert torch.allclose(seasonal + trend, x, atol=1e-6)
    seasonal, trend = net.decomp(torch.full((2, 1, 20), 3.0))   # edge replication keeps constants unchanged
    assert torch.allclose(trend, torch.full((2, 1, 20), 3.0), atol=1e-6)
    assert torch.allclose(seasonal, torch.zeros(2, 1, 20), atol=1e-6)
    first, _ = net.decomp(torch.arange(20.0).reshape(1, 1, 20))
    assert first.shape == (1, 1, 20)


def test_dlinear_target_mode_ignores_other_channels():
    model = get_model("dlinear")()
    net = model.build_network(make_spec(model, L=48, H=6, target_index=3)).eval()
    x = torch.randn(3, 48, 4)
    other = x.clone()
    other[:, :, :3] += 5.0
    with torch.no_grad():
        assert torch.allclose(net(x), net(other))
        moved = x.clone()
        moved[:, :, 3] += 1.0
        assert not torch.allclose(net(x), net(moved))


def test_dlinear_target_mode_needs_target_input():
    model = get_model("dlinear")()
    with pytest.raises(ValueError, match="target_input"):
        model.build_network(make_spec(model, target_index=None))
    allc = get_model("dlinear")(channels="all")
    allc.build_network(make_spec(allc, target_index=None))      # "all" mode does not need it


# ----- parameter validation and registry -----
@pytest.mark.parametrize("name,params", [
    ("lstm", {"hidden": 0}), ("lstm", {"layers": 0}), ("lstm", {"dropout": 1.0}), ("lstm", {"width": 3}),
    ("tcn", {"kernel": 1}), ("tcn", {"levels": 0}), ("tcn", {"channels": 0}), ("tcn", {"dropout": -0.1}),
    ("tcn", {"hidden": 3}),
    ("dlinear", {"moving_avg": 24}), ("dlinear", {"channels": "some"}), ("dlinear", {"individual": True}),
    ("dlinear", {"kernel": 3}),
])
def test_invalid_parameters_rejected(name, params):
    with pytest.raises(ValueError):
        get_model(name)(**params)


def test_default_params_include_training_defaults():
    assert get_model("lstm")().params == {**TRAIN_DEFAULTS, "hidden": 128, "layers": 2, "dropout": 0.1}
    assert get_model("tcn")().params == {**TRAIN_DEFAULTS, "channels": 64, "levels": None, "kernel": 3,
                                         "dropout": 0.1}
    assert get_model("dlinear")().params == {**TRAIN_DEFAULTS, "moving_avg": 25, "channels": "target",
                                             "individual": False}


def test_model_modules_do_not_import_torch_at_import_time():
    code = ("import sys; import wpf_benchmark.models.neural.lstm, wpf_benchmark.models.neural.tcn, "
            "wpf_benchmark.models.neural.dlinear; assert 'torch' not in sys.modules, 'torch imported'")
    env = {"PYTHONPATH": str(ROOT / "src"), "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT)
    assert result.returncode == 0, result.stderr
