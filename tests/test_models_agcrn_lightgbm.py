"""AGCRN and LightGBM: shapes, anchors, training on the synthetic task, determinism, checkpoints."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightgbm")

from wpf_benchmark.models import get_model  # noqa: E402
from wpf_benchmark.models.neural.agcrn import AGCRNForecaster, latest_history_power  # noqa: E402
from wpf_benchmark.models.neural.trainer import NetworkSpec, load_checkpoint  # noqa: E402
from wpf_benchmark.runner import (forecast, load_task_config, load_task_data, make_context,  # noqa: E402
                                  run_one, seed_all)

SMALL = {"hidden": 8, "emb": 4, "epochs": 2, "batch": 32}


def _spec(skip: bool, target_index=3, zero: float = 0.07, **extra) -> NetworkSpec:
    params = {"hidden": 8, "layers": 2, "emb": 4, "cheb_k": 2, "current_power_skip": skip, **extra}
    return NetworkSpec(L=12, H=5, F=4, N=3, input_names=("ws", "sin", "cos", "power"),
                       target_input_index=target_index, target_zero=zero, params=params)


def test_agcrn_output_shape():
    for skip in (False, True):
        network = AGCRNForecaster(current_power_skip=skip).build_network(_spec(skip))
        x = torch.rand(2, 12, 3, 4)
        assert tuple(network(x).shape) == (2, 5, 3)


def test_agcrn_skip_starts_at_last_valid_power():
    zero = 0.07
    network = AGCRNForecaster(current_power_skip=True).build_network(_spec(True, zero=zero))
    rng = np.random.default_rng(0)
    x = rng.random((4, 12, 3, 4)).astype(np.float32)
    x[:, :, :, 0] = np.nan                       # other features may be missing too
    x[0, 8:, 0, 3] = np.nan                      # last valid power of turbine 0 is at step 7
    x[1, :, 2, 3] = np.nan                       # turbine 2 has no valid power at all
    x[2, -1, 1, 3] = np.nan
    with torch.no_grad():
        out = network(torch.from_numpy(x)).numpy()
    expected = np.empty((4, 3), np.float32)
    for b in range(4):
        for n in range(3):
            finite = np.flatnonzero(np.isfinite(x[b, :, n, 3]))
            expected[b, n] = x[b, finite[-1], n, 3] if len(finite) else zero
    assert expected[0, 0] == x[0, 7, 0, 3] and expected[1, 2] == np.float32(zero)
    assert np.allclose(out, expected[:, None, :].repeat(5, axis=1), atol=1e-6)
    anchor = latest_history_power(torch.from_numpy(x).permute(0, 2, 3, 1), 3, zero)
    assert np.allclose(anchor.squeeze(-1).numpy(), expected, atol=1e-6)


def test_agcrn_skip_requires_target_input():
    with pytest.raises(ValueError, match="target_input"):
        AGCRNForecaster(current_power_skip=True).build_network(_spec(True, target_index=None))
    assert AGCRNForecaster(current_power_skip=True).raw_inputs is True
    assert AGCRNForecaster().raw_inputs is False
    with pytest.raises(ValueError):
        AGCRNForecaster(unknown=1)


def test_agcrn_state_dict_names_match_legacy():
    keys = set(AGCRNForecaster().build_network(_spec(False)).state_dict())
    assert {"node_embeddings", "end_conv.weight", "end_conv.bias", "cells.0.gate.weight_pool",
            "cells.0.gate.bias_pool", "cells.1.candidate.weight_pool", "cells.1.candidate.bias_pool"} <= keys


@pytest.mark.parametrize("skip", [False, True])
def test_agcrn_trains_and_checkpoint_roundtrip(synthetic_root, skip, tmp_path):
    cfg = load_task_config(synthetic_root, "synthetic_6h")
    data = load_task_data(synthetic_root, cfg)
    ctx = make_context(data)
    seed_all(0)
    model = get_model("agcrn")(current_power_skip=skip, **SMALL)
    model.fit(ctx, data.train, data.valid)
    assert model.n_params > 0 and model.validation_metrics["MAE_kW"] > 0
    batch = data.test.batch(data.test.issues[:6])
    before = model.predict(batch)
    assert before.shape == (6, ctx.H, len(ctx.entities)) and np.isfinite(before).all()
    assert (before >= 0).all() and (before <= ctx.rated_kw[None, None, :]).all()
    model.save_checkpoint(tmp_path / "checkpoint.pt")
    loaded = load_checkpoint(tmp_path / "checkpoint.pt", data.task)
    assert loaded.raw_inputs is skip
    assert np.array_equal(loaded.predict(batch), before)


def test_agcrn_run_one_smoke(synthetic_root):
    directory, run = run_one(synthetic_root, "synthetic_6h", {"model": "agcrn", "params": SMALL}, seed=0)
    assert (directory / "checkpoint.pt").is_file() and np.isfinite(run["summary"]["turbine_MAE_kW"])


# ----- LightGBM -----
LGB = {"n_estimators": 20, "n_jobs": 2}


def _fit_lightgbm(synthetic_root, **params):
    data = load_task_data(synthetic_root, load_task_config(synthetic_root, "synthetic_6h"))
    ctx = make_context(data)
    model = get_model("lightgbm")(**{**LGB, **params})
    model.fit(ctx, data.train, data.valid)
    return model, data, ctx


def test_lightgbm_predicts_in_range(synthetic_root):
    model, data, ctx = _fit_lightgbm(synthetic_root)
    pred = forecast(model, data)
    assert pred.shape == (len(data.test.issues), ctx.H, len(ctx.entities)) and pred.dtype == np.float32
    assert np.isfinite(pred).all() and (pred >= 0).all() and (pred <= ctx.rated_kw[None, None, :]).all()
    assert model.deterministic and model.n_params > 0 and model.validation_metrics["MAE_kW"] > 0


def test_lightgbm_is_reproducible(synthetic_root):
    first, data, _ = _fit_lightgbm(synthetic_root)
    second, _, _ = _fit_lightgbm(synthetic_root)
    batch = data.test.batch(data.test.issues[:50])
    assert np.array_equal(first.predict(batch), second.predict(batch))
    assert first.n_params == second.n_params


@pytest.mark.parametrize("lags", [6, 0, 3])
def test_lightgbm_feature_count(synthetic_root, lags):
    model, _, ctx = _fit_lightgbm(synthetic_root, lags=lags, n_estimators=2)
    assert model.n_features == 6 * len(ctx.input_names) + lags + 2
    assert model.estimators[0].num_feature() == model.n_features


def test_lightgbm_features_by_hand():
    model = get_model("lightgbm")(lags=2)

    class Ctx:
        resolution_minutes, steps_per_day = 10, 144
    model.ctx, model.target_index = Ctx(), 1
    x = np.zeros((1, 4, 2, 2), np.float32)
    x[0, :, 0, 1] = [1, 2, np.nan, 4]                 # NaN -> 0
    when = np.array(["2020-01-02T06:00:00"], dtype="datetime64[us]")
    f = model._features(x, when)
    assert f.shape == (2, 6 * 2 + 2 + 2)
    row = f[0]                                        # turbine 0
    # blocks of F=2 columns: last, mean, std, min, max, slope; column 1 of each block is the target input
    assert row[1] == 4 and np.isclose(row[3], 1.75) and row[7] == 0 and row[9] == 4
    assert np.isclose(row[5], np.std([1, 2, 0, 4]))
    assert np.isclose(row[11], (-1.5 * 1 - 0.5 * 2 + 0 + 1.5 * 4) / 5)           # slope
    assert list(row[12:14]) == [4, 0]                 # newest lag first, missing -> 0
    assert np.isclose(row[14], 0.25) and row[15] == 0 and f[1, 15] == 1


def test_lightgbm_rejects_bad_setup():
    with pytest.raises(ValueError):
        get_model("lightgbm")(n_estimators=0)
    with pytest.raises(ValueError):
        get_model("lightgbm")(unknown=1)
