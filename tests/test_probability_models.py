"""Quantile model forward/backward, checkpoints, leakage and opt-in runner checks."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from wpf_benchmark.data.store import write_dataset
from wpf_benchmark.datasets.synthetic import make_panel
from wpf_benchmark.models import get_model
from wpf_benchmark.models.neural.trainer import NetworkSpec, load_checkpoint
from wpf_benchmark.paths import ProjectPaths
from wpf_benchmark.probability import LEVELS, postprocess, torch_pinball
from wpf_benchmark.profile_probability import history_features, make_strata
from wpf_benchmark.runner import TaskData, make_context, seed_all
from wpf_benchmark.tasks import TaskConfig, build_split, resolve_task
from wpf_benchmark.tasks.probabilistic import window_audit
from wpf_benchmark.probability_runner import run_probability, score_probability


@pytest.fixture
def small_data():
    panel = make_panel(days=24, entities=2)
    cfg = TaskConfig.from_dict({"name": "probability_test", "dataset": "synthetic", "target": "power",
        "target_input": "power", "inputs": ["wind_speed", "wind_dir_sin", "wind_dir_cos", "power"],
        "history_minutes": 1440, "horizon_minutes": 30, "train_mask": "observed", "eval_mask": "observed",
        "splits": {"train": ["2020-01-01", "2020-01-17"], "valid": ["2020-01-17", "2020-01-20"],
                   "test": ["2020-01-20", "2020-01-25"]},
        "extra": {"window_protocol": "causal_history_v1", "probability_protocol": "unit_quantiles_v1"}})
    task = resolve_task(cfg, panel.meta | {"time_range": ["2020-01-01", "2020-01-25"]})
    splits = [build_split(task, panel, name) for name in ("train", "valid", "test")]
    # Bound software smoke to a handful of training windows.
    splits[0].issues = splits[0].issues[:8]
    splits[1].issues = splits[1].issues[:4]
    splits[2].issues = splits[2].issues[:6]
    return TaskData(task, *splits, {"panel_sha256": "synthetic", "meta_sha256": "synthetic"})


CASES = [("dlinear_q", {}), ("lstm_q", {"hidden": 8, "layers": 1}),
         ("tcn_q", {"channels": 4, "levels": 1}),
         ("patchtst_q", {"d_model": 8, "heads": 2, "layers": 1, "ff": 16, "patch": 12}),
         ("itransformer_q", {"d_model": 8, "heads": 2, "layers": 1, "ff": 16}),
         ("agcrn_q", {"hidden": 4, "layers": 1, "emb": 2})]


@pytest.mark.parametrize("name,params", CASES)
def test_q_models_backward_checkpoint_and_point(name, params, small_data, tmp_path):
    seed_all(0)
    model = get_model(name)(**params, epochs=1, batch=4)
    ctx = make_context(small_data)
    model.fit(ctx, small_data.train, small_data.valid)
    batch = small_data.test.batch(small_data.test.issues[:2])
    raw = model.predict_quantiles(batch)
    assert raw.shape == (2, ctx.H, 2, 9)
    assert np.isfinite(raw).all() and (np.diff(raw, axis=-1) >= 0).all()
    assert model.validation_metrics["best_epoch"] == 1
    assert np.isfinite(model.validation_metrics["normalized_mean_pinball"])
    np.testing.assert_array_equal(model.predict(batch), postprocess(raw)[..., 4])
    model.save_checkpoint(tmp_path / "checkpoint.pt")
    loaded = load_checkpoint(tmp_path / "checkpoint.pt", small_data.task, device="cpu")
    np.testing.assert_allclose(loaded.predict_quantiles(batch), raw, rtol=1e-5, atol=1e-3)
    empty = replace(small_data.train, target_mask=np.zeros_like(small_data.train.target_mask))
    with pytest.raises(ValueError, match="No valid"):
        get_model(name)(**params, epochs=1, batch=4).fit(ctx, empty, small_data.valid)


@pytest.mark.parametrize("mode", ["none", "level", "level_volatility"])
def test_persistence_repeatability_and_no_future_labels(mode, small_data):
    ctx = make_context(small_data)
    model = get_model("pers_q")(conditioning=mode)
    model.fit(ctx, small_data.train, small_data.valid)
    batch = small_data.test.batch(small_data.test.issues)
    before = model.predict_quantiles(batch)
    np.testing.assert_array_equal(model.predict_quantiles(batch), before)
    small_data.valid.target[:] = 1e8
    second = get_model("pers_q")(conditioning=mode)
    second.fit(ctx, small_data.train, small_data.valid)
    np.testing.assert_array_equal(second.pooled, model.pooled)
    assert before.shape == (6, ctx.H, 2, 9)
    assert np.all(np.diff(before, axis=-1) >= 0)
    assert model.fallback


def test_empirical_quantiles_minimize_pinball():
    rng = np.random.default_rng(0)
    # AR(1) innovations, conditional on each preceding value.
    history = np.zeros(20001)
    noise = rng.normal(size=20000)
    for k in range(20000):
        history[k+1] = 0.8 * history[k] + noise[k]
    residual = history[1:] - 0.8*history[:-1]
    q = np.quantile(residual, LEVELS)
    from scipy.stats import norm
    np.testing.assert_allclose(q, norm.ppf(LEVELS), atol=0.04)
    def loss(values):
        e = residual[:, None] - values
        return np.maximum(LEVELS * e, (LEVELS-1)*e).mean(0)
    assert np.all(loss(q) <= loss(q+0.1)) and np.all(loss(q) <= loss(q-0.1))


def test_lightgbm_q_i0_and_booster_mapping(small_data):
    ctx = make_context(small_data)
    params = {"n_estimators": 2, "n_jobs": 1, "num_leaves": 3}
    model = get_model("lightgbm_q")(**params)
    model.fit(ctx, small_data.train, small_data.valid)
    batch = small_data.test.batch(small_data.test.issues[:2])
    raw = model.predict_quantiles(batch)
    assert raw.shape == (2, ctx.H, 2, 9)
    assert len(model.estimators) == ctx.H
    for boosters in model.estimators:
        assert len(boosters) == 9
        assert [m.params["alpha"] for m in boosters] == LEVELS.tolist()
        assert all(m.params["objective"] == "quantile" for m in boosters)
    X = model._features(batch.x_past, batch.issue_time)
    assert X.shape[-1] == 6 * len(ctx.input_names) + model.params["lags"]
    np.testing.assert_array_equal(X, model._features(batch.x_past, batch.issue_time + np.timedelta64(4, "h")))
    second = get_model("lightgbm_q")(**params)
    second.fit(ctx, small_data.train, small_data.valid)
    np.testing.assert_array_equal(raw, second.predict_quantiles(batch))


@pytest.mark.parametrize("L,H", [(144,36), (576,36), (576,64)])
def test_causal_windows_and_legacy_opt_in(L,H,small_data):
    cfg = replace(small_data.task.config, history_minutes=L*10, horizon_minutes=H*10)
    panel = make_panel(days=24, entities=2)
    task = resolve_task(cfg, small_data.task.meta)
    for name in ("train", "valid", "test"):
        split = build_split(task, panel, name)
        row = window_audit(task, split)
        assert row["n_issues"] > 0 and row["label_cross_boundary_count"] == 0
        if name == "valid":
            assert split.issue_times[0] == task.bounds["valid"][0] and row["history_borrowed"]
    legacy = resolve_task(replace(cfg, extra={}), small_data.task.meta)
    if L == 576:
        with pytest.raises(ValueError, match="too short"):
            build_split(legacy, panel, "valid")
    else:
        assert build_split(legacy,panel,"valid").issue_times[0] > legacy.bounds["valid"][0]


def test_history_features_ignore_future(small_data):
    test, rated = small_data.test, small_data.task.rated_kw
    first = history_features(test, rated)
    altered = replace(test, target=test.target.copy())
    altered.target[test.issues[-1]+1:] = 1e9
    second = history_features(altered, rated)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    strata = make_strata(test, rated, [0.1,0.2], first)
    assert "known/level/0" in strata and "posthoc/zero" in strata
    assert strata["known/level/0"].shape == (len(test.issues),len(test.entities))


def test_opt_in_runner_roundtrip_and_old_score_rejects(tmp_path, small_data):
    from wpf_benchmark.runner import score_predictions
    paths = ProjectPaths(tmp_path)
    paths.configs.joinpath("tasks").mkdir(parents=True)
    panel = make_panel(days=24, entities=2)
    write_dataset(paths.dataset("synthetic"), panel)
    cfg = small_data.task.config
    (paths.configs / "tasks" / "probability_test.json").write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    directory = tmp_path / "run"
    result = run_probability(paths, "probability_test", "pers_q", 0, directory)
    scored = score_probability(paths, "probability_test", directory / "predictions_quantiles.npz", tmp_path / "score")
    assert result == scored
    assert (directory / "DONE").exists()
    with pytest.raises(ValueError, match="shape"):
        score_predictions(paths,"probability_test",directory / "predictions_quantiles.npz")
    with pytest.raises(ValueError, match="empty"):
        run_probability(paths,"probability_test","pers_q",0,directory)
