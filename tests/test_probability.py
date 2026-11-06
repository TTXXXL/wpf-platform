"""Probability arithmetic, protocol identity, loss masks and numerical CDF checks."""
import json

import numpy as np
import pytest
from scipy.stats import t

from wpf_benchmark.probability import (LEVELS, check_levels, check_quantiles, load_quantiles,
                                      masked_pinball, ordered_head, postprocess, save_quantiles, torch_pinball)
from wpf_benchmark.evaluation.probabilistic import probability_scores
from wpf_benchmark.models.mixture_quantiles import mixture_cdf, mixture_quantiles, validate_parameters


def test_pinball_known_value_and_mask():
    y = np.asarray([[[2.0, np.nan]]])
    pred = np.zeros((1, 1, 2, 9))
    pred[..., 1, :] = np.nan
    loss = masked_pinball(pred, y, np.ones_like(y, bool), [2.0, 2.0])
    np.testing.assert_allclose(loss, LEVELS[None])
    assert loss.mean() == pytest.approx(0.5)
    with pytest.raises(ValueError, match="No valid"):
        masked_pinball(pred, y, np.zeros_like(y, bool), [2.0, 2.0])


def test_wis_identity_and_zero_atom():
    rng = np.random.default_rng(0)
    raw = rng.normal(1000, 700, (4, 6, 3, 9))
    truth = rng.uniform(0, 2500, (4, 6, 3))
    truth[0] = 0
    truth[1, 0, 0] = np.nan
    scores = probability_scores(raw, truth, np.ones(truth.shape, bool), [2000, 2050, 2100])
    assert scores["n_valid"] == truth.size - 1
    assert scores["n_quantile_terms"] == 9 * (truth.size - 1)
    assert scores["WIS_kW"] == pytest.approx(2 * scores["mean_pinball_kW"])
    assert scores["normalized_WIS"] == pytest.approx(2 * scores["normalized_mean_pinball"])
    assert sum(scores["WIS_components_kW"].values()) == pytest.approx(scores["WIS_kW"])
    ties = probability_scores(np.zeros((1, 1, 1, 9)), np.zeros((1, 1, 1)), np.ones((1, 1, 1), bool), [2050])
    assert ties["coverage_lt"] == [0.0] * 9 and ties["coverage_le"] == [1.0] * 9
    assert ties["WIS_kW"] == 0


def test_processing_and_crossing():
    raw = np.asarray([[[[-2, -1, 0, 1, 2, 3, 4000, 3000, 5000]]]], dtype=np.float32)
    processed = postprocess(raw)
    assert processed.min() == 0 and processed.max() == 5000
    assert np.all(np.diff(processed) >= 0)
    with pytest.raises(ValueError, match="crossing"):
        check_quantiles(raw, monotone=True)
    score = probability_scores(raw, np.ones((1, 1, 1)), np.ones((1, 1, 1), bool), [2050])
    assert score["crossing_rate"] == 1 and score["raw_negative_rate"] == pytest.approx(2/9)
    assert score["raw_above_rated_rate"] == pytest.approx(3/9)


def test_file_roundtrip_and_rejections(tmp_path):
    raw = np.arange(2 * 3 * 2 * 9, dtype=np.float32).reshape(2, 3, 2, 9) - 20
    times = np.array(["2021-01-01", "2021-01-02"], dtype="datetime64[us]")
    entities = ["A", "B"]
    metadata = {"task_sha256": "task", "dataset": {"panel_sha256": "panel"}, "target": "power",
                "eval_mask": "observed", "normalization": {"rated_kw": [2050, 2050]}, "model": "synthetic"}
    path = tmp_path / "predictions_quantiles.npz"
    save_quantiles(path, raw, times, entities, metadata, True)
    back, scored, identity = load_quantiles(path, times, entities, raw.shape, metadata)
    np.testing.assert_array_equal(back, raw)
    np.testing.assert_array_equal(scored, postprocess(raw))
    assert identity["upper_clip"] is None
    for wrong_times, wrong_entities, wrong_shape, wrong_meta in (
        (times[::-1], entities, raw.shape, metadata), (times, entities[::-1], raw.shape, metadata),
        (times, entities, (2, 3, 3, 9), metadata), (times, entities, raw.shape, {"target": "other"})):
        with pytest.raises(ValueError):
            load_quantiles(path, wrong_times, wrong_entities, wrong_shape, wrong_meta)
    old = tmp_path / "point.npz"
    np.savez(old, predictions=raw[..., 4], issue_times=times, entities=entities)
    with pytest.raises(ValueError, match="Incomplete"):
        load_quantiles(old, times, entities, raw.shape, metadata)
    with pytest.raises(ValueError):
        check_quantiles(raw[..., 4])
    with pytest.raises(ValueError):
        check_levels(LEVELS[::-1])
    with pytest.raises(FileExistsError):
        save_quantiles(path, raw, times, entities, metadata)


def test_torch_loss_nan_gradient_and_ties():
    import torch
    raw = torch.tensor([[[[1., -1., -1., 0., 0., 0., 0., 0., 0.], [0.] * 9]]], requires_grad=True)
    pred = ordered_head(raw)
    assert torch.all(pred[0, 0, 0] == 1)
    y = torch.tensor([[[2., float("nan")]]])
    mask = torch.ones((1, 1, 2), dtype=torch.bool)
    loss = torch_pinball(pred, y, mask)
    assert loss.item() == pytest.approx(0.5)
    loss.backward()
    assert torch.isfinite(raw.grad).all()
    assert torch.all(raw.grad[0, 0, 1] == 0)
    with pytest.raises(ValueError, match="No valid"):
        torch_pinball(pred, y, torch.zeros_like(mask))


def test_mixture_single_component_closed_form():
    df = np.asarray([[3., 3.]])
    w = np.asarray([[0.3, 0.7]])
    loc, scale = np.ones_like(df)*2, np.ones_like(df)*4
    q = mixture_quantiles(w, df, loc, scale)
    np.testing.assert_allclose(q[0], 2 + 4*t.ppf(LEVELS, 3), atol=1e-6)
    assert np.all(np.diff(q) >= 0)
    for j, tau in enumerate(LEVELS):
        np.testing.assert_allclose(mixture_cdf(q[..., j], w, df, loc, scale), tau, atol=1e-7)
    np.testing.assert_array_equal(q, mixture_quantiles(w, df, loc, scale))


def test_mixture_fixed_sampling():
    rng = np.random.default_rng(0)
    weights, df, loc, scale = map(np.asarray, ([0.2, 0.8], [3.0, 7.0], [-3.0, 2.0], [1.0, 2.0]))
    q = mixture_quantiles(weights, df, loc, scale)
    chosen = rng.choice(2, size=10000, p=weights)
    samples = loc[chosen] + scale[chosen] * rng.standard_t(df[chosen])
    # Rank error is distribution-scale independent and fixed before seeing samples.
    empirical = (samples[:, None] <= q).mean(0)
    assert np.max(np.abs(empirical - LEVELS)) < 0.03
    empirical_q = np.quantile(samples, LEVELS)
    assert np.max(np.abs(mixture_cdf(empirical_q, weights, df, loc, scale) - LEVELS)) < 0.03


@pytest.mark.parametrize("w,df,loc,scale", [([1.], [0.], [0.], [1.]), ([1.], [3.], [np.nan], [1.]),
                                            ([1.], [3.], [0.], [-1.]), ([0.5], [3.], [0.], [1.])])
def test_mixture_rejects_invalid(w, df, loc, scale):
    with pytest.raises(ValueError):
        mixture_quantiles(w, df, loc, scale)
