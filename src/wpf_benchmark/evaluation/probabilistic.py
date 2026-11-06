"""Unit-marginal probability scoring; no farm quantile aggregation."""
from __future__ import annotations

import numpy as np

from ..probability import LEVELS, MEDIAN, check_levels, check_quantiles, masked_pinball, postprocess


def probability_scores(raw, truth, valid, rated_kw):
    raw = np.asarray(raw, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(truth)
    if raw.shape != truth.shape + (len(LEVELS),) or valid.shape != truth.shape:
        raise ValueError("Probability scoring prediction, truth and mask shapes disagree")
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite probability forecast")
    count = int(valid.sum())
    if not count:
        return {"n_valid": 0, "normalized_mean_pinball": None, "mean_pinball_kW": None}
    pred = postprocess(raw)
    pinball = masked_pinball(pred, truth, valid, rated_kw)
    y, q, r = truth[valid], pred[valid], raw[valid]
    rated = np.broadcast_to(np.asarray(rated_kw), truth.shape)[valid].astype(np.float64)
    error = y[:, None] - q
    kw_loss = np.maximum(LEVELS * error, (LEVELS - 1) * error)
    intervals = []
    # WIS = [0.5*|y-m| + sum(alpha/2 * interval_score)] / 4.5 = 2*mean_pinball.
    dispersion = np.zeros(count)
    under = np.zeros(count)
    over = np.zeros(count)
    for lo in range(4):
        hi, alpha = 8 - lo, 2 * LEVELS[lo]
        width = q[:, hi] - q[:, lo]
        lower_penalty = np.maximum(q[:, lo] - y, 0)
        upper_penalty = np.maximum(y - q[:, hi], 0)
        dispersion += alpha / 2 * width
        over += lower_penalty
        under += upper_penalty
        intervals.append({"central_probability": float(1 - alpha), "lower_tau": float(LEVELS[lo]),
                          "upper_tau": float(LEVELS[hi]), "n_valid": count,
                          "coverage_inclusive": float(((q[:, lo] <= y) & (y <= q[:, hi])).mean()),
                          "mean_width_kW": float(width.mean()),
                          "normalized_mean_width": float((width / rated).mean())})
    median_error = y - q[:, MEDIAN]
    under += 0.5 * np.maximum(median_error, 0)
    over += 0.5 * np.maximum(-median_error, 0)
    divisor = 4.5
    return {
        "n_valid": count, "n_quantile_terms": count * len(LEVELS),
        "normalized_mean_pinball": float(pinball.mean()), "mean_pinball_kW": float(kw_loss.mean()),
        "pinball_by_tau": [{"tau": float(t), "normalized_pinball": float(pinball[:, j].mean()),
                            "pinball_kW": float(kw_loss[:, j].mean())} for j, t in enumerate(LEVELS)],
        "q50_MAE_kW": float(np.abs(median_error).mean()),
        "q50_RMSE_kW": float(np.sqrt(np.square(median_error).mean())),
        "coverage_lt": (y[:, None] < q).mean(axis=0).tolist(),
        "coverage_le": (y[:, None] <= q).mean(axis=0).tolist(),
        "intervals": intervals,
        "WIS_kW": float(((dispersion + under + over) / divisor).mean()),
        "normalized_WIS": float(((dispersion + under + over) / (divisor * rated)).mean()),
        "WIS_components_kW": {"dispersion": float(dispersion.mean() / divisor),
                              "underprediction": float(under.mean() / divisor),
                              "overprediction": float(over.mean() / divisor)},
        "raw_negative_rate": float((r < 0).mean()),
        "raw_above_rated_rate": float((r > rated[:, None]).mean()),
        "crossing_rate": float(np.any(np.diff(r, axis=-1) < 0, axis=-1).mean()),
        "postprocessing_mean_absolute_change_kW": float(np.abs(r - q).mean()),
    }


def evaluate_quantiles(task, test, raw, levels=LEVELS):
    check_levels(levels)
    shape = (len(test.issues), task.H, len(test.entities), len(LEVELS))
    raw = check_quantiles(raw, shape)
    out = {"primary_metric": "normalized_mean_pinball", "quantile_levels": LEVELS.tolist(),
           "probability_object": "unit_marginals", "farm_probability": False,
           "interval_endpoints": "inclusive; quantile diagnostics report both strict and inclusive CDF",
           "WIS_identity": "WIS = 2 * nine-level mean Pinball; not independent evidence",
           "n_issues": len(test.issues), "scores": {}}
    for mask_name, target in task.scorings:
        spec = test.scoring[mask_name]
        truth = test.future_values(spec["target"])
        valid = test.future_values(spec["mask"]) & np.isfinite(truth)
        score = probability_scores(raw, truth, valid, task.rated_kw)
        score["target"] = target
        score["per_horizon"] = [dict(horizon=h + 1, minutes=(h + 1) * task.resolution_minutes,
                                     **probability_scores(raw[:, h:h+1], truth[:, h:h+1],
                                                          valid[:, h:h+1], task.rated_kw))
                                 for h in range(task.H)]
        blocks = ((test.issue_times - test.issue_times[0]) // np.timedelta64(7, "D")).astype(np.int64)
        score["stability_blocks"] = [dict(block=int(k), first_issue=str(test.issue_times[blocks == k][0]),
                                         last_issue=str(test.issue_times[blocks == k][-1]),
                                         **probability_scores(raw[blocks == k], truth[blocks == k],
                                                              valid[blocks == k], task.rated_kw))
                                     for k in np.unique(blocks)]
        out["scores"][mask_name] = score
    return out


def stratified_scores(task, test, raw, strata, minimum=200):
    """Strata are issue-turbine [I,N] or scoring-cell [I,H,N] masks, frozen before forecasts."""
    rows = []
    for mask_name, _ in task.scorings:
        spec = test.scoring[mask_name]
        truth = test.future_values(spec["target"])
        valid = test.future_values(spec["mask"]) & np.isfinite(truth)
        for name, selected in strata.items():
            selected = np.asarray(selected, dtype=bool)
            if selected.shape == (len(test.issues), len(test.entities)):
                unit_count = int(selected.sum())
                selected = np.broadcast_to(selected[:, None, :], truth.shape)
            elif selected.shape == truth.shape:
                unit_count = int(selected.any(axis=1).sum())
            else:
                raise ValueError(f"Stratum {name} has invalid shape {selected.shape}")
            rows.append({"mask": mask_name, "stratum": name, "n_issue_turbine": unit_count,
                         "insufficient_n": unit_count < minimum,
                         **probability_scores(raw, truth, valid & selected, task.rated_kw)})
    return rows
