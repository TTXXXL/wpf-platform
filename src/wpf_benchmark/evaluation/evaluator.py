"""Score a complete [I, H, N] forecast on the test split, independent of the model."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..tasks import ResolvedTask, SplitData, last_finite
from .metrics import point_metrics, safe_ratio, skill


def reference_persistence(task: ResolvedTask, test: SplitData) -> np.ndarray:
    """Persistence [I, N]: latest finite target_input (or target) in each history window."""
    cfg = task.config
    series = test.inputs[:, :, list(cfg.inputs).index(cfg.target_input)] if cfg.target_input else test.target
    return last_finite(series, test.issues, task.L, fallback=np.nan)


def evaluate(task: ResolvedTask, test: SplitData, predictions: np.ndarray) -> Dict[str, Any]:
    I, H, N = len(test.issues), task.H, len(test.entities)
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.shape != (I, H, N):
        raise ValueError(f"Predictions have shape {predictions.shape}, expected {(I, H, N)}")
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions must be finite at every issue time, horizon and turbine")
    reference = reference_persistence(task, test)[:, None, :]                  # [I, 1, N]
    wind = test.future_values(test.aux["wind"]) if "wind" in test.aux else None
    scores = {}
    for mask_name, target_name in task.scorings:
        spec = test.scoring[mask_name]
        truth = test.future_values(spec["target"])
        valid = test.future_values(spec["mask"]) & np.isfinite(truth)
        scores[mask_name] = score(task, test, predictions, truth, valid, reference, wind)
        scores[mask_name]["target"] = target_name
    return {"eval_mask": task.config.eval_mask, "n_issues": I, "horizon_steps": H, "n_entities": N,
            "first_issue": str(test.issue_times[0]), "last_issue": str(test.issue_times[-1]),
            "scores": scores}


def score(task: ResolvedTask, test: SplitData, pred: np.ndarray, truth: np.ndarray, valid: np.ndarray,
          reference: np.ndarray, wind) -> Dict[str, Any]:
    cfg, N = task.config, len(test.entities)
    rated = np.broadcast_to(task.rated_kw[None, None, :], truth.shape)
    err = pred - truth
    ref_err = reference - truth
    both = valid & np.isfinite(reference)
    out: Dict[str, Any] = {}

    turbine = point_metrics(err[valid], rated[valid])
    per_entity = {test.entities[j]: float(np.abs(err[:, :, j][valid[:, :, j]]).mean(dtype=np.float64))
                  if valid[:, :, j].any() else None for j in range(N)}
    finite = [v for v in per_entity.values() if v is not None]
    turbine["macro_MAE_kW"] = float(np.mean(finite)) if finite else float("nan")
    ref = point_metrics(ref_err[both], rated[both])
    model_on_both = point_metrics(err[both], rated[both])
    turbine.update(persistence_MAE_kW=ref["MAE_kW"], persistence_RMSE_kW=ref["RMSE_kW"],
                   SS_MAE_pct=skill(model_on_both["MAE_kW"], ref["MAE_kW"]),
                   SS_RMSE_pct=skill(model_on_both["RMSE_kW"], ref["RMSE_kW"]),
                   over_frac=float((err[valid] > 0).mean()) if turbine["n"] else float("nan"),
                   under_frac=float((err[valid] < 0).mean()) if turbine["n"] else float("nan"))
    out["turbine"], out["per_entity_MAE_kW"] = turbine, per_entity

    count = valid.sum(axis=2)
    farm_ok = (count / N >= cfg.farm_min_valid_frac) & (count > 0)
    capacity = np.where(valid, rated, 0.0).sum(axis=2, dtype=np.float64)[farm_ok]
    farm_err = np.where(valid, err, 0.0).sum(axis=2, dtype=np.float64)[farm_ok]
    farm_ref = np.where(valid, ref_err, 0.0).sum(axis=2, dtype=np.float64)[farm_ok]
    farm = point_metrics(farm_err, capacity)
    farm_ref_ok = np.isfinite(farm_ref)
    farm.update(persistence_MAE_kW=float(np.abs(farm_ref[farm_ref_ok]).mean()) if farm_ref_ok.any() else float("nan"),
                normalization="valid_turbine_capacity", min_valid_frac=cfg.farm_min_valid_frac)
    farm["SS_MAE_pct"] = skill(float(np.abs(farm_err[farm_ref_ok]).mean()) if farm_ref_ok.any() else float("nan"),
                               farm["persistence_MAE_kW"])
    out["farm"] = farm

    rows = []
    for h in range(task.H):
        v, b = valid[:, h], both[:, h]
        row = point_metrics(err[:, h][v], rated[:, h][v])
        row.update(horizon=h + 1, minutes=(h + 1) * task.resolution_minutes,
                   persistence_MAE_kW=point_metrics(ref_err[:, h][b], rated[:, h][b])["MAE_kW"])
        rows.append(row)
    out["per_horizon"] = rows
    out["report_horizons"] = [rows[h - 1] for h in task.report_horizons if 0 < h <= task.H]
    out.update(_conditional(task, test, pred, truth, valid, err, rated, reference, wind))
    return out


def _conditional(task, test, pred, truth, valid, err, rated, reference, wind) -> Dict[str, Any]:
    cfg, out = task.config, {}
    n = int(valid.sum())
    if wind is not None:
        bins = {}
        for lo, hi in cfg.wind_bins:
            chosen = valid & (wind >= lo) & (wind < hi)
            bins[f"[{lo}, {hi})"] = point_metrics(err[chosen], rated[chosen])
        out["wind_bins"] = bins
    issue_times = test.issue_times
    block = np.timedelta64(cfg.stability_block_days, "D")
    start = issue_times[0]
    blocks: List[Dict[str, Any]] = []
    index = ((issue_times - start) // block).astype(np.int64)
    for k in np.unique(index):
        rows = index == k
        sub = valid[rows]
        blocks.append(dict(point_metrics(err[rows][sub], rated[rows][sub]),
                           start=str(issue_times[rows][0]), end=str(issue_times[rows][-1])))
    out["stability_blocks"] = blocks
    threshold = cfg.ramp_threshold_frac * rated
    ramp_true = valid & (np.abs(truth - reference) >= threshold)
    ramp_pred = valid & (np.abs(pred - reference) >= threshold)
    tp, pos, pred_pos = int((ramp_true & ramp_pred).sum()), int(ramp_true.sum()), int(ramp_pred.sum())
    out["ramp"] = {"threshold_frac_of_rated": cfg.ramp_threshold_frac, "n_true": pos, "n_pred": pred_pos,
                   "n_hit": tp, "recall": safe_ratio(tp, pos), "precision": safe_ratio(tp, pred_pos),
                   "f1": safe_ratio(2 * tp, pos + pred_pos),
                   "MAE_on_true_ramps_kW": float(np.abs(err[ramp_true]).mean(dtype=np.float64)) if pos else float("nan")}
    if task.eval_stride == 1:
        out["daily_energy"] = _daily_energy(task, test, pred, truth, valid)
    out["sanity"] = {"negative_rate_pct": 100.0 * safe_ratio(int((valid & (pred < 0)).sum()), n),
                     "over_rated_rate_pct": 100.0 * safe_ratio(int((valid & (pred > rated)).sum()), n)}
    return out


def _daily_energy(task, test, pred, truth, valid) -> Dict[str, Any]:
    """h=1 forecasts, one per 10-minute target, summed per calendar day of the target time."""
    step = np.timedelta64(task.resolution_minutes, "m").astype("timedelta64[us]")
    days = (test.issue_times + step).astype("datetime64[D]")
    to_mwh = task.resolution_minutes / 60.0 / 1000.0
    rows = []
    for day in np.unique(days):
        take = days == day
        m = valid[take, 0]
        actual = float(np.where(m, truth[take, 0], 0.0).sum(dtype=np.float64) * to_mwh)
        forecast = float(np.where(m, pred[take, 0], 0.0).sum(dtype=np.float64) * to_mwh)
        rows.append({"day": str(day), "n_steps": int(take.sum()), "actual_MWh": actual, "forecast_MWh": forecast,
                     "abs_error_MWh": abs(forecast - actual),
                     "APE_pct": 100.0 * abs(forecast - actual) / actual if actual else float("nan")})
    ape = [r["APE_pct"] for r in rows if np.isfinite(r["APE_pct"])]
    return {"n_days": len(rows), "MAE_MWh": float(np.mean([r["abs_error_MWh"] for r in rows])),
            "MAPE_pct": float(np.mean(ape)) if ape else float("nan"), "per_day": rows}
