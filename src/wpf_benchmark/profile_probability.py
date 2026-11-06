"""CPU-only data profile and pre-result strata for probability experiments.

This module never imports model code, torch or a checkpoint loader. Windows are
streamed in bounded chunks; all thresholds are estimated from training histories.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .data.store import load_meta, load_panel, verify_dataset
from .provenance import canonical_sha256, sha256_file
from .tasks import build_split, resolve_task
from .tasks.probabilistic import CUTOFFS, probability_task, window_audit

LEVEL_EDGES = np.asarray([0.02, 0.2, 0.5, 0.8, 0.97])


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def history_features(split, rated, steps_per_day=144, chunk=128):
    """Only information available at issue time; arrays are [issue,turbine]."""
    output = {k: [] for k in ("r0", "volatility", "missing_fraction", "last_hour_missing", "last_6h_missing",
                              "power_missing", "wind_missing", "direction_missing", "history_zero_fraction",
                              "history_above_rated_fraction", "any_missing")}
    target_index = split.input_names.index("power")
    wind_index = split.input_names.index("wind_speed")
    direction = [split.input_names.index(n) for n in ("wind_dir_sin", "wind_dir_cos")]
    for first in range(0, len(split.issues), chunk):
        take = split.issues[first:first+chunk]
        history = take[:, None] + np.arange(-split.L + 1, 1)
        x = split.inputs[history]
        missing = ~np.isfinite(x)
        power = x[..., target_index].astype(np.float64)
        finite = np.isfinite(power)
        count = finite.sum(1)
        zero = ((power == 0) & finite).sum(1) / np.maximum(count, 1)
        above = ((power > rated) & finite).sum(1) / np.maximum(count, 1)
        ticks = np.where(finite, np.arange(split.L)[None, :, None], -1).max(1)
        safe = np.where(finite, power, 0)
        latest = np.take_along_axis(safe, ticks.clip(min=0)[:, None], axis=1)[:, 0] / rated
        latest[ticks < 0] = np.nan
        recent = power[:, -steps_per_day:]
        good = np.isfinite(recent)
        n = good.sum(1)
        mean = np.where(good, recent, 0).sum(1) / np.maximum(n, 1)
        variance = np.where(good, (np.where(good, recent, 0) - mean[:, None]) ** 2, 0).sum(1) / np.maximum(n, 1)
        vol = np.sqrt(variance) / rated
        vol[n == 0] = np.nan
        values = {"r0": latest, "volatility": vol, "missing_fraction": missing.mean(axis=(1, 3)),
                  "any_missing": missing.any(axis=(1, 3)), "last_hour_missing": missing[:, -6:].any(axis=(1, 3)),
                  "last_6h_missing": missing[:, -36:].any(axis=(1, 3)),
                  "power_missing": missing[..., target_index].mean(1), "wind_missing": missing[..., wind_index].mean(1),
                  "direction_missing": missing[..., direction].mean(axis=(1, 3)),
                  "history_zero_fraction": zero, "history_above_rated_fraction": above}
        for key, value in values.items():
            output[key].append(value)
    return {key: np.concatenate(parts) for key, parts in output.items()}


def make_strata(split, rated, volatility_edges, features=None):
    features = history_features(split, rated) if features is None else features
    r0, vol, missing = (features[k] for k in ("r0", "volatility", "missing_fraction"))
    strata = {}
    level = np.digitize(r0, LEVEL_EDGES)
    for k in range(6):
        strata[f"known/level/{k}"] = (level == k) & np.isfinite(r0)
    vbin = np.digitize(vol, volatility_edges)
    for k in range(3):
        strata[f"known/volatility/{k}"] = (vbin == k) & np.isfinite(vol)
    for k, chosen in enumerate((missing == 0, (missing > 0) & (missing <= 0.05), missing > 0.05)):
        strata[f"known/missing/{k}"] = chosen
    strata["known/last_hour_missing"] = features["last_hour_missing"]
    hours = (split.issue_times.astype("datetime64[h]").astype(np.int64) % 24) // 4
    for k in range(6):
        strata[f"known/utc_4h/{k}"] = np.broadcast_to((hours == k)[:, None], r0.shape)
    quarters = (split.issue_times.astype("datetime64[M]").astype(np.int64) % 12) // 3 + 1
    for k in np.unique(quarters):
        strata[f"known/quarter/{k}"] = np.broadcast_to((quarters == k)[:, None], r0.shape)
    H, N, I = split.H, len(split.entities), len(split.issues)
    horizons = np.arange(1, H + 1)
    for a, b in ((1, 6), (7, 12), (13, 24), (25, 36)):
        strata[f"known/horizon/{a}-{b}"] = np.broadcast_to(((horizons >= a) & (horizons <= b))[None, :, None], (I, H, N))
    truth = split.future_values(split.target)
    strata["posthoc/zero"] = truth == 0
    strata["posthoc/above_rated"] = truth > rated
    strata["posthoc/change_20pct"] = np.abs(truth / rated - r0[:, None]) >= 0.2
    return strata


def _rate(chosen, valid):
    n = int(valid.sum())
    return {"count": int((chosen & valid).sum()), "denominator": n,
            "fraction": float((chosen & valid).sum() / n) if n else None}


def profile_split(task, split, edges, farm, cutoff, grid_dir):
    features = history_features(split, task.rated_kw)
    truth = split.future_values(split.target)
    valid = split.future_values(split.target_mask) & np.isfinite(truth)
    wind = split.future_values(split.aux["wind"]) if "wind" in split.aux else None
    N, I = len(split.entities), len(split.issues)
    shared = {"farm": farm, "cutoff": cutoff, "split": split.name}
    missing_rows, zero_rows, above_rows = [], [], []
    for entity in range(N):
        observed = valid[..., entity]
        y = truth[..., entity]
        miss = features["missing_fraction"][:, entity]
        distribution, counts = np.unique(miss, return_counts=True)
        missing_rows.append({**shared, "entity": split.entities[entity], "n_issue_turbine": I,
                             "any_history_missing_count": int(features["any_missing"][:, entity].sum()),
                             "any_history_missing_fraction": float(features["any_missing"][:, entity].mean()),
                             "whole_history_missing_fraction": float(miss.mean()),
                             "power_missing_fraction": float(features["power_missing"][:, entity].mean()),
                             "wind_missing_fraction": float(features["wind_missing"][:, entity].mean()),
                             "direction_missing_fraction": float(features["direction_missing"][:, entity].mean()),
                             "last_1h_missing_fraction": float(features["last_hour_missing"][:, entity].mean()),
                             "last_6h_missing_fraction": float(features["last_6h_missing"][:, entity].mean()),
                             "future_any_missing_fraction": float((~observed).any(1).mean()),
                             "n_target_cells": int(observed.size), "n_observed_target_cells": int(observed.sum()),
                             "observed_target_fraction": float(observed.mean()),
                             **{f"missing_P{p}": float(np.percentile(miss, p)) for p in (50, 90, 95, 99)},
                             "exact_missing_fraction_histogram": json.dumps(list(zip(distribution.tolist(), counts.tolist())))})
        for h in range(task.H):
            v, t = observed[:, h], y[:, h]
            future_wind = wind[:, h, entity] if wind is not None else None
            stopped = (future_wind >= 4) & (t == 0) if future_wind is not None else np.zeros_like(v)
            zero_rows.append({**shared, "entity": split.entities[entity], "horizon": h+1,
                              **{f"zero_{k}": val for k, val in _rate(t == 0, v).items()},
                              **{f"below_2pct_{k}": val for k, val in _rate(t < 0.02*task.rated_kw[entity], v).items()},
                              **{f"below_5pct_{k}": val for k, val in _rate(t < 0.05*task.rated_kw[entity], v).items()},
                              **{f"below_20pct_{k}": val for k, val in _rate(t < 0.2*task.rated_kw[entity], v).items()},
                              **{f"stop_candidate_{k}": val for k, val in _rate(stopped, v & np.isfinite(future_wind) if future_wind is not None else np.zeros_like(v)).items()},
                              "latest_zero_fraction": float((features["r0"][:, entity] == 0).mean()),
                              "history_zero_fraction": float(features["history_zero_fraction"][:, entity].mean())})
            above_rows.append({**shared, "entity": split.entities[entity], "horizon": h+1,
                               **{f"above_{delta}kW_{k}": val for delta in (0, 10, 50)
                                  for k, val in _rate(t > task.rated_kw[entity] + delta, v).items()},
                               "maximum_kW": float(t[v].max()) if v.any() else None,
                               "P99_kW": float(np.percentile(t[v], 99)) if v.any() else None,
                               "history_above_rated_fraction": float(features["history_above_rated_fraction"][:, entity].mean())})
    strata = make_strata(split, task.rated_kw, edges, features)
    rows = []
    for name, selected in strata.items():
        if selected.ndim == 2:
            unit_count = int(selected.sum())
            selected_cells = selected[:, None] & valid
        else:
            unit_count = int(selected.any(1).sum())
            selected_cells = selected & valid
        rows.append({**shared, "stratum": name, "n_issue_turbine": unit_count,
                     "n_issue_turbine_denominator": I * N, "fraction": unit_count / (I*N),
                     "n_target_cells": int(selected_cells.sum()), "insufficient_n": unit_count < 200})
    grid = None
    if split.name == "test":
        masks = {name: split.future_values(spec["mask"]) & np.isfinite(split.future_values(spec["target"]))
                 for name, spec in split.scoring.items()}
        target_fingerprints = {name: {
            "target_values_sha256": hashlib.sha256(
                np.asarray(split.future_values(spec["target"]), dtype="<f4").tobytes()).hexdigest(),
            "valid_mask_sha256": hashlib.sha256(masks[name].tobytes()).hexdigest()}
            for name, spec in split.scoring.items()}
        filename = f"{farm}_{cutoff}_score_grid.npz"
        np.savez_compressed(grid_dir / filename, issue_times=split.issue_times.astype("datetime64[us]").astype(np.int64),
                            entities=np.asarray(split.entities), horizons=np.arange(1, task.H+1), **masks)
        strata_file = f"{farm}_{cutoff}_strata.npz"
        np.savez_compressed(grid_dir / strata_file, **strata)
        grid = {**shared, "file": filename, "sha256": sha256_file(grid_dir / filename),
                "strata_file": strata_file, "strata_sha256": sha256_file(grid_dir / strata_file),
                "shape": [I, task.H, N], "entity_order": split.entities,
                "n_valid_by_mask": {name: int(mask.sum()) for name, mask in masks.items()},
                "target_and_mask_identity": target_fingerprints,
                "issue_times_sha256": hashlib.sha256(split.issue_times.astype("datetime64[us]").astype("<i8").tobytes()).hexdigest()}
    missing_fraction = float(features["any_missing"].mean())
    summary = {**shared, "n_issue": I, "n_issue_turbine": I*N, "n_target_cells": int(valid.size),
               "n_observed_target_cells": int(valid.sum()), "any_history_missing_fraction": missing_fraction,
               "H4_priority": "appendix" if missing_fraction < 0.02 else ("primary_data_condition" if missing_fraction > 0.1 else "retain_F3_I2"),
               "zero_target": _rate(truth == 0, valid),
               "above_rated_target": _rate(truth > task.rated_kw, valid)}
    return summary, missing_rows, zero_rows, above_rows, rows, grid


def run_profile(dataset_root, out, allow_synthetic=False):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Profile output must be empty; frozen files are never overwritten")
    # Check all data dependencies before creating any apparent profile artifacts.
    identities = {farm: verify_dataset(Path(dataset_root) / farm) for farm in ("kelmarsh", "penmanshiel")}
    out.mkdir(parents=True, exist_ok=True)
    grid_dir = out / "grids"
    grid_dir.mkdir()
    summaries, missing, zeros, above, windows, strata_rows, grids = [], [], [], [], [], [], []
    boundaries = {}
    for farm in ("kelmarsh", "penmanshiel"):
        directory = Path(dataset_root) / farm
        meta = load_meta(directory)
        if meta.get("adapter") == "synthetic" and not allow_synthetic:
            raise ValueError("Synthetic panels are for tests only; pass allow_synthetic=True explicitly in a software test")
        if not np.all(np.asarray([e["rated_kw"] for e in meta["entities"]]) == 2050):
            raise ValueError("Stage 0/1 frozen K/P protocol requires 2050 kW capacities")
        panel = load_panel(directory, ["wind_speed", "wind_dir_sin", "wind_dir_cos", "power"],
                           [m for m in ("observed", "normal_b", "normal_c") if m in meta["masks"]], end="2022-01-01")
        boundaries[farm] = {}
        for cutoff in CUTOFFS:
            task = resolve_task(probability_task(meta, cutoff), meta)
            splits = {s: build_split(task, panel, s) for s in ("train", "valid", "test")}
            training_features = history_features(splits["train"], task.rated_kw)
            vol = training_features["volatility"]
            if not np.isfinite(vol).any():
                raise ValueError(f"No training volatility for {farm}/{cutoff}")
            edges = np.quantile(vol[np.isfinite(vol)], [1/3, 2/3], method="linear")
            boundaries[farm][cutoff] = {"tertiles": edges.tolist(), "source": "training issue-turbine historical windows only",
                                        "n_finite": int(np.isfinite(vol).sum()), "train_end": str(task.bounds["train"][1])}
            del training_features
            for s, split in splits.items():
                if s != "test":
                    lo = panel.index_of(split.times[0])
                    split.aux["wind"] = panel.variable("wind_speed")[lo:lo+len(split.times)]
                summary, m, z, a, sr, grid = profile_split(task, split, edges, farm, cutoff, grid_dir)
                summaries.append(summary)
                missing.extend(m)
                zeros.extend(z)
                above.extend(a)
                strata_rows.extend(sr)
                audit = window_audit(task, split)
                audit.update(farm=farm, cutoff=cutoff, n_target_cells=summary["n_target_cells"],
                             n_observed_target_cells=summary["n_observed_target_cells"])
                windows.append(audit)
                if grid is not None:
                    grids.append(grid)
            for L, H in ((576, 36), (576, 64)):
                long_task = resolve_task(probability_task(meta, cutoff, L, H), meta)
                for s in ("train", "valid", "test"):
                    split = build_split(long_task, panel, s)
                    audit = window_audit(long_task, split)
                    audit.update(farm=farm, cutoff=cutoff, n_target_cells=len(split.issues)*H*len(split.entities),
                                 n_observed_target_cells=int((split.future_values(split.target_mask) & np.isfinite(split.future_values(split.target))).sum()))
                    windows.append(audit)
    frozen = {"version": "stage1-20261106-v1", "frozen_before_model_results": True,
              "level_edges": [0.0, 0.02, 0.2, 0.5, 0.8, 0.97, None],
              "volatility_bins": boundaries, "missing_bins": [0.0, 0.05, None],
              "time_bins_utc_hours": [0,4,8,12,16,20,24], "horizon_bins": [[1,6],[7,12],[13,24],[25,36]],
              "minimum_issue_turbines": 200, "merged_bins": [],
              "low_sample_policy": "retain and mark insufficient_n; any merge requires a new frozen version",
              "software_test_only": bool(allow_synthetic)}
    write_json(out / "strata_boundaries.json", frozen)
    write_json(out / "common_score_grid_manifest.json", {"datasets": identities, "grids": grids,
                                                         "index_order": ["issue_time", "horizon", "entity"],
                                                         "validity": "registered target mask AND finite truth"})
    write_json(out / "profile.json", {"status": "software_test_only" if allow_synthetic else "passed",
                                     "software_test_only": bool(allow_synthetic),
                                     "datasets": identities, "summaries": summaries,
                                     "no_model_weights_loaded": True, "no_training_or_prediction": True,
                                     "frozen_boundaries_sha256": canonical_sha256(frozen)})
    for name, rows in (("window_profile", windows), ("missingness_by_farm_quarter", missing),
                       ("zero_power_profile", zeros), ("above_rated_profile", above), ("strata_counts", strata_rows)):
        write_csv(out / f"{name}.csv", rows)
    (out / "README.md").write_text("# Stage 1 数据画像\n\n本文件在本批概率模型预测结果生成前冻结。后续改变边界须创建 v2，保留 v1 主要分析。\n\n"
                                  "目标：只读两场面板，统计四 cutoff 数据支持与因果窗口。\n\n"
                                  "分母：issue、issue×turbine、issue×horizon×turbine 分开记录；目标有效性由 mask 与 finite 标签共同决定。\n\n"
                                  "低样本层保留并标 insufficient_n，不自动删除或合并。停机候选是事后诊断。\n\n"
                                  "交付：profile.json、冻结边界、窗口与条件计数、共同评分网格及哈希。\n\n"
                                  "复现：`python -m wpf_benchmark.profile_probability --datasets <面板父目录> --out <新的空输出目录>`\n\n"
                                  "ClawsGO Science Agent\n", encoding="utf-8")
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run_profile(args.datasets, args.out)


if __name__ == "__main__":
    main()
