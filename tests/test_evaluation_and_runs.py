"""Evaluator arithmetic, run directories, external scoring, studies and the CLI."""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd

from wpf_benchmark import cli, study
from wpf_benchmark.evaluation import evaluate, reference_persistence
from wpf_benchmark.runner import is_done, load_task_config, load_task_data, run_one, score_predictions
from wpf_benchmark.tasks.forecasts import align_forecasts


def _data(paths):
    return load_task_data(paths, load_task_config(paths, "synthetic_6h"))


def test_constant_offset_scores(synthetic_root):
    data = _data(synthetic_root)
    truth = data.test.future_values(data.test.target)
    pred = np.where(np.isfinite(truth), truth + 10.0, 0.0).astype(np.float32)
    s = evaluate(data.task, data.test, pred)["scores"]["observed"]
    assert abs(s["turbine"]["MAE_kW"] - 10.0) < 1e-4 and abs(s["turbine"]["ME_kW"] - 10.0) < 1e-4
    assert abs(s["turbine"]["NMAE_pct"] - 100 * 10 / 2050) < 1e-6
    full = s["farm"]["n"]
    assert full > 0 and abs(s["farm"]["NMAE_pct"] - 100 * 10 / 2050) < 1e-6
    assert all(abs(r["MAE_kW"] - 10.0) < 1e-4 for r in s["per_horizon"])


def test_persistence_has_zero_skill(synthetic_root):
    data = _data(synthetic_root)
    ref = reference_persistence(data.task, data.test)
    pred = np.repeat(np.nan_to_num(ref)[:, None, :], data.task.H, axis=1).astype(np.float32)
    turbine = evaluate(data.task, data.test, pred)["scores"]["observed"]["turbine"]
    assert abs(turbine["SS_MAE_pct"]) < 1e-9


def test_run_score_and_resume(synthetic_root, tmp_path):
    directory, run = run_one(synthetic_root, "synthetic_6h", "persistence", 0, save_predictions=True)
    assert is_done(directory) and run["summary"]["SS_MAE_pct"] == 0.0
    external = score_predictions(synthetic_root, "synthetic_6h", directory / "predictions.npz", tmp_path / "ext")
    saved = json.loads((directory / "metrics.json").read_text())
    assert external["scores"]["observed"]["turbine"]["MAE_kW"] == saved["scores"]["observed"]["turbine"]["MAE_kW"]
    plan = synthetic_root.configs / "studies" / "smoke.json"
    plan.parent.mkdir(exist_ok=True)
    plan.write_text(json.dumps({"name": "smoke", "tasks": ["synthetic_6h"], "models": ["persistence", "ridge"],
                                "seeds": [0, 1]}))
    lines = []
    study.run_study(synthetic_root, plan, log=lines.append)
    assert "2 done, 0 failed" in lines[-1]                       # deterministic models: one seed each
    lines.clear()
    study.run_study(synthetic_root, plan, log=lines.append)
    assert sum("skipped" in line for line in lines) == 2
    archive = study.package(synthetic_root, plan)
    names = zipfile.ZipFile(archive).namelist()
    assert "smoke/summary.csv" in names and any(n.endswith("ridge/seed0/metrics.json") for n in names)
    assert cli.main(["--root", str(synthetic_root.root), "study", "--plan", "smoke", "--preflight"]) == 0


def test_forecast_alignment_has_no_lookahead():
    runs = pd.date_range("2020-01-01", periods=4, freq="6h")
    rows = []
    for run in runs:
        for k in range(1, 13):
            rows.append({"issue_time": run, "available_time": run + pd.Timedelta(hours=2),
                         "valid_time": run + pd.Timedelta(hours=k), "entity": "A", "ws": run.hour * 100 + k})
    table = pd.DataFrame(rows)
    issues = np.array(["2020-01-01T07:00", "2020-01-01T08:00"], dtype="datetime64[us]")
    x = align_forecasts(table, issues, H=2, resolution_minutes=60, entities=["A"], variables=["ws"])
    np.testing.assert_array_equal(x[0, :, 0, 0], [8, 9])         # 06:00 run not yet available at 07:00
    np.testing.assert_array_equal(x[1, :, 0, 0], [603, 604])     # 06:00 run available from 08:00, leads 3 and 4
