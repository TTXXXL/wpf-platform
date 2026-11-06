"""Release checks for resume, offline delivery and preflight validation."""
from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from wpf_benchmark import study
from wpf_benchmark.datasets import build_dataset
from wpf_benchmark.paths import ProjectPaths
from wpf_benchmark.runner import load_task_config, load_task_data

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    paths = ProjectPaths.resolve(tmp_path)
    build_dataset("synthetic", paths)
    raw = json.loads((paths.configs / "tasks" / "synthetic_6h.json").read_text(encoding="utf-8"))
    raw.update(history_minutes=60, horizon_minutes=20, report_horizons_minutes=[10, 20])
    (paths.configs / "tasks" / "synthetic_6h.json").write_text(json.dumps(raw), encoding="utf-8")
    return paths


def plan_file(paths, models=None):
    path = paths.configs / "studies" / "release.json"
    path.write_text(json.dumps({"name": "release", "tasks": ["synthetic_6h"],
                                "models": models or ["persistence"]}), encoding="utf-8")
    return path


@pytest.mark.parametrize("model,params", [
    ("seasonal_persistence", {}),
    ("ridge", {"inputs": ["missing_feature"]}),
    ("lightgbm", {"lags": 12}),
])
def test_preflight_rejects_non_neural_incompatible_tasks(project, model, params):
    plan = study.load_plan(plan_file(project, [{"model": model, "params": params}]))
    report = study.preflight(project, plan)
    assert report["problems"] and report["n_runs"] == 0


def test_preflight_counts_labels_in_forecast_windows(project):
    data = load_task_data(project, load_task_config(project, "synthetic_6h"))
    mask = data.train.target_mask.copy()
    mask[:] = False
    mask[:data.task.L] = True  # only the initial history, never any training target
    data.train = dataclasses.replace(data.train, target_mask=mask)
    problems = []
    description = study._describe_task(data, problems, "synthetic_6h")
    assert description["train"]["labelled_target_cells"] == 0
    assert any("train" in problem["problem"] for problem in problems)


def test_summary_does_not_suppress_data_integrity_errors(project):
    path = plan_file(project)
    directory = study.run_study(project, path, log=lambda _: None)
    with (project.dataset("synthetic") / "panel.parquet").open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(ValueError, match="SHA256"):
        study.write_summary(directory, study.load_plan(path), project)


def test_offline_summary_still_checks_configuration(project):
    path = plan_file(project, [{"model": "ridge", "params": {"alpha": 1}}])
    directory = study.run_study(project, path, log=lambda _: None)
    (project.dataset("synthetic") / "panel.parquet").unlink()
    path = plan_file(project, [{"model": "ridge", "params": {"alpha": 1000}}])
    summary = study.write_summary(directory, study.load_plan(path), project)
    row = pd.read_csv(summary).iloc[0]
    assert row["status"] == "stale"
    assert "turbine_MAE_kW" not in row.index or pd.isna(row["turbine_MAE_kW"])


def test_study_can_add_a_task_using_another_dataset(project):
    path = plan_file(project)
    study.run_study(project, path, log=lambda _: None)
    source = project.dataset("synthetic")
    destination = project.dataset("synthetic_extra")
    shutil.copytree(source, destination)
    meta_path = destination / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["name"] = "synthetic_extra"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    task = json.loads((project.configs / "tasks" / "synthetic_6h.json").read_text(encoding="utf-8"))
    task.update(name="extra", dataset="synthetic_extra")
    (project.configs / "tasks" / "extra.json").write_text(json.dumps(task), encoding="utf-8")
    plan = study.load_plan(path)
    plan["tasks"].append("extra")
    path.write_text(json.dumps(plan), encoding="utf-8")
    lines = []
    study.run_study(project, path, log=lines.append)
    assert sum("done, skipped" in line for line in lines) == 1
    assert "2 done, 0 failed" in lines[-1]
