"""Regression tests for the 2026-11-04 code review (R1-R7, R10, R11): result identity, data checks, scoring."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wpf_benchmark import cli, study
from wpf_benchmark.data.panel import Panel
from wpf_benchmark.datasets import build_dataset
from wpf_benchmark.paths import ProjectPaths
from wpf_benchmark.runner import RunLocked, load_task_config, load_task_data, run_one

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fresh(tmp_path) -> ProjectPaths:
    """A private project root; tests here change configs and data, so they must not share one."""
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    paths = ProjectPaths.resolve(tmp_path)
    build_dataset("synthetic", paths)
    return paths


def write_plan(paths, models, name="p", seeds=(0,), save_predictions=False):
    plan = paths.configs / "studies" / f"{name}.json"
    plan.parent.mkdir(exist_ok=True)
    plan.write_text(json.dumps({"name": name, "tasks": ["synthetic_6h"], "models": models, "seeds": list(seeds),
                                "save_predictions": save_predictions}))
    return plan


def run_json(paths, name, label, seed=0):
    path = paths.results / "studies" / name / "runs" / "synthetic_6h" / label / f"seed{seed}" / "run.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ----- R1: resuming must not reuse a result made from another configuration -----
def test_changed_model_parameters_are_not_skipped(fresh):
    write_plan(fresh, [{"label": "r", "model": "ridge", "params": {"alpha": 1.0}}])
    study.run_study(fresh, fresh.configs / "studies" / "p.json", log=lambda s: None)
    plan = write_plan(fresh, [{"label": "r", "model": "ridge", "params": {"alpha": 1000.0}}])
    with pytest.raises(RuntimeError, match="params"):
        study.run_study(fresh, plan, log=lambda s: None)
    assert run_json(fresh, "p", "r")["params"]["alpha"] == 1.0                 # nothing was overwritten
    study.run_study(fresh, plan, log=lambda s: None, rerun_stale=True)
    assert run_json(fresh, "p", "r")["params"]["alpha"] == 1000.0


def test_changed_task_is_not_skipped(fresh):
    plan = write_plan(fresh, ["persistence"])
    study.run_study(fresh, plan, log=lambda s: None)
    path = fresh.configs / "tasks" / "synthetic_6h.json"
    raw = json.loads(path.read_text())
    raw["history_minutes"] = 720
    path.write_text(json.dumps(raw))
    with pytest.raises(RuntimeError, match="task"):
        study.run_study(fresh, plan, log=lambda s: None)


def test_unchanged_run_is_skipped_and_new_models_may_be_added(fresh):
    plan = write_plan(fresh, ["persistence"])
    study.run_study(fresh, plan, log=lambda s: None)
    plan = write_plan(fresh, ["persistence", "ridge"])
    lines = []
    study.run_study(fresh, plan, log=lines.append)
    assert sum("done, skipped" in line for line in lines) == 1 and "2 done, 0 failed" in lines[-1]


# ----- R2: the data file is hashed, not just the declared hash read -----
def test_changed_panel_file_is_rejected(fresh):
    path = fresh.dataset("synthetic") / "panel.parquet"
    table = pq.read_table(path)
    power = table["power"].to_numpy().copy()
    power[:1000] += 1.0
    column = pa.chunked_array([pa.array(power, type=pa.float32())])
    pq.write_table(table.set_column(table.schema.get_field_index("power"), "power", column), path)
    with pytest.raises(ValueError, match="SHA256"):
        load_task_data(fresh, load_task_config(fresh, "synthetic_6h"))


def test_run_records_verified_data_and_metadata_hashes(fresh):
    _, run = run_one(fresh, "synthetic_6h", "persistence", 0)
    meta = json.loads((fresh.dataset("synthetic") / "meta.json").read_text())
    assert run["dataset_sha256"] == meta["panel_sha256"]
    assert len(run["dataset_meta_sha256"]) == 64 and run["identity"]["parts"]["dataset"]["panel_sha256"]


def test_changed_metadata_changes_identity(fresh):
    before = run_one(fresh, "synthetic_6h", "persistence", 0)[1]["identity"]["sha256"]
    path = fresh.dataset("synthetic") / "meta.json"
    meta = json.loads(path.read_text())
    meta["entities"][0]["rated_kw"] = 1234.0
    path.write_text(json.dumps(meta))
    after = run_one(fresh, "synthetic_6h", "persistence", 0)[1]["identity"]["sha256"]
    assert before != after


# ----- R10: DONE is a verified manifest, reruns start clean, one writer per directory -----
def test_missing_run_json_is_not_done_and_is_rerun(fresh):
    plan = write_plan(fresh, ["persistence"])
    study.run_study(fresh, plan, log=lambda s: None)
    out = fresh.results / "studies" / "p" / "runs" / "synthetic_6h" / "persistence" / "seed0"
    (out / "run.json").unlink()
    assert not run_status_done(out)
    lines = []
    study.run_study(fresh, plan, log=lines.append)
    assert not any("done, skipped" in line for line in lines) and (out / "run.json").is_file()


def test_edited_metrics_are_detected(fresh):
    directory, _ = run_one(fresh, "synthetic_6h", "persistence", 0)
    assert run_status_done(directory)
    with (directory / "metrics.json").open("a") as handle:
        handle.write(" ")
    assert not run_status_done(directory)


def test_rerun_removes_optional_files_of_the_earlier_run(fresh):
    directory, _ = run_one(fresh, "synthetic_6h", "persistence", 0, save_predictions=True)
    assert (directory / "predictions.npz").is_file()
    run_one(fresh, "synthetic_6h", "persistence", 0, save_predictions=False)
    assert not (directory / "predictions.npz").exists()


def test_second_writer_is_refused(fresh):
    directory, _ = run_one(fresh, "synthetic_6h", "persistence", 0)
    (directory / "RUNNING").write_text("pid 1 on elsewhere")
    with pytest.raises(RunLocked, match="another process"):
        run_one(fresh, "synthetic_6h", "persistence", 0)
    (directory / "RUNNING").unlink()
    run_one(fresh, "synthetic_6h", "persistence", 0)
    assert not (directory / "RUNNING").exists()                      # released after a normal run


def run_status_done(directory) -> bool:
    from wpf_benchmark.runner import is_done
    return is_done(directory)


# ----- R3: a day without scorable cells is not a zero-error day -----
def test_daily_energy_skips_days_without_valid_cells_and_reports_coverage(synthetic_root):
    from wpf_benchmark.evaluation.evaluator import _daily_energy
    data = load_task_data(synthetic_root, load_task_config(synthetic_root, "synthetic_6h"))
    task, test = data.task, data.test
    shape = (len(test.issues), task.H, len(test.entities))
    truth = np.full(shape, 1000.0, np.float32)
    pred = truth + 50.0
    valid = np.ones(shape, bool)
    days = (test.issue_times + np.timedelta64(task.resolution_minutes, "m")).astype("datetime64[D]")
    unique = np.unique(days)
    valid[days == unique[1]] = False                                  # a whole day with no valid target
    valid[days == unique[2], :, 1:] = False                           # a day with one of three turbines scorable
    result = _daily_energy(task, test, pred, truth, valid)
    listed = [r["day"] for r in result["per_day"]]
    assert str(unique[1]) not in listed and result["n_days_no_valid_cell"] == 1
    by_day = {r["day"]: r for r in result["per_day"]}
    assert abs(by_day[str(unique[2])]["coverage"] - 1 / 3) < 1e-9
    assert result["n_days_below_min_coverage"] >= 1
    assert result["covered"]["n_days"] == result["n_days"] - result["n_days_below_min_coverage"]
    assert result["MAE_MWh"] > 0 and np.isfinite(result["MAE_MWh"])


# ----- R7: the ramp score separates "a large change happened" from "the right direction" -----
def test_ramp_hit_in_the_wrong_direction_is_reported_as_such(synthetic_root):
    from wpf_benchmark.evaluation.evaluator import _conditional
    data = load_task_data(synthetic_root, load_task_config(synthetic_root, "synthetic_6h"))
    task, test = data.task, data.test
    shape = (len(test.issues), task.H, len(test.entities))
    rated = np.broadcast_to(task.rated_kw[None, None, :], shape)
    reference = np.full(shape, 1000.0, np.float32)
    truth, pred = reference + 300.0, reference - 300.0              # real rise, predicted fall
    out = _conditional(task, test, pred, truth, np.ones(shape, bool), pred - truth, rated, reference, None)
    assert "ramp" not in out
    detect = out["ramp_change_detection"]
    assert detect["precision"] == detect["recall"] == detect["f1"] == 1.0
    directional = out["ramp_directional"]
    assert directional["up"]["recall"] == 0.0 and directional["down"]["precision"] == 0.0
    assert directional["direction_accuracy"] == 0.0 and directional["amplitude_bias_kW"] == 0.0


# ----- R4: preflight finds runs that cannot start, and lists every problem -----
def test_preflight_reports_missing_data_file(fresh):
    plan = write_plan(fresh, ["persistence"])
    (fresh.dataset("synthetic") / "panel.parquet").unlink()
    report = study.preflight(fresh, study.load_plan(plan))
    assert report["problems"] and "synthetic_6h" in report["problems"][0]["task"]
    assert cli.main(["--root", str(fresh.root), "study", "--plan", "p", "--preflight"]) == 1


def test_preflight_builds_networks_and_lists_all_failures(fresh):
    plan = write_plan(fresh, [{"label": "tooBig", "model": "patchtst", "params": {"patch": 500}},
                              {"label": "typo", "model": "ridge", "params": {"alhpa": 1}},
                              {"label": "fine", "model": "dlinear", "params": {"epochs": 1}}, "persistence"])
    report = study.preflight(fresh, study.load_plan(plan))
    broken = {p["model"]: p["problem"] for p in report["problems"]}
    assert set(broken) == {"tooBig", "typo"} and "patch" in broken["tooBig"] and "alhpa" in broken["typo"]
    assert report["n_runs"] == 2 and report["n_neural"] == 1                       # dlinear and persistence only
    assert report["tasks"]["synthetic_6h"]["test"]["windows"] > 0


def test_preflight_passes_a_valid_plan(fresh):
    report = study.preflight(fresh, study.load_plan(write_plan(fresh, ["persistence", "ridge"])))
    assert report["problems"] == [] and report["n_runs"] == 2
    assert cli.main(["--root", str(fresh.root), "study", "--plan", "p", "--preflight"]) == 0


# ----- R5: a checkpoint is restored only for the task meaning it was trained for -----
def _task(paths, **changes):
    import dataclasses
    from wpf_benchmark.data.store import load_meta
    from wpf_benchmark.tasks import resolve_task
    cfg = dataclasses.replace(load_task_config(paths, "synthetic_6h"), **changes)
    return resolve_task(cfg, load_meta(paths.dataset("synthetic")))


def test_checkpoint_rejects_a_changed_target_input(fresh):
    from wpf_benchmark.models.neural.trainer import load_checkpoint
    directory, _ = run_one(fresh, "synthetic_6h", {"model": "dlinear", "params": {"epochs": 1}}, 0)
    path = directory / "checkpoint.pt"
    assert load_checkpoint(path, _task(fresh), device="cpu").ctx.target_input == "power"
    with pytest.raises(ValueError, match="target_input"):
        load_checkpoint(path, _task(fresh, target_input="wind_speed"), device="cpu")
    reused = load_checkpoint(path, _task(fresh, target_input="wind_speed"), device="cpu", strict=False)
    assert reused.ctx.target_input == "wind_speed"


def test_checkpoint_rejects_a_changed_history_length(fresh):
    from wpf_benchmark.models.neural.trainer import load_checkpoint
    directory, _ = run_one(fresh, "synthetic_6h", {"model": "dlinear", "params": {"epochs": 1}}, 0)
    with pytest.raises(ValueError, match="history_minutes"):
        load_checkpoint(directory / "checkpoint.pt", _task(fresh, history_minutes=720), device="cpu")


# ----- R6: grid step, declared resolution, ids and rated power must agree -----
def _panel(step_minutes=10, resolution=10, ids=("a", "b"), rated=(2000.0, 2000.0)):
    times = np.datetime64("2020-01-01", "us") + np.arange(6) * np.timedelta64(step_minutes, "m")
    meta = {"resolution_minutes": resolution, "entities": [{"id": i, "rated_kw": r} for i, r in zip(ids, rated)]}
    return Panel(meta, times, list(ids), {"power": np.zeros((6, len(ids)), np.float32)})


def test_grid_step_must_match_declared_resolution():
    _panel()
    with pytest.raises(ValueError, match="resolution_minutes"):
        _panel(step_minutes=10, resolution=20)


def test_duplicate_ids_and_nonpositive_rated_power_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        _panel(ids=("a", "a"))
    with pytest.raises(ValueError, match="rated_kw"):
        _panel(rated=(2000.0, 0.0))


@pytest.mark.parametrize("minutes", [45, 0, 720])
def test_report_horizons_must_divide_and_fit(fresh, minutes):
    with pytest.raises(ValueError, match="report_horizons_minutes|exceeds"):
        _task(fresh, report_horizons_minutes=(60, minutes))


# ----- R11: adapter options arrive as text and are parsed with the option's name in the error -----
def test_numeric_adapter_options_are_validated(fresh, capsys):
    with pytest.raises(ValueError, match="days"):
        build_dataset("synthetic", fresh, days="many")
    with pytest.raises(ValueError, match="workers"):
        build_dataset("kelmarsh", fresh, source=str(fresh.root), workers="0")
    with pytest.raises(ValueError, match="start"):
        build_dataset("kelmarsh", fresh, source=str(fresh.root), start="yesterday")
    with pytest.raises(FileNotFoundError):                       # "2" parses; the empty source then fails later
        build_dataset("kelmarsh", fresh, source=str(fresh.root), workers="2")
    assert cli.main(["--root", str(fresh.root), "build", "synthetic", "-o", "days=4", "-o", "entities=2"]) == 0


def test_duplicate_timestamp_keeps_first_row_and_counts_filled_cells(tmp_path):
    from wpf_benchmark.datasets import greenbyte
    header = ["Date and time", "Wind speed (m/s)", "Power (kW)"]
    rows = [["2021-01-01 00:00:00", "5.0", "NaN"], ["2021-01-01 00:00:00", "NaN", "900.0"],
            ["2021-01-01 00:10:00", "6.0", "800.0"]]
    text = "# Turbine: T\n#\n" * 4 + "# " + ",".join(header) + "\n" + "\n".join(",".join(r) for r in rows) + "\n"
    archive = tmp_path / "Kelmarsh_SCADA_test.zip"
    import zipfile
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("Turbine_Data_Kelmarsh_1_2021-01-01_-_2021-01-02_1.csv", text)
    res = greenbyte.read_scada_member((str(archive), "Turbine_Data_Kelmarsh_1_2021-01-01_-_2021-01-02_1.csv"))
    assert res["raw_rows"] == 3 and res["unique_rows"] == 2
    assert res["wind_speed"][0] == 5.0 and np.isnan(res["power_raw"][0])      # first row kept, not merged
    assert res["dup_filled_cells"] == 1 and res["power_raw"][1] == 800.0
