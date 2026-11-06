"""Store, task resolution, splits, scaling and the persistence rule."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from wpf_benchmark.data.store import load_panel, write_dataset
from wpf_benchmark.datasets.synthetic import make_panel
from wpf_benchmark.runner import load_task_config, load_task_data, make_context
from wpf_benchmark.tasks import last_finite, resolve_task


def test_store_roundtrip(tmp_path):
    panel = make_panel(days=3, entities=2)
    meta = write_dataset(tmp_path / "ds", panel)
    back = load_panel(tmp_path / "ds", verify=True)
    assert meta["n_times"] == 432 and back.entities == panel.entities
    for name, array in panel.values.items():
        np.testing.assert_array_equal(back.values[name], array)
    for name, array in panel.masks.items():
        np.testing.assert_array_equal(back.masks[name], array)
    part = load_panel(tmp_path / "ds", ["power"], ["observed"], start="2020-01-02", end="2020-01-03")
    assert len(part.times) == 144 and str(part.times[0]).startswith("2020-01-02T00:00")


def test_store_rejects_undeclared_variable(tmp_path):
    panel = make_panel(days=2, entities=2)
    panel.values["extra"] = panel.values["power"].copy()
    with pytest.raises(ValueError):
        write_dataset(tmp_path / "ds", panel)


def test_split_windows_stay_inside_splits(synthetic_root):
    data = load_task_data(synthetic_root, load_task_config(synthetic_root, "synthetic_6h"))
    task = data.task
    for split in (data.train, data.valid):
        a, b = task.bounds[split.name]
        assert split.times[split.issues[0] - task.L + 1] >= a
        assert split.times[split.issues[-1] + task.H] < b
        assert np.all(np.diff(split.issues) == task.train_stride)
    test = data.test
    assert test.issue_times[0] == task.bounds["test"][0]
    assert test.times[0] < task.bounds["test"][0]                    # borrowed history
    assert test.times[test.issues[-1] + task.H] < task.bounds["test"][1]
    assert test.issues[-1] + task.H + task.eval_stride > len(test.times) - 1


def test_resolve_errors(synthetic_root):
    cfg = load_task_config(synthetic_root, "synthetic_6h")
    meta = load_task_data(synthetic_root, cfg).task.meta
    with pytest.raises(ValueError):
        resolve_task(dataclasses.replace(cfg, eval_mask="nope"), meta)
    with pytest.raises(ValueError):
        resolve_task(dataclasses.replace(cfg, horizon_minutes=365), meta)
    with pytest.raises(ValueError):
        resolve_task(cfg, dict(meta, fit_window=["2020-01-01", "2020-01-22"]))


def test_last_finite_and_scaler(synthetic_root):
    values = np.array([[1.0], [np.nan], [np.nan], [4.0], [np.nan]], dtype=np.float32)
    np.testing.assert_array_equal(last_finite(values, np.array([2, 4]), window=3)[:, 0], [1.0, 4.0])
    np.testing.assert_array_equal(last_finite(values, np.array([2]), window=2, fallback=-1)[:, 0], [-1.0])
    data = load_task_data(synthetic_root, load_task_config(synthetic_root, "synthetic_6h"))
    ctx = make_context(data)
    j = ctx.input_index("power")
    assert ctx.scaler.minimum["power"] == float(np.nanmin(data.train.inputs[:, :, j]))
    assert ctx.scaler.target_name == "power"
