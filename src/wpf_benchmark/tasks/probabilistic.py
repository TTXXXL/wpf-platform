"""Explicit causal-history probability tasks, leaving existing task JSON unchanged."""
from __future__ import annotations

import numpy as np

from .config import TaskConfig

CUTOFFS = ("2021-01-01", "2021-04-01", "2021-07-01", "2021-10-01")
INPUTS = ("wind_speed", "wind_dir_sin", "wind_dir_cos", "power")


def probability_task(meta, cutoff, L=144, H=36):
    if cutoff not in CUTOFFS:
        raise ValueError("Stage 0/1 requires a registered 2021 quarterly cutoff")
    index = CUTOFFS.index(cutoff)
    end = CUTOFFS[index + 1] if index < 3 else "2022-01-01"
    valid = str(np.datetime64(cutoff) - np.timedelta64(14, "D"))
    train_start = {"kelmarsh": "2016-04-15", "penmanshiel": "2016-09-01"}.get(meta["name"], meta["time_range"][0])
    if np.datetime64(train_start) < np.datetime64(meta["time_range"][0]):
        train_start = meta["time_range"][0]
    return TaskConfig.from_dict({
        "name": f"{meta['name']}_q_{cutoff}_L{L}_H{H}", "dataset": meta["name"],
        "target": "power", "target_input": "power", "inputs": list(INPUTS),
        "history_minutes": L * 10, "horizon_minutes": H * 10,
        "splits": {"train": [train_start, valid], "valid": [valid, cutoff], "test": [cutoff, end]},
        "train_mask": "observed", "valid_mask": "observed", "eval_mask": "observed",
        "train_stride_minutes": 60, "eval_stride_minutes": 10,
        "extra_eval_masks": [{"mask": m, "target": "power"} for m in ("normal_b", "normal_c") if m in meta["masks"]],
        "extra": {"window_protocol": "causal_history_v1", "probability_protocol": "unit_quantiles_v1"}})


def window_audit(task, split):
    issues = split.issues
    target_start = split.times[issues + 1]
    target_end = split.times[issues + task.H]
    start, end = task.bounds[split.name]
    crosses = (target_start < start) | (target_end >= end)
    if np.any(crosses) or np.any(split.issue_times < start) or np.any(split.issue_times >= end):
        raise ValueError(f"{split.name} has issue times or labels crossing split boundaries")
    return {"split": split.name, "L": task.L, "H": task.H, "n_issues": len(issues),
            "first_issue": str(split.issue_times[0]), "last_issue": str(split.issue_times[-1]),
            "history_start_min": str(split.times[issues[0] - task.L + 1]),
            "label_start_min": str(target_start[0]), "label_end_max": str(target_end[-1]),
            "history_borrowed": bool(split.times[issues[0] - task.L + 1] < start),
            "label_cross_boundary_count": int(crosses.sum())}
