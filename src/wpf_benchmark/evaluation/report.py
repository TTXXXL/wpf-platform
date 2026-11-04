"""Strict JSON output and flat tables derived from metrics."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, data: Any) -> None:
    """Write strict JSON (NaN becomes null) atomically."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(data), ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    os.replace(temporary, path)


def per_horizon_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for mask, scores in metrics["scores"].items():
        for row in scores["per_horizon"]:
            rows.append({"mask": mask, "horizon": row["horizon"], "minutes": row["minutes"],
                         "MAE_kW": row["MAE_kW"], "RMSE_kW": row["RMSE_kW"], "NMAE_pct": row["NMAE_pct"],
                         "persistence_MAE_kW": row["persistence_MAE_kW"], "n": row["n"]})
    return rows


def summary_row(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """One flat row on the task's eval_mask: the numbers used to compare models."""
    main = metrics["scores"][metrics["eval_mask"]]
    t, f = main["turbine"], main["farm"]
    row = {"eval_mask": metrics["eval_mask"], "turbine_MAE_kW": t["MAE_kW"], "turbine_RMSE_kW": t["RMSE_kW"],
           "turbine_NMAE_pct": t["NMAE_pct"], "turbine_macro_MAE_kW": t["macro_MAE_kW"],
           "SS_MAE_pct": t["SS_MAE_pct"], "persistence_MAE_kW": t["persistence_MAE_kW"], "n": t["n"],
           "farm_MAE_kW": f["MAE_kW"], "farm_NMAE_pct": f["NMAE_pct"], "farm_n": f["n"]}
    for r in main["report_horizons"]:
        row[f"MAE_kW_{r['minutes']}min"] = r["MAE_kW"]
    return row
