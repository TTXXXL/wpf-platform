"""Error metrics on boolean-selected cells. All reductions accumulate in float64."""
from __future__ import annotations

from typing import Dict

import numpy as np


def nan() -> float:
    return float("nan")


def point_metrics(err: np.ndarray, rated: np.ndarray) -> Dict[str, float]:
    """MAE, RMSE, NMAE (% of each cell's rated power), mean error and count."""
    n = int(err.size)
    if n == 0:
        return {"MAE_kW": nan(), "RMSE_kW": nan(), "NMAE_pct": nan(), "ME_kW": nan(), "n": 0}
    absolute = np.abs(err)
    return {"MAE_kW": float(absolute.mean(dtype=np.float64)),
            "RMSE_kW": float(np.sqrt(np.square(err).mean(dtype=np.float64))),
            "NMAE_pct": float(100.0 * (absolute / rated).mean(dtype=np.float64)),
            "ME_kW": float(err.mean(dtype=np.float64)), "n": n}


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else nan()


def skill(model_mae: float, reference_mae: float) -> float:
    return 100.0 * (1.0 - model_mae / reference_mae) if reference_mae > 0 else nan()
