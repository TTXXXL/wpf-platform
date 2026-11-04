"""Reserved: align issued forecasts (NWP) to forecast issue times without look-ahead.

Not wired into the runner yet; no dataset provides forecasts. See docs/平台接口.md section 9.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd


def align_forecasts(forecasts: pd.DataFrame, issue_times: np.ndarray, H: int, resolution_minutes: int,
                    entities: Sequence[str], variables: List[str]) -> np.ndarray:
    """Return x_future [I, H, N, K] from the latest run available at each issue time.

    ``forecasts`` has columns issue_time, available_time, valid_time, entity and ``variables``.
    For issue time t the run used is the one with the largest available_time <= t; its values
    at valid_time = t + h * resolution (h = 1..H) are returned, NaN where the run has none.
    """
    issue_times = np.asarray(issue_times, dtype="datetime64[us]")
    runs = (forecasts[["issue_time", "available_time"]].drop_duplicates("issue_time")
            .sort_values("available_time").reset_index(drop=True))
    available = runs["available_time"].to_numpy().astype("datetime64[us]")
    chosen = np.searchsorted(available, issue_times, side="right") - 1
    I, N, K = len(issue_times), len(entities), len(variables)
    out = np.full((I, H, N, K), np.nan, dtype=np.float32)
    has_run = chosen >= 0
    if not has_run.any():
        return out
    step = np.timedelta64(resolution_minutes, "m").astype("timedelta64[us]")
    rows, horizons = np.nonzero(np.broadcast_to(has_run[:, None], (I, H)))
    wanted = pd.DataFrame({
        "row": rows, "h": horizons,
        "issue_time": runs["issue_time"].to_numpy().astype("datetime64[us]")[chosen[rows]],
        "valid_time": issue_times[rows] + step * (horizons + 1)})
    wanted = wanted.loc[wanted.index.repeat(N)].reset_index(drop=True)
    wanted["n"] = np.tile(np.arange(N), len(wanted) // N)
    wanted["entity"] = np.asarray(entities, dtype=object)[wanted["n"].to_numpy()]
    table = forecasts.assign(issue_time=forecasts["issue_time"].astype("datetime64[us]"),
                             valid_time=forecasts["valid_time"].astype("datetime64[us]"),
                             entity=forecasts["entity"].astype(str))
    merged = wanted.merge(table[["issue_time", "valid_time", "entity"] + list(variables)],
                          on=["issue_time", "valid_time", "entity"], how="left")
    out[merged["row"].to_numpy(), merged["h"].to_numpy(), merged["n"].to_numpy()] = \
        merged[list(variables)].to_numpy(dtype=np.float32)
    return out
