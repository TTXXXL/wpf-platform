"""SDWPF (KDD Cup 2022, 134 turbines, 245 days, 10 min) adapter.

Two build modes give the same panel:
- ``mode="raw"`` (default) runs the legacy cleaning pipeline (sdwpf_clean.py) on the two raw CSVs;
- ``mode="legacy_clean"`` converts an existing legacy ``sdwpf_clean.parquet``.

The raw data has only day numbers; Day 1 is stored as the nominal date 1970-01-01.
Masks reproduce the legacy protocol: ``m1`` (cleaned power, no base flags, no official bad
rule) scores ``power``; ``m2`` (raw power and wind present, no official bad rule) scores
``power_obs``; ``power_finite`` is the legacy "all" table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.panel import Panel
from ..provenance import sha256_file
from . import sdwpf_clean

ADAPTER_VERSION = "1"
RATED_KW = 1550.0
EPOCH = np.datetime64("1970-01-01T00:00", "us")
BASE_FLAGS = ("m_missing", "m_imputed", "m_outlier", "f_fault", "f_curtail", "f_farm")
VARIABLES = {  # platform name: (legacy column, unit, description)
    "power": ("Patv", "kW", "Active power after cleaning: negatives set to 0, gaps and outliers imputed"),
    "power_obs": ("Patv_obs", "kW", "Raw active power with negatives set to 0; NaN where not recorded"),
    "wind_speed": ("Wspd", "m/s", "Nacelle wind speed (cleaned)"),
    "wind_dir_rel": ("Wdir", "deg", "Wind direction relative to the nacelle, wrapped to [-180, 180)"),
    "wind_dir_rel_sin": ("Wsin", "1", "sin(wind_dir_rel)"),
    "wind_dir_rel_cos": ("Wcos", "1", "cos(wind_dir_rel)"),
    "nacelle_dir": ("Ndir", "deg", "Nacelle direction (yaw angle)"),
    "temp_ambient": ("Etmp", "degC", "Ambient temperature"),
    "temp_internal": ("Itmp", "degC", "Nacelle internal temperature"),
    "pitch_1": ("Pab1", "deg", "Pitch angle, blade 1"),
    "pitch_2": ("Pab2", "deg", "Pitch angle, blade 2"),
    "pitch_3": ("Pab3", "deg", "Pitch angle, blade 3"),
    "reactive_power": ("Prtv", "kvar", "Reactive power"),
}
MASKS = {
    "m1": ("power", "Legacy M1: finite cleaned power, none of " + ", ".join(BASE_FLAGS) + ", not o_bad"),
    "m2": ("power_obs", "Legacy M2: raw power and wind present, not o_bad, finite power_obs"),
    "power_finite": ("power", "Legacy 'all' table: finite cleaned power"),
}


def _masks(cols: Dict[str, np.ndarray], flags: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    finite = np.isfinite(cols["Patv"])
    base = finite.copy()
    for flag in BASE_FLAGS:
        base &= ~flags[flag]
    not_bad = ~flags["o_bad"]
    return {"m1": base & not_bad, "m2": flags["raw_valid"] & not_bad & np.isfinite(cols["Patv_obs"]),
            "power_finite": finite}


def _entities(tids, loc: Optional[pd.DataFrame]):
    rows = []
    for tid in tids:
        row = {"id": str(int(tid)), "rated_kw": RATED_KW}
        if loc is not None:
            row.update(x=float(loc.loc[tid, "x"]), y=float(loc.loc[tid, "y"]))
        rows.append(row)
    return rows


def _panel(name, times, tids, cols, flags, loc, sources, build, train_days, fit=None) -> Tuple[Panel, str]:
    masks = _masks(cols, flags)
    values = {new: np.asarray(cols[old], dtype=np.float32) for new, (old, _, _) in VARIABLES.items()}
    meta: Dict[str, Any] = {
        "name": name, "title": "SDWPF (Baidu KDD Cup 2022), 134 turbines, 245 days", "adapter": "sdwpf",
        "adapter_version": ADAPTER_VERSION, "resolution_minutes": 10, "nominal_dates": True,
        "fit_window": [str(EPOCH)[:10], str(EPOCH + np.timedelta64(train_days, "D"))[:10]],
        "default_target": "power", "entities": _entities(tids, loc),
        "variables": {k: {"unit": u, "description": d, "source": old} for k, (old, u, d) in VARIABLES.items()},
        "masks": {k: {"target": t, "description": d, "fit_window": "fit_window"} for k, (t, d) in MASKS.items()},
        "sources": sources, "build": build}
    panel = Panel(meta, times, [e["id"] for e in meta["entities"]], values, masks, fit or {})
    lines = [f"# {name} build report", "", f"Mode: {build['mode']}; train_days: {train_days}; "
             f"grid {len(times)} x {len(tids)}.", "", "| item | count | share |", "|---|---|---|"]
    cells = values["power"].size
    for label, array in [("power NaN", ~np.isfinite(values["power"])), ("power_obs NaN", ~np.isfinite(values["power_obs"]))] + \
            [(f"flag {k}", v) for k, v in flags.items()] + [(f"mask {k}", v) for k, v in masks.items()]:
        lines.append(f"| {label} | {int(array.sum()):,} | {100 * array.sum() / cells:.3f}% |")
    return panel, "\n".join(lines) + "\n"


def _location(raw_dir: Path) -> Optional[pd.DataFrame]:
    path = raw_dir / sdwpf_clean.LOCATION_FILE
    return pd.read_csv(path, encoding="utf-8-sig").set_index("TurbID") if path.is_file() else None


def build(paths, name: str = "sdwpf", mode: str = "raw", source: Optional[str] = None, train_days=196,
          clean_parquet: Optional[str] = None) -> Tuple[Panel, str]:
    train_days = int(train_days)
    raw_dir = Path(source) if source else paths.raw / "sdwpf"
    if mode == "raw":
        result = sdwpf_clean.clean(raw_dir, train_days)
        cols = dict(result["wide"], Wsin=result["Wsin"], Wcos=result["Wcos"], Patv_obs=result["Patv_obs"])
        times = EPOCH + result["ts_index"].to_numpy().astype("timedelta64[us]")
        files = [raw_dir / sdwpf_clean.RAW_FILE, raw_dir / sdwpf_clean.LOCATION_FILE]
        fit = {k: np.asarray(v) for k, v in result["fit"].items()}
        fit["turbine_ids"] = np.asarray(result["tids"])
        return _panel(name, times, result["tids"], cols, result["flags"], result["loc"],
                      [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files],
                      {"mode": "raw", "train_days": train_days, "short_gap": sdwpf_clean.SHORT_GAP,
                       "knn_k": sdwpf_clean.KNN_K, "farm_days": result["farm_days"]}, train_days, fit)
    if mode != "legacy_clean":
        raise ValueError("mode must be 'raw' or 'legacy_clean'")
    path = Path(clean_parquet) if clean_parquet else paths.root / "data" / "processed" / "sdwpf_clean.parquet"
    df = pd.read_parquet(path)
    tids = np.sort(df["TurbID"].unique())
    T, N = len(df) // len(tids), len(tids)
    if len(df) != T * N or not (df["TurbID"].to_numpy().reshape(T, N) == tids[None, :]).all():
        raise ValueError("Legacy parquet must be sorted by (ts, TurbID) on a complete grid")
    wide = lambda c: df[c].to_numpy().reshape(T, N)
    cols = {old: wide(old) for old, _, _ in VARIABLES.values()}
    flags = {k: wide(k).astype(bool) for k in BASE_FLAGS + ("f_stuck", "o_pitch", "o_zero_windy",
                                                             "o_dir_abnormal", "o_bad", "raw_valid")}
    times = EPOCH + wide("ts")[:, 0].astype("timedelta64[us]")
    return _panel(name, times, tids, cols, flags, _location(raw_dir),
                  [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}],
                  {"mode": "legacy_clean", "train_days": train_days}, train_days)
