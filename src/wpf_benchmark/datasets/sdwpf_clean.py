"""SDWPF cleaning pipeline, ported unchanged from the legacy ``preprocessing/sdwpf.py``.

Numerical steps are identical to the legacy code; plots and report text were removed.
Missing values: the first 3 steps of a gap use the causal previous value, later steps use
the same-time spatial kNN (k=12, inverse-squared distance weights). Outliers: robust power
curves and residual scales are fitted on the training days only and applied to all days.
Flags: f_fault (Patv<=0 at Wspd>=3), f_curtail (large negative residual at Wspd>=5),
f_stuck (wind speed constant for 6 steps), f_farm (days with >50% stopped cells).
Official KDD Cup rules (o_*) are captured from the raw values before any change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

FEATURES = ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "Patv"]
SHORT_GAP = 3
KNN_K = 12
STUCK_WIN = 6
CURT_WSPD = 5.0
SIGMA = 3.0
RAW_FILE = "sdwpf_245days_v1.csv"
LOCATION_FILE = "sdwpf_baidukddcup2022_turb_location.csv"


def load_raw(raw_dir: Path):
    df = pd.read_csv(Path(raw_dir) / RAW_FILE)
    df["ts"] = pd.to_timedelta(df["Day"] - 1, unit="D") + pd.to_timedelta(df["Tmstamp"] + ":00")
    df["TurbID"] = df["TurbID"].astype(int)
    loc = pd.read_csv(Path(raw_dir) / LOCATION_FILE, encoding="utf-8-sig").set_index("TurbID")
    tids = sorted(loc.index.to_numpy())
    wide = {c: df.pivot(index="ts", columns="TurbID", values=c)[tids].to_numpy(dtype=float).copy()
            for c in FEATURES}
    ts_index = df.pivot(index="ts", columns="TurbID", values="TurbID")[tids].index
    return loc.loc[tids], wide, ts_index, tids


def fill_short(X: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Fill only the first SHORT_GAP steps of each gap with the previous observation."""
    out = X.copy()
    T, N = X.shape
    for j in range(N):
        isna = missing[:, j]
        if not np.any(isna):
            continue
        positions = np.arange(T)
        last_observed = np.maximum.accumulate(np.where(isna, -1, positions))
        fillable = isna & (last_observed >= 0) & (positions - last_observed <= SHORT_GAP)
        previous = pd.Series(np.where(isna, np.nan, X[:, j])).ffill().to_numpy()
        out[fillable, j] = previous[fillable]
    return out


def knn_fill(Xm: np.ndarray, D: np.ndarray, k: int, todo: np.ndarray) -> np.ndarray:
    """Fill cells marked in ``todo`` with the inverse-squared-distance mean of k neighbours."""
    out = Xm.copy()
    T, N = Xm.shape
    order = np.argsort(D, axis=1)
    for j in range(N):
        rows = np.where(todo[:, j])[0]
        if len(rows) == 0:
            continue
        neigh = order[j, 1:k + 1]
        vals = Xm[:, neigh][rows]
        w = np.tile((1.0 / D[j, neigh] ** 2), (len(rows), 1))
        ok = np.isfinite(vals)
        w = w * ok
        den = w.sum(axis=1)
        num = np.nansum(vals * w, axis=1)
        out[rows, j] = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    return out


def impute(X: np.ndarray, D: np.ndarray):
    missing = ~np.isfinite(X)
    short = fill_short(X, missing)
    still = ~np.isfinite(short)
    filled = knn_fill(short, D, KNN_K, still & missing)
    return filled, missing & np.isfinite(filled)


def fit_power_curve(Wspd: np.ndarray, Patv: np.ndarray, excl: np.ndarray):
    """Per-turbine median power in 0.5 m/s bins; empty bins linearly interpolated."""
    edges = np.arange(0.0, 26.01, 0.5)
    centers = 0.5 * (edges[:-1] + edges[1:])
    N = Wspd.shape[1]
    curves = np.full((N, len(centers)), np.nan)
    for j in range(N):
        v, p = Wspd[:, j], Patv[:, j]
        ok = np.isfinite(v) & np.isfinite(p) & ~excl[:, j] & (p > 0) & (v >= 0.25) & (v < 26)
        if ok.sum() < 500:
            continue
        idx = np.clip(np.digitize(v[ok], edges) - 1, 0, len(centers) - 1)
        for b in range(len(centers)):
            sel = idx == b
            if sel.sum() >= 30:
                curves[j, b] = np.median(p[ok][sel])
        valid = np.isfinite(curves[j])
        if valid.sum() >= 2:
            curves[j] = np.interp(centers, centers[valid], curves[j][valid])
    return centers, curves


def curve_interp(v: np.ndarray, centers: np.ndarray, curve: np.ndarray) -> np.ndarray:
    out = np.interp(v, centers, curve, left=curve[0], right=curve[-1])
    out[~np.isfinite(v)] = np.nan
    return out


def residual_flags(Wspd, Patv, resid, f_fault, train_rows):
    """Robust residual scales fitted on training rows, then applied to every row."""
    m_curtail = np.zeros_like(f_fault)
    m_outlier = np.zeros_like(f_fault)
    edges = np.arange(0.0, 26.01, 1.0)
    global_sigma = np.full(Wspd.shape[1], np.nan)
    bin_sigma = np.full((Wspd.shape[1], len(edges) - 1), np.nan)
    for j in range(Wspd.shape[1]):
        v, r = Wspd[:, j], resid[:, j]
        ok = np.isfinite(v) & np.isfinite(r) & (Patv[:, j] > 0) & ~f_fault[:, j]
        train_ok = ok & train_rows
        train_resid = r[train_ok]
        if train_resid.size == 0:
            continue
        mad_all = max(float(np.median(np.abs(train_resid - np.median(train_resid))) * 1.4826), 5.0)
        global_sigma[j] = mad_all
        bin_sigma[j] = mad_all
        sigma = np.full(len(v), mad_all)
        for b in range(len(edges) - 1):
            in_bin = (v >= edges[b]) & (v < edges[b + 1])
            training_bin = train_ok & in_bin
            if training_bin.sum() >= 100:
                rb = r[training_bin]
                sig = np.median(np.abs(rb - np.median(rb))) * 1.4826
                sigma[in_bin] = max(sig, 5.0)
                bin_sigma[j, b] = max(sig, 5.0)
        z = r / np.where(sigma > 0, sigma, np.nan)
        cand = ok & np.isfinite(z)
        m_curtail[:, j] = cand & (z <= -SIGMA) & (v >= CURT_WSPD)
        m_outlier[:, j] = cand & (np.abs(z) > SIGMA) & (v >= 3.0) & ~m_curtail[:, j]
    return m_curtail, m_outlier, {"residual_edges": edges, "residual_global_sigma": global_sigma,
                                  "residual_bin_sigma": bin_sigma}


def clean(raw_dir: Path, train_days: int = 196) -> Dict[str, Any]:
    """Run the legacy pipeline; returns wide [T, N] arrays, flags and fitted objects."""
    loc, wide, ts_index, tids = load_raw(raw_dir)
    T, N = wide["Wspd"].shape
    # Official KDD Cup rules, from raw values before wrapping or imputation.
    o_pitch = (wide["Pab1"] > 89) | (wide["Pab2"] > 89) | (wide["Pab3"] > 89)
    o_zero_windy = (wide["Patv"] <= 0) & (wide["Wspd"] > 2.5)
    o_dir_abnormal = (np.abs(wide["Ndir"]) > 720) | (np.abs(wide["Wdir"]) > 180)
    raw_valid = np.isfinite(wide["Patv"]) & np.isfinite(wide["Wspd"])
    patv_obs = np.maximum(wide["Patv"], 0).astype(np.float32)
    train_rows = np.asarray(ts_index < pd.Timedelta(days=train_days))
    if not train_rows.any() or train_rows.all():
        raise ValueError("Training boundary must leave both training and later rows")
    xy = loc[["x", "y"]].to_numpy()
    D = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    missing0 = {c: ~np.isfinite(wide[c]) for c in FEATURES}
    for c in ["Etmp", "Itmp"]:
        bad = np.isfinite(wide[c]) & ((wide[c] < -40) | (wide[c] > 60))
        wide[c][bad] = np.nan
    bad = np.isfinite(wide["Wspd"]) & ((wide["Wspd"] < 0) | (wide["Wspd"] > 40))
    wide["Wspd"][bad] = np.nan
    wide["Wdir"] = ((wide["Wdir"] + 180.0) % 360.0) - 180.0
    wide["Patv"][wide["Patv"] < 0] = 0.0
    imputed = {}
    for c in FEATURES:
        wide[c], imputed[c] = impute(wide[c], D)
    f_fault = np.isfinite(wide["Patv"]) & np.isfinite(wide["Wspd"]) & (wide["Patv"] <= 0) & (wide["Wspd"] >= 3.0)
    centers, curves = fit_power_curve(wide["Wspd"][train_rows], wide["Patv"][train_rows], excl=f_fault[train_rows])
    resid = np.full_like(wide["Patv"], np.nan)
    for j in range(N):
        ok = np.isfinite(wide["Patv"][:, j]) & np.isfinite(wide["Wspd"][:, j])
        resid[np.where(ok)[0], j] = wide["Patv"][ok, j] - curve_interp(wide["Wspd"][ok, j], centers, curves[j])
    m_curtail, m_outlier, fitted = residual_flags(wide["Wspd"], wide["Patv"], resid, f_fault, train_rows)
    wide["Patv"][m_outlier] = np.nan
    wide["Patv"], imp2 = impute(wide["Patv"], D)
    f_stuck = np.zeros_like(f_fault)
    for j in range(N):
        s = pd.Series(wide["Wspd"][:, j])
        size = s.groupby((s != s.shift()).cumsum()).transform("size").to_numpy()
        f_stuck[:, j] = (size >= STUCK_WIN) & np.isfinite(s.to_numpy())
    day = ts_index.days.to_numpy() + 1
    daily_fault = pd.DataFrame({"day": day, "fault": f_fault.mean(axis=1)}).groupby("day")["fault"].mean()
    farm_days = [int(d) for d in daily_fault[daily_fault > 0.5].index]
    f_farm = np.broadcast_to(np.isin(day, farm_days)[:, None], (T, N))
    rad = np.deg2rad(wide["Wdir"])
    flags = {"m_missing": np.stack([missing0[c] for c in FEATURES], axis=2).any(axis=2),
             "m_imputed": np.stack([imputed[c] for c in FEATURES], axis=2).any(axis=2) | imp2,
             "m_outlier": m_outlier, "f_fault": f_fault, "f_curtail": m_curtail, "f_stuck": f_stuck,
             "f_farm": f_farm, "o_pitch": o_pitch, "o_zero_windy": o_zero_windy,
             "o_dir_abnormal": o_dir_abnormal, "o_bad": o_pitch | o_zero_windy | o_dir_abnormal,
             "raw_valid": raw_valid}
    return {"ts_index": ts_index, "tids": tids, "loc": loc, "wide": wide, "Wsin": np.sin(rad), "Wcos": np.cos(rad),
            "Patv_obs": patv_obs, "flags": flags, "farm_days": farm_days,
            "fit": {"curve_centers": centers, "curves": curves, **fitted}, "train_days": train_days}
