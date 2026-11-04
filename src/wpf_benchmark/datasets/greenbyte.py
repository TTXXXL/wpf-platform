"""Kelmarsh and Penmanshiel (Greenbyte export) adapter: 10-minute SCADA zips plus status logs."""
from __future__ import annotations

import concurrent.futures as cf
import csv
import io
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.panel import Panel
from ..provenance import code_digest, sha256_file

ADAPTER_VERSION = "1"
RAW_ROOT = Path("D:/codex/风电预测/datasets/OpenWPF_8datasets")
FARMS = {
    "kelmarsh": {"title": "Kelmarsh wind farm (UK)", "folder": "02_Kelmarsh_UK", "static": "Kelmarsh_WT_static.csv",
                 "start": "2016-01-03", "cod": "2016-04-15"},
    "penmanshiel": {"title": "Penmanshiel wind farm (UK)", "folder": "03_Penmanshiel_UK",
                    "static": "Penmanshiel_WT_static.csv", "start": "2016-06-02", "cod": "2016-09-01"},
}
GRID_END = "2025-01-01"
STEP_MIN = 10
MAX_WORKERS = 4
CLIP_TOL_KW = 1e-3                      # power differences below this are not reported as conflicts
MEMBER_RE = re.compile(r"^(Turbine_Data|Status)_([A-Za-z]+)_(\d+)_(\d{4}-\d{2}-\d{2})_-_(\d{4}-\d{2}-\d{2})_")
EXACT = {"wind_speed": "Wind speed (m/s)", "power_raw": "Power (kW)",
         "lp_downtime": "Lost Production to Downtime (kWh)",
         "lp_curtailment": "Lost Production to Curtailment (Total) (kWh)", "rotor_rpm": "Rotor speed (RPM)"}
PREFIX = {"wind_dir": "Wind direction (", "nacelle_dir": "Nacelle position (",
          "temp_ambient": "Nacelle ambient temperature (", "pitch": "Blade angle (pitch position) A ("}
RAW_VARS = ["wind_speed", "wind_dir", "nacelle_dir", "power_raw", "temp_ambient", "pitch", "rotor_rpm",
            "lp_downtime", "lp_curtailment"]
IEC_BAD = {"Forced outage", "Scheduled Maintenance", "Requested Shutdown", "Out of Electrical Specification",
           "Partial Performance", "Technical Standby", "Out of Environmental Specification"}
STATUS_COLS = ["Timestamp start", "Timestamp end", "Status", "Code", "IEC category"]
VARIABLES = {
    "power": ("kW", "Active power, negative values clipped to 0 (target)", "Power (kW)"),
    "power_raw": ("kW", "Active power as recorded, including standby consumption", "Power (kW)"),
    "wind_speed": ("m/s", "Nacelle anemometer wind speed", "Wind speed (m/s)"),
    "wind_dir": ("deg", "Absolute wind direction", "Wind direction"),
    "wind_dir_sin": ("1", "sin(absolute wind direction)", "derived from wind_dir"),
    "wind_dir_cos": ("1", "cos(absolute wind direction)", "derived from wind_dir"),
    "nacelle_dir": ("deg", "Nacelle position (yaw)", "Nacelle position"),
    "temp_ambient": ("degC", "Nacelle ambient temperature", "Nacelle ambient temperature"),
    "pitch": ("deg", "Blade pitch angle, blade A", "Blade angle (pitch position) A"),
    "rotor_rpm": ("rpm", "Rotor speed", "Rotor speed (RPM)"),
    "lp_downtime": ("kWh", "Lost production to downtime (computed after the fact; mask use only)", "Lost Production to Downtime (kWh)"),
    "lp_curtailment": ("kWh", "Lost production to curtailment, total (computed after the fact; mask use only)",
                       "Lost Production to Curtailment (Total) (kWh)"),
}
MASKS = {
    "observed": "power_raw and wind_speed both present",
    "normal_b": "rule B: observed and Greenbyte lost production to downtime and to curtailment both present and zero",
    "normal_c": "rule C: observed and no overlap with Stop/Curtailment/Communication, non-full-performance IEC "
                "events (except Informational 10, 65) or Informational 245",
}


def _head(zip_path: str, member: str, n: int = 40) -> List[str]:
    with zipfile.ZipFile(zip_path) as z, z.open(member) as f:
        return [f.readline().decode("utf-8", "replace").lstrip("\ufeff") for _ in range(n)]


def read_scada_member(job: Tuple[str, str]) -> Dict[str, Any]:
    """Read one Turbine_Data member: needed columns only, duplicate timestamps collapsed to the first row."""
    zip_path, member = job
    head = _head(zip_path, member)
    at = next(i for i, line in enumerate(head) if line.startswith("# Date and time") or line.startswith("Date and time"))
    cols = next(csv.reader([head[at].lstrip("# ").rstrip("\r\n")]))
    picked, missing = {}, []
    for key in RAW_VARS:
        if key in EXACT:
            hits = [i for i, c in enumerate(cols) if c == EXACT[key]]
        else:
            hits = [i for i, c in enumerate(cols) if c.startswith(PREFIX[key]) and "," not in c]
        if hits:
            picked[key] = hits[0]
        else:
            missing.append(key)
    order = sorted(picked, key=picked.get)
    with zipfile.ZipFile(zip_path) as z, z.open(member) as f:
        df = pd.read_csv(f, skiprows=at + 1, header=None, usecols=[0] + [picked[k] for k in order],
                         names=["ts"] + order, na_values=["NaN"], dtype={0: str, **{picked[k]: np.float32 for k in order}},
                         engine="c")
    raw_rows = len(df)
    df["ts"] = pd.to_datetime(df["ts"], format="%Y-%m-%d %H:%M:%S")
    df = df.dropna(subset=["ts"])
    unique = df["ts"].nunique()
    df = df.groupby("ts", sort=True).first() if unique < len(df) else df.sort_values("ts").set_index("ts")
    out = {"times": df.index.to_numpy().astype("datetime64[s]"), "raw_rows": raw_rows, "unique_rows": int(unique),
           "missing_cols": missing}
    for k in RAW_VARS:
        out[k] = df[k].to_numpy(np.float32) if k in df else np.full(len(df), np.nan, np.float32)
    return out


def read_status_member(job: Tuple[str, str]) -> pd.DataFrame:
    """Read one Status member into the five columns the masks need."""
    zip_path, member = job
    head = _head(zip_path, member)
    at = next(i for i, line in enumerate(head) if line.startswith("Timestamp start"))
    with zipfile.ZipFile(zip_path) as z, z.open(member) as f:
        text = io.TextIOWrapper(f, encoding="utf-8-sig", errors="replace").read()
    df = pd.read_csv(io.StringIO(text), skiprows=at, dtype=str, keep_default_na=False, usecols=STATUS_COLS)
    return df[STATUS_COLS]


def hash_file(path: str) -> Tuple[str, int, str]:
    return path, Path(path).stat().st_size, sha256_file(Path(path))


def discover(source: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Find usable SCADA zips and list members; unfinished or unreadable files go to the skipped list."""
    zips, skipped = [], []
    for path in sorted(Path(source).rglob("*SCADA*")):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".zip":
            skipped.append({"path": str(path), "reason": "not a finished .zip (unfinished download)"})
            continue
        try:
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
        except (zipfile.BadZipFile, OSError) as error:
            skipped.append({"path": str(path), "reason": f"unreadable zip: {error}"})
            continue
        members = []
        for name in names:
            m = MEMBER_RE.match(Path(name).name)
            if m:
                members.append({"member": name, "kind": "scada" if m.group(1) == "Turbine_Data" else "status",
                                "wt": int(m.group(3)), "start": np.datetime64(m.group(4), "s"),
                                "end": np.datetime64(m.group(5), "s")})
        zips.append({"path": str(path), "members": members})
    return zips, skipped


def mark_superseded(zips: List[Dict[str, Any]]) -> None:
    """A member is superseded when another one for the same turbine and kind covers it and spans longer."""
    flat = [(z, m) for z in zips for m in z["members"]]
    for z, m in flat:
        m["zip"] = z["path"]
    for _, m in flat:
        m["superseded"] = any(o is not m and o["kind"] == m["kind"] and o["wt"] == m["wt"]
                              and o["start"] <= m["start"] and o["end"] >= m["end"]
                              and (o["end"] - o["start"]) > (m["end"] - m["start"]) for _, o in flat)


def read_static(source: Path, farm: str) -> pd.DataFrame:
    path = Path(source) / "raw" / FARMS[farm]["static"]
    if not path.is_file():
        raise FileNotFoundError(f"Turbine table not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig").dropna(subset=["Title"])
    df["wt"] = df["Title"].str.extract(r"(\d+)\s*$")[0].astype(int)
    return df.sort_values("wt").reset_index(drop=True)


def entity_record(row: pd.Series) -> Dict[str, Any]:
    cod = pd.to_datetime(row["Commercial Operations Date"], dayfirst=True, errors="coerce")
    return {"id": str(row["Title"]), "rated_kw": float(row["Rated power (kW)"]), "lat": float(row["Latitude"]),
            "lon": float(row["Longitude"]), "hub_height_m": float(row["Hub Height (m)"]),
            "model": f"{row['Manufacturer']} {row['Model']}", "cod": None if pd.isna(cod) else cod.strftime("%Y-%m-%d")}


def event_cover(events: pd.DataFrame, start: np.datetime64, n_bins: int) -> np.ndarray:
    """Bool [n_bins] marking 10-minute bins [t, t+10min) that have positive-length overlap with any event."""
    if events.empty:
        return np.zeros(n_bins, bool)
    step = STEP_MIN * 60
    t0 = ((events["t0"].to_numpy("datetime64[s]") - start) // np.timedelta64(1, "s")).astype(np.int64)
    t1 = ((events["t1"].to_numpy("datetime64[s]") - start) // np.timedelta64(1, "s")).astype(np.int64)
    lo, hi = np.floor_divide(t0, step), -np.floor_divide(-t1, step)
    diff = np.zeros(n_bins + 1, np.int32)
    ok = (hi > lo) & (lo < n_bins) & (hi > 0)
    np.add.at(diff, np.clip(lo[ok], 0, n_bins), 1)
    np.add.at(diff, np.clip(hi[ok], 0, n_bins), -1)
    return np.cumsum(diff)[:n_bins] > 0


def run_jobs(func, jobs: List[Any], workers: int) -> List[Any]:
    """Run func over jobs in input order; more than one worker uses a process pool."""
    if workers <= 1 or len(jobs) <= 1:
        return [func(job) for job in jobs]
    with cf.ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS)) as pool:
        return list(pool.map(func, jobs, chunksize=1))


def _place(res: Dict[str, Any], start: np.datetime64, n_times: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Grid row for each result row; returns (source index, grid index, rows dropped off the grid or range)."""
    offset = (res["times"] - start) // np.timedelta64(1, "s")
    on_grid = (offset % (STEP_MIN * 60) == 0)
    row = offset // (STEP_MIN * 60)
    keep = on_grid & (row >= 0) & (row < n_times)
    return np.flatnonzero(keep), row[keep].astype(np.int64), int((~keep).sum())


def _assemble(plan: List[Dict[str, Any]], results: List[Dict[str, Any]], col_of: Dict[int, int],
              start: np.datetime64, n_times: int) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[Dict[str, Any]]]:
    """Write kept members (later periods win on overlap) and compare superseded members with what is there."""
    shape = (n_times, len(col_of))
    values = {k: np.full(shape, np.nan, np.float32) for k in RAW_VARS}
    written = np.zeros(shape, bool)
    notes: List[Dict[str, Any]] = []
    order = sorted(range(len(plan)), key=lambda i: (plan[i]["superseded"], plan[i]["start"], plan[i]["end"]))
    for i in order:
        m, res = plan[i], results[i]
        j = col_of.get(m["wt"])
        src, row, dropped = _place(res, start, n_times)
        note = {"zip": Path(m["zip"]).name, "member": m["member"], "wt": m["wt"], "raw_rows": res["raw_rows"],
                "unique_rows": res["unique_rows"], "dropped": dropped, "missing_cols": res["missing_cols"],
                "superseded": m["superseded"], "overlap_cells": 0, "power_conflicts": 0, "max_power_diff": 0.0,
                "presence_mismatch": 0}
        notes.append(note)
        if j is None:
            note["unknown_turbine"] = True
            continue
        if m["superseded"]:
            have = written[row, j]
            old, new = values["power_raw"][row[have], j], res["power_raw"][src[have]]
            both = np.isfinite(old) & np.isfinite(new)
            diff = np.abs(old[both] - new[both])
            note.update(overlap_cells=int(have.sum()), power_conflicts=int((diff > CLIP_TOL_KW).sum()),
                        max_power_diff=float(diff.max()) if diff.size else 0.0,
                        presence_mismatch=int((np.isfinite(old) != np.isfinite(new)).sum()))
            continue
        note["overlap_cells"] = int(written[row, j].sum())
        for k in RAW_VARS:
            values[k][row, j] = res[k][src]
        written[row, j] = True
    return values, written, notes


def _events(frames: List[pd.DataFrame], wts: List[int]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Concatenate status logs, drop duplicates, keep events with a valid end, flag rule C events."""
    parts = [f.assign(wt=w) for f, w in zip(frames, wts)]
    ev = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=STATUS_COLS + ["wt"])
    stats = {"rows": len(ev)}
    ev = ev.drop_duplicates(["wt", "Timestamp start", "Timestamp end", "Code"])
    stats["after_dedup"] = len(ev)
    ev["t0"] = pd.to_datetime(ev["Timestamp start"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    ev["t1"] = pd.to_datetime(ev["Timestamp end"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    ev = ev[ev["t0"].notna() & ev["t1"].notna() & (ev["t1"] > ev["t0"])].copy()
    stats["with_valid_end"] = len(ev)
    code = pd.to_numeric(ev["Code"], errors="coerce")
    info = ev["Status"].eq("Informational")
    base = ev["Status"].isin(["Stop", "Curtailment", "Communication"]) | ev["IEC category"].isin(IEC_BAD)
    ev["rule_c"] = (base & ~(info & code.isin([10, 65]))) | (info & code.eq(245))
    stats["rule_c_events"] = int(ev["rule_c"].sum())
    return ev, stats


def _masks(values: Dict[str, np.ndarray], ev: pd.DataFrame, wts: List[int], start: np.datetime64) -> Dict[str, np.ndarray]:
    n_times = values["power_raw"].shape[0]
    observed = np.isfinite(values["power_raw"]) & np.isfinite(values["wind_speed"])
    zero = (values["lp_downtime"] == 0) & (values["lp_curtailment"] == 0)
    covered = np.zeros_like(observed)
    flagged = ev[ev["rule_c"]]
    for j, wt in enumerate(wts):
        covered[:, j] = event_cover(flagged[flagged["wt"] == wt], start, n_times)
    return {"observed": observed, "normal_b": observed & zero, "normal_c": observed & ~covered}


def _derived(values: Dict[str, np.ndarray]) -> None:
    raw = values["power_raw"]
    values["power"] = np.where(np.isfinite(raw), np.maximum(raw, 0), np.nan).astype(np.float32)
    angle = np.deg2rad(values["wind_dir"].astype(np.float64))
    values["wind_dir_sin"] = np.sin(angle).astype(np.float32)
    values["wind_dir_cos"] = np.cos(angle).astype(np.float32)


def _pct(num: float, den: float) -> str:
    return "-" if not den else f"{100.0 * num / den:.2f}%"


def _year_table(panel: Panel) -> List[str]:
    years = panel.times.astype("datetime64[Y]").astype(int) + 1970
    v, m = panel.values, panel.masks
    lines = ["| 年份 | 网格格数 | 有数据行占比 | 功率缺失率 | 风速缺失率 | 负功率占比 | observed | B 保留 | C 保留 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for y in np.unique(years):
        s = years == y
        cells = int(s.sum()) * len(panel.entities)
        pr, ws, ob = v["power_raw"][s], v["wind_speed"][s], m["observed"][s]
        present = np.isfinite(pr) | np.isfinite(ws) | np.isfinite(v["rotor_rpm"][s]) | np.isfinite(v["lp_downtime"][s])
        lines.append(f"| {y} | {cells} | {_pct(present.sum(), cells)} | {_pct((~np.isfinite(pr)).sum(), cells)} | "
                     f"{_pct((~np.isfinite(ws)).sum(), cells)} | {_pct((pr[np.isfinite(pr)] < 0).sum(), np.isfinite(pr).sum())} | "
                     f"{_pct(ob.sum(), cells)} | {_pct((m['normal_b'][s]).sum(), ob.sum())} | "
                     f"{_pct((m['normal_c'][s]).sum(), ob.sum())} |")
    return lines


def _range_line(panel: Panel, label: str, select: np.ndarray) -> str:
    ob = panel.masks["observed"][select]
    return (f"| {label} | {int(ob.sum())} | {_pct(panel.masks['normal_b'][select].sum(), ob.sum())} | "
            f"{_pct(panel.masks['normal_c'][select].sum(), ob.sum())} |")


def _report(name: str, panel: Panel, meta: Dict[str, Any], notes: List[Dict[str, Any]], skipped: List[Dict[str, str]],
            ev_stats: Dict[str, int], seconds: float) -> str:
    t = panel.times
    day = lambda s: np.datetime64(s, "us")
    L = [f"# {name} 构建报告", "",
         f"适配器 greenbyte v{ADAPTER_VERSION}；网格 {t[0]} 至 {t[-1] + panel.resolution}（不含），{len(t)} 个 10 分钟时刻，"
         f"{len(panel.entities)} 台机组；耗时 {seconds:.0f} 秒。", "", "## 来源文件", "",
         "| 文件 | 字节 | 作用 |", "|---|---|---|"]
    for s in meta["sources"]:
        L.append(f"| {Path(s['path']).name} | {s['bytes']} | {s['role']} |")
    L += ["", "## 跳过的文件", ""] + ([f"- {s['path']}：{s['reason']}" for s in skipped] or ["无。"])
    dup = sum(n["raw_rows"] - n["unique_rows"] for n in notes if not n["superseded"])
    L += ["", "## 逐成员行数与去重", "", f"采用的数据文件合计去重行数 {dup}（同一时刻只保留第一行）。", "",
          "| zip | 机组 | 原始行 | 唯一时刻 | 去重行 | 网格外丢弃 | 缺列 | 重叠格 | 被取代 | 功率冲突 | 最大差 kW | 有无不一致 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in notes:
        L.append(f"| {n['zip']} | {n['wt']} | {n['raw_rows']} | {n['unique_rows']} | {n['raw_rows'] - n['unique_rows']} | "
                 f"{n['dropped']} | {','.join(n['missing_cols']) or '-'} | {n['overlap_cells']} | "
                 f"{'是' if n['superseded'] else '否'} | {n['power_conflicts']} | {n['max_power_diff']:.4g} | {n['presence_mismatch']} |")
    conflicts = sum(n["power_conflicts"] for n in notes)
    L += ["", f"被取代文件与新版重叠时刻的功率冲突共 {conflicts} 格，一律以新版为准。", "", "## 状态事件", "",
          f"读入 {ev_stats['rows']} 条，去重后 {ev_stats['after_dedup']} 条，结束时间有效且晚于开始 {ev_stats['with_valid_end']} 条，"
          f"其中规则 C 事件 {ev_stats['rule_c_events']} 条。", "", "## 逐年统计", ""] + _year_table(panel)
    core = (t >= day("2016-01-01")) & (t < day("2023-01-01"))
    check = core & ~((t >= day("2021-07-01")) & (t < day("2022-01-01")))
    L += ["", "## 掩码保留比例（分母 observed）", "", "| 范围 | observed 格数 | B | C |", "|---|---|---|---|",
          _range_line(panel, "全部", np.ones(len(t), bool)),
          _range_line(panel, "2016-01-01 至 2023-01-01", core),
          _range_line(panel, "2016-01-01 至 2023-01-01，去掉 2021-07-01 至 2022-01-01", check)]
    return "\n".join(L) + "\n"


def build(paths, name: str, source: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None,
         workers: int = MAX_WORKERS) -> Tuple[Panel, str]:
    """Build the kelmarsh or penmanshiel panel from the Greenbyte SCADA zips under ``source``."""
    if name not in FARMS:
        raise KeyError(f"greenbyte adapter builds {sorted(FARMS)}, not {name!r}")
    t_begin = time.time()
    farm = FARMS[name]
    source = Path(source) if source else RAW_ROOT / farm["folder"]
    start64 = np.datetime64(start or farm["start"], "s")
    n_times = int((np.datetime64(end or GRID_END, "s") - start64) // np.timedelta64(STEP_MIN * 60, "s"))
    static = read_static(source, name)
    entities = [entity_record(r) for _, r in static.iterrows()]
    wts = [int(w) for w in static["wt"]]
    zips, skipped = discover(source)
    mark_superseded(zips)
    for z in list(zips):
        if not any(m["kind"] == "scada" for m in z["members"]):
            skipped.append({"path": z["path"], "reason": "no Turbine_Data members"})
            zips.remove(z)
    members = [m for z in zips for m in z["members"]]
    scada = sorted((m for m in members if m["kind"] == "scada"), key=lambda m: (m["wt"], m["start"], m["zip"]))
    status = sorted((m for m in members if m["kind"] == "status" and not m["superseded"] and m["wt"] in wts),
                    key=lambda m: (m["wt"], m["start"], m["zip"]))
    results = run_jobs(read_scada_member, [(m["zip"], m["member"]) for m in scada], workers)
    values, _, notes = _assemble(scada, results, {w: j for j, w in enumerate(wts)}, start64, n_times)
    frames = run_jobs(read_status_member, [(m["zip"], m["member"]) for m in status], workers)
    ev, ev_stats = _events(frames, [m["wt"] for m in status])
    masks = _masks(values, ev, wts, start64)
    _derived(values)
    used = {m["zip"]: False for m in scada}
    for m in scada:
        used[m["zip"]] = used[m["zip"]] or not m["superseded"]
    hashed = run_jobs(hash_file, sorted(used), workers)
    sources = [{"path": p, "bytes": n, "sha256": h,
                "role": "used" if used[p] else "superseded (compared only)"} for p, n, h in hashed]
    times = start64.astype("datetime64[us]") + np.arange(n_times) * np.timedelta64(STEP_MIN, "m")
    meta = {"name": name, "title": farm["title"], "adapter": "greenbyte", "adapter_version": ADAPTER_VERSION,
            "resolution_minutes": STEP_MIN, "nominal_dates": False, "fit_window": None, "default_target": "power",
            "entities": entities,
            "variables": {k: {"unit": u, "description": d, "source": s} for k, (u, d, s) in VARIABLES.items()},
            "masks": {k: {"target": "power", "description": d, "fit_window": None} for k, d in MASKS.items()},
            "sources": sources,
            "build": {"source": str(source), "grid_start": str(start64), "n_times": n_times, "skipped": skipped,
                      "code_digest": code_digest(), "events": ev_stats, "commercial_operation": farm["cod"]}}
    panel = Panel(meta, times, [e["id"] for e in entities], {k: values[k] for k in VARIABLES}, masks)
    seconds = time.time() - t_begin
    meta["build"]["seconds"] = round(seconds, 1)
    return panel, _report(name, panel, meta, notes, skipped, ev_stats, seconds)
