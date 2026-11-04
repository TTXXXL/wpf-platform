"""Read and write dataset directories: panel.parquet plus meta.json."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..provenance import canonical_sha256, sha256_file
from .panel import Panel

STORE_VERSION = 1
MASK_PREFIX = "mask__"
REQUIRED_META = ("name", "adapter", "resolution_minutes", "entities", "variables", "masks",
                 "default_target")


def write_dataset(directory: Path, panel: Panel, report: str = "") -> Dict[str, Any]:
    """Write a complete panel atomically and return the final metadata."""
    meta = dict(panel.meta)
    missing = [key for key in REQUIRED_META if key not in meta]
    if missing:
        raise ValueError("Dataset metadata lacks: " + ", ".join(missing))
    if [str(e["id"]) for e in meta["entities"]] != list(panel.entities):
        raise ValueError("meta.entities must list panel entities in panel order")
    undeclared = sorted(set(panel.values) ^ set(meta["variables"]))
    if undeclared or sorted(set(panel.masks) ^ set(meta["masks"])):
        raise ValueError("Variables and masks must match meta exactly: " + ", ".join(undeclared))
    for name, spec in meta["masks"].items():
        if spec.get("target") not in panel.values:
            raise ValueError(f"Mask {name} names an unknown target {spec.get('target')!r}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    T, N = len(panel.times), len(panel.entities)
    columns = {"time": pa.array(np.repeat(panel.times, N), type=pa.timestamp("us")),
               "entity": pa.array(np.tile(np.asarray(panel.entities, dtype=object), T), type=pa.string())}
    for name, array in panel.values.items():
        columns[name] = pa.array(np.asarray(array, dtype=np.float32).ravel(), type=pa.float32())
    for name, array in panel.masks.items():
        columns[MASK_PREFIX + name] = pa.array(np.asarray(array, dtype=bool).ravel(), type=pa.bool_())
    table = pa.table(columns)
    temporary = directory / "panel.parquet.tmp"
    pq.write_table(table, temporary, compression="zstd", row_group_size=max(N * 4096, 1))
    os.replace(temporary, directory / "panel.parquet")
    end = panel.times[-1] + panel.resolution
    meta.update(store_version=STORE_VERSION, time_convention="interval_start", timezone="UTC",
                time_range=[str(panel.times[0]), str(end)], n_times=T, n_entities=N,
                panel_sha256=sha256_file(directory / "panel.parquet"),
                written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    if panel.fit_arrays:
        np.savez_compressed(directory / "fit.npz", **panel.fit_arrays)
        meta["fit_arrays"] = {"path": "fit.npz", "sha256": sha256_file(directory / "fit.npz"),
                              "keys": sorted(panel.fit_arrays)}
    (directory / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if report:
        (directory / "build_report.md").write_text(report, encoding="utf-8")
    return meta


def load_meta(directory: Path) -> Dict[str, Any]:
    path = Path(directory) / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset is not built: {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("store_version") != STORE_VERSION:
        raise ValueError(f"Unsupported dataset store version {meta.get('store_version')} in {path}")
    return meta


VOLATILE_META = ("written_utc", "build")      # change with every rebuild without changing the data
_VERIFIED: Dict[Any, str] = {}


def meta_digest(meta: Dict[str, Any]) -> str:
    """Hash of the metadata that fixes data semantics: units, rated power, masks, resolution, sources."""
    return canonical_sha256({k: v for k, v in meta.items() if k not in VOLATILE_META})


def verify_dataset(directory: Path) -> Dict[str, str]:
    """Hash panel.parquet itself and compare it with meta.json; cached per process by size and mtime."""
    directory = Path(directory)
    meta = load_meta(directory)
    path = directory / "panel.parquet"
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, meta["panel_sha256"])
    if key not in _VERIFIED:
        actual = sha256_file(path)
        if actual != meta["panel_sha256"]:
            raise ValueError(f"{path} does not match the SHA256 recorded in meta.json "
                             f"(file {actual[:12]}, meta {meta['panel_sha256'][:12]})")
        _VERIFIED[key] = actual
    return {"panel_sha256": _VERIFIED[key], "meta_sha256": meta_digest(meta)}


def _as_datetime(value) -> datetime:
    return np.datetime64(value, "us").astype(datetime)


def load_panel(directory: Path, variables: Optional[Sequence[str]] = None,
               masks: Optional[Sequence[str]] = None, start=None, end=None,
               verify: bool = False) -> Panel:
    """Load [start, end) of a dataset; None for variables or masks loads all of them."""
    directory = Path(directory)
    meta = load_meta(directory)
    path = directory / "panel.parquet"
    if verify:
        verify_dataset(directory)
    variables = list(meta["variables"]) if variables is None else list(dict.fromkeys(variables))
    masks = list(meta["masks"]) if masks is None else list(dict.fromkeys(masks))
    unknown = [v for v in variables if v not in meta["variables"]] + [m for m in masks if m not in meta["masks"]]
    if unknown:
        raise KeyError(f"Dataset {meta['name']} has no variable or mask: {', '.join(unknown)}")
    filters = []
    if start is not None:
        filters.append(("time", ">=", _as_datetime(start)))
    if end is not None:
        filters.append(("time", "<", _as_datetime(end)))
    table = pq.read_table(path, columns=["time", "entity"] + variables + [MASK_PREFIX + m for m in masks],
                          filters=filters or None)
    entities = [str(e["id"]) for e in meta["entities"]]
    N = len(entities)
    if table.num_rows % N:
        raise ValueError("Loaded rows do not form a complete time x entity grid")
    T = table.num_rows // N
    codes = pc.index_in(table["entity"], value_set=pa.array(entities)).to_numpy(zero_copy_only=False)
    if T and not np.array_equal(codes.reshape(T, N), np.broadcast_to(np.arange(N), (T, N))):
        raise ValueError("Panel rows are not sorted by (time, entity) in metadata order")
    times = table["time"].to_numpy().astype("datetime64[us]").reshape(T, N)
    if T and not (times == times[:, :1]).all():
        raise ValueError("Each time block must hold every entity exactly once")
    values = {v: table[v].to_numpy(zero_copy_only=False).astype(np.float32, copy=False).reshape(T, N)
              for v in variables}
    loaded = {m: table[MASK_PREFIX + m].to_numpy(zero_copy_only=False).astype(bool).reshape(T, N)
              for m in masks}
    return Panel(meta=meta, times=times[:, 0] if T else times.reshape(0), entities=entities,
                 values=values, masks=loaded)
