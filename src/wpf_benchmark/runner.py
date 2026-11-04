"""One run: task x model x seed -> fit, forecast the test split, score, write results."""
from __future__ import annotations

import csv
import json
import os
import platform
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import models  # noqa: F401  (registers models)
from .data.store import load_meta, load_panel, verify_dataset
from .evaluation import evaluate, per_horizon_rows, summary_row, write_json
from .models.base import Forecaster, get_model
from .paths import ProjectPaths
from .provenance import canonical_sha256, code_digest, environment, git_commit, sha256_file
from .tasks import ResolvedTask, Scaler, SplitData, TaskConfig, TaskContext, build_split, panel_columns, resolve_task

PREDICT_CELLS = 40_000_000          # float32 cells per prediction chunk (about 160 MB of inputs)
DONE_FORMAT = 2
REQUIRED_FILES = ("run.json", "metrics.json", "per_horizon.csv")     # hashed in DONE
OPTIONAL_FILES = ("predictions.npz", "checkpoint.pt", "train.log", "error.txt")
LOCK_NAME = "RUNNING"


def _config_path(paths: ProjectPaths, kind: str, name_or_path: str) -> Optional[Path]:
    candidate = Path(name_or_path)
    if candidate.suffix == ".json":
        return candidate if candidate.is_absolute() else paths.root / candidate
    path = paths.configs / kind / f"{name_or_path}.json"
    return path if path.is_file() else None


def load_task_config(paths: ProjectPaths, task: str) -> TaskConfig:
    path = _config_path(paths, "tasks", task)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"No task config for {task!r} under {paths.configs / 'tasks'}")
    return TaskConfig.load(path)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model: str
    params: Dict[str, Any]

    @classmethod
    def load(cls, paths: ProjectPaths, entry) -> "ModelSpec":
        """Entry is a model name, a config path, or {"label", "model", "params"}."""
        if isinstance(entry, dict):
            return cls(entry.get("label", entry["model"]), entry["model"], dict(entry.get("params", {})))
        path = _config_path(paths, "models", str(entry))
        if path is None:
            return cls(str(entry), str(entry), {})
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(raw.get("label", path.stem), raw.get("model", path.stem), dict(raw.get("params", {})))


@dataclass
class TaskData:
    task: ResolvedTask
    train: SplitData
    valid: SplitData
    test: SplitData
    dataset_ident: Dict[str, str]          # panel_sha256 (verified against the file) and meta_sha256

    @property
    def dataset_sha256(self) -> str:
        return self.dataset_ident["panel_sha256"]


def load_task_data(paths: ProjectPaths, cfg: TaskConfig) -> TaskData:
    """Resolve a task against its dataset and slice the three splits from one panel load.

    The panel file is hashed and compared with meta.json first; a changed file raises.
    """
    directory = paths.dataset(cfg.dataset)
    ident = verify_dataset(directory)
    meta = load_meta(directory)
    task = resolve_task(cfg, meta)
    variables, masks = panel_columns(task)
    step = np.timedelta64(task.resolution_minutes, "m").astype("timedelta64[us]")
    start = min(task.bounds["train"][0], task.bounds["test"][0] - (task.L - 1) * step)
    panel = load_panel(directory, variables, masks, start=start, end=task.bounds["test"][1])
    splits = {name: build_split(task, panel, name) for name in ("train", "valid", "test")}
    return TaskData(task, splits["train"], splits["valid"], splits["test"], ident)


def make_context(data: TaskData) -> TaskContext:
    cfg = data.task.config
    scaler = Scaler.fit(data.train, cfg.target_input)
    return TaskContext(data.task, list(cfg.inputs), cfg.target, cfg.target_input, data.task.L, data.task.H,
                       list(data.train.entities), data.task.rated_kw, scaler)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def forecast(model: Forecaster, data: TaskData) -> np.ndarray:
    """Predict every test issue time in chunks, returning [I, H, N]."""
    test, task = data.test, data.task
    per_issue = task.L * len(test.entities) * len(test.input_names)
    chunk = max(1, PREDICT_CELLS // max(per_issue, 1))
    out = np.empty((len(test.issues), task.H, len(test.entities)), np.float32)
    for first in range(0, len(test.issues), chunk):
        positions = test.issues[first:first + chunk]
        block = np.asarray(model.predict(test.batch(positions)), dtype=np.float32)
        if block.shape != (len(positions), task.H, len(test.entities)):
            raise ValueError(f"{model.name} returned shape {block.shape}")
        if not np.isfinite(block).all():
            raise ValueError(f"{model.name} returned nonfinite predictions")
        out[first:first + len(positions)] = block
    return out


def run_directory(base: Path, task: str, label: str, seed: int) -> Path:
    return base / task / label / f"seed{seed}"


def run_identity(cfg: TaskConfig, spec: ModelSpec, seed: int, dataset_ident: Dict[str, str]) -> Dict[str, Any]:
    """Everything a result depends on except the code: resolved task, model with effective
    parameters, seed and verified data. Code is compared at study level (see study.fingerprints)."""
    task = cfg.to_dict()
    task.pop("description", None)
    parts = {"task": task, "model": spec.model, "params": get_model(spec.model)(**spec.params).params,
             "seed": int(seed), "dataset": dict(dataset_ident)}
    parts = json.loads(json.dumps(parts, sort_keys=True, default=str))
    return {"sha256": canonical_sha256(parts), "parts": parts}


def _read_done(directory: Path) -> Optional[Dict[str, Any]]:
    path = directory / "DONE"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    return data if isinstance(data, dict) else {"legacy": text}


def run_status(directory: Path, identity: Optional[Dict[str, Any]] = None) -> str:
    """missing | incomplete | legacy | stale | done.

    legacy: a pre-identity DONE (metrics hash only) whose metrics.json still matches; it cannot be
    compared with a configuration. stale: complete, but made from a different configuration or data.
    """
    done = _read_done(directory)
    if done is None:
        return "missing"
    if "legacy" in done:
        metrics = directory / "metrics.json"
        return "legacy" if metrics.is_file() and done["legacy"] == sha256_file(metrics) else "incomplete"
    files = done.get("files", {})
    if done.get("format") != DONE_FORMAT or set(REQUIRED_FILES) - set(files):
        return "incomplete"
    for name, digest in files.items():
        if not (directory / name).is_file() or sha256_file(directory / name) != digest:
            return "incomplete"
    for name, size in done.get("optional", {}).items():
        if not (directory / name).is_file() or (directory / name).stat().st_size != size:
            return "incomplete"
    if identity is not None and done.get("identity") != identity["sha256"]:
        return "stale"
    return "done"


def is_done(directory: Path, identity: Optional[Dict[str, Any]] = None) -> bool:
    return run_status(directory, identity) == "done"


def stale_reason(directory: Path, identity: Dict[str, Any]) -> str:
    """Name the identity parts that differ between a finished run and the expected one."""
    try:
        saved = json.loads((directory / "run.json").read_text(encoding="utf-8")).get("identity", {}).get("parts", {})
    except (OSError, ValueError):
        saved = {}
    if not saved:
        return "no identity recorded"
    parts = identity["parts"]
    return ", ".join(k for k in parts if saved.get(k) != parts[k]) or "identity hash"


class RunLocked(RuntimeError):
    """Another process holds the run directory's RUNNING lock."""


def acquire_lock(directory: Path) -> Path:
    """Create RUNNING exclusively so two processes cannot write one run directory."""
    lock = directory / LOCK_NAME
    try:
        handle = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RunLocked(f"{directory} is being written by another process ({lock.read_text().strip()}); "
                        f"if that process is gone, delete {lock} and rerun") from None
    with os.fdopen(handle, "w") as out:
        out.write(f"pid {os.getpid()} on {platform.node()} since {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    return lock


def clear_run(directory: Path) -> None:
    """Remove every artifact of an earlier run, so a rerun cannot be mixed with old results."""
    (directory / "DONE").unlink(missing_ok=True)
    for name in REQUIRED_FILES + OPTIONAL_FILES:
        (directory / name).unlink(missing_ok=True)


def finish_run(directory: Path, identity: Dict[str, Any]) -> None:
    """Write DONE last and atomically: identity, required-file hashes, optional-file sizes."""
    optional = {n: (directory / n).stat().st_size for n in ("predictions.npz", "checkpoint.pt")
                if (directory / n).is_file()}
    manifest = {"format": DONE_FORMAT, "identity": identity["sha256"],
                "files": {n: sha256_file(directory / n) for n in REQUIRED_FILES}, "optional": optional}
    temporary = directory / "DONE.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, directory / "DONE")


def write_table(path: Path, rows) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def run_one(paths: ProjectPaths, task_name: str, model_entry, seed: int, out_base: Optional[Path] = None,
            save_predictions: bool = False, save_checkpoint: bool = True,
            data: Optional[TaskData] = None) -> Tuple[Path, Dict[str, Any]]:
    """Fit one model on one task with one seed and write its run directory.

    Artifacts of an earlier run in the same directory are removed first, and a RUNNING lock file keeps
    a second process out until this one finishes.
    """
    cfg = load_task_config(paths, task_name)
    spec = ModelSpec.load(paths, model_entry)
    data = data or load_task_data(paths, cfg)
    identity = run_identity(cfg, spec, seed, data.dataset_ident)
    directory = run_directory(out_base or paths.results / "runs", cfg.name, spec.label, seed)
    directory.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(directory)
    try:
        clear_run(directory)
        return _execute(paths, cfg, spec, seed, directory, save_predictions, save_checkpoint, data, identity)
    finally:
        lock.unlink(missing_ok=True)


def _execute(paths: ProjectPaths, cfg: TaskConfig, spec: ModelSpec, seed: int, directory: Path,
             save_predictions: bool, save_checkpoint: bool, data: TaskData,
             identity: Dict[str, Any]) -> Tuple[Path, Dict[str, Any]]:
    started = datetime.now(timezone.utc)
    ctx = make_context(data)
    seed_all(seed)
    model = get_model(spec.model)(**spec.params)
    if cfg.future_inputs and not model.supports_future_inputs:
        raise ValueError(f"{spec.model} cannot use future_inputs")
    t0 = time.perf_counter()
    model.fit(ctx, data.train, data.valid)
    fit_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    predictions = forecast(model, data)
    predict_seconds = time.perf_counter() - t0
    metrics = evaluate(data.task, data.test, predictions)
    run = {"task": cfg.to_dict(), "label": spec.label, "model": spec.model, "params": model.params, "seed": seed,
           "dataset": cfg.dataset, "dataset_sha256": data.dataset_sha256,
           "dataset_meta_sha256": data.dataset_ident["meta_sha256"], "identity": identity,
           "code_sha256": code_digest(),
           "git_commit": git_commit(paths.root), "environment": environment(),
           "started_utc": started.isoformat(timespec="seconds"), "fit_seconds": fit_seconds,
           "predict_seconds": predict_seconds, "n_params": int(model.n_params),
           "validation_metrics": model.validation_metrics, "scaler": ctx.scaler.to_dict(),
           "grid": {"L": data.task.L, "H": data.task.H, "n_issues": int(len(data.test.issues)),
                    "train_windows": int(len(data.train.issues)), "valid_windows": int(len(data.valid.issues))},
           "summary": summary_row(metrics)}
    (directory / "train.log").write_text("\n".join(model.log_lines) + "\n", encoding="utf-8")
    if save_predictions:
        np.savez_compressed(directory / "predictions.npz", predictions=predictions,
                            issue_times=data.test.issue_times.astype("datetime64[us]").astype(np.int64),
                            entities=np.asarray(data.test.entities))
    if save_checkpoint and hasattr(model, "save_checkpoint"):
        model.save_checkpoint(directory / "checkpoint.pt", metadata={"seed": seed, "label": spec.label})
        run["checkpoint_sha256"] = sha256_file(directory / "checkpoint.pt")
    write_json(directory / "run.json", run)
    write_table(directory / "per_horizon.csv", per_horizon_rows(metrics))
    write_json(directory / "metrics.json", metrics)
    finish_run(directory, identity)
    return directory, run


def score_predictions(paths: ProjectPaths, task_name: str, predictions_path: Path,
                      out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Score externally produced forecasts on exactly this task's test grid."""
    cfg = load_task_config(paths, task_name)
    data = load_task_data(paths, cfg)
    saved = np.load(predictions_path, allow_pickle=False)
    times = data.test.issue_times.astype("datetime64[us]").astype(np.int64)
    if not np.array_equal(np.asarray(saved["issue_times"], dtype=np.int64), times):
        raise ValueError("issue_times differ from the task's test issue times")
    if [str(e) for e in saved["entities"]] != list(data.test.entities):
        raise ValueError("entities differ from the dataset entity order")
    metrics = evaluate(data.task, data.test, np.asarray(saved["predictions"], dtype=np.float32))
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "metrics.json", metrics)
        write_table(out_dir / "per_horizon.csv", per_horizon_rows(metrics))
        write_json(out_dir / "run.json", {"task": cfg.to_dict(), "label": "external", "model": "external",
                                          "predictions": str(predictions_path),
                                          "predictions_sha256": sha256_file(Path(predictions_path)),
                                          "dataset_sha256": data.dataset_sha256,
                                          "dataset_meta_sha256": data.dataset_ident["meta_sha256"],
                                          "summary": summary_row(metrics)})
    return metrics
