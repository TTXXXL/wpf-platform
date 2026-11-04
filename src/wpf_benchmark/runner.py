"""One run: task x model x seed -> fit, forecast the test split, score, write results."""
from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import models  # noqa: F401  (registers models)
from .data.store import load_meta, load_panel
from .evaluation import evaluate, per_horizon_rows, summary_row, write_json
from .models.base import Forecaster, get_model
from .paths import ProjectPaths
from .provenance import code_digest, environment, git_commit, sha256_file
from .tasks import ResolvedTask, Scaler, SplitData, TaskConfig, TaskContext, build_split, panel_columns, resolve_task

PREDICT_CELLS = 40_000_000          # float32 cells per prediction chunk (about 160 MB of inputs)


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
    dataset_sha256: str


def load_task_data(paths: ProjectPaths, cfg: TaskConfig) -> TaskData:
    """Resolve a task against its dataset and slice the three splits from one panel load."""
    directory = paths.dataset(cfg.dataset)
    meta = load_meta(directory)
    task = resolve_task(cfg, meta)
    variables, masks = panel_columns(task)
    step = np.timedelta64(task.resolution_minutes, "m").astype("timedelta64[us]")
    start = min(task.bounds["train"][0], task.bounds["test"][0] - (task.L - 1) * step)
    panel = load_panel(directory, variables, masks, start=start, end=task.bounds["test"][1])
    splits = {name: build_split(task, panel, name) for name in ("train", "valid", "test")}
    return TaskData(task, splits["train"], splits["valid"], splits["test"], meta["panel_sha256"])


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


def is_done(directory: Path) -> bool:
    done, metrics = directory / "DONE", directory / "metrics.json"
    return done.is_file() and metrics.is_file() and done.read_text().strip() == sha256_file(metrics)


def write_table(path: Path, rows) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def run_one(paths: ProjectPaths, task_name: str, model_entry, seed: int, out_base: Optional[Path] = None,
            save_predictions: bool = False, save_checkpoint: bool = True,
            data: Optional[TaskData] = None) -> Tuple[Path, Dict[str, Any]]:
    """Fit one model on one task with one seed and write its run directory."""
    cfg = load_task_config(paths, task_name)
    spec = ModelSpec.load(paths, model_entry)
    directory = run_directory(out_base or paths.results / "runs", cfg.name, spec.label, seed)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "DONE").unlink(missing_ok=True)
    started = datetime.now(timezone.utc)
    data = data or load_task_data(paths, cfg)
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
           "dataset": cfg.dataset, "dataset_sha256": data.dataset_sha256, "code_sha256": code_digest(),
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
    (directory / "DONE").write_text(sha256_file(directory / "metrics.json"), encoding="utf-8")
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
                                          "dataset_sha256": data.dataset_sha256, "summary": summary_row(metrics)})
    return metrics
