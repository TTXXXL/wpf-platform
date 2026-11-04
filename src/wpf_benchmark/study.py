"""Matrix batch runs: tasks x models x seeds, resumable, packaged into one ZIP."""
from __future__ import annotations

import gc
import json
import time
import traceback
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .data.store import load_meta
from .evaluation import write_json
from .models.base import get_model
from .paths import ProjectPaths
from .provenance import code_digest, environment, sha256_file
from .runner import ModelSpec, is_done, load_task_config, load_task_data, run_directory, run_one, write_table
from .tasks import resolve_task

PLAN_FIELDS = {"name", "description", "tasks", "models", "seeds", "save_predictions", "save_checkpoints"}
PACKAGE_FILES = ("run.json", "metrics.json", "per_horizon.csv", "train.log", "error.txt")


@dataclass(frozen=True)
class Job:
    task: str
    spec: ModelSpec
    seed: int


def load_plan(path: Path) -> Dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = sorted(set(plan) - PLAN_FIELDS)
    if unknown:
        raise ValueError(f"Study plan has unknown fields: {', '.join(unknown)}")
    for key in ("name", "tasks", "models"):
        if not plan.get(key):
            raise ValueError(f"Study plan needs a non-empty {key!r}")
    plan.setdefault("seeds", [0])
    plan.setdefault("save_predictions", False)
    plan.setdefault("save_checkpoints", True)
    return plan


def expand(paths: ProjectPaths, plan: Dict[str, Any]) -> List[Job]:
    """All runs in plan order; deterministic models run only the first seed."""
    jobs = []
    for task in plan["tasks"]:
        for entry in plan["models"]:
            spec = ModelSpec.load(paths, entry)
            model = get_model(spec.model)(**spec.params)
            seeds = plan["seeds"][:1] if model.deterministic else plan["seeds"]
            jobs.extend(Job(task, spec, int(seed)) for seed in seeds)
    labels = {(j.task, j.spec.label, j.seed) for j in jobs}
    if len(labels) != len(jobs):
        raise ValueError("Two model entries share a label; give them distinct 'label' values")
    return jobs


def study_directory(paths: ProjectPaths, plan: Dict[str, Any]) -> Path:
    return paths.results / "studies" / plan["name"]


def fingerprints(paths: ProjectPaths, plan: Dict[str, Any]) -> Dict[str, Any]:
    datasets = {}
    for task in plan["tasks"]:
        cfg = load_task_config(paths, task)
        datasets[cfg.dataset] = load_meta(paths.dataset(cfg.dataset))["panel_sha256"]
    return {"code_sha256": code_digest(), "datasets": datasets}


def preflight(paths: ProjectPaths, plan: Dict[str, Any]) -> Dict[str, Any]:
    """Check datasets, tasks and models without training. Raises on the first problem."""
    report: Dict[str, Any] = {"tasks": {}, "environment": environment()}
    for task in plan["tasks"]:
        cfg = load_task_config(paths, task)
        meta = load_meta(paths.dataset(cfg.dataset))
        resolved = resolve_task(cfg, meta)
        report["tasks"][task] = {"dataset": cfg.dataset, "L": resolved.L, "H": resolved.H,
                                 "entities": len(meta["entities"]), "splits": cfg.splits}
    jobs = expand(paths, plan)
    neural = [j for j in jobs if hasattr(get_model(j.spec.model), "build_network")]
    if neural:
        import torch
        report["cuda"] = bool(torch.cuda.is_available())
        report["device"] = torch.cuda.get_device_name(0) if report["cuda"] else "cpu"
    if any(j.spec.model == "lightgbm" for j in jobs):
        import lightgbm  # noqa: F401
    directory = study_directory(paths, plan)
    done = sum(is_done(run_directory(directory / "runs", load_task_config(paths, j.task).name, j.spec.label, j.seed))
               for j in jobs)
    report.update(n_runs=len(jobs), n_done=int(done), n_neural=len(neural), output=str(directory))
    return report


def _free_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_study(paths: ProjectPaths, plan_path: Path, allow_changed_code: bool = False,
              only: Optional[str] = None, log=print) -> Path:
    """Run every job not yet done; one failure does not stop the others."""
    plan = load_plan(plan_path)
    directory = study_directory(paths, plan)
    (directory / "runs").mkdir(parents=True, exist_ok=True)
    current = fingerprints(paths, plan)
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        changed = {k for k in ("code_sha256", "datasets") if manifest["fingerprints"][k] != current[k]}
        if changed and not allow_changed_code:
            raise RuntimeError(f"Code or data changed since this study started ({', '.join(sorted(changed))}); "
                               "rerun with --allow-changed-code to mix versions, or use a new study name")
        manifest.setdefault("resumes", []).append({"utc": _now(), "fingerprints": current})
    else:
        manifest = {"plan": plan, "plan_sha256": sha256_file(Path(plan_path)), "created_utc": _now(),
                    "fingerprints": current, "environment": environment()}
    write_json(manifest_path, manifest)
    jobs = [j for j in expand(paths, plan) if only is None or only in f"{j.task}/{j.spec.label}"]
    failures_path = directory / "failures.json"
    failures = json.loads(failures_path.read_text(encoding="utf-8")) if failures_path.is_file() else {}
    cache_task, cache = None, None
    for index, job in enumerate(jobs, 1):
        cfg = load_task_config(paths, job.task)
        out = run_directory(directory / "runs", cfg.name, job.spec.label, job.seed)
        key = f"{cfg.name}/{job.spec.label}/seed{job.seed}"
        if is_done(out):
            log(f"[{index}/{len(jobs)}] {key}: done, skipped")
            continue
        started = time.perf_counter()
        try:
            if cache_task != cfg.name:
                cache, cache_task = None, None
                _free_memory()
                cache, cache_task = load_task_data(paths, cfg), cfg.name
            run_one(paths, job.task, {"label": job.spec.label, "model": job.spec.model, "params": job.spec.params},
                    job.seed, out_base=directory / "runs", save_predictions=plan["save_predictions"],
                    save_checkpoint=plan["save_checkpoints"], data=cache)
            failures.pop(key, None)
            (out / "error.txt").unlink(missing_ok=True)
            summary = json.loads((out / "run.json").read_text(encoding="utf-8"))["summary"]
            log(f"[{index}/{len(jobs)}] {key}: MAE {summary['turbine_MAE_kW']:.2f} kW "
                f"({time.perf_counter() - started:.0f} s)")
        except Exception:  # keep going; the error is recorded for the delivery
            text = traceback.format_exc()
            out.mkdir(parents=True, exist_ok=True)
            (out / "error.txt").write_text(text, encoding="utf-8")
            failures[key] = {"utc": _now(), "error": text.strip().splitlines()[-1]}
            log(f"[{index}/{len(jobs)}] {key}: FAILED: {failures[key]['error']}")
        write_json(failures_path, failures)
        _free_memory()
    write_summary(directory, plan, paths)
    log(f"finished: {len(jobs) - len(failures)} done, {len(failures)} failed -> {directory}")
    return directory


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_summary(directory: Path, plan: Dict[str, Any], paths: ProjectPaths) -> Path:
    """summary.csv: one row per finished run, in plan order."""
    rows = []
    for job in expand(paths, plan):
        cfg = load_task_config(paths, job.task)
        out = run_directory(directory / "runs", cfg.name, job.spec.label, job.seed)
        if not is_done(out):
            rows.append({"task": cfg.name, "model": job.spec.label, "seed": job.seed, "status": "missing"
                         if not (out / "error.txt").is_file() else "failed"})
            continue
        run = json.loads((out / "run.json").read_text(encoding="utf-8"))
        rows.append({"task": cfg.name, "model": job.spec.label, "seed": job.seed, "status": "done",
                     **run["summary"], "n_params": run["n_params"], "fit_seconds": round(run["fit_seconds"], 1)})
    columns = list(dict.fromkeys(k for row in rows for k in row))
    write_table(directory / "summary.csv", [{c: row.get(c, "") for c in columns} for row in rows])
    return directory / "summary.csv"


def package(paths: ProjectPaths, plan_path: Path, include_predictions: bool = False,
            include_checkpoints: bool = False) -> Path:
    """Zip the plan, manifest, summary and per-run results (not predictions or weights by default)."""
    plan = load_plan(plan_path)
    directory = study_directory(paths, plan)
    write_summary(directory, plan, paths)
    target = directory / f"{plan['name']}_delivery.zip"
    names = set(PACKAGE_FILES) | ({"predictions.npz"} if include_predictions else set()) | \
        ({"checkpoint.pt"} if include_checkpoints else set())
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(plan_path, f"{plan['name']}/plan.json")
        for name in ("manifest.json", "summary.csv", "failures.json"):
            if (directory / name).is_file():
                archive.write(directory / name, f"{plan['name']}/{name}")
        for path in sorted((directory / "runs").rglob("*")):
            if path.is_file() and path.name in names:
                archive.write(path, f"{plan['name']}/{path.relative_to(directory).as_posix()}")
    return target
