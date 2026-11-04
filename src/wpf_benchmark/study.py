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

import numpy as np

from .data.store import verify_dataset
from .evaluation import write_json
from .models.base import get_model
from .paths import ProjectPaths
from .provenance import code_digest, environment, sha256_file
from .runner import (ModelSpec, RunLocked, TaskData, load_task_config, load_task_data, make_context, run_directory,
                     run_identity, run_one, run_status, stale_reason, write_table)

PLAN_FIELDS = {"name", "description", "tasks", "models", "seeds", "save_predictions", "save_checkpoints"}
PACKAGE_FILES = ("DONE", "run.json", "metrics.json", "per_horizon.csv", "train.log", "error.txt")


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
    """Code hash plus, per dataset, the panel hash computed from the file and the metadata hash."""
    datasets = {}
    for task in plan["tasks"]:
        cfg = load_task_config(paths, task)
        datasets[cfg.dataset] = verify_dataset(paths.dataset(cfg.dataset))
    return {"code_sha256": code_digest(), "datasets": datasets}


def job_identity(paths: ProjectPaths, job: Job, prints: Dict[str, Any]) -> Dict[str, Any]:
    cfg = load_task_config(paths, job.task)
    return run_identity(cfg, job.spec, job.seed, prints["datasets"][cfg.dataset])


def job_status(paths: ProjectPaths, directory: Path, job: Job, prints: Dict[str, Any]) -> str:
    cfg = load_task_config(paths, job.task)
    return run_status(run_directory(directory / "runs", cfg.name, job.spec.label, job.seed),
                      job_identity(paths, job, prints))


def _describe_task(data: TaskData, problems: List[Dict[str, Any]], task: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {"dataset": data.task.config.dataset, "L": data.task.L, "H": data.task.H,
                            "entities": len(data.train.entities), "splits": data.task.config.splits}
    for name in ("train", "valid", "test"):
        split = getattr(data, name)
        per_time = (split.target_mask & np.isfinite(split.target)).sum(axis=1, dtype=np.int64)
        cumulative = np.concatenate(([0], np.cumsum(per_time, dtype=np.int64)))
        labelled = int((cumulative[split.issues + split.H + 1] - cumulative[split.issues + 1]).sum())
        info[name] = {"windows": int(len(split.issues)), "labelled_target_cells": labelled,
                      "label_count_basis": "forecast window x horizon x entity"}
        if not len(split.issues) or not labelled:
            problems.append({"task": task, "problem": f"the {name} split has no usable windows or labelled targets"})
    return info


def _check_models(paths: ProjectPaths, plan: Dict[str, Any], task: str, data: TaskData,
                  problems: List[Dict[str, Any]]):
    """Yield (job, is_neural) for each model entry that can be built on this task."""
    ctx, seen = None, set()
    for entry in plan["models"]:
        try:
            spec = ModelSpec.load(paths, entry)
            if spec.label in seen:
                raise ValueError(f"label {spec.label!r} is used twice; give the entries distinct labels")
            seen.add(spec.label)
            model = get_model(spec.model)(**spec.params)
            if data.task.config.future_inputs and not model.supports_future_inputs:
                raise ValueError(f"{spec.model} cannot use future_inputs")
            ctx = ctx or make_context(data)
            model.validate_context(ctx)
            neural = hasattr(model, "build_network")
            if neural:
                model.dry_run(ctx, data.train)
            if spec.model == "lightgbm":
                import lightgbm  # noqa: F401
        except Exception as error:  # one broken entry must not hide the others
            label = entry.get("label", entry.get("model")) if isinstance(entry, dict) else entry
            problems.append({"task": task, "model": str(label), "problem": f"{type(error).__name__}: {error}"})
            continue
        for seed in (plan["seeds"][:1] if model.deterministic else plan["seeds"]):
            yield Job(task, spec, int(seed)), int(neural)


def preflight(paths: ProjectPaths, plan: Dict[str, Any]) -> Dict[str, Any]:
    """Check that every run can start, without training; problems are listed per task and model.

    Per task: the dataset file is hashed, the three splits are built and must hold labelled targets.
    Per model: it is constructed, and a neural network is built and run forward on two windows.
    """
    directory = study_directory(paths, plan)
    report: Dict[str, Any] = {"tasks": {}, "problems": [], "environment": environment(), "output": str(directory)}
    counts: Dict[str, int] = {}
    n_runs = n_neural = 0
    for task in plan["tasks"]:
        try:
            data = load_task_data(paths, load_task_config(paths, task))
        except Exception as error:  # report and go on with the next task
            report["problems"].append({"task": task, "problem": f"{type(error).__name__}: {error}"})
            continue
        report["tasks"][task] = _describe_task(data, report["problems"], task)
        prints = {data.task.config.dataset: data.dataset_ident}
        for job, neural in _check_models(paths, plan, task, data, report["problems"]):
            n_runs += 1
            n_neural += neural
            status = job_status(paths, directory, job, {"datasets": prints})
            counts[status] = counts.get(status, 0) + 1
        del data
    if n_neural:
        import torch
        report["cuda"] = bool(torch.cuda.is_available())
        report["device"] = torch.cuda.get_device_name(0) if report["cuda"] else "cpu"
    report.update(n_runs=n_runs, n_done=counts.get("done", 0), n_neural=n_neural, status=counts)
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
              only: Optional[str] = None, log=print, rerun_stale: bool = False) -> Path:
    """Run every job not yet done; one failure does not stop the others.

    A finished run is skipped only if its stored identity (task, model parameters, seed, data) equals the
    current one. Finished runs that differ, or predate identities, stop the study unless ``rerun_stale``.
    """
    plan = load_plan(plan_path)
    directory = study_directory(paths, plan)
    (directory / "runs").mkdir(parents=True, exist_ok=True)
    current = fingerprints(paths, plan)
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous = manifest["fingerprints"]
        changed = {"code_sha256"} if previous["code_sha256"] != current["code_sha256"] else set()
        known_datasets = dict(previous["datasets"])
        for resume in manifest.get("resumes", []):
            known_datasets.update(resume["fingerprints"]["datasets"])
        if any(name in known_datasets and known_datasets[name] != ident
               for name, ident in current["datasets"].items()):
            changed.add("datasets")
        if changed and not allow_changed_code:
            raise RuntimeError(f"Code or data changed since this study started ({', '.join(sorted(changed))}); "
                               "rerun with --allow-changed-code to mix versions, or use a new study name")
        manifest.setdefault("resumes", []).append({"utc": _now(), "fingerprints": current,
                                                   "plan_sha256": sha256_file(Path(plan_path))})
    else:
        manifest = {"plan": plan, "plan_sha256": sha256_file(Path(plan_path)), "created_utc": _now(),
                    "fingerprints": current, "environment": environment()}
    write_json(manifest_path, manifest)
    jobs = [j for j in expand(paths, plan) if only is None or only in f"{j.task}/{j.spec.label}"]
    queue, blocked = [], []
    for job in jobs:
        cfg = load_task_config(paths, job.task)
        out = run_directory(directory / "runs", cfg.name, job.spec.label, job.seed)
        identity = job_identity(paths, job, current)
        status = run_status(out, identity)
        queue.append((job, cfg, out, identity, status))
        if status == "stale":
            blocked.append(f"{cfg.name}/{job.spec.label}/seed{job.seed}: differs in {stale_reason(out, identity)}")
        elif status == "legacy":
            blocked.append(f"{cfg.name}/{job.spec.label}/seed{job.seed}: finished before run identities existed")
    if blocked and not rerun_stale:
        raise RuntimeError("Finished runs here were made from a different configuration or cannot be checked:\n  "
                           + "\n  ".join(blocked) + "\nUse a new study name, or pass --rerun-stale to redo them.")
    failures_path = directory / "failures.json"
    failures = json.loads(failures_path.read_text(encoding="utf-8")) if failures_path.is_file() else {}
    cache_task, cache = None, None
    for index, (job, cfg, out, identity, status) in enumerate(queue, 1):
        key = f"{cfg.name}/{job.spec.label}/seed{job.seed}"
        if status == "done":
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
        except RunLocked:  # another process runs this plan; stop instead of recording a failure
            raise
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
    """summary.csv: one row per run in plan order, with its status.

    done: complete and identical to the current configuration. legacy: complete but from before run
    identities. stale: complete but made from a different configuration or data; its numbers are
    left out. Without local datasets, configuration is checked against the saved data identity and
    data_verified is False; an actual integrity error is never treated as an offline delivery.
    """
    try:
        current = fingerprints(paths, plan)
    except FileNotFoundError:
        current = None
    rows = []
    for job in expand(paths, plan):
        cfg = load_task_config(paths, job.task)
        out = run_directory(directory / "runs", cfg.name, job.spec.label, job.seed)
        expected = job_identity(paths, job, current) if current else None
        if current is None and (out / "run.json").is_file():
            saved_run = json.loads((out / "run.json").read_text(encoding="utf-8"))
            saved_data = saved_run.get("identity", {}).get("parts", {}).get("dataset")
            if saved_data:
                expected = run_identity(cfg, job.spec, job.seed, saved_data)
        status = run_status(out, expected)
        head = {"task": cfg.name, "model": job.spec.label, "seed": job.seed}
        if status not in ("done", "legacy"):
            if status == "missing" and (out / "error.txt").is_file():
                status = "failed"
            rows.append({**head, "status": status})
            continue
        run = json.loads((out / "run.json").read_text(encoding="utf-8"))
        rows.append({**head, "status": status, "data_verified": current is not None,
                     **run["summary"], "n_params": run["n_params"],
                     "fit_seconds": round(run["fit_seconds"], 1)})
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
