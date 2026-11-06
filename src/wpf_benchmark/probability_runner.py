"""Opt-in probability runs and external scoring, separate from legacy point files."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .evaluation import evaluate, write_json
from .evaluation.probabilistic import evaluate_quantiles
from .models import get_model
from .probability import LEVELS, MEDIAN, load_quantiles, postprocess, save_quantiles
from .provenance import canonical_sha256, code_digest, environment, git_commit, sha256_file
from .runner import ModelSpec, acquire_lock, load_task_config, load_task_data, make_context, seed_all, write_table


def forecast_quantiles(model, data):
    test, task = data.test, data.task
    chunk = max(1, min(128, 2_000_000 // (task.L * len(test.entities) * len(test.input_names))))
    output, means, medians = [], [], []
    degenerate = not callable(getattr(model, "predict_quantiles", None))
    for first in range(0, len(test.issues), chunk):
        batch = test.batch(test.issues[first:first+chunk])
        if degenerate:
            block = np.repeat(model.predict(batch)[..., None], len(LEVELS), axis=-1)
        else:
            block = model.predict_quantiles(batch)
        if block.shape != (len(batch.issue_time), task.H, len(test.entities), len(LEVELS)):
            raise ValueError(f"Invalid probability output shape {block.shape}")
        output.append(block)
        if hasattr(model, "native_mean"):
            means.append(model.native_mean.copy())
            medians.append(model.native_median.copy())
    native = {"native_mean": np.concatenate(means), "native_median": np.concatenate(medians)} if means else {}
    return np.concatenate(output), degenerate, native


def probability_identity(data):
    cfg = data.task.config
    return {"task_sha256": canonical_sha256(cfg.to_dict()), "dataset": data.dataset_ident,
            "target": cfg.target, "eval_mask": cfg.eval_mask,
            "normalization": {"scale": "power/entity_rated_kw", "rated_kw": data.task.rated_kw.tolist()}}


def write_probability_metrics(directory, data, raw):
    metrics = evaluate_quantiles(data.task, data.test, raw)
    write_json(directory / "probability_metrics.json", metrics)
    rows = []
    for mask, result in metrics["scores"].items():
        for row in result["per_horizon"]:
            rows.append({"mask": mask, **{k: v for k, v in row.items() if not isinstance(v, (dict, list))}})
    write_table(directory / "probability_per_horizon.csv", rows)
    point = evaluate(data.task, data.test, postprocess(raw)[..., MEDIAN])
    point["point_statistic"] = "unit_q50; farm_point_from_unit_median = sum(unit_q50), not a farm quantile"
    write_json(directory / "metrics_unit_median.json", point)
    return metrics


def run_probability(paths, task_name, model_entry, seed, directory):
    cfg = load_task_config(paths, task_name)
    if cfg.extra.get("probability_protocol") != "unit_quantiles_v1":
        raise ValueError("Probability runs require an explicit probability task configuration")
    data = load_task_data(paths, cfg)
    spec = ModelSpec.load(paths, model_entry)
    model = get_model(spec.model)(**spec.params)
    ctx = make_context(data)
    model.validate_context(ctx)
    frozen_path = None
    boundaries = None
    frozen = cfg.extra.get("strata_boundaries_path")
    if frozen:
        import json
        frozen_path = Path(frozen)
        if not frozen_path.is_absolute():
            frozen_path = paths.root / frozen_path
        boundaries = json.loads(frozen_path.read_text(encoding="utf-8"))
        if not boundaries.get("frozen_before_model_results") or boundaries.get("software_test_only"):
            raise ValueError("Strata must be real-data boundaries frozen before model results")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Probability run output directory must be empty; existing results are never overwritten")
    lock = acquire_lock(directory)
    try:
        seed_all(seed)
        started = time.perf_counter()
        model.fit(ctx, data.train, data.valid)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        raw, degenerate, native = forecast_quantiles(model, data)
        predict_seconds = time.perf_counter() - started
        identity = save_quantiles(directory / "predictions_quantiles.npz", raw, data.test.issue_times,
                                  data.test.entities, {**probability_identity(data), "model": spec.model,
                                  "params": model.params, "seed": seed, "degenerate_point": degenerate,
                                  "model_metadata": getattr(model, "probability_metadata", {})},
                                  monotone=getattr(model, "monotone_quantiles", False))
        write_json(directory / "quantile_levels.json", LEVELS.tolist())
        if native:
            np.savez_compressed(directory / "native_point_statistics.npz", **native,
                                q50=raw[..., MEDIAN], issue_times=data.test.issue_times.astype("datetime64[us]").astype(np.int64),
                                entities=np.asarray(data.test.entities))
        metrics = write_probability_metrics(directory, data, raw)
        if boundaries is not None:
            from .profile_probability import make_strata
            from .evaluation.probabilistic import stratified_scores
            cutoff = str(data.task.bounds["test"][0].astype("datetime64[D]"))
            edges = boundaries["volatility_bins"][cfg.dataset][cutoff]["tertiles"]
            strata = make_strata(data.test, data.task.rated_kw, edges)
            write_json(directory / "probability_strata.json", {
                "boundaries_sha256": sha256_file(frozen_path),
                "rows": stratified_scores(data.task, data.test, raw, strata)})
        run = {"identity": identity, "code_sha256": code_digest(), "git_commit": git_commit(paths.root),
               "environment": environment(), "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
               "validation_metrics": model.validation_metrics, "n_params": model.n_params,
               "scaler": ctx.scaler.to_dict(), "task": cfg.to_dict()}
        if hasattr(model, "save_checkpoint"):
            model.save_checkpoint(directory / "checkpoint.pt", metadata={"seed": seed, "probability_identity": identity})
        (directory / "train.log").write_text("\n".join(model.log_lines) + "\n", encoding="utf-8")
        write_json(directory / "run.json", run)
        files = [p for p in directory.iterdir() if p.is_file() and p.name != "RUNNING"]
        write_json(directory / "DONE", {"protocol": "unit_quantiles_v1", "identity": canonical_sha256(identity),
                                        "files": {p.name: sha256_file(p) for p in files}})
        return metrics
    finally:
        lock.unlink(missing_ok=True)


def score_probability(paths, task_name, predictions, directory):
    data = load_task_data(paths, load_task_config(paths, task_name))
    shape = (len(data.test.issues), data.task.H, len(data.test.entities), len(LEVELS))
    raw, _, identity = load_quantiles(predictions, data.test.issue_times, data.test.entities, shape,
                                      probability_identity(data))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Probability score output directory must be empty")
    metrics = write_probability_metrics(directory, data, raw)
    write_json(directory / "external_probability_identity.json", identity)
    return metrics
