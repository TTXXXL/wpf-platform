"""Command line entry point: wpf-benchmark <command>."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


def _options(pairs: Sequence[str]) -> dict:
    out = {}
    for pair in pairs or ():
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"Option must be key=value: {pair}")
        out[key] = value
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wpf-benchmark", description="Wind power forecasting platform")
    p.add_argument("--root", help="Project root holding configs/, data/ and results/")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("datasets", help="List known and built datasets")
    b = sub.add_parser("build", help="Build a dataset from its raw sources")
    b.add_argument("dataset")
    b.add_argument("--option", "-o", action="append", default=[], help="Adapter option key=value")
    sub.add_parser("tasks", help="List task configs and check them against built datasets")
    sub.add_parser("models", help="List registered models and their default parameters")
    r = sub.add_parser("run", help="Fit and score one model on one task")
    r.add_argument("--task", required=True)
    r.add_argument("--model", required=True, help="Model name or configs/models/<name>.json")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--save-predictions", action="store_true")
    r.add_argument("--no-checkpoint", action="store_true")
    s = sub.add_parser("score", help="Score an external predictions.npz on a task's test grid")
    s.add_argument("--task", required=True)
    s.add_argument("--predictions", required=True)
    s.add_argument("--out", required=True, help="Directory for metrics.json and per_horizon.csv")
    for command in ("run-probability", "score-probability"):
        qp = sub.add_parser(command, help="Explicit unit-marginal probability path")
        qp.add_argument("--task", required=True)
        qp.add_argument("--out", required=True)
        if command == "run-probability":
            qp.add_argument("--model", required=True)
            qp.add_argument("--seed", type=int, default=0)
        else:
            qp.add_argument("--predictions", required=True)
    st = sub.add_parser("study", help="Run a batch plan from configs/studies/")
    st.add_argument("--plan", required=True, help="Plan name or path")
    st.add_argument("--preflight", action="store_true", help="Only check; do not train")
    st.add_argument("--package", action="store_true", help="Only (re)build the delivery ZIP")
    st.add_argument("--allow-changed-code", action="store_true")
    st.add_argument("--rerun-stale", action="store_true",
                    help="Redo finished runs whose task, model, seed or data differ from the plan")
    st.add_argument("--only", help="Run only jobs whose task/model contains this text")
    st.add_argument("--include-predictions", action="store_true")
    st.add_argument("--include-checkpoints", action="store_true")
    return p


def _plan_path(paths, plan: str) -> Path:
    candidate = Path(plan)
    if candidate.suffix == ".json":
        return candidate if candidate.is_absolute() else paths.root / candidate
    return paths.configs / "studies" / f"{plan}.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    from .paths import ProjectPaths
    paths = ProjectPaths.resolve(args.root)
    if args.command == "datasets":
        from .datasets import ADAPTERS
        for name in sorted(ADAPTERS):
            meta_path = paths.dataset(name) / "meta.json"
            state = "not built"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                state = f"built: {meta['n_times']} times x {meta['n_entities']} entities, {meta['time_range']}"
            print(f"{name:14s} {state}")
    elif args.command == "build":
        from .datasets import build_dataset
        meta = build_dataset(args.dataset, paths, **_options(args.option))
        print(f"built {args.dataset}: {meta['n_times']} x {meta['n_entities']} -> {paths.dataset(args.dataset)}")
    elif args.command == "tasks":
        from .data.store import load_meta
        from .tasks import TaskConfig, resolve_task
        for path in sorted((paths.configs / "tasks").glob("*.json")):
            cfg = TaskConfig.load(path)
            try:
                task = resolve_task(cfg, load_meta(paths.dataset(cfg.dataset)))
                state = f"ok: L={task.L} H={task.H} masks={[m for m, _ in task.scorings]}"
            except FileNotFoundError:
                state = "dataset not built"
            print(f"{cfg.name:24s} {cfg.dataset:12s} {state}")
    elif args.command == "models":
        from .models import MODEL_REGISTRY
        for name, cls in sorted(MODEL_REGISTRY.items()):
            print(f"{name:22s} {json.dumps(cls.default_params)}")
    elif args.command == "run":
        from .runner import run_one
        directory, run = run_one(paths, args.task, args.model, args.seed, save_predictions=args.save_predictions,
                                 save_checkpoint=not args.no_checkpoint)
        print(json.dumps(run["summary"], indent=2))
        print(f"results -> {directory}")
    elif args.command == "score":
        from .evaluation import summary_row
        from .runner import score_predictions
        print(json.dumps(summary_row(score_predictions(paths, args.task, Path(args.predictions), Path(args.out))), indent=2))
    elif args.command in ("run-probability", "score-probability"):
        from .probability_runner import run_probability, score_probability
        if args.command == "run-probability":
            result = run_probability(paths, args.task, args.model, args.seed, Path(args.out))
        else:
            result = score_probability(paths, args.task, Path(args.predictions), Path(args.out))
        print(json.dumps({mask: score.get("normalized_mean_pinball") for mask, score in result["scores"].items()}, indent=2))
    elif args.command == "study":
        from . import study
        plan = _plan_path(paths, args.plan)
        if args.preflight:
            report = study.preflight(paths, study.load_plan(plan))
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
            return 1 if report["problems"] else 0
        elif args.package:
            print(study.package(paths, plan, args.include_predictions, args.include_checkpoints))
        else:
            study.run_study(paths, plan, args.allow_changed_code, args.only, rerun_stale=args.rerun_stale)
            print(study.package(paths, plan, args.include_predictions, args.include_checkpoints))
    return 0


if __name__ == "__main__":
    sys.exit(main())
