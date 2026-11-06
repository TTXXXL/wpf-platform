"""Write explicit Stage 0/1 configs from verified data metadata; does not run models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data.store import load_meta
from .tasks.probabilistic import CUTOFFS, probability_task


MODEL_CONFIGS = {
    "pers_q": {"model": "pers_q", "params": {"conditioning": "none", "min_samples": 200}},
    "pers_cq": {"model": "pers_q", "params": {"conditioning": "level", "min_samples": 200}},
    "pers_cq2": {"model": "pers_q", "params": {"conditioning": "level_volatility", "min_samples": 200}},
    **{f"{name}_q": {"model": f"{name}_q", "params": {"loss": "pinball"}}
       for name in ("dlinear", "lstm", "tcn", "patchtst", "itransformer", "agcrn")},
    "lightgbm_q": {"model": "lightgbm_q", "params": {}},
}


def write_configs(datasets, out, model_dir=None):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Config output must be empty")
    metas = {farm: load_meta(Path(datasets) / farm) for farm in ("kelmarsh", "penmanshiel")}
    tasks, models = out / "tasks", out / "models"
    tasks.mkdir(parents=True)
    models.mkdir()
    for meta in metas.values():
        for cutoff in CUTOFFS:
            cfg = probability_task(meta, cutoff)
            (tasks / f"{cfg.name}.json").write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    configs = dict(MODEL_CONFIGS)
    if model_dir:
        configs["toto1_zs_q"] = {"model": "toto1_zs_q", "params": {"model_dir": str(model_dir)}}
        for mode in ("head", "last4", "full"):
            configs[f"toto1_wpf_q_{mode}"] = {"model": "toto1_wpf_q", "params": {"model_dir": str(model_dir),
                                                                                       "adaptation": mode}}
    for name, cfg in configs.items():
        (models / f"{name}.json").write_text(json.dumps({"label": name, **cfg}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-dir")
    args = parser.parse_args()
    write_configs(args.datasets, args.out, args.model_dir)


if __name__ == "__main__":
    main()
