"""CPU profile orchestration tested with clearly labelled synthetic K/P-shaped panels."""
import json

import numpy as np

from wpf_benchmark.data.store import write_dataset
from wpf_benchmark.datasets.synthetic import make_panel
from wpf_benchmark.profile_probability import run_profile


def test_cpu_profile_four_cutoffs_and_frozen_training_edges(tmp_path):
    datasets = tmp_path / "datasets"
    for farm in ("kelmarsh", "penmanshiel"):
        panel = make_panel(days=390, entities=1, name=farm)
        # Synthetic dates cover the protocol's train prefix and four 2021 quarters.
        panel.times = panel.times + (np.datetime64("2020-12-08") - np.datetime64("2020-01-01"))
        write_dataset(datasets / farm, panel)
    out = tmp_path / "profile"
    run_profile(datasets, out, allow_synthetic=True)
    profile = json.loads((out / "profile.json").read_text(encoding="utf-8"))
    frozen = json.loads((out / "strata_boundaries.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "common_score_grid_manifest.json").read_text(encoding="utf-8"))
    assert len(profile["summaries"]) == 24
    assert profile["status"] == "software_test_only" and profile["software_test_only"]
    assert profile["no_model_weights_loaded"] and profile["no_training_or_prediction"]
    assert len(manifest["grids"]) == 8
    from wpf_benchmark.configure_probability import write_configs
    config_out = tmp_path / "configs"
    write_configs(datasets, config_out, model_dir="local-checkpoint")
    assert len(list((config_out / "tasks").glob("*.json"))) == 8
    assert len(list((config_out / "models").glob("*.json"))) == 14
    for farm in ("kelmarsh", "penmanshiel"):
        assert len(frozen["volatility_bins"][farm]) == 4
    for grid in manifest["grids"]:
        with np.load(out / "grids" / grid["file"], allow_pickle=False) as saved:
            assert saved["observed"].shape == tuple(grid["shape"])
            assert int(saved["observed"].sum()) == grid["n_valid_by_mask"]["observed"]
    import csv
    rows = list(csv.DictReader((out / "window_profile.csv").open(encoding="utf-8")))
    assert len(rows) == 72
    assert all(int(r["n_issues"]) > 0 and int(r["label_cross_boundary_count"]) == 0 for r in rows)
