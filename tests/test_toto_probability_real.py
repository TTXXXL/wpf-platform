"""Opt-in actual-checkpoint shared trainer roundtrip, using synthetic data only."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from wpf_benchmark.data.store import write_dataset
from wpf_benchmark.datasets.synthetic import make_panel
from wpf_benchmark.models import get_model
from wpf_benchmark.models.neural.trainer import load_checkpoint
from wpf_benchmark.runner import TaskData, make_context, seed_all
from wpf_benchmark.tasks import TaskConfig, build_split, resolve_task


@pytest.mark.skipif(not os.environ.get("WPF_TOTO_DIR"), reason="Real checkpoint test requires WPF_TOTO_DIR")
def test_real_toto_shared_training_and_checkpoint(tmp_path):
    panel = make_panel(days=8,entities=1)
    meta = write_dataset(tmp_path / "dataset",panel)
    cfg = TaskConfig.from_dict({"name":"toto_synthetic_acceptance","dataset":"synthetic","target":"power",
        "inputs":["wind_speed","wind_dir_sin","wind_dir_cos","power"],
        "history_minutes":1440,"horizon_minutes":360,"train_mask":"observed","eval_mask":"observed",
        "splits":{"train":["2020-01-01","2020-01-05"],"valid":["2020-01-05","2020-01-07"],
                  "test":["2020-01-07","2020-01-09"]},
        "extra":{"window_protocol":"causal_history_v1","probability_protocol":"unit_quantiles_v1"}})
    task = resolve_task(cfg,meta)
    splits = [build_split(task,panel,name) for name in ("train","valid","test")]
    for split in splits:
        split.issues = split.issues[:2]
    ctx = make_context(TaskData(task,*splits,{"panel_sha256":"synthetic","meta_sha256":"synthetic"}))
    seed_all(0)
    model = get_model("toto1_wpf_q")(model_dir=os.environ["WPF_TOTO_DIR"],epochs=1,batch=1)
    model.fit(ctx,splits[0],splits[1])
    batch = splits[2].batch(splits[2].issues[:1])
    raw = model.predict_quantiles(batch)
    assert raw.shape == (1,36,1,9)
    assert model.probability_metadata["adaptation"] == "head"
    assert model.probability_metadata["trainable_parameters"] == 6921
    assert len(model.probability_metadata["epoch_seconds"]) == 1
    path = tmp_path / "checkpoint.pt"
    model.save_checkpoint(path)
    loaded = load_checkpoint(path,task,device="cpu")
    np.testing.assert_allclose(raw,loaded.predict_quantiles(batch),atol=1e-3,rtol=1e-5)
    assert loaded.n_params == model.n_params
