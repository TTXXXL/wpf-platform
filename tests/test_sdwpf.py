"""SDWPF adapter on a tiny raw CSV in the original KDD Cup layout."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wpf_benchmark.data.store import load_panel
from wpf_benchmark.datasets import build_dataset
from wpf_benchmark.paths import ProjectPaths


def _raw(root, days=3, turbines=4):
    raw = root / "data" / "raw" / "sdwpf"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(1)
    pd.DataFrame({"TurbID": np.arange(1, turbines + 1), "x": rng.uniform(0, 3000, turbines),
                  "y": rng.uniform(0, 3000, turbines)}).to_csv(raw / "sdwpf_baidukddcup2022_turb_location.csv", index=False)
    rows = []
    for tid in range(1, turbines + 1):
        for day in range(1, days + 1):
            for k in range(144):
                ws = 6 + 3 * np.sin(k / 20 + tid)
                rows.append({"TurbID": tid, "Day": day, "Tmstamp": f"{k // 6:02d}:{k % 6 * 10:02d}", "Wspd": ws,
                             "Wdir": 5.0, "Etmp": 20.0, "Itmp": 25.0, "Ndir": 100.0, "Pab1": 1.0, "Pab2": 1.0,
                             "Pab3": 1.0, "Prtv": 0.0, "Patv": 120 * ws - 300})
    df = pd.DataFrame(rows)
    df.loc[5, ["Wspd", "Patv"]] = np.nan                       # a recorded gap
    df.loc[10, "Patv"] = -5.0                                  # standby consumption
    df.loc[20, "Pab1"] = 95.0                                  # official abnormal pitch
    df.to_csv(raw / "sdwpf_245days_v1.csv", index=False)
    return raw


def test_build_from_raw(tmp_path):
    _raw(tmp_path)
    paths = ProjectPaths.resolve(tmp_path)
    meta = build_dataset("sdwpf", paths, train_days="2")
    assert meta["fit_window"] == ["1970-01-01", "1970-01-03"] and meta["n_entities"] == 4
    p = load_panel(paths.dataset("sdwpf"))
    assert str(p.times[0]).startswith("1970-01-01T00:00") and len(p.times) == 3 * 144
    t1 = p.entities.index("1")
    assert np.isnan(p.values["power_obs"][5, t1]) and p.values["power_obs"][10, t1] == 0
    assert not p.masks["m2"][5, t1] and not p.masks["m1"][20, t1] and not p.masks["m2"][20, t1]
    both = p.masks["m1"]
    np.testing.assert_allclose(p.values["power"][both], p.values["power_obs"][both], atol=1e-3)
    assert (p.masks["m1"] <= p.masks["power_finite"]).all()
    assert (paths.dataset("sdwpf") / "fit.npz").is_file()
