"""Greenbyte adapter on a hand-made source: two turbines, two days, duplicate rows, status logs."""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from wpf_benchmark.data.store import load_panel
from wpf_benchmark.datasets import build_dataset
from wpf_benchmark.datasets import greenbyte
from wpf_benchmark.paths import ProjectPaths
from wpf_benchmark.tasks.config import TaskConfig

ROOT = Path(__file__).resolve().parents[1]
FARMS = {"kelmarsh": ("Kelmarsh", lambda n: f"Kelmarsh {n}", lambda n: f"Kelmarsh_{n}"),
         "penmanshiel": ("Penmanshiel", lambda n: f"Penmanshiel {n:02d}", lambda n: f"Penmanshiel_{n:02d}")}
COLUMNS = ["Date and time", "Wind speed (m/s)", "Wind speed, Standard deviation (m/s)", "Wind direction (°)",
           "Wind direction, Standard deviation (°)", "Nacelle position (°)", "Lost Production to Downtime (kWh)",
           "Lost Production to Curtailment (Total) (kWh)", "Power (kW)", "Nacelle ambient temperature (°C)",
           "Nacelle ambient temperature, Max (°C)", "Rotor speed (RPM)", "Blade angle (pitch position) A (°)"]
START = np.datetime64("2021-01-01T00:00:00")
NAN = float("nan")


def stamp(k: int) -> str:
    return str(START + np.timedelta64(10 * k, "m")).replace("T", " ")


def cells(row):
    return ["NaN" if isinstance(x, float) and x != x else x for x in row]


def scada_text(n_rows: int, turbine: str, power_shift: float = 0.0, odd: bool = False, mojibake: bool = False) -> str:
    out = io.StringIO()
    out.write(f"# This file was exported by Greenbyte.\n#\n# Turbine: {turbine}\n# Turbine type: Senvion\n"
              "# Time zone: UTC\n# Time interval: x\n#\n# Data that is missing has been marked with the value \"NaN\"\n#\n")
    w = csv.writer(out, lineterminator="\n")
    header = [c.replace("°", "��") if mojibake else c for c in COLUMNS]
    out.write("# ")
    w.writerow(header)
    for k in range(n_rows):
        ws, power, lpd, lpc = 5 + (k % 10) * 0.1, 100.0 + k + power_shift, 0.0, 0.0
        if odd and k == 10:
            power = -3.0
        if odd and k == 20:
            ws = NAN
        if odd and k == 30:
            lpd = 5.0
        if odd and k == 40:
            lpc = NAN
        w.writerow(cells([stamp(k), ws, 0.1, (5 * k) % 360, 1.0, (5 * k) % 360, lpd, lpc, power, 10.0, 12.0, 11.0, 1.5]))
        if odd and k == 100:
            for _ in range(3):
                w.writerow([stamp(k)] + ["NaN"] * (len(COLUMNS) - 1))
    return out.getvalue()
#__CONTINUE__


HEAD = ("Timestamp start,Timestamp end,Duration,Status,Code,Message,Comment,Service contract category,"
        "IEC category,Global contract category,Custom contract category")
T1_EVENTS = [("2020-12-31 23:00:00", "2021-01-01 00:15:00", "Stop", "6540", "Out of Environmental Specification"),
             ("2021-01-01 01:03:00", "2021-01-01 01:25:00", "Stop", "6540", "Forced outage"),
             ("2021-01-01 01:03:00", "2021-01-01 01:25:00", "Stop", "6540", "Forced outage"),
             ("2021-01-01 02:00:00", "2021-01-01 02:30:00", "Informational", "10", "Out of Environmental Specification"),
             ("2021-01-01 02:30:00", "2021-01-01 03:00:00", "Informational", "65", "Out of Environmental Specification"),
             ("2021-01-01 03:00:00", "2021-01-01 03:10:00", "Informational", "245", ""),
             ("2021-01-01 03:30:00", "2021-01-01 04:00:00", "Communication", "9997", ""),
             ("2021-01-01 04:30:00", "2021-01-01 05:30:00", "Warning", "9000", "Full Performance"),
             ("2021-01-01 06:00:00", "2021-01-01 06:00:00", "Stop", "1", "Forced outage"),
             ("2021-01-01 07:00:00", "-", "Informational", "245", ""),
             ("2021-01-01 08:00:00", "2021-01-01 08:05:00", "Curtailment", "108", "Partial Performance")]
T2_EVENTS = [("2021-01-01 09:00:00", "2021-01-01 09:10:00", "Stop", "1", "Forced outage")]
COVERED_T1 = [0, 1, 6, 7, 8, 18, 21, 22, 23, 48]
COVERED_T2 = [54]


def status_text(events) -> str:
    rows = [f"{a},{b},00:00:00,{s},{c},message,,,{iec},," for a, b, s, c, iec in events]
    return "# This file was exported by Greenbyte.\n#\n# Turbine: x\n#\n" + "\n".join([HEAD] + rows) + "\n"


def write_zip(path: Path, members: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in members.items():
            z.writestr(name, text.encode("utf-8"))


def static_text(farm: str, numbers) -> str:
    title = FARMS[farm][1]
    head = ("Wind Farm,Title,Alternative Title,Identity,Manufacturer,Model,Rated power (kW),Hub Height (m),"
            "Rotor Diameter (m),Latitude,Longitude,Elevation (m),Country,Commercial Operations Date")
    rows = [f"{FARMS[farm][0]},{title(n)},T{n},ID{n},Senvion,MM92,2050,{60 + n},92,52.{n},-0.9{n},100,UK,15/04/2016"
            for n in numbers]
    blanks = [",,,,,,,,,,,,,"] * 3 if farm == "penmanshiel" else []
    return "\ufeff" + "\r\n".join([head] + rows + blanks) + "\r\n"


def make_source(root: Path, farm: str, numbers) -> Path:
    src = root / farm
    (src / "raw").mkdir(parents=True)
    (src / "raw" / greenbyte.FARMS[farm]["static"]).write_text(static_text(farm, numbers), encoding="utf-8", newline="")
    label = FARMS[farm][2]
    new, old = {}, {}
    for i, n in enumerate(numbers):
        new[f"Turbine_Data_{label(n)}_2021-01-01_-_2021-01-03_{228 + i}.csv"] = scada_text(
            288, FARMS[farm][1](n), odd=(i == 0), mojibake=(i == 1))
        new[f"Status_{label(n)}_2021-01-01_-_2021-01-03_{228 + i}.csv"] = status_text(T1_EVENTS if i == 0 else T2_EVENTS)
        old[f"Turbine_Data_{label(n)}_2021-01-01_-_2021-01-02_{228 + i}.csv"] = scada_text(144, FARMS[farm][1](n), power_shift=50.0)
        old[f"Status_{label(n)}_2021-01-01_-_2021-01-02_{228 + i}.csv"] = status_text(
            [("2021-01-01 00:00:00", "2021-01-02 00:00:00", "Stop", "1", "Forced outage")])
    write_zip(src / "2022-2024" / f"{farm}_SCADA_2021_new.zip", new)
    write_zip(src / "archives" / f"{farm}_SCADA_2021_old.zip", old)
    write_zip(src / "archives" / f"{farm}_Grid_1.zip", {"Grid_x.csv": "a,b\n1,2\n"})
    (src / "2021").mkdir()
    (src / "2021" / f"{farm}_SCADA_2021_full.zip.part").write_bytes(b"unfinished")
    (src / "2021" / f"{farm}_SCADA_broken.zip").write_bytes(b"not a zip")
    return src


@pytest.fixture(scope="module", params=["kelmarsh", "penmanshiel"])
def built(request, tmp_path_factory):
    farm = request.param
    root = tmp_path_factory.mktemp(farm)
    numbers = [1, 2] if farm == "kelmarsh" else [1, 4]
    src = make_source(root, farm, numbers)
    paths = ProjectPaths.resolve(root / "project")
    meta = build_dataset(farm, paths, source=str(src), start="2021-01-01", end="2021-01-03", workers=1)
    return farm, src, paths, meta, load_panel(paths.dataset(farm))


def test_grid_entities_and_meta(built):
    farm, _, _, meta, panel = built
    assert panel.values["power"].shape == (288, 2)
    expected = ["Kelmarsh 1", "Kelmarsh 2"] if farm == "kelmarsh" else ["Penmanshiel 01", "Penmanshiel 04"]
    assert panel.entities == expected
    entity = meta["entities"][0]
    assert entity["rated_kw"] == 2050 and entity["cod"] == "2016-04-15" and entity["hub_height_m"] == 61
    assert {"lat", "lon", "model"} <= set(entity)
    assert meta["fit_window"] is None and meta["default_target"] == "power" and meta["resolution_minutes"] == 10
    assert set(meta["masks"]) == {"observed", "normal_b", "normal_c"}
    assert all(m["target"] == "power" and m["fit_window"] is None for m in meta["masks"].values())
    assert set(meta["variables"]) == set(panel.values) and "lp_curtailment" in panel.values


def test_values_dedup_clip_and_supersede(built):
    _, _, _, _, panel = built
    raw, power = panel.values["power_raw"], panel.values["power"]
    assert raw[0, 0] == 100 and raw[287, 0] == 100 + 287          # new version wins over old (+50)
    assert raw[100, 0] == 200                                     # first of the duplicate rows kept
    assert raw[10, 0] == -3 and power[10, 0] == 0                 # negative clipped, raw kept
    assert np.isnan(panel.values["wind_speed"][20, 0]) and np.isfinite(raw[20, 0])
    assert np.isfinite(panel.values["wind_dir"]).all()            # mojibake degree sign still matched
    assert panel.values["temp_ambient"][0, 0] == 10 and panel.values["pitch"][0, 0] == 1.5
    k = 7
    np.testing.assert_allclose(panel.values["wind_dir_sin"][k, 0], np.sin(np.deg2rad(5 * k)), atol=1e-6)
    np.testing.assert_allclose(panel.values["wind_dir_cos"][k, 0], np.cos(np.deg2rad(5 * k)), atol=1e-6)
    assert all(a.dtype == np.float32 for a in panel.values.values())


def test_masks(built):
    _, _, _, _, panel = built
    obs, b, c = (panel.masks[k] for k in ("observed", "normal_b", "normal_c"))
    assert obs.all() == False and obs[20, 0] == False and obs.sum() == 2 * 288 - 1
    assert not b[30, 0] and not b[40, 0] and b[10, 0] and b[0, 0]
    assert not b[20, 0] and b.sum() == obs.sum() - 2
    covered = np.zeros((288, 2), bool)
    covered[COVERED_T1, 0] = True
    covered[COVERED_T2, 1] = True
    np.testing.assert_array_equal(c, obs & ~covered)
    assert c[12:18, 0].all() and c[19, 0] and c[24, 0]            # exempt 10/65, 245 boundary, Warning, open-ended
    assert c[42, 0] and c[43, 0]                                  # zero-length stop and open-ended 245 are ignored
    assert c[0, 1]                                                # the superseded zip's whole-day Stop is not read
    assert (b & ~obs).sum() == 0 and (c & ~obs).sum() == 0


def test_sources_report_and_skips(built):
    farm, src, paths, meta, _ = built
    roles = {Path(s["path"]).name: s for s in meta["sources"]}
    assert set(roles) == {f"{farm}_SCADA_2021_new.zip", f"{farm}_SCADA_2021_old.zip"}
    new = roles[f"{farm}_SCADA_2021_new.zip"]
    assert new["bytes"] == (src / "2022-2024" / f"{farm}_SCADA_2021_new.zip").stat().st_size
    assert len(new["sha256"]) == 64 and new["role"] == "used" and roles[f"{farm}_SCADA_2021_old.zip"]["role"].startswith("superseded")
    skipped = " ".join(s["path"] + s["reason"] for s in meta["build"]["skipped"])
    assert ".zip.part" in skipped and "broken" in skipped and "Grid" not in skipped
    report = (paths.dataset(farm) / "build_report.md").read_text(encoding="utf-8")
    assert "功率冲突共 288 格" in report and "去重行数 3" in report and "2021" in report
    assert "规则 C 事件 6 条" in report
    assert json.loads((paths.dataset(farm) / "meta.json").read_text(encoding="utf-8"))["adapter"] == "greenbyte"


def test_unknown_farm_and_missing_static(tmp_path):
    with pytest.raises(KeyError):
        greenbyte.build(ProjectPaths.resolve(tmp_path), "sdwpf")
    (tmp_path / "raw").mkdir()
    with pytest.raises(FileNotFoundError):
        greenbyte.build(ProjectPaths.resolve(tmp_path), "kelmarsh", source=str(tmp_path))


@pytest.mark.parametrize("farm", ["kelmarsh", "penmanshiel"])
@pytest.mark.parametrize("horizon", ["6h", "2h"])
def test_task_configs_match_adapter(farm, horizon):
    cfg = TaskConfig.load(ROOT / "configs" / "tasks" / f"{farm}_actual_{horizon}.json")
    assert cfg.dataset == farm and cfg.target == "power" and cfg.target_input == "power"
    assert list(cfg.inputs) == ["wind_speed", "wind_dir_sin", "wind_dir_cos", "power"]
    assert cfg.history_minutes == 1440 and cfg.horizon_minutes == (360 if horizon == "6h" else 120)
    assert cfg.train_stride_minutes == 60 and cfg.eval_stride_minutes == 10
    assert cfg.splits["train"] == (greenbyte.FARMS[farm]["cod"], "2022-01-01")
    assert cfg.splits["valid"] == ("2022-01-01", "2023-01-01") and cfg.splits["test"] == ("2023-01-01", "2025-01-01")
    assert (cfg.train_mask, cfg.valid_mask, cfg.eval_mask) == ("observed",) * 3
    assert cfg.extra_eval_masks == (("normal_b", "power"), ("normal_c", "power"))
    assert cfg.report_horizons_minutes == ((60, 120, 360) if horizon == "6h" else (60, 120))
