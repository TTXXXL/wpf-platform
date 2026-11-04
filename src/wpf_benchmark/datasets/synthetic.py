"""Small deterministic dataset for tests and model smoke runs. No raw files needed."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from ..data.panel import Panel

ADAPTER_VERSION = "1"


def make_panel(days: int = 30, entities: int = 3, seed: int = 0, name: str = "synthetic") -> Panel:
    rng = np.random.default_rng(seed)
    T = days * 144
    times = np.datetime64("2020-01-01T00:00", "us") + np.arange(T) * np.timedelta64(10, "m")
    t = np.arange(T)[:, None]
    phase = rng.uniform(0, 2 * np.pi, entities)[None, :]
    wind = 7 + 3 * np.sin(2 * np.pi * t / 144 + phase) + 2 * np.sin(2 * np.pi * t / 1008) + rng.normal(0, 0.6, (T, entities))
    wind = np.clip(wind, 0, None).astype(np.float32)
    power = (2000 / (1 + np.exp(-(wind - 8) / 1.3))).astype(np.float32)
    power[wind < 3] = 0
    power = np.clip(power + rng.normal(0, 30, power.shape), -10, 2050).astype(np.float32)
    direction = np.deg2rad((200 + 40 * np.sin(2 * np.pi * t / 2000) + 0 * phase) % 360)
    missing = rng.random((T, entities)) < 0.01
    power_raw = np.where(missing, np.nan, power).astype(np.float32)
    clipped = np.maximum(power_raw, 0).astype(np.float32)
    observed = np.isfinite(clipped) & np.isfinite(wind)
    normal = observed & ~((wind >= 4) & (clipped <= 5))
    ids = [f"T{i + 1:02d}" for i in range(entities)]
    meta = {"name": name, "title": "Synthetic test farm", "adapter": "synthetic", "adapter_version": ADAPTER_VERSION,
            "resolution_minutes": 10, "nominal_dates": True, "fit_window": None, "default_target": "power",
            "entities": [{"id": i, "rated_kw": 2050.0} for i in ids],
            "variables": {"power": {"unit": "kW", "description": "Active power, negatives clipped to 0", "source": "synthetic"},
                          "power_raw": {"unit": "kW", "description": "Active power as generated", "source": "synthetic"},
                          "wind_speed": {"unit": "m/s", "description": "Wind speed", "source": "synthetic"},
                          "wind_dir_sin": {"unit": "1", "description": "sin(wind direction)", "source": "synthetic"},
                          "wind_dir_cos": {"unit": "1", "description": "cos(wind direction)", "source": "synthetic"}},
            "masks": {"observed": {"target": "power", "description": "power and wind present", "fit_window": None},
                      "normal": {"target": "power", "description": "observed and not stopped in wind", "fit_window": None}},
            "sources": [], "build": {"days": days, "entities": entities, "seed": seed}}
    values = {"power": clipped, "power_raw": power_raw, "wind_speed": wind,
              "wind_dir_sin": np.broadcast_to(np.sin(direction), (T, entities)).astype(np.float32),
              "wind_dir_cos": np.broadcast_to(np.cos(direction), (T, entities)).astype(np.float32)}
    return Panel(meta, times, ids, values, {"observed": observed, "normal": normal})


def build(paths, name: str = "synthetic", days: int = 30, entities: int = 3, seed: int = 0) -> Tuple[Panel, str]:
    from . import option_int
    panel = make_panel(option_int("days", days), option_int("entities", entities),
                       option_int("seed", seed, minimum=0), name)
    return panel, f"# {name}\n\nSynthetic panel: {days} days, {entities} turbines, seed {seed}.\n"
