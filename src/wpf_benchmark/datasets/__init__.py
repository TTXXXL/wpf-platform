"""Dataset adapters. Each adapter turns raw source files into a stored Panel.

An adapter module defines ``build(paths, name, **options) -> (Panel, report_markdown)``
and is listed in ``ADAPTERS``. Modules are imported only when that dataset is built.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict

from ..data.store import write_dataset
from ..paths import ProjectPaths

def option_int(name: str, value: Any, minimum: int = 1) -> int:
    """Parse an adapter option that the CLI passes as text; the error names the option."""
    try:
        number = int(str(value).strip())
    except ValueError:
        raise ValueError(f"Option {name} must be an integer, got {value!r}") from None
    if number < minimum:
        raise ValueError(f"Option {name} must be at least {minimum}, got {number}")
    return number


ADAPTERS: Dict[str, str] = {
    "sdwpf": "wpf_benchmark.datasets.sdwpf",
    "kelmarsh": "wpf_benchmark.datasets.greenbyte",
    "penmanshiel": "wpf_benchmark.datasets.greenbyte",
    "synthetic": "wpf_benchmark.datasets.synthetic",
}


def build_dataset(name: str, paths: ProjectPaths, **options: Any) -> Dict[str, Any]:
    """Build one dataset from its raw sources and write data/datasets/<name>/."""
    if name not in ADAPTERS:
        raise KeyError(f"Unknown dataset {name!r}; available: {', '.join(sorted(ADAPTERS))}")
    module = importlib.import_module(ADAPTERS[name])
    panel, report = module.build(paths, name=name, **options)
    if panel.meta.get("name") != name:
        raise ValueError(f"Adapter returned metadata for {panel.meta.get('name')!r}, expected {name!r}")
    return write_dataset(paths.dataset(name), panel, report)
