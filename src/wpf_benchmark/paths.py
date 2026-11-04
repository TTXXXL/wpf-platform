"""Project directories shared by commands, adapters and the runner."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def resolve(cls, root: Optional[PathLike] = None) -> "ProjectPaths":
        if root is not None:
            candidate = Path(root)
        elif os.environ.get("WPF_BENCHMARK_ROOT"):
            candidate = Path(os.environ["WPF_BENCHMARK_ROOT"])
        else:
            candidate = cls._discover()
        return cls(candidate.expanduser().resolve())

    @staticmethod
    def _discover() -> Path:
        for candidate in (Path.cwd(), *Path.cwd().parents):
            if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
                return candidate
        return Path(__file__).resolve().parents[2]

    @property
    def datasets(self) -> Path:
        return self.root / "data" / "datasets"

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def results(self) -> Path:
        return self.root / "results"

    def dataset(self, name: str) -> Path:
        return self.datasets / name
