"""Synthetic task fixture, independent of model weights and external observations."""
import shutil
from pathlib import Path

import pytest

from wpf_benchmark.datasets import build_dataset
from wpf_benchmark.paths import ProjectPaths


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_project")
    shutil.copytree(Path(__file__).resolve().parents[1] / "configs", root / "configs")
    paths = ProjectPaths.resolve(root)
    build_dataset("synthetic", paths)
    return paths
