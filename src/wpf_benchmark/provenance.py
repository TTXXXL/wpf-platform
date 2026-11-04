"""File, code and environment fingerprints recorded with every result."""
from __future__ import annotations

import functools
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """SHA256 of a JSON-serialisable value with sorted keys, so equal content hashes equal."""
    text = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=None)
def code_digest(root: Path = PACKAGE_ROOT) -> str:
    """SHA256 over the relative path and bytes of every package source file."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def git_commit(root: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def environment() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {"python": platform.python_version(),
                                          "platform": platform.platform()}
    for module in ("numpy", "pandas", "pyarrow", "torch", "lightgbm"):
        try:
            versions[module] = __import__(module).__version__
        except ImportError:
            versions[module] = None
    return versions
