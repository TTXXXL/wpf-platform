"""Create and refresh the uploadable server copy from this development project.

Run this script only after local validation. The server copy is a generated mirror
that includes the SDWPF raw files but excludes built datasets and results.
The local .runtime, tests, diagnostic scripts, historical notes, and generated
reports stay in the development project. Docs and scripts use explicit allowlists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Dict, Optional


SOURCE = Path(__file__).resolve().parents[1]
TARGET = SOURCE.parent / "wpf-benchmark-server"
ARCHIVE = SOURCE.parent / "wpf-benchmark-server.zip"
MANIFEST = ".server-sync-manifest.json"
ROOT_FILES = (".gitignore", ".gitattributes", "README.md", "pyproject.toml", "THIRD_PARTY_NOTICES.md")
LEGACY_ROOT_FILES = ()
DATA_FILES = (
    "data/raw/sdwpf/sdwpf_245days_v1.csv",
    "data/raw/sdwpf/sdwpf_baidukddcup2022_turb_location.csv",
)
PROCESSED_FILES = (
    "data/datasets/sdwpf/panel.parquet",
    "data/datasets/kelmarsh/panel.parquet",
    "data/datasets/penmanshiel/panel.parquet",
)
TREE_DIRS = ("src", "configs", "docs", "scripts")
SERVER_SCRIPTS = {"scripts/sync_server_project.py"}
SERVER_DOCS = {
    "docs/平台接口.md", "docs/使用指南.md", "docs/数据集.md", "docs/模型.md",
    "docs/评估.md", "docs/服务器部署与批跑.md",
}
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
PLACEHOLDERS = (
    "data/raw/sdwpf/.gitkeep",
    "data/datasets/.gitkeep",
    "results/.gitkeep",
)


def digest(path: Path) -> str:
    hash_value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hash_value.update(chunk)
    return hash_value.hexdigest()


def source_files() -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for name in ROOT_FILES + DATA_FILES:
        path = SOURCE / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("Missing or linked project file: {}".format(path))
        files[name] = path
    for dirname in TREE_DIRS:
        root = SOURCE / dirname
        if not root.is_dir() or root.is_symlink():
            raise RuntimeError("Missing or linked project directory: {}".format(root))
        for path in root.rglob("*"):
            relative = path.relative_to(SOURCE)
            if any(part in SKIP_DIRS for part in relative.parts):
                continue
            if path.is_symlink():
                raise RuntimeError("Linked source entries are not supported: {}".format(path))
            if not path.is_file() or path.suffix in {".pyc", ".pyo"}:
                continue
            key = relative.as_posix()
            if dirname == "scripts" and key not in SERVER_SCRIPTS:
                continue
            if dirname == "docs" and key not in SERVER_DOCS:
                continue
            files[key] = path
    absent = (SERVER_SCRIPTS | SERVER_DOCS) - set(files)
    if absent:
        raise RuntimeError("Missing server docs/scripts: {}".format(sorted(absent)))
    return dict(sorted(files.items()))


def safe_destination(relative: str) -> Path:
    """Restrict manifest paths to managed source trees inside the server copy."""
    if "\\" in relative or ":" in relative:
        raise RuntimeError("Unsafe manifest path: {}".format(relative))
    parts = PurePosixPath(relative).parts
    if (
        not parts
        or PurePosixPath(relative).is_absolute()
        or any(part in {"", ".", ".."} for part in parts)
        or (
            relative not in ROOT_FILES
            and relative not in LEGACY_ROOT_FILES
            and relative not in DATA_FILES
            and relative not in PROCESSED_FILES
            and (len(parts) < 2 or parts[0] not in TREE_DIRS)
        )
    ):
        raise RuntimeError("Unsafe manifest path: {}".format(relative))
    destination = TARGET.joinpath(*parts)
    try:
        destination.resolve().relative_to(TARGET.resolve())
    except ValueError as error:
        raise RuntimeError("Manifest path leaves server project: {}".format(relative)) from error
    return destination


def old_manifest() -> Optional[Dict[str, str]]:
    path = TARGET / MANIFEST
    if not path.exists():
        return None
    content = json.loads(path.read_text(encoding="utf-8"))
    if content.get("schema") != 1 or not isinstance(content.get("managed"), dict):
        raise RuntimeError("Unsupported server sync manifest")
    managed = content["managed"]
    for relative, hash_value in managed.items():
        safe_destination(relative)
        if not isinstance(hash_value, str) or len(hash_value) != 64:
            raise RuntimeError("Invalid hash in server sync manifest: {}".format(relative))
    return managed


def verify(managed: Dict[str, str], current: Dict[str, str]) -> None:
    if managed != current:
        missing = sorted(set(current) - set(managed))
        stale = sorted(set(managed) - set(current))
        changed = sorted(key for key in set(managed) & set(current) if managed[key] != current[key])
        raise RuntimeError(
            "Development copy has unsynced changes (new={}, removed={}, changed={}). "
            "Run sync without --check.".format(len(missing), len(stale), len(changed))
        )
    for relative, expected in current.items():
        target_file = safe_destination(relative)
        if not target_file.is_file() or digest(target_file) != expected:
            raise RuntimeError("Server copy differs: {}".format(relative))
    for relative in PROCESSED_FILES:
        if safe_destination(relative).exists():
            raise RuntimeError("Server copy contains generated data: {}".format(relative))
    if not ARCHIVE.is_file():
        raise RuntimeError("Missing upload archive: {}".format(ARCHIVE))
    expected_names = {TARGET.name + "/" + relative for relative in current}
    expected_names.add(TARGET.name + "/" + MANIFEST)
    expected_names.update(TARGET.name + "/" + relative for relative in PLACEHOLDERS)
    with zipfile.ZipFile(ARCHIVE) as package:
        if set(package.namelist()) != expected_names:
            raise RuntimeError("Upload archive file list differs from the server copy")
        for relative, expected in current.items():
            hash_value = hashlib.sha256()
            with package.open(TARGET.name + "/" + relative) as member:
                for chunk in iter(lambda: member.read(1024 * 1024), b""):
                    hash_value.update(chunk)
            if hash_value.hexdigest() != expected:
                raise RuntimeError("Upload archive differs: {}".format(relative))


def sync(files: Dict[str, Path], current: Dict[str, str], previous: Optional[Dict[str, str]]) -> None:
    # Detect edits to the generated copy before writing any project files.
    for relative, expected in (previous or {}).items():
        target_file = safe_destination(relative)
        if target_file.exists() and (not target_file.is_file() or digest(target_file) != expected):
            raise RuntimeError("Server copy was edited directly: {}".format(relative))
    if previous is None:
        for relative in files:
            if safe_destination(relative).exists():
                raise RuntimeError("Existing server file has no sync manifest: {}".format(relative))

    TARGET.mkdir(parents=True, exist_ok=True)
    copied = 0
    for relative, source_file in files.items():
        target_file = safe_destination(relative)
        if not target_file.exists() or digest(target_file) != current[relative]:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)
            copied += 1

    removed = 0
    for relative in sorted(set(previous or {}) - set(current)):
        target_file = safe_destination(relative)
        if target_file.is_file():
            target_file.unlink()
            removed += 1

    for relative in PROCESSED_FILES:
        target_file = safe_destination(relative)
        if target_file.exists():
            if not target_file.is_file() or target_file.is_symlink():
                raise RuntimeError("Unexpected generated data entry: {}".format(relative))
            target_file.unlink()
            removed += 1

    # Retired directories should not remain visible in the generated copy.
    # Only remove empty ordinary directories within a managed tree.
    for dirname in TREE_DIRS:
        tree = TARGET / dirname
        for directory in sorted(tree.rglob("*"), key=lambda path: len(path.parts), reverse=True):
            if directory.is_symlink() or not directory.is_dir():
                continue
            directory.resolve().relative_to(tree.resolve())
            try:
                directory.rmdir()
            except OSError:
                pass  # Keep directories containing generated/unmanaged files.

    for relative in PLACEHOLDERS:
        placeholder = TARGET.joinpath(*PurePosixPath(relative).parts)
        placeholder.parent.mkdir(parents=True, exist_ok=True)
        placeholder.touch(exist_ok=True)

    manifest_path = TARGET / MANIFEST
    temporary_manifest = TARGET / (MANIFEST + ".tmp")
    temporary_manifest.write_text(
        json.dumps({"schema": 1, "managed": current}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary_manifest), str(manifest_path))

    temporary_archive = ARCHIVE.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for relative in sorted(current):
            package.write(safe_destination(relative), TARGET.name + "/" + relative)
        package.write(manifest_path, TARGET.name + "/" + MANIFEST)
        for relative in PLACEHOLDERS:
            package.write(TARGET / relative, TARGET.name + "/" + relative)
    os.replace(str(temporary_archive), str(ARCHIVE))
    print("Server copy: {}".format(TARGET))
    print("Upload ZIP: {} ({} bytes)".format(ARCHIVE, ARCHIVE.stat().st_size))
    print("Managed files: {}; copied: {}; removed: {}".format(len(current), copied, removed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the copy and ZIP without changing them")
    args = parser.parse_args()
    if TARGET.is_symlink() or (TARGET.exists() and not TARGET.is_dir()):
        raise RuntimeError("Server project target must be an ordinary directory")
    if ARCHIVE.is_symlink():
        raise RuntimeError("Upload archive must not be a symbolic link")
    files = source_files()
    current = {relative: digest(path) for relative, path in files.items()}
    previous = old_manifest()
    if args.check:
        if previous is None:
            raise RuntimeError("Server project has not been generated yet")
        verify(previous, current)
        print("Server copy and upload ZIP match {} development files.".format(len(current)))
    else:
        sync(files, current, previous)
        verify(current, current)


if __name__ == "__main__":
    main()
