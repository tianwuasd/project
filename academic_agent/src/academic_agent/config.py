"""Portable project configuration and atomic file primitives."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .filesystem import io_path
from .models import ProjectConfig


def atomic_json(path: Path, value: Any) -> None:
    physical = io_path(path)
    physical.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    fd, temporary = tempfile.mkstemp(dir=physical.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, physical)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_config(root: Path) -> ProjectConfig:
    return ProjectConfig.model_validate_json(io_path(root / "project.json").read_text(encoding="utf-8"))


def create_project(root: Path, config: ProjectConfig) -> None:
    io_path(root).mkdir(parents=True, exist_ok=True)
    if io_path(root / "project.json").exists():
        raise ValueError("Project already exists; use the existing project or a new directory")
    atomic_json(root / "project.json", config.model_dump())
