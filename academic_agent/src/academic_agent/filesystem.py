"""Platform-specific filesystem path forms that preserve logical path identity."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote


def io_path(path: Path) -> Path:
    """Return an absolute path suitable for long Windows filesystem operations."""
    resolved = Path(path).resolve()
    if os.name != "nt":
        return resolved
    value = str(resolved)
    if value.startswith("\\\\?\\"):
        return resolved
    if value.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + value[2:])
    return Path("\\\\?\\" + value)


def lexical_io_path(path: Path) -> Path:
    """Return a long-path form without resolving links in the supplied lexical path."""
    value = os.path.abspath(os.fspath(path))
    if os.name != "nt" or value.startswith("\\\\?\\"):
        return Path(value)
    if value.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + value[2:])
    return Path("\\\\?\\" + value)


def sqlite_path(path: Path) -> str:
    """Return a physical filename for SQLite without changing logical project paths."""
    return str(io_path(path))


def sqlite_uri(path: Path, *, mode: str = "ro") -> str:
    """Return an escaped SQLite URI for literal #, %, spaces and Unicode paths."""
    if mode not in {"ro", "rw", "rwc"}:
        raise ValueError("SQLite URI mode must be ro, rw or rwc")
    physical = io_path(path)
    if os.name == "nt":
        return "file:" + quote(str(physical), safe="/:") + f"?mode={mode}"
    return physical.as_uri() + f"?mode={mode}"
