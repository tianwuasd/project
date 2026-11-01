"""Host-controlled, integrity-checked immutable research version directories."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import atomic_json
from .filesystem import io_path, lexical_io_path
from .models import digest
from .research_context import (
    ResearchSnapshot,
    _assert_locked,
    assert_research_current,
    research_fingerprint,
)
from .store import Store

KINDS = {"innovation", "challenge", "assessment", "domain", "writing"}
PAYLOADS = {
    "domain": {"context.json", "domain.json", "domain-review.json", "comparison.md", "references.bib",
               "evidence.json"},
    "writing": {"context.json", "outline.json", "draft.json", "review.json", "revision.json", "raw-edit.bin",
                "report.md", "references.bib", "evidence.json"},
}
_VERSION = re.compile(r"v-[0-9a-f]{32}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\Z")
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
             *(f"LPT{i}" for i in range(1, 10))}


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1]
    kind: Literal["innovation", "challenge", "assessment", "domain", "writing"]
    version: str
    snapshot: ResearchSnapshot
    fingerprint: str
    files: dict[str, str] = Field(min_length=1)
    manifest_hash: str


class _Latest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1]
    version: str
    manifest_hash: str


def _canonical(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def _safe_path(path: Path) -> None:
    # Check before resolve(), otherwise a symlink would already be hidden.
    for part in [path, *path.parents]:
        physical = lexical_io_path(part)
        if physical.is_symlink() or getattr(physical, "is_junction", lambda: False)():
            raise ValueError("Research artifact path is a symlink or junction")


def _filename(name: str) -> None:
    if (not isinstance(name, str) or not _NAME.fullmatch(name) or name.endswith(".")
            or name.split(".")[0].upper() in _RESERVED
            or name.casefold() in {"manifest.json", "latest.json"}):
        raise ValueError("Unsafe or reserved research payload filename")


def _read_json(path: Path) -> tuple[bytes, dict]:
    _safe_path(path)
    try:
        data = io_path(path).read_bytes()
        value = json.loads(data)
        if not isinstance(value, dict) or data != _canonical(value).encode("utf-8"):
            raise ValueError("Research JSON is malformed or not canonical")
        return data, value
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Research artifact is missing or malformed") from exc


def load_research(directory: Path) -> dict:
    """Load a version, or the verified latest version from analyses/<kind>.

    Returns exactly {"manifest": dict, "files": dict[str, bytes]}.
    Hashes detect changes, not author authenticity.
    """
    directory = Path(directory).absolute()
    _safe_path(directory)
    latest = None
    if directory.name in KINDS:
        if directory.parent.name != "analyses":
            raise ValueError("Research kind directory is outside the expected layout")
        _, pointer = _read_json(directory / "latest.json")
        latest = _Latest.model_validate(pointer)
        if not _VERSION.fullmatch(latest.version) or not _HASH.fullmatch(latest.manifest_hash):
            raise ValueError("Unsafe research latest pointer")
        directory = directory / latest.version
    if (not _VERSION.fullmatch(directory.name) or directory.parent.name not in KINDS
            or directory.parent.parent.name != "analyses"):
        raise ValueError("Invalid research version path")
    _safe_path(directory)
    _, raw = _read_json(directory / "manifest.json")
    manifest = _Manifest.model_validate(raw)
    body = {k: v for k, v in raw.items() if k != "manifest_hash"}
    if (manifest.kind != directory.parent.name or manifest.version != directory.name
            or manifest.manifest_hash != digest(_canonical(body))
            or manifest.fingerprint != research_fingerprint(manifest.snapshot)
            or latest and latest.manifest_hash != manifest.manifest_hash):
        raise ValueError("Research manifest integrity mismatch")
    names = list(manifest.files)
    expected = PAYLOADS.get(manifest.kind)
    if expected is not None and set(names) != expected:
        raise ValueError("Research payload set is invalid for artifact kind")
    for name in names:
        _filename(name)
    if len({n.casefold() for n in names}) != len(names):
        raise ValueError("Research payload filenames collide")
    if {p.name for p in io_path(directory).iterdir()} != set(names) | {"manifest.json"}:
        raise ValueError("Research version has extra or missing files")
    files = {}
    for name, sha in manifest.files.items():
        path = directory / name
        _safe_path(path)
        try:
            data = io_path(path).read_bytes()
        except OSError as exc:
            raise ValueError("Research payload is missing or unreadable") from exc
        if not _HASH.fullmatch(sha) or hashlib.sha256(data).hexdigest() != sha:
            raise ValueError("Research payload hash mismatch; preserve and inspect human edits")
        files[name] = data
    return {"manifest": raw, "files": files}


def _check_history(base: Path) -> None:
    _safe_path(base)
    physical_base = io_path(base)
    if not physical_base.exists():
        return
    if not physical_base.is_dir():
        raise ValueError("Research kind path is not a directory")
    for entry in physical_base.iterdir():
        path = base / entry.name
        _safe_path(path)
        if _VERSION.fullmatch(path.name):
            load_research(path)
        elif path.name == "latest.json":
            load_research(base)
        elif not path.name.startswith(".pending-") or not io_path(path).is_dir():
            raise ValueError("Unexpected research history entry; inspect without overwriting")


def _check_dependency(dependency_check: Callable[[], None] | None, output: Path | None = None) -> None:
    if dependency_check is None:
        return
    try:
        dependency_check()
    except Exception as exc:
        from .writing_models import WritingError

        if output is not None and isinstance(exc, WritingError) and exc.output is None:
            exc.output = output
        raise


def publish_research(store: Store, kind: str, snapshot: ResearchSnapshot,
                     payloads: dict[str, bytes], *,
                     dependency_check: Callable[[], None] | None = None) -> Path:
    if kind not in KINDS:
        raise ValueError("Unknown research artifact kind")
    if not isinstance(payloads, dict) or not payloads:
        raise ValueError("Research publication requires named byte payloads")
    expected = PAYLOADS.get(kind)
    if expected is not None and set(payloads) != expected:
        raise ValueError("Research payload set is invalid for artifact kind")
    for name, data in payloads.items():
        _filename(name)
        if not isinstance(data, bytes):
            raise ValueError("Research payload must be bytes")
    if len({n.casefold() for n in payloads}) != len(payloads):
        raise ValueError("Research payload filenames collide")
    _check_dependency(dependency_check)
    fingerprint = research_fingerprint(snapshot)
    assert_research_current(store, snapshot)
    base = store.root / "analyses" / kind
    _check_history(base)
    io_path(base).mkdir(parents=True, exist_ok=True)
    version = "v-" + uuid.uuid4().hex
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=io_path(base)))
    hashes = {}
    for name, data in sorted(payloads.items()):
        with (staging / name).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        hashes[name] = hashlib.sha256(data).hexdigest()
    manifest = {"schema_version": 1, "kind": kind, "version": version,
                "snapshot": snapshot.model_dump(mode="json"), "fingerprint": fingerprint, "files": hashes}
    manifest["manifest_hash"] = digest(_canonical(manifest))
    atomic_json(staging / "manifest.json", json.loads(_canonical(manifest)))
    assert_research_current(store, snapshot)
    _check_dependency(dependency_check)
    destination = base / version
    with store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        _assert_locked(store, connection, snapshot)
        _check_history(base)
        _safe_path(destination)
        if io_path(destination).exists():
            raise ValueError("Research version already exists")
        _check_dependency(dependency_check)
        staging.rename(io_path(destination))
        # Verify bytes before moving the pointer; config/blob files have no SQLite lock.
        load_research(destination)
        _assert_locked(store, connection, snapshot)
        _check_dependency(dependency_check, destination)
        atomic_json(base / "latest.json", json.loads(_canonical(
            {"schema_version": 1, "version": version, "manifest_hash": manifest["manifest_hash"]})))
        _check_dependency(dependency_check, destination)
    return destination
