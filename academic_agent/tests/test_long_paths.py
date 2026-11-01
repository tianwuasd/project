import hashlib
import sqlite3
from pathlib import Path

import pytest

from academic_agent.config import create_project, load_config
from academic_agent.documents import parse_document
from academic_agent.filesystem import io_path, sqlite_path, sqlite_uri
from academic_agent.models import ProjectConfig, Work
from academic_agent.store import Store


def _long_root(tmp_path: Path) -> Path:
    root = tmp_path
    index = 0
    while len(str((root / "中文 project #100%" / "research.db").resolve())) <= 300:
        root /= f"nested {index} abcdefghijklmnopqrstuvwxyz"
        index += 1
    root /= "中文 project #100%"
    assert len(str(root.resolve())) > 260
    assert not str(root.resolve()).startswith("\\\\?\\")
    return root


def test_long_project_config_store_and_document_keep_logical_paths(tmp_path):
    root = _long_root(tmp_path)
    config = ProjectConfig(title="Long path project", sources=[], backend="demo")
    create_project(root, config)
    assert load_config(root) == config

    store = Store(root)
    assert store.root == root.resolve()
    assert not str(store.root).startswith("\\\\?\\")
    work = store.upsert_work(Work(title="Long path fixture", screening="include", screening_reason="test"))
    source = root / "inputs" / "source 中文 #100%.txt"
    io_path(source).parent.mkdir(parents=True)
    io_path(source).write_text("Exact source text.", encoding="utf-8")
    parsed = parse_document(source, work.id)
    assert parsed.source_path == str(source.resolve())
    stored = store.add_document(parsed)
    assert stored.source_path.startswith("artifacts/")
    assert not stored.source_path.startswith("\\\\?\\")
    assert store.active_document(work.id) == stored
    assert store.evidence(work.id)[0].text == "Exact source text."


def test_long_sqlite_readonly_uri_sees_wal_without_writing_or_prefix_leak(tmp_path):
    root = _long_root(tmp_path)
    create_project(root, ProjectConfig(title="Long SQLite", sources=[]))
    store = Store(root)
    writer = sqlite3.connect(sqlite_path(store.db))
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("INSERT INTO logs VALUES ('long-path-log', 'probe', '{}', '2026-09-12T00:00:00+00:00')")
        writer.commit()
        wal = Path(str(store.db) + "-wal")
        before = {
            "db": hashlib.sha256(io_path(store.db).read_bytes()).hexdigest(),
            "wal": hashlib.sha256(io_path(wal).read_bytes()).hexdigest(),
        }
        uri = sqlite_uri(store.db)
        assert "%23" in uri and "%25" in uri
        reader = sqlite3.connect(uri, uri=True)
        try:
            assert reader.execute("SELECT kind FROM logs WHERE id='long-path-log'").fetchone() == ("probe",)
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                reader.execute("INSERT INTO logs VALUES ('forbidden', 'probe', '{}', 'now')")
        finally:
            reader.close()
        after = {
            "db": hashlib.sha256(io_path(store.db).read_bytes()).hexdigest(),
            "wal": hashlib.sha256(io_path(wal).read_bytes()).hexdigest(),
        }
        assert after == before
    finally:
        writer.close()
