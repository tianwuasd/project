import json
import sqlite3

import pytest

from academic_agent.config import atomic_json
from academic_agent.demo import seed_demo
from academic_agent.research_artifacts import load_research, publish_research
from academic_agent.research_context import freeze_research
from academic_agent.workflow import Runner
from academic_agent.writing_models import WritingError


def _domain_payloads(marker: bytes = b"{}"):
    return {
        "context.json": marker, "domain.json": b"{}", "domain-review.json": b"{}",
        "comparison.md": b"", "references.bib": b"", "evidence.json": b"{}",
    }


def _writing_payloads(marker: bytes = b"{}"):
    return {
        "context.json": marker, "outline.json": b"{}", "draft.json": b"{}", "review.json": b"{}",
        "revision.json": b"{}", "raw-edit.bin": b"", "report.md": b"", "references.bib": b"",
        "evidence.json": b"{}",
    }


@pytest.fixture
def research(tmp_path):
    seed_demo(tmp_path)
    runner = Runner(tmp_path, offline=True)
    runner.run()
    return runner.store, freeze_research(runner.store)


def test_versions_and_latest(research):
    store, snapshot = research
    first = publish_research(store, "innovation", snapshot, {"candidate.json": b'{"candidate":1}'})
    original = {p.name: p.read_bytes() for p in first.iterdir()}
    second = publish_research(store, "innovation", snapshot, {"candidate.json": b'{"candidate":2}'})
    assert first != second
    assert load_research(first)["files"]["candidate.json"] == b'{"candidate":1}'
    assert load_research(second.parent)["files"]["candidate.json"] == b'{"candidate":2}'
    assert {p.name: p.read_bytes() for p in first.iterdir()} == original


@pytest.mark.parametrize("defect", ["payload", "manifest", "extra", "missing", "latest", "manifest_extra"])
def test_reject_modified_records(research, defect):
    store, snapshot = research
    directory = publish_research(store, "assessment", snapshot, {"report.md": b"original"})
    if defect == "payload":
        (directory / "report.md").write_bytes(b"human edit")
    elif defect == "manifest":
        data = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        data["snapshot"]["question"] = "Altered"
        (directory / "manifest.json").write_bytes(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2).encode())
    elif defect == "extra":
        (directory / "extra.txt").write_text("unexpected", encoding="utf-8")
    elif defect == "missing":
        (directory / "report.md").unlink()
    elif defect == "latest":
        path = directory.parent / "latest.json"
        data = json.loads(path.read_bytes())
        data["version"] = "../../outside"
        path.write_bytes(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2).encode())
    else:
        data = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        data["unrecognized"] = "unexpected"
        (directory / "manifest.json").write_bytes(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2).encode())
    with pytest.raises(ValueError):
        load_research(directory.parent if defect == "latest" else directory)
    with pytest.raises(ValueError):
        publish_research(store, "assessment", snapshot, {"report.md": b"replacement"})


@pytest.mark.parametrize("name", ["../outside", "a/b", "a\\b", "manifest.json", "CON", "a:stream", "REPORT."])
def test_reject_hostile_names(research, name):
    store, snapshot = research
    with pytest.raises(ValueError):
        publish_research(store, "innovation", snapshot, {name: b"bad"})
    with pytest.raises(ValueError):
        publish_research(store, "../outside", snapshot, {"safe.txt": b"bad"})


def test_stale_publication_rejected(research):
    store, snapshot = research
    store.screen(snapshot.work_ids[0], "exclude", "New scope")
    with pytest.raises(ValueError):
        publish_research(store, "challenge", snapshot, {"search.json": b"{}"})
    assert not (store.root / "analyses" / "challenge" / "latest.json").exists()


def test_mutation_during_publish(research, monkeypatch):
    import academic_agent.research_artifacts as artifacts

    store, snapshot = research
    original = artifacts.atomic_json

    def mutate(path, value):
        original(path, value)
        if path.name == "manifest.json":
            store.screen(snapshot.work_ids[0], "exclude", "Concurrent change")

    monkeypatch.setattr(artifacts, "atomic_json", mutate)
    with pytest.raises(ValueError, match="changed"):
        publish_research(store, "innovation", snapshot, {"report.md": b"report"})
    assert not (store.root / "analyses" / "innovation" / "latest.json").exists()


def test_locked_final_check_catches_change_after_read_fence(research, monkeypatch):
    import academic_agent.research_artifacts as artifacts

    store, snapshot = research
    original = artifacts.assert_research_current
    calls = 0

    def mutate(store, snapshot):
        nonlocal calls
        original(store, snapshot)
        calls += 1
        if calls == 2:
            store.screen(snapshot.work_ids[0], "exclude", "Changed before lock")

    monkeypatch.setattr(artifacts, "assert_research_current", mutate)
    with pytest.raises(ValueError, match="changed"):
        publish_research(store, "challenge", snapshot, {"query.json": b"{}"})
    assert not (store.root / "analyses" / "challenge" / "latest.json").exists()


def test_source_mutation_after_version_rename_does_not_replace_latest(research, monkeypatch):
    from pathlib import Path

    store, snapshot = research
    first = publish_research(store, "innovation", snapshot, {"report.md": b"first"})
    pointer = (first.parent / "latest.json").read_bytes()
    original = Path.rename

    def mutate(path, destination):
        result = original(path, destination)
        if path.name.startswith(".pending-"):
            document = next(iter(snapshot.documents.values()))
            (store.root / document.source_path).write_bytes(b"changed source")
        return result

    monkeypatch.setattr(Path, "rename", mutate)
    with pytest.raises(ValueError, match="hash mismatch"):
        publish_research(store, "innovation", snapshot, {"report.md": b"second"})
    assert (first.parent / "latest.json").read_bytes() == pointer
    assert load_research(first)["files"]["report.md"] == b"first"


def test_historical_human_edits_are_not_repaired(research):
    store, snapshot = research
    first = publish_research(store, "innovation", snapshot, {"report.md": b"first"})
    second = publish_research(store, "innovation", snapshot, {"report.md": b"second"})
    (first / "report.md").write_bytes(b"human edit")
    pointer = (second.parent / "latest.json").read_bytes()
    with pytest.raises(ValueError, match="hash mismatch"):
        publish_research(store, "innovation", snapshot, {"report.md": b"third"})
    assert (first / "report.md").read_bytes() == b"human edit"
    assert (second.parent / "latest.json").read_bytes() == pointer


@pytest.mark.parametrize("target", ["payload", "version", "kind", "latest"])
def test_symlinks_rejected(research, target):
    store, snapshot = research
    directory = publish_research(store, "innovation", snapshot, {"report.md": b"original"})
    path = {"payload": directory / "report.md", "version": directory,
            "kind": directory.parent, "latest": directory.parent / "latest.json"}[target]
    destination = store.root / ("original-" + target)
    is_dir = path.is_dir()
    path.rename(destination)
    try:
        path.symlink_to(destination, target_is_directory=is_dir)
    except OSError as exc:
        destination.rename(path)
        pytest.skip(f"Host lacks symlink permission: {exc}")
    with pytest.raises(ValueError, match="symlink"):
        load_research(directory.parent)
    with pytest.raises(ValueError, match="symlink"):
        publish_research(store, "innovation", snapshot, {"report.md": b"new"})


def test_invalid_current_manual_review_preserves_latest(research):
    store, snapshot = research
    first = publish_research(store, "innovation", snapshot, {"report.md": b"first"})
    pointer = (first.parent / "latest.json").read_bytes()
    versions = {path.name for path in first.parent.iterdir()}
    entry = snapshot.decisions[snapshot.supported_claim_ids[0]].model_dump(exclude={"type"})
    entry["claim_id"] = "unknown_claim"
    store.log("manual_review", entry)
    with pytest.raises(ValueError, match="manual"):
        publish_research(store, "innovation", snapshot, {"report.md": b"second"})
    assert (first.parent / "latest.json").read_bytes() == pointer
    assert {path.name for path in first.parent.iterdir()} == versions
    assert load_research(first.parent)["files"]["report.md"] == b"first"


def test_domain_and_writing_payload_sets_are_exact(research):
    store, snapshot = research
    domain = _domain_payloads()
    directory = publish_research(store, "domain", snapshot, domain)
    assert load_research(directory)["files"] == domain
    writing = _writing_payloads()
    assert load_research(publish_research(store, "writing", snapshot, writing))["files"] == writing


@pytest.mark.parametrize("kind,case", [
    ("domain", "missing"),
    ("domain", "extra"),
    ("domain", "foreign"),
    ("writing", "missing"),
    ("writing", "extra"),
    ("writing", "foreign"),
])
def test_new_artifact_kinds_reject_missing_extra_and_foreign_payload_sets(research, kind, case):
    store, snapshot = research
    valid = _domain_payloads() if kind == "domain" else _writing_payloads()
    prior = publish_research(store, kind, snapshot, valid)
    pointer = (prior.parent / "latest.json").read_bytes()
    if case == "missing":
        payloads = {name: value for name, value in valid.items() if name != next(iter(valid))}
    elif case == "extra":
        payloads = {**valid, "extra.json": b"{}"}
    else:
        payloads = _writing_payloads() if kind == "domain" else _domain_payloads()
    with pytest.raises(ValueError, match="payload"):
        publish_research(store, kind, snapshot, payloads)
    assert (prior.parent / "latest.json").read_bytes() == pointer
    assert load_research(prior.parent)["files"] == valid


@pytest.mark.parametrize("failure_at", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("status", ["stale", "failed"])
def test_publication_dependency_callback_fences_and_reports_late_output(research, failure_at, status):
    store, snapshot = research
    prior = publish_research(store, "domain", snapshot, _domain_payloads(b'{"version":"prior"}'))
    pointer = (prior.parent / "latest.json").read_bytes()
    versions = {path.name for path in prior.parent.glob("v-*")}
    calls = []

    def check():
        calls.append(len(calls) + 1)
        if len(calls) == failure_at:
            raise WritingError("dependency changed", status=status)

    with pytest.raises(WritingError) as caught:
        publish_research(store, "domain", snapshot, _domain_payloads(b'{"version":"attempt"}'),
                         dependency_check=check)
    assert calls == list(range(1, failure_at + 1))
    assert caught.value.status == status
    assert (caught.value.output is not None) == (failure_at >= 4)
    if caught.value.output is not None:
        assert caught.value.output.is_dir()
        assert load_research(caught.value.output)["files"]["context.json"] == b'{"version":"attempt"}'
    expected_versions = versions | ({caught.value.output.name} if caught.value.output else set())
    assert {path.name for path in prior.parent.glob("v-*")} == expected_versions
    if failure_at <= 4:
        assert (prior.parent / "latest.json").read_bytes() == pointer
        assert load_research(prior.parent)["files"]["context.json"] == b'{"version":"prior"}'
    else:
        assert (prior.parent / "latest.json").read_bytes() != pointer
        assert load_research(prior.parent)["files"]["context.json"] == b'{"version":"attempt"}'


def test_failure_after_latest_does_not_roll_back_a_later_writer_pointer(research):
    store, snapshot = research
    publish_research(store, "domain", snapshot, _domain_payloads(b'{"version":"prior"}'))
    later = publish_research(store, "domain", snapshot, _domain_payloads(b'{"version":"later"}'))
    later_manifest = load_research(later)["manifest"]
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 5:
            atomic_json(later.parent / "latest.json", {
                "manifest_hash": later_manifest["manifest_hash"],
                "schema_version": 1,
                "version": later.name,
            })
            raise WritingError("dependency became invalid", status="failed")

    with pytest.raises(WritingError) as caught:
        publish_research(store, "domain", snapshot, _domain_payloads(b'{"version":"attempt"}'),
                         dependency_check=check)
    assert caught.value.status == "failed"
    assert caught.value.output is not None and caught.value.output.is_dir()
    assert load_research(caught.value.output)["files"]["context.json"] == b'{"version":"attempt"}'
    assert load_research(later.parent)["files"]["context.json"] == b'{"version":"later"}'


def test_dependency_callback_observes_sqlite_lock_but_external_fence_is_explicit(research):
    store, snapshot = research
    calls = 0
    locked = []

    def check():
        nonlocal calls
        calls += 1
        if calls >= 3:
            connection = sqlite3.connect(store.root / "research.db", timeout=0)
            try:
                connection.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError:
                locked.append(calls)
            finally:
                connection.close()

    publish_research(store, "domain", snapshot, {
        "context.json": b"{}", "domain.json": b"{}", "domain-review.json": b"{}",
        "comparison.md": b"", "references.bib": b"", "evidence.json": b"{}",
    }, dependency_check=check)
    assert calls == 5
    assert locked == [3, 4, 5]
