"""Independent arithmetic, provenance and adversarial artifact acceptance."""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from academic_agent import trends
from academic_agent.config import atomic_json, create_project
from academic_agent.documents import abstract_document
from academic_agent.models import ProjectConfig, PublicationCheck, Work
from academic_agent.store import Store
from academic_agent.trends import (
    CorpusAnnotations,
    CorpusSnapshot,
    build_trends,
    compute_trends,
    freeze_corpus,
    load_verified_trends,
    verify_trends,
)


def project(path):
    create_project(path, ProjectConfig(title="中文趋势", sources=["crossref"], queries=["test"], backend="demo"))
    return Store(path)


def add(store, title, year=None, **kwargs):
    return store.upsert_work(Work(title=title, year=year, screening="include", **kwargs))


@pytest.mark.skipif(os.name != "nt", reason="Windows MAX_PATH regression")
def test_verify_trends_cli_accepts_ordinary_directory_with_exact_max_path_child(tmp_path):
    store = project(tmp_path)
    add(store, "Long path fixture", 2024)
    published = build_trends(store)
    parent = published.parent.resolve()
    fixed_length = len(str(parent)) + 1 + len("\\computation.json")
    component_length = 260 - fixed_length
    assert 1 <= component_length <= 255
    destination = parent / ("v" + "x" * (component_length - 1))
    published.rename(destination)
    assert len(str((destination / "computation.json").resolve())) == 260

    completed = subprocess.run(
        [sys.executable, "-m", "academic_agent.cli", "verify-trends", str(destination)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert json.loads(completed.stdout)["valid"] is True


@pytest.fixture
def corpus(tmp_path):
    store = project(tmp_path)
    works = [add(store, f"Included {i}", year) for i, year in enumerate([2022, 2022, 2023, None, 2025])]
    store.upsert_work(Work(title="Pending", year=2022))
    store.upsert_work(Work(title="Excluded", year=2022, screening="exclude"))
    add(store, "Retracted", 2022, publication_checks=[PublicationCheck(source="fixture", status="retracted")])
    annotations = CorpusAnnotations(version="1", annotator="研究者", works={
        works[0].id: {"labels": [" 图模型 ", "图模型", "优化"], "rationale": "方法章节"},
        works[1].id: {"labels": ["图模型"], "rationale": "方法章节"},
        works[3].id: {"labels": ["优化"], "rationale": "方法章节"},
    })
    return store, works, annotations


def test_independent_arithmetic_and_json_roundtrip(corpus):
    store, works, annotations = corpus
    snapshot = freeze_corpus(store, annotations=annotations, as_of_year=2024)
    result = compute_trends(snapshot)
    assert compute_trends(CorpusSnapshot.model_validate_json(snapshot.model_dump_json())) == result
    assert {row.year: row.count for row in result.annual} == {2022: 2, 2023: 1}
    assert all(row.denominator == 3 for row in result.annual)
    assert result.annual[0].share == pytest.approx(2 / 3)
    assert result.counts == {
        "candidates": 8, "included": 6, "excluded": 1, "pending": 1, "retracted": 1,
        "eligible": 5, "missing_year": 1, "invalid_year": 0, "future_year": 1,
        "valid_year": 3, "labelled": 3, "unlabelled": 2, "related": 0,
        "arxiv_identified": 0,
    }
    labels = {row.label: row for row in result.labels}
    assert labels["图模型"].count == labels["优化"].count == 2
    assert labels["优化"].denominator == 5
    assert labels["优化"].share == 0.4
    assert labels["图模型"].annual[0].count == 2
    assert labels["图模型"].annual[0].denominator == 2
    assert result.work_ids["eligible"] == sorted(w.id for w in works)
    assert result.work_ids["missing_year"] == [works[3].id]
    expected = hashlib.sha256(json.dumps(snapshot.model_dump(mode="json"), sort_keys=True,
                                        separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    assert result.input_hash == expected
    assert "descriptive_only" in {d.code for d in result.diagnostics}


@pytest.mark.parametrize("change", [
    {"version": " "}, {"annotator": " "}, {"surprise": True},
    {"works": {"w": {"year": True, "year_source": "source"}}},
    {"works": {"w": {"year": "2023", "year_source": "source"}}},
    {"works": {"w": {"year": 2023.0, "year_source": "source"}}},
    {"works": {"w": {"year": 999, "year_source": "source"}}},
    {"works": {"w": {"year": 10000, "year_source": "source"}}},
    {"works": {"w": {"year": 2023, "year_source": " "}}},
    {"works": {"w": {"labels": [" "]}}},
    {"works": {"w": {"labels": ["A"], "rationale": " "}}},
    {"works": {"w": {"labels": [3], "rationale": "source"}}},
])
def test_strict_annotations(change):
    with pytest.raises(ValidationError):
        CorpusAnnotations.model_validate({"version": "1", "annotator": "R", **change})


def test_unknown_ids_and_mutated_annotations_rejected(corpus):
    store, _, annotations = corpus
    with pytest.raises(ValueError, match="Unknown.*ID"):
        freeze_corpus(store, annotations=CorpusAnnotations(version="1", annotator="R", works={"unknown": {}}))
    annotations.works["unknown"] = next(iter(annotations.works.values()))
    with pytest.raises(ValueError):
        freeze_corpus(store, annotations=annotations)


@pytest.mark.parametrize("basis", ["first_publication_year", "formal_publication_year"])
def test_alternate_basis_never_falls_back(corpus, basis):
    store, works, _ = corpus
    annotations = CorpusAnnotations(version="2", annotator="R", works={
        works[0].id: {"year": 2020, "year_source": "Published date on paper"}})
    result = compute_trends(freeze_corpus(store, annotations=annotations, date_basis=basis, as_of_year=2024))
    assert [(row.year, row.count) for row in result.annual] == [(2020, 1)]
    assert result.counts["missing_year"] == 4
    assert result.parameters.date_basis == basis
    record_result = compute_trends(freeze_corpus(store, annotations=annotations, as_of_year=2024))
    assert record_result.annual[0].year == 2022


@pytest.mark.parametrize("year", [True, False, "2022", 2022.5, 999, 10000, 10**30])
def test_invalid_metadata_dates_never_coerced_or_expanded(tmp_path, year):
    store = project(tmp_path)
    work = add(store, "Bad metadata", 2022)
    with store.connect() as connection:
        data = work.model_dump()
        data["year"] = year
        connection.execute("UPDATE works SET data=? WHERE id=?", (json.dumps(data), work.id))
    snapshot = freeze_corpus(store, as_of_year=2024)
    assert snapshot.works[0].year == year
    result = compute_trends(snapshot)
    assert result.counts["invalid_year"] == 1
    assert result.annual == []


@pytest.mark.parametrize("year", [True, "2024", 999, 10000])
def test_invalid_cutoff_rejected(tmp_path, year):
    with pytest.raises(ValueError):
        freeze_corpus(project(tmp_path), as_of_year=year)


def test_empty_current_year_and_gaps(tmp_path):
    store = project(tmp_path)
    empty = compute_trends(freeze_corpus(store, as_of_year=2024))
    assert empty.annual == empty.labels == []
    assert empty.arxiv_share is None
    add(store, "First", 2022)
    add(store, "Current", 2024)
    result = compute_trends(freeze_corpus(store, as_of_year=2024))
    assert [(r.year, r.count, r.current_year) for r in result.annual] == [
        (2022, 1, False), (2023, 0, False), (2024, 1, True)]
    assert freeze_corpus(store).as_of_year == datetime.now(timezone.utc).year
    with pytest.raises(ValueError):
        freeze_corpus(store, date_basis="invented")


def test_provenance_documents_relations_and_retrieval_limits(corpus):
    store, works, _ = corpus
    work = store.work(works[0].id)
    work.arxiv_id = "2201.12345"
    work.abstract = "Original fixture abstract"
    work.relations = [{"type": "preprint_of", "id": works[1].id}]
    work.provenance = [{"source": "local", "detail": "unverified"}]
    store.update_work(work)
    document = store.add_document(abstract_document(work))
    store.log("search", {"query_key": "key", "summary": {"query": "bounded search", "retained": 2,
        "dropped_at_project_cap": 7, "sources": [{"source": "crossref", "status": "partial", "records": 2,
        "truncated": True, "errors": ["HTTP 429"]}]}, "responses": [{"source": "crossref",
        "status": "partial", "queries": ["bounded search"], "total": 999, "truncated": True,
        "errors": ["HTTP 429"], "retrieved_at": "2024-02-01T00:00:00+00:00", "records": [], "raw_responses": []}]})
    snapshot = freeze_corpus(store, as_of_year=2024)
    assert snapshot.documents[0].id == document.id
    assert snapshot.documents[0].content_hash == document.content_hash
    assert snapshot.works[[w.id for w in snapshot.works].index(work.id)].record_revision == work.record_revision
    assert snapshot.search_log[0].responses[0].retrieved_at.startswith("2024-02-01")
    result = compute_trends(snapshot)
    assert result.counts["related"] == 1
    assert result.arxiv_share == 0.2
    assert result.document_levels["abstract_only"] == 1
    codes = {d.code for d in result.diagnostics}
    assert {"search_incomplete", "project_cap", "related_records"} <= codes
    directory = build_trends(store)
    report = (directory / "report.md").read_text(encoding="utf-8")
    assert "HTTP 429" in report and "999" in report and "bounded search" in report
    assert "arXiv 标识覆盖" in report and "未执行人工评测" in report


def test_deterministic_repeated_freeze_and_old_edits_preserved(corpus):
    store, _, annotations = corpus
    a = freeze_corpus(store, annotations=annotations)
    b = freeze_corpus(store, annotations=annotations)
    assert a == b
    first = build_trends(store, annotations=annotations)
    assert verify_trends(first)["valid"] is True
    (first / "report.md").write_text("Human draft", encoding="utf-8")
    second = build_trends(store, annotations=annotations)
    assert second != first
    assert (first / "report.md").read_text(encoding="utf-8") == "Human draft"
    assert (first / "computation.json").read_bytes() == (second / "computation.json").read_bytes()
    assert json.loads((second.parent / "latest.json").read_text(encoding="utf-8"))["version"] == second.name
    assert "local_coverage_unknown" in {d.code for d in compute_trends(a).diagnostics}


@pytest.mark.parametrize("filename", ["corpus.json", "computation.json", "annual.csv", "labels.csv", "report.md", "manifest.json"])
@pytest.mark.parametrize("operation", ["corrupt", "missing"])
def test_all_artifacts_verified(corpus, filename, operation):
    directory = build_trends(corpus[0], annotations=corpus[2])
    path = directory / filename
    if operation == "missing":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError):
        verify_trends(directory)


def reseal_manifest(directory, update_files=True):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if update_files:
        manifest["files"] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                             for name in manifest["files"]}
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                                                        separators=(",", ":")).encode("utf-8")).hexdigest()
    atomic_json(path, manifest)


@pytest.mark.parametrize("filename", ["corpus.json", "computation.json", "annual.csv", "labels.csv", "report.md"])
def test_resealed_outputs_must_recompute(corpus, filename):
    directory = build_trends(corpus[0])
    path = directory / filename
    if filename == "corpus.json":
        value = json.loads(path.read_text(encoding="utf-8"))
        value["works"][0]["title"] += "tamper"
        atomic_json(path, value)
    elif filename == "computation.json":
        value = json.loads(path.read_text(encoding="utf-8"))
        value["counts"]["eligible"] += 1
        atomic_json(path, value)
    else:
        path.write_bytes(path.read_bytes() + b"tamper")
    reseal_manifest(directory)
    with pytest.raises(ValueError):
        verify_trends(directory)


@pytest.mark.parametrize("name", ["../outside", "C:/outside", "..\\outside", "extra.json"])
def test_manifest_never_controls_read_paths(corpus, name):
    directory = build_trends(corpus[0])
    path = directory / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["files"][name] = "0" * 64
    atomic_json(path, value)
    reseal_manifest(directory, update_files=False)
    with pytest.raises(ValueError, match="files"):
        verify_trends(directory)


def test_verify_needs_no_store(corpus, monkeypatch):
    directory = build_trends(corpus[0])
    def forbidden(*args, **kwargs):
        raise AssertionError("Verifier consulted mutable database")
    monkeypatch.setattr(trends, "Store", forbidden)
    monkeypatch.setattr(trends, "load_config", forbidden)
    assert verify_trends(directory)["valid"]


def test_verified_trends_returns_the_exact_validated_bytes_and_legacy_summary(corpus):
    directory = build_trends(corpus[0])
    verified = load_verified_trends(directory)
    assert set(verified.files) == {
        "manifest.json", "corpus.json", "computation.json", "annual.csv", "labels.csv", "report.md",
    }
    assert all(verified.files[name] == (directory / name).read_bytes() for name in verified.files)
    assert verified.snapshot == CorpusSnapshot.model_validate_json(verified.files["corpus.json"])
    assert verified.computation == compute_trends(verified.snapshot)
    assert set(verified.summary) == {"valid", "algorithm_version", "input_hash", "snapshot", "files_verified"}
    assert verified.summary["files_verified"] == 6
    assert verify_trends(directory) == verified.summary


def test_verified_trends_uses_captured_bytes_when_file_changes_after_read(corpus, monkeypatch):
    directory = build_trends(corpus[0])
    original = type(directory).read_bytes
    swapped = False

    def swap_after_capture(path):
        nonlocal swapped
        data = original(path)
        if path.name == "computation.json" and not swapped:
            swapped = True
            path.write_bytes(data + b"tamper after captured read")
        return data

    monkeypatch.setattr(type(directory), "read_bytes", swap_after_capture)
    verified = load_verified_trends(directory)
    assert verified.computation == compute_trends(verified.snapshot)
    assert verified.files["computation.json"] != (directory / "computation.json").read_bytes()
    with pytest.raises(ValueError, match="hash"):
        load_verified_trends(directory)


def test_freeze_race_rejected(corpus, monkeypatch):
    store = corpus[0]
    original = store.logs
    def mutate(*args, **kwargs):
        value = original(*args, **kwargs)
        add(store, "Concurrent", 2024)
        return value
    monkeypatch.setattr(store, "logs", mutate)
    with pytest.raises(ValueError, match="changed"):
        freeze_corpus(store)


@pytest.mark.parametrize("mutation", ["work", "config", "log", "document"])
def test_publish_race_preserves_previous(corpus, monkeypatch, mutation):
    store, works, _ = corpus
    previous = build_trends(store)
    pointer = (previous.parent / "latest.json").read_bytes()
    original = trends.compute_trends
    def mutate(snapshot):
        result = original(snapshot)
        if mutation == "work":
            add(store, "Concurrent", 2024)
        elif mutation == "config":
            config = json.loads((store.root / "project.json").read_text(encoding="utf-8"))
            config["max_candidates"] += 1
            atomic_json(store.root / "project.json", config)
        elif mutation == "log":
            store.log("manual", {"changed": True})
        else:
            store.add_document(abstract_document(works[0]))
        return result
    monkeypatch.setattr(trends, "compute_trends", mutate)
    with pytest.raises(ValueError, match="changed"):
        build_trends(store)
    assert (previous.parent / "latest.json").read_bytes() == pointer
    assert previous.exists()


def test_strict_snapshot_duplicate_ids_and_extra_fields(corpus):
    snapshot = freeze_corpus(corpus[0])
    for mutate in [lambda d: d.update(extra=True), lambda d: d["works"].append(d["works"][0]),
                   lambda d: d.update(as_of_year=True)]:
        value = snapshot.model_dump(mode="json")
        mutate(value)
        with pytest.raises(ValueError):
            CorpusSnapshot.model_validate(value)


def test_multilabel_can_exceed_one_and_definitions_are_fixed(tmp_path):
    store = project(tmp_path)
    work = add(store, "Methods", 2023)
    annotations = CorpusAnnotations(version="1", annotator="R", label_definitions={
        "A": "Defined A", "B": "Defined B", "Unused": "Prespecified but unused"}, works={
        work.id: {"labels": ["A", "B", "B"], "rationale": "Methods"}})
    result = compute_trends(freeze_corpus(store, annotations=annotations, as_of_year=2024))
    assert sum(label.share for label in result.labels) == 2.0
    assert [(label.label, label.count) for label in result.labels] == [("A", 1), ("B", 1), ("Unused", 0)]
    assert result.labels[0].definition == "Defined A"
    with pytest.raises(ValueError, match="definitions"):
        CorpusAnnotations(version="1", annotator="R", label_definitions={"B": "Only B"},
                          works={work.id: {"labels": ["A"], "rationale": "Methods"}})


@pytest.mark.parametrize("counts,small", [([10, 40], False), ([9, 41], True), ([10, 39], True)])
def test_small_sample_thresholds_are_exact(tmp_path, counts, small):
    store = project(tmp_path)
    for year, count in zip([2022, 2023], counts):
        for number in range(count):
            add(store, f"Year {year} record {number}", year)
    result = compute_trends(freeze_corpus(store, as_of_year=2024))
    assert ("small_sample" in {d.code for d in result.diagnostics}) is small
    assert "descriptive_only" in {d.code for d in result.diagnostics}


def test_gap_year_label_denominator_zero_is_null(tmp_path):
    store = project(tmp_path)
    work = add(store, "2021", 2021)
    add(store, "2023", 2023)
    annotations = CorpusAnnotations(version="1", annotator="R", works={
        work.id: {"labels": ["A"], "rationale": "Methods"}})
    result = compute_trends(freeze_corpus(store, annotations=annotations, as_of_year=2024))
    assert result.labels[0].annual[1].year == 2022
    assert result.labels[0].annual[1].denominator == 0
    assert result.labels[0].annual[1].share is None


def test_empty_annotation_and_snapshot_models_are_strict(corpus):
    store, _, _ = corpus
    result = compute_trends(freeze_corpus(store, annotations=CorpusAnnotations(version="1", annotator="R")))
    assert result.counts["labelled"] == 0
    assert result.counts["unlabelled"] == 5
    snapshot = freeze_corpus(store).model_dump(mode="json")
    snapshot["works"][0]["record_revision"] = "1"
    with pytest.raises(ValueError):
        CorpusSnapshot.model_validate_json(json.dumps(snapshot))
    with pytest.raises(ValueError):
        CorpusAnnotations(version="1", annotator="R", schema_version=True)


def test_no_proxy_credentials_exported_and_untrusted_text_escaped(tmp_path):
    store = project(tmp_path)
    work = add(store, "<script>alert(1)</script>", 2023)
    config = json.loads((store.root / "project.json").read_text(encoding="utf-8"))
    config["proxy_url"] = "http://fake_user:SECRET_PROXY_PASSWORD@127.0.0.1:7897"
    config["title"] = "<script>malicious</script>"
    atomic_json(store.root / "project.json", config)
    annotations = CorpusAnnotations(version="1", annotator="R", works={work.id: {
        "labels": ["=HYPERLINK(\"bad\")", "<script>"], "rationale": "Explicit labels"}})
    directory = build_trends(store, annotations=annotations)
    for path in directory.iterdir():
        assert b"SECRET_PROXY_PASSWORD" not in path.read_bytes()
    assert "<script>" not in (directory / "report.md").read_text(encoding="utf-8")
    assert "'=HYPERLINK" in (directory / "labels.csv").read_text(encoding="utf-8")


def test_synthetic_source_is_explicitly_marked(tmp_path):
    store = project(tmp_path)
    add(store, "Original fixture", 2023, source_ids={"synthetic": "1"})
    result = compute_trends(freeze_corpus(store))
    assert "synthetic_demo" in {d.code for d in result.diagnostics}


@pytest.mark.parametrize("field,value", [("algorithm_version", "future-unsupported"), ("schema_version", True),
                                        ("input_hash", "0" * 64), ("source_snapshot", [0, 0, 0, "0" * 64])])
def test_manifest_contract_and_binding(corpus, field, value):
    directory = build_trends(corpus[0])
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    atomic_json(path, manifest)
    reseal_manifest(directory)
    with pytest.raises(ValueError):
        verify_trends(directory)


def test_snapshot_token_is_checked_inside_immediate_transaction(corpus, monkeypatch):
    original = trends.snapshot_token
    checked = []
    def inspect(store, connection=None):
        if connection is not None:
            checked.append(connection.in_transaction)
        return original(store, connection)
    monkeypatch.setattr(trends, "snapshot_token", inspect)
    directory = build_trends(corpus[0])
    assert checked == [True]
    assert verify_trends(directory)["snapshot"] == json.loads(
        (directory.parent / "latest.json").read_text(encoding="utf-8"))["snapshot"]


def test_mixed_local_import_retains_unknown_coverage(tmp_path):
    store = project(tmp_path)
    add(store, "Local", 2023, provenance=[{"source": "local_import", "file": "local.ris"}])
    store.log("search", {"summary": {"query": "online"}, "responses": [{
        "source": "crossref", "status": "ok", "truncated": False, "records": [], "errors": [],
        "retrieved_at": "2024-01-01T00:00:00Z"}]})
    result = compute_trends(freeze_corpus(store))
    assert "local_coverage_unknown" in {d.code for d in result.diagnostics}


def test_mutation_cannot_bypass_annotation_year_validation(corpus):
    snapshot = freeze_corpus(corpus[0], annotations=corpus[2])
    snapshot.annotations.works[next(iter(snapshot.annotations.works))].labels.append(" ")
    with pytest.raises(ValueError):
        compute_trends(snapshot)


def test_tampered_active_document_blocks_freeze(corpus):
    store, works, _ = corpus
    document = store.add_document(abstract_document(works[0]))
    with store.connect() as connection:
        path = connection.execute("SELECT path FROM documents WHERE id=?", (document.id,)).fetchone()[0]
    (store.root / path).write_bytes(b"tampered document")
    with pytest.raises(ValueError, match="hash"):
        freeze_corpus(store)


def test_report_retrieval_and_coverage_have_readable_chinese_fields(tmp_path):
    store = project(tmp_path)
    add(store, "Returned record", 2023)
    store.log("search", {"summary": {"query": "bounded search", "retained": 1,
        "dropped_at_project_cap": 7, "sources": [{"source": "crossref", "status": "partial", "records": 18,
        "truncated": True, "errors": ["HTTP 429"]}]}, "responses": [{"source": "crossref",
        "status": "partial", "queries": ["bounded search"], "total": 2037921, "truncated": True,
        "errors": ["HTTP 429"], "retrieved_at": "2024-02-01T00:00:00Z", "records": [{}] * 18}]})
    directory = build_trends(store)
    report = (directory / "report.md").read_text(encoding="utf-8")
    assert "| 文档层级 | 数量 |" in report
    assert "| 仅元数据 | 1 |" in report
    assert "| 需 OCR（文字识别） | 0 |" in report
    assert "### 检索范围表" in report
    assert "项目保留记录数：1；因项目候选上限丢弃：7。" in report
    assert "| 来源 | 实际检索式 | 状态 | 检索时间 | 来源报告总量 | 返回记录数 | 截断 | 已记录错误 |" in report
    assert "| Crossref | bounded search | 部分返回 | 2024-02-01T00:00:00Z | 2037921 | 18 | 是 | HTTP 429 |" in report
    assert "| 来源 | 摘要状态 | 摘要返回记录数 | 摘要截断 | 摘要已记录错误 |" in report
    assert "| Crossref | 部分返回 | 18 | 是 | HTTP 429 |" in report
    assert "- 年度占比：该年合格记录数 ÷ 所有合法且不晚于截止年的合格记录数。" in report
    assert "metadata_only" not in report and "annual_count" not in report
    assert '"truncated"' not in report and '"retrieved_at"' not in report
    assert verify_trends(directory)["valid"]


def test_report_unknown_source_fields_and_blank_project_have_explicit_fallbacks(tmp_path):
    store = project(tmp_path)
    config = json.loads((store.root / "project.json").read_text(encoding="utf-8"))
    config.update(question="  ", sources=[], queries=[" "])
    atomic_json(store.root / "project.json", config)
    store.log("search", {"summary": {}, "responses": [{"source": "", "records": []}]})
    report = (build_trends(store) / "report.md").read_text(encoding="utf-8")
    assert "研究问题：未填写。" in report
    assert "配置来源：未配置。" in report
    assert "配置检索式（不保证已执行）：未配置。" in report
    assert "执行检索式：未知（未记录）。" in report
    assert "| 未知（未记录） | 未知（未记录） | 未知 | 未知（未记录） | 未知 | 0 | 未知 | 无已记录错误 |" in report
    assert "研究问题：。" not in report


def test_report_summary_only_and_unknown_status_preserve_escaped_values(tmp_path):
    store = project(tmp_path)
    store.log("search", {"summary": {"query": "<script>|query", "sources": [{"source": "<source>|",
        "status": "custom|state", "records": 2, "errors": ["<error>|line\nnext"]}]}, "responses": []})
    report = (build_trends(store) / "report.md").read_text(encoding="utf-8")
    assert "执行检索式：&lt;script&gt;\\|query。" in report
    assert "未记录来源响应；来源报告总量、检索时间及实际检索式未知。" in report
    assert "| &lt;source&gt;\\| | 未知（原值：custom\\|state） | 2 | 未知 | &lt;error&gt;\\|line next |" in report
    assert "<script>" not in report and "<error>" not in report
