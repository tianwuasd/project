import pytest
from pypdf import PdfWriter

from academic_agent.documents import abstract_document, make_evidence, parse_document, verify_span
from academic_agent.models import Work
from academic_agent.store import Store


def test_unicode_offsets_exact_and_tamper_rejected(tmp_path):
    path = tmp_path / "paper.md"
    path.write_text("测量值未提高。\n" * 400, encoding="utf-8")
    doc = parse_document(path, "work")
    spans = make_evidence(doc)
    assert len(spans) > 1
    assert "".join(span.text for span in spans) == doc.pages[0].text
    assert all(verify_span(span, doc) for span in spans)
    assert not verify_span(spans[0].model_copy(update={"text": "测量值提高。"}), doc)
    assert not verify_span(spans[0].model_copy(update={"page": 99}), doc)


def test_scanned_and_encrypted_pdf(tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    path = tmp_path / "scan.pdf"
    writer.write(path)
    doc = parse_document(path, "work")
    assert doc.level == "needs_ocr" and doc.diagnostics
    assert make_evidence(doc) == []
    writer.encrypt("private")
    writer.write(path)
    with pytest.raises(ValueError, match="Encrypted"):
        parse_document(path, "work")


def test_abstract_is_never_full_text():
    work = Work(id="work", title="Study", abstract="Evidence in abstract only.")
    doc = abstract_document(work)
    assert doc.level == "abstract_only" and doc.pages[0].number == 0
    assert abstract_document(Work(id="work", title="No abstract")).level == "metadata_only"


def test_attached_source_is_snapshotted_and_versioned(tmp_path):
    source = tmp_path / "note.txt"
    source.write_text("First version.", encoding="utf-8")
    store = Store(tmp_path / "project")
    work = store.upsert_work(Work(title="Study"))
    old = store.add_document(parse_document(source, work.id))
    source.write_text("Second version.", encoding="utf-8")
    latest = store.add_document(parse_document(source, work.id))
    assert old.id != latest.id
    assert store.document(old.id).pages[0].text == "First version."
    assert (store.root / old.source_path).read_text() == "First version."


def test_changed_file_after_parse_is_rejected(tmp_path):
    source = tmp_path / "paper.txt"
    source.write_text("Before", encoding="utf-8")
    store = Store(tmp_path / "project")
    work = store.upsert_work(Work(title="Study"))
    document = parse_document(source, work.id)
    source.write_text("After", encoding="utf-8")
    with pytest.raises(ValueError, match="Source changed"):
        store.add_document(document)
