import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / 'skills/medical-ai-research/scripts'
sys.path.insert(0, str(SCRIPTS))
try:
    import corpus
except ImportError:
    corpus = None


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(corpus, '论文索引模块尚未实现')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self):
        import fitz
        folder = self.root / '论文'
        folder.mkdir()
        doc = fitz.open()
        doc.new_page().insert_text((40, 70), 'Cardiac segmentation research evidence.')
        doc.new_page().insert_text((40, 70), 'Patient split and external hospital testing.')
        doc.save(folder / 'heart.pdf')
        doc.close()
        (folder / 'copy.pdf').write_bytes((folder / 'heart.pdf').read_bytes())
        (folder / 'heart_精读.md').write_text('二手笔记：心脏分割', encoding='utf-8')
        (folder / 'meta.json').write_text(json.dumps({'papers': [
            {'title': 'Unknown evidence', 'abstract': 'Only an abstract, no full text.'}]}), encoding='utf-8')
        config = self.root / 'sources.json'
        config.write_text(json.dumps({'sources': [
            {'type': 'papers', 'path': str(folder), 'notes': True},
            {'type': 'records', 'path': str(folder / 'meta.json')}]}), encoding='utf-8')
        return config

    def test_dedup_preserves_locations_and_page_evidence(self):
        db = self.root / 'corpus.sqlite'
        report = corpus.build(self.fixture(), db)
        self.assertEqual(report['documents'], 3)
        self.assertEqual(report['sources'], 4)
        results = corpus.search(db, 'hospital', 5)
        self.assertEqual(results[0]['page'], 2)
        page = corpus.read_page(db, results[0]['document_id'], 2)
        self.assertIn('external hospital', page['text'])
        self.assertEqual(page['kind'], 'pdf_text')

    def test_abstract_is_not_fulltext_and_sql_characters_are_safe(self):
        db = self.root / 'corpus.sqlite'
        corpus.build(self.fixture(), db)
        hit = corpus.search(db, 'Only an abstract', 5)[0]
        self.assertEqual(hit['kind'], 'abstract_only')
        self.assertIsNone(hit['page'])
        self.assertEqual(corpus.search(db, "'; DROP TABLE documents; --", 5), [])
        self.assertEqual(corpus.stats(db)['documents'], 3)

    def test_failed_build_preserves_previous_database(self):
        db = self.root / 'corpus.sqlite'
        corpus.build(self.fixture(), db)
        before = db.read_bytes()
        bad = self.root / 'bad.json'
        bad.write_text(json.dumps({'sources': [{'type': 'papers', 'path': str(self.root / 'absent')}]}))
        with self.assertRaises((ValueError, FileNotFoundError)):
            corpus.build(bad, db)
        self.assertEqual(db.read_bytes(), before)

    def test_build_rejects_source_or_config_as_output(self):
        config = self.fixture()
        source = self.root / '论文/heart_精读.md'
        for target in [config, source, self.root / '论文/new.sqlite']:
            before = target.read_bytes() if target.exists() else None
            with self.assertRaises(ValueError):
                corpus.build(config, target)
            self.assertEqual(target.read_bytes() if target.exists() else None, before)

    def test_build_does_not_replace_unrelated_existing_file(self):
        config = self.fixture()
        target = self.root / 'important.txt'
        target.write_text('Existing user document', encoding='utf-8')
        with self.assertRaises(ValueError):
            corpus.build(config, target)
        self.assertEqual(target.read_text(encoding='utf-8'), 'Existing user document')


if __name__ == '__main__':
    unittest.main()
