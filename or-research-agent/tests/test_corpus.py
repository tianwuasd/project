import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'skill' / 'scripts'))
from corpus import build, search, evidence, parse_wos, winpath, doi, excerpt


class CorpusTests(unittest.TestCase):
    def test_markdown_and_html_doi_do_not_split_papers(self):
        self.assertEqual(doi('10.1016/j.trb.2026.103449](https://doi.org/10.1016/j.trb.2026.103449)'), '10.1016/j.trb.2026.103449')
        self.assertEqual(doi('10.1016/j.tre.2026.105010"'), '10.1016/j.tre.2026.105010')
        self.assertEqual(doi('10.1000/example(2)'), '10.1000/example(2)')

    def test_changed_source_rejects_stale_excerpt(self):
        with tempfile.TemporaryDirectory() as temp:
            config, db = self.fixture(temp)
            build(config, db)
            hit = search(db, 'column', 1)[0]
            doc = evidence(db, hit['paper_id'])['documents'][0]
            Path(config['sources'][0]['path']).write_text('[]', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed'):
                excerpt(db, doc['document_id'])

    def test_wos_continuation_and_locator(self):
        rows = list(parse_wos('FN Test\nPT J\nTI Drone routing\n   under uncertainty\nAB A drone model.\nDI 10.1/X\nUT WOS:1\nER\nEF\n'))
        self.assertEqual(rows[0]['title'], 'Drone routing under uncertainty')
        self.assertEqual(rows[0]['line'], 2)
        self.assertEqual(rows[0]['ut'], 'WOS:1')

    def fixture(self, folder):
        p = Path(folder)
        data = p / 'papers.json'
        data.write_text(json.dumps([
            {'title': 'Drone routing', 'doi': 'HTTPS://DOI.ORG/10.1/X', 'abstract': 'Drone delivery by column generation', 'ut': 'WOS:1'},
            {'title': 'Drone routing revised', 'doi': '10.1/x', 'abstract': 'Drone delivery', 'ut': 'WOS:2'},
            {'title': 'Drone routing', 'doi': '10.1/y', 'abstract': None, 'ut': 'WOS:3'},
        ]), encoding='utf-8')
        return {'sources': [{'type': 'json_records', 'path': str(data), 'label': 'test'}]}, p / 'index.sqlite'

    def test_dedup_keeps_sources_distinct_dois_and_chinese_query(self):
        with tempfile.TemporaryDirectory() as temp:
            config, db = self.fixture(temp)
            stats = build(config, db)
            self.assertEqual(stats['papers'], 2)
            self.assertEqual(stats['documents'], 3)
            hits = search(db, '无人机 列生成', 5)
            self.assertEqual(hits[0]['doi'], '10.1/x')
            ev = evidence(db, hits[0]['paper_id'])
            self.assertEqual(len(ev['documents']), 2)
            self.assertEqual(search(db, 'revised', 5)[0]['paper_id'], hits[0]['paper_id'])
            self.assertIn('Drone routing revised', {d['source_title'] for d in ev['documents']})
            self.assertTrue(all(x['sha256'] and x['locator'] for x in ev['documents']))
            self.assertEqual(search(db, 'completelyunknownword', 5), [])
            self.assertEqual(build(config, db)['papers'], 2)

    def test_same_title_without_doi_different_source_ids_not_merged(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / 'data.json'
            data.write_text(json.dumps([
                {'title': 'Editorial', 'year': 2025, 'ut': 'WOS:11'},
                {'title': 'Editorial', 'year': 2026, 'ut': 'WOS:12'},
            ]), encoding='utf-8')
            stats = build({'sources': [{'path': str(data), 'type': 'json_records', 'label': 'test'}]}, Path(temp) / 'db.sqlite')
            self.assertEqual(stats['papers'], 2)

    def test_missing_input_preserves_previous_index(self):
        with tempfile.TemporaryDirectory() as temp:
            config, db = self.fixture(temp)
            build(config, db)
            config['sources'][0]['path'] += '.missing'
            with self.assertRaises(FileNotFoundError):
                build(config, db)
            self.assertTrue(search(db, 'drone', 2))

    def test_note_is_secondary_and_pdf_is_not_read(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p / 'Drone routing_精读.md').write_text('# Drone routing\nDOI: 10.1/x\n这是整理笔记。', encoding='utf-8')
            (p / 'Drone routing.pdf').write_bytes(b'%PDF-not-read')
            config = {'sources': [{'type': 'notes', 'path': str(p), 'label': 'notes'}]}
            db = p / 'test.sqlite'
            build(config, db)
            docs = evidence(db, search(db, 'routing', 1)[0]['paper_id'])['documents']
            self.assertEqual({d['kind'] for d in docs}, {'reading_note', 'pdf_unread'})


if __name__ == '__main__':
    unittest.main()
