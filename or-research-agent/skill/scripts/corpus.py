"""Local, read-only source adapters and a rebuildable literature evidence index."""
import argparse
import hashlib
import json
import os
import re
import sqlite3
import uuid
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


def winpath(path):
    p = os.path.abspath(str(path))
    if os.name == 'nt' and not p.startswith('\\\\?\\'):
        p = '\\\\?\\UNC\\' + p[2:] if p.startswith('\\\\') else '\\\\?\\' + p
    return Path(p)


def cleanpath(path):
    s = str(path)
    return '\\\\' + s[8:] if s.startswith('\\\\?\\UNC\\') else s.removeprefix('\\\\?\\')


def sha(path):
    with winpath(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def doi(value):
    value = re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)', '', str(value or '').strip(), flags=re.I)
    value = re.split(r'[\s<>\[\]"\x27`]', value, maxsplit=1)[0].lower().rstrip('.,;*')
    while value.endswith(')') and value.count(')') > value.count('('):
        value = value[:-1]
    return value


def normalized_title(value):
    return ''.join(c for c in value.casefold() if c.isalnum())


def parse_wos(text):
    record, key, line = {}, None, 0
    for number, raw in enumerate(text.splitlines(), 1):
        tag = raw[:2]
        if tag == 'PT':
            record, key, line = {}, None, number
        if tag == 'ER':
            if record.get('TI'):
                yield {'title': ' '.join(record['TI']), 'abstract': ' '.join(record.get('AB', [])),
                       'doi': ' '.join(record.get('DI', [])), 'ut': ' '.join(record.get('UT', [])),
                       'year': ' '.join(record.get('PY', [])), 'source': ' '.join(record.get('SO', [])),
                       'authors': record.get('AF', record.get('AU', [])), 'line': line}
            record, key = {}, None
        elif raw.startswith('   ') and key:
            record[key].append(raw.strip())
        elif re.match(r'^[A-Z0-9]{2} ', raw):
            key = tag
            record.setdefault(key, []).append(raw[3:].strip())
    if record.get('TI'):
        raise ValueError('WoS record is incomplete: missing ER')


def source_documents(config):
    docs = []
    for source in config['sources']:
        path = winpath(source['path'])
        if not path.exists():
            raise FileNotFoundError(cleanpath(path))
        label, typ = source['label'], source['type']
        if typ in ('json_records', 'wos'):
            text = path.read_text(encoding='utf-8-sig')
            records = json.loads(text) if typ == 'json_records' else list(parse_wos(text))
            digest = sha(path)
            for number, r in enumerate(records, 1):
                if not r.get('title'):
                    raise ValueError(f'{label} record {number} has no title')
                abstract = r.get('abstract') or ''
                docs.append({**r, 'abstract': abstract, 'body': abstract,
                             'kind': 'abstract' if abstract else 'metadata_only', 'path': cleanpath(path),
                             'locator': f'json record {number} (1-based)' if typ == 'json_records' else f'line {r["line"]}',
                             'sha256': digest, 'label': label})
        elif typ == 'notes':
            for note in sorted(path.rglob('*_精读.md')):
                text = note.read_text(encoding='utf-8-sig')
                title = note.stem.removesuffix('_精读')
                # Notes may contain cited DOIs; accept only an explicitly labelled header.
                header = re.search(r'DOI[^\n]*?(10\.\d{4,9}/[^\s<>|]+)', text[:3000], re.I)
                note_doi = doi(header.group(1)) if header else ''
                docs.append({'title': title, 'doi': note_doi, 'kind': 'reading_note', 'body': text,
                             'abstract': '', 'path': cleanpath(note), 'locator': f'lines 1-{len(text.splitlines())}',
                             'sha256': sha(note), 'label': label})
            for pdf in sorted(path.rglob('*.pdf')):
                docs.append({'title': pdf.stem, 'kind': 'pdf_unread', 'body': '', 'abstract': '',
                             'path': cleanpath(pdf), 'locator': 'file inventory; pages not extracted',
                             'sha256': sha(pdf), 'label': label})
        else:
            raise ValueError(f'Unsupported source type: {typ}')
    return docs


def build(config, db):
    """Rebuild atomically after successful reads; never modify corpus source files."""
    docs = source_documents(config)
    target = winpath(db)
    source_paths = {os.path.normcase(os.path.abspath(d['path'])) for d in docs}
    if os.path.normcase(cleanpath(target)) in source_paths:
        raise ValueError('Index path must not be a source path')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with closing(sqlite3.connect(str(target))) as old:
            marker = old.execute('PRAGMA application_id').fetchone()[0]
            if marker != 1330791239:
                raise ValueError('Refusing to replace a database not owned by or-research')
    temp = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
    title_ids = defaultdict(set)
    for d in docs:
        if doi(d.get('doi')):
            d['identity'] = 'doi:' + doi(d['doi'])
        elif d.get('ut'):
            d['identity'] = 'ut:' + d['ut']
        elif d['kind'] in ('abstract', 'metadata_only'):
            d['identity'] = 'bibliographic:' + json.dumps([normalized_title(d['title']), d.get('year'), d.get('authors')], ensure_ascii=False)
        if d.get('identity'):
            title_ids[normalized_title(d['title'])].add(d['identity'])
    paper_rows, doc_rows = {}, []
    for d in docs:
        nd, nt = doi(d.get('doi')), normalized_title(d['title'])
        candidates = title_ids[nt]
        key = d.get('identity') or (next(iter(candidates)) if len(candidates) == 1 else 'localtitle:' + nt)
        pid = 'p_' + hashlib.sha256(key.encode()).hexdigest()[:16]
        if pid not in paper_rows:
            paper_rows[pid] = {'paper_id': pid, 'title': d['title'], 'doi': nd, 'year': str(d.get('year') or ''),
                               'venue': d.get('source', ''), 'authors': d.get('authors', [])}
        row = paper_rows[pid]
        if nd and not row['doi']:
            row['doi'] = nd
        for field, source_field in [('year', 'year'), ('venue', 'source'), ('authors', 'authors')]:
            if not row[field] and d.get(source_field):
                row[field] = str(d[source_field]) if field == 'year' else d[source_field]
        did = 'd_' + hashlib.sha256((d['path'] + '|' + d['locator']).encode()).hexdigest()[:16]
        source_metadata = {k: d[k] for k in ('title', 'doi', 'year', 'source', 'authors', 'ut', 'source_file', 'source_record_number') if k in d}
        doc_rows.append((did, pid, d['kind'], d['label'], d['path'], d['locator'], d['sha256'],
                         d['body'], str(d.get('ut') or ''), d['title'], json.dumps(source_metadata, ensure_ascii=False)))
    try:
        with closing(sqlite3.connect(str(temp))) as conn:
            conn.executescript('''PRAGMA application_id=1330791239;
                CREATE TABLE papers(paper_id TEXT PRIMARY KEY,title TEXT,doi TEXT,year TEXT,venue TEXT,authors TEXT);
                CREATE TABLE documents(document_id TEXT PRIMARY KEY,paper_id TEXT,kind TEXT,label TEXT,path TEXT,
                    locator TEXT,sha256 TEXT,body TEXT,source_id TEXT,source_title TEXT,source_metadata TEXT);
                CREATE INDEX document_paper ON documents(paper_id);
                CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);''')
            conn.executemany('INSERT INTO papers VALUES(?,?,?,?,?,?)',
                [(r['paper_id'], r['title'], r['doi'], r['year'], r['venue'], json.dumps(r['authors'], ensure_ascii=False)) for r in paper_rows.values()])
            conn.executemany('INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?)', doc_rows)
            stats = {'papers': len(paper_rows), 'documents': len(doc_rows), 'by_kind': dict(Counter(d['kind'] for d in docs)),
                     'by_source': dict(Counter(d['label'] for d in docs)), 'built_at': datetime.now(timezone.utc).isoformat(),
                     'source_policy': 'abstract/metadata and secondary notes; PDF inventory is not full-text reading'}
            conn.execute('INSERT INTO metadata VALUES(?,?)', ('stats', json.dumps(stats, ensure_ascii=False)))
            conn.commit()
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()
    return stats


ALIASES = {'无人机': ['drone', 'uav', 'unmanned aerial'], '配送': ['delivery', 'distribution'],
           '即时': ['on-demand', 'same-day', 'real-time'], '骑手': ['courier', 'rider'],
           '中继': ['relay', 'transshipment'], '列生成': ['column generation', 'column-and-row', 'branch-and-price'],
           '鲁棒': ['robust'], '不确定': ['uncertain', 'stochastic'], '调度': ['scheduling', 'dispatch'],
           '选址': ['location', 'facility'], '路径': ['routing', 'route'], '学习': ['learning'],
           '建模': ['formulation', 'modeling', 'modelling'], '低空': ['drone', 'uav', 'air mobility'],
           '物流': ['logistics', 'delivery'], '容量': ['capacity', 'capacitated'], '充电': ['charging', 'battery']}


def query_groups(query):
    groups = []
    remaining = query.casefold()
    for word in sorted(ALIASES, key=len, reverse=True):
        if word in remaining:
            groups.append([word] + ALIASES[word])
            remaining = remaining.replace(word, ' ')
    for token in re.findall(r'[^\W_]+(?:[-][^\W_]+)*', remaining, re.UNICODE):
        groups.append([token])
    return groups


def readonly(db):
    p = winpath(db)
    if not p.is_file():
        raise FileNotFoundError(cleanpath(p))
    conn = sqlite3.connect(Path(cleanpath(p)).as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def search(db, query, limit=10):
    groups = query_groups(query)
    if not groups or not 1 <= limit <= 100:
        raise ValueError('Provide non-empty query and limit between 1 and 100')
    with closing(readonly(db)) as conn:
        rows = conn.execute('''SELECT p.*,group_concat(d.source_title || ' ' || d.body,' ') AS text,
            group_concat(DISTINCT d.kind) AS evidence_levels FROM papers p
            LEFT JOIN documents d USING(paper_id) GROUP BY p.paper_id''').fetchall()
    results = []
    for row in rows:
        r = dict(row)
        title, body = r['title'].casefold(), (r.pop('text') or '').casefold()
        if not all(any(w in title + ' ' + body for w in group) for group in groups):
            continue
        r['match_score'] = sum(max(5 * (w in title) + (w in body) for w in g) for g in groups)
        r['authors'] = json.loads(r['authors'])
        results.append(r)
    return sorted(results, key=lambda r: (-r['match_score'], r['paper_id']))[:limit]


def evidence(db, paper_id):
    with closing(readonly(db)) as conn:
        paper = conn.execute('SELECT * FROM papers WHERE paper_id=?', (paper_id,)).fetchone()
        if paper is None:
            raise ValueError('Unknown paper_id: ' + paper_id)
        docs = [dict(r) for r in conn.execute('SELECT * FROM documents WHERE paper_id=?', (paper_id,))]
        for d in docs:
            d['source_metadata'] = json.loads(d['source_metadata'])
    return {'paper': dict(paper), 'documents': docs}


def excerpt(db, document_id, start=1, end=None):
    with closing(readonly(db)) as conn:
        row = conn.execute('SELECT * FROM documents WHERE document_id=?', (document_id,)).fetchone()
    if row is None:
        raise ValueError('Unknown document_id')
    d = dict(row)
    if sha(d['path']) != d['sha256']:
        raise ValueError('Source changed since indexing; rebuild before extracting evidence')
    if d['kind'] == 'pdf_unread':
        import pymupdf
        with pymupdf.open(str(winpath(d['path']))) as pdf:
            stop = end or start
            if not 1 <= start <= stop <= len(pdf):
                raise ValueError('Invalid PDF page range (1-based)')
            content = [{'page': i + 1, 'text': pdf[i].get_text()} for i in range(start - 1, stop)]
        return {'document_id': document_id, 'kind': 'pdf_text_extract', 'path': d['path'], 'sha256': d['sha256'],
                'pages': content, 'warning': 'Text extraction only; inspect page images for equations and tables. Empty pages need OCR.'}
    lines = d['body'].splitlines()
    stop = end or min(start + 79, max(1, len(lines)))
    if start < 1 or stop < start:
        raise ValueError('Invalid line range')
    return {**{k: d[k] for k in ('document_id', 'kind', 'path', 'locator', 'sha256')},
            'lines': [{'line': n + 1, 'text': t} for n, t in enumerate(lines) if start <= n + 1 <= stop]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(os.environ.get('OR_RESEARCH_HOME', '.'))
    parser.add_argument('--db', default=str(project / 'data/corpus.sqlite'))
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build'); b.add_argument('--config', default=str(project / 'config/sources.json'))
    s = sub.add_parser('search'); s.add_argument('query'); s.add_argument('--limit', type=int, default=8)
    e = sub.add_parser('evidence'); e.add_argument('paper_id')
    x = sub.add_parser('excerpt'); x.add_argument('document_id'); x.add_argument('--start', type=int, default=1); x.add_argument('--end', type=int)
    sub.add_parser('stats')
    args = parser.parse_args()
    if args.command == 'build': result = build(json.loads(winpath(args.config).read_text(encoding='utf-8-sig')), args.db)
    elif args.command == 'search': result = search(args.db, args.query, args.limit)
    elif args.command == 'evidence': result = evidence(args.db, args.paper_id)
    elif args.command == 'excerpt': result = excerpt(args.db, args.document_id, args.start, args.end)
    else:
        with closing(readonly(args.db)) as conn:
            result = json.loads(conn.execute("SELECT value FROM metadata WHERE key='stats'").fetchone()[0])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
