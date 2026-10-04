"""Read-only local paper ingestion; page-addressable, content-addressed evidence."""
import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(os.environ.get('MEDICAL_AI_PROJECT', str(Path(__file__).resolve().parents[3])))
SYNONYMS = {
    '心脏': ['cardiac', 'heart', 'myocardial'], '心肌': ['myocardial', 'myocardium'],
    '分割': ['segmentation'], '特征': ['feature', 'representation'],
    '可解释': ['interpretab', 'explainab', 'concept'], '跨中心': ['domain', 'external', 'multicenter'],
    '多模态': ['multimodal', 'multi-modality', 'multisequence'],
    '拓扑': ['topology', 'topological', 'connectivity'], '不确定': ['uncertainty', 'evidential'],
    '先心病': ['congenital heart'], '缺失': ['missing', 'incomplete'],
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def connect(db):
    path = Path(db).resolve()
    if not path.exists():
        raise FileNotFoundError(f'索引不存在，先 build: {path}')
    con = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    return con


def stats(db):
    with closing(connect(db)) as con:
        return {'documents': con.execute('select count(*) from documents').fetchone()[0],
                'sources': con.execute('select count(*) from sources').fetchone()[0],
                'pages': con.execute('select count(*) from pages where page > 0').fetchone()[0],
                'kinds': dict(con.execute('select kind,count(*) from documents group by kind')),
                'note': '文档数不是独立研究数；可提取全文不代表人工审阅。'}


def pdf_pages(path):
    exe = os.environ.get('PDFTOTEXT') or shutil.which('pdftotext')
    if not exe:
        raise RuntimeError('pdftotext 不可用，需安装或配置 PATH')
    p = subprocess.run([exe, '-layout', '-enc', 'UTF-8', str(path), '-'],
                       capture_output=True, timeout=90, check=True)
    text = p.stdout.decode('utf-8', errors='replace')
    pages = text.split('\f')
    if pages and not pages[-1].strip():
        pages.pop()
    if sum(len(x.strip()) for x in pages) < 30:
        raise ValueError('无可靠文本层；需要 OCR 或页面视觉检查')
    return pages


def records(path):
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    value = value.get('papers', value.get('records', [])) if isinstance(value, dict) else value
    if not isinstance(value, list):
        raise ValueError(f'JSON 应为记录列表: {path}')
    return value


def build(config, db):
    config, db = Path(config).resolve(), Path(db).resolve()
    cfg = json.loads(config.read_text(encoding='utf-8-sig'))
    if db == config:
        raise ValueError('数据库输出不能覆盖配置文件')
    sources = cfg.get('sources', [])
    if not sources:
        raise ValueError('sources 不能为空')
    for source in sources:
        path = Path(source['path'])
        if not path.is_absolute():
            path = config.parent / path
        source['resolved'] = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if source.get('type') not in {'papers', 'records'}:
            raise ValueError('source.type 只支持 papers / records')
        root = source['resolved']
        if db == root or (root.is_dir() and db.is_relative_to(root)):
            raise ValueError('数据库输出必须位于原始资料之外，不能覆盖源文件或写入源目录')
        if root.is_dir() and any(p.resolve() == db for p in root.rglob('*') if p.is_file()):
            raise ValueError('数据库输出不能覆盖源目录链接所指向的文件')
    if db.exists():
        try:
            with closing(connect(db)) as previous:
                tables = {row[0] for row in previous.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if tables != {'documents', 'sources', 'pages', 'build_info'}:
                    raise ValueError('现有输出不是本工具的文献索引，拒绝替换')
                if previous.execute('SELECT count(*) FROM build_info').fetchone()[0] != 1:
                    raise ValueError('现有索引缺少构建标识，拒绝替换')
        except sqlite3.Error as exc:
            raise ValueError('现有输出不是文献索引，拒绝替换原文件') from exc
    db.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='corpus-', suffix='.sqlite', dir=db.parent)
    os.close(fd)
    con = sqlite3.connect(tmp)
    errors = []
    try:
        con.executescript('''
        CREATE TABLE documents(id TEXT PRIMARY KEY,title TEXT,kind TEXT,sha256 TEXT,metadata TEXT);
        CREATE TABLE sources(document_id TEXT,path TEXT,label TEXT,PRIMARY KEY(document_id,path));
        CREATE TABLE pages(document_id TEXT,page INTEGER,text TEXT,PRIMARY KEY(document_id,page));
        CREATE TABLE build_info(config_sha256 TEXT,built_at TEXT,errors TEXT);
        ''')

        def add(title, kind, content_hash, location, label, pages, meta):
            key = 'd_' + digest((kind + content_hash).encode())[:20]
            fresh = con.execute('INSERT OR IGNORE INTO documents VALUES(?,?,?,?,?)',
                                (key, title, kind, content_hash, json.dumps(meta, ensure_ascii=False))).rowcount
            con.execute('INSERT OR IGNORE INTO sources VALUES(?,?,?)', (key, location, label))
            if fresh:
                con.executemany('INSERT INTO pages VALUES(?,?,?)', [(key, i, t) for i, t in pages])

        def add_record(record, location, label):
            title = record.get('title') or 'Untitled record'
            abstract = record.get('abstract') or ''
            kind = 'abstract_only' if abstract.strip() else 'metadata_only'
            raw = json.dumps(record, sort_keys=True, ensure_ascii=False)
            # Personal reading notes are metadata, never mislabelled as an author abstract.
            text = title + '\n' + (abstract or raw)
            add(title, kind, digest(raw.encode()), location, label, [(0, text)], record)

        for source in sources:
            root, label = source['resolved'], source.get('label', '')
            if source['type'] == 'records':
                for i, record in enumerate(records(root)):
                    add_record(record, f'{root.as_posix()}#record={i}', label)
                continue
            files = sorted(root.rglob('*.pdf')) if root.is_dir() else [root]
            by_file = {}
            if root.is_dir():
                metadata_file = root / '文献元数据与来源.json'
                if metadata_file.exists():
                    for i, record in enumerate(records(metadata_file)):
                        filename = record.get('local_pdf', '')
                        if filename and (root / filename).is_file():
                            by_file[(root / filename).resolve()] = record
                        else:
                            add_record(record, f'{metadata_file.as_posix()}#record={i}', label)
                if source.get('notes'):
                    files += sorted(p for p in root.rglob('*.md') if '精读' in p.name)
            for path in files:
                if path.name.startswith('._') or '__MACOSX' in path.parts:
                    continue
                raw = path.read_bytes()
                meta = by_file.get(path.resolve(), {})
                title = meta.get('title', path.stem)
                if path.suffix.lower() == '.pdf':
                    try:
                        pages = list(enumerate(pdf_pages(path), 1))
                        kind = 'pdf_text'
                    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                        kind, pages = 'pdf_unread', [(0, title)]
                        errors.append({'path': str(path), 'error': str(exc)})
                elif path.suffix.lower() == '.md':
                    kind, pages = 'secondary_note', [(0, raw.decode('utf-8-sig'))]
                else:
                    raise ValueError(f'仅支持 PDF/Markdown: {path}')
                add(title, kind, digest(raw), path.as_posix(), label, pages, meta)
        con.execute('INSERT INTO build_info VALUES(?,?,?)',
                    (digest(config.read_bytes()), datetime.now(timezone.utc).isoformat(), json.dumps(errors)))
        con.commit()
        con.close()
        os.replace(tmp, db)
    except BaseException:
        con.close()
        Path(tmp).unlink(missing_ok=True)
        raise
    return {**stats(db), 'extraction_errors': errors, 'database': str(db)}


def query_terms(query):
    terms = re.findall(r'[\w-]+', query.lower())
    for word, synonyms in SYNONYMS.items():
        if word in query:
            terms.extend(synonyms)
    return sorted(set(terms))


def search(db, query, limit=8):
    terms = query_terms(query)
    if not terms or limit < 1:
        return []
    best = {}
    with closing(connect(db)) as con:
        rows = con.execute('''SELECT d.id,d.title,d.kind,p.page,p.text,
            (SELECT path FROM sources s WHERE s.document_id=d.id ORDER BY path LIMIT 1) AS source
            FROM documents d JOIN pages p ON p.document_id=d.id''')
        for row in rows:
            title, body = row['title'].lower(), row['text'].lower()
            hits = [t for t in terms if t in title or t in body]
            if not hits:
                continue
            score = sum((3 if t in title else 0) + min(body.count(t), 4) for t in hits)
            if row['kind'] == 'pdf_text':
                score += 1
            start = max(0, min([body.find(t) for t in hits if t in body] or [0]) - 130)
            result = {'document_id': row['id'], 'title': row['title'], 'kind': row['kind'],
                      'source': row['source'], 'page': row['page'] or None, 'score': score,
                      'snippet': row['text'][start:start + 900], 'matched_terms': hits}
            old = best.get(row['id'])
            if old is None or score > old['score']:
                best[row['id']] = result
    return sorted(best.values(), key=lambda x: (-x['score'], x['title']))[:limit]


def read_page(db, document_id, page):
    with closing(connect(db)) as con:
        row = con.execute('''SELECT d.id AS document_id,d.title,d.kind,d.sha256,p.page,p.text
            FROM documents d JOIN pages p ON d.id=p.document_id WHERE d.id=? AND p.page=?''',
                          (document_id, page)).fetchone()
        if not row:
            raise ValueError('文档或页码不存在；笔记/摘要使用 page=0')
        return {**dict(row), 'sources': [r[0] for r in con.execute(
            'SELECT path FROM sources WHERE document_id=?', (document_id,))]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', type=Path, default=PROJECT / 'data/corpus.sqlite')
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build'); b.add_argument('--config', type=Path, default=PROJECT / 'config/sources.json')
    sub.add_parser('stats')
    s = sub.add_parser('search'); s.add_argument('query'); s.add_argument('--limit', type=int, default=8)
    r = sub.add_parser('read'); r.add_argument('document_id'); r.add_argument('--page', type=int, required=True)
    args = p.parse_args()
    if args.command == 'build': result = build(args.config, args.db)
    elif args.command == 'stats': result = stats(args.db)
    elif args.command == 'search': result = search(args.db, args.query, args.limit)
    else: result = read_page(args.db, args.document_id, args.page)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
