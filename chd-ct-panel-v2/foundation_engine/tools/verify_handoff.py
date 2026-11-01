"""Verify this archive offline; never call a model or a remote service."""
import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / '.codex/ARCHIVE_MANIFEST.json').read_text(encoding='utf-8'))
catalog = json.loads((root / 'literature/catalog.json').read_text(encoding='utf-8'))
errors = []


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


for item in manifest['copied_files']:
    p = root / item['destination']
    if not p.is_file() or digest(p) != item['sha256']:
        errors.append({'kind': 'copied_file_integrity', 'path': item['destination']})

pdfs = [r for r in catalog if r.get('pdf')]
notes = [n for r in catalog for n in r.get('notes', [])]
for r in pdfs:
    p = root / r['pdf']
    if not p.is_file() or digest(p) != r['sha256']:
        errors.append({'kind': 'paper_integrity', 'paper': r['id']})
    elif p.read_bytes()[:5] != b'%PDF-':
        errors.append({'kind': 'pdf_header', 'paper': r['id']})
for n in notes:
    if not (root / n).is_file():
        errors.append({'kind': 'missing_note', 'path': n})

json_count = 0
for p in root.rglob('*.json'):
    try:
        json.loads(p.read_text(encoding='utf-8-sig'))
        json_count += 1
    except (UnicodeError, json.JSONDecodeError) as ex:
        errors.append({'kind': 'json_decode', 'path': p.relative_to(root).as_posix(), 'detail': str(ex)})

link_count = 0
md_files = [root / 'README.md', root / 'AGENTS.md']
for sub in ['.codex', 'docs', 'experiments', 'literature']:
    md_files.extend((root / sub).rglob('*.md'))
for p in md_files:
    text = p.read_text(encoding='utf-8-sig')
    if '\ufffd' in text:
        errors.append({'kind': 'replacement_character', 'path': p.relative_to(root).as_posix()})
    for target in re.findall(r'(?<!!)\[[^\]\n]+\]\(([^\n)]+)\)', text):
        target = target.strip('<>').split('#')[0]
        if not target or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target):
            continue
        target = unquote(target)
        link_count += 1
        if not (p.parent / target).exists():
            # The verification result is written below.
            if target == 'ARCHIVE_VALIDATION.json':
                continue
            errors.append({'kind': 'broken_project_link', 'path': p.relative_to(root).as_posix(), 'target': target})

snapshot = root / 'research/academic_agent_snapshot'
with sqlite3.connect(snapshot.joinpath('research.db').as_uri() + '?mode=ro', uri=True) as db:
    integrity = db.execute('pragma integrity_check').fetchone()[0]
    if integrity != 'ok':
        errors.append({'kind': 'sqlite_integrity', 'detail': integrity})
    works = db.execute('select count(*) from works').fetchone()[0]
    jobs = db.execute('select status, count(*) from jobs group by status').fetchall()
latest = json.loads((snapshot / 'exports/latest.json').read_text(encoding='utf-8'))
quality = json.loads((snapshot / 'exports' / latest['version'] / 'quality.json').read_text(encoding='utf-8'))
if works != 61 or len(pdfs) != 17 or len(notes) != 18 or sum(r['app_fulltext_reviewed'] for r in catalog) != 12:
    errors.append({'kind': 'unexpected_counts'})

result = {
    'checked_at': datetime.now(timezone.utc).isoformat(),
    'status': 'passed' if not errors else 'failed',
    'copied_file_checks': len(manifest['copied_files']),
    'mentioned_papers': len(catalog), 'unique_pdf_papers': len(set(r['sha256'] for r in pdfs)),
    'existing_notes': len(notes), 'valid_json_files': json_count,
    'checked_markdown_files': len(md_files), 'checked_project_links': link_count,
    'sqlite_integrity': integrity, 'app_records': works, 'app_job_status_counts': dict(jobs),
    'export_version': latest['version'], 'app_quality_fulltext': quality['document_levels']['full_text'],
    'errors': errors,
    'limits': 'Offline integrity and navigation checks only; no new literature evaluation or segmentation experiment. Historical originals retain original links and statements.'
}
(root / '.codex/ARCHIVE_VALIDATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(0 if not errors else 1)
