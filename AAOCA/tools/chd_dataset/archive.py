"""Safe, resumable AES ZIP extraction. No credentials are persisted."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
import stat

import pyzipper


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def signature(path):
    path = Path(path).resolve()
    s = path.stat()
    return {'path': str(path), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def safe_members(infos):
    """Return flat basenames after validating every archive path and collision."""
    names = set()
    result = []
    for info in infos:
        raw = info.filename.replace('\\', '/')
        p = PurePosixPath(raw)
        if p.is_absolute() or '..' in p.parts or any(':' in x for x in p.parts):
            raise ValueError('Unsafe ZIP member path')
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError('ZIP symbolic links are not supported')
        if info.is_dir():
            continue
        if not raw.endswith('.nii.gz'):
            raise ValueError('Unexpected non-NIfTI archive member')
        key = p.name.casefold()
        if key in names:
            raise ValueError('Duplicate ZIP basename')
        names.add(key)
        result.append((info, p.name))
    return result


def extract_labels(archive, output, password=None, progress=lambda **kw: None):
    archive, output = Path(archive), Path(output)
    dest = output / 'extracted_labels'
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'extraction.json'
    old = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    archive_sha = sha256(archive)
    previous = {r['name']: r for r in old.get('files', [])} if old.get('archive_sha256') == archive_sha else {}
    files, errors = [], []
    with pyzipper.AESZipFile(archive) as z:
        members = safe_members(z.infolist())
        todo = []
        for info, name in members:
            target = dest / name
            prior = previous.get(name)
            if prior and target.is_file() and target.stat().st_size == info.file_size and sha256(target) == prior['sha256']:
                files.append(prior)
            else:
                todo.append((info, name))
        progress(phase='extract', completed=len(files), total=len(members), message='解压标注，验证密码与文件完整性')
        if todo and password is None:
            raise ValueError('Password required for remaining encrypted labels')
        if password is not None:
            z.setpassword(password)
        # A failed password must stop here, before creating hundreds of failed records.
        if todo:
            with z.open(todo[0][0]) as src:
                src.read(1)
        for info, name in todo:
            target = dest / name
            tmp = target.with_name(target.name + '.partial')
            h, size = hashlib.sha256(), 0
            try:
                with z.open(info) as src, tmp.open('wb') as dst:
                    for chunk in iter(lambda: src.read(1 << 20), b''):
                        dst.write(chunk)
                        h.update(chunk)
                        size += len(chunk)
                if size != info.file_size:
                    raise ValueError('Extracted size mismatch')
                os.replace(tmp, target)
                files.append({'name': name, 'member': info.filename, 'bytes': size, 'sha256': h.hexdigest()})
            except Exception as exc:
                errors.append({'name': name, 'error': f'{type(exc).__name__}: {exc}'})
            finally:
                if tmp.exists():
                    tmp.unlink()
            # Save completed members and failures independently so one damaged member
            # does not stop unrelated labels. Failed members are retried next run.
            write_json(manifest_path, {'archive': str(archive.resolve()), 'archive_sha256': archive_sha,
                                      'files': files, 'errors': errors, 'total': len(members), 'updated': stamp()})
            progress(phase='extract', completed=len(files) + len(errors), total=len(members), message=name)
    return {'archive_sha256': archive_sha, 'files': files, 'errors': errors, 'total': len(members)}


if __name__ == '__main__':
    import argparse
    import getpass
    import sys
    p = argparse.ArgumentParser()
    p.add_argument('--archive', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--password-stdin', action='store_true')
    a = p.parse_args()
    password = sys.stdin.readline().rstrip('\r\n') if a.password_stdin else getpass.getpass('ZIP password: ')
    def progress(**state):
        write_json(Path(a.output) / 'progress.json', {**state, 'updated': stamp(), 'status': 'running'})
        if state['completed'] % 100 == 0 or state['completed'] == state['total']:
            print(json.dumps(state, ensure_ascii=False), flush=True)
    result = extract_labels(a.archive, a.output, password.encode(), progress)
    write_json(Path(a.output) / 'progress.json', {'phase': 'extracted', 'status': 'completed',
               'completed': len(result['files']), 'failed': len(result['errors']), 'total': result['total'], 'updated': stamp()})
    print('EXTRACTION_COMPLETE', len(result['files']), 'FAILED', len(result['errors']), flush=True)
    raise SystemExit(1 if result['errors'] else 0)
