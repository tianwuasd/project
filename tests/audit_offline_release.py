import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from update import allowed


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit_assets(directory, output):
    values = {}
    for path in sorted(Path(directory).glob('*_材料引用.json')):
        for asset in json.loads(path.read_text(encoding='utf-8'))['assets']:
            values[asset['path']] = asset
    results = []
    for asset in values.values():
        path = Path(asset['path']).resolve(strict=True)
        if path.is_dir():
            files = [p for p in path.rglob('*') if p.is_file()]
            rows = [dict(name=p.relative_to(path).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                    for p in sorted(files)]
            actual = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            size, count = sum(p.stat().st_size for p in files), len(files)
        else:
            actual, size, count = sha(path), path.stat().st_size, 1
        results.append(dict(path=str(path), bytes=size, files=count, sha256=actual,
            unchanged=(actual, size, count)==(asset['sha256'], asset['bytes'], asset['file_count'])))
    receipt = dict(state='passed' if all(r['unchanged'] for r in results) else 'changed',
        assets=results, reused_bytes=sum(r['bytes'] for r in results), large_files_copied=0,
        server_upload_status='not_verified_no_server_login')
    Path(output).write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    assert receipt['state']=='passed', receipt
    print(json.dumps(dict(state=receipt['state'], assets=len(results), reused_bytes=receipt['reused_bytes'])))


def public_files(offset, limit):
    def git(*words):
        return subprocess.check_output(['git', '-C', str(ROOT), *words])
    if git('status', '--porcelain').strip():
        raise ValueError('发布读取仅接受已提交且干净的工作树')
    names = git('ls-tree', '-r', '--name-only', 'HEAD').decode().splitlines()
    names = [name for name in names if allowed(name)]
    assert len(names)<=200
    entries = []
    for name in names[offset:offset+limit]:
        data = git('show', 'HEAD:'+name)
        entries.append(dict(path=name, sha=git('rev-parse', 'HEAD:'+name).decode().strip(),
            bytes=len(data), content=data.decode('utf-8')))
    print(json.dumps(dict(head=git('rev-parse', 'HEAD').decode().strip(), total=len(names), entries=entries)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets')
    parser.add_argument('--output')
    parser.add_argument('--public-files', action='store_true')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--limit', type=int, default=10)
    args = parser.parse_args()
    if args.public_files:
        public_files(args.offset, args.limit)
    else:
        audit_assets(args.assets, args.output)
