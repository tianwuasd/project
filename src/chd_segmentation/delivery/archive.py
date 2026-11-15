from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
from zipfile import ZipFile, ZIP_DEFLATED


LIMIT_BYTES = 1024**3
LIMIT_FILES = 10000


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True).encode('utf-8')).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if not isinstance(name, str) or '\\' in name or ':' in name or path.is_absolute() or \
            not path.parts or any(part in ('', '.', '..') for part in name.split('/')):
        raise ValueError('交付包路径必须为安全相对路径')
    return name


def stream_digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024**2), b''):
        result.update(block)
    return result.hexdigest()


def file_digest(path):
    with Path(path).open('rb') as stream:
        return stream_digest(stream)


@dataclass
class Package:
    archive: ZipFile
    manifest: dict


@contextmanager
def read_package(path):
    with ZipFile(path) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)) or len(names) > LIMIT_FILES or \
                sum(item.file_size for item in infos) > LIMIT_BYTES:
            raise ValueError('交付包条目重复或超过容量限制')
        for item in infos:
            safe_name(item.filename)
            if item.is_dir() or stat.S_IFMT(item.external_attr >> 16) not in (0, stat.S_IFREG):
                raise ValueError('交付包只允许普通文件')
        manifest = json.loads(archive.read('manifest.json'))
        payload = {key: value for key, value in manifest.items() if key != 'identity'}
        if manifest.get('schema') != 1 or identity(payload) != manifest.get('identity'):
            raise ValueError('交付包清单身份无效')
        files = manifest['files']
        targets, objects = set(), {}
        for row in files:
            target = safe_name(row['target'])
            sha = row['sha256']
            if target in targets or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha) or \
                    row['object'] != 'objects/'+sha or type(row['bytes']) is not int or row['bytes'] < 0:
                raise ValueError('交付包清单中的路径或摘要无效')
            targets.add(target)
            if row['object'] in objects and objects[row['object']] != row['bytes']:
                raise ValueError('相同对象的容量不同')
            objects[row['object']] = row['bytes']
        if set(names) != {'manifest.json', *objects} or len(files) > LIMIT_FILES or \
                sum(row['bytes'] for row in files) > LIMIT_BYTES:
            raise ValueError('交付包对象与清单不同')
        for name, size in objects.items():
            if archive.getinfo(name).file_size != size:
                raise ValueError('交付包对象容量不同')
            with archive.open(name) as stream:
                if stream_digest(stream) != name.removeprefix('objects/'):
                    raise ValueError('交付包对象摘要不同')
        yield Package(archive, manifest)


def build_package(entries, output, kind):
    output = Path(output)
    files, objects, targets = [], {}, set()
    if not entries or kind not in ('private', 'q-frozen'):
        raise ValueError('交付包内容或类型无效')
    for item in entries:
        source = Path(item['source']).resolve(strict=True)
        target = safe_name(item['target'])
        if not source.is_file() or source.is_symlink() or target in targets:
            raise ValueError('交付来源必须为普通文件，目标不得重复')
        targets.add(target)
        sha = file_digest(source)
        files.append(dict(target=target, object='objects/'+sha,
                          sha256=sha, bytes=source.stat().st_size))
        objects.setdefault(sha, source)
    if len(files) > LIMIT_FILES or sum(row['bytes'] for row in files) > LIMIT_BYTES:
        raise ValueError('交付包内容超过限制')
    payload = dict(schema=1, kind=kind, files=sorted(files, key=lambda row: row['target']))
    manifest = dict(payload, identity=identity(payload))
    if output.exists():
        with read_package(output) as old:
            if old.manifest != manifest:
                raise FileExistsError('已有交付包身份不同，保留原包')
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name+'.pending')
    if pending.exists():
        raise FileExistsError('存在未完成交付包，请核对归属')
    try:
        with ZipFile(pending, 'x', compression=ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr('manifest.json', json.dumps(manifest, sort_keys=True, ensure_ascii=True))
            for sha, source in sorted(objects.items()):
                archive.write(source, 'objects/'+sha)
        with read_package(pending):
            pass
        pending.replace(output)
    finally:
        if pending.exists():
            pending.unlink()
    return manifest
