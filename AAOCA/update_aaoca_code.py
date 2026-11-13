import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import uuid

REPOSITORY = 'tianwuasd/project'
LIMIT = 8 * 1024 * 1024
FILE_LIMIT = 512 * 1024
RECEIPT = '.aaoca-code-receipt.json'
LOCK = '.aaoca-code-update.lock'
SUFFIXES = {'.py', '.sh', '.cmd', '.ps1', '.md', '.txt', '.toml', '.json',
            '.yaml', '.yml', '.html', '.js', '.css', '.csv', '.ini', '.cfg'}
REQUIRED = {'start_segmentation.py', 'start_segmentation.sh',
            'tools/coarse_segmentation/aaoca_seg/direct.py',
            'tools/coarse_segmentation/aaoca_seg/workbench.py'}


def blob_hash(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or str(path) != name or '..' in path.parts
            or any(part in {'.git', '.venv', 'runtime', '__pycache__'} or part.endswith(('.', ' '))
                   for part in path.parts)
            or any(character in name for character in '\\:\r\n\0')
            or (path.suffix.lower() not in SUFFIXES and path.name not in {'.gitignore', '.gitattributes'})):
        raise ValueError('源码清单包含非代码路径或不安全路径')
    return path


def checked_path(root, relative):
    path = root.joinpath(*safe_name(relative).parts)
    for item in (root, *path.relative_to(root).parents):
        item = root / item if not item.is_absolute() else item
        if item.exists() and (item.is_symlink() or
                             getattr(item.lstat(), 'st_file_attributes', 0) & 1024):
            raise ValueError('代码目标路径包含链接')
        if item.exists() and not item.is_dir():
            raise ValueError('代码父路径已被普通文件占用')
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError('代码目标必须为普通文件')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('代码目标路径越界')
    return path


def validate_snapshot(snapshot, data):
    rows = snapshot['files']
    names = [row['path'] for row in rows]
    if (not re.fullmatch('[0-9a-f]{40}', snapshot['commit']) or not 1 <= len(rows) <= 512
            or len(names) != len(set(names)) or set(names) != set(data) or not REQUIRED <= set(names)
            or sum(row['size'] for row in rows) > LIMIT):
        raise ValueError('AAOCA 源码快照不完整或超出代码容量限制')
    for row in rows:
        safe_name(row['path'])
        content = data[row['path']]
        if (row['mode'] not in {'100644', '100755'} or not 0 <= row['size'] <= FILE_LIMIT
                or len(content) != row['size'] or blob_hash(content) != row['sha']):
            raise ValueError('源码文件长度或 Git blob 哈希不符：' + row['path'])
        if row['path'].endswith('.py'):
            compile(content, row['path'], 'exec')


def remove_owned(directory, parent):
    if directory.parent != parent or directory.is_symlink():
        raise ValueError('更新临时目录归属不符，拒绝删除')
    for path in directory.rglob('*'):
        if path.is_symlink():
            raise ValueError('更新临时目录包含链接，拒绝删除')
        if path.is_file():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
    shutil.rmtree(directory)


class Download:
    def __init__(self, mirror, work_root, timeout=900):
        self.url = ('https://gh-proxy.com/' if mirror == 'gh-proxy' else '') + f'https://github.com/{REPOSITORY}.git'
        self.work_root = Path(work_root).resolve()
        self.deadline = time.monotonic() + timeout

    def git(self, cache, *arguments, online=False, trace=False, input_data=None):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('代码下载超过总时间限制')
        env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
        env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT='0',
                   GCM_INTERACTIVE='Never', GIT_NO_LAZY_FETCH='1', LC_ALL='C')
        if not online:
            env['GIT_ALLOW_PROTOCOL'] = ''
        if trace:
            env['GIT_TRACE_PACKET'] = '1'
        command = ['git', '-c', 'protocol.version=2', '-c', 'credential.helper=',
                   '-c', 'http.extraHeader=', '-c', 'core.askPass=',
                   '-c', 'http.followRedirects=false', '-c', 'gc.auto=0',
                   '-c', 'maintenance.auto=false', '-C', str(cache), *arguments]
        child = subprocess.Popen(command, stdin=subprocess.PIPE if input_data is not None else subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                 start_new_session=os.name != 'nt',
                                 creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
        completed = False
        try:
            stdout, stderr = child.communicate(input=input_data, timeout=min(remaining, 180 if online else 30))
            completed = True
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, command, stdout, stderr)
            return stdout, stderr
        finally:
            if not completed:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False)
                else:
                    subprocess.run(['kill', '-KILL', '--', '-' + str(child.pid)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False)
                child.wait(timeout=15)

    def snapshot(self, commit=None):
        cache = Path(tempfile.mkdtemp(prefix='.aaoca-code-update-', dir=self.work_root)).resolve()
        try:
            self.git(cache, 'init', '--bare')
            stdout, capabilities = self.git(cache, 'ls-remote', self.url, 'refs/heads/main', online=True, trace=True)
            if not re.search(rb'fetch=[^\r\n]*\bfilter\b', capabilities):
                raise ValueError('远端未明确支持按需对象过滤，拒绝下载整个仓库')
            known = stdout.decode('ascii').split()[0]
            commit = known if commit is None else commit
            if not re.fullmatch('[0-9a-f]{40}', commit):
                raise ValueError('请使用完整提交 SHA')
            print('只下载提交/目录元信息：' + commit, flush=True)
            for name, value in [('remote.origin.url', self.url), ('remote.origin.promisor', 'true'),
                                ('remote.origin.partialclonefilter', 'blob:none')]:
                self.git(cache, 'config', name, value)
            self.git(cache, 'fetch', '--depth=1', '--no-tags', '--filter=blob:none', '--progress',
                     'origin', commit, online=True)
            actual = self.git(cache, 'rev-parse', 'FETCH_HEAD')[0].decode().strip()
            if actual != commit:
                raise ValueError('下载的提交身份不符')
            listing = self.git(cache, 'ls-tree', '-r', '-z', '--full-tree', commit, '--', 'AAOCA/')[0]
            rows = []
            for entry in listing.split(b'\0'):
                if not entry:
                    continue
                metadata, path = entry.split(b'\t', 1)
                mode, kind, oid = metadata.decode('ascii').split()
                path = path.decode('utf-8')
                if not path.startswith('AAOCA/') or kind != 'blob' or mode not in {'100644', '100755'}:
                    raise ValueError('AAOCA 包含非普通代码对象')
                name = path[len('AAOCA/'):]
                safe_name(name)
                rows.append({'path': name, 'sha': oid, 'mode': mode})
            if not 1 <= len(rows) <= 512 or not REQUIRED <= {row['path'] for row in rows}:
                raise ValueError('AAOCA 源码文件清单不完整')
            wanted = {row['sha'] for row in rows}
            inventory = self.git(cache, 'cat-file', '--batch-all-objects', '--batch-check=%(objectname) %(objecttype)')[0]
            available = {line.split()[0].decode() for line in inventory.splitlines()}
            missing = sorted(wanted - available)
            print(f'仅获取 AAOCA：{len(rows)} 文件、{len(missing)} 个缺失 blob', flush=True)
            for offset in range(0, len(missing), 64):
                self.git(cache, 'fetch', '--no-tags', '--no-write-fetch-head', '--progress', 'origin',
                         *missing[offset:offset + 64], online=True)
                print(f'已获取 {min(offset + 64, len(missing))}/{len(missing)} 个代码 blob', flush=True)
            inventory = self.git(cache, 'cat-file', '--batch-all-objects', '--batch-check=%(objectname) %(objecttype)')[0]
            available = {line.split()[0].decode() for line in inventory.splitlines()}
            obtained = {line.split()[0].decode() for line in inventory.splitlines() if line.split()[1] == b'blob'}
            if obtained - wanted:
                raise ValueError('下载缓存包含 AAOCA 清单之外的文件内容，拒绝应用')
            if not wanted <= available:
                raise ValueError('所需代码对象未下载完整；离线读取禁止隐式联网')
            sizes = self.git(cache, 'cat-file', '--batch-check=%(objectname) %(objectsize)',
                             input_data=('\n'.join(sorted(wanted)) + '\n').encode())[0]
            sizes = {oid: int(size) for oid, size in (line.decode().split() for line in sizes.splitlines())}
            for row in rows:
                row['size'] = sizes[row['sha']]
            if any(row['size'] > FILE_LIMIT for row in rows) or sum(row['size'] for row in rows) > LIMIT:
                raise ValueError('下载的源码超出代码容量限制')
            tree = self.git(cache, 'rev-parse', commit + ':AAOCA')[0].decode().strip()
            snapshot = {'schema': 1, 'commit': commit, 'tree': tree, 'files': rows}
            def one(row):
                return row['path'], self.git(cache, 'cat-file', 'blob', row['sha'])[0]
            data = {}
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(one, row) for row in rows]
                for future in as_completed(futures):
                    name, content = future.result()
                    data[name] = content
            validate_snapshot(snapshot, data)
            print(f'AAOCA 校验完成：{len(rows)} 文件，{sum(map(len, data.values()))} 字节', flush=True)
            return snapshot, data
        finally:
            remove_owned(cache, self.work_root)


def prepare_file(path, data, mode, prefix):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    completed = False
    try:
        with tempfile.NamedTemporaryFile(prefix=prefix, dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        completed = True
        return temporary
    finally:
        if not completed and temporary is not None:
            temporary.unlink(missing_ok=True)


def apply_snapshot(target, snapshot, data):
    validate_snapshot(snapshot, data)
    receipt_path = target / RECEIPT
    if receipt_path.is_symlink():
        raise ValueError('更新回执不能是链接')
    previous = json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else None
    old_rows = previous['files'] if previous else []
    if previous and (previous.get('schema') != 1 or previous.get('target') != str(target)):
        raise ValueError('旧更新回执不属于当前代码目录')
    new_rows = {row['path']: row for row in snapshot['files']}
    removed = [row for row in old_rows if row['path'] not in new_rows]
    for row in removed:
        path = checked_path(target, row['path'])
        if path.exists() and blob_hash(path.read_bytes()) != row['sha']:
            raise ValueError('退役源码已被本地修改，保留并停止更新：' + row['path'])
    paths = {name: checked_path(target, name) for name in new_rows}
    paths.update({row['path']: checked_path(target, row['path']) for row in removed})
    paths[RECEIPT] = receipt_path
    existed = {name: path.exists() for name, path in paths.items()}
    new_directories = set()
    for path in paths.values():
        parent = path.parent
        while parent != target and not parent.exists():
            new_directories.add(parent)
            parent = parent.parent
    payloads = {}
    for name, row in new_rows.items():
        mode = 0o755 if row['mode'] == '100755' else 0o644
        if (existed[name] and paths[name].read_bytes() == data[name]
                and (os.name == 'nt' or stat.S_IMODE(paths[name].stat().st_mode) == mode)):
            continue
        payloads[name] = (data[name], mode)
    for row in removed:
        if existed[row['path']]:
            payloads[row['path']] = None
    record = dict(snapshot, target=str(target))
    payloads[RECEIPT] = ((json.dumps(record, ensure_ascii=False, indent=2) + '\n').encode(), 0o600)
    prefix = '.aaoca-code-' + uuid.uuid4().hex + '-'
    backups = {name: paths[name].with_name(prefix + 'backup-' + paths[name].name)
               for name in payloads if existed[name]}
    if any(path.exists() for path in backups.values()):
        raise ValueError('更新备份路径已存在')
    candidates, committed, restored = {}, False, False
    try:
        # 全部候选文件先在目标磁盘写完，应用与恢复只执行同盘改名。
        for name, payload in payloads.items():
            candidates[name] = prepare_file(paths[name], *payload, prefix) if payload is not None else None
        for name, candidate in candidates.items():
            if existed[name]:
                paths[name].replace(backups[name])
            if candidate is not None:
                candidate.replace(paths[name])
        for name, row in new_rows.items():
            if blob_hash(paths[name].read_bytes()) != row['sha']:
                raise ValueError('应用后的代码校验失败：' + name)
        if json.loads(receipt_path.read_text(encoding='utf-8')) != record:
            raise ValueError('更新回执写入核验失败')
        committed = True
    finally:
        if not committed:
            for name in reversed(list(payloads)):
                if name in backups and backups[name].exists():
                    backups[name].replace(paths[name])
                elif not existed[name] and name in candidates and candidates[name] is not None and not candidates[name].exists():
                    paths[name].unlink(missing_ok=True)
            restored = True
        if committed or restored:
            for parent in {path.parent for path in paths.values()}:
                if parent.is_dir():
                    for path in parent.glob(prefix + '*'):
                        path.unlink()
        if not committed:
            for directory in sorted(new_directories, key=lambda item: len(item.parts), reverse=True):
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
    for row in removed:
        parent = paths[row['path']].parent
        while parent != target and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    return {'commit': snapshot['commit'], 'files': len(new_rows), 'updated': len(payloads) - 1,
            'removed_retired_files': sum(existed[row['path']] for row in removed)}


def update(target, work_root, mirror='gh-proxy', commit=None):
    requested = Path(target).expanduser().absolute()
    if requested.is_symlink() or getattr(requested.lstat(), 'st_file_attributes', 0) & 1024:
        raise ValueError('AAOCA 代码目录不能是链接')
    target = requested.resolve()
    work_root = Path(work_root).expanduser().resolve()
    if (target.name != 'AAOCA' or not work_root.is_dir() or work_root.is_relative_to(target)
            or not (target / 'start_segmentation.py').is_file()
            or not (target / 'tools/coarse_segmentation').is_dir()):
        raise ValueError('请指定已有 AAOCA 代码目录及目录之外的工作位置')
    lock = target / LOCK
    lock.mkdir()
    staging = None
    started = time.monotonic()
    try:
        staging = Path(tempfile.mkdtemp(prefix='.aaoca-code-update-', dir=work_root)).resolve()
        snapshot, data = Download(mirror, staging).snapshot(commit)
        for name, content in data.items():
            path = staging.joinpath(*safe_name(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        result = apply_snapshot(target, snapshot, data)
        result['seconds'] = round(time.monotonic() - started, 3)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
    finally:
        try:
            if staging is not None:
                remove_owned(staging, work_root)
        finally:
            lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description='按 v2 的按需清单与校验顺序，仅更新 AAOCA 目录代码')
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--work-root', required=True, type=Path)
    parser.add_argument('--mirror', choices=['gh-proxy', 'official'], default='gh-proxy')
    parser.add_argument('--commit', help='完整提交 SHA；省略时查询 main')
    args = parser.parse_args()
    def stopped(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    update(args.target, args.work_root, args.mirror, args.commit)


if __name__ == '__main__':
    main()
