import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
import zipfile

import psutil


REVISION = '96f3e1b404ba14f2cfac60ee6ae87c269a7b7923'
FILES = {'model.safetensors': 3439938512, 'config.json': 25843,
         'processor_config.json': 1712, 'merges.txt': 524619,
         'special_tokens_map.json': 588, 'tokenizer.json': 3642073,
         'tokenizer_config.json': 799, 'vocab.json': 862328,
         'LICENSE': 7352, 'README.md': 26038}
PACKAGES = ['modelscope==1.40.3', 'modelscope-hub==0.7.1', 'transformers==5.24.0',
            'regex==2026.11.19', 'tokenizers==0.23.3', 'typer==0.29.2',
            'cryptography==50.0.2', 'cffi==2.1.1', 'pycparser==3.12',
            'shellingham==1.5.4', 'annotated-doc==0.0.7']


def inventory(root):
    root = Path(root)
    if not root.exists():
        return []
    if root.is_symlink() or root.lstat().st_file_attributes & 0x400:
        raise ValueError('本次目录含链接，拒绝清理或预算遍历')
    entries = []
    with os.scandir(root) as listing:
        for entry in listing:
            # Windows DirEntry 保存枚举时元数据，文件重命名不会使采样改读已消失路径。
            info = entry.stat(follow_symlinks=False)
            if info.st_file_attributes & 0x400:
                raise ValueError('本次目录含链接，拒绝清理或预算遍历')
            if stat.S_ISDIR(info.st_mode):
                entries.extend(inventory(entry.path))
            elif stat.S_ISREG(info.st_mode):
                entries.append({'path': str(Path(entry.path).absolute()), 'bytes': info.st_size})
    return entries


def write(path, record):
    Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')


def remove_owned(root, owner_root):
    root, owner_root = Path(root).resolve(), Path(owner_root).resolve()
    if root == owner_root or not root.is_relative_to(owner_root):
        raise ValueError('清理目标必须位于本次独占目录内')
    files = inventory(root)
    if root.exists():
        shutil.rmtree(root)
    return dict(files=len(files), bytes=sum(f['bytes'] for f in files), path=str(root))


def environment(work, dependencies, direct):
    env = dict(os.environ)
    for key in tuple(env):
        if key.lower() in ('http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'):
            del env[key]
    if direct:
        env['NO_PROXY'] = '*'
    else:
        env.update(HTTP_PROXY='http://127.0.0.1:7897', HTTPS_PROXY='http://127.0.0.1:7897')
    env.update(PYTHONPATH=str(dependencies), PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', PIP_NO_CACHE_DIR='1',
               PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK='1')
    for key, name in [('TEMP', 'temp'), ('TMP', 'temp'), ('HF_HOME', 'huggingface'),
                      ('XDG_CACHE_HOME', 'cache'), ('TORCH_HOME', 'torch'),
                      ('MODELSCOPE_CACHE', 'modelscope')]:
        value = Path(work) / name
        value.mkdir(parents=True, exist_ok=True)
        env[key] = str(value)
    return env


def supervised(command, env, roots, log, deadline, record, static_files=None):
    began = time.monotonic()
    free_at_start = psutil.disk_usage(str(Path(roots[0]).anchor)).free
    with Path(log).open('w', encoding='utf-8') as stream:
        child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                if time.time() >= deadline:
                    raise TimeoutError('达到本轮固定截止时间')
                # pip 会移动其暂存目录；安装期用已核对的 wheel 展开上界监督，避免读取正在消失的路径。
                entries = ([f for root in roots for f in inventory(root)]
                           if static_files is None else static_files)
                size = (sum(f['bytes'] for f in entries) if static_files is None else
                        max(0, free_at_start - psutil.disk_usage(str(Path(roots[0]).anchor)).free))
                record['peak_files'] = max(record.get('peak_files', 0), len(entries))
                record['peak_disk_bytes'] = max(record.get('peak_disk_bytes', 0), size)
                if len(entries) > 12400 or size > 9 * 2**30:
                    raise RuntimeError('超过已批准文件或磁盘上限')
                if child.poll() is not None:
                    break
                process = psutil.Process(child.pid)
                family = [process, *process.children(recursive=True)]
                rss = sum(p.memory_info().rss for p in family if p.is_running())
                record['peak_group_rss_bytes'] = max(record.get('peak_group_rss_bytes', 0), rss)
                if rss > 8 * 2**30 or psutil.virtual_memory().percent > 80:
                    raise MemoryError('超过已批准进程组或整机内存上限')
                time.sleep(1)
            if child.returncode != 0:
                raise RuntimeError(f'材料命令失败，退出码 {child.returncode}，日志 {log}')
        finally:
            if child.poll() is None:
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                child.wait()
            record.update(exit_code=child.returncode, seconds=time.monotonic() - began)


def acquire(args):
    work, deps, weights, report = map(Path, (args.work, args.dependencies, args.weights, args.report))
    # 本次目录必须新建，防止清理共享材料或其他批次产物。
    for root in (work, deps, weights):
        root.mkdir(parents=True, exist_ok=False)
    report.mkdir(parents=True, exist_ok=True)
    record = dict(started_utc=datetime.now(timezone.utc).isoformat(), deadline_epoch=args.deadline,
                  revision=REVISION, packages=PACKAGES, requested_files=FILES, phases={}, cleanup=[])
    write(report / 'materials.json', record)
    roots = [work, deps, weights]
    success = False
    installed = False
    try:
        stage = record['phases']['install'] = {}
        env = environment(work, deps, direct=False)
        stage_deadline = min(args.deadline, time.time() + 600)
        wheels = work / 'wheels'
        wheels.mkdir()
        supervised([sys.executable, '-B', '-m', 'pip', 'download', '--index-url',
                    'https://pypi.org/simple', '--dest', str(wheels), '--no-deps',
                    '--no-cache-dir', '--only-binary=:all:', *PACKAGES], env, roots,
                   report / 'package-download.log', stage_deadline, stage)
        members = []
        for wheel in wheels.glob('*.whl'):
            with zipfile.ZipFile(wheel) as archive:
                members.extend(archive.infolist())
        if len(list(wheels.glob('*.whl'))) != 11 or sum(p.stat().st_size for p in wheels.glob('*.whl')) != 27279575:
            raise ValueError('固定依赖包数量或总大小改变')
        # pip 的 target 暂存与目标位于同一卷，目录通过 rename 移动；加 100 个元数据及入口余量。
        if wheels.anchor != deps.anchor:
            raise ValueError('依赖暂存与安装目标必须位于同一卷')
        bound = len(members) + 100
        if bound > 12400 or 2 * sum(i.file_size for i in members) > 2**30:
            raise ValueError('依赖展开上界超过安装预算')
        stage['transient_file_upper_bound'] = bound
        stage['installation_disk_measurement'] = '整卷剩余空间变化，包含同时写入的其他任务，采取保守上限'
        supervised([sys.executable, '-B', '-m', 'pip', 'install', '--no-index', '--find-links',
                    str(wheels), '--target', str(deps), '--no-deps', '--no-compile',
                    '--no-cache-dir', '--only-binary=:all:', *PACKAGES], env, roots,
                   report / 'install.log', stage_deadline, stage,
                   static_files=[{'bytes': 0}] * bound)
        entries = inventory(deps)
        if len(entries) > 12000 or sum(f['bytes'] for f in entries) > 2**30:
            raise RuntimeError('隔离依赖超过批准范围')
        record['dependencies'] = dict(files=len(entries), bytes=sum(f['bytes'] for f in entries))
        installed = True
        record['cleanup'].append(remove_owned(work / 'temp', work))
        record['cleanup'].append(remove_owned(wheels, work))
        write(report / 'materials.json', record)
        env = environment(work, deps, direct=True)
        stage = record['phases']['download'] = {}
        supervised([sys.executable, '-B', __file__, 'sdk', '--weights', str(weights)], env,
                   roots, report / 'download.log', min(args.deadline, time.time() + 2700), stage)
        for name, size in FILES.items():
            if (weights / name).stat().st_size != size:
                raise ValueError('官方材料大小不符：' + name)
        record['model'] = dict(files=len(FILES), bytes=sum(FILES.values()), path=str(weights))
        success = True
    finally:
        record['success'] = success
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        record['work_before_cleanup'] = inventory(work)
        for path in list(work.iterdir()):
            if path.is_dir():
                record['cleanup'].append(remove_owned(path, work))
        if not installed:
            record['failed_dependencies'] = inventory(deps)
            remove_owned(deps, deps.parent)
        if not success:
            record['failed_model_files'] = inventory(weights)
            remove_owned(weights, weights.parent)
        write(report / 'materials.json', record)
    print(json.dumps(record, ensure_ascii=False, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    download = commands.add_parser('sdk')
    download.add_argument('--weights', required=True)
    acquire_parser = commands.add_parser('acquire')
    for name in ('work', 'dependencies', 'weights', 'report'):
        acquire_parser.add_argument('--' + name, required=True)
    acquire_parser.add_argument('--deadline', required=True, type=float)
    args = parser.parse_args()
    if args.command == 'sdk':
        from modelscope.hub.snapshot_download import snapshot_download
        snapshot_download('facebook/sam3', revision=REVISION, local_dir=args.weights,
                          allow_patterns=list(FILES), max_workers=2)
    else:
        acquire(args)


if __name__ == '__main__':
    main()
