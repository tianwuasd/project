import hashlib
import json
import os
from pathlib import Path
import stat
import time

from ..observability.locking import try_lock, unlock
from ..observability.runlog import write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def writable_output(value):
    """校验生成目录，数据专用目录 /data_nas 不接收新的过程文件。"""
    path = Path(value).expanduser().resolve()
    for name in (str(value).replace('\\', '/'), path.as_posix()):
        if name == '/data_nas' or name.startswith('/data_nas/'):
            raise ValueError('过程产物请写入 workspace；/data_nas 仅用于读取数据集：'+str(path))
    return path


def owned_file(root, relative, *, directory=False):
    path = root / relative
    for item in (path, *path.parents):
        if item == root.parent:
            break
        if item.exists() or item.is_symlink():
            info = item.lstat()
            if item.is_symlink() or (os.name == 'nt' and info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                raise ValueError('清理路径包含链接：'+str(item))
    if not path.resolve().is_relative_to(root):
        raise ValueError('清理路径越界：'+str(path))
    if path.exists() and not (path.is_dir() if directory else path.is_file()):
        raise ValueError('路径类型与登记内容不一致：'+str(path))
    return path


def identifier(value):
    if not isinstance(value, str) or not value or value in ('.', '..') or '/' in value or '\\' in value:
        raise ValueError('病例标识不能包含路径：'+str(value))
    return value


def inventory(root):
    files = [p for p in root.rglob('*') if p.is_file() and p != root/'cleanup.json']
    return dict(files=len(files), logical_bytes=sum(p.stat().st_size for p in files),
                excludes=['cleanup.json'], measurement='logical_file_sizes')


def candidates(root):
    """只列出已完成病例的可重建任务文件，数据、日志和审计样本全部保留。"""
    summary = read_json(root/'summary.json')
    status = read_json(root/'status.json')
    if status.get('state') != 'completed':
        raise ValueError('只精简已完成的预处理运行；失败和未完成任务保持原状')
    removable = []
    directories = []
    if (root/'signature.json').is_file():
        if summary.get('failed_cases') or summary.get('audit_only'):
            raise ValueError('预处理存在失败或仅完成审计，暂不精简')
        manifest = read_json(root/'manifests/imagechd_input.json')
        signature = read_json(root/'signature.json')
        if summary.get('protocol') != manifest['protocol'] or manifest['identity'] != signature['identity']:
            raise ValueError('预处理来源与汇总不一致')
        rows = {identifier(r['case_id']): r for r in manifest['cases']}
        for fold in summary['requested_folds']:
            if fold not in (0, 1, 2):
                raise ValueError('未知预处理折')
            folder = f'fold_{fold}'
            contract = read_json(root/folder/'training_contract.json')['preprocessing_contract']
            for cid in summary['requested_case_ids']:
                identifier(cid)
                marker = read_json(root/folder/'completion'/f'{cid}.json')
                required = {cid+'.b2nd', cid+'_seg.b2nd', cid+'.pkl'}
                if (not marker.get('success') or marker.get('case_id') != cid
                        or marker.get('contract') != contract
                        or marker.get('source_hashes') != rows[cid]['source_hashes']
                        or set(marker.get('file_sha256', {})) != required):
                    raise ValueError('病例完成记录不一致：'+cid)
                for name in required:
                    if not owned_file(root, f'{folder}/data/{name}').is_file():
                        raise ValueError('已完成病例缺少数据：'+cid)
                removable.extend([f'tasks/{folder}/{cid}.json', f'{folder}/results/{cid}.json',
                                  f'{folder}/results/{cid}.progress.json', f'{folder}/completion/{cid}.json.partial'])
                for staging in (root/folder/'staging').glob(cid+'-*'):
                    if not staging.name[len(cid)+1:].isdigit():
                        continue
                    relative = staging.relative_to(root).as_posix()
                    owned_file(root, relative, directory=True)
                    names = {name+suffix for name in required for suffix in ('', '.partial')}
                    children = list(staging.iterdir())
                    if any(p.name not in names or not p.is_file() for p in children):
                        raise ValueError('暂存目录包含未登记内容：'+str(staging))
                    removable.extend(p.relative_to(root).as_posix() for p in children)
                    directories.append(relative)
        for path in (root/'audit/cases').glob('*.json'):
            row = read_json(path)
            cid = identifier(row['case_id'])
            if not row.get('success') or path.stem != cid:
                raise ValueError('审计尚未完成：'+cid)
            removable.append(f'tasks/audit/{cid}.json')
    elif (root/'manifest.json').is_file():
        manifest = read_json(root/'manifest.json')
        if (summary.get('status') != 'completed' or summary.get('failed')
                or summary.get('manifest_identity') != manifest['identity']):
            raise ValueError('MedSAM2 清单与完成汇总不一致')
        rows = {identifier(r['case_id']): r for r in manifest['cases']}
        if set(summary['completed']) != set(rows):
            raise ValueError('MedSAM2 完成病例不完整')
        for cid, identity in summary['completed'].items():
            marker = read_json(root/'cases'/cid/'complete.json')
            if (marker['identity'] != identity or marker.get('case_id') != cid
                    or marker.get('source_hashes') != rows[cid]['source_hashes']
                    or set(marker['files']) != {'ct.npy', 'labels.npy'}):
                raise ValueError('MedSAM2 病例完成记录不一致：'+cid)
            for name in ('ct.npy', 'labels.npy'):
                if not owned_file(root, f'cases/{cid}/{name}').is_file():
                    raise ValueError('MedSAM2 已完成病例缺少数组：'+cid)
            removable.extend([f'jobs/{cid}-task.json', f'jobs/{cid}-result.json',
                              f'jobs/{cid}-result.progress.json', f'cases/{cid}/ct.partial',
                              f'cases/{cid}/labels.partial', f'cases/{cid}/complete.json.partial'])
    else:
        raise ValueError('指定目录不是已知的预处理运行')
    # 原子 JSON 写出的遗留副本只针对同一批已确认可删除的文件。
    removable += [name+'.partial' for name in list(removable)]
    paths = [owned_file(root, name) for name in sorted(set(removable))]
    return [p for p in paths if p.is_file()], directories


def compact_prepared(output, *, apply=False):
    """新旧运行共用的精简入口；默认只读，应用时取得原任务的互斥锁。"""
    root = Path(output).resolve(strict=True)
    if apply:
        writable_output(root)
    lock = root/'.run.lock' if (root/'signature.json').is_file() else root.with_name(root.name+'.prepare.lock')
    # 只使用原运行已存在的锁，不为任意输入目录创建文件。
    with lock.open('r+b' if apply else 'rb') as stream:
        if not try_lock(stream):
            raise RuntimeError('预处理运行仍被其他进程占用，未清理任何文件')
        try:
            began = time.monotonic()
            paths, directories = candidates(root)
            before = inventory(root)
            records = []
            for path in paths:
                info = path.stat()
                records.append(dict(path=path.relative_to(root).as_posix(), bytes=info.st_size,
                                    mtime_ns=info.st_mtime_ns, inode=info.st_ino))
            result = dict(schema=1, policy='prepared-essential-v1', applied=apply,
                          scope='completed_preprocessing_only', cleanup_state='planned',
                          before=before, candidates=records, removed_files=0, removed_logical_bytes=0,
                          staging_directories=directories,
                          retained=['training_data', 'completion_markers', 'audit_samples', 'audit_records',
                                    'manifests', 'fold_plans', 'source_code', 'logs', 'resources'])
            if apply:
                write_json(root/'cleanup.json', result)
                for record in records:
                    path = owned_file(root, record['path'])
                    info = path.stat()
                    if (info.st_size, info.st_mtime_ns, info.st_ino) != (record['bytes'], record['mtime_ns'], record['inode']):
                        raise ValueError('精简期间文件发生变化：'+str(path))
                    path.unlink()
                    result['removed_files'] += 1
                    result['removed_logical_bytes'] += record['bytes']
                for relative in directories:
                    owned_file(root, relative, directory=True).rmdir()
            result.update(cleanup_state='completed' if apply else 'planned', after=inventory(root),
                          elapsed_seconds=time.monotonic()-began)
            if apply:
                write_json(root/'cleanup.json', result)
            return result
        finally:
            unlock(stream)


def verify_snapshot(output):
    """恢复继续运行本任务冻结的生产者，不把新默认配置写入旧合同。"""
    root = Path(output).resolve(strict=True)
    signature = read_json(root/'signature.json')
    record = {k: v for k, v in signature.items() if k != 'identity'}
    identity = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
    if identity != signature['identity']:
        raise ValueError('旧预处理签名不完整')
    code = root/'code'
    for name, expected in signature['producer_sha256'].items():
        path = owned_file(code, name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('冻结的预处理代码已变化：'+name)
    return code
