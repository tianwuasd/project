import argparse
import json
import os
from pathlib import Path
import shutil
import stat
from .contracts import atomic_json, digest, read_json, sha256_file
from .observability.status import lock_active, operation_lock
from .storage_paths import assert_output_location


def _inventory(root):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Migration source must be a directory')
    for lock in root.rglob('operation.lock'):
        if lock.is_symlink():
            raise ValueError('Linked migration lock')
        if lock_active(lock):
            raise RuntimeError('Active operation found in migration source: ' + str(lock))
    rows = {}
    for path in [root, *sorted(root.rglob('*'))]:
        info = path.lstat()
        if path.is_symlink() or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise ValueError('Linked or special migration input: ' + str(path))
        if hasattr(os, 'getuid') and info.st_uid != os.getuid():
            raise ValueError('Migration requires ownership of every source entry: ' + str(path))
        if path.name == 'operation.lock' and lock_active(path):
            raise RuntimeError('Active operation found in migration source: ' + str(path))
        if path.is_file():
            before = path.stat()
            sha = sha256_file(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('Source changed during inventory: ' + str(path))
            rows[path.relative_to(root).as_posix()] = {'size': after.st_size, 'sha256': sha}
    if not rows:
        raise ValueError('Migration source is empty')
    return rows


def migration_plan(source, destination):
    source = Path(source).expanduser().resolve(strict=True)
    destination = assert_output_location(destination)
    if destination.exists():
        raise ValueError('Migration destination must not exist')
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Migration source and destination overlap')
    if hasattr(os, 'getuid') and source.stat().st_uid != os.getuid():
        raise ValueError('Migration source must be owned by the current user')
    inventory = _inventory(source)
    plan = {'schema_version': 1, 'source': str(source), 'destination': str(destination),
            'inventory': inventory, 'files': len(inventory),
            'bytes': sum(row['size'] for row in inventory.values()), 'source_deletion': False}
    plan['identity'] = digest(plan)
    return plan


def _immutable_contracts(source, inventory):
    contracts = {}
    excluded = {'attempts', '.input-cache', 'prediction-cache', '.runtime', 'code_snapshot'}
    hu_roots = set()
    for name in inventory:
        relative = Path(name)
        if excluded.intersection(relative.parts):
            continue
        if relative.name == 'request.json' and read_json(source / relative).get('method') == 'HU-marginal-v1':
            hu_roots.add(relative.parent)
    hu_names = {'request.json', 'complete.json', 'cohort.json', 'training_contract.json',
                'dataset.json', 'splits_final.json', 'partial_supervision.json'}
    for name, row in inventory.items():
        relative = Path(name)
        if excluded.intersection(relative.parts):
            continue
        immutable = relative.name in ('run.json', 'contract.json')
        hu_contract = relative.name in hu_names and any(root == relative.parent or root in relative.parents for root in hu_roots)
        if immutable or hu_contract:
            contracts[name] = row['sha256']
    return contracts


def apply_migration(plan_file, home, move=False):
    home = assert_output_location(home)
    plan = read_json(plan_file)
    if plan.get('schema_version') != 1 or plan.get('source_deletion') is not False:
        raise ValueError('Unsupported migration plan')
    if digest({key: value for key, value in plan.items() if key != 'identity'}) != plan['identity']:
        raise ValueError('Migration plan identity differs')
    with operation_lock(home, 'storage-migration'):
        current = migration_plan(plan['source'], plan['destination'])
        if current != plan:
            raise ValueError('Source content differs from approved migration plan')
        source = Path(plan['source']); target = Path(plan['destination'])
        if move and (os.name == 'nt' or str(source) != '/data_nas/zhangruichen/dataset1282_preprocessing'):
            raise ValueError('Move is limited to the exact owned legacy server process directory')
        # 复制失败时保留目标现场，源目录保持完整。
        target.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(target.parent).free < plan['bytes'] + 1024**2:
            raise ValueError('Insufficient destination disk space')
        target.mkdir()
        for name in plan['inventory']:
            old = source / name; new = target / name
            if old.is_symlink() or not old.is_file():
                raise ValueError('Source changed during migration: ' + name)
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, new)
        if _inventory(source) != plan['inventory'] or _inventory(target) != plan['inventory']:
            raise ValueError('Full migration size/hash verification failed')
        receipt = {**plan, 'verified': True,
                   'contracts': _immutable_contracts(source, plan['inventory'])}
        receipt['receipt_identity'] = digest(receipt)
        sidecar = target.parent / (target.name + '.relocation.json')
        if sidecar.exists():
            raise ValueError('Relocation receipt already exists')
        atomic_json(sidecar, receipt)
        atomic_json(target / '.storage-relocation.json', receipt)
        if move:
            if _inventory(source) != plan['inventory']:
                raise ValueError('Source changed before move deletion; source preserved')
            # 仅删除已确认的固定旧目录；复制验证已完成，源文件从不提前删除。
            if source.is_symlink() or source.resolve() != source or str(source) != '/data_nas/zhangruichen/dataset1282_preprocessing':
                raise ValueError('Legacy move source safety check failed')
            shutil.rmtree(source)
        return {'destination': str(target), 'sidecar': str(sidecar), 'files': plan['files'],
                'bytes': plan['bytes'], 'source_preserved': not move}


def load_relocation(path):
    path = Path(path).resolve()
    for parent in [path, *path.parents]:
        sidecar = parent / '.storage-relocation.json'
        if not sidecar.is_file():
            continue
        if sidecar.is_symlink():
            raise ValueError('Linked relocation receipt')
        receipt = read_json(sidecar)
        if receipt.get('schema_version') != 1 or receipt.get('verified') is not True:
            raise ValueError('Unverified relocation receipt')
        if digest({k: v for k, v in receipt.items() if k != 'receipt_identity'}) != receipt['receipt_identity']:
            raise ValueError('Relocation receipt identity differs')
        if Path(receipt['destination']).resolve() != parent:
            raise ValueError('Relocation receipt destination differs')
        assert_output_location(parent)
        for name, expected in receipt['contracts'].items():
            contract = parent / name
            if parent not in contract.resolve().parents or contract.is_symlink() or sha256_file(contract) != expected:
                raise ValueError('Relocated contract identity differs: ' + name)
        return receipt
    return None


def canonical_path(path):
    path = Path(path).resolve()
    relocation = load_relocation(path)
    if relocation is None:
        return str(path)
    return str(Path(relocation['source']) / path.relative_to(Path(relocation['destination'])))


def resolve_relocated_path(original, sidecar, contract_file):
    receipt = read_json(sidecar)
    if receipt.get('verified') is not True or receipt.get('schema_version') != 1:
        raise ValueError('Unverified relocation receipt')
    if digest({key: value for key, value in receipt.items() if key != 'receipt_identity'}) != receipt['receipt_identity']:
        raise ValueError('Relocation receipt identity differs')
    source = Path(receipt['source']).resolve()
    target = assert_output_location(receipt['destination'])
    contract = Path(contract_file).resolve(strict=True)
    relative_contract = contract.relative_to(target).as_posix()
    expected = receipt['contracts'].get(relative_contract)
    if expected is None or sha256_file(contract) != expected:
        raise ValueError('Original contract identity is not preserved')
    relative = Path(original).resolve().relative_to(source)
    relocated = target / relative
    selected = {name: row for name, row in receipt['inventory'].items()
                if name == relative.as_posix() or relative in Path(name).parents}
    if not selected:
        raise ValueError('Requested relocated input is absent from migration inventory')
    for name, row in selected.items():
        path = target / name
        if path.is_symlink() or target not in path.resolve().parents or path.stat().st_size != row['size'] or sha256_file(path) != row['sha256']:
            raise ValueError('Relocated input differs from recorded content: ' + name)
    return relocated


def main(argv=None):
    parser = argparse.ArgumentParser(description='Plan or apply a verified process-directory copy; preserve all source files')
    actions = parser.add_subparsers(dest='action', required=True)
    plan = actions.add_parser('plan')
    plan.add_argument('--source', required=True); plan.add_argument('--destination', required=True)
    plan.add_argument('--plan-file', required=True)
    apply = actions.add_parser('apply')
    apply.add_argument('--plan-file', required=True); apply.add_argument('--home', required=True)
    apply.add_argument('--move', action='store_true', help='Delete the exact legacy NAS process directory after verified copy and repeated lock checks')
    args = parser.parse_args(argv)
    if args.action == 'plan':
        result = migration_plan(args.source, args.destination)
        plan_file = assert_output_location(args.plan_file)
        if plan_file.exists():
            raise ValueError('Plan file already exists')
        atomic_json(plan_file, result)
    else:
        result = apply_migration(args.plan_file, args.home, args.move)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
