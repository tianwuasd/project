import argparse
import json
import os
from pathlib import Path
import re

from .contracts import atomic_json, digest, read_json
from .observability.status import operation_lock
from .storage_migration import _immutable_contracts, _inventory, load_relocation
from .storage_paths import assert_output_location


def _adopt_unlocked(source, destination, experiment):
    if not re.fullmatch(r'xu-train-[0-9]+', experiment):
        raise ValueError('Expected a named Xu training experiment')
    receipt_path = destination / '.storage-relocation.json'
    if receipt_path.exists():
        raise ValueError('This process directory already has a relocation receipt')
    if os.path.exists(source):
        raise ValueError('Original process directory still exists; use verified copy migration')
    inventory = _inventory(destination)
    run_relative = Path('experiments') / experiment / 'run.json'
    if run_relative.as_posix() not in inventory:
        raise ValueError('Xu training contract is absent from the moved directory')
    run = read_json(destination / run_relative)
    if run.get('protocol') != 'xu-model-run-4':
        raise ValueError('Unexpected Xu training protocol')
    original_prepared = Path(run['prepared'])
    prepared_relative = original_prepared.relative_to(source)
    prepared = destination / prepared_relative
    if not prepared.is_dir():
        raise ValueError('Moved preparation is missing')
    request = read_json(prepared / 'request.json')
    complete = read_json(prepared / 'complete.json')
    request_identity = digest(request)
    if (request.get('method') != 'HU-marginal-v1' or complete.get('state') != 'completed'
            or complete.get('run_dir') != str(original_prepared)
            or complete.get('request_digest') != request_identity
            or run.get('request') != request_identity):
        raise ValueError('Moved preparation and Xu training contract disagree')
    cases = request['cases']
    if (complete['train_cases'] != sum(row['role'] == 'train' for row in cases.values())
            or complete['development_cases'] != sum(row['role'] == 'development' for row in cases.values())):
        raise ValueError('Moved preparation case counts disagree')
    committed_processes = set()
    commits = prepared / 'commits'
    for marker_path in sorted(commits.glob('*.json')):
        marker = read_json(marker_path)
        if marker['contract'] != request_identity or marker['job']['id'] != marker_path.stem:
            raise ValueError('Moved preparation job contract differs: ' + marker_path.name)
        if marker_path.stem.startswith('process-'):
            committed_processes.add(marker_path.stem.removeprefix('process-'))
        for relative, checksum in marker['artifacts'].items():
            artifact = (prepared / relative).resolve(strict=True)
            if artifact != prepared and prepared not in artifact.parents:
                raise ValueError('Preparation artifact escapes the moved directory')
            key = artifact.relative_to(destination).as_posix()
            if key not in inventory or inventory[key]['sha256'] != checksum:
                raise ValueError('Moved preparation artifact differs: ' + key)
    if committed_processes != set(cases):
        raise ValueError('Moved preparation has missing or additional committed cases')
    plan = {'schema_version': 1, 'source': str(source), 'destination': str(destination),
            'inventory': inventory, 'files': len(inventory),
            'bytes': sum(row['size'] for row in inventory.values()), 'source_deletion': False}
    plan['identity'] = digest(plan)
    receipt = {**plan, 'verified': True,
               'contracts': _immutable_contracts(destination, inventory)}
    receipt['receipt_identity'] = digest(receipt)
    atomic_json(receipt_path, receipt)
    load_relocation(destination)
    return {'destination': str(destination), 'experiment': experiment,
            'files': plan['files'], 'bytes': plan['bytes'],
            'committed_cases': len(committed_processes), 'receipt': str(receipt_path)}


def adopt_moved_process_root(source, destination, experiment):
    source = Path(source).expanduser().resolve()
    destination = assert_output_location(destination)
    if not source.is_absolute() or not destination.is_dir() or source == destination:
        raise ValueError('Expected distinct absolute source and existing destination directories')
    if source in destination.parents or destination in source.parents:
        raise ValueError('Source and destination overlap')
    with operation_lock(destination.parent, 'storage-adoption'):
        return _adopt_unlocked(source, destination, experiment)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Verify an already moved Xu process directory and record its original paths')
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--experiment', required=True)
    args = parser.parse_args(argv)
    result = adopt_moved_process_root(args.source, args.destination, args.experiment)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
