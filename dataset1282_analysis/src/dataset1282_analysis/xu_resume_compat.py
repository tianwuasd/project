from pathlib import Path, PurePosixPath
import re

from .contracts import atomic_json, digest, read_json, sha256_file


AUDIT_PATH = Path(__file__).resolve().parents[2]/'configs/xu-resume-compat.json'
SCHEMA = 'xu-resume-compat-1'


def _source_map(value):
    if not isinstance(value, dict) or not value:
        raise ValueError('Expected a complete nonempty source SHA256 mapping')
    for name, checksum in value.items():
        if (not isinstance(name, str) or not name or '\\' in name
                or PurePosixPath(name).is_absolute() or ':' in name
                or any(part in ('', '.', '..') for part in name.split('/'))
                or not isinstance(checksum, str) or re.fullmatch('[0-9a-f]{64}', checksum) is None):
            raise ValueError('Invalid source path or SHA256 in compatibility mapping')


def _audit(path):
    audit = read_json(path)
    if (set(audit) != {'schema', 'transitions'} or audit['schema'] != SCHEMA
            or not isinstance(audit['transitions'], list)):
        raise ValueError('Invalid source compatibility audit schema')
    identifiers = set()
    for edge in audit['transitions']:
        if (set(edge) != {'id', 'review', 'sources'} or not isinstance(edge['id'], str)
                or not edge['id'].strip() or edge['id'] in identifiers
                or not isinstance(edge['review'], str) or not edge['review'].strip()
                or not isinstance(edge['sources'], list) or not edge['sources']):
            raise ValueError('Invalid source compatibility transition')
        identifiers.add(edge['id'])
        for maps in edge['sources']:
            if (set(maps) != {'path_prefix', 'stored', 'current'}
                    or maps['path_prefix'] not in ('', 'src/dataset1282_analysis/')):
                raise ValueError('Invalid source compatibility consumer')
            _source_map(maps['stored'])
            _source_map(maps['current'])
    return audit


def _matching_transition(stored_map, current_map, path_prefix, audit_path):
    _source_map(stored_map)
    _source_map(current_map)
    if path_prefix not in ('', 'src/dataset1282_analysis/'):
        raise ValueError('Unknown source compatibility path prefix')
    if not audit_path.is_file():
        return None
    for edge in _audit(audit_path)['transitions']:
        for maps in edge['sources']:
            if (maps['path_prefix'] == path_prefix and maps['stored'] == stored_map
                    and maps['current'] == current_map):
                return edge['id']
    return None


def source_compatible(stored_map, current_map, path_prefix='', *, audit_path=AUDIT_PATH):
    if stored_map == current_map:
        return True
    return _matching_transition(stored_map, current_map, path_prefix, Path(audit_path)) is not None


def preserve_contract(stored, current, out, *, field='implementation', path_prefix='',
                      audit_path=AUDIT_PATH):
    if stored == current:
        return stored
    if (field not in stored or field not in current
            or {k: v for k, v in stored.items() if k != field}
            != {k: v for k, v in current.items() if k != field}):
        raise ValueError('Use --resume with identical inputs and model settings')
    audit_path = Path(audit_path)
    audit_sha = sha256_file(audit_path) if audit_path.is_file() else None
    transition = _matching_transition(stored[field], current[field], path_prefix, audit_path)
    if transition is None:
        raise ValueError('Use --resume with identical inputs and audited implementation changes')
    if sha256_file(audit_path) != audit_sha:
        raise ValueError('Source compatibility audit changed while checking the contract')
    evidence = dict(schema=SCHEMA, transition=transition, field=field, path_prefix=path_prefix,
                    frozen_contract=digest(stored), execution_contract=digest(current),
                    stored_sources=stored[field], execution_sources=current[field],
                    audit_sha256=audit_sha)
    path = Path(out)/'.runtime/resume-compat'/(digest(evidence)+'.json')
    if path.exists():
        if read_json(path) != evidence:
            raise ValueError('Source compatibility execution evidence changed')
    else:
        atomic_json(path, evidence)
    return stored


def assert_execution_sources(expected, current, audit_sha256=None, *, audit_path=AUDIT_PATH):
    if expected != current:
        raise ValueError('Implementation changed while the pipeline was running')
    if audit_sha256 is not None and sha256_file(audit_path) != audit_sha256:
        raise ValueError('Source compatibility audit changed while the pipeline was running')
