"""Configuration precedence: CLI > task config > machine config > release defaults."""
import os
from pathlib import Path
from .contracts import read_json
from .storage_paths import assert_output_location, validate_input_output_paths, default_server_work_root, SERVER_DATA_ROOT

PATH_KEYS = {'data_root', 'metadata_excel', 'work_root', 'environment'}


def resolve_config(overrides=None, local_file=None, code_root=None, task_file=None):
    root = Path(code_root).resolve()
    home = Path(local_file).resolve().parent if local_file else (root.parent.parent if root.parent.name=='code_versions' else root)
    config = read_json(root / 'configs/default.json')
    allowed = set(config)
    sources = []
    for path in [local_file, task_file]:
        if path and Path(path).is_file():
            source = Path(path).resolve()
            values = read_json(source)
            for key in PATH_KEYS:
                if values.get(key):
                    if key in ('work_root', 'environment'):
                        assert_output_location(values[key])
                    p = Path(values[key]).expanduser()
                    values[key] = str(p.resolve() if p.is_absolute() else (source.parent / p).resolve())
            sources.append(values)
    sources.append({k: v for k, v in (overrides or {}).items() if v is not None})
    for values in sources:
        unknown = set(values) - allowed
        if unknown:
            raise ValueError('Unknown configuration fields: ' + ', '.join(sorted(unknown)))
        config.update(values)
    if config['schema_version'] != 1:
        raise ValueError('Unsupported configuration schema')
    if config['data_root'] is None and os.name != 'nt':
        config['data_root'] = SERVER_DATA_ROOT
    for key in PATH_KEYS:
        if key in ('work_root', 'environment') and config[key]:
            assert_output_location(config[key])
        if config[key] == '':
            raise ValueError(f'Empty path: {key}')
        if config[key] is not None:
            config[key] = str(Path(config[key]).expanduser().resolve())
    config['work_root'] = config['work_root'] or (default_server_work_root() if os.name != 'nt' else str(home.parent / (home.name + '_work')))
    config['environment'] = config['environment'] or str(Path(config['work_root']) / 'environment')
    for name in ('cpu_fraction', 'ram_fraction'):
        if not 0 < config[name] <= 0.8:
            raise ValueError(f'{name} must be in (0, 0.8]')
    for name in ('read_concurrency', 'max_workers', 'chunk_slices', 'percentile_samples', 'visual_examples'):
        if not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if config['reserve_gib'] < 4 or config['resource_wait_seconds'] < 0:
        raise ValueError('reserve_gib >= 4 and resource_wait_seconds >= 0 are required')
    if config['connectivity'] not in (1, 2, 3):
        raise ValueError('connectivity must be 1, 2 or 3')
    if not isinstance(config.get('max_label_values',128),int) or not 2<=config.get('max_label_values',128)<=4096:
        raise ValueError('max_label_values must be an integer between 2 and 4096')
    validate_input_output_paths(config['data_root'], {key: config[key] for key in ('work_root', 'environment')})
    for name in ('work_root', 'environment'):
        out = Path(config[name])
        if out == root or root in out.parents:
            raise ValueError(f'{name} must be outside the source directory')
    return config
