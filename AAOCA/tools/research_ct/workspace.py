"""Resolve research workspace locations without embedding patient-specific paths."""
import json
import os
from pathlib import Path


def workspace_root(start=__file__):
    configured = os.environ.get('CHD_PROJECT_ROOT')
    if configured:
        return Path(configured).expanduser().resolve()
    for parent in Path(start).resolve().parents:
        if (parent / 'CT总清单.json').is_file() and (parent / 'results').is_dir():
            return parent
    return Path.cwd().resolve()


def work_directory(root):
    return Path(os.environ.get('CHD_WORK_DIR') or
                (Path(root) / 'results' / 'docs' / 'chd_ct_prepare_20261017')).resolve()


def reference_directory(root):
    config = Path(root) / 'project.local.json'
    settings = json.loads(config.read_text(encoding='utf-8')) if config.is_file() else {}
    value = os.environ.get('CHD_REFERENCE_ROOT') or settings.get('chd_reference_root')
    return Path(value).expanduser().resolve() if value else Path(root) / 'reference_CHD'


def relocate(value, root):
    """Translate known moved paths using a local map; leave evidence files unchanged."""
    config = Path(root) / 'project.local.json'
    settings = json.loads(config.read_text(encoding='utf-8')) if config.is_file() else {}
    mappings = sorted(settings.get('path_mappings', []), key=lambda r:len(r['old']), reverse=True)
    def visit(obj):
        if isinstance(obj, str):
            normalized = obj.replace('\\', '/')
            for mapping in mappings:
                old = mapping['old'].replace('\\', '/').rstrip('/')
                if normalized.casefold() == old.casefold() or normalized.casefold().startswith(old.casefold() + '/'):
                    return mapping['new'].replace('\\', '/').rstrip('/') + normalized[len(old):]
            return obj
        if isinstance(obj, list):return [visit(x) for x in obj]
        if isinstance(obj, dict):return {visit(k):visit(v) for k,v in obj.items()}
        return obj
    return visit(value)
