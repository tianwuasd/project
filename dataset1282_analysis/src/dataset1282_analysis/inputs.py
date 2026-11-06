"""A manifest row is one research image/label pair, independent of historical copies."""
from pathlib import Path, PureWindowsPath
import re
from .contracts import CaseInput, read_json


def load_cases(config):
    root = Path(config['data_root']).resolve()
    rows = read_json(root / 'manifest.json')
    if not isinstance(rows, list):
        raise ValueError('Manifest must be a list of case records')
    cases, seen = [], set()
    for row in rows:
        name = row['case_id']
        group = row['collection']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in seen:
            raise ValueError('Invalid or duplicated case_id')
        if group not in ('native', 'review_required'):
            raise ValueError('Unsupported collection')
        seen.add(name)
        paths = [root/group/'images'/(name+'_0000.nii.gz'), root/group/'labels'/(name+'.nii.gz')]
        for path in paths:
            if path.resolve() != path or root not in path.resolve().parents:
                raise ValueError('Linked or escaping input path')
        expected = {}
        for output in row.get('outputs', []):
            # A relocated Windows manifest remains usable on a Linux server.
            basename = PureWindowsPath(output.get('path', '')).name
            for kind, path in zip(('image', 'label'), paths):
                if basename == path.name and output.get('sha256'):
                    expected[kind] = output['sha256']
        cases.append(CaseInput(name, str(row['source_key']), group, str(paths[0]), str(paths[1]), expected))
    return sorted(cases, key=lambda c: c.case_id)


def inventory(cases):
    from collections import Counter
    missing, total = [], 0
    for case in cases:
        for kind in ('image', 'label'):
            path = Path(getattr(case, kind))
            if path.is_file(): total += path.stat().st_size
            else: missing.append({'case_id': case.case_id, 'kind': kind, 'path': str(path)})
    return {'case_count': len(cases), 'collections': dict(Counter(c.collection for c in cases)),
            'file_count_expected': 2 * len(cases), 'total_bytes': total, 'missing': missing,
            'denominator': 'manifest image/label pairs; extracted/history copies excluded'}
