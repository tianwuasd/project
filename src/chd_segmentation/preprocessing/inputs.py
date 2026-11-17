from copy import deepcopy
from pathlib import Path

import numpy as np

from ..artifacts.store import read_json


class OfflineDataset:
    def __init__(self, root, catalog, cases, dataset):
        self.root, self.catalog, self.cases, self.dataset = root, catalog, cases, dataset

    def open(self, case_id, *, with_label=True):
        pointer = self.cases[case_id]
        directory = self.root / pointer['key'][:16]
        record = read_json(directory / 'case.json')
        entry = deepcopy(record['entry'])
        entry['image'] = np.load(directory / 'image.npy', mmap_mode='r', allow_pickle=False)
        entry['label'] = np.load(directory / 'label.npy', mmap_mode='r', allow_pickle=False) if with_label else None
        locations = {}
        if with_label:
            with np.load(directory / 'locations.npz', allow_pickle=False) as values:
                for i, key in enumerate(record['location_keys']):
                    array = values[f'p{i}']
                    locations[tuple(key) if isinstance(key, list) else key] = array if array.size else []
        entry['properties']['class_locations'] = locations
        recipe = self.catalog['recipe']
        plan = recipe.get('plan', {})
        entry.update(processed_spacing=plan.get('configurations', {}).get('3d_fullres', {}).get('spacing',
                     entry.get('spacing_xyz', [])[::-1]),
                     axis_order=plan.get('transpose_forward', [0, 1, 2]),
                     transform=recipe['transform'], source_directory=str(directory),
                     intensity={'gray_window': recipe.get('gray_window'),
                                'statistics': plan.get('foreground_intensity_properties_per_channel')})
        return entry


def resolve_preprocessed(path, *, dataset, case_ids):
    path = Path(path).resolve(strict=True)
    if path.is_file():
        prepared = read_json(path)
        if prepared['dataset'] != dataset:
            raise ValueError('指定预处理数据集不一致')
        storage = prepared['storage']
        root, cases = Path(storage['root']).resolve(strict=True), storage['cases']
        catalog = read_json(root / 'catalog.json')
    else:
        root, cases = path, {}
        catalog = read_json(root / 'catalog.json')
        for key, pointer in catalog['cases'].items():
            record = read_json(root / key[:16] / 'case.json')
            cases[record['entry']['case_id']] = pointer
    if catalog['recipe']['dataset'] != dataset:
        raise ValueError('指定预处理数据集不一致')
    missing = set(case_ids) - cases.keys()
    if missing:
        raise ValueError('预处理数据缺少当前名单：' + ', '.join(sorted(missing)))
    return OfflineDataset(root, catalog, cases, dataset)
