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
        from .offline_store import record_identity
        if (record['case_key']!=pointer['key'] or record['recipe_identity']!=self.catalog['recipe_identity']
                or record['entry']['case_id']!=case_id or pointer.get('case_id',case_id)!=case_id
                or record_identity(record)!=pointer['identity']):
            raise ValueError('选定病例指针与实际病例、版本或配方不一致：'+case_id)
        entry = deepcopy(record['entry'])
        entry['image'] = np.load(directory / 'image.npy', mmap_mode='r', allow_pickle=False)
        if list(entry['image'].shape)!=pointer['shape_czyx']:
            raise ValueError('选定病例版本的数组尺寸不符：'+case_id)
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
        native = recipe['transform'].startswith('medsam2-native')
        entry.update(source_plan=deepcopy(plan), processed_spacing=plan.get('configurations', {}).get('3d_fullres', {}).get('spacing',
                     entry.get('spacing_xyz', [])[::-1] if native else None),
                     axis_order=plan.get('transpose_forward', [0, 1, 2] if native else None),
                     transform=recipe['transform'], source_directory=str(directory),
                     intensity={'kind': 'native_window_uint8' if native else 'ct_normalized', 'gray_window': recipe.get('gray_window'),
                                'statistics': plan.get('foreground_intensity_properties_per_channel')})
        return entry


def resolve_preprocessed(path, *, dataset, case_ids, storage=None):
    path = Path(path).resolve(strict=True)
    if path.is_file():
        if storage is not None:
            raise ValueError('预处理描述与显式storage不能重复指定')
        prepared = read_json(path)
        if prepared['dataset'] != dataset:
            raise ValueError('指定预处理数据集不一致')
        storage = prepared['storage']
        root, cases = Path(storage['root']).resolve(strict=True), storage['cases']
        catalog = read_json(root / 'catalog.json')
    elif storage is not None:
        root = path
        if root!=Path(storage['root']).resolve(strict=True):
            raise ValueError('选定预处理版本的根目录不同')
        cases = storage['cases']
        catalog = read_json(root/'catalog.json')
    else:
        root, cases = path, {}
        catalog = read_json(root / 'catalog.json')
        wanted = set(case_ids)
        index = catalog.get('case_index', {})
        for name in wanted & index.keys():
            cases[name] = catalog['cases'][index[name]]
        for key, pointer in catalog['cases'].items():
            if wanted <= cases.keys():
                break
            name = pointer.get('case_id')
            if name is None:
                # 旧catalog没有病例索引，只读小记录定位所需病例，不读取数组内容。
                name = read_json(root / key[:16] / 'case.json')['entry']['case_id']
            if name in wanted:
                cases[name] = pointer
    if catalog['recipe']['dataset'] != dataset:
        raise ValueError('指定预处理数据集不一致')
    if storage is not None and storage['recipe_identity']!=catalog['recipe_identity']:
        raise ValueError('选定预处理配方与catalog不一致')
    missing = set(case_ids) - cases.keys()
    if missing:
        raise ValueError('预处理数据缺少当前名单：' + ', '.join(sorted(missing)))
    return OfflineDataset(root, catalog, {name:cases[name] for name in case_ids}, dataset)
