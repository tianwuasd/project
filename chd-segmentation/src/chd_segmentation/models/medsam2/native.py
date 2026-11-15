from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ...artifacts.store import read_json
from ...observability.events import event, stage
from ...preprocessing.offline import case_pointer
from ...preprocessing.offline_store import FORMAT, OfflineTransform, case_directory, recipe, save_case, validate_storage
from ...preprocessing.validation import SourceReader


CONTRACT = 'medsam2-native-window-uint8-v1'


def window_image(image, window):
    low, high = window
    if not np.isfinite(window).all() or low >= high:
        raise ValueError('MedSAM2 CT窗范围需要两个递增的有限数值')
    clipped = np.clip(image, low, high)
    minimum, maximum = float(clipped.min()), float(clipped.max())
    if maximum <= minimum:
        raise ValueError('MedSAM2 CT窗内没有有效灰度变化')
    return ((clipped-minimum)/(maximum-minimum)*255.).astype(np.uint8)


def resolve_window(config, prepared):
    if config['gray_window'] is not None:
        values = config['gray_window']
    else:
        stats = prepared['plan']['foreground_intensity_properties_per_channel']['0']
        values = [stats['percentile_00_5'], stats['percentile_99_5']]
    if not isinstance(values, (list, tuple)) or len(values) != 2 or any(
            type(v) not in (int, float) or not np.isfinite(v) for v in values) or values[0] >= values[1]:
        raise ValueError('MedSAM2 CT窗范围需要两个递增的有限数值')
    return list(map(float, values))


def native_recipe(prepared, config, *, window, derived_window):
    source = recipe(prepared, config)
    statistics = dict(kind='explicit')
    if derived_window:
        statistics = dict(kind='training_percentiles', fold=prepared['fold'],
            statistics_source=config['preprocessing']['statistics_source'],
            statistics_inputs=source['statistics_inputs'], fit_ids=prepared['plan'].get('fit_ids', []))
    return dict(format=FORMAT, transform=CONTRACT, dataset=prepared['dataset'],
        scope=prepared['scope'], technical=prepared['technical'], labels=prepared['dataset_json'],
        gray_window=list(window), window_source=statistics,
        versions={name: version(name) for name in ('numpy', 'nibabel')})


def produce_native_case(prepared, config, root, name, key, recipe_identity, *, window):
    torch.set_num_threads(1)
    began = perf_counter()
    row = next(r for r in prepared['manifest']['cases'] if r['case_id'] == name)
    event(config['run_dir'], 'preprocess_case_started', case_id=name, method=CONTRACT)
    reader = SourceReader(prepared, config['resources']['reserve_ram_bytes'],
        with_label=row.get('label') is not None, log_dir=config['run_dir'], logging=config['logging'])
    reader.console = True
    data, label, image, spacing, roi, read_seconds = reader.read_case(row)
    with stage(config, 'native_gray_window', case_id=name):
        pixels = window_image(data.transpose(2, 1, 0), window)[None]
        entry = dict(case_id=name, image=pixels,
            label=np.ascontiguousarray(label.transpose(2, 1, 0)) if label is not None else None,
            unknown_classes=row['unknown_classes'], source_shape_xyz=list(image.shape),
            source_affine=image.affine.tolist(), source_units=image.header.get_xyzt_units()[0],
            spacing_xyz=spacing.tolist(), roi_xyz=roi, properties=dict(class_locations={}),
            medsam_native=True, medsam_windowed=True, gray_window=list(window),
            timing=dict(read_decompress_seconds=read_seconds, native_seconds=perf_counter()-began))
    with stage(config, 'preprocess_case_save', case_id=name):
        record = save_case(case_directory(root, key), key, recipe_identity, entry)
    event(config['run_dir'], 'preprocess_case_completed', case_id=name, method=CONTRACT,
        seconds=perf_counter()-began, bytes=sum(v['stamp'][0] for v in record['files'].values()))
    return name, case_pointer(record)


class NativeCases:
    def __init__(self, prepared, with_label=True, log_dir=None, logging=None, reserve_ram_bytes=4*2**30):
        validate_storage(prepared, expected_transform=CONTRACT)
        self.prepared, self.with_label = prepared, with_label
        self.rows = {row['case_id']: row for row in prepared['manifest']['cases']}
        self.reader = OfflineTransform(prepared, with_label)
        self.window = read_json(Path(prepared['storage']['root'])/'catalog.json')['recipe']['gray_window']
        self.current_name, self.current = None, None

    def get(self, name):
        self.reader.check_sources(self.rows[name])
        if name == self.current_name:
            return self.current
        entry = self.reader(self.rows[name])
        if not entry.get('medsam_native') or not entry.get('medsam_windowed') or entry['image'].dtype != np.uint8:
            raise ValueError('MedSAM2需要原生uint8灰度窗输入')
        if entry['gray_window'] != self.window:
            raise ValueError('MedSAM2病例灰度窗与完整配方不同')
        self.current_name, self.current = name, entry
        return entry

    def clear(self):
        self.current_name, self.current = None, None


def restore_native(probability, entry):
    label = probability.argmax(0).astype(np.uint8).transpose(2, 1, 0)
    result = np.zeros(entry['source_shape_xyz'], dtype=np.uint8)
    region = tuple(slice(low, high) for low, high in entry['roi_xyz'])
    if result[region].shape != label.shape:
        raise ValueError('MedSAM2导出尺寸与源图范围不同')
    result[region] = label
    return result
