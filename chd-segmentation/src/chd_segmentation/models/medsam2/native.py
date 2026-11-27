from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ...artifacts.store import read_json
from ...observability.events import event, stage
from ...preprocessing.offline import case_pointer, worker_cancel, case_guard
from ...preprocessing.offline_store import FORMAT, OfflineTransform, case_directory, recipe, save_case, validate_storage, begin_case
from ...preprocessing.validation import SourceReader
from ...models.stunet.streaming import BlockWorkspace, read_sources
from ...training.memory_guard import MemoryGuard


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
    source = recipe(prepared, {**config, 'preprocessing': {**config['preprocessing'], 'storage_format': 'npy'}})
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
    native = deepcopy(prepared)
    native['plan']['transpose_forward'] = [0, 1, 2]
    reader.prepared = native
    directory = case_directory(root, key)
    pending = begin_case(directory, key, recipe_identity)
    with case_guard(config) as guard:
        image, label_header, spacing, roi = reader.read_metadata(row)
        shape = [high-low for low, high in roi]
        whole_bytes = int(np.prod(shape))*32
        snapshot = guard.snapshot()
        available = snapshot['budget']['headroom_bytes']
        mode = config['preprocessing'].get('execution', 'auto')
        if mode == 'auto' and whole_bytes <= config.get('_whole_capacity', 0):
            mode = 'whole'
        if mode != 'blocks' and (whole_bytes <= available or mode == 'whole'):
            with guard.admit(whole_bytes, label='native_whole_case') as ticket:
                if ticket is None:
                    raise InterruptedError('原生整例处理已取消')
                data, label, image, spacing, roi, read_seconds = reader.read_case(row)
                pixels = window_image(data.transpose(2,1,0), window)[None]
                entry = dict(case_id=name, image=pixels,
                    label=None if label is None else label.transpose(2,1,0),
                    unknown_classes=row['unknown_classes'], source_shape_xyz=list(image.shape),
                    source_affine=image.affine.tolist(), source_units=image.header.get_xyzt_units()[0],
                    spacing_xyz=spacing.tolist(), roi_xyz=roi, properties=dict(class_locations={}),
                    medsam_native=True, medsam_windowed=True, gray_window=list(window),
                    timing=dict(read_decompress_seconds=read_seconds, native_seconds=perf_counter()-began))
                event(config['run_dir'], 'preprocess_execution_selected', case_id=name,
                      execution='whole', whole_peak_bytes=whole_bytes)
                record = save_case(directory, key, recipe_identity, entry)
            return name, case_pointer(record)
        settings = dict(config['preprocessing'])
        explicit = config.get('_explicit_preprocessing', [])
        if 'block_bytes' not in explicit:
            settings['block_bytes'] = max(1024, available//4)
        if 'max_block_bytes' not in explicit:
            settings['max_block_bytes'] = max(settings['max_block_bytes'], settings['block_bytes'])
        config = {**config, 'preprocessing': settings}
        workspace = BlockWorkspace(pending/'work', config, guard, name)
        try:
            image, spacing, roi, data, label, _, _, has_label = read_sources(reader, row, workspace)
            read_seconds = perf_counter()-began
            workspace.drop('mask')
            minimum, maximum = np.inf, -np.inf
            for part in workspace.blocks(data.shape, 16, label='native_window_bounds'):
                values = np.clip(data[part], *window)
                minimum, maximum = min(minimum, float(values.min())), max(maximum, float(values.max()))
                del values
            if maximum <= minimum:
                raise ValueError('MedSAM2 CT窗内没有有效灰度变化')
            pixels = workspace.create('result-image', (1, *data.shape), np.uint8)
            for part in workspace.blocks(data.shape, 24, label='native_gray_window'):
                values = np.clip(data[part], *window)
                pixels[(0, *part)] = ((values-minimum)/(maximum-minimum)*255.).astype(np.uint8)
                del values
            entry = dict(case_id=name, image=pixels, label=label if has_label else None,
                unknown_classes=row['unknown_classes'], source_shape_xyz=list(image.shape),
                source_affine=image.affine.tolist(), source_units=image.header.get_xyzt_units()[0],
                spacing_xyz=spacing.tolist(), roi_xyz=roi, properties=dict(class_locations={}),
                medsam_native=True, medsam_windowed=True, gray_window=list(window),
                timing=dict(read_decompress_seconds=read_seconds, native_seconds=perf_counter()-began))
            with stage(config, 'preprocess_case_save', case_id=name):
                record = save_case(directory, key, recipe_identity, entry, workspace)
        finally:
            workspace.release_handles()
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
