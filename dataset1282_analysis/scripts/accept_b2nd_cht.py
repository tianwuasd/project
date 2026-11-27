import argparse
import json
from pathlib import Path
import sys
import time


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'compressed', 'code-snapshot', 'xu-code', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--case')
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError('输出必须是已有目录内尚不存在的JSON文件')
    if output.suffix.lower() != '.json':
        raise ValueError('验收报告使用.json扩展名')

    code = args.code_snapshot.resolve(strict=True)
    snapshot = read_json(code / '.snapshot.json')
    if not (code / 'src/chd_segmentation/preprocessing/arrays.py').is_file():
        raise ValueError('指定的已有代码快照缺少压缩读取模块')
    sys.dont_write_bytecode = True
    xu_code = args.xu_code.resolve(strict=True)
    sys.path.insert(0, str(xu_code / 'src'))
    sys.path.insert(0, str(code / 'src'))
    import numpy as np
    import psutil
    from chd_segmentation.preprocessing import inputs
    from chd_segmentation.preprocessing.arrays import B2ND_FORMAT, NPY_FORMAT, CompressedArray
    from chd_segmentation.data.dataset import sample_patch
    from dataset1282_analysis import contracts as xu_contracts

    if not Path(inputs.__file__).resolve().is_relative_to(code):
        raise ValueError('实际读取模块未来自指定的已有代码快照')
    if not Path(xu_contracts.__file__).resolve().is_relative_to(xu_code):
        raise ValueError('验收身份算法未来自指定的Xu代码')
    process = psutil.Process()
    peak_rss = process.memory_info().rss

    def memory_boundary(extra=0):
        nonlocal peak_rss
        peak_rss = max(peak_rss, process.memory_info().rss)
        memory = psutil.virtual_memory()
        if extra > 256 * 1024**2 or memory.available - extra < memory.total * .2:
            raise MemoryError('单例验收的解压预算或整机80%内存条件不满足')

    source_path = args.source.resolve(strict=True)
    prepared_path = args.compressed.resolve(strict=True)
    original = read_json(source_path)
    prepared = read_json(prepared_path)
    if original['storage']['format'] != NPY_FORMAT or prepared['storage']['format'] != B2ND_FORMAT:
        raise ValueError('验收要求原NPY描述与新B2ND描述')
    if not original['storage']['complete'] or not prepared['storage']['complete']:
        raise ValueError('验收要求两份预处理均已完成')
    if original['dataset'] != prepared['dataset']:
        raise ValueError('压缩前后数据集不同')
    names = list(original['storage']['cases'])
    if not names:
        raise ValueError('原预处理描述没有病例')
    case_id = args.case or names[0]
    a = inputs.resolve_preprocessed(source_path, dataset=original['dataset'], case_ids=[case_id])
    b = inputs.resolve_preprocessed(prepared_path, dataset=prepared['dataset'], case_ids=[case_id])
    memory_boundary()
    before = a.open(case_id, with_label=True)
    after = b.open(case_id, with_label=True)
    if not all(isinstance(after[k], CompressedArray) for k in ('image', 'label')):
        raise ValueError('CHT实际入口未使用压缩影像及标签')
    if not all(isinstance(before[k], np.memmap) and before[k].mode == 'r' for k in ('image', 'label')):
        raise ValueError('原NPY必须通过只读映射打开')
    if before['image'].ndim != 4 or before['image'].shape[0] != 1 or before['label'].ndim != 3:
        raise ValueError('要求单通道影像[1,Z,Y,X]和标签[Z,Y,X]')
    if before['image'].shape[1:] != before['label'].shape:
        raise ValueError('影像与标签网格不同')
    for kind in ('image', 'label'):
        if before[kind].shape != after[kind].shape or before[kind].dtype != after[kind].dtype:
            raise ValueError('压缩前后数组尺寸或类型不同：' + kind)
    if before['label'].dtype.kind != 'i':
        raise ValueError('标签必须使用可保留-1忽略值的有符号整数')

    records = []
    for source in (a, b):
        pointer = source.cases[case_id]
        records.append(read_json(source.root / pointer['key'][:16] / 'case.json'))
    if records[0]['entry'] != records[1]['entry'] or records[0]['location_keys'] != records[1]['location_keys']:
        raise ValueError('病例来源、几何或监督元数据不同')
    for field in ('source_plan', 'processed_spacing', 'axis_order', 'transform', 'intensity'):
        if before[field] != after[field]:
            raise ValueError('CHT实际读取后的元数据不同：' + field)
    locations_a = before['properties']['class_locations']
    locations_b = after['properties']['class_locations']
    if locations_a.keys() != locations_b.keys():
        raise ValueError('前景采样类别不同')
    for label, points in locations_a.items():
        if not np.array_equal(points, locations_b[label]):
            raise ValueError('前景采样坐标不同')

    workspace = sum(after[k].decode_workspace_bytes for k in ('image', 'label'))
    memory_boundary(workspace + 8 * 1024**2)
    seed = 20261127
    patch_a = sample_patch(before, (32, 32, 32), np.random.default_rng(seed), .33)
    patch_b = sample_patch(after, (32, 32, 32), np.random.default_rng(seed), .33)
    for left, right in zip(patch_a, patch_b):
        if left.dtype != right.dtype or not np.array_equal(left, right):
            raise ValueError('CHT实际训练取块数值不同')
    del patch_a, patch_b
    shape = before['label'].shape
    boxes = [[(0, min(32, n)) for n in shape], [(max(0, n - 32), n) for n in shape]]
    for box in boxes:
        memory_boundary(workspace + 8 * 1024**2)
        region = tuple(slice(start, stop) for start, stop in box)
        for kind in ('image', 'label'):
            selection = (slice(None), *region) if kind == 'image' else region
            if not np.array_equal(before[kind][selection], after[kind][selection]):
                raise ValueError('固定边界区域不同：' + kind)
    memory_boundary()
    if read_json(source_path) != original or read_json(prepared_path) != prepared:
        raise ValueError('验收期间预处理描述发生变化')
    record = dict(
        protocol='chd-b2nd-read-acceptance-v1', acceptance_party='CHT', passed=True,
        source=str(source_path), source_identity=xu_contracts.digest(original),
        prepared=str(prepared_path), prepared_identity=xu_contracts.digest(prepared),
        recipe_identity=prepared['storage']['recipe_identity'], cases=[case_id],
        code_snapshot=str(code), code_sha=snapshot['sha'], reader_file=inputs.__file__,
        identity_reader_file=xu_contracts.__file__,
        reader='resolve_preprocessed/OfflineDataset.open/sample_patch', read_mode='r',
        image_shape=list(after['image'].shape), image_dtype=str(after['image'].dtype),
        label_shape=list(after['label'].shape), label_dtype=str(after['label'].dtype),
        seed=seed, patch=[32, 32, 32], foreground_fraction=.33, regions=boxes,
        metadata_equal=True, locations_equal=True, sampled_arrays_equal=True,
        decode_workspace_bytes=workspace, sampled_peak_rss_bytes=peak_rss,
        seconds=time.monotonic() - started,
        scope='一例元数据、一次训练取块及两个固定区域；不证明完整数组一致或全模型运行',
    )
    with output.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
