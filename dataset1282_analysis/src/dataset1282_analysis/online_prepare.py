import argparse
import json
from pathlib import Path
import shutil
from time import perf_counter

from .contracts import atomic_json, sha256_file
from .input_source import open_prepared
from .storage_paths import assert_output_location


def compact(source, output, *, technical_subset=False):
    source = Path(source).resolve(strict=True)
    output = assert_output_location(output)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('紧凑结果需要独立目录')
    if output.exists():
        raise FileExistsError('紧凑结果输出目录必须不存在')
    started = perf_counter()
    cache = open_prepared(source, allow_technical_subset=technical_subset)
    relatives = ['request.json', 'complete.json', 'cohort.json', 'fit.json', 'commits/fit.json']
    base = cache._dataset
    relatives += [f'{base}/{name}' for name in (
        'training_contract.json', 'splits_final.json', 'dataset.json',
        'nnUNetPlans.json', 'partial_supervision.json')]
    for name in sorted(cache._records):
        cache.geometry(name)
        relatives += [f'commits/stage-{name}.json', f'commits/process-{name}.json',
                      f'staged/{name}/metadata.json', f'qc/{name}.json']
    checksums = {relative:sha256_file(cache._path(relative)) for relative in relatives}
    output.mkdir(parents=True, mode=0o700)
    for relative in relatives:
        target = output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache._path(relative), target)
        if sha256_file(target) != checksums[relative]:
            raise ValueError('复制预处理元数据时发生变化')
    atomic_json(output/'online.json', dict(schema='chd-online-1',
                request_digest=cache.request_digest, metadata=checksums))
    result = dict(state='completed', mode='monai-online', cases=len(cache._records),
                  seconds=perf_counter()-started, volume_files=0,
                  metadata_files=len(relatives)+1,
                  metadata_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
                  original_preserved=True)
    (output/'preparation.log').write_text(json.dumps(result, ensure_ascii=False)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description='复用已审核计划，生成仅含元数据的在线预处理结果')
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--technical-subset', action='store_true')
    args = parser.parse_args()
    print(json.dumps(compact(args.prepared, args.output, technical_subset=args.technical_subset),
                     ensure_ascii=False), flush=True)
