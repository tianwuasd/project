"""Explicit opt-in, bounded case benchmarks; never a clinical quality test."""
from dataclasses import replace
from pathlib import Path

from .image import inspect_image
from .io import discover, now, read_json, write_json
from .pipeline import disjoint, run


def benchmark(input_root, output_root, config, manifest=None, samples=3, worker_options=(1, 2, 4)):
    if not 1 <= samples <= 5:
        raise ValueError('benchmark samples must be in [1,5]')
    if not 1 <= len(worker_options) <= 4 or len(set(worker_options)) != len(worker_options):
        raise ValueError('choose 1-4 distinct worker counts')
    candidates = [replace(config, workers=w) for w in worker_options]
    source, output = Path(input_root).resolve(), Path(output_root).resolve()
    if not disjoint(source, output):
        raise ValueError('benchmark output must be separate from input')
    cases = discover(source, manifest)
    ordered = sorted(cases, key=lambda case: inspect_image(case['source'], config)['estimated_working_bytes'])
    count = min(samples, len(ordered))
    indices = sorted(set(round(i * (len(ordered)-1) / max(1, count-1)) for i in range(count)))
    selected = [ordered[i] for i in indices]
    output.mkdir(parents=True, exist_ok=False)
    selection = output / 'inputs.json'
    write_json(selection, [{'input': case['input'], 'sha256': case['expected_sha256']} for case in selected])
    report = {'created_at': now(), 'scope': 'only selected inputs on this machine; warm caches/order may affect timings',
              'source_count': len(selected), 'trials': [], 'numerically_identical': True}
    reference = None
    for index, candidate in enumerate(candidates):
        trial = output / f'trial-{index+1}'
        result = run(source, trial, candidate, selection)
        if result['failed']:
            report['trials'].append({'workers': candidate.workers, 'state': result['state'], 'errors': result['errors']})
            report['numerically_identical'] = False
            continue
        records = [read_json(path) for path in sorted((trial / 'cases').glob('*/record.json'))]
        hashes = {record['case_id']: record['verification']['voxel_payload_sha256'] for record in records}
        reference = hashes if reference is None else reference
        equal = hashes == reference
        report['numerically_identical'] &= equal
        report['trials'].append({'workers': candidate.workers, 'compute_threads': candidate.compute_threads,
                                'resources': result['resources'], 'performance': result['performance'],
                                'pixels_equal_to_first_success': equal,
                                'case_timings': [record['timings_seconds'] for record in records]})
        write_json(output / 'benchmark.json', report)
    write_json(output / 'benchmark.json', report)
    return report
