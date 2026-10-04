"""Generate a wholly synthetic cardiac-style feature fixture; not clinical data."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def generate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261004)
    rows = []
    for i in range(240):
        site = ['A', 'B', 'C'][i % 3]
        age = float(rng.uniform(1, 18))
        anatomy = rng.normal(size=2)
        logit = 1.3 * anatomy[0] - 0.9 * anatomy[1] + 0.025 * (age - 9)
        label = int(rng.random() < 1 / (1 + np.exp(-logit)))
        for scan in range(2):
            quality = float(rng.uniform(0.2, 1.0))
            noise = (1 - quality) * (1.2 if site == 'C' else 0.7)
            volume_ratio = float(np.exp(0.25 * anatomy[0] + rng.normal(scale=noise)))
            shape_index = float(anatomy[1] + rng.normal(scale=noise))
            if rng.random() < 0.12:
                shape_index = ''
            rows.append([f'SYNTH_{i:04d}', site, label, age, volume_ratio, shape_index, quality])
    with (output / 'features.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['patient_id', 'site', 'label', 'age', 'volume_ratio', 'shape_index', 'quality'])
        w.writerows(rows)
    features = ['age', 'volume_ratio', 'shape_index', 'quality']
    manifest = {'schema_version': 1, 'data_kind': 'synthetic', 'patient_column': 'patient_id',
                'target': 'label', 'site_column': 'site', 'external_site': 'C', 'seed': 42, 'bootstrap': 500,
                'prediction_time': 'synthetic pre-intervention index; both scans declared earlier',
                'outcome': 'Bernoulli label generated from latent synthetic anatomy; not a diagnosis',
                'features': [{'name': name, 'available': 'pre_index', 'source': 'make_demo.py synthetic generator'}
                             for name in features],
                'feature_sets': {'clinical_only': ['age'],
                                 'clinical_morphology': ['age', 'volume_ratio', 'shape_index'],
                                 'clinical_morphology_quality': features}}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'data_kind': 'synthetic', 'patients': 240, 'rows': len(rows), 'output': str(output)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    print(json.dumps(generate(p.parse_args().output), ensure_ascii=False))
