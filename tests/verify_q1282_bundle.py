import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from time import perf_counter

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))

from smoke import SmokeWorkspace
from chd_segmentation.artifacts.store import file_digest, read_json
from chd_segmentation.config import load_config
from chd_segmentation.datasets.manifest import load_manifest
from chd_segmentation.datasets.q_split import build_q_split
from chd_segmentation.datasets.splits import apply_split, select_test
from chd_segmentation.datasets.transfer import bind_inputs
from chd_segmentation.models.workflow import training_names, options


def run(args):
    source, delivery = Path(args.source), Path(args.bundle)
    before = {str(p): file_digest(p) for parent in (source, delivery) for p in parent.iterdir() if p.is_file()}
    started = perf_counter()
    with SmokeWorkspace(args.work_root, args.report) as workspace:
        packet = workspace.root/'inputs'
        shutil.copytree(delivery, packet)
        first = bind_inputs(packet, workspace.root, {'chd1282': args.data_root}, args.weights)
        stamps = {p.name: p.stat().st_mtime_ns for p in packet.iterdir() if p.is_file()}
        assert not bind_inputs(packet, workspace.root, {'chd1282': args.data_root}, args.weights)['changed_files']
        assert stamps == {p.name: p.stat().st_mtime_ns for p in packet.iterdir() if p.is_file()}
        original = read_json(source/'chd1282_manifest.json')
        cohort = read_json(source/'chd1282_cohort.json')
        split, groups = build_q_split(original, cohort)
        assert groups == read_json(delivery/'q_groups.json')
        assert split == build_q_split(original, cohort)[0]
        config = load_config(packet/'q1282_prepare.yaml')
        manifest = apply_split(load_manifest(config['manifest'], 'chd1282', 'formal', allow_unassigned=True), config)
        assert len(manifest['cases']) == sum(map(len, groups.values()))
        prepared = dict(scope='formal', manifest=manifest, plan=dict(fit_ids=groups['A']))
        base_config = options(packet/'q1282_base.yaml')
        assert training_names(base_config, prepared, 'base') == groups['A']
        assert training_names(base_config, prepared, 'localizer') == groups['A']
        assert training_names(base_config, prepared, 'local') == groups['B']
        assert {row['case_id'] for row in manifest['cases'] if 'train' in row['roles']} == set(groups['A'])
        patients = [{row['patient'] for row in manifest['cases'] if row['case_id'] in groups[key]} for key in 'ABCD']
        for i, current in enumerate(patients):
            assert all(not current & other for other in patients[i+1:])
        tests = Counter(row['test_group'] for row in manifest['cases'] if 'test' in row['roles'])
        assert tests == dict(source_test=127, target_primary=298, target_secondary=31)
        assert len(select_test(manifest, 'internal')['cases']) == 127
        assert len(select_test(manifest, 'external', 'target_primary')['cases']) == 298
        assert len(select_test(manifest, 'external', 'target_secondary')['cases']) == 31
        train_original = {key for key, row in cohort['cases'].items() if row['role'] == 'train' and
                          row.get('included', True) and row.get('index_scan', True)}
        assert set(groups['A']+groups['B']) == train_original
        path = packet/'q1282_base.yaml'
        manual = yaml.safe_load(path.read_text(encoding='utf-8'))
        manual['updates'] = 123
        path.write_text(yaml.safe_dump(manual), encoding='utf-8')
        bind_inputs(packet, workspace.root, {'chd1282': args.data_root}, args.weights)
        assert options(path)['updates'] == 123
        assert all(file_digest(path) == sha for path, sha in before.items())
        workspace.results.append(dict(scope='冻结名单与文件绑定元数据；不读取CT体素，不训练', binding=first,
            cases={key: len(values) for key, values in groups.items()}, patients=dict(zip('ABCD', map(len, patients))),
            test_groups=dict(tests), patient_disjoint=True, original_train_preserved=True, fit_A_only=True,
            reproducible=True, manual_config_preserved=True, source_and_delivery_unchanged=True,
            seconds=perf_counter()-started, records_per_second=len(manifest['cases'])/(perf_counter()-started)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source', 'bundle', 'data-root', 'weights', 'work-root', 'report'):
        parser.add_argument('--'+name, required=True)
    run(parser.parse_args())
