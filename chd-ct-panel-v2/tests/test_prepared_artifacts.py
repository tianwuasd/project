import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import blosc2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from chd_stunet.maintenance.prepared_artifacts import compact_prepared, verify_snapshot, writable_output
from chd_stunet.observability.locking import run_lock
from chd_stunet.observability.runlog import write_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PreparedArtifactsTests(unittest.TestCase):
    # 两份真实方法配置作为文件内容样本，只验证文件管理，不作为医学数组或模型结果。
    def setUp(self):
        self.began = time.monotonic()
        self.parent = Path(os.environ['CHD_ARTIFACT_TEST_ROOT']).resolve()
        self.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=self.parent)
        self.folder = Path(self.temporary.name)
        self.samples = [(ROOT/'configs/preprocessing'/name).read_bytes()
                        for name in ('stunet-s.json', 'stunet-s-partial.json')]

    def tearDown(self):
        files = [p for p in self.folder.rglob('*') if p.is_file()]
        size = sum(p.stat().st_size for p in files)
        self.assertLess(len(files), 200)
        self.assertLess(size, 20*1024**2)
        print(json.dumps(dict(test=self.id(), samples=2, scope='file_management_only',
              files=len(files), logical_bytes=size, seconds=time.monotonic()-self.began,
              training_updates=0, memory_peak='not_measured', gpu_peak='not_measured')))
        self.temporary.cleanup()

    def make_run(self, model):
        root = self.folder/model
        root.mkdir()
        cid = 'sample_a'
        row = dict(case_id=cid, source_hashes={'image': hashlib.sha256(self.samples[0]).hexdigest(),
                                             'label': hashlib.sha256(self.samples[1]).hexdigest()})
        write_json(root/'status.json', {'state': 'completed'})
        if model == 'stunet':
            (root/'.run.lock').write_bytes(b'0')
            write_json(root/'signature.json', {'identity': 'file-management-test'})
            write_json(root/'manifests/imagechd_input.json', dict(protocol='file-management-only',
                       identity='file-management-test', cases=[row]))
            write_json(root/'summary.json', dict(protocol='file-management-only', failed_cases=[],
                       requested_folds=[0], requested_case_ids=[cid]))
            write_json(root/'fold_0/training_contract.json', {'preprocessing_contract': 'contract'})
            data = root/'fold_0/data'
            data.mkdir(parents=True)
            for content, suffix in zip(self.samples, ('.b2nd', '_seg.b2nd')):
                blosc2.asarray(np.frombuffer(content, dtype=np.uint8), urlpath=str(data/(cid+suffix)))
            with (data/(cid+'.pkl')).open('wb') as stream:
                pickle.dump({'scope': 'file-management-only'}, stream)
            required = {p.name: sha(p) for p in data.iterdir()}
            write_json(root/'fold_0/completion'/f'{cid}.json',
                       dict(row, success=True, contract='contract', file_sha256=required))
            write_json(root/'audit/cases'/f'{cid}.json', dict(row, success=True))
            np.save(root/'audit'/f'{cid}.npy', np.frombuffer(self.samples[0], dtype=np.uint8))
            for name in (f'tasks/audit/{cid}.json', f'tasks/fold_0/{cid}.json',
                         f'fold_0/results/{cid}.json', f'fold_0/results/{cid}.progress.json'):
                write_json(root/name, {'case_id': cid})
            staging = root/'fold_0/staging'/f'{cid}-123'
            staging.mkdir(parents=True)
            shutil.copyfile(data/(cid+'.b2nd'), staging/(cid+'.b2nd'))
        else:
            root.with_name(root.name+'.prepare.lock').write_bytes(b'0')
            write_json(root/'manifest.json', {'identity': 'manifest', 'cases': [row]})
            write_json(root/'summary.json', dict(status='completed', failed={}, completed={cid: 'case'},
                                                manifest_identity='manifest'))
            data = root/'cases'/cid
            data.mkdir(parents=True)
            for content, name in zip(self.samples, ('ct.npy', 'labels.npy')):
                np.save(data/name, np.frombuffer(content, dtype=np.uint8))
            write_json(data/'complete.json', dict(row, identity='case', files={p.name: sha(p) for p in data.iterdir()}))
            for name in ('task', 'result', 'result.progress'):
                write_json(root/'jobs'/f'{cid}-{name}.json', {'case_id': cid})
            (data/'ct.partial').write_bytes(self.samples[0])
        write_json(root/'logs/evidence.json', {'scope': 'file-management-only'})
        return root, data

    def test_stunet_check_apply_repeat_preserve_data(self):
        root, data = self.make_run('stunet')
        before = {p.relative_to(root): sha(p) for p in root.rglob('*') if p.is_file()}
        preview = compact_prepared(root)
        self.assertFalse(preview['applied'])
        checked = subprocess.run([sys.executable, '-B', str(ROOT/'run.py'), 'compact', '--prepared', str(root)],
                                 check=True, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(json.loads(checked.stdout)['before'], preview['before'])
        self.assertEqual(before, {p.relative_to(root): sha(p) for p in root.rglob('*') if p.is_file()})
        result = compact_prepared(root, apply=True)
        print(json.dumps(dict(model='stunet', before=result['before'], after=result['after'],
                              removed_files=result['removed_files'], removed_bytes=result['removed_logical_bytes'])))
        self.assertEqual(result['removed_files'], 5)
        self.assertGreater(result['removed_logical_bytes'], 0)
        for p in data.iterdir():
            self.assertEqual(sha(p), before[p.relative_to(root)])
        self.assertTrue((root/'audit/sample_a.npy').is_file())
        self.assertTrue((root/'logs/evidence.json').is_file())
        self.assertEqual(compact_prepared(root, apply=True)['removed_files'], 0)

    def test_medsam_arrays_survive_compaction(self):
        root, data = self.make_run('medsam2')
        arrays = {p.name: sha(p) for p in data.glob('*.npy')}
        result = compact_prepared(root, apply=True)
        print(json.dumps(dict(model='medsam2', before=result['before'], after=result['after'],
                              removed_files=result['removed_files'], removed_bytes=result['removed_logical_bytes'])))
        self.assertEqual(result['removed_files'], 4)
        self.assertEqual(arrays, {p.name: sha(p) for p in data.glob('*.npy')})

    def test_shared_arrays_survive_preview_and_compaction(self):
        from chd_stunet.preprocessing.shared_cache import MODE, begin_view, commit_view, digest, lease
        root, data = self.make_run('stunet')
        cid = 'sample_a'
        plan = {'fold': 0}
        contract = digest(dict(run='file-management-test', plan=plan))
        write_json(root/'signature.json', dict(identity='file-management-test', method={'storage_mode': MODE}))
        write_json(root/'fold_0/preprocessing_plan.json', plan)
        write_json(root/'fold_0/training_contract.json', {'preprocessing_contract': contract})
        marker_path = root/f'fold_0/completion/{cid}.json'
        marker = json.loads(marker_path.read_text(encoding='utf-8'))
        write_json(marker_path, dict(marker, contract=contract, fold=0))
        with lease(root, enabled=True):
            begin_view(root, 0, [cid], contract)
            target = root/'shared_data'
            target.mkdir()
            for path in data.iterdir():
                path.rename(target/path.name)
            commit_view(root, 0, [cid], contract)
        before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
        compact_prepared(root)
        self.assertEqual(before, {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()})
        result = compact_prepared(root, apply=True)
        self.assertEqual(result['removed_files'], 5)
        self.assertEqual(marker['file_sha256'], {p.name: sha(p) for p in target.iterdir()})

    def test_active_run_and_incomplete_data_are_protected(self):
        root, data = self.make_run('stunet')
        with run_lock(root/'.run.lock'):
            with self.assertRaises(RuntimeError):
                compact_prepared(root, apply=True)
        (data/'sample_a.pkl').unlink()
        with self.assertRaises(ValueError):
            compact_prepared(root, apply=True)
        self.assertTrue((root/'tasks/audit/sample_a.json').is_file())

    def test_unknown_staging_file_stops_cleanup(self):
        root, _ = self.make_run('stunet')
        (root/'fold_0/staging/sample_a-123/keep.txt').write_bytes(self.samples[0])
        with self.assertRaises(ValueError):
            compact_prepared(root, apply=True)
        self.assertTrue((root/'tasks/audit/sample_a.json').is_file())

    def test_data_only_root_is_rejected(self):
        with self.assertRaises(ValueError):
            writable_output('/data_nas/user/derived')
        self.assertEqual(writable_output(self.folder), self.folder.resolve())

    def test_public_path_templates_resolve_from_local_workspace(self):
        from chd_stunet.config import platform_paths
        workspace = self.folder/'sample_workspace'
        values = platform_paths({'workspace_root': str(workspace)}, platform='linux')
        self.assertEqual(values['preprocess_output_root'], str(workspace)+'/chd_ct_v2_stunet_results/preprocessing')
        inferred = platform_paths({}, platform='linux', project_root=workspace/'code/project/chd-ct-panel-v2')
        self.assertEqual(inferred['preprocess_output_root'], values['preprocess_output_root'])
        self.assertNotIn('{', values['raw_root'])

    def test_frozen_producer_bytes_and_signature_are_checked(self):
        root = self.folder/'snapshot'
        relative = 'configs/preprocessing/stunet-s.json'
        path = root/'code'/relative
        path.parent.mkdir(parents=True)
        path.write_bytes(self.samples[0])
        record = {'producer_sha256': {relative: sha(path)}}
        identity = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
        write_json(root/'signature.json', dict(record, identity=identity))
        self.assertEqual(verify_snapshot(root), root/'code')
        path.write_bytes(self.samples[1])
        with self.assertRaises(ValueError):
            verify_snapshot(root)

    def test_resume_script_imports_the_real_frozen_producer(self):
        from chd_stunet.preprocessing.run import producer_sources
        root = self.folder/'real-snapshot'
        paths = producer_sources()
        for relative in paths:
            target = root/'code'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/relative, target)
        record = dict(producer_sha256=paths)
        identity = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
        write_json(root/'signature.json', dict(record, identity=identity))
        code = verify_snapshot(root)
        raw = self.folder/'raw'
        raw.mkdir()
        for index, content in enumerate(self.samples):
            (raw/f'sample-{index}.json').write_bytes(content)
        options = dict(raw_root=str(raw), output=str(root), method_name='stunet-s-partial', resume=True)
        # 输入只含两份文件样本，原生产者必须在发现配对检查处停止，不能进入医学预处理。
        completed = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/resume_prepare.py'),
                                    str(code), json.dumps(options)], capture_output=True, text=True, encoding='utf-8')
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('Incomplete/changed source pairing', completed.stderr)
        self.assertEqual(verify_snapshot(root), code)
        self.assertFalse(list(self.folder.glob('.prepare-resume-*')))


if __name__ == '__main__':
    unittest.main()
