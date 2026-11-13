import json
import os
from pathlib import Path
import shutil
import time
import unittest

from dataset1282_analysis.contracts import digest, sha256_file
from dataset1282_analysis.storage_adoption import _adopt_unlocked
from dataset1282_analysis.storage_migration import canonical_path, resolve_relocated_path


class ExistingProcessAdoptionTest(unittest.TestCase):
    def test_real_case_identity_and_cleanup(self):
        required = {'CHD_RELOCATION_SAMPLE', 'CHD_RELOCATION_TEST_ROOT',
                    'CHD_RELOCATION_APPROVED_PARENT'}
        if not required <= os.environ.keys():
            self.skipTest('Real CT and an approved test directory are required')
        sample = Path(os.environ['CHD_RELOCATION_SAMPLE']).resolve(strict=True)
        root = Path(os.environ['CHD_RELOCATION_TEST_ROOT']).resolve()
        allowed = Path(os.environ['CHD_RELOCATION_APPROVED_PARENT']).resolve(strict=True)
        self.assertEqual(root.parent, allowed)
        self.assertEqual(root.name, 'dataset1282-relocation-check-20261113')
        self.assertFalse(root.exists())
        started = time.perf_counter()
        source = root / 'original'
        destination = root / 'moved'
        prepared_relative = Path('work/prepared/sample')
        prepared = destination / prepared_relative
        case = sample.name.removesuffix('_0000.nii.gz')
        artifact_relative = Path('cases') / sample.name
        artifact = prepared / artifact_relative
        try:
            artifact.parent.mkdir(parents=True)
            shutil.copyfile(sample, artifact)
            checksum = sha256_file(artifact)
            self.assertEqual(checksum, sha256_file(sample))
            request = {'method': 'HU-marginal-v1', 'cases': {case: {'role': 'train'}}}
            request_identity = digest(request)
            original_prepared = source / prepared_relative
            (prepared / 'request.json').write_text(json.dumps(request), encoding='utf-8')
            complete = {'state': 'completed', 'run_dir': str(original_prepared),
                        'request_digest': request_identity, 'train_cases': 1,
                        'development_cases': 0}
            (prepared / 'complete.json').write_text(json.dumps(complete), encoding='utf-8')
            experiment = 'xu-train-123'
            run_file = destination / 'experiments' / experiment / 'run.json'
            run_file.parent.mkdir(parents=True)
            run_file.write_text(json.dumps({'protocol': 'xu-model-run-4',
                                            'prepared': str(original_prepared),
                                            'request': request_identity}), encoding='utf-8')
            marker = prepared / 'commits' / ('process-' + case + '.json')
            marker.parent.mkdir()
            marker.write_text(json.dumps({'contract': request_identity,
                                          'job': {'id': marker.stem},
                                          'artifacts': {artifact_relative.as_posix(): checksum}}),
                              encoding='utf-8')
            result = _adopt_unlocked(source, destination, experiment)
            self.assertEqual(result['committed_cases'], 1)
            self.assertEqual(canonical_path(prepared), str(original_prepared))
            relocated = resolve_relocated_path(source / prepared_relative / artifact_relative,
                                               destination / '.storage-relocation.json', run_file)
            self.assertEqual(relocated, artifact)
            with artifact.open('r+b') as stream:
                first = stream.read(1)
                stream.seek(0)
                stream.write(bytes([first[0] ^ 1]))
            with self.assertRaises(ValueError):
                resolve_relocated_path(source / prepared_relative / artifact_relative,
                                       destination / '.storage-relocation.json', run_file)
            files = [path for path in root.rglob('*') if path.is_file()]
            self.assertEqual(len(files), 6)
            print(json.dumps({'case': case, 'files': len(files),
                              'bytes': sum(path.stat().st_size for path in files),
                              'seconds': time.perf_counter() - started}, ensure_ascii=False))
        finally:
            if root.exists():
                self.assertEqual(root.parent, allowed)
                self.assertEqual(root.name, 'dataset1282-relocation-check-20261113')
                shutil.rmtree(root)
                self.assertFalse(root.exists())


if __name__ == '__main__':
    unittest.main()
