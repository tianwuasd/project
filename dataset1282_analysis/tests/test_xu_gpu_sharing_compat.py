import unittest
from pathlib import Path

from dataset1282_analysis.contracts import sha256_file
from dataset1282_analysis.xu_blood import _source as blood_sources
from dataset1282_analysis import xu_init
from dataset1282_analysis.xu_pipeline import _sources
from dataset1282_analysis.xu_resume_compat import (AUDIT_PATH, GPU_SHARING_PATH,
    _audit, _gpu_sharing_audit, source_compatible)


class SharedGpuCompatibilityTests(unittest.TestCase):
    def test_existing_source_maps_resume_with_exact_gpu_sharing_change(self):
        current = _sources(False)
        patch = _gpu_sharing_audit(GPU_SHARING_PATH)
        previous = dict(current)
        for name, versions in patch['changes'].items():
            self.assertEqual(current[name], versions['current'])
            previous[name] = versions['stored']
        self.assertTrue(source_compatible(previous, current))
        matches = [maps for edge in _audit(AUDIT_PATH)['transitions']
                   for maps in edge['sources']
                   if maps['path_prefix'] == '' and maps['current'] == previous]
        self.assertTrue(matches)
        for maps in matches:
            self.assertTrue(source_compatible(maps['stored'], current))
        changed = dict(current, **{'src/dataset1282_analysis/xu_models.py':
                                   previous['src/dataset1282_analysis/xu_resources.py']})
        self.assertFalse(source_compatible(previous, changed))

    def test_blood_and_init_checkpoints_keep_source_compatibility(self):
        patch = _gpu_sharing_audit(GPU_SHARING_PATH)
        previous_hash = patch['changes']['src/dataset1282_analysis/xu_resume_compat.py']['stored']
        source = Path(xu_init.__file__).parent
        init_names = ('xu_init.py', 'xu_models.py', 'xu_targets.py', 'xu_training.py',
                      'xu_supply.py', 'xu_distributed.py', 'xu_resume_compat.py')
        init_sources = {name: sha256_file(source/name) for name in init_names}
        for current in (blood_sources(), init_sources):
            previous = dict(current, **{'xu_resume_compat.py': previous_hash})
            self.assertTrue(source_compatible(previous, current, 'src/dataset1282_analysis/'))
            matching = [maps for edge in _audit(AUDIT_PATH)['transitions']
                        for maps in edge['sources']
                        if maps['path_prefix'] == 'src/dataset1282_analysis/' and
                        maps['current'] == previous]
            self.assertTrue(matching)
            for maps in matching:
                self.assertTrue(source_compatible(maps['stored'], current,
                                                  'src/dataset1282_analysis/'))


if __name__ == '__main__':
    unittest.main()
