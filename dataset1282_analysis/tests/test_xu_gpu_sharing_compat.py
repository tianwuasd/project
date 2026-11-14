import unittest

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


if __name__ == '__main__':
    unittest.main()
