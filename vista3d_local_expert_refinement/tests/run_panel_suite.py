"""Control-plane regression suite: stdlib only, no network/CT/GPU workloads."""
from pathlib import Path
import sys
import unittest

tests = Path(__file__).resolve().parent
suite = unittest.TestSuite()
for pattern in ('test_task*.py', 'test_panel*.py', 'test_environment_progress.py',
                'test_private_update.py', 'test_update_download.py', 'test_append_authorization.py', 'test_cohort.py',
                'test_baseline_contract.py','test_strict_panel.py','test_publication.py',
                'test_inner_contracts.py','test_evaluation_parallel.py','test_checkpoint_compat.py','test_gpu_memory.py'):
    suite.addTests(unittest.defaultTestLoader.discover(str(tests), pattern=pattern))
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not result.wasSuccessful())
