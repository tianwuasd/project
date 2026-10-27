"""Control-plane regression suite: stdlib only, no network/CT/GPU workloads."""
from pathlib import Path
import sys
import unittest

def main():
    tests = Path(__file__).resolve().parent
    suite = unittest.TestSuite()
    for pattern in ('test_task*.py', 'test_panel*.py', 'test_environment_progress.py',
                'test_private_update.py', 'test_update_download.py', 'test_update_scope.py', 'test_append_authorization.py', 'test_cohort.py',
                'test_baseline_contract.py','test_strict_panel.py','test_publication.py',
                'test_inner_contracts.py','test_evaluation_parallel.py','test_checkpoint_compat.py','test_gpu_memory.py',
                'test_planner_threads.py','test_training_timing.py','test_training_policy.py',
                'test_runtime_observations.py','test_assessment_policy.py','test_simple_panel.py',
                'test_automatic_execution.py','test_stage_boards.py','test_pipeline_catalog.py','test_integration_fixes.py',
                'test_live_resources.py','test_batch_pool.py','test_nnunet_batches.py','test_execution_transfer.py',
                'test_ddp_runtime.py','test_stage_coordinator.py','test_stage_completion.py','test_final_integration.py'):
        suite.addTests(unittest.defaultTestLoader.discover(str(tests), pattern=pattern))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return not result.wasSuccessful()


if __name__ == '__main__':
    sys.exit(main())
