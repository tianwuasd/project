"""ImageCHD scientific/control regressions; no real CT or formal training.

Run in the compute environment for Torch/nnU-Net API contracts. The archived
template suite is available explicitly; its Wholeheart/cohort/publication
fixtures describe a different project and are not ImageCHD acceptance tests.
"""
from pathlib import Path
import sys
import unittest

def main():
    tests = Path(__file__).resolve().parent
    source=str(tests.parent/'05_src')
    sys.path.insert(0,source)
    import os
    # Spawned synthetic workers need the same module path as this test runner.
    os.environ['PYTHONPATH']=source+os.pathsep+os.environ.get('PYTHONPATH','')
    suite = unittest.TestSuite()
    legacy=('--legacy-template' in sys.argv[1:])
    patterns=('test_task*.py', 'test_panel*.py', 'test_environment*.py', 'test_install_routes.py',
                'test_private_update.py', 'test_update_download.py', 'test_update_scope.py', 'test_append_authorization.py', 'test_cohort.py','test_full_source_protocol.py',
                'test_baseline_contract.py','test_strict_panel.py','test_publication.py',
                'test_inner_contracts.py','test_evaluation_parallel.py','test_checkpoint_compat.py','test_gpu_memory.py','test_gpu_admission.py',
                'test_planner_threads.py','test_training_timing.py','test_training_policy.py',
                'test_runtime_observations.py','test_assessment_policy.py','test_simple_panel.py',
                'test_automatic_execution.py','test_stage_boards.py','test_pipeline_catalog.py','test_integration_fixes.py',
                'test_live_resources.py','test_memory_guard.py','test_capped_batch.py','test_performance_repair.py','test_batch_pool.py','test_nnunet_batches.py','test_execution_transfer.py',
                'test_ddp_runtime.py','test_stage_coordinator.py','test_stage_completion.py','test_final_integration.py','test_final_resource_fixes.py','test_final_export_api.py','test_final_loader_api.py') if legacy else (
                'test_imagechd_contract.py','test_stunet_recipe.py','test_environment*.py','test_install_routes.py',
                'test_checkpoint_compat.py','test_gpu_memory.py','test_gpu_admission.py',
                'test_planner_threads.py','test_training_timing.py','test_training_policy.py',
                'test_runtime_observations.py','test_automatic_execution.py','test_stage_boards.py',
                'test_live_resources.py','test_memory_guard.py','test_capped_batch.py','test_performance_repair.py',
                'test_batch_pool.py','test_nnunet_batches.py','test_execution_transfer.py','test_ddp_runtime.py',
                'test_final_resource_fixes.py','test_final_export_api.py','test_final_loader_api.py',
                'test_evaluation_parallel.py','test_stage_coordinator.py')
    print('Scope: '+('archived template fixtures, not ImageCHD acceptance' if legacy else 'ImageCHD contracts, training identity, resource/recovery/worker controls'),flush=True)
    for pattern in patterns:
        suite.addTests(unittest.defaultTestLoader.discover(str(tests), pattern=pattern))
    if not legacy:
        sys.path.insert(0,str(tests))
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('test_task_lifecycle.InputTests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return not result.wasSuccessful()


if __name__ == '__main__':
    sys.exit(main())
