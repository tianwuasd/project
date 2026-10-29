"""Natural-run regression recovery and bounded, append-only feedback evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from cooperative_control import stage_config
from training_feedback import RollingFeedback


class PerformanceRepairTests(unittest.TestCase):
    def test_regression_continues_latest_epoch_with_old_group_and_defers_additions(self):
        import sys
        import cooperative_control as control
        from test_integration_fixes import TopologyAndRegroupTests
        base=[f'GPU-{n}' for n in range(4)];expanded=base+['GPU-4','GPU-5'];calls=[]
        def tune(*args,**kwargs):
            group=list(args[4]) if len(args)>4 else base
            return dict(gpu_uuids=group,global_batch_size=8,local_batch_size=2,
                effective_cpu={},train_patches_per_epoch=2000,reference_train_patches_per_epoch=500)
        def launch(target,args,*other,**kwargs):
            calls.append(args)
            if len(calls)==1:
                result=dict(status='regroup',checkpoint='epoch10',next_epoch=10,
                    epoch_performance=[dict(epoch=e,seconds=200,global_batch=8) for e in range(6,10)])
            elif len(calls)==2:
                self.assertEqual(args['expansion_guard']['reference_seconds'],200)
                result=dict(status='regroup',checkpoint='epoch14',next_epoch=14,
                    regroup_reason='expansion_regression',expansion_guard=dict(rollback=True))
            else:result=dict(status='completed',checkpoint='final',next_epoch=250)
            return [dict(result=dict(result,initial_weight_sha256='seed0'))]
        with tempfile.TemporaryDirectory() as temporary:
            root=TopologyAndRegroupTests().workspace(temporary)
            with patch.dict(sys.modules,{'server_train':Mock()}),patch.object(control,'tune',side_effect=tune),\
                 patch.object(control,'admit',side_effect=lambda g:(g,{},[])),patch.object(control,'devices',return_value=expanded),\
                 patch('ddp_runtime.launch_ranks',side_effect=launch),patch.object(control,'release_devices') as release:
                control.train(root,'ct_holdG','B0')
            self.assertEqual([r['gpu_uuids'] for r in calls],[base,expanded,base])
            self.assertEqual(calls[-1]['checkpoint'],'epoch14')
            self.assertEqual(calls[-1]['global_batch_size'],8)
            self.assertEqual(release.call_args.args[0],{'GPU-4','GPU-5'})
            self.assertTrue(release.call_args.kwargs['defer_for_stage'])

    def test_only_sustained_comparable_expansion_regresses(self):
        from training_performance import ExpansionGuard
        settings=dict(global_batch=8,reference_seconds=200,previous_gpu_uuids=['GPU-a']*4)
        guard=ExpansionGuard(settings,8)
        self.assertFalse(guard.observe(0,1000)['rollback'])  # compile/startup warmup
        self.assertFalse(guard.observe(1,270)['rollback'])
        self.assertFalse(guard.observe(2,280)['rollback'])
        result=guard.observe(3,290)
        self.assertTrue(result['rollback'])
        self.assertEqual(result['samples'],[270,280,290])
        mismatch=ExpansionGuard(settings,6)
        for epoch in range(5):self.assertFalse(mismatch.observe(epoch,999)['rollback'])
        mixed=ExpansionGuard(settings,8)
        for epoch,seconds in enumerate([1000,270,190,280]):
            self.assertFalse(mixed.observe(epoch,seconds)['rollback'])

    def test_feedback_latest_is_bounded_and_history_is_append_only(self):
        resource=stage_config(dict(cpu_budget=80,memory_budget_bytes=100*2**30),[8,8,8],2,1)
        resource['allocation_snapshot']['private_probe_padding']='x'*400000
        class Dist:
            def all_gather_object(self,rows,local):rows[0]=local
            def broadcast_object_list(self,packet,src):pass
        tick=[0.]
        with tempfile.TemporaryDirectory() as temporary:
            f=RollingFeedback(resource,0,1,temporary,clock=lambda:tick[0]);prefix=None
            for index in range(80):
                tick[0]+=2;f.observe(dict(batch_wait_seconds=0),Mock(current_epoch=0),Dist())
                if index==39:prefix=(Path(temporary)/'feedback.rank0.jsonl').read_bytes()
            path=Path(temporary)/'feedback.rank0.json'
            latest=json.loads(path.read_text())
            self.assertIsInstance(latest,dict)
            self.assertLess(path.stat().st_size,50000)
            history=(Path(temporary)/'feedback.rank0.jsonl').read_bytes()
            self.assertTrue(history.startswith(prefix))
            self.assertEqual(len(history.splitlines()),4)
            self.assertNotIn(b'private_probe_padding',history)
            self.assertLessEqual(len(f.events),2)
            self.assertLessEqual(len(f.windows),2)


if __name__=='__main__':unittest.main()
