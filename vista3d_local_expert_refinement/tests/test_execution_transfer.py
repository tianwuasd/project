"""CPU-only execution and logical transfer contracts; no local CUDA workload."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_src'))


class ExecutionPolicyTests(unittest.TestCase):
    def test_configure_preserves_explicit_compile(self):
        from test_planner_threads import configure_for
        with patch.dict(os.environ, {'nnUNet_compile': 'true'}):
            configure_for(Path('unused'))()
            self.assertEqual(os.environ['nnUNet_compile'], 'true')

    def test_capacity_usable_budget_has_no_second_headroom(self):
        from training_execution import capacity_batch
        facts = dict(proven_batch=2, usable_bytes=20, fixed_bytes=4,
                     peak_allocated_bytes=8, peak_reserved_bytes=10,
                     optimizer_initialized=True, slope_bytes=2)
        self.assertEqual(capacity_batch(**facts)['candidate_batch'], 8)
        # Allocator reserved is observation, not another deduction from U.
        self.assertEqual(capacity_batch(**dict(facts,peak_reserved_bytes=18))['candidate_batch'],8)
        for slope in (0, -1, float('nan')):
            self.assertEqual(capacity_batch(**dict(facts, slope_bytes=slope))['candidate_batch'], 2)
        self.assertEqual(capacity_batch(**dict(facts, optimizer_initialized=False))['candidate_batch'], 2)

    def test_capacity_derives_slope_only_from_reliable_initialized_peak(self):
        from training_execution import capacity_batch
        facts=dict(proven_batch=2,usable_bytes=20,fixed_bytes=4,
            peak_allocated_bytes=8,peak_reserved_bytes=10,optimizer_initialized=True)
        result=capacity_batch(**facts)
        self.assertEqual(result['candidate_batch'],8)
        self.assertEqual(result['slope_bytes'],2)
        for peak in (None,0,4,float('nan')):
            self.assertEqual(capacity_batch(**dict(facts,peak_allocated_bytes=peak))['candidate_batch'],2)
        self.assertEqual(capacity_batch(**dict(facts,evidence_reliable=False))['candidate_batch'],2)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'requires pinned API runtime')
class TransferTests(unittest.TestCase):
    def test_large_half_output_probe_does_not_create_artificial_gradient_overflow(self):
        from training_execution import _probe_network
        class Large(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.ones((),dtype=torch.float16))
            def forward(self,x):return [self.weight.expand(131072),self.weight.expand(262144)]
        model=Large()
        _,outputs,gradients=_probe_network(model,torch.ones(1),tuple(model.parameters()),'cpu')
        self.assertEqual(len(outputs),2)
        self.assertTrue(torch.isfinite(gradients[0]))
        self.assertGreater(float(gradients[0]),0)

    def compile_compatibility(self, kind):
        import copy
        import types
        import numpy as np
        import random
        from training_execution import apply_execution,preflight_compile
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__();self.linear=torch.nn.Linear(3,2)
                self.register_buffer('calls',torch.zeros(()))
            def forward(self,x):
                self.calls.add_(1)
                # Each route must start with exactly the same RNG and buffers.
                return self.linear(x)+torch.rand(())+random.random()+float(np.random.rand())+self.calls
        class WrongGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx,value):return value.clone()
            @staticmethod
            def backward(ctx,gradient):return gradient*2
        net=Network()
        optimizer=torch.optim.SGD(net.parameters(),lr=.1,momentum=.9)
        for parameter in net.parameters():parameter.grad=torch.ones_like(parameter)
        optimizer.step()
        trainer=types.SimpleNamespace(network=net,is_ddp=False,device=torch.device('cpu'),print_to_log_file=lambda *a:None)
        from checkpoint_rng import capture
        state={k:v.clone() for k,v in net.state_dict().items()}
        optimizer_before=copy.deepcopy(optimizer.state_dict())
        grads=[p.grad.clone() for p in net.parameters()]
        rng=capture();batch={'data':torch.ones(2,3)}
        def compiler(module,**kwargs):
            def forward(x):
                value=module(x)
                if kind=='bias':return value*2
                if kind=='nan':return value*float('nan')
                if kind=='structure':return [value]
                if kind=='gradient':return WrongGradient.apply(value)
                return value
            return forward
        report=apply_execution(torch,{'compile':True},main_threads=1)
        with patch.object(torch,'compile',side_effect=compiler):preflight_compile(trainer,batch,report)
        self.assertEqual(report['compile_applied'],kind=='equal',report)
        if kind!='equal':
            self.assertIs(trainer.network,net)
            self.assertIn('compatibility',report['fallback_reason'])
        for key,value in state.items():torch.testing.assert_close(net.state_dict()[key],value,rtol=0,atol=0)
        for parameter,grad in zip(net.parameters(),grads):torch.testing.assert_close(parameter.grad,grad,rtol=0,atol=0)
        for key,value in optimizer_before['state'].items():
            torch.testing.assert_close(optimizer.state_dict()['state'][key]['momentum_buffer'],value['momentum_buffer'],rtol=0,atol=0)
        self.assertEqual(random.getstate(),rng['python'])
        np.testing.assert_array_equal(np.random.get_state()[1],rng['numpy'][1])
        self.assertTrue(torch.equal(torch.get_rng_state(),rng['torch']))
        torch.testing.assert_close(batch['data'],torch.ones(2,3),rtol=0,atol=0)
        return report

    def test_compile_rejects_biased_nonfinite_structure_and_gradient_outputs(self):
        for kind in ('bias','nan','structure','gradient'):
            with self.subTest(kind=kind):self.compile_compatibility(kind)

    def test_compile_accepts_equal_stochastic_routes_without_state_change(self):
        report=self.compile_compatibility('equal')
        self.assertTrue(report['compatibility']['passed'])
        self.assertEqual(report['compatibility']['rtol'],1e-4)

    def make(self, steps=4):
        from batch_pool import EpochBatchIterator
        from batch_transfer import BatchTransfer
        class Source:
            def __iter__(self): return self
            def __next__(self): return {'data': torch.rand(2, 3), 'target': [torch.zeros(2, 1)]}
        source = EpochBatchIterator(Source(), workers=0, phase_steps=steps)
        return BatchTransfer(source, torch.device('cpu'))

    def test_prepare_does_not_consume_and_checkpoint_replays(self):
        transfer = self.make()
        try:
            initial = transfer.state_dict()
            batch = next(transfer)
            self.assertEqual(transfer.state_dict()['next_index'], 0)
            torch.testing.assert_close(next(transfer)['data'], batch['data'])
            transfer.mark_consumed()
            self.assertEqual(transfer.state_dict()['next_index'], 1)
            transfer.load_state_dict(initial)
            torch.testing.assert_close(next(transfer)['data'], batch['data'])
        finally: transfer.close()

    def test_phase_limit_pause_and_global_rng_unchanged(self):
        transfer = self.make(2)
        try:
            rng = torch.get_rng_state().clone()
            for _ in range(2):
                next(transfer); transfer.mark_consumed()
            with self.assertRaises(StopIteration): next(transfer)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(transfer.observations()['consumed_index'], 2)
            transfer.set_epoch(1); transfer.set_phase_steps(1)
            next(transfer); transfer.pause()
            self.assertEqual(transfer.state_dict()['next_index'], 0)
        finally: transfer.close()

    def test_modes_and_compile_lazy_failure_restore(self):
        from training_execution import apply_execution, preflight_compile
        import types
        net = torch.nn.Linear(3, 2)
        trainer = types.SimpleNamespace(network=net, is_ddp=False, device=torch.device('cpu'),
            print_to_log_file=lambda *a: None)
        with patch.dict(os.environ, {}):
            report = apply_execution(torch, {'mode':'strict', 'compile':True}, main_threads=2)
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertEqual(os.environ['TORCHINDUCTOR_COMPILE_THREADS'], '2')
            self.assertFalse(report['compile_requested'])
            report = apply_execution(torch, {'mode':'performance', 'compile':True}, main_threads=1)
            self.assertTrue(torch.backends.cudnn.benchmark)
            before = {k:v.clone() for k,v in net.state_dict().items()}
            rng = torch.get_rng_state().clone()
            def broken(*args, **kwargs):
                self.assertEqual(kwargs, {'backend':'inductor','mode':'default'})
                def forward(x):
                    torch.rand(2)
                    raise RuntimeError('lazy compiler failure')
                return forward
            with patch.object(torch, 'compile', side_effect=broken):
                preflight_compile(trainer, {'data':torch.ones(2,3)}, report)
            self.assertEqual(report['compile_applied'], False)
            self.assertIn('lazy compiler failure', report['fallback_reason'])
            self.assertIs(trainer.network, net)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for key, value in before.items(): torch.testing.assert_close(net.state_dict()[key],value)

    def test_two_slots_reuse_events_and_bounded_lookahead_control(self):
        from concurrent.futures import ThreadPoolExecutor
        from contextlib import nullcontext
        import threading
        class Event:
            def record(self, *args): pass
            def synchronize(self): pass
            def elapsed_time(self, other): return 1.
        class Stream:
            def wait_event(self, event): pass
            def synchronize(self): pass
        transfer = self.make(4)
        transfer.cuda = True
        transfer.stream = Stream()
        transfer.executor = ThreadPoolExecutor(max_workers=1)
        pin_threads = []
        def pin(tensor):
            pin_threads.append(threading.get_ident()); return tensor
        with patch.object(torch.cuda,'device',return_value=nullcontext()), \
             patch.object(torch.cuda,'stream',return_value=nullcontext()), \
             patch.object(torch.cuda,'Event',side_effect=lambda **kw:Event()), \
             patch.object(torch.cuda,'current_stream',return_value=Stream()), \
             patch.object(torch.Tensor,'pin_memory',pin):
            try:
                first = next(transfer)
                pointer = first['data'].data_ptr()
                self.assertEqual(transfer.source.state_dict()['next_index'],2)
                self.assertEqual(transfer.consumed,0)
                transfer.mark_consumed()
                next(transfer); transfer.mark_consumed()
                third = next(transfer)
                self.assertEqual(third['data'].data_ptr(),pointer)
                transfer.pause()
                self.assertEqual(transfer.state_dict()['next_index'],2)
                self.assertEqual(transfer.stats['queue_peak'],2)
                self.assertTrue(pin_threads)
                self.assertTrue(all(t != threading.get_ident() for t in pin_threads))
            finally: transfer.close()

    def test_eager_loss_weights_including_ddp_last_scale(self):
        from project_trainers import BaselineNNUNet
        import types
        import numpy as np
        for ddp in (False,True):
            trainer = BaselineNNUNet.__new__(BaselineNNUNet)
            trainer.is_ddp = ddp
            trainer.enable_deep_supervision = True
            trainer.label_manager = types.SimpleNamespace(has_regions=False,ignore_label=None)
            trainer.configuration_manager = types.SimpleNamespace(batch_dice=True)
            trainer._get_deep_supervision_scales = lambda:[1,.5,.25]
            with patch.dict(os.environ, {'nnUNet_compile':'true'}):
                loss = trainer._build_loss()
            expected = np.array([1.,.5,1e-6 if ddp else 0.])
            np.testing.assert_array_equal(loss.weight_factors,expected/expected.sum())

    def trainer(self):
        from project_trainers import BaselineNNUNet
        import types
        t = BaselineNNUNet.__new__(BaselineNNUNet)
        t.device=torch.device('cpu'); t.is_ddp=False; t.local_rank=0
        t.was_initialized=True; t.disable_checkpointing=False; t.grad_scaler=None
        t.network=torch.nn.Linear(3,2); t.optimizer=torch.optim.SGD(t.network.parameters(),lr=.1,momentum=.9)
        t.loss=torch.nn.MSELoss(); t.current_epoch=0; t.my_init_kwargs={}; t._best_ema=.2
        t.inference_allowed_mirroring_axes=None
        t.logger=types.SimpleNamespace(get_checkpoint=lambda:{},load_checkpoint=lambda state:None)
        return t

    def test_single_atomic_save_and_failed_replace_preserves_old_generation(self):
        import tempfile
        trainer=self.trainer()
        trainer.train_step({'data':torch.ones(2,3),'target':torch.zeros(2,2)})
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'checkpoint.pth'
            with patch.object(torch,'save',wraps=torch.save) as save, patch.object(torch,'load',side_effect=AssertionError('save must not load')):
                trainer.save_checkpoint(str(path))
                self.assertEqual(save.call_count,1)
            old=path.read_bytes()
            with patch('project_trainers.os.replace',side_effect=OSError('replace fault')):
                with self.assertRaisesRegex(OSError,'replace fault'): trainer.save_checkpoint(str(path))
            self.assertEqual(path.read_bytes(),old)
            self.assertFalse(Path(str(path)+'.tmp').exists())
            with torch.no_grad(): next(trainer.network.parameters()).add_(1)
            trainer.load_checkpoint(str(path))
            self.assertTrue(trainer.optimizer.state)

    def test_finite_training_matches_pinned_and_rejects_bad_gradient_before_update(self):
        import copy
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        trainer=self.trainer(); reference=copy.deepcopy(trainer)
        batch={'data':torch.ones(2,3),'target':torch.zeros(2,2)}
        expected=nnUNetTrainer.train_step(reference,batch)
        actual=trainer.train_step(batch)
        self.assertEqual(actual['loss'],expected['loss'])
        for p,q in zip(trainer.network.parameters(),reference.network.parameters()): torch.testing.assert_close(p,q,rtol=0,atol=0)
        old=[p.clone() for p in trainer.network.parameters()]
        hook=next(trainer.network.parameters()).register_hook(lambda grad:grad*float('inf'))
        try:
            with self.assertRaisesRegex(RuntimeError,'non-finite'): trainer.train_step(batch)
        finally: hook.remove()
        for p,q in zip(trainer.network.parameters(),old): torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_initialized_optimizer_memory_facts_count_actual_storage(self):
        from training_execution import memory_facts
        trainer=self.trainer()
        self.assertFalse(memory_facts(trainer)['optimizer_initialized'])
        trainer.train_step({'data':torch.ones(2,3),'target':torch.zeros(2,2)})
        facts=memory_facts(trainer)
        expected=sum(p.numel()*p.element_size() for p in trainer.network.parameters())*3
        self.assertTrue(facts['optimizer_initialized'])
        self.assertEqual(facts['fixed_bytes'],expected)  # parameters + grads + SGD momentum
        self.assertIsNone(facts['peak_allocated_bytes'])

    def test_real_worker_prefetch_rewinds_without_restarting_workers(self):
        from batch_pool import EpochBatchIterator
        from batch_transfer import BatchTransfer
        from nnunet_checkpoint_rank_fixture import CheckpointBatches
        source=EpochBatchIterator(CheckpointBatches(),workers=2,prefetch=3,
            phase_steps=3,timeout_seconds=120)
        transfer=BatchTransfer(source,'cpu')
        try:
            next(transfer);transfer.mark_consumed()
            unconsumed=next(transfer)['data'].clone()
            pids=source.observations()['worker_pids']
            state=transfer.state_dict()
            self.assertEqual(state['next_index'],1)
            self.assertEqual(source.observations()['worker_pids'],pids)
            self.assertTrue(source.observations()['paused'])
            torch.testing.assert_close(next(transfer)['data'],unconsumed,rtol=0,atol=0)
            transfer.mark_consumed()
            self.assertEqual(transfer.state_dict()['next_index'],2)
        finally:transfer.close()

    def test_sync_batchnorm_cannot_enter_local_preflight_collective(self):
        import types
        from training_execution import apply_execution,preflight_compile
        network=torch.nn.Sequential(torch.nn.SyncBatchNorm(3),torch.nn.Linear(3,2))
        trainer=types.SimpleNamespace(network=network,is_ddp=False,device=torch.device('cpu'),print_to_log_file=lambda *a:None)
        report=apply_execution(torch,{'compile':True},main_threads=1)
        with patch.object(torch,'compile',side_effect=AssertionError('must reject before compile')):
            preflight_compile(trainer,{'data':torch.ones(2,3)},report)
        self.assertFalse(report['compile_applied'])
        self.assertIn('SyncBatchNorm',report['fallback_reason'])

    @unittest.skipUnless(os.environ.get('VISTA_TEST_IDLE_CUDA') == '1', 'requires explicitly reserved idle CUDA device')
    def test_actual_cuda_transfer_optional(self):
        from batch_transfer import BatchTransfer
        cpu=self.make(3)
        source=cpu.source
        transfer=BatchTransfer(source,torch.device('cuda',0))
        try:
            first=next(transfer)
            self.assertTrue(first['data'].is_cuda)
            expected=first['data'].cpu().clone()
            self.assertEqual(transfer.state_dict()['next_index'],0)
            replay=next(transfer)
            torch.testing.assert_close(replay['data'].cpu(),expected)
            for _ in range(3):
                replay['data'].square().sum().backward() if replay['data'].requires_grad else replay['data'].square().sum()
                transfer.mark_consumed()
                if transfer.consumed < 3: replay=next(transfer)
            self.assertEqual(transfer.stats['queue_peak'],2)
            self.assertGreater(transfer.stats['copy_seconds'],0)
            with self.assertRaises(StopIteration):next(transfer)
        finally: transfer.close()


if __name__ == '__main__': unittest.main()
