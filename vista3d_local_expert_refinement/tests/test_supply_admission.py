"""Dispatch and transfer regressions without CT, subprocesses or CUDA."""
import queue
import tempfile
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_src'))
import cpu_case_queue as cases
from live_resources import case_policy
from batch_transfer import BatchTransfer
from batch_pool import EpochBatchIterator


class DispatchTick(Exception):
    pass


class CaseAdmissionTests(unittest.TestCase):
    def dispatch(self, kind, *, resident=True, external=0):
        active=[]
        def refresh():
            count=len(active)
            facts=dict(cpu_budget=int(.8*(16-count)) if kind=='cpu' else 100,
                memory_budget_bytes=int(.8*(100-(10*count if resident else 0)-(external if count else 0))))
            return case_policy(facts,30,10,supported_threads=1)
        def start(*args,**kwargs):
            proc=SimpleNamespace(poll=lambda:None);active.append(proc);return proc
        with tempfile.TemporaryDirectory() as temp, \
             patch('read_admission.create_gate',return_value='synthetic'), \
             patch('read_admission.update_gate'), \
             patch.object(cases.subprocess,'Popen',side_effect=start), \
             patch.object(cases,'private_memory_bytes',return_value=10 if resident else 0,create=True), \
             patch.object(cases,'stop_owned_process'), \
             patch.object(cases.time,'sleep',side_effect=DispatchTick):
            with self.assertRaises(DispatchTick):
                cases.run([(str(i),['synthetic']) for i in range(30)],30,temp,lambda _:None,refresh=refresh)
        return len(active)

    def test_current_cpu_usage_is_not_subtracted_twice(self):
        # RAM intentionally abundant for this CPU-only scenario.
        active=[]
        def refresh():
            return case_policy(dict(cpu_budget=int(.8*(16-len(active))),memory_budget_bytes=10000),30,1,supported_threads=1)
        def start(*a,**k):
            p=SimpleNamespace(poll=lambda:None);active.append(p);return p
        with tempfile.TemporaryDirectory() as temp, patch('read_admission.create_gate',return_value='synthetic'),patch('read_admission.update_gate'),patch.object(cases.subprocess,'Popen',side_effect=start),patch.object(cases,'stop_owned_process'),patch.object(cases.time,'sleep',side_effect=DispatchTick):
            with self.assertRaises(DispatchTick):cases.run([(str(i),['synthetic']) for i in range(30)],30,temp,lambda _:None,refresh=refresh)
        self.assertEqual(len(active),12)

    def test_resident_ram_is_not_reserved_twice(self):
        self.assertEqual(self.dispatch('ram'),8)

    def test_unallocated_ram_still_has_a_reservation(self):
        self.assertEqual(self.dispatch('ram',resident=False),8)

    def test_external_ram_pressure_still_blocks_new_cases(self):
        self.assertEqual(self.dispatch('ram',external=80),1)

    def test_known_to_unknown_resources_preserves_serial_fallback(self):
        active=[]
        def refresh():
            return case_policy(dict(cpu_budget=4,memory_budget_bytes=100 if not active else None),10,10,supported_threads=1)
        def start(*a,**k):
            p=SimpleNamespace(poll=lambda:None);active.append(p);return p
        with tempfile.TemporaryDirectory() as temp, patch('read_admission.create_gate',return_value='synthetic'),patch('read_admission.update_gate'),patch.object(cases.subprocess,'Popen',side_effect=start),patch.object(cases,'stop_owned_process'),patch.object(cases.time,'sleep',side_effect=DispatchTick):
            with self.assertRaises(DispatchTick):cases.run([(str(i),['synthetic']) for i in range(10)],4,temp,lambda _:None,refresh=refresh)
        self.assertEqual(len(active),1)

    def test_private_credit_rejects_exit_pid_reuse_and_missing_uss(self):
        fake=SimpleNamespace(Process=lambda _:SimpleNamespace(create_time=lambda:1,memory_full_info=lambda:SimpleNamespace(uss=10)))
        with patch.dict(sys.modules,{'psutil':fake}):
            self.assertEqual(cases.private_memory_bytes(SimpleNamespace(pid=1,poll=lambda:0)),0)
            self.assertEqual(cases.private_memory_bytes(SimpleNamespace(pid=1,poll=lambda:None)),10)
            with patch.object(fake,'Process',side_effect=[SimpleNamespace(create_time=lambda:1,memory_full_info=lambda:SimpleNamespace(uss=10)),SimpleNamespace(create_time=lambda:2)]):
                self.assertEqual(cases.private_memory_bytes(SimpleNamespace(pid=1,poll=lambda:None)),0)
            with patch.object(fake,'Process',return_value=SimpleNamespace(create_time=lambda:1,memory_full_info=lambda:SimpleNamespace(rss=1000))):
                self.assertEqual(cases.private_memory_bytes(SimpleNamespace(pid=1,poll=lambda:None)),0)


class TransferAdmissionTests(unittest.TestCase):
    def transfer(self, existing=True, ready=False):
        transfer=BatchTransfer.__new__(BatchTransfer)
        transfer.closed=False;transfer.active=None;transfer.cuda=True;transfer.exhausted=False
        transfer.queue=[1] if existing else []
        future=lambda:SimpleNamespace(result=lambda:None)
        transfer.slots=[dict(data=None),dict(data='ready batch',ready=None,future=future())]
        transfer.stats=dict(cpu_supply_seconds=0.,prepared_batches=0,queue_peak=len(transfer.queue))
        transfer.device='synthetic'
        transfer.torch=SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda _:SimpleNamespace(wait_event=lambda _:None)))
        calls=[]
        class Source:
            def ready(self):return ready
            def __next__(self):
                calls.append('read')
                if existing and not ready:raise AssertionError('waited on later batch despite ready data')
                if len(calls)>1 and not ready:raise AssertionError('first batch delayed by later batch')
                return 'new batch'
        transfer.source=Source()
        def prepare(index,batch):transfer.slots[index].update(data=batch,ready=None)
        transfer._prepare=prepare
        def submit(fn,*args):fn(*args);return future()
        transfer.executor=SimpleNamespace(submit=submit)
        return transfer,calls

    def test_ready_batch_does_not_wait_for_later_supply(self):
        transfer,calls=self.transfer()
        self.assertEqual(next(transfer),'ready batch');self.assertEqual(calls,[])

    def test_first_batch_does_not_wait_for_second(self):
        transfer,calls=self.transfer(existing=False)
        self.assertEqual(next(transfer),'new batch');self.assertEqual(calls,['read'])

    def test_ready_lookahead_keeps_two_slot_overlap(self):
        transfer,calls=self.transfer(ready=True)
        self.assertEqual(next(transfer),'ready batch');self.assertEqual(calls,['read'])
        self.assertEqual(transfer.queue,[0]);self.assertEqual(transfer.stats['queue_peak'],2)

    def test_serial_source_readiness_never_generates_or_changes_cursor(self):
        pool=EpochBatchIterator(iter([1,2]),workers=0,phase_steps=2)
        try:
            self.assertFalse(pool.ready());self.assertEqual(pool.state_dict()['next_index'],0)
        finally:pool.close()

    def test_empty_worker_mailbox_readiness_does_not_wait(self):
        pool=EpochBatchIterator.__new__(EpochBatchIterator)
        pool._closed=False;pool._paused=False;pool.workers=1;pool.phase_steps=2;pool._next_index=0
        pool._ready={};pool._pending={};pool._results=queue.Queue()
        original=pool._results.get
        def get(*args,**kwargs):
            if kwargs.get('block',True):raise AssertionError('blocking receive')
            return original(*args,**kwargs)
        with patch.object(pool._results,'get',side_effect=get):
            self.assertFalse(pool.ready())

    def test_later_sequence_ready_does_not_authorize_missing_next(self):
        pool=EpochBatchIterator.__new__(EpochBatchIterator)
        pool._closed=False;pool._paused=False;pool.workers=1;pool.phase_steps=3;pool._next_index=0
        pool._ready={1:('connection','ok',{})};pool._pending={};pool._results=queue.Queue()
        self.assertFalse(pool.ready())
        self.assertEqual(pool._next_index,0)

    def test_shared_buffer_handshake_does_not_wait_for_batch_completion(self):
        from unittest.mock import Mock
        connection=Mock();pool=EpochBatchIterator.__new__(EpochBatchIterator)
        pool._closed=False;pool._paused=False;pool.workers=1;pool.phase_steps=2;pool._next_index=0
        pool._ready={};pool._pending={connection:(0,0)};pool._results=queue.Queue()
        pool._sequence_slots={0:0};pool._buffers={};pool.max_shared_bytes=0
        pool._results.put((connection,('buffer_request',0,16),0,0))
        self.assertFalse(pool.ready())
        connection.send.assert_called_once_with((None,'shared_budget'))
        self.assertEqual(pool._next_index,0)


if __name__=='__main__':unittest.main()
