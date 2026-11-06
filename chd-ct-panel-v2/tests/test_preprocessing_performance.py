"""Correctness and resource boundaries for execution-only preprocessing changes."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


class PerformanceBoundaryTests(unittest.TestCase):
    def test_atomic_status_retries_transient_windows_reader_conflict(self):
        from chd_stunet.observability.runlog import write_json
        import chd_stunet.observability.runlog as runlog
        scratch=ROOT/'outputs/tests-tmp';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path=Path(directory)/'status.json';write_json(path,dict(old=True))
            original=os.replace;calls=[]
            def replace(source,target):
                calls.append(1)
                if len(calls)<3:
                    error=PermissionError('reader sharing conflict');error.winerror=32;raise error
                original(source,target)
            with patch.object(runlog.os,'replace',side_effect=replace):write_json(path,dict(new=True))
            self.assertEqual(json.loads(path.read_text()),dict(new=True))
            self.assertEqual(len(calls),3)

    def test_capacity_preview_uses_native_rounding_and_unbounded_integer_product(self):
        from chd_stunet.preprocessing.capacity import case_scale
        row=dict(case_id='large',cropped_shape_zyx=[500,512,512],spacing_zyx=[1.5,.91796875,.91796875])
        plan=dict(target_spacing_before_transpose=[.75,.30859375,.30859375],transpose_forward=[0,1,2])
        value=case_scale(row,plan)
        self.assertEqual(value['predicted_shape_zyx'],[1000,1523,1523])
        self.assertEqual(value['output_voxels'],2319529000)
        self.assertGreater(value['output_voxels'],2**31)
        self.assertEqual(value['image_float32_bytes'],4*2319529000)

    def test_chunk_readback_visits_edges_once_and_detects_corruption(self):
        from chd_stunet.preprocessing.verification import verify_full_array
        expected=np.arange(5*7*9,dtype=np.float32).reshape(5,7,9)
        class Stored:
            shape=expected.shape;dtype=expected.dtype;chunks=(2,3,4)
            def __init__(self):self.visits=np.zeros(self.shape,dtype=int)
            def __getitem__(self,key):self.visits[key]+=1;return expected[key]
        stored=Stored();progress=[]
        row=verify_full_array(stored,expected,lambda a,b:progress.append((a,b)))
        np.testing.assert_array_equal(stored.visits,np.ones(expected.shape))
        self.assertEqual(row['verified_voxels'],expected.size)
        self.assertEqual(progress[-1],(expected.size,expected.size))
        class Corrupted(Stored):
            def __getitem__(self,key):
                result=super().__getitem__(key).copy()
                if key[0].stop==5 and key[1].stop==7 and key[2].stop==9:result.flat[-1]=-1
                return result
        with self.assertRaisesRegex(ValueError,'read-back'):verify_full_array(Corrupted(),expected)

    def test_histogram_sampler_matches_original_per_class_counts_and_rng(self):
        from chd_stunet.preprocessing.stunet_s import bounded_locations
        # int16 selects the old per-class counting path. Both cases cross the
        # one-million-voxel boundary and include absent/out-of-range labels.
        seg=np.random.default_rng(22).integers(-1,9,size=(1,101,103,107),dtype=np.int8)
        classes=[1,7,[-1,0,1,2,3,4,5,6,7],[-129,0,0,128],99]
        a=bounded_locations(seg,classes);b=bounded_locations(seg.astype(np.int16),classes)
        self.assertEqual(list(a),list(b))
        for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_inner_threads_are_charged_to_single_stage_budget(self):
        from chd_stunet.acceleration.preprocess_cpu import policy,admits
        gib=1024**3
        facts=dict(cpu_budget=10,available_memory_bytes=100*gib,memory_budget_bytes=80*gib)
        stage=policy(facts,[dict(estimate=gib)]*50,case_threads=4)
        self.assertEqual((stage['workers'],stage['threads_per_worker'],stage['total_cpu_slots']),(3,4,10))
        active=[dict(estimate=gib,private=0,threads=4)]*2
        self.assertFalse(admits(stage,stage,active,gib,4))
        self.assertTrue(admits(stage,stage,active,gib,2))

    def test_partial_thread_team_can_fill_remaining_stage_cpu_slots(self):
        from chd_stunet.acceleration.preprocess_cpu import policy,admits
        gib=1024**3
        facts=dict(cpu_budget=6,available_memory_bytes=100*gib,memory_budget_bytes=80*gib)
        stage=policy(facts,[dict(estimate=gib)]*20,case_threads=4)
        self.assertEqual(stage['workers'],2)
        self.assertEqual(stage['total_cpu_slots'],6)
        self.assertTrue(admits(stage,stage,[dict(estimate=gib,private=0,threads=4)],gib,2))

    def test_phase_identity_and_stale_progress_are_checked(self):
        from chd_stunet.observability.phase_timer import PhaseTimer
        from chd_stunet.acceleration.preprocess_cpu import case_activity
        import psutil
        scratch=ROOT/'outputs/tests-tmp';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            result=Path(directory)/'case.json';timer=PhaseTimer(result)
            with contextlib.redirect_stdout(io.StringIO()),timer.phase('image_resampling'):
                timer.progress(3,8)
            row=json.loads(timer.path.read_text(encoding='utf-8'))
            self.assertEqual(row['worker_pid'],os.getpid())
            class Own:
                pid=os.getpid()
                def poll(self):return None
            item=dict(process=Own(),born=psutil.Process().create_time(),
                name='case',job=dict(result=str(result)),started=0)
            self.assertEqual(case_activity(item)['phase'],'image_resampling')
            row['worker_born']-=1;timer.path.write_text(json.dumps(row),encoding='utf-8')
            self.assertEqual(case_activity(item)['phase'],'starting_or_no_owned_phase')

    def test_next_fold_backfills_while_long_case_is_running(self):
        from chd_stunet.acceleration.preprocess_cpu import execute_jobs
        from chd_stunet.observability.runlog import RunLog,write_json
        gib=1024**3;scratch=ROOT/'outputs/tests-tmp';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            folder=Path(directory)
            entry=folder/'worker.py'
            entry.write_text('import json,os,sys,time\nfrom pathlib import Path\n'
                'j=json.loads(Path(sys.argv[1]).read_text());began=time.time();time.sleep(j["delay"])\n'
                'Path(j["result"]).write_text(json.dumps(dict(success=True,worker_pid=os.getpid(),'
                'began=began,ended=time.time(),threads=j["cpu_threads"])))\n',encoding='utf-8')
            jobs=[]
            for name,delay,estimate in [('fold0_big',1.4,3*gib),('fold0_small',.05,gib),('fold1_small',.05,gib)]:
                task=folder/(name+'.task.json');result=folder/(name+'.json')
                write_json(task,dict(delay=delay,result=str(result)))
                jobs.append(dict(name=name,estimate=estimate,entry=str(entry),task=str(task),result=str(result)))
            facts=dict(cpu_budget=8,available_memory_bytes=16*gib,memory_budget_bytes=12*gib)
            with patch('chd_stunet.acceleration.preprocess_cpu.snapshot',return_value=facts),contextlib.redirect_stdout(io.StringIO()):
                execute_jobs(jobs,folder,RunLog(folder),stage_name='shared',workers=2,case_threads=4)
            rows={j['name']:json.loads(Path(j['result']).read_text()) for j in jobs}
            self.assertLess(rows['fold1_small']['began'],rows['fold0_big']['ended'])
            self.assertTrue(all(row['threads']==4 for row in rows.values()))

    def test_supervisor_builds_one_queue_with_independent_fold_fits(self):
        from types import SimpleNamespace
        from chd_stunet.preprocessing.run import run_prepare
        from chd_stunet.preprocessing.registry import get_method
        method=get_method('stunet-s-partial')
        ids=[f'main_{i:03}' for i in range(96)]+['ct_1170']+[f'aux_{i:02}' for i in range(13)]
        scratch=ROOT/'outputs/tests-tmp';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            folder=Path(directory);raw=folder/'raw';raw.mkdir();calls=[]
            pairs=[dict(case_id=cid,image=str(raw/(cid+'.nii.gz')),label=str(raw/(cid+'_label.nii.gz'))) for cid in ids]
            def execute(jobs,output,log,*,stage_name,completed,**kwargs):
                if stage_name=='audit':
                    for job in jobs:
                        task=json.loads(Path(job['task']).read_text());cid=task['pair']['case_id']
                        sample=Path(task['sample_path']);sample.parent.mkdir(parents=True,exist_ok=True)
                        np.save(sample,np.arange(20,dtype=np.float32))
                        completed(job['name'],dict(success=True,case_id=cid,pair=task['pair'],
                            source_hashes=dict(image=cid,label=cid+'_label'),sample_path=str(sample),
                            missing_label_ids=[5] if cid.startswith('aux_') else [],
                            spacing_xyz=[1,1,1],spacing_zyx=[1,1,1],spacing_evidence={},
                            cropped_shape_zyx=[4,4,4],shape_xyz=[4,4,4]))
                else:
                    calls.append((stage_name,jobs,kwargs))
                    for job in jobs:completed(job['name'],dict(success=True,cohort_group='main'))
            signature=dict(identity='synthetic',spreadsheet_sha256=method['spacing_excel_sha256'],
                           producer_sha256={'preprocess_worker.py':'synthetic'})
            prefix='chd_stunet.preprocessing.run.'
            with patch(prefix+'discover_pairs',return_value=dict(pairs=pairs,missing_labels=[],orphan_labels=[])), \
                 patch(prefix+'build_signature',return_value=signature), \
                 patch(prefix+'read_spacing_excel',return_value={cid:dict(spacing_mm=[1,1,1],excel_row=1) for cid in ids}), \
                 patch(prefix+'descriptor_evidence',return_value=({},{})), \
                 patch(prefix+'nib.load',return_value=SimpleNamespace(shape=(4,4,4))), \
                 patch(prefix+'execute_jobs',side_effect=execute),contextlib.redirect_stdout(io.StringIO()):
                summary=run_prepare(raw,folder/'run',method_name='stunet-s-partial',selected_cases=['main_000'],case_threads=4)
            self.assertEqual(summary['processed_cases'],3)
            self.assertEqual(len(calls),1)
            stage,jobs,kwargs=calls[0];self.assertEqual(kwargs['case_threads'],4)
            tasks=[json.loads(Path(job['task']).read_text()) for job in jobs]
            self.assertEqual({task['plan']['fold'] for task in tasks},{0,1,2})
            self.assertEqual(len({tuple(task['plan']['fit_ids']) for task in tasks}),3)
            for task in tasks:
                self.assertTrue(all(cid.startswith('main_') for cid in task['plan']['fit_ids']))


if __name__=='__main__':unittest.main()
