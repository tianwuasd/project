"""Regression cases from the independent final review; no patient inputs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))


class ReviewRegressions(unittest.TestCase):
    def test_report_failure_preserves_published_version_and_obeys_lock(self):
        from dataset1282_analysis.reporting import build_report
        self.assertIn('home',__import__('inspect').signature(build_report).parameters)
        from dataset1282_analysis.smoke import synthetic_fixture
        from dataset1282_analysis.config import resolve_config
        from dataset1282_analysis.engine.runner import run_analysis
        from dataset1282_analysis.contracts import read_json,sha256_file
        from dataset1282_analysis.observability.status import operation_lock
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);data,excel=synthetic_fixture(base/'input');home=base/'home'
            config=resolve_config({'data_root':str(data),'metadata_excel':str(excel),'work_root':str(base/'work')},None,ROOT)
            run=run_analysis(config,ROOT,home,run_id='report-test')
            directory=Path(run['run_dir'])
            first=build_report(directory,home=home,render_examples=False)
            version=Path(first['report']).parent
            hashes={p.name:sha256_file(p) for p in version.iterdir() if p.is_file()}
            pointer=read_json(directory/'report-current.json')
            rows=read_json(directory/'metadata_records.json');rows[0]['ManufacturerModelName']='changed synthetic source'
            (directory/'metadata_records.json').write_text(json.dumps(rows),encoding='utf-8')
            with patch('dataset1282_analysis.reporting._plots',side_effect=RuntimeError('synthetic plotting failure')):
                with self.assertRaises(RuntimeError):build_report(directory,home=home,render_examples=False)
            self.assertEqual(read_json(directory/'report-current.json'),pointer)
            self.assertEqual({p.name:sha256_file(p) for p in version.iterdir() if p.is_file()},hashes)
            with operation_lock(home,'analysis'):
                with self.assertRaises(RuntimeError):build_report(directory,home=home,render_examples=False)

    def test_qc_reads_label_planes_without_full_float64_volume(self):
        import numpy as np
        from dataset1282_analysis.smoke import synthetic_fixture
        from dataset1282_analysis.config import resolve_config
        from dataset1282_analysis.inputs import load_cases
        from dataset1282_analysis.engine.worker import analyze_case
        from dataset1282_analysis.reporting import _qc_examples
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);data,excel=synthetic_fixture(base/'input')
            config=resolve_config({'data_root':str(data),'metadata_excel':str(excel),'work_root':str(base/'work')},None,ROOT)
            result=analyze_case({'case':load_cases(config)[0].to_dict(),'config':config,'contract':'test','stage':'full','metadata_link_status':'matched'})
            out=base/'report';out.mkdir();original=np.empty
            def bounded(shape,*a,**kw):
                if isinstance(shape,tuple) and shape==(12,14,8): raise AssertionError('full-volume QC allocation')
                return original(shape,*a,**kw)
            with patch.object(np,'empty',side_effect=bounded):
                examples=_qc_examples(out,[result],[],{'case_rows':[]},1)
            self.assertEqual(examples[0]['status'],'rendered')

    def test_hard_killed_supervisor_cannot_leave_writing_workers(self):
        from dataset1282_analysis.engine import worker
        self.assertTrue(hasattr(worker,'start_owner_watchdog'),'worker parent-death guard is missing')
        import psutil
        from dataset1282_analysis.observability.status import operation_lock,process_birth
        with tempfile.TemporaryDirectory() as d:
            home=Path(d)/'home';ready=Path(d)/'worker-ready.json'
            env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')
            parent=subprocess.Popen([sys.executable,'-B',str(ROOT/'tests/orphan_helper.py'),str(home),str(ready)],env=env,
                                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            owned=[]
            try:
                deadline=time.monotonic()+10
                while not (ready.exists() and (home/'supervisor.json').exists()) and time.monotonic()<deadline:time.sleep(.05)
                self.assertTrue(ready.exists(),'synthetic child did not start')
                info=json.loads((home/'supervisor.json').read_text());wid=json.loads(ready.read_text())['pid']
                owned=[info['pid'],info['worker_launcher'],wid]
                before=process_birth(wid)
                psutil.Process(info['pid']).kill()  # Deliberately bypass finally.
                parent.wait(timeout=5)
                deadline=time.monotonic()+5
                while process_birth(wid)==before and time.monotonic()<deadline:time.sleep(.05)
                self.assertNotEqual(process_birth(wid),before,'orphan worker survived supervisor death')
                with operation_lock(home,'resume-after-hard-exit'):pass
            finally:
                for pid in owned:
                    try:psutil.Process(pid).kill()
                    except psutil.NoSuchProcess:pass
                if parent.poll() is None:parent.kill();parent.wait(timeout=5)

    def test_stale_decision_cannot_rename_a_new_live_lock(self):
        from dataset1282_analysis.observability import status
        with tempfile.TemporaryDirectory() as d:
            home=Path(d);(home/'.runtime').mkdir();(home/'.runtime/operation.lock').write_text('{}')
            first=threading.Event();second=threading.Event();release=threading.Event();errors=[]
            def acquire(which):
                try:
                    with status.operation_lock(home,'test'):
                        (first if which==1 else second).set();release.wait(2)
                except RuntimeError: pass
                except Exception as e: errors.append(e)
            with patch.object(status,'lock_active',return_value=False):
                a=threading.Thread(target=acquire,args=(1,));a.start();self.assertTrue(first.wait(1))
                b=threading.Thread(target=acquire,args=(2,));b.start()
                overlapping=second.wait(.25)
                release.set();a.join(3);b.join(3)
            self.assertFalse(overlapping,'two operations simultaneously owned the project lock')
            self.assertFalse(errors)

    def test_crashed_partial_lock_is_recoverable(self):
        from dataset1282_analysis.observability.status import operation_lock
        with tempfile.TemporaryDirectory() as d:
            home=Path(d);(home/'.runtime').mkdir();(home/'.runtime/operation.lock').write_bytes(b'')
            with operation_lock(home,'test'): pass
            self.assertFalse((home/'.runtime/operation.lock').exists())

    def test_work_defaults_are_stable_across_releases(self):
        from dataset1282_analysis.config import resolve_config
        with tempfile.TemporaryDirectory() as d:
            home=Path(d)/'app';candidate=home/'code_versions'/('a'*40)
            for root in (home,candidate):
                (root/'configs').mkdir(parents=True);shutil.copyfile(ROOT/'configs/default.json',root/'configs/default.json')
            a=resolve_config({},home/'project.local.json',home)
            b=resolve_config({},home/'project.local.json',candidate)
            self.assertEqual(a['work_root'],b['work_root']);self.assertEqual(a['environment'],b['environment'])

    def test_task_config_selects_the_interpreter_before_import(self):
        import bootstrap
        self.assertTrue(hasattr(bootstrap,'resolve_launch_config'),'launch configuration must be shared with CLI')
        with tempfile.TemporaryDirectory() as d:
            home=Path(d);a=home/'envA';b=home/'envB'
            (home/'project.local.json').write_text(json.dumps({'environment':str(a),'work_root':str(home.parent/'workA')}))
            task=home/'task.json';task.write_text(json.dumps({'environment':str(b),'work_root':str(home.parent/'workB')}))
            config=bootstrap.resolve_launch_config(home,ROOT,['--config',str(task),'inventory'])
            self.assertEqual(config['environment'],str(b))

    def test_invalid_many_value_labels_have_bounded_explicit_outcome(self):
        import numpy as np
        import nibabel as nib
        from dataset1282_analysis.labels import analyze_label
        from dataset1282_analysis.contracts import sha256_file
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'invalid.nii.gz';x=(np.arange(256,dtype=np.float64)+.25).reshape(8,8,4)
            nib.save(nib.Nifti1Image(x,np.eye(4)),p)
            result=analyze_label(p,{'connectivity':1,'max_label_values':16,'chunk_slices':1})
            self.assertEqual(result.get('per_class_statistics_status'),'not_computed_cardinality_limit')
            self.assertEqual(result['noninteger_count'],256)
            self.assertEqual(result['foreground_voxels'],256)
            self.assertEqual(result['sha256'],sha256_file(p))
            self.assertEqual(result['classes'],[])

    def test_no_finite_image_is_retained_without_sorting_none(self):
        from dataset1282_analysis.reporting import _choose_examples
        rows=[]
        for index,value in enumerate((None,1.0)):
            rows.append({'case_id':str(index),'status':'complete','collection':'native','metadata_link_status':'matched',
                         'image':dict.fromkeys(('mean','std','min','max'),value),'label':{},
                         'geometry':{'image':{'dtype':'float64'},'label':{'dtype':'int8'},'metadata_shape_conflicts':[]}})
        chosen=_choose_examples(rows,[],{'case_rows':[]},24)
        self.assertEqual({r['case_id'] for r,reasons in chosen},{'0','1'})
        self.assertTrue(any('finite' in reason for row,reasons in chosen if row['case_id']=='0' for reason in reasons))

    def test_metadata_report_scope_has_no_unrequested_pending_voxels(self):
        from dataset1282_analysis.reporting import aggregate_results
        self.assertIn('stage',__import__('inspect').signature(aggregate_results).parameters)
        result=aggregate_results([],3,stage='metadata')
        self.assertEqual(result['denominators']['pending'],0)
        self.assertEqual(result['denominators']['voxel_analysis_not_requested'],3)

    def test_top_level_rollback_survives_corrupt_active_release(self):
        from dataset1282_analysis.maintenance import source_files,write_manifest
        with tempfile.TemporaryDirectory() as d:
            home=Path(d)/'app';home.mkdir()
            for p in source_files(ROOT):
                dest=home/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
            write_manifest(home)
            release=home/'code_versions'/('a'*40);release.mkdir(parents=True)
            for p in source_files(home)+[home/'release-manifest.json']:
                dest=release/p.relative_to(home);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
            (home/'project.local.json').write_text(json.dumps({'environment':sys.prefix,'work_root':str(Path(d)/'work')}))
            (home/'current.json').write_text(json.dumps({'commit':'b'*40}))
            result=subprocess.run([sys.executable,'-B',str(home/'update.py'),'--rollback','a'*40],capture_output=True,text=True,encoding='utf-8')
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(json.loads((home/'current.json').read_text())['commit'],'a'*40)


if __name__=='__main__':unittest.main()
