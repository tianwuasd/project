import io
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
from server_data import read_json,write_json,digest
import task_runtime as jobs
import task_inputs as inputs
import task_update as updater


class InputTests(unittest.TestCase):
    def fixture(self,root):
        raw=root/'raw';raw.mkdir();source=root/'old';work=root/'new'
        split=dict(dataset_id=813,train=['Case1'],val=['Case2'],target=['Case3'])
        write_json(source/'00_admin/server_config.json',dict(data_root=str(raw),work_root=str(source),gpu_memory_gb=5,project_id=__import__('cohort').PROJECT_ID,protocol_sha256=__import__('server_data').protocol_hash()))
        (source/'.run.lock').touch()
        write_json(source/'04_data/manifests/splits_v1.json',dict(folds={'imagechd_fold0':split}))
        write_json(source/'04_data/manifests/case_inventory.json',dict(cases=[]))
        prep=source/'04_data/derived/nnUNet_preprocessed/Dataset813_imagechd_fold0'
        write_json(prep/'WholeheartBaselinePlans.json',{'configurations':{'3d_fullres':{'data_identifier':'cache'}}})
        current=Path(inputs.__file__).parent
        write_json(prep/'server_preparation_signature.json',{'implementation_sha256':__import__('preparation_identity').current(current)})
        for case in ['Case1','Case2']:
            for suffix in ['.b2nd','_seg.b2nd','.pkl']:
                path=prep/'cache'/(case+suffix);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'cache')
        write_json(source/'09_reports/imagechd_fold0_preparation.json',dict(status='passed',plan_sha256=digest(prep/'WholeheartBaselinePlans.json')))
        config=dict(input_job=str(source),runtime=str(root/'runtime'),data=str(raw),folds=['imagechd_fold0'],methods=['S1'],tasks=['prepare','train'],gpu_memory=5)
        return config,source,work,prep

    def test_private_cache_copy_does_not_change_source_and_keeps_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            config,source,work,prep=self.fixture(Path(temp))
            before={str(p.relative_to(source)):(p.read_bytes(),p.stat().st_mtime_ns) for p in source.rglob('*') if p.is_file()}
            value=inputs.inspect_input(config)
            plan=dict(config,workspace=str(work),source=value)
            inputs.copy_input(plan)
            self.assertEqual(read_json(work/'00_admin/model_selection.json')['methods'],['S1'])
            copied=work/prep.relative_to(source)/'cache/Case1.b2nd';copied.write_bytes(b'changed independently')
            self.assertEqual(before,{str(p.relative_to(source)):(p.read_bytes(),p.stat().st_mtime_ns) for p in source.rglob('*') if p.is_file()})

    def test_missing_cache_is_specific_mismatch_but_wrong_source_is_not(self):
        with tempfile.TemporaryDirectory() as temp:
            config,source,work,prep=self.fixture(Path(temp))
            with self.assertRaises(inputs.CacheMismatch):inputs.inspect_input(dict(config,gpu_memory=6))
            with self.assertRaises(ValueError) as caught:inputs.inspect_input(dict(config,data=str(Path(temp)/'other')))
            self.assertNotIsInstance(caught.exception,inputs.CacheMismatch)
            (prep/'cache/Case1.b2nd').unlink()
            with self.assertRaises(inputs.CacheMismatch):inputs.inspect_input(config)

    def test_resume_code_integrity_is_not_bypassed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);code=root/'code';code.mkdir();file=code/'a.py';file.write_text('old')
            write_json(root/'plan.json',dict(code=str(code)))
            write_json(root/'code_manifest.json',{'a.py':digest(file)})
            inputs.verify_resume(root,'none')
            file.write_text('changed')
            with self.assertRaises(ValueError):inputs.verify_resume(root,'none')

    def test_append_keeps_original_models_and_completed_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);job=root/'job';code=job/'code';code.mkdir(parents=True)
            plan=dict(name='old',directory=str(job),runtime=str(root/'runtime'),workspace=str(job/'workspace'),code=str(code),
                      tasks=['prepare','train'],methods=['S1'],folds=['imagechd_fold0'],gpus=['GPU-0'],gpu_count=1)
            write_json(job/'plan.json',plan);write_json(job/'code_manifest.json',{})
            state=dict(status='completed',plan_sha256=digest(job/'plan.json'),steps={t:{'status':'completed'} for t in plan['tasks']})
            write_json(job/'state.json',state)
            with patch.object(jobs,'launch',return_value=0):jobs.append_tasks(job,['predict'])
            self.assertEqual(jobs.load_plan(job)['methods'],['S1'])
            current=read_json(job/'state.json')
            self.assertEqual(current['steps']['train']['status'],'completed')
            self.assertEqual(current['steps']['predict']['status'],'pending')
            self.assertEqual(len(list((job/'revisions').glob('*/plan.json'))),1)


class UpdateTests(unittest.TestCase):
    def archive(self,name,link=False):
        data=io.BytesIO()
        with tarfile.open(fileobj=data,mode='w') as archive:
            member=tarfile.TarInfo(name)
            if link:member.type=tarfile.SYMTYPE;member.linkname='/outside'
            else:member.size=2
            archive.addfile(member,None if link else io.BytesIO(b'ok'))
        return data.getvalue()

    def test_archive_cannot_escape_or_include_siblings(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name,link in [('AAOCA/x.py',False),('vista3d_local_expert_refinement/../../bad',False),('vista3d_local_expert_refinement/link',True)]:
                with self.assertRaises(ValueError):updater.extract_project(self.archive(name,link),root)
            updater.extract_project(self.archive('vista3d_local_expert_refinement/05_src/a.py'),root)
            self.assertEqual((root/'05_src/a.py').read_text(),'ok')

    def test_release_pointer_and_all_file_hashes_are_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime=Path(temp);release=runtime/'releases'/'abc';entry=release/'05_src/task_panel.py'
            entry.parent.mkdir(parents=True);entry.write_text('# code')
            write_json(release/'release_manifest.json',{'05_src/task_panel.py':digest(entry)})
            write_json(runtime/'release.json',{'path':str(release)})
            self.assertEqual(updater.selected_entry(runtime),entry)
            entry.write_text('# different')
            with self.assertRaises(ValueError):updater.selected_entry(runtime)

    def test_authenticated_origin_publishes_isolated_release(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);repository=root/'repo';repository.mkdir();runtime=root/'runtime'
            data=io.BytesIO()
            with tarfile.open(fileobj=data,mode='w') as archive:
                for relative in ['05_src/task_panel.py','05_src/task_worker.py','05_src/task_runtime.py',
                                 '06_configs/baseline.json','04_data/manifests/README.md','bootstrap_server.sh','start_server.sh']:
                    member=tarfile.TarInfo(updater.SUBDIR+'/'+relative);member.size=2
                    archive.addfile(member,io.BytesIO(b'ok'))
            routes=[]
            def fake_git(where,*args,**kwargs):
                result=SimpleNamespace(returncode=0,stdout='',stderr=b'')
                if args==('rev-parse','--show-toplevel'):result.stdout=str(repository)
                elif args[:2]==('config','--get'):result.stdout=updater.UPSTREAM
                elif args==('rev-parse','HEAD'):result.stdout='old'
                elif args[:2]==('ls-remote','--get-url'):result.stdout=args[2]
                elif args[0]=='ls-remote':raise subprocess.TimeoutExpired(args,12)
                elif args[0]=='fetch':
                    routes.append(updater.UPSTREAM)
                elif args==('rev-parse','refs/heads/vista-candidate'):result.stdout='new'
                elif args[0]=='ls-tree':
                    result.stdout=b''.join(b'100644 blob '+b'a'*40+b'\t'+(updater.SUBDIR+'/'+name).encode()+b'\0'
                        for name in ['05_src/task_panel.py','05_src/task_worker.py','05_src/task_runtime.py',
                                     '06_configs/baseline.json','04_data/manifests/README.md','bootstrap_server.sh','start_server.sh'])
                elif args[:2]==('cat-file','--batch-all-objects'):
                    result.stdout='a'*40+' blob\n'
                elif args[0]=='archive':result.stdout=data.getvalue()
                return result
            with patch.object(updater,'git',side_effect=fake_git):updater.update(runtime)
            self.assertEqual(routes,[updater.UPSTREAM])
            pointer=read_json(runtime/'release.json')
            self.assertEqual(pointer['commit'],'new')
            self.assertTrue((Path(pointer['backup'])/'code.tar').is_file())
            self.assertTrue(updater.selected_entry(runtime).is_file())
            self.assertEqual(list(repository.iterdir()),[])

    def test_all_failed_routes_keep_existing_checkout_and_no_pointer(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);repo=root/'repo';repo.mkdir();runtime=root/'runtime'
            def fake_git(where,*args,**kwargs):
                result=SimpleNamespace(returncode=0,stdout='',stderr=b'')
                if args==('rev-parse','--show-toplevel'):result.stdout=str(repo)
                elif args[:2]==('config','--get'):result.stdout=updater.UPSTREAM
                elif args==('rev-parse','HEAD'):result.stdout='old'
                elif args[0]=='archive':result.stdout=b'backup'
                elif args[:2]==('ls-remote','--get-url'):result.stdout=args[2]
                elif args[0]=='ls-remote' or args[0]=='fetch':raise subprocess.TimeoutExpired(args,kwargs['timeout'])
                return result
            with patch.object(updater,'git',side_effect=fake_git),self.assertRaises(ValueError):updater.update(runtime)
            self.assertFalse((runtime/'release.json').exists())
            self.assertEqual(list(repo.iterdir()),[])
            write_json(runtime/'release.json',{'path':str(runtime.parent)})
            with self.assertRaises(ValueError):updater.selected_entry(runtime)


if __name__ == '__main__':unittest.main()
