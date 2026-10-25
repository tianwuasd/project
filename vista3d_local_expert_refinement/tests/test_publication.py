"""CT publication contracts, with local files/Git and no real GitHub writes."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch,Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
from server_data import write_json,read_json,digest
from panel_publication import network,upload,code_sync,ssh_upload,updater
import task_panel


class PublicationTests(unittest.TestCase):
    def settings(self,root):
        key=root/'key';key.write_text('synthetic, not a real key')
        return dict(schema_version=1,proxy='',git_transport='github-ssh',ssh_key=str(key))

    def fixture(self,root):
        job=root/'job';work=job/'workspace';runtime=root/'runtime'
        plan=dict(directory=str(job),workspace=str(work),runtime=str(runtime),name='合成任务',tasks=['train'])
        write_json(job/'plan.json',plan)
        write_json(job/'state.json',dict(status='completed',plan_sha256=digest(job/'plan.json'),steps={'train':{'status':'completed'}}))
        for rel,data in [('07_experiments/formal/ct_holdG_B0_seed0/fold_0/checkpoint_final.pth',b'synthetic weights'),
                         ('07_experiments/formal/ct_holdG_B0_seed0/run.json',b'{"status":"completed"}'),
                         ('07_experiments/formal/ct_holdG_B0_seed0/training.log',b'epoch 1'),
                         ('09_reports/final_evaluation/summary.json',b'{"dice":0.5}'),
                         ('04_data/derived/private_cache.b2nd',b'private raw'),
                         ('08_results/predictions/case.nii.gz',b'private prediction')]:
            path=work/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        return job,runtime

    def test_network_is_atomic_private_and_process_scoped(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);settings=self.settings(root)
            before=dict(os.environ)
            with network.use(dict(settings,proxy='http://127.0.0.1:17891')),patch.object(network.shutil,'which',return_value='/usr/bin/nc'):
                env=network.child_env(ssh=True)
                self.assertIn('StrictHostKeyChecking=yes',env['GIT_SSH_COMMAND'])
                self.assertIn('17891',env['GIT_SSH_COMMAND'])
                self.assertEqual(network.routes(network.current()),(network.SSH_UPSTREAM,))
            self.assertEqual(dict(os.environ),before)
            with self.assertRaises(ValueError):network.validate(dict(settings,git_transport='public-mirrors'))
            network.save(root/'runtime',**settings)
            self.assertEqual(network.load(root/'runtime')['ssh_key'],settings['ssh_key'])

    def test_cancel_network_form_does_not_save(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime=Path(temp)/'absent'
            with patch('panel_publication.prompts.ask',side_effect=task_panel.Back),self.assertRaises(task_panel.Back):
                network.configure_interactive(runtime)
            self.assertFalse(runtime.exists())

    def test_archive_includes_weights_logs_metrics_but_no_images_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            job,runtime=self.fixture(Path(temp))
            with patch.object(upload,'CHUNK_BYTES',4096),redirect_stdout(io.StringIO()):
                package=upload.prepare_upload(job,runtime)
            _,manifest,assets=upload._validate_package(package)
            names=[v['archive_path'] for v in manifest['files']]
            self.assertTrue(any('checkpoint_final.pth' in n for n in names))
            self.assertTrue(any('summary.json' in n for n in names))
            self.assertFalse(any('.nii' in n or '04_data' in n or '.b2nd' in n for n in names))
            self.assertGreater(len(assets),1)
            part=package/manifest['assets'][0]['name'];part.write_bytes(b'tamper')
            with self.assertRaises(ValueError):upload._validate_package(package)

    def test_running_or_incomplete_job_cannot_be_packaged(self):
        with tempfile.TemporaryDirectory() as temp:
            job,runtime=self.fixture(Path(temp));state=read_json(job/'state.json')
            state['status']='running';write_json(job/'state.json',state)
            with self.assertRaises(ValueError):upload.prepare_upload(job,runtime)
            self.assertFalse((runtime/'uploads').exists())

    def test_code_allowlist_is_vista_only(self):
        self.assertTrue(code_sync.allowed('vista3d_local_expert_refinement/05_src/task_panel.py'))
        for path in ['chd-ct-reproduction/README.md','vista3d_local_expert_refinement/research/research.db',
                     'vista3d_local_expert_refinement/literature/paper.pdf','vista3d_local_expert_refinement/.codex/STATE.json',
                     'vista3d_local_expert_refinement/06_configs/private_credentials.json']:
            self.assertFalse(code_sync.allowed(path),path)

    def test_code_preview_checks_each_commit_not_only_final_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);repo=root/'repo';repo.mkdir();settings=self.settings(root)
            def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.STDOUT,text=True).strip()
            git('init','-b','main');git('config','user.name','Synthetic');git('config','user.email','synthetic@example.invalid')
            project=repo/'vista3d_local_expert_refinement';project.mkdir();(project/'README.md').write_text('first')
            git('add','.');git('commit','-m','base');base=git('rev-parse','HEAD')
            git('remote','add','origin','https://github.com/tianwuasd/project.git')
            (project/'05_src').mkdir();(project/'05_src/new.py').write_text('print(1)')
            git('add','.');git('commit','-m','code')
            def local(where,*args,**kwargs):
                if args[0]=='fetch':
                    git('update-ref',args[-1].split(':',1)[1],base)
                    return subprocess.CompletedProcess(args,0,'','')
                return subprocess.CompletedProcess(args,0,subprocess.check_output(['git','-C',str(where),*args],text=True),'')
            with network.use(settings),patch.object(ssh_upload,'git',side_effect=local),patch.object(ssh_upload,'check_local_routes'):
                preview=code_sync.prepare(project)
                self.assertEqual(preview['files'],['vista3d_local_expert_refinement/05_src/new.py'])
                secret=project/'secret.txt';secret.write_text('do not upload');git('add','.');git('commit','-m','bad')
                secret.unlink();git('add','.');git('commit','-m','remove')
                with self.assertRaises(ValueError):code_sync.prepare(project)

    def test_completed_package_retry_reuses_same_bytes_and_new_runtime_route(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);job,runtime=self.fixture(root)
            with redirect_stdout(io.StringIO()):package=upload.prepare_upload(job,runtime)
            before=(package/'manifest.json').read_bytes()
            settings=self.settings(root);network.save(runtime,**settings)
            def fake(*args):
                self.assertEqual(network.current()['ssh_key'],settings['ssh_key'])
                return 'https://github.com/tianwuasd/project/tree/vista3d-artifacts/synthetic'
            with patch.object(upload,'_publish',side_effect=fake),redirect_stdout(io.StringIO()):
                self.assertIn('vista3d-artifacts',upload.upload_package(package))
            self.assertEqual(before,(package/'manifest.json').read_bytes())

if __name__=='__main__':unittest.main()
