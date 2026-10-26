from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch, Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
import task_update
import task_runtime
from server_data import read_json

class PrivateUpdateTests(unittest.TestCase):
    def test_two_updates_recover_evicted_cache_with_checkout_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); upstream=root/'upstream'; upstream.mkdir()
            def local_git(where,*args):
                return subprocess.check_output(['git','-C',str(where),*args],stderr=subprocess.STDOUT).decode().strip()
            local_git(upstream,'init','-b','main')
            local_git(upstream,'config','user.name','Synthetic Test')
            local_git(upstream,'config','user.email','synthetic@example.invalid')
            project=upstream/task_update.SUBDIR
            for relative in ['05_src/task_panel.py','05_src/task_worker.py','bootstrap_server.sh',
                             '06_configs/baseline.json','04_data/manifests/README.md']:
                file=project/relative;file.parent.mkdir(parents=True,exist_ok=True);file.write_text('# v1\n')
            local_git(upstream,'add','.');local_git(upstream,'commit','-m','initial')
            original=root/'original'
            local_git(root,'clone','--local',str(upstream),str(original))
            local_git(original,'remote','set-url','origin',task_update.UPSTREAM)
            initial=local_git(original,'rev-parse','HEAD')
            real_git=task_update.git
            def offline_git(where,*args,**kwargs):
                if args[:2]==('ls-remote','--get-url'):
                    return subprocess.CompletedProcess(args,0,stdout=args[2],stderr='')
                # The test transport is local; production still uses its allowlisted URL.
                args=tuple(str(upstream) if a==task_update.UPSTREAM else a for a in args)
                return real_git(where,*args,**kwargs)
            runtime=root/'runtime'
            raw=root/'raw';raw.mkdir()
            with patch.object(task_update,'CODE',original/task_update.SUBDIR),patch.object(task_update,'git',side_effect=offline_git):
                for version in [2,3]:
                    if version == 3:
                        cache=(runtime/'update_cache.git').resolve()
                        assert cache.is_relative_to(root.resolve()) and cache.name=='update_cache.git'
                        evicted=(runtime/'evicted_cache.git').resolve()
                        assert evicted.is_relative_to(root.resolve())
                        cache.rename(evicted)
                    (project/'05_src/task_panel.py').write_text(f'# v{version}\n')
                    local_git(upstream,'add','.');local_git(upstream,'commit','-m',f'version {version}')
                    task_update.update(runtime)
                    self.assertEqual(task_update.selected_entry(runtime).read_text(),f'# v{version}\n')
                    self.assertEqual(local_git(original,'rev-parse','HEAD'),initial)
                    # Selecting a release is not enough: its panel must still be
                    # able to submit and discover a task using the same runtime.
                    release=task_update.selected_entry(runtime).parents[1]
                    config=dict(name=f'after-update-{version}',runtime=str(runtime),results=str(root/'results'),
                                data=str(raw),tasks=['environment'],methods=['B0'],folds=['ct_holdG'],
                                gpus=[],gpu_count=0,gpu_memory=20,input_job=None)
                    with patch.object(task_runtime,'CODE',release),patch.object(task_runtime,'launch',return_value=0):
                        task_runtime.submit(config)
                        job=task_runtime.latest(runtime)
                        self.assertEqual(task_runtime.load_plan(job)['name'],f'after-update-{version}')
                        self.assertEqual(task_runtime.history(runtime)[0]['directory'],str(job))
                        self.assertEqual(read_json(job/'plan.json')['runtime'],str(runtime.resolve()))

    def test_only_authenticated_official_routes(self):
        self.assertEqual(task_update.SUBDIR,'vista3d_local_expert_refinement')
        self.assertIn('ssh://git@ssh.github.com:443/tianwuasd/project.git',task_update.ORIGINS)
        # Historical mirror identity is accepted, while downloads stay official.
        self.assertIn('https://gh-proxy.com/https://github.com/tianwuasd/project.git',task_update.ORIGINS)
        process=Mock(returncode=0);process.communicate.return_value=(b'ok',b'')
        with patch.object(task_update.subprocess,'Popen',return_value=process) as launch:
            task_update.git(Path('.'),'status')
        command=launch.call_args.args[0]
        self.assertNotIn('credential.helper=',command)
        self.assertNotIn('http.extraHeader=',command)

if __name__=='__main__':unittest.main()
