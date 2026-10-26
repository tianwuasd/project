"""Real local Git checks: only runtime blobs, offline archive, bounded stalls."""
import contextlib
import io
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'05_src'))
import task_update as updater
from panel_publication import network


class UpdateScopeTests(unittest.TestCase):
    def test_old_entry_can_bootstrap_only_the_new_updater_without_checkout_changes(self):
        path=Path(__file__).resolve().parents[1]/'tools/bootstrap_code_only_update.py'
        self.assertTrue(path.is_file(),'Missing small-download bootstrap for old full-download entry')
        spec=importlib.util.spec_from_file_location('vista_small_bootstrap',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        import task_runtime
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);git,upstream,original,head,excluded=self.fixture(root)
            target=upstream/updater.SUBDIR/'05_src/task_update.py'
            target.write_bytes(Path(updater.__file__).read_bytes())
            git(upstream,'add','.');git(upstream,'commit','-qm','new updater')
            runtime=root/'runtime';runtime.mkdir()
            with patch.object(task_runtime,'CODE',original/updater.SUBDIR),patch.object(network,'routes',return_value=(upstream.as_uri(),)):
                module.bootstrap(Path(updater.__file__).resolve().parents[1],runtime)
            self.assertEqual(updater.selected_entry(runtime).read_bytes(),b'# code v2 $Format:%H$\n')
            self.assertEqual(git(original,'rev-parse','HEAD'),head)
            self.assertEqual(git(original,'status','--porcelain'),'')

    def fixture(self, root):
        upstream=root/'upstream';upstream.mkdir()
        def git(where,*args):
            return subprocess.check_output(['git','-C',str(where),*args],stderr=subprocess.STDOUT).decode().strip()
        git(upstream,'init','-b','main')
        git(upstream,'config','user.name','Synthetic Test')
        git(upstream,'config','user.email','synthetic@example.invalid')
        git(upstream,'config','uploadpack.allowFilter','true')
        project=upstream/updater.SUBDIR
        files=['05_src/task_panel.py','05_src/task_worker.py','05_src/task_runtime.py','bootstrap_server.sh','start_server.sh',
               '06_configs/baseline.json','04_data/manifests/README.md']
        for relative in files:
            path=project/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'# code v1\n')
        unrelated=upstream/'other-project'/'large.bin';unrelated.parent.mkdir()
        unrelated.write_bytes(bytes(range(256))*8192)
        paper=project/'literature'/'paper.pdf';paper.parent.mkdir();paper.write_bytes(b'paper fixture'*100000)
        attributes=upstream/'.gitattributes';attributes.write_bytes(b'*.py export-subst\n')
        git(upstream,'add','.');git(upstream,'commit','-qm','initial')
        original=root/'original';git(root,'clone','--local',str(upstream),str(original))
        git(original,'remote','set-url','origin',updater.UPSTREAM)
        original_head=git(original,'rev-parse','HEAD')
        (project/'05_src/task_panel.py').write_bytes(b'# code v2 $Format:%H$\n')
        git(upstream,'add','.');git(upstream,'commit','-qm','code update')
        excluded=[git(upstream,'rev-parse','HEAD:'+name) for name in
                  ['other-project/large.bin',updater.SUBDIR+'/literature/paper.pdf','.gitattributes']]
        return git,upstream,original,original_head,excluded

    def test_only_selected_runtime_blobs_are_downloaded_and_published(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);git,upstream,original,head,excluded=self.fixture(root)
            runtime=root/'runtime'
            with patch.object(updater,'CODE',original/updater.SUBDIR),patch.object(network,'routes',return_value=(upstream.as_uri(),)):
                updater.update(runtime)
            cache=runtime/'update_cache.git'
            available=set(git(cache,'cat-file','--batch-all-objects','--batch-check=%(objectname)').splitlines())
            self.assertTrue(set(excluded).isdisjoint(available), 'Unrelated projects, literature or archive attributes were downloaded')
            release=updater.selected_entry(runtime).parents[1]
            self.assertEqual((release/'05_src/task_panel.py').read_bytes(),b'# code v2 $Format:%H$\n')
            self.assertFalse((release/'literature').exists())
            self.assertEqual(git(original,'rev-parse','HEAD'),head)
            calls=[];real_git=updater.git
            def trace(where,*args,**kwargs):
                calls.append(args);return real_git(where,*args,**kwargs)
            with patch.object(updater,'CODE',original/updater.SUBDIR),patch.object(network,'routes',return_value=(upstream.as_uri(),)),patch.object(updater,'git',side_effect=trace):
                updater.update(runtime)
            self.assertFalse(any(args[0]=='fetch' for args in calls), 'An unchanged version should not fetch objects again')
            cache.rename(root/'evicted-cache')
            (upstream/updater.SUBDIR/'05_src/task_panel.py').write_bytes(b'# code v3\n')
            git(upstream,'add','.');git(upstream,'commit','-qm','next code update')
            with patch.object(updater,'CODE',original/updater.SUBDIR),patch.object(network,'routes',return_value=(upstream.as_uri(),)):
                updater.update(runtime)
            available=set(git(cache,'cat-file','--batch-all-objects','--batch-check=%(objectname)').splitlines())
            self.assertTrue(set(excluded).isdisjoint(available))
            self.assertEqual(updater.selected_entry(runtime).read_bytes(),b'# code v3\n')

    def test_missing_promisor_inventory_never_probes_missing_object_types(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);git,upstream,original,head,excluded=self.fixture(root)
            runtime=root/'runtime';inventories=[];real_git=updater.git
            def guarded_git(where,*args,**kwargs):
                if args and args[0]=='cat-file' and any(arg.startswith('--batch-check') for arg in args):
                    if '--batch-all-objects' not in args:
                        raise ValueError('warning: lazy fetching disabled; fatal: could not fetch missing object from promisor remote')
                    self.assertIsNone(kwargs.get('input_bytes'))
                    inventories.append(args)
                return real_git(where,*args,**kwargs)
            with patch.object(updater,'CODE',original/updater.SUBDIR),patch.object(network,'routes',return_value=(upstream.as_uri(),)),patch.object(updater,'git',side_effect=guarded_git):
                updater.update(runtime)
            self.assertGreaterEqual(len(inventories),2)
            self.assertEqual(updater.selected_entry(runtime).read_bytes(),b'# code v2 $Format:%H$\n')
            available=set(git(runtime/'update_cache.git','cat-file','--batch-all-objects','--batch-check=%(objectname)').splitlines())
            self.assertTrue(set(excluded).isdisjoint(available))

    def test_repeated_transfer_counters_do_not_extend_total_deadline(self):
        script="import sys,time\nfor _ in range(12):\n sys.stderr.write('Receiving objects: 13% (28/210), 53.18 MiB | 3.95 MiB/s\\r');sys.stderr.flush();time.sleep(.04)\n"
        real_popen=subprocess.Popen;children=[]
        def spawn(command,**kwargs):
            child=real_popen([sys.executable,'-u','-c',script],**kwargs);children.append(child);return child
        with tempfile.TemporaryDirectory() as temp,patch.object(updater.subprocess,'Popen',side_effect=spawn), \
             patch.object(updater,'HEARTBEAT_SECONDS',.03,create=True),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.TimeoutExpired):
                updater.git(Path(temp),'fetch',timeout=.15,progress_log=Path(temp)/'fetch.log',stage='对象下载')
            self.assertTrue(all(child.poll() is not None for child in children))

    def test_main_advancing_during_download_is_rechecked_before_publishing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);git,upstream,original,head,excluded=self.fixture(root)
            queried=git(upstream,'rev-parse','HEAD');real_git=updater.git;queries=[]
            def moving_main(where,*args,**kwargs):
                result=real_git(where,*args,**kwargs)
                if args and args[0]=='ls-remote' and '--get-url' not in args:
                    queries.append(result.stdout.split()[0])
                    if len(queries)==1:
                        (upstream/updater.SUBDIR/'05_src/task_panel.py').write_bytes(b'# code v3 after version query\n')
                        git(upstream,'add','.');git(upstream,'commit','-qm','concurrent main advance')
                return result
            runtime=root/'runtime'
            with patch.object(updater,'CODE',original/updater.SUBDIR),patch.object(network,'routes',return_value=(upstream.as_uri(),)),patch.object(updater,'git',side_effect=moving_main):
                updater.update(runtime)
            self.assertEqual(updater.selected_entry(runtime).read_bytes(),b'# code v3 after version query\n')
            self.assertEqual(len(queries),2)
            self.assertEqual(queries[0],queried)
            self.assertNotEqual(queries[0],queries[1])

    def test_divergence_or_continually_moving_main_keeps_old_release(self):
        for mutation in ['divergence','moves_again']:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);git,upstream,original,head,excluded=self.fixture(root)
                real_git=updater.git;queries=[]
                def moving_main(where,*args,**kwargs):
                    is_query=args and args[0]=='ls-remote' and '--get-url' not in args
                    if is_query and len(queries)==1 and mutation=='moves_again':
                        (upstream/updater.SUBDIR/'05_src/task_panel.py').write_bytes(b'# code v4\n')
                        git(upstream,'add','.');git(upstream,'commit','-qm','main moves again')
                    result=real_git(where,*args,**kwargs)
                    if is_query:
                        queries.append(result.stdout.split()[0])
                        if len(queries)==1:
                            if mutation=='divergence':git(upstream,'checkout','-B','main',head)
                            (upstream/updater.SUBDIR/'05_src/task_panel.py').write_bytes(b'# code v3\n')
                            git(upstream,'add','.');git(upstream,'commit','-qm','changed main')
                    return result
                runtime=root/'runtime'
                with patch.object(updater,'CODE',original/updater.SUBDIR),patch.object(network,'routes',return_value=(upstream.as_uri(),)),patch.object(updater,'git',side_effect=moving_main):
                    with self.assertRaises(ValueError):updater.update(runtime)
                self.assertFalse((runtime/'release.json').exists())
                self.assertEqual(git(original,'rev-parse','HEAD'),head)
                self.assertEqual(len(queries),2)

    def test_broken_candidate_code_keeps_the_previous_pointer(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);git,upstream,original,head,excluded=self.fixture(root)
            runtime=root/'runtime'
            with patch.object(updater,'CODE',original/updater.SUBDIR),patch.object(network,'routes',return_value=(upstream.as_uri(),)):
                updater.update(runtime)
                before=(runtime/'release.json').read_bytes()
                (upstream/updater.SUBDIR/'05_src/task_panel.py').write_bytes(b'def invalid(:\n')
                git(upstream,'add','.');git(upstream,'commit','-qm','invalid candidate')
                with self.assertRaisesRegex(ValueError,'语法'):
                    updater.update(runtime)
                self.assertEqual((runtime/'release.json').read_bytes(),before)


if __name__=='__main__':unittest.main()
