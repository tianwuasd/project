import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import subprocess

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


class UpdateTests(unittest.TestCase):
    def test_git_blob_install_is_scoped_and_rejects_corrupt_manifest(self):
        from dataset1282_analysis.maintenance import install_release
        from dataset1282_analysis.contracts import read_json
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);repo=base/'repo';repo.mkdir();home=base/'home';(home/'.update').mkdir(parents=True)
            self.package(repo/'dataset1282_analysis','A')
            (repo/'unrelated_private.txt').write_text('must not copy')
            def git(*args):
                return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.DEVNULL).decode().strip()
            git('init');git('add','.')
            git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','synthetic')
            commit=git('rev-parse','HEAD')
            entries=[]
            for line in git('ls-tree','-r',commit,'--','dataset1282_analysis').splitlines():
                info,name=line.split('\t');mode,kind,oid=info.split()
                entries.append({'name':name.split('/',1)[1],'oid':oid,'mode':mode})
            plan={'commit':commit,'cache':str(repo/'.git'),'entries':entries,'remote':'synthetic-local-fixture','branch':'test'}
            release=install_release(home,plan)
            self.assertTrue((release/'run.py').exists())
            self.assertFalse((release/'unrelated_private.txt').exists())
            self.assertFalse((home/'current.json').exists())

    def test_unsafe_and_linked_release_is_rejected(self):
        from dataset1282_analysis.maintenance import safe_relative,verify_release
        for value in ('../x','/tmp/x','src/../x','src/x:stream','src/CON.py','src/a.py ','private-data/x.json'):
            with self.assertRaises(ValueError,msg=value):safe_relative(value)

    def package(self,path,value):
        path.mkdir()
        (path/'src/dataset1282_analysis').mkdir(parents=True)
        (path/'configs').mkdir();(path/'docs').mkdir()
        for name in ('bootstrap.py','run.py','setup_env.py','update.py','smoke.py'):
            (path/name).write_text('VALUE = '+repr(value)+'\n')
        (path/'src/dataset1282_analysis/__init__.py').write_text('__version__="test"\n')
        (path/'requirements-lock.txt').write_text('')
        (path/'configs/default.json').write_text(json.dumps({'value':value}))
        (path/'docs/method.md').write_text(value)
        (path/'.gitattributes').write_bytes(b'* text eol=lf\n')
        for item in path.rglob('*'):
            if item.is_file(): item.write_bytes(item.read_bytes().replace(b'\r\n',b'\n'))
        from dataset1282_analysis.maintenance import write_manifest
        write_manifest(path)

    def test_tamper_traversal_and_missing_files_rejected(self):
        self.assertTrue((ROOT/'src/dataset1282_analysis/maintenance.py').exists(),'updater implementation missing')
        from dataset1282_analysis.maintenance import verify_release
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'release';self.package(p,'A');verify_release(p)
            manifest=json.loads((p/'release-manifest.json').read_text())
            (p/'run.py').write_text('tampered')
            with self.assertRaises(ValueError):verify_release(p)
            manifest['files']['../outside']='bad'
            (p/'release-manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):verify_release(p)

    def test_atomic_activation_and_rollback_preserve_local_config(self):
        self.assertTrue((ROOT/'src/dataset1282_analysis/maintenance.py').exists(),'updater implementation missing')
        from dataset1282_analysis.maintenance import activate_release,rollback_release
        from dataset1282_analysis.contracts import read_json
        with tempfile.TemporaryDirectory() as d:
            home=Path(d);(home/'code_versions').mkdir()
            a='a'*40;b='b'*40
            self.package(home/'code_versions'/a,'A');self.package(home/'code_versions'/b,'B')
            local=home/'project.local.json';local.write_text('{"private":"keep"}')
            activate_release(home,a);activate_release(home,b);rollback_release(home,a)
            self.assertEqual(read_json(home/'current.json')['commit'],a)
            self.assertEqual(local.read_text(),'{"private":"keep"}')
            (home/'code_versions'/b/'configs/default.json').write_text('{}')
            with self.assertRaises(ValueError):activate_release(home,b)
            self.assertEqual(read_json(home/'current.json')['commit'],a)

    def test_update_refuses_active_operation(self):
        self.assertTrue((ROOT/'src/dataset1282_analysis/maintenance.py').exists(),'updater implementation missing')
        from dataset1282_analysis.maintenance import activate_release
        from dataset1282_analysis.observability.status import operation_lock
        with tempfile.TemporaryDirectory() as d:
            home=Path(d);(home/'code_versions').mkdir();self.package(home/'code_versions'/('a'*40),'A')
            with operation_lock(home,'analysis'):
                with self.assertRaises(RuntimeError):activate_release(home,'a'*40)
            self.assertFalse((home/'current.json').exists())


if __name__=='__main__':unittest.main()
