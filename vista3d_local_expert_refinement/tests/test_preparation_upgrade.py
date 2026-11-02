"""Exact scheduling-only upgrade preserves cache identity checks and source bytes."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SRC=Path(__file__).resolve().parents[1]/'05_src'
sys.path.insert(0,str(SRC))
import preparation_identity as identity
import task_inputs
from server_data import read_json,write_json
import test_task_lifecycle


def fingerprint(mapping):
    return hashlib.sha256(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class PreparationUpgradeTests(unittest.TestCase):
    def before(self):
        current=identity.current()
        audit=read_json(SRC.parent/'06_configs/preparation_code_compatibility.json')
        edge=audit['transitions'][-1]
        self.assertEqual(edge['target'],fingerprint(current))
        self.assertEqual(set(edge['changed_files']),{'cpu_case_queue.py','server_prepare.py'})
        old=dict(current)
        for name,values in edge['changed_files'].items():
            self.assertEqual(current[name],values['after']);old[name]=values['before']
        self.assertEqual(edge['source'],fingerprint(old))
        return old

    def test_only_reviewed_complete_producer_map_is_accepted(self):
        before=self.before();self.assertTrue(identity.compatible(before))
        for name in identity.FILES:
            self.assertFalse(identity.compatible(dict(before,**{name:'unknown'})))
        self.assertFalse(identity.compatible(dict(before,unknown='field')))
        with tempfile.TemporaryDirectory() as temp:
            self.assertFalse(identity.compatible(before,identity.current(),Path(temp)/'05_src'))

    def test_other_preparation_signature_fields_remain_strict(self):
        nodes=ast.parse((SRC/'server_prepare.py').read_text(encoding='utf-8'))
        function=next(n for n in nodes.body if isinstance(n,ast.FunctionDef) and n.name=='compatible_signature')
        ns={};exec(compile(ast.Module(body=[function],type_ignores=[]),'server_prepare.py','exec'),ns)
        before=dict(implementation_sha256=self.before(),split={'train':['synthetic']},nnunet_version='fixture')
        current=dict(before,implementation_sha256=identity.current())
        self.assertTrue(ns['compatible_signature'](before,current))
        self.assertFalse(ns['compatible_signature'](before,dict(current,split={'train':['other']})))
        self.assertFalse(ns['compatible_signature'](before,dict(current,unknown='field')))

    def test_old_input_cache_is_accepted_and_copied_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as temp:
            config,source,work,prep=test_task_lifecycle.InputTests().fixture(Path(temp))
            write_json(prep/'server_preparation_signature.json',dict(implementation_sha256=self.before()))
            before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in source.rglob('*') if p.is_file()}
            info=task_inputs.inspect_input(config)
            with patch('task_runtime.load_plan',return_value=dict(workspace=str(source),folds=['ct_holdG'])):
                self.assertIsNone(task_inputs.source_compatibility(source))
            task_inputs.copy_input(dict(config,workspace=str(work),source=info))
            self.assertEqual(before,{p:(p.read_bytes(),p.stat().st_mtime_ns) for p in before})
            self.assertEqual(read_json(work/prep.relative_to(source)/'server_preparation_signature.json')['implementation_sha256'],self.before())


if __name__=='__main__':unittest.main()
