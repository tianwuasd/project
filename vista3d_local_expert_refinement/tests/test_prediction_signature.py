"""Real freeze function with synthetic files; no imaging/GPU dependencies."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SRC=Path(__file__).resolve().parents[1]/'05_src'
sys.path.insert(0,str(SRC))
from server_data import digest,read_json,write_json


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class PredictionSignature(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.work=self.root/'workspace';self.code=self.root/'code'
        source=ast.parse((SRC/'evaluation.py').read_text(encoding='utf-8'))
        names={'establish_protocol','require_completed','compatible_evaluation_code','freeze_models'}
        functions=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names]
        protocol=next(ast.literal_eval(n.value) for n in source.body if isinstance(n,ast.Assign)
                      and any(isinstance(t,ast.Name) and t.id=='PROTOCOL' for t in n.targets))
        self.ns=dict(Path=Path,ROOT=self.work,CODE_ROOT=self.code,PROTOCOL=protocol,FOLDS={'ct_holdG':None},
            digest=digest,read_json=read_json,write_json=write_json,
            load_selection=lambda *a,**k:dict(folds=['ct_holdG'],methods=['B0']))
        exec(compile(ast.Module(body=functions,type_ignores=[]),'evaluation.py','exec'),self.ns)
        self.split=dict(dataset_id=813,train=['synthetic_fit'],val=['synthetic_val'],target=['synthetic_target'])
        write_json(self.work/'04_data/manifests/splits_v1.json',dict(folds={'ct_holdG':self.split}))
        self.prep=self.work/'04_data/derived/nnUNet_preprocessed/Dataset813_ct_holdG/WholeheartBaselinePlans.json'
        write_json(self.prep,{'synthetic':True})
        self.env=self.work/'06_configs';self.env.mkdir(parents=True)
        for name in ('pip_freeze_training.txt','conda_explicit_training.txt'):(self.env/name).write_text('synthetic\n')
        source=self.code/'05_src';source.mkdir(parents=True)
        (source/'evaluation.py').write_text('# synthetic current evaluator\n')
        (source/'trainer.py').write_text('# unchanged training implementation\n')
        self.current={p.name:digest(p) for p in source.glob('*.py')}
        self.folder=self.work/'07_experiments/formal/ct_holdG_B0_seed0'
        for name in ('checkpoint_best.pth','checkpoint_final.pth'):
            p=self.folder/'fold_0'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'synthetic hash fixture')
        self.signature=dict(fold='ct_holdG',method='B0',seed=0,plan_sha256=digest(self.prep),
            splits_sha256=digest(self.work/'04_data/manifests/splits_v1.json'),
            environment_sha256=digest(self.env/'pip_freeze_training.txt'),
            conda_environment_sha256=digest(self.env/'conda_explicit_training.txt'),code_sha256=self.current)
        self.record=dict(status='completed',formal_training=True,signature=self.signature,split=self.split,initial_weight_sha256='synthetic-seed')

    def freeze(self):
        write_json(self.folder/'run.json',self.record)
        with patch('server_data.freeze_source_labels'):
            return self.ns['freeze_models']()

    def audit(self,edges):
        write_json(self.code/'06_configs/prediction_code_compatibility.json',dict(schema_version=1,
            scope='training-preserving-prediction-upgrade',transitions=[dict(source=fingerprint(a),target=fingerprint(b)) for a,b in edges]))

    def test_fixed_and_nondefault_batch_preserved_independent_of_prediction_default(self):
        for batch in (1,2,4,8):
            with self.subTest(batch=batch):
                self.record['signature']=dict(self.signature,selected_global_batch_size=batch)
                frozen=self.work/'08_results/evaluation/frozen_models.json';frozen.unlink(missing_ok=True)
                with patch.dict(os.environ,WHOLE_HEART_GLOBAL_BATCH='99'):
                    models,_=self.freeze()
                self.assertEqual(models[0]['signature'],self.record['signature'])

    def test_same_code_legacy_shape_still_accepted(self):
        self.assertEqual(self.freeze()[0][0]['signature'],self.signature)

    def test_shipped_audit_binds_actual_release_bytes_and_reviewed_two_file_change(self):
        current={p.name:digest(p) for p in SRC.glob('*.py')}
        audit=read_json(SRC.parent/'06_configs/prediction_code_compatibility.json')
        edge=audit['transitions'][-1]
        self.assertEqual(edge['source_version'],'0.7.0')
        self.assertEqual(set(edge['changed_files']),{'evaluation.py','task_panel.py'})
        self.assertEqual(edge['target'],fingerprint(current))
        previous=dict(current)
        for name,values in edge['changed_files'].items():
            self.assertEqual(current[name],values['after'])
            previous[name]=values['before']
        self.assertEqual(edge['source'],fingerprint(previous))

    def test_only_exact_reviewed_transition_accepts_previous_training(self):
        before=dict(self.current,**{'evaluation.py':'a'*64})
        self.audit([(before,self.current)])
        self.record['signature']=dict(self.signature,code_sha256=before,selected_global_batch_size=4)
        self.assertEqual(self.freeze()[0][0]['signature'],self.record['signature'])
        self.assertEqual(read_json(self.folder/'run.json'),self.record)
        bad=dict(before,**{'trainer.py':'f'*64})
        self.assertFalse(self.ns['compatible_evaluation_code'](bad,self.current))

    def test_reviewed_edges_are_cumulative_but_missing_or_corrupt_audit_rejects(self):
        a=dict(self.current,**{'evaluation.py':'a'*64});b=dict(self.current,**{'evaluation.py':'b'*64})
        self.assertFalse(self.ns['compatible_evaluation_code'](a,self.current))
        self.audit([(a,b),(b,self.current),(b,a)])
        self.assertTrue(self.ns['compatible_evaluation_code'](a,self.current))
        (self.code/'06_configs/prediction_code_compatibility.json').write_text('{broken')
        self.assertFalse(self.ns['compatible_evaluation_code'](a,self.current))

    def test_invalid_batch_and_unknown_signature_fields_rejected(self):
        for value in (0,-1,True,'4',None):
            with self.subTest(value=value):
                self.record['signature']=dict(self.signature,selected_global_batch_size=value)
                with self.assertRaises(ValueError):self.freeze()
        self.record['signature']=dict(self.signature,unexpected_future_contract=1)
        with self.assertRaises(ValueError):self.freeze()

    def test_plan_split_environment_and_completion_are_still_strict(self):
        for key in ('plan_sha256','splits_sha256','environment_sha256','conda_environment_sha256'):
            with self.subTest(key=key):
                self.record['signature']=dict(self.signature,**{key:'changed'})
                with self.assertRaises(ValueError):self.freeze()
        self.record['signature']=self.signature
        self.record['split']=dict(self.split,target=['wrong_target'])
        with self.assertRaises(ValueError):self.freeze()
        self.record['split']=self.split;self.record['status']='running'
        with self.assertRaises(ValueError):self.freeze()

    def test_run_and_checkpoint_bytes_are_unchanged_and_freeze_is_repeatable(self):
        self.record['signature']=dict(self.signature,selected_global_batch_size=4)
        write_json(self.folder/'run.json',self.record)
        before={p: p.read_bytes() for p in self.folder.rglob('*') if p.is_file()}
        first=self.freeze();second=self.freeze()
        self.assertEqual(first,second)
        for p,data in before.items():self.assertEqual(p.read_bytes(),data)
        path=self.folder/'fold_0/checkpoint_best.pth';path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'冻结后发生变化'):self.freeze()


if __name__=='__main__':unittest.main()
