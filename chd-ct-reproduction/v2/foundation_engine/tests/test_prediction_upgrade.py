"""Exact audited upgrades accept old models without modifying training identity."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class PredictionUpgrade(unittest.TestCase):
    def test_shipped_transition_binds_complete_current_release(self):
        from server_data import digest,read_json
        root=Path(__file__).resolve().parents[1]
        audit=read_json(root/'06_configs/prediction_code_compatibility.json')
        current={p.name:digest(p) for p in sorted((root/'05_src').glob('*.py'))}
        self.assertEqual(current,audit['target_code_sha256'])
        edge=audit['transitions'][0]
        self.assertEqual(fingerprint(audit['source_code_sha256']),edge['source'])
        self.assertEqual(fingerprint(current),edge['target'])
        import evaluation as ev
        self.assertTrue(ev.compatible_evaluation_code(audit['source_code_sha256'],current))

    def test_complete_maps_only_and_current_release_owns_audit(self):
        import evaluation as ev
        from server_data import write_json
        old={'a.py':'1'*64,'b.py':'2'*64};new=dict(old,**{'panel.py':'3'*64})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_json(root/'06_configs/prediction_code_compatibility.json',dict(schema_version=1,
                scope='training-preserving-prediction-upgrade',transitions=[dict(source=fingerprint(old),target=fingerprint(new))]))
            with patch.object(ev,'CODE_ROOT',root):
                self.assertTrue(ev.compatible_evaluation_code(old,new))
                self.assertFalse(ev.compatible_evaluation_code(dict(old,unknown='x'),new))
                self.assertFalse(ev.compatible_evaluation_code(old,dict(new,**{'a.py':'changed'})))
                self.assertFalse(ev.compatible_evaluation_code(new,old))

    def test_old_signature_preserved_and_unknown_fields_or_training_changes_rejected(self):
        import evaluation as ev
        from server_data import write_json,read_json
        from training_identity import training_signature
        from stunet_recipe import recipe
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);fold='imagechd_fold0';method='S1'
            split=dict(dataset_id=921,train=['a'],val=['b'],target=['c'])
            write_json(work/'04_data/manifests/splits_v1.json',dict(folds={fold:split}))
            write_json(work/'04_data/derived/nnUNet_preprocessed/Dataset921_imagechd_fold0/WholeheartBaselinePlans.json',{})
            for name in ['pip_freeze_training.txt','conda_explicit_training.txt']:
                p=work/'06_configs'/name;p.parent.mkdir(exist_ok=True);p.write_text('synthetic')
            write_json(work/'00_admin/model_selection.json',dict(folds=[fold],methods=[method]))
            signature=training_signature(work,fold,method,global_batch=2)
            signature['code_sha256']={'old.py':'1'*64}
            out=work/f'07_experiments/formal/{fold}_{method}_seed0'
            run=dict(status='completed',formal_training=True,planned_epochs=1000,completed_epochs=1000,
                signature=signature,split=split,initial_weight_sha256='synthetic')
            write_json(out/'run.json',run)
            for name in ['checkpoint_final.pth','checkpoint_best.pth']:
                p=out/'fold_0'/name;p.parent.mkdir(exist_ok=True);p.write_text(name)
            before=(out/'run.json').read_bytes()
            with patch.object(ev,'ROOT',work),patch('server_data.freeze_source_labels'),\
                 patch.object(ev,'compatible_evaluation_code',return_value=True):
                models,_=ev.freeze_models()
                self.assertEqual(signature,models[0]['signature'])
                self.assertEqual(before,(out/'run.json').read_bytes())
                for field,value in [('unexpected','x'),('plan_sha256','wrong'),('training_recipe',dict(recipe('S1'),initial_lr=9))]:
                    tampered=copy.deepcopy(run);tampered['signature'][field]=value
                    write_json(out/'run.json',tampered)
                    with self.assertRaises(ValueError):ev.freeze_models()


if __name__=='__main__':unittest.main()
