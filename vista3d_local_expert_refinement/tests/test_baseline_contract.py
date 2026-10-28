import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_src'))
from test_cohort import cases
from model_catalog import parse_models, parse_folds
from server_data import bind_workspace, read_json, write_json
from cohort import PROJECT_ID

class BaselineContract(unittest.TestCase):
    def test_only_baseline_and_ct(self):
        self.assertEqual(parse_models('all'), ['B0'])
        with self.assertRaises(ValueError): parse_models('B1')
        with self.assertRaises(ValueError): parse_folds('mr_holdE')

    def test_runtime_inventory_and_changed_source_labels_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); code=root/'code'; work=root/'work'; data=root/'data'
            code.mkdir(); data.mkdir(); values=cases()
            for c in values:
                f=data/(c['case_id']+'.label');f.write_bytes(b'original')
                c['label']=str(f)
            with patch('server_data.discover_cases', return_value=values):
                bind_workspace(code,work,data,5)
                split=read_json(work/'04_data/manifests/splits_v1.json')['folds']['ct_holdG']
                from server_data import freeze_source_labels
                first=freeze_source_labels(work,'ct_holdG')
                self.assertEqual(len(first),38)
                self.assertFalse(set(first)&set(split['selector']+split['unlabeled']+split['target']))
                # Hidden labels can change without being read or affecting source fitting.
                (data/(split['target'][0]+'.label')).write_bytes(b'not inspected')
                self.assertEqual(freeze_source_labels(work,'ct_holdG'),first)
                (data/(split['train'][0]+'.label')).write_bytes(b'changed')
                with self.assertRaises(ValueError): freeze_source_labels(work,'ct_holdG')
                config=read_json(work/'00_admin/server_config.json')
                config['project_id']='old_full_label_project'
                write_json(work/'00_admin/server_config.json',config)
                with self.assertRaises(ValueError):bind_workspace(code,work,data,5)

if __name__=='__main__':unittest.main()
