from copy import deepcopy
import os
from pathlib import Path
from time import perf_counter
import unittest

import torch

from chd_segmentation.artifacts.store import read_json,write_json
from chd_segmentation.training.checkpoint import load_checkpoint,save_checkpoint


class ResumeInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ['CHD_REVIEW_WORK']).resolve(strict=True)
        cls.reports = Path(os.environ['CHD_REVIEW_REPORTS']).resolve(strict=True)
        cls.started = perf_counter()
        cls.directory = cls.root/'checks/stu-resume-inputs'
        cls.directory.mkdir(parents=True,exist_ok=False)
        cls.ledger = read_json(cls.reports.parent/'attempts.json')

    @classmethod
    def tearDownClass(cls):
        import psutil
        files = list(cls.directory.iterdir())
        removed = []
        for path in files:
            if path.is_symlink() or not path.is_file() or path.parent.resolve()!=cls.directory.resolve():
                raise ValueError('恢复局部检查目录归属不明确')
            removed.append(dict(path=str(path),bytes=path.stat().st_size))
            path.unlink()
        cls.directory.rmdir()
        if read_json(cls.reports.parent/'attempts.json')!=cls.ledger:
            raise ValueError('零更新恢复局部检查改变了训练台账')
        memory = psutil.Process().memory_info()
        write_json(cls.reports/'stu-resume-inputs.json',dict(cases=['ct_1001','ct_1012'],
            optimizer_updates=0,seconds=perf_counter()-cls.started,peak_rss_bytes=getattr(memory,'peak_wset',None),
            cuda_initialized=torch.cuda.is_initialized(),ledger_unchanged=True,cleanup=removed,
            deleted_files=len(removed),released_bytes=sum(row['bytes'] for row in removed),directory_absent=True))

    def test_new_source_first_complete_validation_can_select_a_higher_loss(self):
        run = self.root/'course/runs/STU'
        previous = read_json(run/'run.json')
        original = load_checkpoint(run,previous['identity'],read_json(run/'best.json'))
        latest = load_checkpoint(run,previous['identity'])
        original = deepcopy(original)
        original['progress']['selection_scope'] = previous['prepared']['storage']['recipe_identity']
        first = save_checkpoint(self.directory,original)
        old_best = read_json(self.directory/'best.json')
        changed = deepcopy(latest)
        changed['progress']['selection_scope'] = read_json(self.root/'course/q-wide-offline.json')['storage']['recipe_identity']
        changed['progress'].pop('selection',None)
        save_checkpoint(self.directory,changed)
        self.assertEqual(read_json(self.directory/'best.json'),old_best)
        self.assertTrue((self.directory/first['file']).is_file())
        value = latest['progress']['validation_loss']
        self.assertGreater(value,old_best['metric'])
        changed['progress']['selection'] = dict(latest['progress']['selection'],best=value,
            best_position=latest['progress']['epoch'])
        final = save_checkpoint(self.directory,changed)
        self.assertEqual(read_json(self.directory/'best.json')['file'],final['file'])
        self.assertEqual(read_json(self.directory/'best.json')['metric'],value)

    def test_initialization_metadata_does_not_block_real_inputs(self):
        from chd_segmentation.config import load_config
        from chd_segmentation.preprocessing.prepare import load_prepared
        from chd_segmentation.training.engine import validate_initialization_restart
        from chd_segmentation.training.resume import resume_inputs,validate_training_inputs,reset_selection,selection_scope
        previous = read_json(self.root/'course/runs/STU/run.json')
        prepared = load_prepared(self.root/'course/shared-offline.json')
        config = load_config(self.root/'course/stu.json')
        current = deepcopy(previous['scientific'])
        current.pop('code_identity',None)
        current['versions'] = {}
        write_json(self.directory/'run.json',previous)
        validate_initialization_restart(self.directory,current,config,prepared)
        changed_metadata = deepcopy(prepared)
        changed_metadata['identity'] = current['prepared_identity']+'-metadata'
        self.assertFalse(resume_inputs(previous,changed_metadata,config))
        other = load_prepared(self.root/'course/q-wide-offline.json')
        self.assertTrue(resume_inputs(previous,other,config))
        reset = reset_selection(dict(selection=dict(best=1.),val_step=1,val_sum=1.,validation_loss=1.),selection_scope(other,config))
        self.assertNotIn('selection',reset)
        self.assertEqual(reset['val_step'],0)
        self.assertIsNone(reset['validation_loss'])
        invalid = deepcopy(prepared)
        invalid['manifest']['cases'][0]['roles'] = ['train','validation']
        with self.assertRaises(ValueError):
            validate_training_inputs(invalid,config)
        invalid = deepcopy(prepared)
        invalid['dataset_json']['channel_names'] = {'0':'MR'}
        with self.assertRaises(ValueError):
            validate_training_inputs(invalid,config)

    def test_storage_startup_needs_pointers_and_consumer_opens_the_actual_case(self):
        from chd_segmentation.data.dataset import CaseDataset
        from chd_segmentation.preprocessing.offline_store import validate_storage
        prepared = read_json(self.root/'course/shared-offline.json')
        metadata = deepcopy(prepared)
        del metadata['storage']['cases']['ct_1001']['identity']
        self.assertEqual(validate_storage(metadata),Path(metadata['storage']['root']).resolve())
        self.assertEqual(CaseDataset(metadata,dict(cache_bytes=0),with_label=False).get('ct_1001')['case_id'],'ct_1001')
        metadata['storage']['cases']['ct_1001']['key'] = '0'*64
        self.assertEqual(validate_storage(metadata),Path(metadata['storage']['root']).resolve())
        with self.assertRaises(FileNotFoundError):
            CaseDataset(metadata,dict(cache_bytes=0),with_label=False).get('ct_1001')


if __name__=='__main__':
    unittest.main()
