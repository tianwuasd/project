from copy import deepcopy
import os
from pathlib import Path
import unittest
import shutil
from time import perf_counter


class ReviewRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from chd_segmentation.preprocessing.prepare import load_prepared
        cls.root = Path(os.environ['CHD_REVIEW_WORK']).resolve(strict=True)
        cls.prepared = load_prepared(cls.root/'course/shared-offline.json')
        cls.started = perf_counter()

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('CHD_REVIEW_REPORTS'):
            import psutil
            from chd_segmentation.artifacts.store import write_json
            memory = psutil.Process().memory_info()
            write_json(Path(os.environ['CHD_REVIEW_REPORTS'])/'regression-runtime.json',
                dict(cases=['ct_1001','ct_1012'],optimizer_updates=0,seconds=perf_counter()-cls.started,
                    rss_bytes=memory.rss,peak_rss_bytes=getattr(memory,'peak_wset',None),
                    vram_peak='not_measured',scope='真实数组和既有NIfTI的局部接口；无新模型预测或训练'))

    def test_selected_real_case_pointer_cannot_be_replaced_by_catalog(self):
        from chd_segmentation.data.dataset import CaseDataset
        source = deepcopy(self.prepared)
        source['storage']['cases']['ct_1001'] = source['storage']['cases']['ct_1012']
        with self.assertRaises(ValueError):
            CaseDataset(source,dict(cache_bytes=0),with_label=False).get('ct_1001')

    def test_selected_real_arrays_and_labels_are_read_from_their_exact_keys(self):
        from chd_segmentation.data.dataset import CaseDataset,sample_patch
        import numpy as np
        dataset = CaseDataset(self.prepared,dict(cache_bytes=0),with_label=True)
        for name in ('ct_1001','ct_1012'):
            entry = dataset.get(name)
            self.assertEqual(entry['case_id'],name)
            self.assertEqual(list(entry['image'].shape),self.prepared['storage']['cases'][name]['shape_czyx'])
            image,label = sample_patch(entry,[32]*3,np.random.default_rng(17),0.)
            self.assertTrue(np.isfinite(image).all())
            self.assertEqual(label.shape,(32,32,32))
        dataset.clear()

    def test_epoch_stop_keeps_the_complete_real_sample_count(self):
        from chd_segmentation.models.medsam2.bounded_loader import require_complete_epoch
        count = len(self.prepared['manifest']['cases'])
        with self.assertRaises(ValueError):
            require_complete_epoch(steps=0,limit=1,batches=count)
        require_complete_epoch(steps=0,limit=count,batches=count)

    def test_med_resource_names_include_the_real_validation_case(self):
        from chd_segmentation.models.execution import training_budget
        from chd_segmentation.models.workflow import options
        config = options(self.root/'course/med.json')
        config['technical'] = False
        metadata = deepcopy(self.prepared)
        metadata['scope'] = 'formal'
        budget = training_budget(config,metadata,'sequence')
        self.assertEqual([row['case_id'] for row in budget['manifest']['cases']],['ct_1001','ct_1012'])
        self.assertTrue(all(row['roles']==['train'] for row in budget['manifest']['cases']))

    def test_inference_cannot_change_q_solver_method(self):
        from chd_segmentation.models.contracts import frozen_inference_config
        from chd_segmentation.models.workflow import options
        config = options(self.root/'course/q-local.json')
        changed = dict(config,method='M02')
        with self.assertRaises(ValueError):
            frozen_inference_config(changed,config)
        self.assertEqual(frozen_inference_config(config,config)['method'],'M07')

    def test_resume_protects_selection_rules_and_allows_explicit_source_change(self):
        from chd_segmentation.models.contracts import require_resume_contract
        from chd_segmentation.models.workflow import options
        config = options(self.root/'course/q-base.json')
        changed = dict(config,validate_every=config['validate_every']+1)
        with self.assertRaises(ValueError):
            require_resume_contract(config,self.prepared,changed,self.prepared,'base')
        other = deepcopy(self.prepared)
        other['plan']['configurations']['3d_fullres']['spacing'] = [1.,1.,1.]
        self.assertTrue(require_resume_contract(config,self.prepared,config,other,'base'))

    def test_pending_real_prediction_commits_without_model_recomputation(self):
        from chd_segmentation.artifacts.store import read_json,write_json
        from chd_segmentation.prediction.commit import CaseCommit
        directory = self.root/'checks/journal-regression'
        directory.mkdir(parents=True,exist_ok=False)
        source = self.root/'result/stu/ct_1001.nii.gz'
        receipt = read_json(self.root/'result/stu/predictions.json')['cases']['ct_1001']
        commit = CaseCommit(directory,self.prepared['identity'])
        records = {}
        try:
            commit.begin('ct_1001',receipt,'ct_1001.nii.gz','ct_1001.pending.nii.gz')
            shutil.copyfile(source,directory/'ct_1001.pending.nii.gz')
            commit.ready(receipt)
            journal_bytes = commit.journal.stat().st_size
            os.replace(directory/'ct_1001.pending.nii.gz',directory/'ct_1001.nii.gz')
            commit.recover(records,lambda values:write_json(directory/'predictions.json',dict(cases=values)))
            self.assertEqual(records['ct_1001'],receipt)
            self.assertFalse(commit.journal.exists())
            self.assertEqual((directory/'ct_1001.nii.gz').stat().st_size,source.stat().st_size)
        finally:
            cleaned = []
            for path in directory.iterdir():
                if path.is_symlink() or path.name not in ('ct_1001.nii.gz','ct_1001.pending.nii.gz',
                        'pending-case.json','predictions.json'):
                    raise ValueError('回归检查目录包含未知文件')
                cleaned.append(dict(path=str(path.resolve()),bytes=path.stat().st_size))
                path.unlink()
            directory.rmdir()
        if os.environ.get('CHD_REVIEW_REPORTS'):
            write_json(Path(os.environ['CHD_REVIEW_REPORTS'])/'journal-regression.json',dict(state='completed',
                model_recomputations=0,source=str(source),cleanup=cleaned,journal_deleted_bytes=journal_bytes,
                deleted_files=len(cleaned)+1,released_bytes=sum(item['bytes'] for item in cleaned)+journal_bytes,
                directory_absent=not directory.exists()))

    def test_cumulative_q_index_keeps_both_real_completed_cases(self):
        from chd_segmentation.artifacts.store import read_json
        from chd_segmentation.models.prediction import evaluation_records
        from chd_segmentation.preprocessing.prepare import load_prepared
        completed = read_json(self.root/'result/q_local/completed-cases.json')['cases']
        prepared = load_prepared(self.root/'course/q-wide-offline.json')
        records = evaluation_records(completed,prepared,str(self.root/'result/q_local/selected-model.pt'))
        self.assertEqual(set(records),{'ct_1001','ct_1012'})
        self.assertEqual(records['ct_1001']['checkpoint'],records['ct_1012']['checkpoint'])

    def test_med_validation_rounds_cover_each_real_case_once(self):
        from chd_segmentation.models.medsam2.validation import validation_rounds,aggregate_validation
        from chd_segmentation.artifacts.store import read_json
        names = [row['case_id'] for row in self.prepared['manifest']['cases']]
        for world in (1,2,3):
            rounds = validation_rounds(names,world)
            self.assertEqual([name for group in rounds for name in group if name is not None],names)
            self.assertTrue(all(len(group)==world for group in rounds))
        evaluation = read_json(self.root/'result/stu-evaluation.json')
        observations = [dict(case_id=row['case_id'],loss=next(item['dice'] for item in row['classes'].values()
            if item['dice'] is not None)) for row in evaluation['cases']]
        rows = {row['case_id']:row for row in self.prepared['manifest']['cases']}
        result = aggregate_validation(names,rows,[[observations[1]],[],[observations[0]]])
        self.assertEqual(result['aggregation_unit'],'case')
        self.assertAlmostEqual(result['value'],sum(row['loss'] for row in observations)/len(observations))
        with self.assertRaises(ValueError):
            aggregate_validation(names,rows,[observations,observations])

    def test_med_workspace_uses_the_actual_processed_grid(self):
        from chd_segmentation.models.memory import case_estimates
        from chd_segmentation.models.workflow import options
        import math
        config = options(self.root/'course/med.json')
        records = case_estimates(config,self.prepared,'prediction')
        for row in records:
            pointer = self.prepared['storage']['cases'][row['case_id']]
            voxels = math.prod(pointer['shape_czyx'][1:])
            self.assertEqual(row['processed_voxels'],voxels)
            self.assertEqual(row['conversion_workspace_bytes'],voxels*24)
            self.assertEqual(row['retained_case_bytes'],voxels*10)
            self.assertEqual(row['propagation_history_bytes'],pointer['shape_czyx'][1]*7*4*2**20)

    def test_missing_gpu_observation_cannot_become_complete_capacity_evidence(self):
        from chd_segmentation.acceleration.measurements import complete_peak,process_vram_observation
        from chd_segmentation.acceleration.registry import inventory
        observed = inventory()
        if observed:
            record = process_vram_observation(observed[0]['uuid'],os.getpid())
            self.assertIsNone(record['bytes'])
            self.assertIsNone(complete_peak([dict(peak_vram_bytes=record['bytes'])]))

    def test_explicit_backend_selection_does_not_initialize_cuda(self):
        import torch
        import torch.distributed as dist
        from chd_segmentation.acceleration.distributed import distributed_backend
        self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual(distributed_backend('cpu'),'gloo')
        with self.assertRaises(ValueError):
            distributed_backend('cpu','nccl')
        if torch.cuda.is_available():
            self.assertEqual(distributed_backend('cuda','gloo'),'gloo')
            if not dist.is_nccl_available():
                with self.assertRaises(RuntimeError):
                    distributed_backend('cuda')
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':
    unittest.main()
