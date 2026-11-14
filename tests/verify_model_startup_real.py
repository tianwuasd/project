import argparse
from copy import deepcopy
from pathlib import Path
import sys
from time import perf_counter
from unittest import TestCase

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))

from smoke import SmokeWorkspace
from chd_segmentation.artifacts.store import write_json,read_json,file_digest
from chd_segmentation.config import DEFAULTS,load_config
from chd_segmentation.models.workflow import options,train,load_state
from chd_segmentation.training.checkpoint import tensors_equal,require_resume_device
from chd_segmentation.preprocessing.prepare import prepare


def run(args):
    item = next(row for row in yaml.safe_load(Path(args.spec).read_text(encoding='utf-8'))['experiments']
                if row['dataset']=='imagechd')
    check = TestCase()
    with SmokeWorkspace(args.work_root,args.report) as workspace:
        started = perf_counter()
        row = deepcopy(item['case'])
        row['roles'] = ['train','validation']
        write_json(workspace.root/'manifest.json',dict(dataset='imagechd',scope='technical',
            data_root=item['data_root'],cases=[row],plan_scope='technical'))
        base = deepcopy(DEFAULTS)
        base.update(dataset='imagechd',scope='technical',manifest=str(workspace.root/'manifest.json'),
            pretrained=item['pretrained'],work_root=str(workspace.root/'course'),run_dir=str(workspace.root/'course/stu'))
        base['training'].update(patch=[32]*3,batch=1,attempts=2)
        base['technical'] = dict(roi_xyz=item['roi_xyz'],allow_role_overlap=True)
        write_json(workspace.root/'base.json',base)
        prepared_path = workspace.root/'course/prepared.json'
        prepare(load_config(workspace.root/'base.json'),prepared_path)
        config = dict(model='q_route1',run_dir=str(workspace.root/'course/q'),work_root=str(workspace.root/'course'),
            device='cpu',precision='fp32',technical=True,updates=2,accumulation=4 if args.world>1 else 1,patch=1,local_size=32,
            pretrained=item['pretrained'],execution=dict(cpu_processes=args.world,workers=0,prefetch=1,
                batch_layout=list(map(int,args.batch_layout.split(','))) if args.batch_layout else None))
        path = workspace.root/'model.json'
        write_json(path,config)
        record = dict(case_id=row['case_id'],roi_xyz=item['roi_xyz'],optimizer_update_limit=2,device='cpu')
        workspace.results.append(record)
        with check.assertRaisesRegex(torch.multiprocessing.ProcessRaisedException,'训练块'):
            train(options(path),prepared_path,'base',stop_after=1)
        record['actual_invalid_patch_initialization_failed'] = True
        stage = Path(config['run_dir'])/'base'
        check.assertFalse((stage/'latest.pt').exists())
        note = stage/'test-owned-note.txt'
        note.write_text('本次测试创建的保护检查文件',encoding='utf-8')
        config['patch'] = 32
        write_json(path,config)
        with check.assertRaises((ValueError,FileExistsError)):
            train(options(path),prepared_path,'base',stop_after=1)
        check.assertTrue(note.exists())
        note.unlink()
        record['unknown_file_preserved'] = True
        record['retry'] = train(options(path),prepared_path,'base',stop_after=1)
        check.assertEqual(record['retry']['update'],1)
        startup = read_json(stage/'startup.json')
        check.assertEqual([item['state'] for item in startup['attempts']],['failed','finished'])
        with check.assertRaises(FileExistsError):
            train(options(path),prepared_path,'base',stop_after=1)
        if args.forget_layout_on_resume:
            config['execution']['batch_layout'] = None
            write_json(path,config)
        record['resume'] = train(options(path),prepared_path,'base',resume=True)
        check.assertEqual(record['resume']['update'],2)
        check.assertTrue(record['resume']['restored_exact'])
        check.assertEqual(record['resume']['batch_layout'],record['retry']['batch_layout'])
        with check.assertRaises(ValueError):
            require_resume_device('cpu','cuda')
        saved = load_state(record['resume']['checkpoint'])
        parameter = next(value for value in saved['model'].values() if value.is_floating_point())
        check.assertFalse(tensors_equal(parameter,parameter.to(dtype=torch.float64)))
        record.update(startup=startup,completed_checkpoint_protected=True,optimizer_updates=2,
            resume_device_type_guard=True,dtype_difference_rejected=True,CUDA_used=False,
            source_unchanged=all(file_digest(row[kind])==row['sha256'][kind] for kind in ('image','label')),
            seconds=perf_counter()-started)
        check.assertTrue(record['source_unchanged'])


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec',required=True)
    parser.add_argument('--work-root',required=True)
    parser.add_argument('--report',required=True)
    parser.add_argument('--world',type=int,choices=(1,2),default=1)
    parser.add_argument('--batch-layout')
    parser.add_argument('--forget-layout-on-resume',action='store_true')
    run(parser.parse_args())
