from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
from time import perf_counter
import unittest

import numpy as np
import torch
import yaml
from filelock import FileLock,Timeout

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))

from smoke import SmokeWorkspace
from chd_segmentation.artifacts.store import write_json,file_digest,digest
from chd_segmentation.config import DEFAULTS,load_config
from chd_segmentation.preprocessing.prepare import prepare
from chd_segmentation.models.workflow import options,train,predict,load_state,dataset_for
from chd_segmentation.models.q_route1.adapter import allowed,export_probabilities


def verify(spec_path,work_root,report_root,selected_model,dataset=None):
    experiments = yaml.safe_load(Path(spec_path).read_text(encoding='utf-8'))['experiments']
    if dataset is not None:
        experiments = [experiment for experiment in experiments if experiment['dataset']==dataset]
    if not experiments:
        raise ValueError('没有选中的真实病例')
    if len(experiments)>2:
        raise ValueError('模型验证最多使用两个真实病例')
    with SmokeWorkspace(work_root,report_root) as workspace:
        for experiment in experiments:
            started = perf_counter()
            name = experiment['dataset']
            directory = workspace.root/name
            directory.mkdir()
            row = deepcopy(experiment['case']);row['roles']=['train','validation']
            roi = experiment['roi_xyz']
            if selected_model=='q_route1':
                roi = [[(a+b)//2-48,(a+b)//2+48] for a,b in roi]
            manifest = dict(dataset=name,scope='technical',data_root=experiment['data_root'],cases=[row],plan_scope='technical')
            write_json(directory/'manifest.json',manifest)
            base = deepcopy(DEFAULTS)
            base.update(dataset=name,scope='technical',manifest=str(directory/'manifest.json'),
                        pretrained=experiment['pretrained'],work_root=str(directory/'course'),run_dir=str(directory/'course/stu'))
            base['training'].update(patch=[32]*3,attempts=2,batch=1)
            base['technical']=dict(roi_xyz=roi,allow_role_overlap=True)
            write_json(directory/'base.json',base)
            prepared = prepare(load_config(directory/'base.json'),directory/'course/prepared.json')
            record = dict(dataset=name,case_id=row['case_id'],roi_xyz=roi,source_identity=row['sha256'],
                          models={},prepare_seconds=perf_counter()-started)
            workspace.results.append(record)
            model_path = directory/'model.yaml'
            config = dict(model=selected_model,run_dir=str(directory/'course'/selected_model),device='cpu',precision='fp32',
                threads=2,updates=2,accumulation=1,patch=32,local_size=32,technical=True,max_candidates=1,
                pretrained=experiment['pretrained'] if selected_model=='q_route1' else str(ROOT/'data/model_weights/MedSAM2_latest.pt'),
                official_root=str(ROOT/'vendor/MedSAM2'),dependency_root=str(ROOT/'env/medsam2'),
                frames=8,resolution=512,freeze_image_encoder=False,batch=1,workers=0)
            write_json(model_path,config)
            stages = ('base','localizer','local') if selected_model=='q_route1' else ('sequence',)
            record['models'][selected_model]={}
            for stage in stages:
                if stage=='local':
                    config['parents']={key:str(Path(config['run_dir'])/key/'latest.pt') for key in ('base','localizer')}
                    write_json(model_path,config)
                applied = options(model_path)
                if selected_model=='medsam2':
                    Path(config['run_dir']).mkdir(parents=True,exist_ok=True)
                    with FileLock(Path(config['run_dir'])/(stage+'.lock'),timeout=0):
                        with unittest.TestCase().assertRaises(Timeout):
                            train(applied,directory/'course/prepared.json',stage,stop_after=1)
                    record['models'][selected_model]['concurrent_training_rejected']=True
                first = train(applied,directory/'course/prepared.json',stage,stop_after=1)
                prior = load_state(first['checkpoint'])
                parameters = {key:value.clone() for key,value in prior['model'].items()}
                del prior
                continued = train(applied,directory/'course/prepared.json',stage,resume=True)
                if selected_model=='medsam2':
                    native = torch.load(continued['official_checkpoint'],map_location='cpu',weights_only=True)
                    if continued['restored_exact'] is not True or native['epoch']!=2 or native['steps']['train']!=2 or len(native['_chd']['history'])!=2:
                        raise ValueError('官方检查点及随机状态恢复未通过')
                    for key in ('python_rng','numpy_rng','torch_rng','prompt_rng','cuda_rng','identity'):
                        if key not in native['_chd']:
                            raise ValueError('完整官方检查点缺少恢复状态：'+key)
                    record['models'][selected_model]['atomic_resume_bundle']=True
                    del native
                current = load_state(continued['checkpoint'])
                changed = any(not torch.equal(parameters[key],value) for key,value in current['model'].items())
                if first['update']!=1 or continued['update']!=2 or not changed:
                    raise ValueError('实际参数更新与保存恢复未通过')
                record['models'][selected_model][stage]=dict(first=first,resume=continued,parameters_changed=changed)
                del parameters,current
                gc.collect()
            prompts=None
            if selected_model=='medsam2':
                from chd_segmentation.models.medsam2.data import NativeCases
                data = NativeCases(prepared,with_label=True)
                entry = data.get(row['case_id'])
                label = entry['label'];values=sorted(set(map(int,np.unique(label)))-{0,-1}-set(entry['unknown_classes']))
                target=values[0]
                frame=int(np.flatnonzero((label==target).any((1,2)))[0])
                yy,xx=np.where(label[frame]==target)
                prompts={row['case_id']:[dict(source='technical_simulated',space='native_dhw',frame=frame,class_id=target,
                          box_xyxy=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1])]}
                record['models'][selected_model]['prompt_protocol']='技术验收用训练标签模拟人工框；不属于正式自动预测'
                data.clear();del data,entry
            independent = deepcopy(prepared)
            independent['manifest'] = deepcopy(prepared['manifest'])
            for prediction_row in independent['manifest']['cases']:
                prediction_row['roles']=['predict']
                prediction_row['label']=None
                prediction_row['source_stamp'].pop('label',None)
                prediction_row['sha256'].pop('label',None)
            independent.pop('identity')
            independent['identity']=digest(independent)
            write_json(directory/'course/prediction-input.json',independent)
            record['models'][selected_model]['predict']=predict(options(model_path),directory/'course/prediction-input.json',
                Path(config['run_dir'])/stages[-1]/'latest.pt',directory/'result'/selected_model,prompts=prompts)
            record['models'][selected_model]['new_manifest_without_labels']=True
            for kind in ('image','label'):
                if file_digest(row[kind])!=row['sha256'][kind]:
                    raise ValueError('原始病例摘要发生变化')
            record.update(seconds=perf_counter()-started,source_unchanged=True)
            write_json(workspace.report/(name+'-'+selected_model+'.json'),record)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec',required=True)
    parser.add_argument('--work-root',required=True)
    parser.add_argument('--report-root',required=True)
    parser.add_argument('--model',choices=('q_route1','medsam2'),required=True)
    parser.add_argument('--dataset',choices=('chd1282','imagechd'))
    args=parser.parse_args()
    verify(args.spec,args.work_root,args.report_root,args.model,args.dataset)
