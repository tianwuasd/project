import argparse
from copy import deepcopy
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))

from smoke import SmokeWorkspace
from chd_segmentation.artifacts.store import write_json,read_json,digest,file_digest
from chd_segmentation.config import DEFAULTS,load_config
from chd_segmentation.models.workflow import options,predict
from chd_segmentation.preprocessing.prepare import prepare
from chd_segmentation.prediction.metrics import evaluate


def run(args):
    sample=next(item for item in yaml.safe_load(Path(args.spec).read_text(encoding='utf-8'))['experiments']
                if item['dataset']=='imagechd')
    with SmokeWorkspace(args.work_root,args.report) as workspace:
        started=perf_counter()
        row=deepcopy(sample['case'])
        row['roles']=['train','validation']
        write_json(workspace.root/'manifest.json',dict(dataset='imagechd',scope='technical',
            data_root=sample['data_root'],cases=[row],plan_scope='technical'))
        config=deepcopy(DEFAULTS)
        config.update(dataset='imagechd',scope='technical',manifest=str(workspace.root/'manifest.json'),
            pretrained=sample['pretrained'],work_root=str(workspace.root/'course'),
            run_dir=str(workspace.root/'course/prepare'),result_root=str(workspace.root/'result'))
        config['training'].update(patch=[32]*3,batch=1,attempts=1)
        config['technical']=dict(roi_xyz=sample['roi_xyz'],allow_role_overlap=True)
        write_json(workspace.root/'prepare.json',config)
        config=load_config(workspace.root/'prepare.json')
        prepared=prepare(config,workspace.root/'course/prepared.json')
        model=dict(model='medsam2',technical=True,device='cpu',run_dir=str(workspace.root/'course/medsam2'),
            work_root=str(workspace.root/'course'),batch=1,workers=0,updates=1,
            official_root=str(ROOT/'vendor/MedSAM2'),dependency_root=str(ROOT/'env/medsam2'),
            pretrained=str(ROOT/'data/model_weights/MedSAM2_latest.pt'),
            execution=dict(cpu_processes=1,workers=0,prefetch=1,reserve_ram_bytes=2*2**30,cache_bytes=16*2**20))
        write_json(workspace.root/'model.json',model)
        applied=options(workspace.root/'model.json')
        from chd_segmentation.models.medsam2.adapter import activate
        activate(applied)
        from chd_segmentation.models.medsam2.data import NativeCases
        from chd_segmentation.models.medsam2.prompts import label_box_prompts
        data=NativeCases(prepared,with_label=True,reserve_ram_bytes=2*2**30)
        assert data.reader.reserve==2*2**30
        entry=data.get(row['case_id'])
        values=sorted(set(map(int,np.unique(entry['label'])))-{0,-1}-set(entry['unknown_classes']))
        target=max(values,key=lambda value:int((entry['label']==value).sum((1,2)).argmax()))
        prompts,_=label_box_prompts(entry,row['sha256']['label'],8,[target])
        for item in prompts[row['case_id']]:
            item['source']='technical_simulated'
        data.clear()
        del data,entry
        independent=deepcopy(prepared)
        independent['manifest']=deepcopy(prepared['manifest'])
        for item in independent['manifest']['cases']:
            item['roles']=['predict']
            item['label']=None
            item['sha256'].pop('label')
            item['source_stamp'].pop('label')
        independent.pop('identity')
        independent['identity']=digest(independent)
        write_json(workspace.root/'without-label.json',independent)
        record=dict(case_id=row['case_id'],roi_xyz=sample['roi_xyz'],optimizer_updates=0,
            device='cpu',prompt_source='technical_simulated',source_identity=row['sha256'])
        workspace.results.append(record)
        prediction=predict(applied,workspace.root/'without-label.json',model['pretrained'],
            workspace.root/'result/prediction',prompts=prompts,official_weights=True)
        metrics=evaluate(workspace.root/'result/prediction',prepared,workspace.root/'result/metrics.json',config)
        assert '5' not in metrics['cases'][0]['classes']
        receipt=read_json(workspace.root/'result/prediction/predictions.json')
        assert receipt['cases'][row['case_id']]['reference_sha256'] is None
        resource_files=list((workspace.root/'course/model-predictions').rglob('resources.json'))
        assert len(resource_files)==1
        resource=read_json(resource_files[0])
        assert resource['resolved']['reserve_ram_bytes']==2*2**30
        assert all(file_digest(row[kind])==row['sha256'][kind] for kind in ('image','label'))
        record.update(prediction=prediction,evaluation=metrics,resources=resource,
            source_unchanged=True,prediction_without_labels=True,seconds=perf_counter()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('spec','work-root','report'):
        parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
