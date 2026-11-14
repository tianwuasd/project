import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys
from time import perf_counter

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))

from smoke import SmokeWorkspace
from chd_segmentation.artifacts.store import write_json,file_digest
from chd_segmentation.config import DEFAULTS,load_config
from chd_segmentation.models.execution import execution_config
from chd_segmentation.models.workflow import options
from chd_segmentation.models.memory import case_estimates
from chd_segmentation.preprocessing.prepare import prepare
from chd_segmentation.training.resources import cpu_plan,host_capacity


def run(args):
    item = next(row for row in yaml.safe_load(Path(args.spec).read_text(encoding='utf-8'))['experiments']
                if row['dataset']=='imagechd')
    with SmokeWorkspace(args.work_root,args.report) as workspace:
        started = perf_counter()
        row = deepcopy(item['case'])
        row['roles'] = ['train','validation']
        write_json(workspace.root/'manifest.json',dict(dataset='imagechd',scope='technical',
            data_root=item['data_root'],cases=[row],plan_scope='technical'))
        config = deepcopy(DEFAULTS)
        config.update(dataset='imagechd',scope='technical',manifest=str(workspace.root/'manifest.json'),
            pretrained=item['pretrained'],work_root=str(workspace.root/'course'),run_dir=str(workspace.root/'course/stu'))
        config['training'].update(patch=[32]*3,batch=1,attempts=2)
        config['technical'] = dict(roi_xyz=item['roi_xyz'],allow_role_overlap=True)
        write_json(workspace.root/'base.json',config)
        prepared = prepare(load_config(workspace.root/'base.json'),workspace.root/'course/prepared.json')
        plans = []
        for model in ('q_route1','medsam2'):
            model_config = dict(model=model,run_dir=str(workspace.root/'course'/model),technical=True,
                device='cpu',precision='fp32',patch=32,local_size=32,updates=2,batch=1,workers=0,
                pretrained=item['pretrained'] if model=='q_route1' else str(ROOT/'data/model_weights/MedSAM2_latest.pt'),
                official_root=str(ROOT/'vendor/MedSAM2'),dependency_root=str(ROOT/'env/medsam2'),
                execution=dict(workers=0,prefetch=1,reserve_ram_bytes=2*2**30,cache_bytes=16*2**20))
            write_json(workspace.root/'model.json',model_config)
            common = execution_config(options(workspace.root/'model.json'))
            common['technical'] = prepared['technical']
            for stage in (('base','localizer','local','prediction') if model=='q_route1' else ('sequence','prediction')):
                common['model_stage'] = stage
                plan = cpu_plan(common,prepared,1,stage='prediction' if stage=='prediction' else 'train')
                if plan.get('model_memory',{}).get('model')!=model:
                    raise ValueError('共享分配器没有使用模型实际的内存预算')
                profile = plan['model_memory']
                actual = case_estimates(common,prepared,stage)
                if model=='q_route1' and stage=='base' and profile['rank_case_bytes'][0] < math.prod(actual[0]['working_shape'])*(11*24+16):
                    raise ValueError('Q基座C组整例验证预算低于实际概率工作区门槛')
                if plan['case_estimates']!=actual or profile['retained_case_bytes']<=0:
                    raise ValueError('病例构建和保留内存未登记')
                if model=='medsam2' and stage=='sequence' and plan['minimum_bytes']<19*2**30:
                    raise ValueError('官方完整CPU训练未计入已实测的模型内存')
                if plan['admitted'] and plan['ram_bytes']>plan['budget_bytes']:
                    raise ValueError('模型资源预算超过主机可用额度')
                plans.append(dict(model=model,stage=stage,plan=plan))
            if model=='medsam2':
                # 只规划CUDA训练的主机内存，不申请或使用CUDA。
                common['resources'].update(device='cuda',workers='auto')
                common['model_stage'] = 'sequence'
                common['model_topology'] = dict(world=1,shares=[1],workers=[1])
                frozen = cpu_plan(common,prepared,1)
                if not frozen['admitted'] or frozen['workers']!=[1]:
                    raise ValueError('恢复规划没有保留原始worker布局')
                common['resources']['workers'] = 0
                restricted = cpu_plan(common,prepared,1)
                if restricted['admitted']:
                    raise ValueError('恢复所需worker不足时应继续等待')
                plans.append(dict(model=model,stage='resume_host_planning_without_cuda',plan=frozen,
                    restricted=restricted))
        for kind in ('image','label'):
            if file_digest(row[kind])!=row['sha256'][kind]:
                raise ValueError('短测改变了原始病例')
        workspace.results.append(dict(case_id=row['case_id'],roi_xyz=item['roi_xyz'],plans=plans,
            host=host_capacity(),seconds=perf_counter()-started,optimizer_updates=0,source_unchanged=True))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec',required=True)
    parser.add_argument('--work-root',required=True)
    parser.add_argument('--report',required=True)
    run(parser.parse_args())
