import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from time import time,perf_counter

import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--reject',choices=('precision','batch','learning_rate','source'))
    parser.add_argument('--verify-boundaries',action='store_true')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(project/'src'))
    from chd_segmentation.models.medsam3.finetune import settings,check
    from chd_segmentation.models.medsam3.resume import scientific_contract,resume_identity
    from chd_segmentation.artifacts.store import read_json,write_json
    root = Path(args.root).resolve(strict=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',
        TEMP=str(root/'temporary'),TMP=str(root/'temporary'),MPLCONFIGDIR=str(root/'matplotlib'))
    config = settings(root/'training.yaml')
    checked = check(config)
    saved = torch.load(root/'work/train/latest.pt',map_location='cpu',weights_only=True)
    contract = scientific_contract(config,checked)
    if args.verify_boundaries:
        began = perf_counter()
        config = settings(root/'training-resume.yaml')
        checked = check(config)
        # 仅扩展同一真实病例的相邻切片，检查整病例共有记录的合同构造。
        from chd_segmentation.models.medsam3.training_data import entries
        from chd_segmentation.artifacts.store import digest
        raw = deepcopy(config)
        raw['training']['prepared'] = None
        for role in ('train','val'):
            item = checked['sources'][role][0]
            raw['slices'][item['row']['case_id']] = [item['slice'],item['slice']+1]
        expanded = dict(checked,sources=entries(raw,raw['training']))
        assert all(len(items) == 2 and items[0]['row'] is items[1]['row'] for items in expanded['sources'].values())
        normalized = scientific_contract(raw,expanded)
        assert all(len(items) == 2 for items in normalized['sources'].values())
        assert all('source_stamp' in items[0]['row'] for items in expanded['sources'].values())
        for key in ('precision','batch','learning_rate','source'):
            result = subprocess.run([sys.executable,'-B',__file__,'--root',str(root),'--reject',key],capture_output=True,text=True,encoding='utf-8')
            assert result.returncode != 0 and 'ValueError' in result.stderr and '不兼容科学合同未被拒绝' not in result.stderr
        assert saved['step'] == config['training']['updates'] == 2 and 'validation_pending' not in saved
        original = dict(identity=saved['identity'],lora=saved['lora'],optimizer=saved['optimizer'])
        from chd_segmentation.training.checkpoint import tensors_equal
        del saved
        with (root/'resume-validation.log').open('w',encoding='utf-8') as output:
            subprocess.run([sys.executable,'-B',str(project/'run.py'),'models','medsam3','train',
                '--config',str(root/'training-resume.yaml'),'--resume','--stop-after','1','--deadline',str(time()+180)],
                check=True,stdout=output,stderr=subprocess.STDOUT,timeout=180)
        final = torch.load(root/'work/train/latest.pt',map_location='cpu',weights_only=True)
        assert final['step'] == 2 and not final['validation_pending'] and final['identity'] == original['identity']
        assert tensors_equal(final['lora'],original['lora']) and tensors_equal(final['optimizer'],original['optimizer'])
        assert final['contract_identity'] == digest(final['contract'])
        write_json(root/'resume-boundaries.json',dict(state='passed',same_case_slices_per_role=2,
            raw_contract_preserved=True,new_contract_rejections=4,pending_validation_completed=True,
            additional_model_updates=0,unchanged_lora_optimizer=True,seconds=perf_counter()-began))
        print(json.dumps(dict(state='passed',report=str(root/'resume-boundaries.json'))))
        return
    if args.reject:
        changed = deepcopy(contract)
        if args.reject == 'precision':
            changed['science']['precision'] = 'fp32'
        elif args.reject == 'source':
            changed['sources']['train'][0]['classes'] = [6]
        else:
            changed['training'][args.reject] = 2 if args.reject == 'batch' else .001
        # 该分支只执行真实合同检查，不启动模型或更新参数。
        if 'contract' in saved:
            resume_identity(saved,config,checked,changed,root/'work/train/resume-proof.v1.json')
        else:
            config['precision'] = changed['science']['precision']
            config['training'].update(changed['training'])
            if args.reject == 'source':
                checked['sources']['train'][0]['classes'] = [6]
            resume_identity(saved,config,checked,changed,root/'work/train/resume-proof.v1.json')
        raise AssertionError('不兼容科学合同未被拒绝')
    began = perf_counter()
    assert saved['step'] == 1
    original_identity = saved['identity']
    first_lora = saved['lora']
    first_checkpoint = dict(identity=original_identity,step=1,optimizer_states=len(saved['optimizer']['state']))
    identity,migration = resume_identity(saved,config,checked,contract,root/'work/train/resume-proof.v1.json')
    assert identity == original_identity and migration['kind'] == 'legacy_contract_verified'
    del saved
    for key in ('precision','batch','learning_rate','source'):
        result = subprocess.run([sys.executable,'-B',__file__,'--root',str(root),'--reject',key],capture_output=True,text=True,encoding='utf-8')
        assert result.returncode != 0 and 'ValueError' in result.stderr and '不兼容科学合同未被拒绝' not in result.stderr
    base = yaml.safe_load((root/'base.yaml').read_text(encoding='utf-8'))
    base.update(fixed_ram_bytes=3*2**30)
    base['resources'].update(cpu_threads=2)
    (root/'base-resume.yaml').write_text(yaml.safe_dump(base,allow_unicode=True),encoding='utf-8')
    training = yaml.safe_load((root/'training.yaml').read_text(encoding='utf-8'))
    training.update(base_config='base-resume.yaml',workers=0,prefetch=1,ram_limit_bytes=12*2**30,
        vram_limit_bytes=13*2**29,max_seconds=600)
    path = root/'training-resume.yaml'
    path.write_text(yaml.safe_dump(training,allow_unicode=True),encoding='utf-8')
    changed_config = settings(path)
    changed_checked = check(changed_config)
    changed = scientific_contract(changed_config,changed_checked)
    assert contract == changed
    deadline = min(time()+600,(root/'report.json').stat().st_ctime+1200)
    with (root/'resume-new.log').open('w',encoding='utf-8') as output:
        subprocess.run([sys.executable,'-B',str(project/'run.py'),'models','medsam3','train',
            '--config',str(path),'--resume','--stop-after','1','--deadline',str(deadline)],
            check=True,stdout=output,stderr=subprocess.STDOUT,timeout=max(.1,deadline-time()))
    latest = torch.load(root/'work/train/latest.pt',map_location='cpu',weights_only=True)
    assert latest['step'] == 2 and latest['identity'] == original_identity
    assert latest['contract'] == contract and latest['complete_update']
    assert all(float(v['step']) == 2 for v in latest['optimizer']['state'].values())
    delta = sum((v-first_lora[k]).abs().sum().item() for k,v in latest['lora'].items())
    assert delta > 0
    write_json(root/'resume-acceptance.json',dict(state='passed',first_checkpoint=first_checkpoint,
        resumed_step=2,retained_origin_identity=latest['identity'],optimizer_states=len(latest['optimizer']['state']),
        lora_delta_resume=delta,migration=latest['migration'],resource_changes=dict(cpu_threads=[4,2],workers=[1,0],
            prefetch=[2,1],ram_limit_gib=[16,12],vram_limit_gib=[7,6.5]),
        rejected_scientific_changes=['precision','batch','learning_rate','source'],seconds=perf_counter()-began,
        training=read_json(root/'work/train/training.json')))
    print(json.dumps(dict(state='passed',updates_total=2,report=str(root/'resume-acceptance.json'))))


if __name__ == '__main__':
    main()
