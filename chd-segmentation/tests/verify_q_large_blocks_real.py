import argparse
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
from time import perf_counter
from unittest import TestCase

import nibabel as nib
import numpy as np
import psutil
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))

from chd_segmentation.artifacts.store import read_json
from chd_segmentation.models.q_route1.adapter import allowed,SOLVER
from chd_segmentation.models.q_route1.localizer import Localizer
from chd_segmentation.models.q_route1.network import LocalNetwork
from chd_segmentation.models.q_route1.targets import build_heatmap_targets,localizer_loss
from chd_segmentation.models.q_route1.training_inputs import prepare_inputs,input_plan,pooled_image
from chd_segmentation.models.q_route1.validation_targets import localizer_statistics
from chd_segmentation.models.q_route1.validation_store import ValidationWorkspace
from chd_segmentation.models.q_route1.validation_runtime import case_demand
from chd_segmentation.models.q_route1.geometry import crop_local_input,sample_context
from chd_segmentation.models.q_route1.comparisons import infer_local
from chd_segmentation.models.q_route1.losses import segmentation_loss,event_loss
from chd_segmentation.models.stunet.adapter import StunetAdapter


def entry_for(row, data, index):
    region = (slice(224,288-index),slice(224,288),slice(64,128-3*index))
    began = perf_counter()
    image_file,label_file = nib.load(data/row['image']),nib.load(data/row['label'])
    image = np.asarray(image_file.dataobj[region],dtype=np.float32).transpose(2,1,0).copy()[None]
    label = np.asarray(label_file.dataobj[region],dtype=np.int16).transpose(2,1,0).copy()
    return dict(case_id=row['case_id'],image=image,label=label,unknown_classes=row['unknown_classes'],
                processed_spacing=list(reversed(row['spacing_xyz']))),perf_counter()-began


def run(args, root, result):
    torch.set_num_threads(2)
    torch.manual_seed(20261122)
    manifest = read_json(args.manifest)
    entries = []
    model = Localizer()
    result['cases'] = []
    for index,name in enumerate(('ct_1003','ct_1012')):
        row = next(row for row in manifest['cases'] if row['case_id']==name)
        entry,seconds = entry_for(row,Path(args.data_root),index)
        entries.append(entry)
        shape = entry['label'].shape
        padding = [v for n in reversed(shape) for v in (0,(-n)%32)]
        full_image = torch.from_numpy(entry['image'])[None]
        full_valid = torch.from_numpy(entry['label']>=0)[None,None]
        mark = perf_counter()
        heat,mask = build_heatmap_targets(allowed(entry['label'],entry['unknown_classes'],'imagechd'),
                                        entry['label']>=0,entry['processed_spacing'])
        expected = (F.avg_pool3d(F.pad(full_image,padding,mode='replicate'),4,4),
                    F.avg_pool3d(F.pad(full_valid.float(),padding),4,4),heat,mask)
        dense_seconds = perf_counter()-mark
        plan = input_plan(shape,entry['processed_spacing'],2*2**30)
        small_budget = plan['minimum_workspace_bytes']
        small_plan = input_plan(shape,entry['processed_spacing'],small_budget)
        assert small_plan['blocks']>1 and plan['blocks']==1
        with TestCase().assertRaises(MemoryError):
            input_plan(shape,entry['processed_spacing'],small_budget-1)
        variants,errors = [],[]
        for budget in (small_budget,2*2**30):
            mark = perf_counter()
            actual = prepare_inputs(entry,'imagechd',budget,log_dir=root/'events')
            for a,b in zip(expected,actual):
                torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
            errors.append([0. if a.dtype==torch.bool else float((a-b).abs().max()) for a,b in zip(expected,actual)])
            variants.append(dict(budget=budget,plan=input_plan(shape,entry['processed_spacing'],budget),seconds=perf_counter()-mark))
        pooled = pooled_image(entry,2*2**30)
        ones = torch.ones_like(full_image,dtype=torch.bool)
        with torch.no_grad():
            original = model(full_image,ones)
            other = model.forward_pooled(*pooled)
        for key in ('logits','heatmap','features','valid'):
            torch.testing.assert_close(original[key],other[key],atol=1e-6,rtol=1e-6)
        model.zero_grad(set_to_none=True)
        loss = localizer_loss(model(full_image,full_valid)['logits'],heat,mask)
        loss.backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        computed = localizer_loss(model.forward_pooled(*actual[:2])['logits'],*actual[2:])
        computed.backward()
        gradient_error = max(float((a-p.grad).abs().max()) for a,p in zip(gradients,model.parameters()))
        for a,p in zip(gradients,model.parameters()):
            torch.testing.assert_close(a,p.grad,atol=2e-5,rtol=2e-4)
        validation_config = dict(run_dir=str(root/'validation'),work_root=str(root),device='cpu',
                                 logging=dict(heartbeat_seconds=15))
        with ValidationWorkspace(validation_config,dict(case_id=name),small_budget) as workspace:
            validation_loss = localizer_statistics(model.forward_pooled(*actual[:2])['logits'][0].detach().numpy(),
                                                  entry,'imagechd',workspace)
        torch.testing.assert_close(computed.detach(),torch.tensor(validation_loss),atol=2e-6,rtol=2e-6)
        minimum = case_demand('localizer',shape,[32]*3,{},entry['processed_spacing'])['minimum']
        assert minimum>=small_budget
        result['cases'].append(dict(case_id=name,shape=list(shape),read_seconds=seconds,dense_seconds=dense_seconds,
            variants=variants,errors=errors,gradient_max_error=gradient_error,loss=float(computed.detach()),
            validation_loss=validation_loss,local_pooled_forward_equal=True))
        if index==0:
            torch.optim.AdamW(model.parameters(),lr=1e-4).step()
            result['optimizer_updates']+=1
        del original,other,gradients,computed,loss,actual,expected
    # 大网格只使用已记录形状与现有正式计划元数据，不创建整例数组。
    spacing = read_json(args.plan)['configurations']['3d_fullres']['spacing']
    result['formal_metadata_spacing'] = spacing
    result['formal_server_plan_verified'] = False
    result['formal_metadata_plans'] = {name:input_plan(shape,spacing,2*2**30) for name,shape in
        (('ct_1120',(275,600,600)),('ct_1158',(363,707,713)))}
    assert all(p['blocks']<10000 for p in result['formal_metadata_plans'].values())
    entry = entries[0]
    del model
    gc.collect()
    source_plan = read_json(args.plan)
    spec = StunetAdapter.spec(dict(dataset='imagechd',training=dict(patch=[32]*3)),source_plan)
    spec = dict(spec) if isinstance(spec,dict) else deepcopy(spec.__dict__)
    spec['num_classes']=11
    base = StunetAdapter.build(spec).eval()
    StunetAdapter.load_pretrained(base,args.pretrained)
    image = torch.from_numpy(entry['image'])[:,:32,:32,:32][None]
    label = entry['label'][:32,:32,:32]
    with torch.no_grad():
        probability = base(image)[0].float().softmax(1)
        locator = Localizer().eval()
        features = locator(image,torch.ones_like(image,dtype=torch.bool))['features']
        inputs,editable = crop_local_input(image,probability,(0,0,0),32)
        context = sample_context(features,(15.5,15.5,15.5),samples=4)
    del base,locator
    local = LocalNetwork(method='M07')
    optimizer = torch.optim.AdamW(local.parameters(),lr=1e-4)
    mark = perf_counter()
    out = local(inputs,context,editable)
    core = (slice(None),slice(None),*out['core'])
    target = torch.from_numpy(allowed(label,entry['unknown_classes'],'imagechd'))[None]
    final,_ = infer_local(out,editable,'M07',SOLVER,checkpointing=True,steps=3)
    loss = segmentation_loss(final,target[core],editable[core])
    loss = loss+.5*segmentation_loss(out['z0'][core],target[core],editable[core])+event_loss(out['q_logits'],target[core],editable[core])
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    result['optimizer_updates']+=1
    result['local'] = dict(size=32,loss=float(loss.detach()),seconds=perf_counter()-mark,
                           inputs=list(inputs.shape),context=list(context.shape))
    assert result['optimizer_updates']==2


def main():
    parser = argparse.ArgumentParser()
    for name in ('manifest','data-root','plan','pretrained'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    root = Path('D:/projects/back/t17/chd-q-large-blocks-20261122').resolve()
    report = Path('D:/projects/back/sync-reports/chd-q-large-blocks-20261122/report.json')
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (root/'.gitignore').write_text('*\n',encoding='utf-8')
    report.parent.mkdir(parents=True,exist_ok=True)
    started = perf_counter()
    result = dict(state='running',optimizer_updates=0,peak_rss_bytes=0,gpu_compute=False,
                  scope='两例真实影像区域≤64³；两个CPU优化器更新；大形状仅元数据规划')
    stop = threading.Event()
    def sample():
        while not stop.wait(.1):
            result['peak_rss_bytes']=max(result['peak_rss_bytes'],psutil.Process().memory_info().rss)
    sampler = threading.Thread(target=sample)
    sampler.start()
    try:
        run(args,root,result)
        result['state']='passed'
    finally:
        stop.set()
        sampler.join()
        result['seconds']=perf_counter()-started
        files = [p for p in root.rglob('*') if p.is_file()]
        assert len(files)<=80 and all(root in p.resolve().parents and not p.is_symlink() for p in files)
        total = sum(p.stat().st_size for p in files)
        assert total<=2*2**30
        result['cleanup']=dict(files=len(files),bytes=total,root=str(root),state='pending')
        report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        shutil.rmtree(root)
        result['cleanup'].update(state='complete',root_absent=not root.exists())
        report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(state=result['state'],seconds=result['seconds'],report=str(report)),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
