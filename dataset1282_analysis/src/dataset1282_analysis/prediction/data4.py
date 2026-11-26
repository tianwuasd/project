import argparse
from collections import deque
from pathlib import Path
import os
import re
import shutil

from ..contracts import atomic_json, digest, read_json, sha256_file
from ..observability.status import operation_lock, write_status
from ..storage_paths import assert_output_location
from ..xu_targets import SEMANTICS
from . import PROTOCOL
from .preprocessed import FORMAT, PreprocessedInputs
from .reuse import SIX
from .source import implementation, verify_frozen


def label_semantics(labels):
    aliases={re.sub('[^a-z0-9]','',name.lower()):name for name in ('background',*SEMANTICS.values())}
    aliases.update({name.lower():meaning for name,meaning in SEMANTICS.items()})
    if (len(labels)!=11 or any(type(value) is not int for value in labels.values())
            or set(labels.values())!=set(range(11))):
        raise ValueError('标签必须保留完整且唯一的0..10整数编号')
    result={}
    for name,value in labels.items():
        token=re.sub('[^a-z0-9]','',name.lower())
        if token not in aliases or aliases[token] in result:
            raise ValueError('标签名称未知或存在重复语义：'+name)
        result[aliases[token]]=value
    if result.get('background')!=0:
        raise ValueError('背景标签编号必须为0')
    return result


def validate_model_input(input_plan,input_labels,model_plan,model_labels):
    if digest(input_plan)!=digest(model_plan):
        keys=sorted(k for k in input_plan.keys()|model_plan.keys() if input_plan.get(k)!=model_plan.get(k))
        raise ValueError('新数据的预处理计划与原训练不同，差异字段：'+', '.join(keys))
    actual,expected=label_semantics(input_labels),label_semantics(model_labels)
    if actual!=expected:
        keys=sorted(k for k in actual if actual[k]!=expected[k])
        raise ValueError('新数据的标签结构编号与原训练不同：'+', '.join(keys))


def check_data(prepared,cht_code,technical=False):
    description=read_json(prepared)
    names=list(description['storage']['cases'])
    rows=description['manifest']['cases']
    groups={g:sum(r.get('test_group')==g for r in rows) for g in
            ('source_test','target_primary','target_secondary')}
    if not technical and (len(names)!=1174 or groups!={'source_test':127,'target_primary':298,'target_secondary':31}):
        raise ValueError('需等待1174例转换全部完成并写出prepared.json')
    source=PreprocessedInputs(prepared,cht_code,names)
    raw=Path(description['manifest']['data_root']).resolve(strict=True)
    if os.name=='posix' and not raw.is_relative_to(Path('/data4/zhangruichen/data')):
        raise ValueError('原始CT和GT必须绑定data4/data目录')
    seen=set()
    patients=set()
    for row in rows:
        name=row['case_id']
        if name in seen or name not in names:
            raise ValueError('预处理病例清单重复或与storage不一致')
        seen.add(name)
        if not row.get('patient') or row['patient'] in patients:
            raise ValueError('病例患者编号缺失或发生重复')
        patients.add(row['patient'])
        for kind in ('image','label'):
            path=Path(row[kind]).resolve(strict=True)
            if not path.is_relative_to(raw) or not path.is_file():
                raise ValueError('原件引用未指向当前RAW目录')
        source.geometry(name)
        directory=source.source.root/description['storage']['cases'][name]['key'][:16]
        if not (directory/'label.npy').is_file() or not (directory/'locations.npz').is_file():
            raise ValueError('预处理病例缺少标签或采样坐标：'+name)
    if seen!=set(names):
        raise ValueError('预处理病例缺少原件记录')
    for name in names[:2]:
        entry=source.source.open(name,with_label=True)
        if entry['label'].shape!=entry['image'].shape[1:] or entry['label'].dtype.kind!='i':
            raise ValueError('CHT训练标签尺寸或ignore整数类型不符')
    print(f'CHT实际读取接口及原件路径检查通过：{len(names)}例，{groups}',flush=True)
    return source


def build_models(args):
    source=check_data(args.preprocessed,args.cht_code)
    old=Path(args.legacy_prepared).resolve(strict=True)/'preprocessed/Dataset282_CHDSource'
    plan,dataset=read_json(old/'nnUNetPlans.json'),read_json(old/'dataset.json')
    validate_model_input(source.plan,source.prepared['dataset_json']['labels'],plan,dataset['labels'])
    identities=[]
    selected={}
    for path in args.prediction:
        path=Path(path).resolve(strict=True)
        record=read_json(path/'run.json')
        identity=verify_frozen(path)
        if (record['source']!=digest(identity) or record['plans']!=sha256_file(old/'nnUNetPlans.json')
                or record['dataset']!=sha256_file(old/'dataset.json') or not set(SIX)<=identity['checkpoints'].keys()):
            raise ValueError('原预测来源与保留的训练计划不一致')
        if identities and (any(identity['checkpoints'][b]!=identities[0][1]['checkpoints'][b] for b in SIX)
                or identity['boxes']!=identities[0][1]['boxes']
                or any(identity['run'][k]!=identities[0][1]['run'][k] for k in ('profile','supervision','request'))):
            raise ValueError('两个原预测目录使用了不同模型或取框设置')
        for name,row in record['cases'].items():
            if name in selected:
                raise ValueError('原预测病例重复')
            selected[name]={k:row[k] for k in ('role','device','patient','shape','spacing','known_absent')}
        identities.append((path,identity))
    out=assert_output_location(args.output)
    if os.name=='posix' and not out.is_relative_to(Path('/data4/zhangruichen/result/chd1282')):
        raise ValueError('模型应放在/data4/zhangruichen/result/chd1282')
    out.mkdir(parents=True,exist_ok=True)
    path,identity=identities[0]
    mapping={k:dataset['labels'][v] for k,v in SEMANTICS.items()}
    model=dict(protocol='xu-sixway-models-data4-v1',profile=identity['run']['profile'],
        supervision=identity['run']['supervision'],request=identity['run']['request'],
        checkpoints={b:identity['checkpoints'][b] for b in SIX},plan=plan,dataset=dataset,
        anatomy=mapping,boxes=identity['boxes'],heldout=selected,padding_percent=20)
    with operation_lock(out,'xu-data4-models',out):
        marker=out/'models.json'
        if marker.exists() and read_json(marker)!=model:
            raise ValueError('目标已有不同模型登记')
        if not marker.exists() and any(x.name not in {'.runtime',*(b+'.pt' for b in SIX)} for x in out.iterdir()):
            raise ValueError('模型目标包含未知文件，拒绝混入')
        for branch in SIX:
            source_file=path/'source'/(branch+'.pt')
            if sha256_file(source_file)!=model['checkpoints'][branch]:
                raise ValueError('原冻结权重内容不一致：'+branch)
            target=out/(branch+'.pt')
            if target.exists():
                if sha256_file(target)!=model['checkpoints'][branch]:
                    raise ValueError('目标已有不同权重')
            else:
                os.link(source_file,target)
        if args.training is not None:
            checkpoint_root=out.parent/'checkpoints'
            checkpoint_root.mkdir(exist_ok=True)
            for branch in SIX:
                stage='auto-'+branch if 'auto-'+branch in identity['receipts'] else branch
                previous=Path(args.training).resolve(strict=True)/stage/'last.pt'
                if previous.is_file():
                    target=checkpoint_root/(branch+'-last.pt')
                    if target.exists() and not target.samefile(previous):
                        raise ValueError('已有不同的最新检查点')
                    if not target.exists():
                        os.link(previous,target)
        atomic_json(marker,model)
    print('六模型已登记：'+str(out),flush=True)


def open_job(args,out):
    model_root=Path(args.models).resolve(strict=True)
    if os.name=='posix' and not model_root.is_relative_to(Path('/data4/zhangruichen/result/chd1282')):
        raise ValueError('预测模型必须位于正式result/chd1282目录')
    model=read_json(model_root/'models.json')
    if model['protocol']!='xu-sixway-models-data4-v1' or model['padding_percent']!=20 or set(model['checkpoints'])!=set(SIX):
        raise ValueError('需要已登记的六网络模型及20%取框')
    description=read_json(args.preprocessed)
    rows={r['case_id']:r for r in description['manifest']['cases']}
    eligible={n:r for n,r in rows.items() if r.get('test_group') in args.groups}
    names=sorted(args.cases or eligible)
    if not names or len(set(names))!=len(names) or set(names)-eligible.keys():
        raise ValueError('请选择冻结留出组中的病例')
    source=PreprocessedInputs(args.preprocessed,args.cht_code,names)
    validate_model_input(source.plan,description['dataset_json']['labels'],model['plan'],model['dataset']['labels'])
    cases={}
    for name in names:
        row,original=rows[name],model['heldout'][name]
        geo=source.geometry(name)
        if row['patient']!=original['patient'] or row['test_group']!=original['role']:
            raise ValueError('患者或冻结测试组发生变化')
        if list(geo.native_shape_xyz)!=original['shape'] or list(geo.spacing_xyz)!=original['spacing']:
            raise ValueError('实际预处理空间与原病例不同')
        cases[name]=dict(original,image=row['image'],label=row['label'])
        for kind in ('image','label'):
            if not Path(row[kind]).resolve(strict=True).is_relative_to(Path(description['manifest']['data_root']).resolve(strict=True)):
                raise ValueError('原图或GT未绑定实际数据集目录')
    job=dict(protocol=PROTOCOL,prepared=str(Path(args.preprocessed).resolve()),
        preprocessed=str(Path(args.preprocessed).resolve()),cht_code=str(Path(args.cht_code).resolve()),
        preprocessed_identity=source.identity,input_format=FORMAT,mode='six-way',cases=cases,
        technical_subset=False,implementation=implementation(),retention='final-only-v1')
    frozen=dict(protocol=PROTOCOL,run={k:model[k] for k in ('profile','supervision','request')},
        checkpoints=model['checkpoints'],boxes=model['boxes'],anatomy=model['anatomy'],
        prepared=job['preprocessed'],training=str(model_root))
    job['source']=digest(frozen)
    if (out/'run.json').exists():
        if not args.resume or read_json(out/'run.json')!=job:
            raise ValueError('更换数据条目后请使用新输出目录；续跑必须保持原输入')
    elif args.resume:
        raise ValueError('目标尚无可续跑任务')
    for branch in SIX:
        path=model_root/(branch+'.pt')
        if not path.is_file() or sha256_file(path)!=model['checkpoints'][branch]:
            raise ValueError('六模型缺少权重或权重内容不符：'+branch)
    target=out/'source'
    target.mkdir(exist_ok=True)
    for branch in SIX:
        src,dst=model_root/(branch+'.pt'),target/(branch+'.pt')
        if dst.exists():
            if not os.path.samefile(src,dst):
                raise ValueError('预测权重未指向登记模型')
        else:
            os.link(src,dst)
    atomic_json(target/'manifest.json',frozen)
    atomic_json(out/'run.json',job)
    return job


def predict(args, owned_empty=False):
    from .cli import execute
    out=assert_output_location(args.output)
    if os.name=='posix' and not out.is_relative_to(Path('/data4/zhangruichen/result/chd1282')):
        raise ValueError('预测结果应放在/data4/zhangruichen/result/chd1282')
    if out.exists() and not args.resume and not (owned_empty and not any(out.iterdir())):
        raise ValueError('新预测使用独立输出目录')
    for protected in (Path(args.models).resolve(),Path(args.preprocessed).resolve().parent,
                      Path(args.cht_code).resolve()):
        if out==protected or out in protected.parents or protected in out.parents:
            raise ValueError('输入与输出目录重叠')
    out.mkdir(parents=True,exist_ok=True)
    with operation_lock(out,'xu-sixway-data4',out) as owner:
        success=False
        try:
            job=open_job(args,out)
            execute(args,out,owner,job,'prepare')
            execute(args,out,owner,job,'predict')
            success=True
        finally:
            write_status(out,state='complete' if success else 'failed',stage='six-way',active=[])


def acceptance(args):
    from .evaluation import evaluate
    from .workflow import verify_job
    out=Path(args.output).resolve(strict=True)
    job=verify_job(out)
    complete=read_json(out/'complete.json')
    if (job.get('input_format')!=FORMAT or job['mode']!='six-way' or len(complete['cases'])!=1
            or complete['rejected']!=0):
        raise ValueError('验收要求1例新预处理输入成功完成无血池预测')
    for row in complete['cases'].values():
        if {'blood','vessels'} & row['result']['qc'].keys():
            raise ValueError('验收输出含血池或血管修正')
    report=evaluate(out)
    atomic_json(args.receipt,dict(protocol='xu-data4-acceptance-v1',preprocessed=job['preprocessed'],
        preprocessed_identity=job['preprocessed_identity'],models=job['source'],cases=1,
        checkpoints=read_json(out/'source/manifest.json')['checkpoints'],implementation=job['implementation'],
        mode='six-way',evaluation=report))


def short(args):
    model=read_json(Path(args.models)/'models.json')
    args.cases=[next(n for n,r in model['heldout'].items() if r['role']=='source_test')]
    args.groups=['source_test']
    args.resume=False
    out=assert_output_location(args.output)
    if os.name=='posix' and out!=Path('/data4/zhangruichen/result/chd1282/.short-one'):
        raise ValueError('短测仅使用指定的独占目录')
    receipt=Path(args.receipt).resolve()
    if receipt.is_relative_to(out):
        raise ValueError('验收记录应保留在短测目录之外')
    out.mkdir(parents=True,exist_ok=False)
    try:
        predict(args,owned_empty=True)
        acceptance(args)
    finally:
        if out.resolve()!=out or out.is_symlink():
            raise ValueError('短测目录解析改变，停止清理')
        receipt.parent.mkdir(parents=True,exist_ok=True)
        workers=sorted((out/'logs').glob('*.log'),key=lambda x:x.stat().st_mtime_ns)
        for name,path in [('short-worker.log',workers[-1] if workers else None),
                          ('short-evaluation.log',out/'evaluation.log')]:
            if path is not None and path.is_file():
                with path.open('r',encoding='utf-8',errors='replace') as stream:
                    tail=''.join(deque(stream,maxlen=1000))
                (receipt.parent/name).write_text(tail,encoding='utf-8')
        shutil.rmtree(out)


def main(argv=None):
    p=argparse.ArgumentParser(description='从正式preprocessed读取并执行无血池六网络预测')
    commands=p.add_subparsers(dest='command',required=True)
    check=commands.add_parser('check')
    check.add_argument('--preprocessed',type=Path,required=True)
    check.add_argument('--cht-code',type=Path,required=True)
    check.add_argument('--technical',action='store_true')
    models=commands.add_parser('models')
    for key in ('preprocessed','cht-code','legacy-prepared','output'):
        models.add_argument('--'+key,type=Path,required=True)
    models.add_argument('--prediction',type=Path,nargs='+',required=True)
    models.add_argument('--training',type=Path,help='保留原训练的六个最新检查点，不重复复制数据')
    accept=commands.add_parser('accept')
    accept.add_argument('--output',type=Path,required=True)
    accept.add_argument('--receipt',type=Path,required=True)
    run=commands.add_parser('predict')
    for key in ('preprocessed','cht-code','models','output'):
        run.add_argument('--'+key,type=Path,required=True)
    run.add_argument('--groups',nargs='+',choices=['source_test','target_primary','target_secondary'],
                     default=['source_test','target_primary','target_secondary'])
    run.add_argument('--case',dest='cases',action='append')
    run.add_argument('--threads',type=int,default=4)
    run.add_argument('--wait-seconds',type=int,default=60)
    run.add_argument('--device')
    run.add_argument('--devices',default='all-visible')
    run.add_argument('--exclude-physical-gpus',type=int,nargs='*',default=[0,1])
    run.add_argument('--resume',action='store_true')
    single=commands.add_parser('short',parents=[run],add_help=False)
    single.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args(argv)
    if args.command=='check':
        check_data(args.preprocessed,args.cht_code,args.technical)
    elif args.command=='models':
        build_models(args)
    elif args.command=='accept':
        acceptance(args)
    elif args.command=='short':
        short(args)
    else:
        if args.threads<1 or not 1<=args.wait_seconds<=300:
            raise ValueError('线程和等待时间无效')
        predict(args)
