"""Primary user choices and private, explicitly bounded real-data assessment."""
from pathlib import Path
from model_catalog import parse_models

PREPROCESSORS={'nnunet_default':'nnU-Net 默认预处理'}
FORMAL_BLOCKS={'1':'prepare','2':'train','3':'predict',
               '预处理':'prepare','训练':'train','预测':'predict',
               'prepare':'prepare','train':'train','predict':'predict'}
QUICK_BUDGET=dict(train_steps=12,warmup_steps=2,measured_steps=10,validation_steps=2,
                  resumed_training_steps=1,training_cases=2,validation_cases=1,
                  prediction_cases=1,max_tuning_candidates=2,tuning_timeout_seconds=600)


def parse_formal_blocks(value):
    tokens=value if isinstance(value,(list,tuple)) else str(value).replace('，',',').replace(',',' ').split()
    if tokens==['all']:tokens=['1','2','3']
    if not tokens or any(token not in FORMAL_BLOCKS for token in tokens):
        raise ValueError('正式任务请选择1预处理、2训练、3预测')
    chosen={FORMAL_BLOCKS[token] for token in tokens}
    return [name for name in ['prepare','train','predict'] if name in chosen]


def automatic_request(profile,data,models='B0',*,preprocessor='nnunet_default',kind='formal',blocks='all'):
    if preprocessor not in PREPROCESSORS:
        raise ValueError('预处理方法不存在')
    if kind not in {'formal','assessment'}:
        raise ValueError('任务类型不存在')
    raw=Path(data).expanduser().resolve()
    if not raw.is_dir() or raw==Path(raw.anchor):
        raise ValueError('请填写实际数据目录')
    methods=parse_models(models)
    results=Path(profile['results']).expanduser().resolve()
    if kind=='assessment':results=results/'assessments'
    return dict(name=('完整测评' if kind=='assessment' else '正式任务')+' '+','.join(methods),
                runtime=str(Path(profile['runtime']).expanduser().resolve()),results=str(results),data=str(raw),
                profile=profile['name'],conda=profile.get('conda',''),dataset='wholeheart',preprocessor=preprocessor,
                job_kind=kind,formal_training=kind=='formal',
                tasks=['assessment'] if kind=='assessment' else parse_formal_blocks(blocks),
                methods=methods,folds=['ct_holdG'],gpus=[],gpu_count=0,gpu_candidates='auto',
                gpu_policy='automatic',gpu_poll_seconds=1,allow_batch_growth=True,max_global_batch_size=8,
                gpu_memory=profile.get('gpu_memory',20),gpu_memory_policy='zmic44-20g-v1',
                training_execution='cooperative_elastic',
                formal_budget=dict(epochs=250,train_iterations=250,validation_iterations=50),
                assessment_budget=dict(QUICK_BUDGET),input_job=None,preprocess_cache='reuse')


def sample_scope(cases,split):
    from cohort import validate_split
    validate_split(split,cases)
    by_id={case['case_id']:case for case in cases}
    groups={by_id[cid]['group'] for cid in split['train']}
    chosen=[]
    for group in sorted(groups):
        candidates=[cid for cid in split['train'] if by_id[cid]['group']==group]
        candidates.sort(key=lambda cid:(-by_id[cid].get('estimated_voxels',0),cid))
        chosen.append(candidates[0])
    chosen=chosen[:QUICK_BUDGET['training_cases']]
    validation=sorted(split['val'],key=lambda cid:(-by_id[cid].get('estimated_voxels',0),cid))[:1]
    if len(chosen)!=2 or len(validation)!=1:
        raise ValueError('来源病例不足以进行独立短测')
    return dict(train_ids=chosen,validation_ids=validation,prediction_ids=validation[:],
                planning_fit_ids=split['train'][:],excluded_target_ids=split['target'][:],
                formal_split_unchanged=True,
                scope='few-case engineering flow only; full source-fit planning preserves formal preprocessing recipe; not a convergence/performance study')
