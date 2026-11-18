from copy import deepcopy

from ..artifacts.store import digest
from ..datasets.manifest import LABELS


def validate_training_inputs(prepared,config):
    if prepared['dataset']!=config['dataset']:
        raise ValueError('当前训练与输入数据集不同')
    labels = {name:index for index,name in enumerate(LABELS[config['dataset']])}
    if prepared['dataset_json']['labels']!=labels or prepared['dataset_json']['channel_names']!={'0':'CT'}:
        raise ValueError('当前输入的类别或CT通道与模型不同')
    rows = prepared['manifest']['cases']
    if not rows or len({row['case_id'] for row in rows})!=len(rows):
        raise ValueError('当前病例名单为空或重复')
    roles = {role:set() for role in ('train','validation','test')}
    patients = {role:set() for role in roles}
    for row in rows:
        selected = set(row['roles'])
        if not selected or not selected<={'train','validation','test','predict'} or len(selected & roles.keys())>1:
            raise ValueError('当前病例角色无效或训练验证测试重叠')
        if selected & {'train','validation'} and not row.get('label'):
            raise ValueError('训练与验证需要实际监督标签')
        for role in selected & roles.keys():
            roles[role].add(row['case_id'])
            if row.get('patient') is not None:
                patients[role].add(row['patient'])
    if not roles['train'] or not roles['validation']:
        raise ValueError('当前训练和验证名单必须非空')
    for left,right in (('train','validation'),('train','test'),('validation','test')):
        if roles[left]&roles[right] or patients[left]&patients[right]:
            raise ValueError('当前病例或患者划分重叠')


def selection_scope(prepared,config):
    return digest(dict(dataset=prepared['dataset'],labels=prepared['dataset_json'],scope=prepared['scope'],
        cases=prepared['manifest']['cases'],storage={key:value for key,value in prepared['storage'].items() if key!='source_scope'},
        plan={key:value for key,value in prepared['plan'].items() if key!='generation'},
        technical=prepared['technical'],training=config['training'],fold=config.get('fold'),
        input_source=config.get('input_source')))


def resume_inputs(previous,prepared,config):
    validate_training_inputs(prepared,config)
    old = previous['prepared']
    if old['dataset']!=prepared['dataset'] or old['dataset_json']!=prepared['dataset_json']:
        raise ValueError('恢复训练的数据集、类别或通道不同')
    return selection_scope(old,previous['config'])!=selection_scope(prepared,config)


def reset_selection(progress,scope):
    result = deepcopy(progress)
    result.pop('selection',None)
    result.update(selection_scope=scope,val_step=0,val_sum=0.,validation_loss=None)
    return result
