from copy import deepcopy


RUNTIME_FIELDS = {'run_dir','threads','dependency_root','official_root','execution','work_root',
    'device','logging','checkpoint_dir','attempt_ledger','save_every_updates','save_every_seconds','distributed_backend',
    'training_cache','resume_budget','cpu_preparation'}
INFERENCE_FIELDS = ('model','method','max_candidates','patch','local_size','frames','resolution','seed','gray_window')


def resume_contract(config,prepared,stage):
    from .workflow import stage_case_names,training_names
    names = stage_case_names(config,prepared,stage)
    return dict(settings={key:value for key,value in config.items() if key not in RUNTIME_FIELDS|{'input_source'}},
        dataset=prepared['dataset'],dataset_json=prepared['dataset_json'],scope=prepared['scope'],
        training_names=training_names(config,prepared,stage),consumed_names=names,
        cases=[{key:row.get(key) for key in ('case_id','roles','patient','unknown_classes','ignore_label_ids')}
               for row in prepared['manifest']['cases'] if row['case_id'] in names])


def require_resume_contract(saved_config,saved_prepared,config,prepared,stage):
    previous = resume_contract(saved_config,saved_prepared,stage)
    current = resume_contract(config,prepared,stage)
    if config['model']=='q_route1':
        for value in (previous,current):
            if value['settings'].get('updates_per_epoch') is None:
                value['settings']['updates_per_epoch'] = 250
    if config['model']=='q_route1' and config.get('resume_budget',False):
        for value in (previous,current):
            for key in ('updates','epochs','updates_per_epoch','validate_every'):
                value['settings'].pop(key,None)
    if previous!=current:
        raise ValueError('恢复的科学设置、训练验证划分或类别监督语义不同')
    return (saved_prepared['storage']!=prepared['storage'] or saved_prepared['plan']!=prepared['plan']
        or saved_prepared['technical']!=prepared['technical']
        or saved_config['input_source']!=config['input_source'])


def inference_settings(config):
    return {key:deepcopy(config[key]) for key in (*INFERENCE_FIELDS,'input_source')}


def frozen_inference_config(config,saved):
    if any(config[key]!=saved[key] for key in INFERENCE_FIELDS):
        raise ValueError('预测科学设置与固定模型快照不同')
    value = deepcopy(saved)
    value.update({key:deepcopy(config[key]) for key in RUNTIME_FIELDS if key in config})
    value['input_source'] = deepcopy(config['input_source'])
    return value
