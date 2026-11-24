from copy import deepcopy
from pathlib import Path

from ...artifacts.store import digest,file_digest,read_json


RUNTIME_FIELDS = {'base_config','run_dir','prepared','workers','prefetch','max_seconds','ram_limit_bytes','vram_limit_bytes'}
SCIENCE_FIELDS = ('precision','gray_window','intensity_semantics','source_mean','source_std',
    'prompts','geometry_policy','spacing_policy','orientation_evidence','intensity_evidence')
IMPLEMENTATION = ('adapter','config','input','training_data','training_worker','finetune','distributed','launcher','resume')


def implementation():
    result = {name:file_digest(Path(__file__).with_name(name+'.py')) for name in IMPLEMENTATION}
    result['gradient_sync'] = file_digest(Path(__file__).parents[2]/'acceleration'/'distributed.py')
    return result


def scientific_contract(config,checked):
    hashes = {}
    def content(path):
        if path not in hashes:
            hashes[path] = file_digest(path)
        return hashes[path]
    sources = deepcopy(checked['sources'])
    for items in sources.values():
        for item in items:
            row = item['row'] = deepcopy(item['row'])
            row.pop('source_stamp')
            for key in ('image','label'):
                row[key] = dict(sha256=content(row[key]))
            if 'prepared' in item:
                item['prepared'] = dict(sha256=content(item['prepared']))
                item.pop('prepared_stamp')
    return dict(schema=2,model='medsam3',method='medical_lora_full_v1',
        upstream_commit=checked['dependencies']['upstream_commit'],device=config['device'],
        sources=sources,initialization={key:content(checked['dependencies']['files'][key]['path'])
            for key in ('base_weights','lora_weights','official_config','bpe_path')},
        training={key:value for key,value in config['training'].items() if key not in RUNTIME_FIELDS},
        science={key:config[key] for key in SCIENCE_FIELDS})


def legacy_contract(config,checked,code):
    return dict(sources=checked['sources'],dependencies=checked['dependencies'],implementation=code,
        training={k:v for k,v in config['training'].items() if k not in ('run_dir','prepared','max_seconds')},
        science={k:config[k] for k in SCIENCE_FIELDS})


def resume_identity(saved,config,checked,current,proof_path):
    if 'contract' in saved:
        if saved.get('contract_identity') != digest(saved['contract']):
            raise ValueError('保存的科学合同摘要不一致')
        if saved['contract'] != current:
            raise ValueError('恢复的科学合同不同：核对数据、精度、global batch与优化配方')
        return saved['identity'],dict(kind='contract',source_identity=saved['identity'])
    audit = read_json(Path(__file__).with_name('legacy_resume.json'))
    if audit['schema'] != 1:
        raise ValueError('旧恢复来源登记格式无效')
    proof = read_json(proof_path) if Path(proof_path).is_file() else None
    if proof:
        proof['contract']['science']['prompts'] = {int(k):v for k,v in proof['contract']['science']['prompts'].items()}
    for source in audit['sources']:
        old = proof['contract'] if proof else legacy_contract(config,checked,source['implementation'])
        if old['implementation'] != source['implementation'] or digest(old) != saved['identity']:
            continue
        if proof and proof['identity'] != saved['identity']:
            raise ValueError('原合同与旧检查点身份不同')
        material = {k:v for k,v in old['dependencies']['files'].items() if k != 'python'}
        expected = {k:v for k,v in checked['dependencies']['files'].items() if k != 'python'}
        if (old['sources'] != checked['sources'] or material != expected or
            old['dependencies']['upstream_commit'] != checked['dependencies']['upstream_commit'] or
            old['science'] != current['science'] or
            {k:v for k,v in old['training'].items() if k not in RUNTIME_FIELDS} != current['training']):
            raise ValueError('旧检查点与当前训练内容不一致')
        return saved['identity'],dict(kind='legacy_contract_verified',source_identity=saved['identity'],
            source_commit=source['commit'])
    raise ValueError('旧摘要无法核验；请使用原训练配置或提供摘要一致的resume-proof.v1.json，禁止跳过身份检查')
