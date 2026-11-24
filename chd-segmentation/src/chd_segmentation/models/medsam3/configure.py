from pathlib import Path

import yaml

from ...artifacts.store import read_json,require_output


def generate(root,manifest=None,cohort=None,python=None,official_root=None,weights_dir=None):
    root = Path(root).resolve(strict=True)
    course = root/'course/chd-segmentation'
    manifest = Path(manifest).resolve(strict=True) if manifest else course/'inputs/chd1282_manifest.json'
    cohort = Path(cohort).resolve(strict=True) if cohort else course/'inputs/chd1282_cohort.json'
    source,roles = read_json(manifest),read_json(cohort)['cases']
    if source['dataset'] != 'chd1282' or source.get('binding_required',True):
        raise ValueError('需要已绑定服务器实际数据目录的CHD1282清单；先执行bind-inputs')
    data_root = Path(source['data_root'])
    if not data_root.is_absolute() or not data_root.is_dir():
        raise ValueError('绑定清单的data_root需要是当前机器上的实际绝对目录')
    cases,patients = {'train':[],'val':[]},{'train':set(),'val':set()}
    seen = set()
    for row in source['cases']:
        name = row['case_id']
        if name in seen:
            raise ValueError('病例清单包含重复case_id')
        seen.add(name)
        record = roles[name]
        if record['role'] not in ('train','development') or not record.get('included',True) or not record.get('index_scan',True):
            continue
        if record['patient'] != row['patient']:
            raise ValueError('病例清单与冻结划分的患者编号不同')
        group = 'train' if record['role'] == 'train' else 'val'
        cases[group].append(name)
        patients[group].add(record['patient'])
    if not all(cases.values()) or patients['train'] & patients['val']:
        raise ValueError('训练和development必须非空且按患者分离')
    official = Path(official_root).resolve() if official_root else root/'data/dependencies/MedSAM3'
    weights = Path(weights_dir).resolve() if weights_dir else root/'data/pretrained/MedSAM3'
    executable = Path(python).resolve() if python else root/'env/chd-medsam3/bin/python'
    templates = Path(__file__).parents[4]/'configs/models'
    base = yaml.safe_load((templates/'medsam3.yaml').read_text(encoding='utf-8'))
    training = yaml.safe_load((templates/'medsam3_finetune.yaml').read_text(encoding='utf-8'))
    base.update(manifest=str(manifest),cohort=str(cohort),data_root=str(data_root),
        work_root=str(course),registry_root=str(course/'resources'),python=str(executable),
        official_root=str(official),official_config=str(official/'configs/full_lora_config.yaml'),
        bpe_path=str(official/'sam3/assets/bpe_simple_vocab_16e6.txt.gz'),
        base_weights=str(weights/'sam3.pt'),lora_weights=str(weights/'best_lora_weights.pt'),
        orientation_evidence='沿用冻结CHD1282清单的RAS研究约定与共享数组索引；未认证扫描坐标')
    training.update(base_config='medsam3.yaml',run_dir=str(course/'runs/medsam3-medical-lora'),
        train_cases=sorted(cases['train']),val_cases=sorted(cases['val']))
    output = require_output(course/'configs',root,[data_root,official,manifest.parent,Path(__file__).parents[2]])
    values = {'medsam3.yaml':base,'medsam3_finetune.yaml':training}
    encoded = {output/name:yaml.safe_dump(value,allow_unicode=True,sort_keys=False) for name,value in values.items()}
    # 写入前检查两个目标，保留用户已编辑的配置。
    for path,text in encoded.items():
        if path.is_symlink() or path.exists() and (not path.is_file() or path.read_text(encoding='utf-8') != text):
            raise FileExistsError('已有配置不同，保留原文件：'+str(path))
    output.mkdir(parents=True,exist_ok=True)
    created = []
    for path,text in encoded.items():
        if not path.exists():
            with path.open('x',encoding='utf-8',newline='\n') as stream:
                stream.write(text)
            created.append(str(path))
    material_fields = ('python','base_weights','lora_weights','official_config','bpe_path')
    return dict(state='configured',files=[str(path) for path in encoded],created=created,
        cases={key:len(value) for key,value in cases.items()},data_root=str(data_root),
        missing_materials=[key for key in material_fields if not Path(base[key]).is_file()],
        model_loaded=False,data_voxels_read=False,training_updates=0,
        validation_scope='配置生成与冻结名单；材料完整性、环境、GPU容量及运行仍需check-training和局部测试')
