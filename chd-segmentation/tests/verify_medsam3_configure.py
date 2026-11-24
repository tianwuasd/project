import argparse
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import yaml


def main():
    parser = argparse.ArgumentParser()
    for field in ('root','manifest','cohort','unbound-manifest','official-root','weights-dir'):
        parser.add_argument('--'+field,required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(project/'src'))
    from chd_segmentation.models.medsam3.finetune import settings
    from chd_segmentation.artifacts.store import write_json,read_json
    root = Path(args.root).resolve(strict=True)
    began = perf_counter()
    command = [sys.executable,'-B',str(project/'run.py'),'models','medsam3','configure','--root',str(root),
        '--manifest',args.manifest,'--cohort',args.cohort,'--python',sys.executable,
        '--official-root',args.official_root,'--weights-dir',args.weights_dir]
    import json
    first = json.loads(subprocess.check_output(command,text=True,encoding='utf-8'))
    first_seconds = perf_counter()-began
    assert first['cases'] == {'train':1,'val':1} and not first['missing_materials']
    assert len(first['created']) == 2 and not first['model_loaded'] and first['training_updates'] == 0
    paths = [Path(path) for path in first['files']]
    stamps = [path.stat().st_mtime_ns for path in paths]
    mark = perf_counter()
    second = json.loads(subprocess.check_output(command,text=True,encoding='utf-8'))
    repeat_seconds = perf_counter()-mark
    assert not second['created'] and stamps == [path.stat().st_mtime_ns for path in paths]
    config = settings(paths[1])
    assert config['resources']['gpu_count'] == 2 and config['training']['batch'] == 4
    assert config['data_root'] == read_json(args.manifest)['data_root']
    changed = yaml.safe_load(paths[0].read_text(encoding='utf-8'))
    changed['resources']['cpu_threads'] = 8
    paths[0].write_text(yaml.safe_dump(changed,allow_unicode=True),encoding='utf-8')
    before = [path.read_bytes() for path in paths]
    rejected = subprocess.run(command,capture_output=True,text=True,encoding='utf-8')
    assert rejected.returncode != 0 and 'FileExistsError' in rejected.stderr
    assert before == [path.read_bytes() for path in paths]
    unbound = list(command)
    unbound[unbound.index('--manifest')+1] = args.unbound_manifest
    rejected = subprocess.run(unbound,capture_output=True,text=True,encoding='utf-8')
    assert rejected.returncode != 0 and '先执行bind-inputs' in rejected.stderr
    size = sum(path.stat().st_size for path in paths)
    for path in paths:
        assert path.resolve().is_relative_to(root)
        path.unlink()
    for path in (root/'course/chd-segmentation/configs',root/'course/chd-segmentation',root/'course'):
        path.rmdir()
    write_json(root/'report.json',dict(state='passed',cases=first['cases'],training_updates=0,
        selected_cases={key:config['training'][key+'_cases'] for key in ('train','val')},
        checks=['真实CLI生成','重复生成不写入','实际配置加载','用户配置冲突保留','未绑定清单拒绝'],
        seconds=perf_counter()-began,first_generation_seconds=first_seconds,repeat_seconds=repeat_seconds,
        config_files_per_second=2/first_seconds,removed_files=2,freed_bytes=size,
        ram_peak=None,vram_peak=None,peak_note='未测量；未加载模型或读取病例体素'))
    print('MedSAM3真实两例配置生成检查通过；已清理两份测试YAML')


if __name__ == '__main__':
    main()
