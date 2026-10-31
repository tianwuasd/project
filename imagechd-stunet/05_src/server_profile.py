"""Server defaults with explicit provenance; CLI and original runtime win."""
import json
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1]


def load(name=None):
    name = name or ('zmic44' if sys.platform.startswith('linux') else 'generic')
    if name not in {'zmic44', 'generic'}:
        raise ValueError('服务器配置请选择 zmic44 或 generic')
    if name == 'zmic44':
        return json.loads((CODE/'06_configs/servers/zmic44.json').read_text(encoding='utf-8'))
    return dict(name='generic', runtime=str(Path.home()/'.local/share/imagechd_stunet_runtime'),
                results=str(Path.home()/'imagechd_stunet_results'), data='', conda='', network=None,
                gpu_count=1, gpu_candidates='auto',gpu_memory=20,gpu_memory_policy='zmic44-20g-v1')


def paths(profile, saved=None, args=None):
    saved = saved or {}
    return {key: getattr(args,key,None) or saved.get(key) or profile[key]
            for key in ('runtime','results','data')}


def show(profile, values):
    print(f'服务器配置：{profile["name"]}（目录按记录预填，提交前核对权限与可用空间）')
    print('原始数据：'+str(values['data'] or '尚未填写'))
    print('环境与下载缓存：'+str(values['runtime']))
    print('结果根目录：'+str(values['results']))
    print('任务预处理缓存：结果根目录/唯一任务/workspace/04_data/derived')
    print('模型/预测/报告：workspace/07_experiments、08_results、09_reports；原始数据不改写。')
