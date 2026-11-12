import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dataset1282_analysis.contracts import read_json, sha256_file


def old_hash(path, base):
    data = subprocess.check_output(['git', 'show', f'{base}:dataset1282_analysis/{path}'], cwd=ROOT)
    return sha256(data).hexdigest()


def function_files(relative, function):
    tree = ast.parse((ROOT/relative).read_text(encoding='utf-8-sig'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant)
            and isinstance(n.value, str) and n.value.endswith('.py')]


def transition(base):
    from dataset1282_analysis.xu_pipeline import _sources
    maps = []
    additions = {'src/dataset1282_analysis/'+name for name in (
        'input_source.py', 'online_input.py', 'online_memory.py', 'online_prediction.py', 'prediction/heldout.py')}
    for technical in (False, True):
        current = _sources(technical)
        stored = {p:old_hash(p, base) for p in current if p not in additions}
        maps.append(dict(path_prefix='', stored=stored, current=current))
    for relative, function in (('xu_blood.py', '_source'), ('xu_init.py', '_inference_contract')):
        names = function_files('src/dataset1282_analysis/'+relative, function)
        prefix = 'src/dataset1282_analysis/'
        maps.append(dict(path_prefix=prefix, stored={p:old_hash(prefix+p, base) for p in names},
                         current={p:sha256_file(ROOT/prefix/p) for p in names}))
    return dict(id='monai-online-input', review='固定原计划及原始来源；两例真实输入、采样和模型保存重载验收。', sources=maps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='f49102f91a065749e89d9b45bd4331e575fff242')
    parser.add_argument('--emit', action='store_true')
    args = parser.parse_args()
    edge = transition(args.base)
    if args.emit:
        print(json.dumps(edge, ensure_ascii=False))
        return
    from dataset1282_analysis.xu_resume_compat import source_compatible
    for mapping in edge['sources']:
        assert source_compatible(mapping['stored'], mapping['current'], mapping['path_prefix']), '缺少已验证源码兼容记录'
        changed = dict(mapping['current'])
        changed[next(iter(changed))] = '0'*64
        assert not source_compatible(mapping['stored'], changed, mapping['path_prefix']), '未审核源码被错误接受'
    print(json.dumps(dict(state='passed', source_groups=len(edge['sources']), changed_source_rejected=True)))


if __name__ == '__main__':
    main()
