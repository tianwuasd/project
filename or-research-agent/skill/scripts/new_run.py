"""Create a fresh research workspace without overwriting existing work."""
import argparse
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path


def create_run(topic, output):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    state = {'schema_version': 1, 'topic': topic, 'scenario': 'draft', 'modeling': 'draft',
             'algorithm': 'draft', 'next_action': '检索最邻近工作，填写场景卡', 'artifacts': []}
    (path / 'state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    (path / 'evidence.json').write_text(json.dumps({'claims': [], 'search_runs': [], 'publication_status': 'unknown'}, ensure_ascii=False, indent=2), encoding='utf-8')
    (path / 'brief.md').write_text(f'# {topic}\n\n研究目的：待补充\n领域/范围：待补充\n交付：找场景、建模、求解算法\n\n下一步：记录已知条件和需确认的关键数据。\n', encoding='utf-8')
    for name, sections in {
        'scenario': ['决策者与业务痛点', '最邻近工作与证据', '候选场景比较', '推荐问题与反对理由', '数据需求与待验证假设'],
        'model': ['信息时序与假设', '集合、参数、变量与单位', '目标函数', '约束与业务含义', '边界反例与模型校验'],
        'algorithm': ['模型结构与算法选择', '基线、主算法与伪代码', '精确性条件与停止规则', '实验、消融与种子', '已运行结果与局限'],
    }.items():
        (path / (name + '.md')).write_text('# ' + topic + '\n\n' + '\n\n'.join('## ' + s + '\n\n尚未执行。' for s in sections) + '\n', encoding='utf-8')
    (path / 'results').mkdir()
    return str(path.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('topic'); parser.add_argument('--output')
    args = parser.parse_args()
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S')
    print(create_run(args.topic, args.output or (Path(os.environ.get('OR_RESEARCH_HOME', '.')) / 'runs' / stamp)))
