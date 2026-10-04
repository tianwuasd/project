"""Create a resumable study without overwriting existing work."""
import argparse
import json
from datetime import datetime
from pathlib import Path


def create_study(topic, output):
    if not topic.strip():
        raise ValueError('研究题目不能为空')
    output = Path(output)
    template = (Path(__file__).resolve().parents[1] / 'assets/study.md').read_text(encoding='utf-8')
    text = template.replace('{{TOPIC}}', topic).replace('{{DATE}}', datetime.now().astimezone().isoformat())
    output.mkdir(parents=True, exist_ok=False)
    for folder in ['literature', 'experiments', 'features']:
        (output / folder).mkdir()
    (output / 'research_package.md').write_text(text, encoding='utf-8')
    state = {'schema_version': 1, 'topic': topic, 'mode': 'full', 'stage': 'evidence', 'status': 'planned',
             'artifacts': ['research_package.md'], 'decisions': [], 'gaps': [],
             'next_action': '检索论文并保存文档ID、SHA256、页码与阅读范围'}
    (output / 'state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    return state


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('topic'); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(create_study(a.topic, a.output), ensure_ascii=False, indent=2))
