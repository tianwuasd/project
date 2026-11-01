"""Static pre-release artifact baseline; preserve after the 0.2.0 release."""

import json
from pathlib import Path

from academic_agent.trends import verify_trends


def test_frozen_v02_artifacts_remain_verifiable():
    directory = Path(__file__).parent / 'fixtures' / 'trends-0.2.0'
    verified = verify_trends(directory)
    assert verified['valid'] and verified['files_verified'] == 6
    record = json.loads((directory / 'computation.json').read_text(encoding='utf-8'))
    assert record['counts']['candidates'] == 3
    assert record['counts']['eligible'] == 2
    assert record['counts']['valid_year'] == record['counts']['missing_year'] == 1
    assert [(row['year'], row['count'], row['share']) for row in record['annual']] == [(2023, 1, 1.0)]
    assert {row['label']: (row['count'], row['denominator']) for row in record['labels']} == {
        '模拟方法': (2, 2), '对照方法': (1, 2),
    }
