from pathlib import Path

from .input_source import open_prepared


def open_q_cache(source_project, prepared, *, technical=False):
    source = Path(source_project).resolve(strict=True)
    expected = source/'src/dataset1282_analysis/q_route1_bridge.py'
    if Path(__file__).resolve() != expected:
        raise ValueError('Q 路线输入适配器来源与指定项目不同')
    return open_prepared(prepared, allow_technical_subset=technical)
