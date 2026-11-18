from contextvars import ContextVar
from functools import wraps
import os
from pathlib import Path
import sys


_pending = ContextVar('training_attempts', default=None)


def is_technical(config):
    return config.get('technical') is True or config.get('scope') == 'technical'


def tool_ledger(config):
    root = os.environ.get('CHD_MINI_TOOL_ROOT')
    path = os.environ.get('CHD_MINI_ATTEMPT_LEDGER', config.get('attempt_ledger'))
    if not root or not path:
        raise ValueError('技术训练需要CHD_MINI_TOOL_ROOT和统一CHD_MINI_ATTEMPT_LEDGER')
    source = str(Path(root).resolve()/'src')
    if source not in sys.path:
        sys.path.insert(0, source)
    from chd_mini.attempts import AttemptLedger
    return AttemptLedger(path)


def tool_stage(config, stage):
    expected = ('q_'+stage if config['model'] == 'q_route1' else
                'med_sequence' if config['model'] == 'medsam2' else 'stu')
    if os.environ.get('CHD_MINI_STAGE', expected) != expected:
        raise ValueError('监督器与生产训练阶段不一致')
    return expected


def settle_attempts(function):
    @wraps(function)
    def execute(*args, **kwargs):
        token = _pending.set([])
        try:
            return function(*args, **kwargs)
        finally:
            # Python错误照常传播；强制退出的进程由监督器登记中断。
            outcome = 'failed' if sys.exc_info()[0] is not None else 'interrupted'
            for ledger, stage, number in _pending.get():
                ledger.finish(stage, number, outcome)
            _pending.reset(token)
    return execute


def remaining_attempts(config, stage):
    if not is_technical(config):
        return None
    ledger = tool_ledger(config)
    state, limits = ledger.read(), ledger.limits()
    key = tool_stage(config, stage)
    return max(0,min(limits['stage_limits'][key]-sum(row['stage']==key for row in state['attempts']),
        limits['total_limit']-len(state['attempts'])))


def claim_attempt(config, stage):
    if not is_technical(config):
        return None
    pending = _pending.get()
    if pending is None:
        raise RuntimeError('训练入口必须使用settle_attempts管理尝试结算')
    ledger, key = tool_ledger(config), tool_stage(config, stage)
    number = ledger.claim(key)
    pending.append((ledger, key, number))
    return number


def finish_attempt(config, stage, number, outcome='successful'):
    if not is_technical(config):
        return
    pending = _pending.get()
    if pending is None:
        raise RuntimeError('训练尝试结算缺少入口上下文')
    key = tool_stage(config, stage)
    entries = [item for item in pending if item[1:] == (key, number)]
    if len(entries) != 1:
        raise ValueError('训练尝试未领取或已结算')
    entries[0][0].finish(key, number, outcome)
    pending.remove(entries[0])
