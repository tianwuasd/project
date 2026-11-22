from concurrent.futures import CancelledError
import math
import subprocess
import time

from ..observability.events import event


def query_settings(resources=None):
    resources = {} if resources is None else resources
    timeout = resources.get('gpu_query_timeout_seconds', 30)
    retries = resources.get('gpu_query_retries', 2)
    delay = resources.get('gpu_query_retry_seconds', 2)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise ValueError('gpu_query_timeout_seconds必须位于(0,300]秒')
    if type(retries) is not int or not 0 <= retries <= 10:
        raise ValueError('gpu_query_retries必须为0至10的整数')
    if type(delay) not in (int, float) or not math.isfinite(delay) or not 0 <= delay <= 60:
        raise ValueError('gpu_query_retry_seconds必须位于[0,60]秒')
    return timeout, retries, delay


def query_nvidia_smi(arguments, *, resources=None, cancel=None, log_dir=None):
    return query_nvidia_smi_batch([arguments],resources=resources,cancel=cancel,log_dir=log_dir)[0]


def query_nvidia_smi_batch(queries, *, resources=None, cancel=None, log_dir=None):
    timeout, retries, delay = query_settings(resources)
    for attempt in range(1, retries + 2):
        if cancel is not None and cancel.is_set():
            raise CancelledError('GPU查询已经取消')
        started = time.monotonic()
        try:
            values = []
            for arguments in queries:
                if cancel is not None and cancel.is_set():
                    raise CancelledError('GPU查询已经取消')
                result = subprocess.run(['nvidia-smi', *arguments], check=True,
                                        capture_output=True, text=True, timeout=timeout)
                values.append(result.stdout)
        except subprocess.TimeoutExpired:
            next_delay = delay if attempt <= retries else 0
            event(log_dir, 'gpu_query', console=False, attempt=attempt, reason='timeout',
                  query_seconds=time.monotonic() - started, timeout=timeout,
                  next_delay=next_delay, query=queries)
            if attempt > retries:
                raise
            if cancel is None:
                time.sleep(next_delay)
            elif cancel.wait(next_delay):
                raise CancelledError('GPU查询已经取消') from None
        else:
            if cancel is not None and cancel.is_set():
                raise CancelledError('GPU查询已经取消')
            event(log_dir, 'gpu_query', console=False, attempt=attempt, reason='success',
                  query_seconds=time.monotonic() - started, timeout=timeout,
                  next_delay=0, query=queries)
            return values
